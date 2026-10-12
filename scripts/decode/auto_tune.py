#!/usr/bin/env python3
# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Auto-tune kernel template parameters for all caselist shapes.

Launches one subprocess per (H,HV,K,V) shape, each on a separate GPU.
Each subprocess benchmarks ALL combos × ALL batch sizes in-process.

Usage:
    python scripts/decode/auto_tune.py [--caselist-dir caselist/decode] [--output tuned_result.csv]
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/decode/auto_tune.py  # limit to specific GPUs
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "_bench_worker.py")

BV_VALUES = [16, 32, 64, 128]
WARP_NUM_VALUES = [4, 8]
STAGE_NUM_VALUES = [4, 5, 6, 8]
HV_PER_CTA_VALUES = [1]


def get_valid_combinations():
    combos = []
    for bv, wn, sn, hv in itertools.product(
        BV_VALUES, WARP_NUM_VALUES, STAGE_NUM_VALUES, HV_PER_CTA_VALUES
    ):
        if sn <= bv // wn:
            combos.append((bv, wn, sn, hv))
    return combos


def parse_caselist_file(path):
    cases = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            params = {}
            if "seqlen_q" in line:
                s = line[line.index("seqlen_q"):].strip(".")
                for m in re.findall(r"(\w+):(\[.*?\]|[^,]+)", s):
                    try:
                        params[m[0]] = int(m[1])
                    except ValueError:
                        params[m[0]] = m[1]
            if "num_heads_qk" not in params:
                continue
            key = (params["num_heads_qk"], params.get("num_heads_v", params["num_heads_qk"]),
                   params.get("head_dim_qk", 128), params.get("head_dim_v", 128),
                   params.get("dtype", "bf16"))
            B = params.get("batch_size", 1)
            cases.setdefault(key, set()).add(B)
    return cases


def parse_all_caselists(caselist_dir):
    all_cases = {}
    for fname in sorted(os.listdir(caselist_dir)):
        if not fname.endswith(".list"):
            continue
        cases = parse_caselist_file(os.path.join(caselist_dir, fname))
        for key, batches in cases.items():
            all_cases.setdefault(key, set()).update(batches)
    return all_cases


def merge_tuned_rows(output_path, fieldnames, new_rows, device_name):
    """Upsert tuned rows while preserving other kernels and state dtypes."""
    existing_rows = []
    if os.path.exists(output_path):
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                normalized = {name: row.get(name, "") for name in fieldnames}
                normalized["device_name"] = row.get("device_name") or device_name
                normalized["state_dtype"] = row.get("state_dtype") or "fp32"
                existing_rows.append(normalized)

    key_fields = (
        "device_name", "kernel", "B", "H", "HV", "K", "V", "dtype", "state_dtype"
    )

    def row_key(row):
        return tuple(str(row[name]) for name in key_fields)

    pending = {row_key(row): row for row in new_rows}
    merged_rows = []
    for row in existing_rows:
        key = row_key(row)
        merged_rows.append(pending.pop(key, row))
    merged_rows.extend(pending.values())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return len(existing_rows), len(merged_rows)


def sample_batch_sizes(all_batches, interval=16, max_batch=None):
    max_b = max(all_batches)
    if max_batch is not None:
        max_b = min(max_b, max_batch)
    sampled = [1]
    b = interval
    while b <= max_b:
        sampled.append(b)
        b += interval
    if max_b not in sampled:
        sampled.append(max_b)
    return sampled


def run_shape_worker(gpu_id, H, HV, K, V, combos, batch_sizes, warmup, iters,
                     results_out, label, kernels=None, detailed=False, use_acu=True,
                     mtp_T=4, state_dtype="fp32"):
    """Run a single shape's benchmarking on a specific GPU. Collects results into results_out list."""
    args_json = json.dumps({
        "H": H, "HV": HV, "K": K, "V": V,
        "combos": [list(c) for c in combos],
        "batch_sizes": batch_sizes,
        "warmup": warmup, "iters": iters,
        "kernels": kernels,
        "detailed": detailed,
        "use_acu": use_acu,
        "mtp_T": mtp_T,
        "state_dtype": state_dtype,
    })
    env = os.environ.copy()
    # Map logical gpu_id to physical device from parent's CUDA_VISIBLE_DEVICES
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_visible:
        physical_gpus = parent_visible.split(",")
        physical_id = physical_gpus[gpu_id] if gpu_id < len(physical_gpus) else str(gpu_id)
    else:
        physical_id = str(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = physical_id

    try:
        proc = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, args_json],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env
        )
        for line in proc.stdout:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            data = json.loads(line)
            if data.get("type") == "detail":
                bv, wn, sn, hpc = data["combo"]
                unit = data.get("unit", "us")
                val = data.get("metric", data.get("us", 0))
                print(f"  [{label}] B={data['B']:>5} {data['kernel']:>15} "
                      f"BV={bv:>3} WN={wn} SN={sn} HPC={hpc} -> {val:.1f} {unit}")
            elif data.get("type") == "result":
                results_out.append(data)
            elif data.get("type") == "done":
                if data.get("failed"):
                    print(f"  [{label}] Failed combos: {data['failed']}")
            elif data.get("type") == "fatal":
                print(f"  [{label}] FATAL: {data.get('error')}")
                break
        proc.wait()
    except Exception as e:
        print(f"  [{label}] Worker error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Auto-tune kernel template parameters")
    parser.add_argument("--caselist-dir", type=str, default=os.path.join(PROJECT_DIR, "caselist", "decode"))
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_DIR, "lut", "tuned_result.csv"))
    parser.add_argument("--batch-interval", type=int, default=16)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--case", type=str, default=None,
                        help="Tune a single case: H,HV,K,V[,B] (e.g. '16,64,128,128,64' or '16,64,128,128')")
    parser.add_argument("--max-batch", type=int, default=512,
                        help="Max batch size when --case is used without B (default: 512)")
    parser.add_argument("--kernel", type=str, default=None,
                        choices=["k_last", "k_last_mtp", "k_last_packed"],
                        help="Tune only a specific kernel type (default: all three)")
    parser.add_argument("--no-acu", action="store_true",
                        help="Use CUDA events instead of acu profiler for measurement")
    parser.add_argument("--mtp-t", type=int, default=4,
                        help="Sequence length T (draft_token_num) for k_last_mtp tuning (default: 4). "
                             "Aligns with sglang verify workload when set to 3.")
    parser.add_argument("--state-dtype", choices=["fp32", "bf16"], default="fp32",
                        help="State dtype for k_last_packed tuning (default: fp32)")
    args = parser.parse_args()

    import torch
    num_gpus = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}, {num_gpus} GPUs available")

    # Find usable GPUs
    usable_gpus = []
    for i in range(num_gpus):
        try:
            torch.randn(1, device=f"cuda:{i}")
            usable_gpus.append(i)
        except:
            pass
    print(f"Usable GPUs: {usable_gpus}")

    all_combos = get_valid_combinations()

    explicit_case_batch = False
    if args.case:
        parts = [int(x) for x in args.case.split(",")]
        if len(parts) == 5:
            H, HV, K, V, B = parts
            batch_sizes = [B]
            explicit_case_batch = True
        elif len(parts) == 4:
            H, HV, K, V = parts
            batch_sizes = None  # will use interval
        else:
            print("Error: --case expects H,HV,K,V[,B]")
            sys.exit(1)
        dtype_str = "bf16"
        if batch_sizes is None:
            batch_sizes = list(range(1, args.max_batch + 1, args.batch_interval))
            if args.max_batch not in batch_sizes:
                batch_sizes.append(args.max_batch)
        all_cases = {(H, HV, K, V, dtype_str): set(batch_sizes)}
    else:
        all_cases = parse_all_caselists(args.caselist_dir)

    shapes = sorted(all_cases.keys())
    print(f"Found {len(shapes)} shapes, {len(all_combos)} total combos")

    # Group shapes by GPU (round-robin), run groups sequentially per GPU, GPUs in parallel
    gpu_shape_groups = {g: [] for g in usable_gpus}
    for idx, shape in enumerate(shapes):
        gpu_id = usable_gpus[idx % len(usable_gpus)]
        gpu_shape_groups[gpu_id].append(shape)

    threads = []
    shape_results = {}

    kernels_filter = [args.kernel] if args.kernel else None
    detailed = args.case is not None
    use_acu = not args.no_acu

    def run_gpu_shapes(gpu_id, shape_list):
        for (H, HV, K, V, dtype_str) in shape_list:
            hv_per_h = HV // H
            valid_combos = [(bv, wn, sn, hpc) for bv, wn, sn, hpc in all_combos
                            if hpc <= hv_per_h and hv_per_h % hpc == 0 and hpc == 1]
            default = (64, 8, 4, 1)
            valid_combos.sort(key=lambda c: (0 if c == default else 1, -c[0], -c[1]))
            if explicit_case_batch:
                batch_sizes_s = sorted(all_cases[(H, HV, K, V, dtype_str)])
            else:
                batch_sizes_s = sample_batch_sizes(
                    all_cases[(H, HV, K, V, dtype_str)],
                    args.batch_interval,
                    args.max_batch,
                )
            label = f"H={H},HV={HV},GPU{gpu_id}"
            print(f"  {label}: {len(valid_combos)} combos × {len(batch_sizes_s)} batches")
            result_list = shape_results.setdefault((H, HV, K, V, dtype_str), [])
            run_shape_worker(gpu_id, H, HV, K, V, valid_combos, batch_sizes_s,
                             args.num_warmup, args.num_iters, result_list, label,
                             kernels=kernels_filter, detailed=detailed, use_acu=use_acu,
                             mtp_T=args.mtp_t, state_dtype=args.state_dtype)

    for gpu_id, shape_list in gpu_shape_groups.items():
        t = threading.Thread(target=run_gpu_shapes, args=(gpu_id, shape_list))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


    # Collect all results into CSV format
    csv_rows = []
    for (H, HV, K, V, dtype_str), result_list in sorted(shape_results.items()):
        for r in result_list:
            bv, wn, sn, hpc = r["combo"]
            metric_val = r.get("metric", r.get("us", 0))
            unit = r.get("unit", "us")
            row_state_dtype = args.state_dtype if r["kernel"] == "k_last_packed" else "fp32"
            csv_rows.append({
                "device_name": device_name, "kernel": r["kernel"],
                "B": r["B"], "H": H, "HV": HV, "K": K, "V": V,
                "dtype": dtype_str, "state_dtype": row_state_dtype,
                "BV": bv, "WARP_NUM": wn, "STAGE_NUM": sn,
                "HV_PER_CTA": hpc, "latency": f"{metric_val:.1f}", "unit": unit
            })

    # Write CSV with device name in filename
    output_path = args.output
    base, ext = os.path.splitext(output_path)
    output_path = f"{base}_{device_name}{ext}"

    fieldnames = ["device_name", "kernel", "B", "H", "HV", "K", "V", "dtype", "state_dtype",
                  "BV", "WARP_NUM", "STAGE_NUM", "HV_PER_CTA", "latency", "unit"]
    old_count, merged_count = merge_tuned_rows(
        output_path, fieldnames, csv_rows, device_name
    )
    print(
        f"\nUpserted {len(csv_rows)} records into {output_path} "
        f"({old_count} existing, {merged_count} total)"
    )


if __name__ == "__main__":
    main()
