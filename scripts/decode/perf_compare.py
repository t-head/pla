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

"""Compare CUDA vs Triton kernel performance across caselist/decode test cases.

Profiles k_last, k_last_mtp, and k_last_packed kernels using acu or CUDA event
timing, then writes per-kernel CSV files with timing ratios.

Usage:
    python scripts/decode/perf_compare.py
    python scripts/decode/perf_compare.py --caselist caselist/decode/Qwen3.5-Max_decode_tp1_kvlimit_200GB_model_len_128K_gdn_op.list
    python scripts/decode/perf_compare.py --kernels k_last k_last_mtp
    python scripts/decode/perf_compare.py --mtp-tokens 8 --output-dir results/
    python scripts/decode/perf_compare.py --max-mem-gb 8
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
RUNNER = os.path.join(SCRIPTS_DIR, "_perf_compare_runner.py")

ACU_METRICS = (
    "ppu__time_duration.sum,"
    "ce__cycles_elapsed.max,"
    "cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"
)

METRIC_KEYS = [
    "ppu__time_duration.sum",
    "ce__cycles_elapsed.max",
    "cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
]

CSV_HEADER = [
    "file", "B", "H", "HV", "K", "V", "dtype", "is_kda",
    "cuda_cycles", "triton_cycles", "cycle_ratio",
    "cuda_time_us", "triton_time_us",
    "cuda_tensor_pct", "triton_tensor_pct",
    "cuda_dram_pct", "triton_dram_pct",
    "cuda_host_us", "triton_host_us", "host_ratio",
    "cuda_e2e_us", "triton_e2e_us", "e2e_ratio",
]


def parse_caselist_line(s):
    result = {}
    target = "seqlen_q"
    if target in s:
        s = s[s.index(target):]
    else:
        return None
    s = s.strip(".")
    param_pattern = r"(\w+):(\[.*?\]|[^,]+)"
    for key, value in re.findall(param_pattern, s):
        try:
            result[key] = int(value)
        except ValueError:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def read_caselist(filepath):
    cases = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = parse_caselist_line(line)
            if parsed:
                # Family from the line itself: files like triton_wins.list mix
                # GDN and KDA lines, and parse_caselist_line slices from
                # seqlen_q onward, which drops the attn marker. Without this,
                # a KDA line in a file whose name lacks "kda" is benchmarked
                # as GDN — silently measuring the wrong kernel.
                if "kda_attn" in line:
                    parsed["_is_kda"] = True
                elif "gdn_attn" in line:
                    parsed["_is_kda"] = False
                cases.append(parsed)
    return cases


def estimate_mem_gb(B, H, HV, K, V, kernel, mtp_T, is_kda):
    """Estimate GPU memory needed in GB for a single case (both backends combined)."""
    T = mtp_T if kernel == "k_last_mtp" else 1
    state_bytes = B * HV * K * V * 4
    qkv_bytes = B * T * (H * K + H * K + HV * V) * 2
    ab_bytes = B * T * HV * (K if is_kda else 1) * 2
    extra = 0
    if kernel == "k_last_mtp":
        extra = B * T * HV * V * K * 4  # intermediate_states_buffer
    if kernel == "k_last_packed":
        state_bytes = (B + 16) * HV * V * K * 4
        qkv_bytes = B * (2 * H * K + HV * V) * 2
    return (state_bytes * 2 + qkv_bytes * 2 + ab_bytes * 2 + extra) / 1e9


def parse_acu_metrics(output):
    results = {}
    for metric in METRIC_KEYS:
        for line in output.split("\n"):
            if metric in line:
                after = line.split(metric)[1]
                match = re.search(r"([\d.]+)", after)
                if match:
                    results[metric] = float(match.group(1))
                    break
    return results


def run_acu(config, kernel_name=None, warmup=3):
    runner_args = json.dumps(config)
    cmd = [
        "acu", "--clock-control", "none",
        "--metrics", ACU_METRICS,
        "--page=details",
        "--launch-skip", str(warmup),
        "--launch-count", "1",
    ]
    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])
    cmd.extend([sys.executable, RUNNER, runner_args])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + "\n" + result.stderr
        metrics = parse_acu_metrics(output)
        if not metrics:
            print(f"    WARNING: no metrics parsed from acu output", file=sys.stderr)
            if result.returncode != 0:
                print(f"    acu exit code: {result.returncode}", file=sys.stderr)
                stderr_lines = result.stderr.strip().split("\n")[-5:]
                for l in stderr_lines:
                    print(f"    stderr: {l}", file=sys.stderr)
        return metrics
    except subprocess.TimeoutExpired:
        print(f"    ERROR: acu timed out after 600s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    ERROR: {e}", file=sys.stderr)
        return None


def run_events(config):
    config = {**config, "mode": "events"}
    config.setdefault("iters", 10)
    runner_args = json.dumps(config)
    cmd = [sys.executable, RUNNER, runner_args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"    ERROR: runner exit code {result.returncode}", file=sys.stderr)
            stderr_tail = result.stderr.strip().split("\n")[-3:]
            for l in stderr_tail:
                print(f"    stderr: {l}", file=sys.stderr)
            return None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return None
    except subprocess.TimeoutExpired:
        print(f"    ERROR: runner timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    ERROR: {e}", file=sys.stderr)
        return None


def get_cuda_kernel_name(kernel):
    if kernel == "k_last_packed":
        return "fused_sigmoid_gating_delta_kernel_packed"
    return "fused_sigmoid_gating_delta_kernel"


def profile_case(B, H, HV, K, V, dtype_str, is_kda, kernel, mtp_T, warmup, mode="acu", iters=10):
    base_config = {
        "B": B, "H": H, "HV": HV, "K": K, "V": V,
        "kernel": kernel, "dtype": dtype_str, "is_kda": is_kda,
        "mtp_T": mtp_T, "warmup": warmup, "iters": iters,
    }

    cuda_config = {**base_config, "backend": "cuda"}
    triton_config = {**base_config, "backend": "triton"}

    if mode == "acu":
        cuda_kernel_name = get_cuda_kernel_name(kernel)
        cuda_metrics = run_acu(cuda_config, kernel_name=cuda_kernel_name, warmup=warmup)
        triton_metrics = run_acu(triton_config, kernel_name=None, warmup=warmup)
    else:
        cuda_metrics = run_events(cuda_config)
        triton_metrics = run_events(triton_config)

    return cuda_metrics, triton_metrics


def flush_csvs(results_by_kernel, output_dir):
    """Write current results to CSV files (append if file exists with header)."""
    for kernel, rows in results_by_kernel.items():
        if not rows:
            continue
        csv_path = os.path.join(output_dir, f"perf_{kernel}.csv")
        file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(CSV_HEADER)
            writer.writerows(rows)


def write_triton_wins(results_by_kernel, output_dir):
    triton_wins = []
    for kernel, rows in results_by_kernel.items():
        for row in rows:
            ratio = float(row[10])
            if 0 < ratio < 1.0:
                B, H, HV, K, V, dtype_str = row[1], row[2], row[3], row[4], row[5], row[6]
                is_kda = row[7]
                prefix = "KDA" if is_kda else "GDN"
                attn = "kda_attn" if is_kda else "gdn_attn"
                line = (
                    f"[{prefix}]:[D][{B}]{attn}:seqlen_q:1,head_dim_qk:{K},"
                    f"head_dim_v:{V},num_heads_qk:{H},num_heads_v:{HV},"
                    f"hidden_act:silu,conv_kdim:4,batch_size:{B},dtype:{dtype_str}."
                )
                triton_wins.append((kernel, ratio, line))

    triton_list_path = os.path.join(output_dir, "triton_wins.list")
    if triton_wins:
        with open(triton_list_path, "w") as f:
            f.write("# Cases where Triton kernel has better performance than CUDA\n")
            f.write(f"# Generated by perf_compare.py on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Format: kernel_type time_ratio(triton/cuda) caseline\n")
            for kernel, ratio, line in triton_wins:
                f.write(f"# [{kernel}] ratio={ratio:.3f}\n")
                f.write(f"{line}\n")
        print(f"Wrote {triton_list_path} ({len(triton_wins)} Triton-wins cases)")
    else:
        with open(triton_list_path, "w") as f:
            f.write("# No cases where Triton beats CUDA\n")
        print("No cases where Triton beats CUDA.")
    return len(triton_wins)


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA vs Triton kernel cycles across caselists")
    parser.add_argument("--caselist", type=str, default=None,
                        help="Specific caselist file. If not provided, runs all under caselist/decode/")
    parser.add_argument("--mtp-tokens", type=int, default=4,
                        help="Token count for k_last_mtp (default: 4)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output CSV files (default: .)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup launches (default: 3)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed iterations per backend in events mode (default: 10)")
    parser.add_argument("--kernels", nargs="+", default=["k_last", "k_last_mtp", "k_last_packed"],
                        choices=["k_last", "k_last_mtp", "k_last_packed"],
                        help="Kernel types to benchmark")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop on first failure")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--mode", type=str, default="events", choices=["acu", "events"],
                        help="Profiling mode: acu (hardware counters) or events (CUDA event timing, default)")
    parser.add_argument("--max-mem-gb", type=float, default=8.0,
                        help="Max estimated GPU memory per case in GB; skip larger cases (default: 8)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    caselist_dir = os.path.join(PROJECT_ROOT, "caselist", "decode")
    if args.caselist:
        caselist_files = [args.caselist]
    else:
        caselist_files = sorted(glob.glob(os.path.join(caselist_dir, "*.list")))

    if not caselist_files:
        print(f"ERROR: No caselist files found in {caselist_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    results_by_kernel = {k: [] for k in args.kernels}
    # CSVs are flushed and cleared per caselist; this keeps every row so the
    # end-of-run Triton-wins list and row summary see the whole sweep.
    all_results_by_kernel = {k: [] for k in args.kernels}

    print("=" * 70)
    print("CUDA vs Triton Performance Comparison")
    print("=" * 70)
    print(f"Kernels: {', '.join(args.kernels)}")
    print(f"Mode: {args.mode}")
    print(f"MTP tokens: {args.mtp_tokens}")
    print(f"Warmup: {args.warmup}")
    print(f"Max mem: {args.max_mem_gb:.1f} GB")
    print(f"Caselist files: {len(caselist_files)}")
    print(f"Output dir: {args.output_dir}")
    print()

    total_cases = 0
    total_ok = 0
    total_fail = 0
    total_skip = 0
    start_time = time.time()

    for filepath in caselist_files:
        filename = os.path.basename(filepath)
        file_is_kda = "kda" in filename.lower()
        cases = read_caselist(filepath)
        if not cases:
            print(f"[SKIP] {filename} (empty)")
            continue

        print(f"[FILE] {filename} ({len(cases)} cases, kda={file_is_kda})")
        print("-" * 50)

        for case in cases:
            # A line-level marker (kda_attn/gdn_attn) wins over the filename;
            # the filename heuristic stays as the fallback for plain files.
            is_kda = case.pop("_is_kda", file_is_kda)
            B = case.get("batch_size", 1)
            K = case.get("head_dim_qk", 128)
            V = case.get("head_dim_v", 128)
            H = case.get("num_heads_qk", 4)
            HV = case.get("num_heads_v", 8)
            dtype_str = case.get("dtype", "bf16")

            for kernel in args.kernels:
                mem_gb = estimate_mem_gb(B, H, HV, K, V, kernel, args.mtp_tokens, is_kda)
                if mem_gb > args.max_mem_gb:
                    tag = f"B={B},H={H},HV={HV},K={K},V={V},{dtype_str},{kernel}"
                    print(f"  [SKIP] {tag} (est {mem_gb:.1f}GB > {args.max_mem_gb}GB)")
                    total_skip += 1
                    continue

                tag = f"B={B},H={H},HV={HV},K={K},V={V},{dtype_str},kda={is_kda},{kernel}"
                print(f"  [{kernel}] {tag} ... ", end="", flush=True)
                total_cases += 1

                cuda_m, triton_m = profile_case(
                    B, H, HV, K, V, dtype_str, is_kda, kernel,
                    args.mtp_tokens, args.warmup, args.mode, args.iters
                )

                if cuda_m is None or triton_m is None:
                    print("FAIL (no metrics)")
                    total_fail += 1
                    if args.stop_on_fail:
                        print("Stopping on first failure.")
                        flush_csvs(results_by_kernel, args.output_dir)
                        write_triton_wins(
                            {k: all_results_by_kernel[k] + results_by_kernel[k]
                             for k in results_by_kernel},
                            args.output_dir,
                        )
                        sys.exit(1)
                    continue

                if args.mode == "acu":
                    cuda_cycles = cuda_m.get("ce__cycles_elapsed.max", 0)
                    triton_cycles = triton_m.get("ce__cycles_elapsed.max", 0)
                    cycle_ratio = triton_cycles / cuda_cycles if cuda_cycles > 0 else 0
                    cuda_time = cuda_m.get("ppu__time_duration.sum", 0)
                    triton_time = triton_m.get("ppu__time_duration.sum", 0)
                    cuda_tensor = cuda_m.get("cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0)
                    triton_tensor = triton_m.get("cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 0)
                    cuda_dram = cuda_m.get("dram__bytes_read.sum.pct_of_peak_sustained_elapsed", 0)
                    triton_dram = triton_m.get("dram__bytes_read.sum.pct_of_peak_sustained_elapsed", 0)
                    cuda_host = 0
                    triton_host = 0
                    host_ratio = 0
                    cuda_e2e = 0
                    triton_e2e = 0
                    e2e_ratio = 0
                else:
                    cuda_cycles = 0
                    triton_cycles = 0
                    cuda_time = cuda_m.get("time_us", 0)
                    triton_time = triton_m.get("time_us", 0)
                    cycle_ratio = triton_time / cuda_time if cuda_time > 0 else 0
                    cuda_tensor = 0
                    triton_tensor = 0
                    cuda_dram = 0
                    triton_dram = 0
                    cuda_host = cuda_m.get("host_us", 0)
                    triton_host = triton_m.get("host_us", 0)
                    host_ratio = triton_host / cuda_host if cuda_host > 0 else 0
                    cuda_e2e = cuda_m.get("e2e_us", 0)
                    triton_e2e = triton_m.get("e2e_us", 0)
                    e2e_ratio = triton_e2e / cuda_e2e if cuda_e2e > 0 else 0

                row = [
                    filename, B, H, HV, K, V, dtype_str, is_kda,
                    int(cuda_cycles), int(triton_cycles), f"{cycle_ratio:.3f}",
                    f"{cuda_time:.1f}", f"{triton_time:.1f}",
                    f"{cuda_tensor:.1f}", f"{triton_tensor:.1f}",
                    f"{cuda_dram:.1f}", f"{triton_dram:.1f}",
                    f"{cuda_host:.1f}", f"{triton_host:.1f}", f"{host_ratio:.3f}",
                    f"{cuda_e2e:.1f}", f"{triton_e2e:.1f}", f"{e2e_ratio:.3f}",
                ]
                results_by_kernel[kernel].append(row)

                winner = "CUDA" if cycle_ratio > 1 else "Triton"
                if args.mode == "events":
                    host_info = f" host: cuda={cuda_host:.1f}us triton={triton_host:.1f}us"
                    print(f"dev: cuda={cuda_time:.1f}us triton={triton_time:.1f}us ratio={cycle_ratio:.2f}x ({winner}){host_info}")
                else:
                    print(f"cuda={int(cuda_cycles)} triton={int(triton_cycles)} ratio={cycle_ratio:.2f}x ({winner})")
                total_ok += 1

        # Flush CSVs after each caselist file
        flush_csvs(results_by_kernel, args.output_dir)
        # Clear accumulated results to avoid duplicates on next append, but keep
        # a cumulative copy: write_triton_wins and the row summary below run
        # after the loop, so clearing alone made them see empty lists — the
        # Triton-wins list was reported as empty on every run.
        for k in results_by_kernel:
            all_results_by_kernel[k].extend(results_by_kernel[k])
            results_by_kernel[k] = []
        print(f"  [flushed CSVs to {args.output_dir}]")
        print()

    write_triton_wins(all_results_by_kernel, args.output_dir)

    elapsed = time.time() - start_time
    total_rows = sum(len(v) for v in all_results_by_kernel.values())
    for kernel in args.kernels:
        rows = all_results_by_kernel[kernel]
        if rows:
            csv_path = os.path.join(args.output_dir, f"perf_{kernel}.csv")
            print(f"  {csv_path} ({len(rows)} rows)")

    print()
    print("=" * 70)
    print(f"Done. {total_ok} profiled, {total_fail} failed, {total_skip} skipped, {elapsed:.0f}s elapsed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
