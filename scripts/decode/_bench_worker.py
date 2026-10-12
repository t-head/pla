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

"""Batch worker: benchmarks ALL combos x ALL batch sizes for one (H,HV,K,V) shape in-process."""
import json, sys, torch

args = json.loads(sys.argv[1])
torch.cuda.set_device(0)
device = "cuda:0"

H, HV, K, V = args["H"], args["HV"], args["K"], args["V"]
combos = [tuple(c) for c in args["combos"]]
batch_sizes = args["batch_sizes"]
num_warmup = args.get("warmup", 3)
num_iters = args.get("iters", 10)
dtype = torch.bfloat16
packed_state_dtype = torch.bfloat16 if args.get("state_dtype", "fp32") == "bf16" else torch.float32

from pla.decode.private import fused_sigmoid_gating_delta_rule_forward_k_last_tunable
from pla.decode.private import fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable

failed_combos = set()

def bench_k_last(B, bv, wn, sn, hpc):
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=dtype, device=device)
    a = torch.randn(1, B, HV, dtype=dtype, device=device)
    b = torch.randn(1, B, HV, dtype=dtype, device=device)
    q = torch.randn(1, B, H, K, dtype=dtype, device=device)
    k = torch.randn(1, B, H, K, dtype=dtype, device=device)
    v = torch.randn(1, B, HV, V, dtype=dtype, device=device)
    state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
    indices = torch.arange(B, dtype=torch.int32, device=device)
    cu = torch.arange(0, B + 1, dtype=torch.int32, device=device)
    def run():
        return fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
            A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
            state, indices, 1.0, True, cu, False, False,
            None, None, None, None, bv, wn, sn, hpc)
    return run

def bench_k_last_mtp(B, T, bv, wn, sn, hpc):
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=dtype, device=device)
    a = torch.randn(B * T, HV, dtype=dtype, device=device)
    b = torch.randn(B * T, HV, dtype=dtype, device=device)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
    indices = torch.arange(B, dtype=torch.int32, device=device)
    cu = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
    # With-cache MTP verify (aligns with sglang target_verify): intermediate state
    # buffer + tree parent tokens are passed, matching _acu_runner.py's ACU path.
    parent_list = [0] + [t for t in range(T - 1)]
    rpt = torch.tensor([parent_list] * B, dtype=torch.int64, device=device).contiguous()
    isb = torch.zeros(B * T, HV, V, K, dtype=torch.float32, device=device)
    def run():
        return fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
            A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
            state, indices, 1.0, True, cu, False, True,
            isb, indices, T, rpt, bv, wn, sn, hpc)
    return run

def bench_k_last_packed(B, bv, wn, sn, hpc):
    qkv_dim = 2 * H * K + HV * V
    mixed = torch.randn(B, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(B, HV, device=device, dtype=dtype)
    bt = torch.randn(B, HV, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=device, dtype=dtype)
    state = torch.randn(B + 16, HV, V, K, device=device, dtype=packed_state_dtype) * 0.1
    indices = torch.arange(B, device=device, dtype=torch.int32)
    cu = torch.arange(B + 1, device=device, dtype=torch.int32)
    out = mixed.new_empty(B, 1, HV, V)
    def run():
        return fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable(
            mixed, a, bt, A_log, dt_bias, 1.0, 20.0,
            K ** -0.5, state, indices, out, True, cu, False, bv, wn, sn, hpc)
    return run

import os, subprocess, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACU_RUNNER = os.path.join(SCRIPT_DIR, "_acu_runner.py")

ACU_CHUNK_SIZE = 10

def _acu_chunk(B, combos_chunk, kernel, mtp_T):
    """Run acu on a chunk of combos. Returns dict {combo: cycles}."""
    runner_args = json.dumps({
        "B": B, "H": H, "HV": HV, "K": K, "V": V,
        "combos": [list(c) for c in combos_chunk],
        "kernel": kernel,
        "mtp_T": mtp_T,
        "state_dtype": args.get("state_dtype", "fp32"),
    })
    num = len(combos_chunk)
    kernel_name = "fused_sigmoid_gating_delta_kernel_packed" if kernel == "k_last_packed" else "fused_sigmoid_gating_delta_kernel"
    cmd = [
        "acu", "--clock-control", "none",
        "--kernel-name", kernel_name,
        "--metrics", "ce__cycles_elapsed.max",
        "--page=details",
        "--launch-skip", str(num),
        "--launch-count", str(num),
        sys.executable, ACU_RUNNER, runner_args,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + "\n" + result.stderr
        cycles_list = []
        for line in output.split("\n"):
            if "ce__cycles_elapsed.max" in line:
                match = re.search(r"(\d+)", line.split("ce__cycles_elapsed.max")[1])
                if match:
                    cycles_list.append(int(match.group(1)))
        results = {}
        for i, combo in enumerate(combos_chunk):
            if i < len(cycles_list):
                results[combo] = cycles_list[i]
        return results
    except Exception:
        return {}


def measure_acu_batch(B, combos_list, kernel, mtp_T=4):
    """Measure all combos for one (B, kernel), chunked into ACU_CHUNK_SIZE per acu call."""
    all_results = {}
    for i in range(0, len(combos_list), ACU_CHUNK_SIZE):
        chunk = combos_list[i:i + ACU_CHUNK_SIZE]
        chunk_results = _acu_chunk(B, chunk, kernel, mtp_T)
        all_results.update(chunk_results)
    return all_results

def measure_events(run_fn):
    """Fallback: measure using CUDA events. Returns latency in us."""
    for _ in range(num_warmup):
        run_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); run_fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    n = len(times)
    return sum(times[n//4:3*n//4]) / max(1, n//2)

mtp_T = args.get("mtp_T", 4)
kernels = args.get("kernels") or ["k_last", "k_last_mtp", "k_last_packed"]
detailed = args.get("detailed", False)
use_acu = args.get("use_acu", True)

# Pre-validate combos per kernel (ACU path). A combo that fails to launch a kernel
# breaks acu's --launch-skip/--launch-count accounting (warmup+profiled launch counts
# get misaligned), yielding ZERO profiled kernels and an empty result CSV. Test-launch
# each combo once with a minimal batch to drop invalid configs up-front, so every combo
# sent to acu produces exactly one warmup + one profiled launch.
valid_by_kernel = {}
if use_acu:
    for _kernel in kernels:
        _good = []
        for _combo in combos:
            _bv, _wn, _sn, _hpc = _combo
            try:
                if _kernel == "k_last":
                    _fn = bench_k_last(1, _bv, _wn, _sn, _hpc)
                elif _kernel == "k_last_mtp":
                    _fn = bench_k_last_mtp(1, mtp_T, _bv, _wn, _sn, _hpc)
                else:
                    _fn = bench_k_last_packed(1, _bv, _wn, _sn, _hpc)
                _fn()
                torch.cuda.synchronize()
                _good.append(_combo)
            except Exception:
                pass
            torch.cuda.empty_cache()
        valid_by_kernel[_kernel] = _good
        print(json.dumps({"type": "info", "kernel": _kernel,
                          "msg": f"prevalidated {len(_good)}/{len(combos)} combos"}), flush=True)

for B in batch_sizes:
    if use_acu:
        # Batched acu: one acu call per kernel type, profiling only pre-validated combos
        for kernel in kernels:
            valid_combos = valid_by_kernel.get(kernel, [])
            if not valid_combos:
                continue
            acu_results = measure_acu_batch(B, valid_combos, kernel, mtp_T)
            best_combo = None
            best_time = float("inf")
            for combo in valid_combos:
                metric_val = acu_results.get(combo)
                if metric_val is None:
                    continue
                if detailed:
                    print(json.dumps({"type": "detail", "kernel": kernel, "B": B,
                                      "combo": list(combo), "metric": round(metric_val, 1),
                                      "unit": "cycles"}), flush=True)
                if metric_val < best_time:
                    best_time = metric_val
                    best_combo = list(combo)
            if best_combo is not None:
                print(json.dumps({"type": "result", "kernel": kernel, "B": B,
                                  "combo": best_combo, "metric": round(best_time, 1),
                                  "unit": "cycles"}), flush=True)
        continue

    for kernel in kernels:
        best_combo = None
        best_time = float("inf")
        for combo in combos:
            if combo in failed_combos:
                continue
            bv, wn, sn, hpc = combo
            try:
                if kernel == "k_last":
                    run_fn = bench_k_last(B, bv, wn, sn, hpc)
                elif kernel == "k_last_mtp":
                    run_fn = bench_k_last_mtp(B, mtp_T, bv, wn, sn, hpc)
                else:
                    run_fn = bench_k_last_packed(B, bv, wn, sn, hpc)
                metric_val = measure_events(run_fn)
                metric_unit = "us"
                del run_fn
                torch.cuda.empty_cache()
                if detailed:
                    print(json.dumps({"type": "detail", "kernel": kernel, "B": B,
                                      "combo": list(combo), "metric": round(metric_val, 1),
                                      "unit": metric_unit}), flush=True)
                if metric_val < best_time:
                    best_time = metric_val
                    best_combo = list(combo)
            except Exception as ex:
                failed_combos.add(combo)
                torch.cuda.empty_cache()
                try:
                    torch.cuda.synchronize()
                except:
                    print(json.dumps({"type": "fatal", "error": str(ex)}), flush=True)
                    sys.exit(1)
        if best_combo is not None:
            print(json.dumps({"type": "result", "kernel": kernel, "B": B,
                              "combo": best_combo, "metric": round(best_time, 1),
                              "unit": "cycles" if use_acu else "us"}), flush=True)

print(json.dumps({"type": "done", "failed": [list(c) for c in failed_combos]}), flush=True)
