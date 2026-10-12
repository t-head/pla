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

"""Runner for perf_compare.py: launched under acu or standalone for CUDA-event timing.

Input: JSON on argv[1] with {B, H, HV, K, V, kernel, backend, mtp_T, is_kda, dtype, warmup, mode}
  kernel  ∈ {k_last, k_last_mtp, k_last_packed}
  backend ∈ {cuda, triton}
  mode    ∈ {acu, events}  (default: acu)

acu mode:  warmup launches skipped by acu --launch-skip; the final launch is profiled.
events mode: warmup + timed launches using torch.cuda.Event; prints JSON result to stdout.
"""
import json, sys, os, time, torch

args = json.loads(sys.argv[1])
if "seed" in args:
    torch.manual_seed(args["seed"])
torch.cuda.set_device(0)
device = "cuda:0"

B = args["B"]
H, HV, K, V = args["H"], args["HV"], args["K"], args["V"]
kernel = args["kernel"]
backend = args["backend"]
mtp_T = args.get("mtp_T", 4)
is_kda = args.get("is_kda", False)
warmup = args.get("warmup", 3)

dtype_str = args.get("dtype", "bf16")
dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype_str, torch.bfloat16)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

A_log = torch.randn(HV, dtype=torch.float32, device=device)
indices = torch.arange(B, dtype=torch.int32, device=device)

if is_kda:
    # Only the KDA MTP-verify kernel reads dt_bias as fp32. k_last here is one token
    # per request (pure decode) and k_last_packed builds its own dt_bias below, both
    # of which read it at the activation dtype.
    dt_bias = torch.randn(HV, K, device=device,
                          dtype=torch.float32 if kernel == "k_last_mtp" else dtype)
else:
    dt_bias = torch.randn(HV, dtype=dtype, device=device)


def make_run_fn():
    if kernel == "k_last":
        T = 1
        if is_kda:
            a = torch.randn(1, B, HV, K, dtype=dtype, device=device)
        else:
            a = torch.randn(1, B, HV, dtype=dtype, device=device)
        b = torch.randn(1, B, HV, dtype=dtype, device=device)
        q = torch.randn(1, B, H, K, dtype=dtype, device=device)
        k = torch.randn(1, B, H, K, dtype=dtype, device=device)
        v = torch.randn(1, B, HV, V, dtype=dtype, device=device)
        state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=device)

        if backend == "cuda":
            from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last
            state_work = state.clone()
            def run():
                # NOTE: no state.clone() here — sglang production passes the
                # persistent state pool directly, so clone must stay out of
                # the timed region.
                fused_sigmoid_gating_delta_rule_forward_k_last(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    state_work, indices, 1.0, True, cu, is_kda,
                    False, None, None, None, None)
        else:
            from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
            state_work = state.clone()
            def run():
                fused_sigmoid_gating_delta_rule_update(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    state_work, indices, scale=1.0,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=is_kda)
        return run

    elif kernel == "k_last_mtp":
        T = mtp_T
        if is_kda:
            a = torch.randn(B * T, HV, K, dtype=dtype, device=device)
            # A/B uses the flattened layout that both versions support correctly, avoiding the baseline's old token-stride bug.
            if args.get("flat_kda_gate", False):
                a = a.view(B * T, HV * K)
        else:
            a = torch.randn(B * T, HV, dtype=dtype, device=device)
        b = torch.randn(B * T, HV, dtype=dtype, device=device)
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
        state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
        cu = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)
        parent_list = [0] + [t for t in range(T - 1)]
        rpt = torch.tensor([parent_list] * B, dtype=torch.int64, device=device).contiguous()
        isb = torch.zeros(B * T, HV, V, K, dtype=torch.float32, device=device)

        if backend == "cuda":
            from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last
            state_work = state.clone()
            def run():
                fused_sigmoid_gating_delta_rule_forward_k_last(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    state_work, indices, 1.0, True, cu, is_kda,
                    True, isb, indices, T, rpt)
        else:
            from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
            state_work = state.clone()
            isb_work = isb.clone()
            def run():
                fused_sigmoid_gating_delta_rule_update(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    state_work, indices, scale=1.0,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=is_kda,
                    disable_state_update=True,
                    intermediate_states_buffer=isb_work,
                    intermediate_state_indices=indices,
                    cache_steps=T, retrieve_parent_token=rpt)
        return run

    elif kernel == "k_last_packed":
        qkv_dim = 2 * H * K + HV * V
        mixed = torch.randn(B, qkv_dim, device=device, dtype=dtype)
        state = torch.randn(B + 16, HV, V, K, device=device, dtype=torch.float32) * 0.1
        cu = torch.arange(B + 1, device=device, dtype=torch.int32)
        out = mixed.new_empty(B, 1, HV, V)

        if is_kda:
            a_packed = torch.randn(B, HV * K, device=device, dtype=dtype)
            bt = torch.randn(B, HV, device=device, dtype=dtype)
            dt_bias_kda = torch.randn(HV * K, dtype=dtype, device=device)
        else:
            a_packed = torch.randn(B, HV, device=device, dtype=dtype)
            bt = torch.randn(B, HV, device=device, dtype=dtype)

        if backend == "cuda":
            from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last_packed
            state_work = state.clone()
            out_work = out.clone()
            def run():
                fused_sigmoid_gating_delta_rule_forward_k_last_packed(
                    mixed, a_packed, bt, A_log,
                    dt_bias_kda if is_kda else dt_bias,
                    1.0, 20.0,
                    K ** -0.5, state_work, indices, out_work,
                    True, cu, is_kda)
        else:
            state_work = state.clone()
            out_work = out.clone()
            if is_kda:
                from fused_recurrent import fused_recurrent_kda_packed_decode
                def run():
                    fused_recurrent_kda_packed_decode(
                        mixed, a_packed, bt, A_log, dt_bias_kda,
                        K ** -0.5, state_work, out_work,
                        indices, use_qk_l2norm_in_kernel=True)
            else:
                from fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
                def run():
                    fused_recurrent_gated_delta_rule_packed_decode(
                        mixed, a_packed, bt, A_log, dt_bias,
                        K ** -0.5, state_work, out_work,
                        indices, use_qk_l2norm_in_kernel=True)
        return run
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


run_fn = make_run_fn()
mode = args.get("mode", "acu")

if mode == "events":
    num_iters = args.get("iters", 10)
    for _ in range(warmup):
        try:
            run_fn()
        except Exception:
            if args.get("strict", False):
                raise
    torch.cuda.synchronize()

    # Device kernel time via CUDA events
    dev_times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_fn()
        e.record()
        torch.cuda.synchronize()
        dev_times.append(s.elapsed_time(e) * 1000)  # us

    # Host time (async call-to-return, pure host overhead)
    host_times = []
    for _ in range(num_iters):
        t0 = time.perf_counter_ns()
        run_fn()
        t1 = time.perf_counter_ns()
        host_times.append((t1 - t0) / 1000.0)  # ns -> us
    torch.cuda.synchronize()

    # E2E time (host + device + sync)
    e2e_times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        run_fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        e2e_times.append((t1 - t0) / 1000.0)  # ns -> us

    def iqm(times_list):
        # warmup stays at 3; here we take the fastest (min) of several timed iterations:
        # min is the least polluted by DVFS ramp-up / cold-start jitter and closest to the kernel's true peak performance.
        times_list.sort()
        n = len(times_list)
        med = times_list[n // 2]
        min_val = times_list[0]
        return round(min_val, 2), round(med, 2)

    dev_iqm, dev_med = iqm(dev_times)
    host_iqm, host_med = iqm(host_times)
    e2e_iqm, e2e_med = iqm(e2e_times)
    print(json.dumps({
        "time_us": dev_iqm, "median_us": dev_med,
        "host_us": host_iqm, "host_median_us": host_med,
        "e2e_us": e2e_iqm, "e2e_median_us": e2e_med,
    }))
else:
    for _ in range(warmup):
        try:
            run_fn()
        except Exception:
            if args.get("strict", False):
                raise
    torch.cuda.synchronize()

    run_fn()
    torch.cuda.synchronize()
