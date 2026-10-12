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

"""Benchmark a single case with specific kernel type and template configuration.

Usage:
    # k_last decode, B=64, H=16, HV=64, K=128, V=128, BV=64, WN=8, SN=4, HV_PER_CTA=1
    python scripts/decode/bench_single.py --kernel k_last --case 16,64,128,128,64 --config 64,8,4,1

    # k_last_mtp, B=4, T=8
    python scripts/decode/bench_single.py --kernel k_last_mtp --case 4,8,128,128,4 --config 64,8,4,1 --mtp-T 8

    # k_last_packed with native BF16 recurrent state
    python scripts/decode/bench_single.py --kernel k_last_packed --case 8,16,128,128,32 --config 64,8,4,1 --state-dtype bf16

    # Use default tuned config (omit --config)
    python scripts/decode/bench_single.py --kernel k_last --case 16,64,128,128,64

    # Correctness check against Triton reference
    python scripts/decode/bench_single.py --kernel k_last --case 16,32,128,128,64 --check
"""

import argparse
import sys
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a single kernel case")
    parser.add_argument("--kernel", choices=["k_last", "k_last_mtp", "k_last_packed"],
                        required=True, help="Kernel type")
    parser.add_argument("--case", type=str, required=True,
                        help="H,HV,K,V,B (e.g. '16,64,128,128,64')")
    parser.add_argument("--config", type=str, default=None,
                        help="BV,WN,SN,HV_PER_CTA (e.g. '64,8,4,1'). If omitted, uses tuned dispatch.")
    parser.add_argument("--mtp-T", type=int, default=4,
                        help="Sequence length T for k_last_mtp (default: 4)")
    parser.add_argument("--is-kda", action="store_true", help="Enable KDA mode")
    parser.add_argument("--state-dtype", choices=["fp32", "bf16"], default="fp32",
                        help="Recurrent state dtype for k_last_packed (default: fp32)")
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--check", action="store_true",
                        help="Run correctness check against Triton reference")
    return parser.parse_args()


def main():
    args = parse_args()

    parts = [int(x) for x in args.case.split(",")]
    if len(parts) != 5:
        print("Error: --case expects H,HV,K,V,B")
        sys.exit(1)
    H, HV, K, V, B = parts

    bv, wn, sn, hpc = None, None, None, None
    if args.config:
        cfg = [int(x) for x in args.config.split(",")]
        if len(cfg) != 4:
            print("Error: --config expects BV,WN,SN,HV_PER_CTA")
            sys.exit(1)
        bv, wn, sn, hpc = cfg

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    print(f"Kernel: {args.kernel}")
    print(f"Case: B={B}, H={H}, HV={HV}, K={K}, V={V}, is_kda={args.is_kda}")
    if args.kernel == "k_last_packed":
        print(f"State dtype: {args.state_dtype}")
    if args.config:
        print(f"Config: BV={bv}, WN={wn}, SN={sn}, HV_PER_CTA={hpc}")
    else:
        print(f"Config: tuned dispatch (auto)")
    print()

    if args.kernel == "k_last":
        from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last
        from pla.decode.private import fused_sigmoid_gating_delta_rule_forward_k_last_tunable

        A_log = torch.randn(HV, dtype=torch.float32, device=device)
        if args.is_kda:
            # One token per request and no cache buffers: pure decode, so the KDA
            # kernel reads dt_bias at the activation dtype (fp32 is MTP-only).
            dt_bias = torch.randn(HV, K, dtype=dtype, device=device)
            a = torch.randn(1, B, HV, K, dtype=dtype, device=device).view(B, -1).contiguous()
        else:
            dt_bias = torch.randn(HV, dtype=dtype, device=device)
            a = torch.randn(1, B, HV, dtype=dtype, device=device)
        b = torch.randn(1, B, HV, dtype=dtype, device=device)
        q = torch.randn(1, B, H, K, dtype=dtype, device=device)
        k = torch.randn(1, B, H, K, dtype=dtype, device=device)
        v = torch.randn(1, B, HV, V, dtype=dtype, device=device)
        state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
        indices = torch.arange(B, dtype=torch.int32, device=device)
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=device)

        if args.config:
            def run_fn(s):
                return fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    s, indices, 1.0, True, cu, args.is_kda, False,
                    None, None, None, None, bv, wn, sn, hpc)
        else:
            def run_fn(s):
                return fused_sigmoid_gating_delta_rule_forward_k_last(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    s, indices, 1.0, True, cu, args.is_kda, False,
                    None, None, None, None)

        if args.check:
            from fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as triton_fn
            s_ref = state.clone()
            out_ref = triton_fn(A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                                s_ref, indices, scale=1.0, use_qk_l2norm_in_kernel=True,
                                cu_seqlens=cu, is_kda=args.is_kda, disable_state_update=False)
            s_test = state.clone()
            out_test = run_fn(s_test)
            torch.cuda.synchronize()
            diff = (out_test - out_ref).abs()
            print(f"Correctness: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
            sdiff = (s_test - s_ref).abs()
            print(f"State: max_diff={sdiff.max().item():.6f}")
            try:
                torch.testing.assert_close(out_test, out_ref, atol=1e-2, rtol=1e-2)
                print("PASS")
            except AssertionError:
                print("FAIL")
            print()

    elif args.kernel == "k_last_mtp":
        from pla.decode.private import fused_sigmoid_gating_delta_rule_forward_k_last_tunable

        T = args.mtp_T
        print(f"MTP: T={T}, disable_state_update=True, tree_attn=causal, intermediate_cache=True")

        A_log = torch.randn(HV, dtype=torch.float32, device=device)
        if args.is_kda:
            # MTP verify: the KDA kernel reads dt_bias as fp32 here.
            dt_bias = torch.randn(HV, K, dtype=torch.float32, device=device)
            a = torch.randn(1, B * T, HV, K, dtype=dtype, device=device).view(B * T, -1).contiguous()
        else:
            dt_bias = torch.randn(HV, dtype=dtype, device=device)
            a = torch.randn(1, B * T, HV, dtype=dtype, device=device).view(B * T, -1).contiguous()
        b = torch.randn(1, B * T, HV, dtype=dtype, device=device).view(B * T, -1).contiguous()
        q = torch.randn(1, B * T, H, K, dtype=dtype, device=device).view(B, T, H, K).contiguous()
        k = torch.randn(1, B * T, H, K, dtype=dtype, device=device).view(B, T, H, K).contiguous()
        v = torch.randn(1, B * T, HV, V, dtype=dtype, device=device).view(B, T, HV, V).contiguous()
        state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
        indices = torch.arange(B, dtype=torch.int32, device=device)
        cu = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=device)

        # Tree attention: standard causal parent mapping
        parent_list = [0] + [t for t in range(T - 1)]
        retrieve_parent_token = torch.tensor([parent_list] * B, dtype=torch.int32, device=device).contiguous()

        # Intermediate states buffer for caching
        cache_steps = T
        intermediate_states_buffer = torch.zeros(B * cache_steps, HV, V, K, dtype=torch.float32, device=device)
        intermediate_state_indices = indices

        if args.config:
            def run_fn(s):
                return fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    s, indices, 1.0, True, cu, args.is_kda, True,
                    intermediate_states_buffer, intermediate_state_indices,
                    cache_steps, retrieve_parent_token, bv, wn, sn, hpc)
        else:
            from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last
            def run_fn(s):
                return fused_sigmoid_gating_delta_rule_forward_k_last(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                    s, indices, 1.0, True, cu, args.is_kda, True,
                    intermediate_states_buffer, intermediate_state_indices,
                    cache_steps, retrieve_parent_token)

    elif args.kernel == "k_last_packed":
        from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last_packed
        from pla.decode.private import fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable

        qkv_dim = 2 * H * K + HV * V
        mixed_qkv = torch.randn(B, qkv_dim, device=device, dtype=dtype)
        A_log = torch.randn(HV, device=device, dtype=torch.float32)
        if args.is_kda:
            dt_bias = torch.randn(HV, K, device=device, dtype=dtype)
            a = torch.randn(B, HV * K, device=device, dtype=dtype)
        else:
            dt_bias = torch.randn(HV, device=device, dtype=dtype)
            a = torch.randn(B, HV, device=device, dtype=dtype)
        b = torch.randn(B, HV, device=device, dtype=dtype)
        state_dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
        state = torch.randn(B + 16, HV, V, K, device=device, dtype=state_dtype) * 0.1
        indices = torch.arange(B, device=device, dtype=torch.int32)
        cu = torch.arange(B + 1, device=device, dtype=torch.int32)
        out = mixed_qkv.new_empty(B, 1, HV, V)

        if args.config:
            def run_fn(s):
                o = out.clone()
                return fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable(
                    mixed_qkv, a, b, A_log, dt_bias, 1.0, 20.0,
                    K ** -0.5, s, indices, o, True, cu, args.is_kda,
                    bv, wn, sn, hpc)
        else:
            def run_fn(s):
                o = out.clone()
                return fused_sigmoid_gating_delta_rule_forward_k_last_packed(
                    mixed_qkv, a, b, A_log, dt_bias, 1.0, 20.0,
                    K ** -0.5, s, indices, o, True, cu, args.is_kda)

    # Benchmark
    print(f"Benchmarking ({args.num_warmup} warmup, {args.num_iters} iters)...")
    for _ in range(args.num_warmup):
        run_fn(state.clone())
    torch.cuda.synchronize()

    times = []
    for _ in range(args.num_iters):
        s = state.clone()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_fn(s)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)

    times.sort()
    n = len(times)
    median = times[n // 2]
    iqr_mean = sum(times[n // 4: 3 * n // 4]) / (n // 2)
    print(f"  Median:   {median:.1f} us")
    print(f"  IQR mean: {iqr_mean:.1f} us")
    print(f"  Min:      {min(times):.1f} us")
    print(f"  Max:      {max(times):.1f} us")
    if args.kernel == "k_last_packed":
        state_bytes = 2 * B * HV * V * K * state.element_size()
        print(f"  Effective state bandwidth: {state_bytes / (median * 1e3):.1f} GB/s")


if __name__ == "__main__":
    main()
