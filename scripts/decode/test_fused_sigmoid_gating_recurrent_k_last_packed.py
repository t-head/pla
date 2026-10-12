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

"""
Benchmark & Correctness: GDN Packed Decode vs Baseline Decode.

Compares:
  - Baseline: split(mixed_qkv) → view → fused_sigmoid_gating_delta_rule_update
  - Packed:   fused_recurrent_gated_delta_rule_packed_decode (single kernel)

The packed path eliminates:
  - torch.split() + .view() tensor materialization
  - Separate gating kernel launches
  - Intermediate tensor allocations

Reports correctness (output & state matching) and performance (ms, speedup).

Usage:
    python bench_gdn_decode.py                        # default sweep
    python bench_gdn_decode.py --mode bench           # benchmark only
    python bench_gdn_decode.py --mode correctness     # correctness only
    python bench_gdn_decode.py --preset qwen3.5-35b   # Qwen3.5-35B-A3B config
"""

import argparse
import math
import os
import sys
import time

# Add scripts directory to path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import torch
import triton

from fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as triton_fused_gating,
)

from fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_kda_packed_decode,
)
from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last_packed
from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last

# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def make_inputs(
    B: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    device: str,
    dtype: torch.dtype,
    seed: int = 42,
    is_kda: bool = False,
    state_dtype: torch.dtype = torch.float32,
):
    """Create all input tensors for a single benchmark / correctness run."""
    torch.manual_seed(seed)

    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(B, qkv_dim, device=device, dtype=dtype)
    b = torch.randn(B, HV, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=torch.float32)

    if is_kda:
        a = torch.randn(B, HV * K, device=device, dtype=dtype)
        dt_bias = torch.randn(HV * K, device=device, dtype=dtype)
    else:
        a = torch.randn(B, HV, device=device, dtype=dtype)
        dt_bias = torch.randn(HV, device=device, dtype=dtype)

    ssm_states = torch.randn(pool_size, HV, V, K, device=device, dtype=state_dtype) * 0.1
    cache_indices = torch.arange(B, device=device, dtype=torch.int32)

    cu_seqlens = torch.arange(B + 1, device=device, dtype=torch.int32)

    return dict(
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        qkv_dim=qkv_dim,
        pool_size=pool_size,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        is_kda=is_kda,
    )


# ---------------------------------------------------------------------------
# Runner wrappers
# ---------------------------------------------------------------------------


def run_baseline(inp):
    """Baseline path: Triton packed decode kernel (GDN or KDA)."""
    B, H, HV, K, V = inp["B"], inp["H"], inp["HV"], inp["K"], inp["V"]
    is_kda = inp.get("is_kda", False)
    out_buf = inp["mixed_qkv"].new_empty(B, 1, HV, V)
    ssm_states = inp["ssm_states"].clone()

    if is_kda:
        fused_recurrent_kda_packed_decode(
            mixed_qkv=inp["mixed_qkv"],
            a=inp["a"],
            b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            scale=K**-0.5,
            initial_state=ssm_states,
            out=out_buf,
            ssm_state_indices=inp["cache_indices"],
            use_qk_l2norm_in_kernel=True,
        )
    else:
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=inp["mixed_qkv"],
            a=inp["a"],
            b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            scale=K**-0.5,
            initial_state=ssm_states,
            out=out_buf,
            ssm_state_indices=inp["cache_indices"],
            use_qk_l2norm_in_kernel=True,
        )

    return out_buf.transpose(0, 1), ssm_states

def run_packed(inp):
    """Packed path: single fused kernel directly on mixed_qkv."""
    B, HV, K, V = inp["B"], inp["HV"], inp["K"], inp["V"]
    is_kda = inp.get("is_kda", False)
    ssm_states = inp["ssm_states"].clone()
    out = inp["mixed_qkv"].new_empty(B, 1, HV, V)

    out = fused_sigmoid_gating_delta_rule_forward_k_last_packed(
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        1.0,  # softplus_beta
        20.0,  # softplus_threshold
        inp["K"] ** -0.5,  # scale
        ssm_states,  # initial_state_source
        inp["cache_indices"],  # initial_state_indices
        out,  # out
        True,  # use_qk_l2norm_in_kernel
        inp["cu_seqlens"],  # cu_seqlens
        is_kda,
    )

    # Convert [B, 1, HV, V] → [1, B, HV, V] to match baseline layout
    return out.transpose(0, 1), ssm_states


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness(B, H, HV, K, V, pool_size, device, dtype, seed=42, is_kda=False):
    """Run correctness check for a single config. Returns True if PASS."""
    kda_tag = " KDA" if is_kda else ""
    tag = f"B={B:>4} H={H:>2} HV={HV:>2} K={K:>3} V={V:>3} pool={pool_size:>4}{kda_tag}"

    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, seed=seed, is_kda=is_kda)

    o_baseline, state_baseline = run_baseline(inp)
    o_packed, state_packed = run_packed(inp)

    # Output comparison
    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4

    try:
        torch.testing.assert_close(o_packed, o_baseline, atol=atol, rtol=rtol)
        output_ok = True
    except AssertionError as e:
        output_ok = False
        out_diff = (o_packed - o_baseline).abs().max().item()

    # State comparison (only for slots that were updated)
    indices = inp["cache_indices"]
    try:
        torch.testing.assert_close(
            state_packed[indices], state_baseline[indices], atol=atol, rtol=rtol
        )
        state_ok = True
    except AssertionError:
        state_ok = False
        st_diff = (state_packed[indices] - state_baseline[indices]).abs().max().item()

    passed = output_ok and state_ok

    if passed:
        print(f"  [PASS] {tag}")
    else:
        details = []
        if not output_ok:
            details.append(f"output max_diff={out_diff:.6f}")
        if not state_ok:
            details.append(f"state max_diff={st_diff:.6f}")
        print(f"  [FAIL] {tag}  ({', '.join(details)})")

    return passed


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_shape(B, H, HV, K, V, pool_size, device, dtype, is_kda=False):
    """Benchmark baseline vs packed for a single config."""
    inp = make_inputs(B, H, HV, K, V, pool_size, device, dtype, is_kda=is_kda)

    # ── Baseline: Triton packed decode ──
    out_buf = inp["mixed_qkv"].new_empty(B, 1, HV, V)

    if is_kda:
        def fn_baseline():
            fused_recurrent_kda_packed_decode(
                mixed_qkv=inp["mixed_qkv"],
                a=inp["a"],
                b=inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                scale=K**-0.5,
                initial_state=inp["ssm_states"],
                out=out_buf,
                ssm_state_indices=inp["cache_indices"],
                use_qk_l2norm_in_kernel=True,
            )
    else:
        def fn_baseline():
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=inp["mixed_qkv"],
                a=inp["a"],
                b=inp["b"],
                A_log=inp["A_log"],
                dt_bias=inp["dt_bias"],
                scale=K**-0.5,
                initial_state=inp["ssm_states"],
                out=out_buf,
                ssm_state_indices=inp["cache_indices"],
                use_qk_l2norm_in_kernel=True,
            )

    # ── Packed: single kernel ──
    def fn_packed():
        fused_sigmoid_gating_delta_rule_forward_k_last_packed(
            inp["mixed_qkv"],
            inp["a"],
            inp["b"],
            inp["A_log"],
            inp["dt_bias"],
            1.0,  # softplus_beta
            20.0,  # softplus_threshold
            inp["K"] ** -0.5,  # scale
            inp["ssm_states"],  # initial_state_source
            inp["cache_indices"],  # initial_state_indices
            out_buf,  # out
            True,  # use_qk_l2norm_in_kernel
            inp["cu_seqlens"],  # cu_seqlens
            is_kda,
        )

    # Warmup
    for _ in range(10):
        fn_baseline()
        fn_packed()
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]

    try:
        ms_baseline, ms_base_lo, ms_base_hi = triton.testing.do_bench(
            fn_baseline, quantiles=quantiles, warmup=50, rep=200
        )
        ms_packed, ms_pack_lo, ms_pack_hi = triton.testing.do_bench(
            fn_packed, quantiles=quantiles, warmup=50, rep=200
        )
    except Exception:
        # Fallback to manual timing
        torch.cuda.synchronize()
        N = 200
        start = time.perf_counter()
        for _ in range(N):
            fn_baseline()
        torch.cuda.synchronize()
        ms_baseline = (time.perf_counter() - start) / N * 1000

        start = time.perf_counter()
        for _ in range(N):
            fn_packed()
        torch.cuda.synchronize()
        ms_packed = (time.perf_counter() - start) / N * 1000

    speedup = ms_baseline / ms_packed if ms_packed > 0 else float("inf")
    saved_us = (ms_baseline - ms_packed) * 1000

    print(
        f"  {B:>5}  {H:>3}  {HV:>3}  {K:>3}  {V:>3} | "
        f"{ms_baseline * 1000:>18.1f} | "
        f"{ms_packed * 1000:>10.1f} | "
        f"{speedup:>7.2f}x | "
        f"{saved_us:>+9.1f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_correctness(device, dtype):
    print("=" * 70)
    print("Correctness: Baseline GDN Decode vs Packed GDN Decode")
    print("=" * 70)

    shapes = [
        # (B,   H,  HV,  K,   V,   pool_size)
        # --- Qwen3.5-35B-A3B style (TP=2: H=8, HV=16) ---
        (1, 8, 16, 128, 128, 32),
        (4, 8, 16, 128, 128, 32),
        (16, 8, 16, 128, 128, 64),
        (32, 8, 16, 128, 128, 128),
        (64, 8, 16, 128, 128, 128),
        (128, 8, 16, 128, 128, 256),
        (256, 8, 16, 128, 128, 512),
        # --- Qwen3.5-35B-A3B style (TP=1: H=16, HV=32) ---
        (1, 16, 32, 128, 128, 32),
        (32, 16, 32, 128, 128, 128),
        (64, 16, 32, 128, 128, 128),
        # --- Qwen3-Next-80B-A3B style ---
        (32, 16, 16, 128, 128, 128),
        (64, 16, 16, 128, 128, 128),
        # --- With PAD_SLOT_ID ---
        (32, 8, 16, 128, 128, 128),
        # --- Edge cases ---
        (1, 8, 16, 128, 128, 32),
        (2, 8, 16, 128, 128, 32),
    ]

    all_pass = True
    for B, H, HV, K, V, pool_size in shapes:
        if not check_correctness(B, H, HV, K, V, pool_size, device, dtype):
            all_pass = False

    # PAD_SLOT_ID test: simulate CUDA Graph batch padding with -1 indices
    print("\n  PAD_SLOT_ID test (indices with -1):")
    pad_configs = [
        # (B, H, HV, K, V, pool_size, num_padded)
        (32, 8, 16, 128, 128, 128, 8),   # 8 out of 32 are padded
        (16, 16, 32, 128, 128, 64, 4),   # 4 out of 16 are padded
        (4, 8, 16, 128, 128, 32, 2),     # 2 out of 4 are padded (50% padding)
        (1, 8, 16, 128, 128, 32, 1),     # entire batch is padded
    ]
    for B, H_pad, HV_pad, K_pad, V_pad, pool_pad, num_padded in pad_configs:
        inp = make_inputs(B, H_pad, HV_pad, K_pad, V_pad, pool_pad, device, dtype)
        # Inject PAD_SLOT_ID=-1 for the last num_padded slots
        inp["cache_indices"][B - num_padded:] = -1
        o_baseline, st_baseline = run_baseline(inp)
        o_packed, st_packed = run_packed(inp)

        tag = f"B={B} pad={num_padded} H={H_pad} HV={HV_pad}"
        try:
            torch.testing.assert_close(o_packed, o_baseline, atol=2e-2, rtol=1e-2)
            print(f"  [PASS] PAD_SLOT_ID: {tag}")
        except AssertionError as e:
            print(f"  [FAIL] PAD_SLOT_ID: {tag}")
            diff = (o_packed - o_baseline).abs().max().item()
            print(f"         max diff = {diff:.6f}")
            all_pass = False

    # --- KDA (Key-Dim Attention) tests ---
    print()
    print("=" * 70)
    print("Correctness: Baseline KDA Decode vs Packed KDA Decode")
    print("=" * 70)

    kda_shapes = [
        # (B,   H,  HV,  K,   V,   pool_size)
        # --- Kimi-Linear-48B-A3B style (TP=2: H=16, HV=16) ---
        (1, 16, 16, 128, 128, 32),
        (4, 16, 16, 128, 128, 32),
        (32, 16, 16, 128, 128, 128),
        (64, 16, 16, 128, 128, 128),
        # --- Kimi-Linear-48B-A3B style (TP=4: H=8, HV=8) ---
        (1, 8, 8, 128, 128, 32),
        (32, 8, 8, 128, 128, 128),
        (64, 8, 8, 128, 128, 128),
        # --- GVA style (HV > H) ---
        (4, 8, 32, 128, 128, 32),
        (32, 8, 32, 128, 128, 128),
        # --- Edge cases ---
        (1, 8, 16, 128, 128, 32),
        (2, 8, 16, 128, 128, 32),
    ]

    for B, H, HV, K, V, pool_size in kda_shapes:
        if not check_correctness(B, H, HV, K, V, pool_size, device, dtype, is_kda=True):
            all_pass = False

    # KDA PAD_SLOT_ID test: same CUDA Graph padding mechanism as GDN
    print("\n  KDA PAD_SLOT_ID test (indices with -1):")
    kda_pad_configs = [
        # (B, H, HV, K, V, pool_size, num_padded)
        (32, 16, 16, 128, 128, 128, 8),   # Kimi-Linear TP=2 style, 25% padding
        (16, 8, 8, 128, 128, 64, 4),      # Kimi-Linear TP=4 style, 25% padding
        (4, 8, 32, 128, 128, 32, 2),      # GVA style, 50% padding
        (1, 16, 16, 128, 128, 32, 1),     # entire batch is padded
    ]
    for B, H_pad, HV_pad, K_pad, V_pad, pool_pad, num_padded in kda_pad_configs:
        inp = make_inputs(B, H_pad, HV_pad, K_pad, V_pad, pool_pad, device, dtype, is_kda=True)
        # Inject PAD_SLOT_ID=-1 for the last num_padded slots
        inp["cache_indices"][B - num_padded:] = -1
        o_baseline, st_baseline = run_baseline(inp)
        o_packed, st_packed = run_packed(inp)

        tag = f"B={B} pad={num_padded} H={H_pad} HV={HV_pad}"
        try:
            torch.testing.assert_close(o_packed, o_baseline, atol=2e-2, rtol=1e-2)
            print(f"  [PASS] KDA PAD_SLOT_ID: {tag}")
        except AssertionError as e:
            print(f"  [FAIL] KDA PAD_SLOT_ID: {tag}")
            diff = (o_packed - o_baseline).abs().max().item()
            print(f"         max diff = {diff:.6f}")
            all_pass = False

    # --- BF16 State tests ---
    # Verify native bf16 state I/O produces equivalent results to fp32 state
    print()
    print("=" * 70)
    print("Correctness: BF16 State vs FP32 State (packed kernel)")
    print("=" * 70)

    bf16_state_shapes = [
        # (B,   H,  HV,  K,   V,   pool_size, is_kda)
        (4, 8, 16, 128, 128, 32, False),
        (32, 8, 16, 128, 128, 128, False),
        (64, 16, 32, 128, 128, 128, False),
        (128, 8, 16, 128, 128, 256, False),
        # KDA variants
        (4, 16, 16, 128, 128, 32, True),
        (32, 16, 16, 128, 128, 128, True),
        (64, 8, 8, 128, 128, 128, True),
        (32, 8, 32, 128, 128, 128, True),
    ]

    for B, H_s, HV_s, K_s, V_s, pool_s, is_kda_s in bf16_state_shapes:
        kda_tag = " KDA" if is_kda_s else ""
        tag = f"B={B:>4} H={H_s:>2} HV={HV_s:>2} state=bf16{kda_tag}"

        # Run packed with fp32 state (reference)
        inp_fp32 = make_inputs(B, H_s, HV_s, K_s, V_s, pool_s, device, dtype,
                               seed=42, is_kda=is_kda_s, state_dtype=torch.float32)
        o_fp32, st_fp32 = run_packed(inp_fp32)

        # Run packed with bf16 state (test)
        inp_bf16 = make_inputs(B, H_s, HV_s, K_s, V_s, pool_s, device, dtype,
                               seed=42, is_kda=is_kda_s, state_dtype=torch.bfloat16)
        o_bf16, st_bf16 = run_packed(inp_bf16)

        # Compare output: bf16 state introduces quantization noise; use relaxed tolerance
        try:
            torch.testing.assert_close(o_bf16, o_fp32, atol=5e-2, rtol=2e-2)
            output_ok = True
        except AssertionError:
            output_ok = False
            out_diff = (o_bf16 - o_fp32).abs().max().item()

        # Compare state: bf16 result vs fp32 result truncated to bf16
        indices = inp_bf16["cache_indices"]
        st_fp32_bf16 = st_fp32[indices].to(torch.bfloat16)
        try:
            torch.testing.assert_close(st_bf16[indices], st_fp32_bf16, atol=5e-2, rtol=2e-2)
            state_ok = True
        except AssertionError:
            state_ok = False
            st_diff = (st_bf16[indices].float() - st_fp32_bf16.float()).abs().max().item()

        passed_bf16 = output_ok and state_ok
        if passed_bf16:
            print(f"  [PASS] {tag}")
        else:
            details = []
            if not output_ok:
                details.append(f"output max_diff={out_diff:.6f}")
            if not state_ok:
                details.append(f"state max_diff={st_diff:.6f}")
            print(f"  [FAIL] {tag}  ({', '.join(details)})")
            all_pass = False

    # BF16 state with CUDA Graph padding. Start the FP32 comparison from the
    # exact BF16 values so this isolates kernel I/O from input quantization.
    print("\n  BF16 state PAD_SLOT_ID tests:")
    bf16_pad_configs = [
        # (B, H, HV, K, V, pool_size, num_padded, is_kda)
        (8, 8, 16, 128, 128, 32, 3, False),
        (8, 8, 16, 128, 128, 32, 3, True),
    ]
    for B, H_s, HV_s, K_s, V_s, pool_s, num_padded, is_kda_s in bf16_pad_configs:
        inp_bf16 = make_inputs(
            B, H_s, HV_s, K_s, V_s, pool_s, device, dtype,
            seed=42, is_kda=is_kda_s, state_dtype=torch.bfloat16,
        )
        inp_bf16["cache_indices"][-num_padded:] = -1
        inp_fp32 = dict(inp_bf16)
        inp_fp32["ssm_states"] = inp_bf16["ssm_states"].float()

        o_fp32, st_fp32 = run_packed(inp_fp32)
        o_bf16, st_bf16 = run_packed(inp_bf16)
        valid = inp_bf16["cache_indices"] >= 0
        valid_indices = inp_bf16["cache_indices"][valid].long()
        padded_slots = torch.arange(B - num_padded, B, device=device)
        tag = f"B={B} pad={num_padded} {'KDA' if is_kda_s else 'GDN'}"
        try:
            torch.testing.assert_close(
                o_bf16[:, valid], o_fp32[:, valid], atol=5e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                st_bf16[valid_indices],
                st_fp32[valid_indices].to(torch.bfloat16),
                atol=5e-2,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                st_bf16[padded_slots], inp_bf16["ssm_states"][padded_slots]
            )
            print(f"  [PASS] {tag}")
        except AssertionError as exc:
            print(f"  [FAIL] {tag}: {exc}")
            all_pass = False

    print()
    if all_pass:
        print("ALL PASSED.")
    else:
        print("SOME FAILED.")
    return all_pass


def run_benchmark(device, dtype, H, HV, K, V, batch_size, pool_size=512):
    print()
    print("=" * 85)
    print("Benchmark: GDN Triton Packed Decode vs CUDA Packed GDN Decode")
    print("=" * 85)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    bench_configs = [(B, H, HV) for B in batch_sizes if B <= batch_size * 2]

    print(f"  Config: K={K}, V={V}, pool_size={pool_size}, dtype={dtype}")
    print(
        f"  {'B':>5}  {'H':>3}  {'HV':>3}  {'K':>3}  {'V':>3} | "
        f"{'triton_packed (μs)':>15} | "
        f"{'packed (μs)':>10} | "
        f"{'speedup':>8} | "
        f"{'saved (μs)':>10}"
    )
    print("  " + "-" * 75)

    for B, h, hv in bench_configs:
        actual_pool = max(pool_size, B + 16)
        bench_shape(B, h, hv, K, V, actual_pool, device, dtype)


def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
    }
    key = dtype_str.lower()
    if key not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {sorted(dtype_map.keys())}")
    return dtype_map[key]


def main():
    parser = argparse.ArgumentParser(description="k_last_packed kernel correctness & benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmark mode")
    parser.add_argument("--batch", type=int, default=129, help="Batch size (default: 129)")
    parser.add_argument("--tp", type=int, default=4, help="TP size (default: 4)")
    parser.add_argument("--qwen35", action="store_true", help="Test Qwen35 model (else Qwen3-Next)")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--alogtype", type=str, default="fp32", help="bf16/fp16/fp32")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)

    if args.qwen35:
        H, HV, K, V = 16, 64, 128, 128
    else:
        H, HV, K, V = 16, 32, 128, 128

    H = H // args.tp
    HV = HV // args.tp

    device = "cuda"

    cap = torch.cuda.get_device_capability()
    dev_name = torch.cuda.get_device_name()
    print(f"Device: {dev_name}  (SM {cap[0]}{cap[1]})")
    print(f"Config: batch={args.batch}, tp={args.tp}, H={H}, HV={HV}, K={K}, V={V}, dtype={dtype}")

    all_pass = run_correctness(device, dtype)

    if args.benchmark:
        if not all_pass:
            print("\nSkipping benchmark due to correctness failures.")
            return 1
        run_benchmark(device, dtype, H, HV, K, V, args.batch)

    return 0


if __name__ == "__main__":
    sys.exit(main())
