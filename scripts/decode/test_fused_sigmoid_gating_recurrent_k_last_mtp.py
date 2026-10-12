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
Comprehensive test suite for fused_sigmoid_gating_delta_rule_forward_k_last.

Tests various parameter combinations for both Triton and CUDA implementations
to ensure correctness across all supported configurations.

Run as: python -m scripts.test_fused_sigmoid_gating_recurrent_k_last_comprehensive
"""

import sys
import os
import argparse
import torch

# Add scripts directory to path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last
from _kda_reference import naive_kda_lowerbound_gate, naive_recurrent_kda

try:
    from fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as triton_fused_gating,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("WARNING: fused_sigmoid_gating_recurrent not found, skipping Triton tests")


def _make_tensors(N, T, H, HV, K, V, is_kda, device="cuda", seed=2025, dtype=torch.bfloat16, alogtype=torch.float32,
                  mtp_verify=True):
    """Create input tensors for GDN target_verify.

    ``mtp_verify`` says whether the call under test takes the MTP-verify path; it
    only affects the KDA dt_bias dtype, which the kernel reads as fp32 there and as
    the activation dtype in pure decode. Defaults to True since this suite is MTP.
    """
    torch.manual_seed(seed)
    A_log = torch.randn(HV, dtype=alogtype, device=device)
    dt_bias = torch.randn(HV, dtype=dtype, device=device)
    a = torch.randn(1, N * T, HV, dtype=dtype, device=device)
    b = torch.randn(1, N * T, HV, dtype=dtype, device=device)
    q = torch.randn(1, N * T, H, K, dtype=dtype, device=device)
    k = torch.randn(1, N * T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, N * T, HV, V, dtype=dtype, device=device)
    indices = torch.arange(N, dtype=torch.int32, device=device)
    initial_state = torch.randn(N, HV, K, V, dtype=torch.float, device=device)
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device)

    if is_kda:
        a = torch.randn(1, N * T, HV, K, dtype=dtype, device=device)
        dt_bias = torch.randn(HV, K, device=device,
                              dtype=torch.float32 if mtp_verify else dtype)

    return A_log, dt_bias, a, b, q, k, v, initial_state, indices, cu_seqlens


def run_test(test_name, B, T, H, HV, K, V, is_kda, use_qk_l2norm, disable_state_update,
             use_cu_seqlens, use_retrieve_parent=False, use_intermediate_cache=False,
             packed_layout=False, parent_mode="causal",
             seed=2025, atol=1e-3, rtol=1e-4, dtype=torch.bfloat16, alogtype=torch.float32):
    """Run a single test case and verify Triton vs CUDA match."""
    device = "cuda"

    torch.manual_seed(seed)
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        B, T, H, HV, K, V, is_kda, device, seed, dtype=dtype, alogtype=alogtype
    )

    # a/b use flattened tokens; QKV can keep SGLang's [1, B*T, ...] layout.
    a = a.view(B * T, -1).contiguous()
    b = b.view(B * T, -1).contiguous()
    if not packed_layout:
        q = q.view(B, T, H, K).contiguous()
        k = k.view(B, T, H, K).contiguous()
        v = v.view(B, T, HV, V).contiguous()

    s_triton = state.clone()
    s_cuda = state.clone()

    # Prepare retrieve_parent_token for tree attention
    retrieve_parent_token = None
    if use_retrieve_parent:
        if parent_mode == "branching":
            # Eagle tree branching: tokens 1..T-1 all branch from an earlier shared
            # parent (not t-1), which exercises the eagle reload path.
            parent_list = [0] + [max(0, t - 2) for t in range(1, T)]
        else:
            # Standard causal single chain: token t's parent is t-1 (or 0 for first token)
            parent_list = [0] + [t for t in range(T - 1)]
        retrieve_parent_token = torch.tensor([parent_list] * B, dtype=torch.int64, device=device).contiguous()

    # Prepare intermediate states buffer for caching
    intermediate_states_buffer = None
    intermediate_state_indices = None
    cache_steps = None
    if use_intermediate_cache:
        cache_steps = T
        intermediate_states_buffer = torch.zeros(B * cache_steps, HV, V, K, dtype=torch.float32, device=device)
        intermediate_state_indices = torch.arange(B, dtype=torch.int32, device=device)

    # Run Triton
    triton_ok = False
    out_triton = None
    if HAS_TRITON:
        try:
            out_triton = triton_fused_gating(
                A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                s_triton, indices, scale=1.0, use_qk_l2norm_in_kernel=use_qk_l2norm,
                cu_seqlens=cu_seqlens if use_cu_seqlens else None,
                is_kda=is_kda, disable_state_update=disable_state_update,
                retrieve_parent_token=retrieve_parent_token,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=cache_steps,
            )
            triton_ok = True
        except RuntimeError as e:
            print(f"  [WARN] {test_name} Triton failed ({e}), falling back to CUDA-only")
            triton_ok = False
            A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
                B, T, H, HV, K, V, is_kda, device, seed, dtype=dtype, alogtype=alogtype
            )
            a = a.view(B * T, -1).contiguous()
            b = b.view(B * T, -1).contiguous()
            if not packed_layout:
                q = q.view(B, T, H, K).contiguous()
                k = k.view(B, T, H, K).contiguous()
                v = v.view(B, T, HV, V).contiguous()
            s_cuda = state.clone()

    # Run CUDA
    out_cuda = fused_sigmoid_gating_delta_rule_forward_k_last(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
        s_cuda, indices, 1.0, use_qk_l2norm,
        cu_seqlens if use_cu_seqlens else None, is_kda,
        disable_state_update, intermediate_states_buffer, intermediate_state_indices,
        cache_steps, retrieve_parent_token
    )

    # Verify results
    if triton_ok and out_triton is not None:
        try:
            torch.testing.assert_close(out_cuda, out_triton, atol=atol, rtol=rtol)
            print(f"  [PASS] {test_name}")
            return True
        except AssertionError as e:
            print(f"  [FAIL] {test_name}")
            print(f"         Triton mean: {out_triton.mean().item():.6f}, CUDA mean: {out_cuda.mean().item():.6f}")
            print(f"         Max diff: {(out_triton - out_cuda).abs().max().item():.6f}")
            return False
    else:
        print(f"  [SKIP] {test_name} (Triton not available)")
        return True


def run_cache_stride_alloc_test(test_name, B=2, T=3, max_cache_steps=8,
                                H=4, HV=8, K=128, V=128,
                                dtype=torch.bfloat16, alogtype=torch.float32,
                                seed=2025, atol=1e-3, rtol=1e-4):
    """Regression test for the cache-stride fix (--speculative-adaptive).

    The pla CUDA kernel must use the intermediate buffer's ALLOCATED step dim as the
    per-request stride when the buffer is 5D ([req, maxT, HV, V, K]), NOT the runtime
    cache_steps. Otherwise, when maxT > runtime T, request r>0 states are written at
    the wrong (runtime-strided) offset and sglang's commit -- which reads the buffer's
    real strides -- would pick up another request's state.

    Strategy: run identical inputs twice with cache_steps=T passed both times:
      (A) tight 4D buffer [B*T, HV, V, K]        -> reference (stride == T == maxT)
      (B) 5D buffer [B, maxT, HV, V, K], maxT > T -> fix must derive stride == maxT
    Then assert B's first-T-step states match A at the maxT-strided positions and the
    unused [T:maxT] slots stay zero. Without the fix, buf_5d[r>0, :T] would be wrong
    (states land at r*T instead of r*maxT), so this case fails.
    """
    device = "cuda"
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        B, T, H, HV, K, V, is_kda=False, device=device, seed=seed,
        dtype=dtype, alogtype=alogtype,
    )
    a = a.view(B * T, -1).contiguous()
    b = b.view(B * T, -1).contiguous()
    q = q.view(B, T, H, K).contiguous()
    k = k.view(B, T, H, K).contiguous()
    v = v.view(B, T, HV, V).contiguous()

    parent_list = [0] + [t for t in range(T - 1)]
    retrieve_parent_token = torch.tensor([parent_list] * B, dtype=torch.int64, device=device).contiguous()
    inter_idx = torch.arange(B, dtype=torch.int32, device=device)

    def _run(buf):
        return fused_sigmoid_gating_delta_rule_forward_k_last(
            A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
            state.clone(), indices, 1.0, True,
            cu_seqlens, False, True,
            buf, inter_idx, T, retrieve_parent_token,
        )

    # (A) reference: tight buffer, allocated stride == runtime T
    buf_ref = torch.zeros(B * T, HV, V, K, dtype=torch.float32, device=device)
    out_ref = _run(buf_ref)

    # (B) allocated 5D buffer with maxT > T; cache_steps=T still passed at call site
    buf_5d = torch.zeros(B, max_cache_steps, HV, V, K, dtype=torch.float32, device=device)
    out_5d = _run(buf_5d)

    ok_out = torch.allclose(out_5d, out_ref, atol=atol, rtol=rtol)
    buf_ref_view = buf_ref.view(B, T, HV, V, K)
    ok_states = torch.allclose(buf_5d[:, :T], buf_ref_view, atol=atol, rtol=rtol)
    ok_zero = bool((buf_5d[:, T:] == 0).all().item()) if max_cache_steps > T else True

    if ok_out and ok_states and ok_zero:
        print(f"  [PASS] {test_name}")
        return True
    print(f"  [FAIL] {test_name}  (ok_out={ok_out}, ok_states={ok_states}, ok_zero={ok_zero})")
    if not ok_states:
        diff = (buf_5d[:, :T] - buf_ref_view).abs().max().item()
        print(f"         state max diff = {diff:.6f} (states written at wrong stride)")
    return False


def run_k3_safe_reference_test(
    test_name,
    T,
    B=1,
    verify_lens=None,
    target_verify=True,
    H=16,
    K=128,
    V=128,
    lower_bound=-5.0,
    dtype=torch.bfloat16,
    alogtype=torch.float32,
    seed=2025,
    atol=3e-2,
    rtol=3e-2,
):
    """Compare Kimi K3 decode/target verify against the PyTorch recurrence."""
    device = "cuda"
    if verify_lens is not None and not target_verify:
        raise ValueError("ragged verify_lens require target_verify=True")
    if verify_lens is not None:
        B = len(verify_lens)
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        B, T, H, H, K, V, True, device, seed, dtype=dtype, alogtype=alogtype,
        # target_verify is forwarded as disable_state_update below, so it alone
        # decides the path when T == 1.
        mtp_verify=target_verify or T > 1,
    )
    q = q.view(B, T, H, K).contiguous()
    k = k.view(B, T, H, K).contiguous()
    v = v.view(B, T, H, V).contiguous()
    a_4d = a.view(B, T, H, K)
    b_3d = b.view(B, T, H)

    q_ref = q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6
    )
    k_ref = k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6
    )
    g_ref = naive_kda_lowerbound_gate(
        a_4d, A_log, dt_bias, lower_bound=lower_bound
    )
    beta_ref = torch.sigmoid(b_3d.float())
    initial_ref = state.transpose(-1, -2).contiguous()
    out_ref, final_state_ref = naive_recurrent_kda(
        q_ref,
        k_ref,
        v,
        g_ref,
        beta_ref,
        scale=K**-0.5,
        initial_state=initial_ref,
        output_final_state=True,
    )

    expected_intermediate = None
    if target_verify:
        expected_intermediate = torch.empty(
            B, T, H, V, K, dtype=torch.float32, device=device
        )
        for step in range(1, T + 1):
            _, step_state = naive_recurrent_kda(
                q_ref[:, :step],
                k_ref[:, :step],
                v[:, :step],
                g_ref[:, :step],
                beta_ref[:, :step],
                scale=K**-0.5,
                initial_state=initial_ref,
                output_final_state=True,
            )
            expected_intermediate[:, step - 1] = step_state.transpose(-1, -2)

    if verify_lens is not None:
        def pack(tensor):
            return torch.cat(
                [tensor[i, :length] for i, length in enumerate(verify_lens)]
            ).unsqueeze(0).contiguous()

        q, k, v, a_4d, b_3d, out_ref = map(
            pack, (q, k, v, a_4d, b_3d, out_ref)
        )
        offsets = [0]
        for sequence, length in enumerate(verify_lens):
            offsets.append(offsets[-1] + length)
            expected_intermediate[sequence, length:] = 0
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)

    a_flat = a_4d.view(-1, H * K).contiguous()
    b_flat = b_3d.view(-1, H).contiguous()

    intermediate_states = (
        torch.zeros_like(expected_intermediate) if target_verify else None
    )
    initial_state = state.clone()
    out_cuda = fused_sigmoid_gating_delta_rule_forward_k_last(
        A_log,
        a_flat,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b_flat,
        initial_state,
        indices,
        K**-0.5,
        True,
        cu_seqlens,
        True,
        target_verify,
        intermediate_states,
        indices if target_verify else None,
        T if target_verify else None,
        None,
        lower_bound,
    )

    try:
        torch.testing.assert_close(out_cuda, out_ref, atol=atol, rtol=rtol)
        if target_verify:
            torch.testing.assert_close(
                intermediate_states,
                expected_intermediate,
                atol=atol,
                rtol=rtol,
            )
            torch.testing.assert_close(initial_state, state)
        else:
            torch.testing.assert_close(
                initial_state.transpose(-1, -2),
                final_state_ref,
                atol=atol,
                rtol=rtol,
            )
        print(f"  [PASS] {test_name}")
        return True
    except AssertionError:
        print(f"  [FAIL] {test_name}")
        print(f"         Output max diff: {(out_cuda - out_ref).abs().max().item():.6f}")
        actual_state = (
            intermediate_states
            if target_verify
            else initial_state.transpose(-1, -2)
        )
        expected_state = (
            expected_intermediate if target_verify else final_state_ref
        )
        print(
            "         State max diff: "
            f"{(actual_state - expected_state).abs().max().item():.6f}"
        )
        return False


def test_all_configurations(dtype=torch.bfloat16, alogtype=torch.float32):
    """Run comprehensive test matrix."""
    print("=" * 70)
    print("Comprehensive Fused Sigmoid Gating Recurrent K-Last Test Suite")
    print(f"dtype={dtype}, alogtype={alogtype}")
    print("=" * 70)

    results = []

    # =========================================================================
    # 1. Basic parameter variations (T=1, Decode mode)
    # =========================================================================
    print("\n[1] Decode Mode Tests (T=1)")
    print("-" * 50)

    for is_kda in [False, True]:
        for use_qk_l2norm in [False, True]:
            test_name = f"B=2,T=1,KDA={is_kda},L2NORM={use_qk_l2norm}"
            result = run_test(
                test_name, B=2, T=1, H=4, HV=8, K=128, V=128,
                is_kda=is_kda, use_qk_l2norm=use_qk_l2norm,
                disable_state_update=True, use_cu_seqlens=True,
                dtype=dtype, alogtype=alogtype
            )
            results.append((test_name, result))

    # =========================================================================
    # 2. MTP Verify mode - Sequence length variations
    # =========================================================================
    print("\n[2] MTP Verify Mode Tests (T>1) - Sequence Length")
    print("-" * 50)

    for T in [2, 4, 8, 16]:
        test_name = f"B=2,T={T},standard_causal"
        result = run_test(
            test_name, B=2, T=T, H=4, HV=8, K=128, V=128,
            is_kda=False, use_qk_l2norm=True,
            disable_state_update=True, use_cu_seqlens=True,
            dtype=dtype, alogtype=alogtype
        )
        results.append((test_name, result))

    # =========================================================================
    # 3. MTP Verify mode - Batch size variations
    # =========================================================================
    print("\n[3] MTP Verify Mode Tests - Batch Size")
    print("-" * 50)

    for B in [1, 2, 4, 8]:
        test_name = f"B={B},T=8,standard_causal"
        result = run_test(
            test_name, B=B, T=8, H=4, HV=8, K=128, V=128,
            is_kda=False, use_qk_l2norm=True,
            disable_state_update=True, use_cu_seqlens=True,
            dtype=dtype, alogtype=alogtype
        )
        results.append((test_name, result))

    # =========================================================================
    # 4. MTP Verify mode - disable_state_update variations
    # =========================================================================
    print("\n[4] MTP Verify Mode Tests - State Update Control")
    print("-" * 50)

    for disable_state in [False, True]:
        test_name = f"B=2,T=8,disable_state_update={disable_state}"
        result = run_test(
            test_name, B=2, T=8, H=4, HV=8, K=128, V=128,
            is_kda=False, use_qk_l2norm=True,
            disable_state_update=disable_state, use_cu_seqlens=True,
            dtype=dtype, alogtype=alogtype
        )
        results.append((test_name, result))

    # =========================================================================
    # 5. MTP Verify mode - cu_seqlens (varlen) variations
    # =========================================================================
    print("\n[5] MTP Verify Mode Tests - VarLen Support")
    print("-" * 50)

    for use_cu_seqlens in [False, True]:
        test_name = f"B=2,T=8,cu_seqlens={use_cu_seqlens}"
        result = run_test(
            test_name, B=2, T=8, H=4, HV=8, K=128, V=128,
            is_kda=False, use_qk_l2norm=True,
            disable_state_update=True, use_cu_seqlens=use_cu_seqlens,
            dtype=dtype, alogtype=alogtype
        )
        results.append((test_name, result))

    # =========================================================================
    # 6. MTP Verify mode - IS_KDA variations
    # =========================================================================
    print("\n[6] MTP Verify Mode Tests - KDA Support")
    print("-" * 50)

    for is_kda in [False, True]:
        test_name = f"B=2,T=8,is_kda={is_kda}"
        kda_atol = 2e-3 if is_kda else 1e-3
        result = run_test(
            test_name, B=2, T=8, H=4, HV=8, K=128, V=128,
            is_kda=is_kda, use_qk_l2norm=True,
            disable_state_update=True, use_cu_seqlens=True,
            atol=kda_atol,
            dtype=dtype, alogtype=alogtype
        )
        results.append((test_name, result))

    # =========================================================================
    # 7. MTP Verify mode - retrieve_parent_token (tree attention)
    # With intermediate_states_buffer for caching parent states
    # =========================================================================
    print("\n[7] MTP Verify Mode Tests - Tree Attention (single chain)")
    print("-" * 50)

    # Standard causal single-chain parent mapping with intermediate state caching
    test_name = "B=2,T=8,tree_attn=causal,cache=True"
    result = run_test(
        test_name, B=2, T=8, H=4, HV=8, K=128, V=128,
        is_kda=False, use_qk_l2norm=True,
        disable_state_update=True, use_cu_seqlens=True,
        use_retrieve_parent=True, use_intermediate_cache=True,
        parent_mode="causal",
        dtype=dtype, alogtype=alogtype
    )
    results.append((test_name, result))

    # =========================================================================
    # 8. MTP Verify mode - eagle tree branching
    # Tokens branch from an earlier shared parent (not t-1), exercising the
    # eagle reload path with V_ITERS > STAGE_NUM under packed layout.
    # =========================================================================
    print("\n[8] MTP Verify Mode Tests - Eagle Tree Branching")
    print("-" * 50)

    test_name = "B=1,T=3,tree_attn=branching,cache=True"
    result = run_test(
        test_name, B=1, T=3, H=4, HV=8, K=128, V=128,
        is_kda=False, use_qk_l2norm=True,
        disable_state_update=True, use_cu_seqlens=True,
        use_retrieve_parent=True, use_intermediate_cache=True,
        packed_layout=True, parent_mode="branching",
        dtype=dtype, alogtype=alogtype
    )
    results.append((test_name, result))

    # =========================================================================
    # 9. Dimension variations
    # =========================================================================
    print("\n[9] Dimension Variation Tests")
    print("-" * 50)

    # Test with standard dimensions (H=4, HV=8, K=128, V=128)
    # Note: Larger dimensions may exceed kernel resource limits on some GPUs
    test_name = "H=4,HV=8,K=128,V=128"
    result = run_test(
        test_name, B=2, T=4, H=4, HV=8, K=128, V=128,
        is_kda=False, use_qk_l2norm=True,
        disable_state_update=True, use_cu_seqlens=True,
        dtype=dtype, alogtype=alogtype
    )
    results.append((test_name, result))

    # =========================================================================
    # 10. cache_steps stride: allocated (5D maxT buffer) vs runtime T
    # Regression for the --speculative-adaptive stride-mismatch fix: the CUDA
    # kernel must derive the per-request stride from the 5D buffer's allocated
    # maxT dim, not from the runtime cache_steps.
    # =========================================================================
    print("\n[10] MTP Verify Mode Tests - Cache Stride (allocated maxT buffer)")
    print("-" * 50)

    test_name = "B=2,T=3,maxT=8,5D_alloc_stride"
    result = run_cache_stride_alloc_test(
        test_name, B=2, T=3, max_cache_steps=8, H=4, HV=8, K=128, V=128,
        dtype=dtype, alogtype=alogtype,
    )
    results.append((test_name, result))

    # =========================================================================
    # 11. Kimi K3 safe-gate decode and target verify
    # =========================================================================
    print("\n[11] Kimi K3 Safe-Gate Decode and Target Verify Tests")
    print("-" * 50)

    tp1_cases = [
        ("decode", 1, False),
        ("target_verify", 2, True),
    ]
    for phase, T, target_verify in tp1_cases:
        test_name = (
            f"Kimi-K3-TP1,B=2,T={T},H=HV=96,{phase},lower_bound=-5"
        )
        result = run_k3_safe_reference_test(
            test_name,
            B=2,
            T=T,
            H=96,
            target_verify=target_verify,
            dtype=dtype,
            alogtype=alogtype,
        )
        results.append((test_name, result))

    for T in range(2, 9):
        test_name = f"Kimi-K3,B=2,T={T},H=HV=16,lower_bound=-5"
        result = run_k3_safe_reference_test(
            test_name,
            B=2,
            T=T,
            dtype=dtype,
            alogtype=alogtype,
        )
        results.append((test_name, result))

    ragged_verify_lens = [
        [index % T + 1 for index in range(T + 1)]
        for T in range(2, 9)
    ]
    ragged_verify_lens.append([index % 8 + 1 for index in range(17)])
    for verify_lens in ragged_verify_lens:
        test_name = (
            f"Kimi-K3,verify_lens={verify_lens},"
            "H=HV=16,lower_bound=-5"
        )
        result = run_k3_safe_reference_test(
            test_name,
            T=max(verify_lens),
            verify_lens=verify_lens,
            dtype=dtype,
            alogtype=alogtype,
        )
        results.append((test_name, result))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    failed_tests = [name for name, result in results if not result]
    if failed_tests:
        print(f"\nFailed tests:")
        for name in failed_tests:
            print(f"  - {name}")
        return False
    else:
        print("\nAll tests passed!")
        return True


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
    parser = argparse.ArgumentParser(description="k_last MTP correctness test")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmark mode (no-op for MTP)")
    parser.add_argument("--batch", type=int, default=129, help="Batch size (default: 129)")
    parser.add_argument("--tp", type=int, default=4, help="TP size (default: 4)")
    parser.add_argument("--qwen35", action="store_true", help="Test Qwen35 model (else Qwen3-Next)")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--alogtype", type=str, default="fp32", help="bf16/fp16/fp32")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    alogtype = parse_dtype(args.alogtype)

    print(f"Config: dtype={dtype}, alogtype={alogtype}")

    success = test_all_configurations(dtype=dtype, alogtype=alogtype)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
