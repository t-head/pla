# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
"""Vendored sglang Triton fold reference for on-PPU A/B vs the CUDA C++ fold.

Copied verbatim (structure) from
  sglang/kernels/ops/attention/fla/kda_replayssm_spec_decode.py
  :: kda_replayssm_exact_fold_kernel + commit_kda_replayssm_spec_all_layers
so the CUDA C++ fold (kda_fold_ppu.cu) can be timed against the exact upstream
Triton implementation on the same inputs. Track / onorm paths are omitted
(HAS_TRACK=False) — the K3 commit hot path does not use them for the ssm fold.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def kda_replayssm_exact_fold_kernel(
    h0, rawv_cache, rawk_cache, gk_cache, beta_cache,
    ssm_state_indices, accept_lens, mamba_track_indices, mamba_steps_to_track,
    stride_state_slot: tl.constexpr, stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr, stride_gk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr, stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr, stride_rawk_layer: tl.constexpr,
    stride_gk_layer: tl.constexpr, stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr, stride_accept: tl.constexpr,
    stride_track: tl.constexpr, stride_steps: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr, NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hvl = tl.program_id(2)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_h0 = (h0 + state_idx * stride_state_slot + i_hv * V * K
            + o_v[None, :] * K + o_k[:, None])
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, n_commit):
        phys = t.to(tl.int64)
        b_k = tl.load(rawk_cache + state_idx * stride_rawk_slot
                      + (i_h * MAX_CACHE_LEN + phys) * K + o_k,
                      mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(rawv_cache + state_idx * stride_rawv_slot
                      + (i_hv * MAX_CACHE_LEN + phys) * V + o_v,
                      mask=mask_v, other=0.0).to(tl.float32)
        b_gk = tl.load(gk_cache + state_idx * stride_gk_slot
                       + (i_hv * MAX_CACHE_LEN + phys) * K + o_k,
                       mask=mask_k, other=0.0).to(tl.float32)
        b_beta = tl.load(beta_cache + state_idx * stride_beta_slot
                         + i_hv * MAX_CACHE_LEN + phys).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_gk[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def triton_fold_all_layers(temporal, rawv, rawk, gk, beta, state_indices,
                           accept_lens, max_cache_len, num_k_heads,
                           null_block_id=-1, use_qk_l2norm=True):
    num_layers, num_slots, HV, V, K = temporal.shape
    B = state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), B, HV * num_layers)
    kda_replayssm_exact_fold_kernel[grid](
        temporal, rawv, rawk, gk, beta,
        state_indices, accept_lens, state_indices, accept_lens,
        temporal.stride(1), rawv.stride(1), rawk.stride(1), gk.stride(1),
        beta.stride(1), temporal.stride(0), rawv.stride(0), rawk.stride(0),
        gk.stride(0), beta.stride(0), state_indices.stride(0),
        accept_lens.stride(0), 0, 0,
        H=num_k_heads, HV=HV, K=K, V=V, BK=BK, BV=BV,
        MAX_CACHE_LEN=max_cache_len, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        NULL_BLOCK_ID=null_block_id, HAS_TRACK=False,
        num_warps=1, num_stages=3,
    )
