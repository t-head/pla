# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
"""Inline copy of the vLLM triton reference kernels used to score the PPU KDA
decode mega kernel.

Two entry points are exported:

* ``causal_conv1d_update(x, conv_state, weight, ...)``
      short causal-conv1d update, adapted verbatim from
      ``vllm/model_executor/layers/mamba/ops/causal_conv1d.py``
      (update path only — the training-side ``causal_conv1d_fn`` is not
      needed for decode).

* ``fused_recurrent_kda_packed_decode(mixed_qkv, raw_g, raw_beta, ...)``
      one-token KDA decode from packed post-conv QKV, adapted from
      ``vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py``.

The two triton ``@triton.jit`` kernels are copied verbatim so any numerical
behavior matches upstream vLLM exactly. Only the vLLM-specific import graph
(``vllm.platforms``, ``vllm.triton_utils``, ``vllm.third_party.flash_linear_attention``
etc.) is stubbed out locally so the module has no vLLM runtime dependency.

Origins:
    ``causal_conv1d.py``       : Apache-2.0 (vLLM project) / BSD-3 (Tri Dao)
    ``fused_recurrent.py``     : Apache-2.0 (vLLM) / MIT (flash-linear-attention)
"""

# ruff: noqa: E501
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Minimal shims for the pieces vLLM injects at import time.
# ---------------------------------------------------------------------------
# vLLM tags "hopper+" hardware to enable programmatic dependent launch (PDL).
# On PPU / SM80-class NV parts we simply disable PDL — the kernels branch on
# this constexpr and skip the ``gdc_wait / gdc_launch_dependents`` intrinsics.
def _is_arch_support_pdl() -> bool:
    return False


# vLLM's block cache uses block id 0 as the "null / padding" placeholder.
NULL_BLOCK_ID = 0

# ``exp`` / ``log`` upstream can be swapped for the fast_expf/fast_logf
# intrinsics via env FLA_USE_FAST_OPS. Keep the default (tl.exp / tl.log)
# because we compare against a torch reference at bf16 atol=3e-2 — the tiny
# accuracy gap of fast_expf is irrelevant here and matters more for perf.
exp = tl.exp
log = tl.log


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# ===========================================================================
# 1) causal_conv1d_update  (vLLM/mamba ops)
#    Source: vllm/model_executor/layers/mamba/ops/causal_conv1d.py (lines 763-1279)
# ===========================================================================
@triton.jit(do_not_specialize_on_alignment=["num_cache_lines"])
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.int64,
    # others
    null_block_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        # Get the state from the initial_state_idx
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if HAS_NULL_BLOCK:  # noqa
        if conv_states_input_coord == null_block_id:
            # not processing as this is not the actual sequence
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        # revise state_len and seqlen
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    if IS_SPEC_DECODING:
        conv_state_token_offset = (
            tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
        )
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)

    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )
    mask = (
        (conv_states_input_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)

    x_ptrs = x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (
        idx_tokens * stride_conv_state_tok
    )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4: PRE-LOAD WEIGHTS
    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: "torch.Tensor | None" = None,
    activation=None,
    conv_state_indices: "torch.Tensor | None" = None,
    num_accepted_tokens: "torch.Tensor | None" = None,
    query_start_loc: "torch.Tensor | None" = None,
    max_query_len: int = -1,
    null_block_id: int = NULL_BLOCK_ID,
    block_idx_last_scheduled_token: "torch.Tensor | None" = None,
    initial_state_idx: "torch.Tensor | None" = None,
    validate_data: bool = False,
    out: "torch.Tensor | None" = None,
):
    """vLLM ``causal_conv1d_update`` port (see upstream for full docstring)."""
    if validate_data:
        assert null_block_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    if out is None:
        out = x
    else:
        if out.shape != x.shape:
            raise ValueError(
                f"`out` shape {tuple(out.shape)} must match `x` shape {tuple(x.shape)}."
            )
        if out.dtype != original_x_dtype or out.device != x.device:
            raise ValueError(
                "`out` must have the same dtype and device as the input `x`."
            )
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
        out = out.unsqueeze(-1)
    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert state_len >= width - 1
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert batch == conv_state_indices.shape[0]
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1

    stride_w_dim, stride_w_width = weight.stride()

    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        null_block_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        HAS_NULL_BLOCK=null_block_id is not None,
        BLOCK_N=256,
        launch_pdl=_is_arch_support_pdl(),
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)


# ===========================================================================
# 2) fused_recurrent_kda_packed_decode  (vLLM/kimi_k3 kda)
#    Source: vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py (lines 486-671)
# ===========================================================================
@triton.jit(do_not_specialize=["stride_beta_token", "stride_state_indices"])
def _fused_recurrent_kda_packed_decode_kernel(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token,
    stride_state_token: tl.constexpr,
    stride_state_indices,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    state_idx = tl.load(state_indices + i_n * stride_state_indices).to(tl.int64)
    p_out = out + (i_n * H + i_h) * V + o_v
    if state_idx <= 0:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=mask_v)
        return

    p_state = state + state_idx * stride_state_token
    p_state += i_h * V * K + o_v[:, None] * K + o_k[None, :]
    b_state = tl.load(p_state, mask=mask_state, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_token
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    b_v = tl.load(
        p_mixed + 2 * H * K + i_h * V + o_v,
        mask=mask_v,
        other=0,
    ).to(tl.float32)

    b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_g = raw_g + i_n * stride_g_token + i_h * K + o_k
    b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_g += b_bias
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_softplus = tl.where(
            b_g > SOFTPLUS_THRESHOLD,
            b_g,
            log(1.0 + tl.exp(b_g)),
        )
        b_gate = -b_a * b_softplus

    b_state *= exp(b_gate[None, :])
    b_v -= tl.sum(b_state * b_k[None, :], axis=1)
    b_beta = tl.sigmoid(
        tl.load(raw_beta + i_n * stride_beta_token + i_h).to(tl.float32)
    )
    b_v *= b_beta
    b_state += b_v[:, None] * b_k[None, :]
    b_out = tl.sum(b_state * b_q[None, :], axis=1)

    tl.store(p_out, b_out.to(p_out.dtype.element_ty), mask=mask_v)
    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=mask_state)


def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale=None,
):
    """One-token KDA decode from packed post-conv QKV (vLLM port)."""
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be 2D and contiguous in its last dim.")
    if raw_g.ndim != 4 or raw_g.shape[0] != 1:
        raise ValueError("`raw_g` must have shape [1, B, H, K].")
    if raw_beta.ndim != 3 or raw_beta.shape[0] != 1:
        raise ValueError("`raw_beta` must have shape [1, B, H].")
    if initial_state.ndim != 4:
        raise ValueError("`initial_state` must have shape [cache, H, V, K].")
    _, H, V, K = initial_state.shape
    if raw_g.stride()[2:] != (K, 1):
        raise ValueError("`raw_g` must be contiguous within each token.")
    if raw_beta.stride(2) != 1:
        raise ValueError("`raw_beta` heads must be contiguous.")
    if initial_state.stride()[1:] != (V * K, K, 1):
        raise ValueError("`initial_state` must be contiguous within each cache slot.")
    if state_indices.ndim != 1:
        raise ValueError("`state_indices` must be one-dimensional.")
    if A_log.ndim != 1 or not A_log.is_contiguous():
        raise ValueError("`A_log` must be contiguous and one-dimensional.")
    if not dt_bias.is_contiguous():
        raise ValueError("`dt_bias` must be contiguous.")

    device = mixed_qkv.device
    if any(
        x.device != device
        for x in (raw_g, raw_beta, A_log, dt_bias, initial_state, state_indices)
    ):
        raise ValueError("All packed KDA inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if raw_g.shape != (1, B, H, K):
        raise ValueError(f"Unexpected raw gate shape {tuple(raw_g.shape)}.")
    if raw_beta.shape != (1, B, H):
        raise ValueError(f"Unexpected raw beta shape {tuple(raw_beta.shape)}.")
    if mixed_qkv.shape[1] != 2 * H * K + H * V:
        raise ValueError(f"Unexpected packed QKV shape {tuple(mixed_qkv.shape)}.")
    if A_log.numel() != H or dt_bias.numel() != H * K:
        raise ValueError("`A_log` or `dt_bias` has an incompatible shape.")
    if state_indices.shape[0] != B:
        raise ValueError("`state_indices` must contain one entry per token.")

    BK = next_power_of_2(K)
    BV = min(next_power_of_2(V), 32)
    if scale is None:
        scale = K ** -0.5

    out = torch.empty((1, B, H, V), dtype=mixed_qkv.dtype, device=device)
    grid = (cdiv(V, BV), B * H)
    _fused_recurrent_kda_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        out=out,
        state=initial_state,
        state_indices=state_indices,
        lower_bound=lower_bound or 0.0,
        scale=scale,
        stride_mixed_token=mixed_qkv.stride(0),
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        stride_state_token=initial_state.stride(0),
        stride_state_indices=state_indices.stride(0),
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=4,
        num_stages=2,
        launch_pdl=_is_arch_support_pdl(),
    )
    return out, initial_state


__all__ = [
    "causal_conv1d_update",
    "fused_recurrent_kda_packed_decode",
    "NULL_BLOCK_ID",
]
