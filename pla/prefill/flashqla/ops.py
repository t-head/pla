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

"""FlashQLA PPU1.5 CuTe — high-level end-to-end forward API.

Mirrors ppu15/flashqla_ppu_ops.py. Non-CP single/multi-sequence inference path:

    g_cum = chunk_local_cumsum(g, cu_seqlens, chunk_indices)
    A     = kkt_solve(k, beta)             # WY lower-tri factor (beta baked in)
    o, ht = fused_fwd(q, k, v, A, g_cum, beta, h0)

The CP-preprocessing sub-pipeline (get_warmup_chunks -> prepare_h -> correct_h0)
is used only when a long sequence is split across CUs; for standard inference
the fused_fwd kernel is fed an identity CP mapping so every cp segment is its
own raw batch.
"""
import math
from typing import Optional, Set, Tuple

import torch

from . import _C as flashqla_cute
from .. import _sanity
from .private.cp_preprocess import (
    calc_cp_seqs,
    tc01_overlap_warmup_threshold,
)

CHUNK_SIZE = 32          # Unified chunk size (cumsum, get_warmup, CP splitting, fused_fwd, etc.)
FF_CHUNK_SIZE = 32       # Same as CHUNK_SIZE (kept for backward compat references)
DK = 128
DV = 128
# Recurrent-state pool dtypes. bf16 halves the resident pool; the recurrence
# itself is fp32 either way.
_STATE_DTYPES = (torch.float32, torch.bfloat16)
_chunk_indices_cache = []
_input_validation_signature = None
SUPPORTED_HEAD_CONFIGS = frozenset({
    (128, 16),
    (64, 16),
    (64, 8),
    (48, 16),
    (32, 16),
    (32, 8),
    (32, 4),
    (24, 8),
    (16, 8),
    (16, 4),
    (16, 2),
    (12, 4),
    (8, 4),
    (8, 2),
    (8, 1),
    (6, 2),
    (4, 2),
})


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _resolve_raw_cu(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    raw_cu: Optional[Tuple[int, ...]] = None,
) -> Tuple[Tuple[int, ...], Optional[int]]:
    """Resolve host offsets needed by chunk/CP planning."""
    try:
        tensor_version = cu_seqlens._version
    except RuntimeError:
        tensor_version = None
    if raw_cu is not None:
        return tuple(raw_cu), tensor_version

    # Reuse the values embedded in a cached chunk plan. Inference tensors
    # without a version counter synchronize again rather than trusting object
    # identity. Serving avoids this path by supplying CPU offsets directly.
    if tensor_version is not None:
        for cached_cu, cached_version, cached_key, _, _, _ in _chunk_indices_cache:
            if (cached_cu is cu_seqlens and cached_version == tensor_version and
                    cached_key[2] == chunk_size):
                return cached_key[3], tensor_version
    _sanity.require_host_read(cu_seqlens, "chunk planning")
    resolved = tuple(cu_seqlens.tolist())
    return resolved, tensor_version


def _current_stream_key(tensor: torch.Tensor) -> Tuple[Optional[int], int]:
    device_index = tensor.device.index
    stream_id = int(torch._C._cuda_getCurrentStream(device_index)[0])
    return device_index, stream_id


def _activate_chunk_indices(
    chunk_indices: torch.Tensor,
    ready_event: torch.cuda.Event,
    active_streams: Set[Tuple[Optional[int], int]],
) -> None:
    """Make cached device metadata ready and allocator-safe on this stream."""
    stream_key = _current_stream_key(chunk_indices)
    if stream_key in active_streams:
        return
    stream = torch.cuda.current_stream(chunk_indices.device)
    stream.wait_event(ready_event)
    chunk_indices.record_stream(stream)
    active_streams.add(stream_key)


def _prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    raw_cu: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Map packed chunks to (sequence, local chunk), caching recent metadata."""
    raw_cu, tensor_version = _resolve_raw_cu(
        cu_seqlens, chunk_size, raw_cu)

    cache_key = (cu_seqlens.device.type, cu_seqlens.device.index,
                 chunk_size, raw_cu)
    for cache_idx, (_, _, cached_key, cached_indices, ready_event,
                    active_streams) in enumerate(_chunk_indices_cache):
        if cached_key == cache_key:
            _activate_chunk_indices(
                cached_indices, ready_event, active_streams)
            entry = (cu_seqlens, tensor_version, cache_key, cached_indices,
                     ready_event, active_streams)
            if cache_idx:
                cache_order = list(_chunk_indices_cache)
                cache_order.pop(cache_idx)
                _chunk_indices_cache[:] = [entry, *cache_order[:3]]
            else:
                _chunk_indices_cache[0] = entry
            return cached_indices

    indices = [
        (seq_idx, chunk_idx)
        for seq_idx in range(len(raw_cu) - 1)
        for chunk_idx in range(_cdiv(raw_cu[seq_idx + 1] - raw_cu[seq_idx], chunk_size))
    ]
    chunk_indices = torch.tensor(
        indices, dtype=torch.int32, device=cu_seqlens.device)
    stream = torch.cuda.current_stream(chunk_indices.device)
    ready_event = torch.cuda.Event()
    ready_event.record(stream)
    chunk_indices.record_stream(stream)
    active_streams = {
        (chunk_indices.device.index, int(stream.stream_id))
    }
    entry = (cu_seqlens, tensor_version, cache_key, chunk_indices,
             ready_event, active_streams)
    cache_order = [
        cached for cached in _chunk_indices_cache
        if cached[2] != cache_key
    ]
    _chunk_indices_cache[:] = [entry, *cache_order[:3]]
    return chunk_indices


def _chunk_local_cumsum(
    g: torch.Tensor,                       # [1, T, H] fp32
    chunk_size: int = CHUNK_SIZE,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Chunk-local cumulative sum of gate values.

    GPU path uses the dedicated flashqla_cute.chunk_local_cumsum kernel
    (replaces the old pure-Python per-chunk loop). CPU tensors fall back to
    the reference loop.
    """
    _, T, _ = g.shape
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=g.device)
    if _sanity.enabled():
        _sanity.check(chunk_size > 0, "chunk_size must be positive")
    raw_cu = _sanity.offsets(cu_seqlens, T, reference=g)
    if g.is_cuda and chunk_size == CHUNK_SIZE:
        if chunk_indices is None:
            chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size, raw_cu)
        return flashqla_cute.chunk_local_cumsum(
            g.contiguous(), cu_seqlens.to(torch.int32), chunk_indices)
    # Reference fallback (CPU / non-default chunk size).
    _sanity.require_host_read(cu_seqlens, "non-default chunk cumsum")
    g_cum = torch.zeros_like(g)
    batch_size = cu_seqlens.size(0) - 1
    for b in range(batch_size):
        start = int(cu_seqlens[b].item())
        end = int(cu_seqlens[b + 1].item())
        for c_start in range(start, end, chunk_size):
            c_end = min(c_start + chunk_size, end)
            g_cum[:, c_start:c_end, :] = torch.cumsum(g[:, c_start:c_end, :], dim=1)
    return g_cum


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,                       # [1, T, Hg, DK] fp16
    k: torch.Tensor,                       # [1, T, Hg, DK] fp16
    v: torch.Tensor,                       # [1, T, H, DV] fp16
    g: Optional[torch.Tensor],             # [1, T, H] fp32 log gate
    beta: Optional[torch.Tensor],          # [1, T, H] fp32 activated beta
    scale: Optional[float] = None,
    # Legacy state is [batch,H,DK,DV]; indexed state pool is [slots,H,DV,DK].
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,  # [batch] int32 slots
    cu_seqlens: Optional[torch.Tensor] = None,      # [batch+1] int32
    output_final_state: bool = True,
    auto_cp: bool = True,
    overlap_warmup_threshold: Optional[float] = None,
    force_tc01_cp_candidate: bool = False,
    normalize_qk: bool = False,
    raw_cu_seqlens: Optional[Tuple[int, ...]] = None,
    raw_gate_a: Optional[torch.Tensor] = None,   # [T,H] b16
    raw_beta_b: Optional[torch.Tensor] = None,   # [T,H] b16
    A_log: Optional[torch.Tensor] = None,        # [H] fp32
    dt_bias: Optional[torch.Tensor] = None,      # [H] fp32
    final_state_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, Optional[torch.Tensor]]:
    """Returns (g_cum_32, A, O, None, final_state).

    The pool is updated in place even when output_final_state is false.
    ``cu_seqlens`` covers the real-token prefix and may end before the physical
    token length; in that case ``O`` keeps the physical shape and its padded
    suffix is zero. Serving callers may pass the same offsets as a trusted Host
    tuple in ``raw_cu_seqlens`` to avoid a Device-to-Host planning read.

    The recurrent state may be fp32 or bf16. The recurrence itself always runs
    in fp32 registers and only converts at the two memory boundaries, so bf16
    halves the resident pool at the cost of one rounding per prefill segment.
    A single dtype covers ``initial_state``, the indexed pool and the returned
    ``final_state``: ``final_state_dtype`` selects it and must agree with
    ``initial_state`` when one is supplied, defaulting to that dtype, or to fp32
    when there is no incoming state.

    CP preserves full prefix history, including states created by early tokens
    when initial_state is None. No log-decay threshold truncates that history.
    ``overlap_warmup_threshold`` enables the overlap-replay candidate for CP
    plans. Indexed state is snapshotted before replay so continuation CTAs never
    race the final-state writer. Legacy nonzero state keeps the established
    prepare/correct path. ``force_tc01_cp_candidate`` is an offline sweep hook
    that bypasses production admission while preserving planner feasibility.
    """
    global _input_validation_signature
    raw_gate_args = (raw_gate_a, raw_beta_b, A_log, dt_bias)
    use_raw_gating = any(tensor is not None for tensor in raw_gate_args)
    num_tokens = q.size(1)
    H = v.size(2)
    Hg = k.size(2)
    if scale is None:
        scale = q.size(-1) ** -0.5
    validation_tensors = (
        q, k, v, g, beta, raw_gate_a, raw_beta_b, A_log, dt_bias,
        initial_state, initial_state_indices,
    )
    validation_signature = (
        tuple(
            None if tensor is None else (
                tensor.device.type, tensor.device.index, tensor.dtype,
                tuple(tensor.shape), tuple(tensor.stride()),
            )
            for tensor in validation_tensors
        ),
        normalize_qk, auto_cp, overlap_warmup_threshold,
        force_tc01_cp_candidate,
    )
    if validation_signature != _input_validation_signature:
        if not q.is_cuda:
            raise ValueError("q must be a CUDA tensor")
        if (overlap_warmup_threshold is not None and
                (not math.isfinite(overlap_warmup_threshold) or
                 overlap_warmup_threshold >= 0.0)):
            raise ValueError("overlap_warmup_threshold must be finite and negative")
        if force_tc01_cp_candidate and not auto_cp:
            raise ValueError("force_tc01_cp_candidate requires auto_cp=True")
        if use_raw_gating and not all(tensor is not None for tensor in raw_gate_args):
            raise ValueError(
                "raw_gate_a, raw_beta_b, A_log, and dt_bias must be supplied together")
        if use_raw_gating and not normalize_qk:
            raise ValueError("raw GDN gating requires normalize_qk=True")
        if not use_raw_gating and (g is None or beta is None):
            raise ValueError("g and beta are required unless raw GDN gating is used")

        named_inputs = [("k", k), ("v", v)]
        if use_raw_gating:
            named_inputs.extend((
                ("raw_gate_a", raw_gate_a), ("raw_beta_b", raw_beta_b),
                ("A_log", A_log), ("dt_bias", dt_bias),
            ))
        else:
            named_inputs.extend((("g", g), ("beta", beta)))
        for name, tensor in named_inputs:
            if tensor.device != q.device:
                raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
        if initial_state is not None and initial_state.device != q.device:
            raise ValueError(
                f"initial_state must be on {q.device}, got {initial_state.device}")
        if initial_state_indices is not None and initial_state is None:
            raise ValueError("initial_state is required with initial_state_indices")

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must be rank-4 tensors")
        if use_raw_gating:
            if raw_gate_a.ndim != 2 or raw_beta_b.ndim != 2:
                raise ValueError("raw_gate_a and raw_beta_b must be rank-2 tensors")
            if A_log.ndim != 1 or dt_bias.ndim != 1:
                raise ValueError("A_log and dt_bias must be rank-1 tensors")
        elif g.ndim != 3 or beta.ndim != 3:
            raise ValueError("g and beta must be rank-3 tensors")
        for name, tensor in (("q", q), *named_inputs):
            if use_raw_gating and name in ("raw_gate_a", "raw_beta_b"):
                if tensor.stride(1) != 1 or tensor.stride(0) < tensor.size(1):
                    raise ValueError(
                        f"{name} must be head-contiguous with a positive token stride")
            elif use_raw_gating and name in ("q", "k", "v"):
                if (tensor.ndim != 4 or tensor.stride(3) != 1 or
                        tensor.stride(2) != tensor.size(3) or
                        tensor.stride(1) < tensor.size(2) * tensor.size(3)):
                    raise ValueError(
                        f"{name} must be head-contiguous with a positive token stride")
            elif not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("q, k, and v must be fp16 or bf16")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q, k, and v must have the same dtype")
        if use_raw_gating:
            if raw_gate_a.dtype != q.dtype or raw_beta_b.dtype != q.dtype:
                raise ValueError("raw_gate_a and raw_beta_b must match q dtype")
            if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
                raise ValueError("A_log and dt_bias must be fp32")
        elif g.dtype != torch.float32 or beta.dtype != torch.float32:
            raise ValueError("g and beta must be fp32")

        if q.size(0) != 1 or k.size(0) != 1 or v.size(0) != 1:
            raise ValueError("q, k, and v must have batch dimension 1")
        if k.size(1) != num_tokens or v.size(1) != num_tokens:
            raise ValueError("q, k, and v must have the same token length")
        if q.size(2) != Hg:
            raise ValueError(
                f"q and k must have the same number of heads, got {q.size(2)} and {Hg}")
        if use_raw_gating:
            if (raw_gate_a.shape != (num_tokens, H) or
                    raw_beta_b.shape != (num_tokens, H)):
                raise ValueError(
                    f"raw_gate_a and raw_beta_b must be [{num_tokens},{H}]")
            if A_log.numel() != H or dt_bias.numel() != H:
                raise ValueError(f"A_log and dt_bias must have H={H} elements")
        else:
            if (g.size(0) != 1 or beta.size(0) != 1 or
                    g.size(1) != num_tokens or beta.size(1) != num_tokens):
                raise ValueError(
                    "g and beta must be [1,T,H] with the same token length as q")
            if g.size(2) != H or beta.size(2) != H:
                raise ValueError(f"g and beta must have H={H} value heads")
        if (H, Hg) not in SUPPORTED_HEAD_CONFIGS:
            raise ValueError(
                f"FlashQLA does not support head configuration (H={H}, Hg={Hg})")
        if q.size(-1) != DK or k.size(-1) != DK or v.size(-1) != DV:
            raise ValueError(
                "FlashQLA requires q/k head_dim=128 and v head_dim=128, got "
                f"q={q.size(-1)}, k={k.size(-1)}, v={v.size(-1)}")

        if num_tokens == 0:
            raise ValueError("FlashQLA does not support an empty token dimension")
        _input_validation_signature = validation_signature

    raw_cu = _sanity.offsets(
        cu_seqlens, num_tokens, raw_cu_seqlens, reference=q)
    if cu_seqlens is None:
        raw_cu = (0, num_tokens)
        cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=q.device)
    cu32 = cu_seqlens.to(
        device=q.device, dtype=torch.int32).contiguous()
    if cu32.ndim != 1:
        raise ValueError("cu_seqlens must be a rank-1 tensor")
    batch = cu32.numel() - 1
    indexed_state = initial_state_indices is not None
    if indexed_state:
        if (initial_state.dtype not in _STATE_DTYPES or
                not initial_state.is_contiguous()):
            raise ValueError(
                "indexed initial_state must be contiguous fp32 or bf16")
        if (initial_state.ndim != 4 or initial_state.size(1) != H or
                initial_state.size(2) != DV or initial_state.size(3) != DK):
            raise ValueError("indexed initial_state must be [num_slots,H,DV,DK]")
        if (initial_state_indices.device != q.device or
                initial_state_indices.dtype != torch.int32 or
                not initial_state_indices.is_contiguous() or
                initial_state_indices.ndim != 1 or
                initial_state_indices.numel() != batch):
            raise ValueError(
                "initial_state_indices must be contiguous int32 [batch] on q.device")
    elif initial_state is not None:
        if (initial_state.dtype not in _STATE_DTYPES or
                not initial_state.is_contiguous()):
            raise ValueError("initial_state must be contiguous fp32 or bf16")
        if (initial_state.ndim != 4 or initial_state.size(0) < batch or
                initial_state.size(1) != H or initial_state.size(2) != DK or
                initial_state.size(3) != DV):
            raise ValueError(
                "initial_state must be [num_states>=batch,H,DK,DV]")

    # The kernels carry a single compile-time state dtype shared by the incoming
    # state, the pool and the returned state, so an incoming state fixes it.
    if final_state_dtype is None:
        state_dtype = (initial_state.dtype if initial_state is not None
                       else torch.float32)
    else:
        if final_state_dtype not in _STATE_DTYPES:
            raise ValueError("final_state_dtype must be fp32 or bf16")
        if initial_state is not None and final_state_dtype != initial_state.dtype:
            raise ValueError(
                "final_state_dtype must match the initial_state dtype")
        state_dtype = final_state_dtype

    if raw_cu is None:
        raw_cu, _ = _resolve_raw_cu(cu32, CHUNK_SIZE)
    valid_num_tokens = raw_cu[-1]
    if initial_state is not None:
        initial_state.record_stream(
            torch.cuda.current_stream(initial_state.device))
    if initial_state_indices is not None:
        initial_state_indices.record_stream(
            torch.cuda.current_stream(initial_state_indices.device))
    chunk_indices = _prepare_chunk_indices(cu32, CHUNK_SIZE, raw_cu)
    # Framework adapters may pass raw q/k and fold both L2 normalizations into
    # the cumsum launch. The default preserves the existing normalized-q/k API.
    if use_raw_gating:
        g_cum_32, beta, q, k = (
            flashqla_cute.chunk_local_cumsum_qk_norm_gated(
                raw_gate_a, raw_beta_b, A_log, dt_bias,
                cu32, chunk_indices, q, k, valid_num_tokens)
        )
    elif normalize_qk:
        g_cum_32, q, k = flashqla_cute.chunk_local_cumsum_qk_norm(
            g, cu32, chunk_indices, q, k)
    else:
        g_cum_32 = _chunk_local_cumsum(g, CHUNK_SIZE, cu32, chunk_indices)

    # Step 2: KKT solve (WY factor). kkt_solve internally computes K@K^T at CS=32.
    # A padded single sequence also needs the explicit chunk map: the implicit
    # single-sequence path derives its grid from the physical token dimension.
    A = flashqla_cute.kkt_solve(
        k, beta, cu32, H,
        chunk_indices if batch > 1 or valid_num_tokens < num_tokens else None)

    production_overlap_threshold = None
    if (auto_cp and overlap_warmup_threshold is None and
            not force_tc01_cp_candidate):
        production_overlap_threshold = tc01_overlap_warmup_threshold(
            cu32, raw_cu, CHUNK_SIZE, H, Hg,
            use_raw_gating=use_raw_gating,
            indexed_state=indexed_state,
            has_initial_state=initial_state is not None,
        )

    cp_plan = (
        calc_cp_seqs(
            cu32, CHUNK_SIZE, H, raw_cu, num_k_heads=Hg,
            tc01_force_candidate=force_tc01_cp_candidate,
            tc01_enable_overlap_candidate=(
                production_overlap_threshold is not None),
            tc01_packed_raw_gate=use_raw_gating,
        ) if auto_cp else None
    )

    # Step 3: CP decision + state reconstruction. The established path computes
    # every segment state before fused_fwd. The overlap candidate instead gives
    # each continuation CTA enough predecessor chunks to reconstruct its state.
    cp_h0 = None
    raw_state_input = None
    num_warmup_flat = fallback_mask_flat = None
    cp_cu_i32 = cp_seq_map_i32 = raw_cu_i32 = None
    if cp_plan is not None:
        use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask = cp_plan
        if use_cp:
            cp_cu_i32 = cp_cu_seqlens.to(torch.int32)
            cp_batch = cp_cu_i32.numel() - 1
            effective_overlap_threshold = (
                overlap_warmup_threshold
                if overlap_warmup_threshold is not None
                else production_overlap_threshold
            )
            # A fresh sequence starts at zero, but earlier tokens can create
            # the entire state consumed by a continuation segment. Log decay
            # alone bounds neither that state's magnitude nor relative error.
            # Preserve all prefix updates and incoming-state transitions in
            # both overlap replay and prepare/correct, even without raw h0.
            segment_chunks = torch.div(
                cp_cu_i32[1:] - cp_cu_i32[:-1] + CHUNK_SIZE - 1,
                CHUNK_SIZE, rounding_mode="floor")
            num_warmup = segment_chunks[:, None].expand(-1, H).contiguous()
            fallback_mask = torch.ones(
                (cp_batch, H), dtype=torch.int8, device=q.device)
            use_overlap_warmup = (
                effective_overlap_threshold is not None and
                (indexed_state or initial_state is None)
            )
            if use_overlap_warmup:
                num_warmup_flat = num_warmup.reshape(-1).contiguous()
                fallback_mask_flat = fallback_mask.reshape(-1).contiguous()
                if indexed_state:
                    # The compact read-only snapshot prevents first/final CP
                    # CTAs from racing through the in-place serving state pool.
                    raw_state_input = torch.index_select(
                        initial_state, 0, initial_state_indices)
            else:
                ht_buf, mt_buf = flashqla_cute.prepare_h(
                    k, v, A, g_cum_32, beta, cp_cu_i32,
                    num_warmup.reshape(-1).contiguous(), H, Hg)
                ht_buf = ht_buf.view(cp_batch, H, DK, DV)
                mt_buf = mt_buf.view(cp_batch, H, DK, DK)
                fb_2d = fallback_mask.view(cp_batch, H).contiguous()
                # cp_h0 goes straight back into fused_fwd as its h0, so it is
                # produced in the state pool's own dtype.
                cp_h0 = flashqla_cute.correct_h0(
                    ht_buf, mt_buf, fb_2d, seq_map_r2c.to(torch.int32),
                    batch, initial_state,
                    initial_state_indices if indexed_state else None,
                    indexed_state, state_dtype == torch.bfloat16)
            cp_seq_map_i32 = seq_map_c2r.to(torch.int32)
            raw_cu_i32 = cu32

    # Step 4: fused forward — CP path (grid = cp_batch*H) or non-CP path.
    if cp_seq_map_i32 is not None:
        o, ht = flashqla_cute.fused_fwd(
            q, k, v, A, beta, g_cum_32, cp_cu_i32, cp_seq_map_i32, raw_cu_i32,
            valid_num_tokens, H, Hg, float(scale), cp_h0,
            initial_state if indexed_state else None,
            initial_state_indices if indexed_state else None,
            output_final_state, state_dtype == torch.bfloat16,
            raw_state_input, num_warmup_flat, fallback_mask_flat)
    else:
        h0 = (initial_state if initial_state is not None and not indexed_state
              else torch.empty(0, device=q.device))
        o, ht = flashqla_cute.fused_fwd(
            q, k, v, A, beta, g_cum_32, cu32, None, cu32,
            valid_num_tokens, H, Hg, float(scale), h0,
            initial_state if indexed_state else None,
            initial_state_indices if indexed_state else None,
            output_final_state, state_dtype == torch.bfloat16,
            None, None, None)

    if output_final_state:
        ht = initial_state[initial_state_indices] if indexed_state else ht
    else:
        ht = None
    return g_cum_32, A, o, None, ht


def gdn_prefill_packed_qkv_fwd(
    packed_qkv: torch.Tensor,                # [T, 2*Hg*DK + H*DV] b16
    raw_gate_a: torch.Tensor,                # [T,H] b16
    raw_beta_b: torch.Tensor,                # [T,H] b16
    A_log: torch.Tensor,                     # [H] fp32
    dt_bias: torch.Tensor,                   # [H] fp32
    cu_seqlens: torch.Tensor,                # [batch+1] int32
    *,
    num_qk_heads: int,
    num_v_heads: int,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    auto_cp: bool = True,
    overlap_warmup_threshold: Optional[float] = None,
    force_tc01_cp_candidate: bool = False,
    raw_cu_seqlens: Optional[Tuple[int, ...]] = None,
    final_state_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, Optional[torch.Tensor]]:
    """Run GDN prefill directly on a post-convolution ``Q|K|V`` buffer.

    ``packed_qkv`` follows the common SGLang/vLLM layout ``[T, Q|K|V]``.
    Features inside one token must be contiguous; the token stride may be
    larger than the logical row width. Q/K/V views are aliases of this tensor:
    the only materialized Q/K tensors are the normalized buffers already
    required by KKT and the recurrence, while V is consumed in place.

    The return contract and state semantics are identical to
    :func:`chunk_gated_delta_rule_fwd`.
    """
    if not packed_qkv.is_cuda:
        raise ValueError("packed_qkv must be a CUDA tensor")
    if packed_qkv.ndim != 2:
        raise ValueError("packed_qkv must be rank-2 [T,QKV]")
    if packed_qkv.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("packed_qkv must be fp16 or bf16")
    if packed_qkv.stride(1) != 1:
        raise ValueError("packed_qkv features must be contiguous")
    if num_qk_heads <= 0 or num_v_heads <= 0:
        raise ValueError("num_qk_heads and num_v_heads must be positive")

    T = packed_qkv.size(0)
    if T == 0:
        raise ValueError("GDN packed prefill does not support an empty token dimension")
    Hg = int(num_qk_heads)
    H = int(num_v_heads)
    qk_width = Hg * DK
    packed_width = 2 * qk_width + H * DV
    if packed_qkv.size(1) != packed_width:
        raise ValueError(
            f"packed_qkv width must be {packed_width} for H={H}, Hg={Hg}, "
            f"got {packed_qkv.size(1)}")
    token_stride = packed_qkv.stride(0)
    if token_stride < packed_width:
        raise ValueError(
            "packed_qkv token stride must cover the complete Q|K|V row")

    storage_offset = packed_qkv.storage_offset()
    batch_stride = max(T, 1) * token_stride
    q = packed_qkv.as_strided(
        (1, T, Hg, DK), (batch_stride, token_stride, DK, 1),
        storage_offset=storage_offset)
    k = packed_qkv.as_strided(
        (1, T, Hg, DK), (batch_stride, token_stride, DK, 1),
        storage_offset=storage_offset + qk_width)
    v = packed_qkv.as_strided(
        (1, T, H, DV), (batch_stride, token_stride, DV, 1),
        storage_offset=storage_offset + 2 * qk_width)

    return chunk_gated_delta_rule_fwd(
        q, k, v, None, None,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
        auto_cp=auto_cp,
        overlap_warmup_threshold=overlap_warmup_threshold,
        force_tc01_cp_candidate=force_tc01_cp_candidate,
        normalize_qk=True,
        raw_cu_seqlens=raw_cu_seqlens,
        raw_gate_a=raw_gate_a,
        raw_beta_b=raw_beta_b,
        A_log=A_log,
        dt_bias=dt_bias,
        final_state_dtype=final_state_dtype,
    )
