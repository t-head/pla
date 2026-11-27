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
import atexit
import math
import os
from typing import Optional, Sequence, Set, Tuple, Union

import torch

from .. import _sanity
from . import _C as flashqla_cute
from .private.tc02_folded_policy import folded_schedule
from .private.cp_preprocess import (
    calc_cp_seqs,
    _cuda_device_properties,
    _cp_plan_cache,
)
from .private.metadata import capture_metadata, metadata_tensor, metadata_upload
from .private.packed_coefficients import tc01_packed_coefficients_mode

CHUNK_SIZE = 32          # Unified chunk size (cumsum, get_warmup, CP splitting, fused_fwd, etc.)
FF_CHUNK_SIZE = 32       # Same as CHUNK_SIZE (kept for backward compat references)
DK = 128
DV = 128
# Recurrent-state pool dtypes. bf16 halves the resident pool; the recurrence
# itself is fp32 either way.
_STATE_DTYPES = (torch.float32, torch.bfloat16)
_MAX_REPAIR_SNAPSHOT_BYTES = 128 << 20
_chunk_indices_cache = []


def _shutdown_metadata_graph_owner(
    chunk_cache=_chunk_indices_cache,
    plan_cache=_cp_plan_cache,
    shutdown=flashqla_cute._metadata_owner_shutdown,
):
    # Bind owning lists and the native callable at import time: shutdown must
    # not depend on imports or module globals during interpreter teardown.
    chunk_cache.clear()
    plan_cache.clear()
    shutdown()


atexit.register(_shutdown_metadata_graph_owner)

_input_validation_signature = None
_packed_coefficients_available = bool(getattr(
    flashqla_cute, "_packed_coefficients_tc01_enabled", False))
SUPPORTED_HEAD_CONFIGS = frozenset({
    (128, 16),
    (64, 16),
    (64, 8),
    (48, 16),
    (32, 16),
    (32, 8),
    (32, 4),
    (24, 8),
    (16, 16),
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
        scope = capture_metadata.get()
        candidates = (*scope.chunks.values(), *_chunk_indices_cache) if scope else _chunk_indices_cache
        for cached_cu, cached_version, cached_key, _, _, _ in candidates:
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
    if active_streams is None:
        # The native entry contains only C++ Tensor/Event/stream state.
        ready_event.activate()
        return
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
    scope = capture_metadata.get()
    if scope is not None and cache_key in scope.chunks:
        _, _, _, indices, event, streams = scope.chunks[cache_key]
        _activate_chunk_indices(indices, event, streams)
        return indices
    for cache_idx, (_, _, cached_key, cached_indices, ready_event,
                    active_streams) in enumerate(_chunk_indices_cache):
        if cached_key == cache_key:
            _activate_chunk_indices(
                cached_indices, ready_event, active_streams)
            entry = (cu_seqlens, tensor_version, cache_key, cached_indices,
                     ready_event, active_streams)
            if scope is not None and active_streams is None:
                scope.chunks[cache_key] = entry
            if cache_idx:
                cache_order = list(_chunk_indices_cache)
                cache_order.pop(cache_idx)
                _chunk_indices_cache[:] = [entry, *cache_order[:3]]
            else:
                _chunk_indices_cache[0] = entry
            return cached_indices

    native_owner = _cuda_device_properties(cu_seqlens.device.index) == (8, 9, 39)
    if native_owner:
        with torch.cuda.device(cu_seqlens.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("chunk metadata cold miss must be warmed before capture")
    indices = [
        (seq_idx, chunk_idx)
        for seq_idx in range(len(raw_cu) - 1)
        for chunk_idx in range(_cdiv(raw_cu[seq_idx + 1] - raw_cu[seq_idx], chunk_size))
    ]
    with metadata_upload(cu_seqlens.device, native_owner):
        chunk_indices = metadata_tensor(indices, torch.int32, cu_seqlens.device, native_owner)
        if native_owner:
            ready_event = flashqla_cute._make_chunk_metadata_entry(chunk_indices)
    if native_owner:
        ready_event.activate()
        active_streams = None
    else:
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
    if scope is not None and active_streams is None:
        scope.chunks[cache_key] = entry
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
    raw_cu_seqlens: Optional[Union[Sequence[int], torch.Tensor]] = None,
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
    suffix is zero. Serving callers may supply the same offsets as a host
    sequence or CPU tensor in ``raw_cu_seqlens`` to avoid a device-to-host
    planning read. Pass device tensors as ``cu_seqlens``.

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
    race the final-state writer. Automatic tc01 CP admission is disabled;
    an explicit threshold alone does not opt a shape into CP. Legacy nonzero
    state keeps the established prepare/correct path.
    ``force_tc01_cp_candidate`` is an offline sweep hook
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

    if raw_cu_seqlens is not None and not isinstance(raw_cu_seqlens, tuple):
        if isinstance(raw_cu_seqlens, torch.Tensor):
            if (raw_cu_seqlens.device.type != "cpu" or raw_cu_seqlens.ndim != 1
                    or raw_cu_seqlens.dtype not in (torch.int32, torch.int64)):
                raise ValueError("raw_cu_seqlens must be a 1-D CPU int32 or int64 tensor")
            raw_cu_seqlens = raw_cu_seqlens.tolist()
        raw_cu_seqlens = tuple(raw_cu_seqlens)
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
    direct_precision = os.environ.get("PLA_FLASHQLA_DIRECT_PRECISION") == "1"
    force_precision = os.environ.get("PLA_FLASHQLA_FORCE_PRECISION_REPAIR") == "1"
    folded_max_local_chunks = None
    complete_layout = valid_num_tokens == num_tokens
    if (os.environ.get("PLA_FLASHQLA_TC02_FOLDED", "1") != "0"
            and not direct_precision and not force_precision
            and q.dtype == torch.bfloat16 and indexed_state
            and initial_state.dtype == torch.float32
            and overlap_warmup_threshold is None and not force_tc01_cp_candidate
            and (normalize_qk or (q.is_contiguous() and k.is_contiguous()))
            and (not use_raw_gating or auto_cp and complete_layout)
            and _cuda_device_properties(q.device.index) == (8, 9, 39)):
        folded_max_local_chunks = folded_schedule(
            H, Hg, raw_cu, use_raw_gating, auto_cp,
            auto_cp and complete_layout and normalize_qk)
    use_folded = folded_max_local_chunks is not None
    packed_short = (
        indexed_state and H == 8 and Hg in (2, 4) and batch in (11, 22)
        and all(0 < b - a <= 256 for a, b in zip(raw_cu, raw_cu[1:])))
    # BF16 can round each weak CP decay to one. Use the FP32 transition
    # path for long sequences and the 24K+ single-sequence 12/4 schedule.
    precise_transitions = use_folded and (
        (batch == 1 and (H, Hg) == (12, 4) and valid_num_tokens >= 24576)
        or any(
            b - a > 65536 for a, b in zip(raw_cu, raw_cu[1:])))
    precise_folded = use_folded and (
        (H, Hg) != (24, 8) or precise_transitions)
    # Framework adapters may pass raw q/k and fold both L2 normalizations into
    # the cumsum launch. The default preserves the existing normalized-q/k API.
    if use_raw_gating:
        # The precision guard classifies saturation using the FP32 sigmoid.
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

    precision_inputs = None
    precision_flags = None
    live_initial = None
    # The structural guard unconditionally selects every nonempty sequence
    # in this range. Returning directly avoids computing output twice.
    repair_all_short = (
        indexed_state and batch == 1 and 0 < num_tokens <= 3 and
        _cuda_device_properties(q.device.index) == (8, 9, 39))
    tiny_indexed = repair_all_short and H == 32
    if not flashqla_cute._precision_bypass_enabled():
        if os.environ.get("PLA_FLASHQLA_PRECISION_ACCUMULATOR", "fp64") != "fp64":
            raise ValueError("precision diagnostics support only FP64 accumulation")
        repair_dv_override = os.environ.get("PLA_FLASHQLA_PRECISION_DV_PER_WARP")
        dv_per_warp = int(repair_dv_override if repair_dv_override is not None else "2")
        if dv_per_warp not in (1, 2, 4):
            raise ValueError("precision DV per warp must be 1, 2, or 4")
        repair_stage_override = os.environ.get("PLA_FLASHQLA_PRECISION_STAGE_TOKENS")
        staged_tokens = int(repair_stage_override if repair_stage_override is not None else "32")
        if staged_tokens not in (0, 32):
            raise ValueError("precision staged tokens must be 0 or 32")
        # Preserve raw recurrence inputs before fast state writers. The guard
        # is a structural fallback class, not a general error certificate.
        pool_storage = (initial_state.untyped_storage().data_ptr()
                        if indexed_state else None)

        def preserve_repair_input(tensor):
            if (tensor is not None and pool_storage is not None and
                    tensor.untyped_storage().data_ptr() == pool_storage):
                return tensor.clone(memory_format=torch.contiguous_format)
            return tensor

        repair_values = tuple(preserve_repair_input(tensor) for tensor in
                              (q, k, v, beta, g if not use_raw_gating else None,
                               raw_gate_a, A_log, dt_bias))
        # Unselected heads still use the fast path. They must read the same
        # preserved operands when a state write could cover a future token.
        q, k, v, beta = repair_values[:4]
        # Metadata must also remain available to the fast state writer and the
        # returned indexed gather when it shares the mutable pool's storage.
        cu32 = preserve_repair_input(cu32)
        if indexed_state:
            initial_state_indices = preserve_repair_input(initial_state_indices)
        # Four measured tc02 geometries. The original short path below stays
        # byte-for-byte intact; this distinct entry owns medium-only scheduling.
        medium_guard_snapshot = (
            not force_precision and not direct_precision and initial_state is not None and
            (((cu32.numel() == 33 and num_tokens == 4964 and H == 48 and Hg == 16)) or
             (cu32.numel() == 3 and
              ((num_tokens == 920 and H == 8 and Hg == 4) or
               (num_tokens == 1041 and H == 8 and Hg == 2))) or
             (cu32.numel() == 2 and H == 16 and
              ((Hg == 2 and num_tokens == 847) or (Hg == 4 and num_tokens == 823)))) and
            len(raw_cu) == cu32.numel() and
            _cuda_device_properties(q.device.index) == (8, 9, 39))
        quick_guard = (
            ((H in (8, 16, 24, 32) and 0 < num_tokens <= 640) or
             (H in (48, 64, 128) and 0 < num_tokens <= 192) or
             (H == 64 and Hg in (8, 16) and 192 < num_tokens <= 256) or
             (H == 8 and Hg == 2 and 768 <= num_tokens < 1024))
            and Hg in (1, 2, 4, 8, 16, 32, 64, 128)
            and len(raw_cu) == 2 and cu32.numel() == 2
            and _cuda_device_properties(q.device.index) == (8, 9, 39))
        short_guard_snapshot = (
            not force_precision and not direct_precision and initial_state is not None
            and quick_guard and repair_values[1].is_contiguous() and repair_values[3].is_contiguous())
        compact_guard_snapshot = (
            not force_precision and not direct_precision and packed_short
            and H == 8 and Hg in (2, 4) and batch in (11, 22)
            and repair_values[1].is_contiguous() and repair_values[3].is_contiguous()
            and _cuda_device_properties(q.device.index) == (8, 9, 39))
        wide_short_guard = (
            not force_precision and not direct_precision and initial_state is not None
            and H == 8 and Hg in (2, 4) and 768 <= num_tokens <= 1023
            and len(raw_cu) == 2 and cu32.numel() == 2
            and repair_values[1].is_contiguous() and repair_values[3].is_contiguous()
            and _cuda_device_properties(q.device.index) == (8, 9, 39))
        tc01_live_indexed = (
            indexed_state and use_raw_gating and complete_layout and q.dtype == torch.bfloat16
            and initial_state.dtype == torch.float32
            and ((batch > 1 and 12 <= H <= 32 and batch * H >= 128 and num_tokens >= 256)
                 or (batch == 1 and H >= 48 and 4096 <= num_tokens <= 32768))
            and _cuda_device_properties(q.device.index) == (8, 0, 64))
        # Packed PnD selects all single-token rows for repair. Their state
        # stays untouched by the fast forward, so repair can load it in place.
        tc02_live_pnd = False
        if (indexed_state and use_raw_gating and complete_layout
                and q.dtype == torch.bfloat16 and state_dtype == torch.float32
                and 8 <= batch <= 136 and 8 <= H <= 48 and 512 <= num_tokens <= 32768
                and _cuda_device_properties(q.device.index) == (8, 9, 39)):
            lengths = tuple(end - start for start, end in zip(raw_cu, raw_cu[1:]))
            decode_rows = lengths.count(1)
            prefill_rows = batch - decode_rows
            long_pnd = num_tokens >= 2048 and 1 <= prefill_rows <= 6
            # Short mixed batches can spend more time snapshotting than
            # repairing. Reuse the skipped state for these packed grids.
            short_pnd = ((H, Hg) in ((8, 1), (24, 8)) and batch <= 48
                         and max(lengths) <= 384 and 1 <= prefill_rows <= 7)
            tc02_live_pnd = (min(lengths) > 0 and 2 * decode_rows >= batch
                             and (long_pnd or short_pnd))
        if (initial_state is not None and not force_precision and not direct_precision
                and (tc01_live_indexed or tc02_live_pnd
                     or batch * H * DK * DV * 4 > _MAX_REPAIR_SNAPSHOT_BYTES)):
            # Fast kernels skip selected heads. Their original state remains
            # available to FP64 repair even if the other heads update in place.
            live_initial = initial_state
        if live_initial is not None:
            if tc01_live_indexed and num_tokens <= 512 and batch * H <= 256:
                short_requests = sum(0 < end - start <= 3 for start, end in zip(raw_cu, raw_cu[1:]))
                if short_requests * 2 >= batch:
                    # Most repairs cover only a few tokens. Read them directly
                    # and reuse each Q/K load across more value columns.
                    if repair_dv_override is None:
                        dv_per_warp = 4
                    if repair_stage_override is None:
                        staged_tokens = 0
            precision_flags = flashqla_cute._precision_guard(
                repair_values[1], cu32, not indexed_state, repair_values[3],
                complete_layout=complete_layout)
            snapshot = torch.empty(0, device=q.device, dtype=torch.float32)
        elif tiny_indexed:
            snapshot = torch.empty(0, device=q.device, dtype=torch.float32)
        elif medium_guard_snapshot or compact_guard_snapshot or wide_short_guard:
            precision_flags, snapshot = flashqla_cute._precision_guard_snapshot_medium(
                repair_values[1], cu32, not indexed_state, repair_values[3],
                initial_state, initial_state_indices, H)
        elif short_guard_snapshot:
            precision_flags, snapshot = flashqla_cute._precision_guard_snapshot_short(
                repair_values[1], cu32, not indexed_state, repair_values[3],
                initial_state, initial_state_indices, H)
        else:
            precision_flags = (None if force_precision or direct_precision else
                               flashqla_cute._precision_guard(
                                   repair_values[1], cu32, not indexed_state, repair_values[3],
                                   complete_layout=complete_layout))
            snapshot = (flashqla_cute._precision_snapshot(
                initial_state, initial_state_indices, cu32, H, precision_flags)
                if initial_state is not None else
                torch.empty(0, device=q.device, dtype=torch.float32))
        precision_inputs = repair_values + (snapshot,)

        def apply_precision(output, final):
            repair_q, repair_k, repair_v, repair_beta, repair_g, repair_a, repair_alog, repair_bias, original_h0 = precision_inputs
            arguments = (repair_q, repair_k, repair_v, repair_beta, repair_g,
                         repair_a, repair_alog, repair_bias, output, original_h0, final,
                         cu32, initial_state_indices if indexed_state else None, float(scale))
            flashqla_cute._precision_repair(
                *arguments, dv_per_warp=dv_per_warp, staged_tokens=staged_tokens,
                flags=precision_flags,
                fuse_suffix=(quick_guard or compact_guard_snapshot or wide_short_guard or ((medium_guard_snapshot and cu32.numel() <= 3) or
                    (cu32.numel() == 2 and H == 16 and Hg in (2, 4, 8) and
                     768 <= num_tokens <= 1408))),
                load_indexed_state=tiny_indexed, live_initial=live_initial)


    # Step 2: KKT solve (WY factor). kkt_solve internally computes K@K^T at CS=32.
    # A padded single sequence also needs the explicit chunk map: the implicit
    # single-sequence path derives its grid from the physical token dimension.
    packed = None
    packed_mode = 0
    if q.is_cuda and not direct_precision:
        major, minor, _ = _cuda_device_properties(q.device.index)
        packed_mode = tc01_packed_coefficients_mode(
            major, minor, valid_num_tokens, H, Hg,
            batch_size=batch, indexed_state=indexed_state,
            full_sequence=(raw_cu == (0, num_tokens) and valid_num_tokens == num_tokens),
            force_cp=force_tc01_cp_candidate,
            group_override=os.environ.get("FF_TC01_NGRP2", "")[:1] in ("0", "1"),
            available=_packed_coefficients_available)
    coefficients = None
    if use_folded:
        A, coefficients = flashqla_cute.folded_coefficients(
            q, k, beta, g_cum_32, cu32, chunk_indices, float(scale), precise_folded)
    elif packed_mode == 1:
        A, pg, ag, exp, rev = flashqla_cute._packed_coefficients_tc01(
            q, k, beta, g_cum_32, cu32, chunk_indices, H, float(scale))
        packed = (pg, ag, exp, rev)
    else:
        A = flashqla_cute.kkt_solve(
            k, beta, cu32, H,
            chunk_indices if batch > 1 or valid_num_tokens < num_tokens else None)
        if packed_mode == 2:
            _, pg, ag, exp, rev = flashqla_cute._packed_coefficients_tc01(
                q, k, beta, g_cum_32, cu32, chunk_indices, H, float(scale), A)
            packed = (pg, ag, exp, rev)

    if (direct_precision or repair_all_short) and precision_inputs is not None:
        # Offline direct candidate: preserve the public gCum/A return values,
        # then compute output/state without running the BF16 chunk recurrence.
        o = torch.empty(tuple(v.shape), device=v.device, dtype=v.dtype)
        final = (initial_state if indexed_state else
                 (torch.empty((batch, H, DK, DV), device=q.device, dtype=state_dtype)
                  if output_final_state else None))
        apply_precision(o, final)
        ht = (initial_state[initial_state_indices] if indexed_state else final) if output_final_state else None
        return g_cum_32, A, o, None, ht

    cp_plan = (
        calc_cp_seqs(
            cu32, CHUNK_SIZE, H, raw_cu, num_k_heads=Hg,
            tc01_force_candidate=force_tc01_cp_candidate,
            tc01_packed_raw_gate=use_raw_gating,
            folded_prefill=use_folded,
            folded_max_local_chunks=folded_max_local_chunks or 0,
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
            # A fresh sequence starts at zero, but earlier tokens can create
            # the entire state consumed by a continuation segment. Log decay
            # alone bounds neither that state's magnitude nor relative error.
            # Preserve all prefix updates and incoming-state transitions in
            # both overlap replay and prepare/correct, even without raw h0.
            warmup_metadata = getattr(cp_plan, "warmup_metadata", None)
            if warmup_metadata is not None:
                num_warmup, fallback_mask = warmup_metadata
            else:
                segment_chunks = torch.div(
                    cp_cu_i32[1:] - cp_cu_i32[:-1] + CHUNK_SIZE - 1,
                    CHUNK_SIZE, rounding_mode="floor")
                num_warmup = segment_chunks[:, None].expand(-1, H).contiguous()
                fallback_mask = torch.ones(
                    (cp_batch, H), dtype=torch.int8, device=q.device)
            use_overlap_warmup = (
                overlap_warmup_threshold is not None and
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
                # The compact grid saves a prepare wave for this short single-
                # sequence shape on the 39-CU M890P.
                seed_first = (
                    use_folded and batch == 1 and (H, Hg) == (24, 8)
                    and 8192 <= valid_num_tokens <= 32768
                    and torch.cuda.get_device_properties(q.device).multi_processor_count == 39
                )
                seeded_h0 = None
                if seed_first:
                    seeded_h0 = torch.empty(
                        (cp_batch, H, DK, DV), dtype=torch.float32, device=q.device)
                # Seeded prepare already computes the first continuation in
                # FP32. Keep the later transitions in FP32 as well.
                selected_correct = (
                    not use_folded and precision_flags is not None and batch == 64 and H == 8 and
                    Hg in (2, 4) and _cuda_device_properties(q.device.index) == (8, 9, 39))
                prepare_args = (
                    k, v, A, g_cum_32, beta, cp_cu_i32,
                    num_warmup.reshape(-1).contiguous(), H, Hg)
                if use_folded:
                    ht_buf, mt_buf = flashqla_cute.prepare_h(
                        *prepare_args, coefficients=coefficients, precise=precise_transitions or seed_first,
                        skip_last_segment=True, seeded_cp_h0=seeded_h0,
                        initial_pool=initial_state if seed_first else None,
                        first_slot=initial_state_indices if seed_first else None)
                elif selected_correct:
                    ht_buf, mt_buf = flashqla_cute._prepare_h_selected_t1_skip(
                        *prepare_args, precision_flags, seq_map_c2r.to(torch.int32),
                        seq_map_r2c.to(torch.int32), cu32)
                else:
                    ht_buf, mt_buf = flashqla_cute.prepare_h(*prepare_args)
                ht_buf = ht_buf.view(cp_batch, H, DK, DV)
                mt_buf = mt_buf.view(cp_batch, H, DK, DK)
                fb_2d = fallback_mask.view(cp_batch, H).contiguous()
                # cp_h0 goes straight back into fused_fwd as its h0, so it is
                # produced in the state pool's own dtype.
                correct_args = (
                    ht_buf, mt_buf, fb_2d, seq_map_r2c.to(torch.int32),
                    batch, initial_state,
                    initial_state_indices if indexed_state else None,
                    indexed_state, state_dtype == torch.bfloat16)
                if use_folded:
                    cp_h0 = flashqla_cute.correct_h0(
                        *correct_args, folded=True, seeded_cp_h0=seeded_h0)
                elif selected_correct:
                    cp_h0 = flashqla_cute._correct_h0_selected_skip(*correct_args, precision_flags, Hg)
                else:
                    cp_h0 = flashqla_cute.correct_h0(*correct_args)
            cp_seq_map_i32 = seq_map_c2r.to(torch.int32)
            raw_cu_i32 = cu32

    # The folded fast path only needs the guard mask when repair reads live state.
    skip_folded_heads = use_folded and live_initial is not None
    forward_flags = precision_flags if not use_folded or skip_folded_heads else None
    # Step 4: fused forward — CP path (grid = cp_batch*H) or non-CP path.
    if cp_seq_map_i32 is not None:
        o, ht = flashqla_cute.fused_fwd(
            q, k, v, A, beta, g_cum_32, cp_cu_i32, cp_seq_map_i32, raw_cu_i32,
            valid_num_tokens, H, Hg, float(scale), cp_h0,
            initial_state if indexed_state else None,
            initial_state_indices if indexed_state else None,
            output_final_state, state_dtype == torch.bfloat16,
            raw_state_input, num_warmup_flat, fallback_mask_flat,
            precision_flags=forward_flags, coefficients=coefficients,
            skip_folded_repair_heads=skip_folded_heads)
    else:
        h0 = (initial_state if initial_state is not None and not indexed_state
              else torch.empty(0, device=q.device))
        main_a, main_beta, main_g = (A, beta, g_cum_32)
        packed_argument = ()
        if packed is not None:
            pg, ag, exp, rev = packed
            main_a, main_beta, main_g = ag, rev, exp
            packed_argument = (pg,)
        o, ht = flashqla_cute.fused_fwd(
            q, k, v, main_a, main_beta, main_g, cu32, None, cu32,
            valid_num_tokens, H, Hg, float(scale), h0,
            initial_state if indexed_state else None,
            initial_state_indices if indexed_state else None,
            output_final_state, state_dtype == torch.bfloat16,
            None, None, None, *packed_argument, precision_flags=forward_flags, coefficients=coefficients,
            skip_folded_repair_heads=skip_folded_heads)

    if precision_inputs is not None:
        apply_precision(o, initial_state if indexed_state else (ht if output_final_state else None))

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
    raw_cu_seqlens: Optional[Union[Sequence[int], torch.Tensor]] = None,
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
