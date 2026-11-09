# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Native, request-row KDA replay contracts for a vLLM-style runner.

This namespace has no vLLM, Triton or CuTe runtime dependency. Verify and commit
must use the same replay records; RecoverSSM correction records are incompatible.
"""
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import torch

from . import _C

__all__ = ["KDAReplayRecords", "KDAReplayCommitPlan", "KDAReplayCommitContext",
           "kda_replay_verify"]


@dataclass(frozen=True)
class KDAReplayRecords:
    rawv: torch.Tensor
    rawk: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    conv: torch.Tensor

    @classmethod
    def allocate(cls, max_num_reqs, num_heads, spec_query_len, device):
        if max_num_reqs < 0 or num_heads < 1 or not 1 <= spec_query_len <= 32:
            raise ValueError("records require R >= 0, H > 0 and Q in [1, 32]")
        shape = (max_num_reqs, num_heads, spec_query_len, 128)
        return cls(torch.empty(shape, dtype=torch.bfloat16, device=device),
                   torch.empty(shape, dtype=torch.bfloat16, device=device),
                   torch.empty(shape, dtype=torch.float32, device=device),
                   torch.empty(shape[:-1], dtype=torch.float32, device=device),
                   torch.empty((max_num_reqs, spec_query_len, 3 * num_heads * 128, 3),
                               dtype=torch.bfloat16, device=device))


def kda_replay_verify(*, mixed_qkv, raw_g, raw_beta, conv_weight, conv_state,
                      checkpoint_state, A_log, dt_bias, lower_bound, records,
                      query_start_loc, state_indices, conv_state_layout,
                      out: Optional[torch.Tensor] = None, scale=None,
                      null_block_id=0):
    """Read a paged checkpoint and write only real query intervals and records.

    Output normalization is external. Q/K/V and gates may be projection views
    with a positive token stride and nonzero storage offset. Persistent state
    and conv history remain read-only. Conv history beyond its first three items
    is neither read nor written.

    Native replay uses FP32 state arithmetic and BF16 outputs. Severe numerical
    cancellation is not repaired by default; finite-input zero gate products
    remain protected. Build with PLA_KDA_NATIVE_FP64_REPAIR=1 to enable the
    diagnostic conditioning checks and FP64 repair path. This is a build option,
    not a runtime switch or a uniform relative-error guarantee for arbitrary
    states and parameters.
    """
    if not isinstance(lower_bound, Real) or (scale is not None and not isinstance(scale, Real)):
        raise TypeError("scale and lower_bound must be host numeric scalars")
    if not isinstance(null_block_id, Integral):
        raise TypeError("null_block_id must be a host integer")
    return _C.verify(mixed_qkv, raw_g, raw_beta, conv_weight, conv_state, checkpoint_state,
                     A_log, dt_bias, out, records.rawv, records.rawk, records.g,
                     records.beta, records.conv, query_start_loc, state_indices,
                     conv_state_layout, 128 ** -0.5 if scale is None else float(scale),
                     float(lower_bound), null_block_id)


@dataclass(frozen=True)
class KDAReplayCommitPlan:
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor


def _intersecting_runs(left, left_bytes, right, right_bytes):
    """Yield intersecting pairs of byte runs from two strided sequences."""
    a, b = left.start, right.start
    while a < left.stop and b < right.stop:
        a_end, b_end = a + left_bytes, b + right_bytes
        if a_end <= b:
            a += ((b - a_end) // left.step + 1) * left.step
        elif b_end <= a:
            b += ((a - b_end) // right.step + 1) * right.step
        else:
            yield a, b
            if a_end <= b_end:
                a += left.step
            else:
                b += right.step


@dataclass(frozen=True)
class _CacheWrites:
    pages: range
    offsets: range
    run_bytes: int

    @classmethod
    def from_tensor(cls, tensor, run_size, run_count=1, run_stride=1):
        item_bytes = tensor.element_size()
        start = tensor.data_ptr()
        page_stride = tensor.stride(0) * item_bytes
        pages = range(start, start + tensor.size(0) * page_stride, page_stride)
        if run_stride == run_size:
            return cls(pages, range(1), run_size * run_count * item_bytes)
        stride_bytes = run_stride * item_bytes
        return cls(pages, range(0, run_count * stride_bytes, stride_bytes),
                   run_size * item_bytes)

    def overlaps(self, other):
        left, right = self, other
        if left.pages.start > right.pages.start:
            left, right = right, left
        a_pages, b_pages = left.pages, right.pages
        if a_pages.step == b_pages.step:
            # All later page pairs repeat the relative offsets of the first
            # right page and its at most two left neighbors.
            first = (b_pages.start - a_pages.start) // a_pages.step
            a_pages, b_pages = a_pages[first:first + 2], b_pages[:1]
        a_extent = left.offsets[-1] + left.run_bytes
        b_extent = right.offsets[-1] + right.run_bytes
        for a, b in _intersecting_runs(a_pages, a_extent, b_pages, b_extent):
            a_runs = range(a, a + left.offsets.stop, left.offsets.step)
            b_runs = range(b, b + right.offsets.stop, right.offsets.step)
            if next(_intersecting_runs(a_runs, left.run_bytes, b_runs,
                                       right.run_bytes), None) is not None:
                return True
        return False


def _cache_writes(conv, state, layout):
    # The native descriptor has validated both tensors and their page strides.
    state_writes = _CacheWrites.from_tensor(state, state.numel() // state.size(0))
    conv_ds = conv if layout == "DS" else conv.transpose(1, 2)
    channels = conv_ds.size(1)
    # Scatter writes only the first three history items of each channel.
    if conv_ds.stride(2) == 1:
        conv_writes = _CacheWrites.from_tensor(conv_ds, 3, channels, conv_ds.stride(1))
    else:
        conv_writes = _CacheWrites.from_tensor(conv_ds, channels, 3, conv_ds.stride(2))
    return state_writes, conv_writes


class KDAReplayCommitContext:
    """Own immutable layer pointers and reusable GPU plan buffers.

    Create outside capture and keep cache and record tensors in place for the
    context lifetime. Writable cache regions of different layers must be disjoint.

    Serialize commits, reads of the returned plan, and replays of graphs
    captured with this context on one stream or through stream event dependencies.
    The next commit or replay overwrites the plan. Use separate contexts and
    nonconflicting cache accesses for concurrent work.

    The plan has max_num_reqs entries, with zeroes in inactive capacity entries.
    """
    @classmethod
    def create(cls, layers, spec_query_len, max_num_reqs,
               conv_state_layout="SD", null_block_id=0):
        layers = tuple(layers)
        if not layers or not 1 <= spec_query_len <= 32 or max_num_reqs < 0:
            raise ValueError("context requires layers, R >= 0 and Q in [1, 32]")
        if null_block_id not in (-1, 0):
            raise ValueError("null_block_id must be -1 or 0")
        device = layers[0].kv_cache[1].device
        if device.type != "cuda":
            raise ValueError("commit context requires CUDA/PPU tensors")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("create commit context before graph capture")
        slots, heads = layers[0].kv_cache[1].shape[:2]
        descriptors = []
        keepalive = []
        layer_writes = []
        for index, layer in enumerate(layers):
            conv, state = layer.kv_cache
            records = layer.replay_records
            tensors = (conv, state, records.rawv, records.rawk, records.g,
                       records.beta, records.conv)
            descriptors.append(_C.layer_descriptor(*tensors, conv_state_layout,
                                                   heads, slots, spec_query_len,
                                                   max_num_reqs))
            if any(t.device != device for t in tensors):
                raise ValueError("all context layers must share one device")
            writes = _cache_writes(conv, state, conv_state_layout)
            for previous, other_writes in enumerate(layer_writes):
                if any(a.overlaps(b) for a in writes for b in other_writes):
                    raise ValueError(
                        f"context layers {previous} and {index} have overlapping writable caches")
            layer_writes.append(writes)
            keepalive.extend(tensors)
        result = cls()
        result._keepalive = tuple(keepalive)
        result._layers = layers
        result._table = torch.tensor(descriptors, device=device, dtype=torch.int64)
        result._candidate = torch.empty((4, max_num_reqs), device=device, dtype=torch.int32)
        result._buffer = torch.empty_like(result._candidate)
        result.plan = KDAReplayCommitPlan(*result._buffer.unbind(0))
        result._heads, result._slots = heads, slots
        result._q, result._capacity = spec_query_len, max_num_reqs
        result._null = null_block_id
        return result

    def commit(self, num_sampled, state_indices, query_start_loc,
               request_indices=None, block_table=None, num_computed_tokens=None,
               mamba_block_size=None, *, token_capacity):
        """Commit clamped sampled-token counts, without subtracting a bonus token.

        Pass the verify projection's ``mixed_qkv.shape[0]`` as token_capacity.
        The plan skips query intervals that exceed this capacity.
        Align mode requires all three align arguments and block_size >= Q.
        Different active requests may not overlap source/destination slots.
        Invalid device metadata is diagnosed by PLA_SANITY_CHECK in eager mode;
        the graph path safely masks invalid plans without a tensor-value read.
        """
        if not isinstance(token_capacity, Integral):
            raise TypeError("token_capacity must be a host integer")
        align = (block_table is not None, num_computed_tokens is not None,
                 mamba_block_size is not None)
        if any(align) and not all(align):
            raise ValueError("align requires block_table, computed tokens and block size")
        if all(align) and not isinstance(mamba_block_size, Integral):
            raise TypeError("mamba_block_size must be a host integer")
        if all(align) and mamba_block_size < self._q:
            raise ValueError("align block_size must be >= spec_query_len")
        _C.commit(self._table, self._candidate, self._buffer, num_sampled,
                  state_indices, query_start_loc, request_indices, block_table,
                  num_computed_tokens, 0 if not all(align) else mamba_block_size,
                  self._heads, self._slots, self._q, self._null, token_capacity)
        return self.plan
