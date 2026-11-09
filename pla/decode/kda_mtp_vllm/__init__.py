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


class KDAReplayCommitContext:
    """Own immutable layer pointers and reusable GPU plan buffers.

    Create outside capture. Cache/record tensors must not be resized or replaced
    during the context lifetime. The returned plan is overwritten by the next
    commit and has max_num_reqs entries; inactive capacity entries are zero.
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
        for layer in layers:
            conv, state = layer.kv_cache
            records = layer.replay_records
            tensors = (conv, state, records.rawv, records.rawk, records.g,
                       records.beta, records.conv)
            descriptors.append(_C.layer_descriptor(*tensors, conv_state_layout,
                                                   heads, slots, spec_query_len,
                                                   max_num_reqs))
            if any(t.device != device for t in tensors):
                raise ValueError("all context layers must share one device")
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
               mamba_block_size=None):
        """Commit clamped sampled-token counts, without subtracting a bonus token.

        Align mode requires all three align arguments and block_size >= Q.
        Different active requests may not overlap source/destination slots.
        Invalid device metadata is diagnosed by PLA_SANITY_CHECK in eager mode;
        the graph path safely masks invalid plans without a tensor-value read.
        """
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
                  self._heads, self._slots, self._q, self._null)
        return self.plan
