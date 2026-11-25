# SPDX-License-Identifier: Apache-2.0
"""Verify/decode: compact or cached double. Materialize also supports raw windows."""

from __future__ import annotations

from numbers import Integral, Real

import torch

from ._C import cached_verify as _cached_verify
from ._C import materialize as _materialize
from ._C import verify as _verify
from .contracts import (
    HistoryFormat,
    MaterializeWork,
    NumericPolicy,
    ReplayHistory,
    ReplayInputs,
    ReplayMapping,
    ReplayStep,
)

__all__ = [
    "HistoryFormat", "NumericPolicy", "ReplayInputs", "ReplayHistory",
    "ReplayMapping", "ReplayStep", "MaterializeWork",
    "gdn_replay_verify", "gdn_replay_decode", "gdn_replay_materialize",
    "gdn_replay_workspace_elements",
]

_NORM_POLICIES = {"none": 0, "l2_inside_sqrt": 1, "l2_outside_sqrt": 2}
_HISTORY_FORMATS = {
    HistoryFormat.COMPACT_DKG_RING: 0,
    HistoryFormat.RAW_VKG_BETA_WINDOW: 1,
    HistoryFormat.CACHED_UKG_DOUBLE: 2,
}


def _host_integer(value, name, minimum=0, maximum=2**31 - 1):
    if (not isinstance(value, Integral) or isinstance(value, bool)
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be a host integer in [{minimum}, {maximum}]")
    return int(value)


def _history_format(history, *, compact_only=False):
    fmt = HistoryFormat(history.format)
    if compact_only and fmt != HistoryFormat.COMPACT_DKG_RING:
        raise ValueError("expected COMPACT_DKG_RING history")
    residual = history.value_residual is not None
    if residual != (history.key_residual is not None):
        raise ValueError("value_residual and key_residual must be supplied together")
    if fmt == HistoryFormat.COMPACT_DKG_RING:
        if history.beta is not None:
            raise ValueError("compact history must not contain beta")
    elif fmt == HistoryFormat.RAW_VKG_BETA_WINDOW:
        if residual or history.beta is None:
            raise ValueError("raw window history requires beta and forbids residuals")
    elif residual or history.beta is not None:
        raise ValueError("cached double history forbids residuals and extra beta")
    return fmt


def _compact_capacity(history):
    _history_format(history, compact_only=True)
    if history.values.ndim != 4 or history.keys.ndim != 4 or history.gates.ndim != 3:
        raise ValueError("verify/decode history must have ranks 4, 4, 3")
    length = history.values.shape[2]
    if length <= 0 or length & (length - 1):
        raise ValueError("compact history capacity must be a positive power of two")
    return length


def _cached_capacity(history):
    if history.values.ndim != 5 or history.keys.ndim != 5 or history.gates.ndim != 4:
        raise ValueError("cached verify/decode history must have ranks 5, 5, 4")
    slots, banks, length, hv, value_dim = history.values.shape
    heads, key_dim = history.keys.shape[-2:]
    if (banks != 2 or not 1 <= length <= 16 or heads <= 0 or hv <= 0
            or hv % heads or not 1 <= key_dim <= 256 or value_dim <= 0
            or history.keys.shape != (slots, 2, length, heads, key_dim)
            or history.gates.shape != (slots, 2, hv, length)):
        raise ValueError("invalid cached double history shape or capacity (must be in [1, 16])")
    return length


def _verify_capacity(history):
    fmt = _history_format(history)
    if fmt == HistoryFormat.RAW_VKG_BETA_WINDOW:
        raise ValueError("verify/decode require compact or cached double history")
    return fmt, (_cached_capacity(history) if fmt == HistoryFormat.CACHED_UKG_DOUBLE
                 else _compact_capacity(history))


def _run_verify(inputs, state, history, mapping, step, out, workspace,
                query_start_loc, max_verify_len, numeric_policy):
    fmt, length = _verify_capacity(history)
    cached = fmt == HistoryFormat.CACHED_UKG_DOUBLE
    max_steps = _host_integer(max_verify_len, "max_verify_len", 1, 8 if cached else 32)
    if cached:
        if step.buffers is None:
            raise ValueError("cached verify/decode require ReplayStep.buffers")
        if max_steps > length:
            raise ValueError("cached max_verify_len must not exceed history capacity")
    elif step.starts is None or step.buffers is not None:
        raise ValueError("compact verify/decode require starts and forbid buffers")
    null_slot = _host_integer(mapping.null_block_id, "null_block_id", -(2**31))
    if (inputs.A_log is None) != (inputs.dt_bias is None):
        raise ValueError("A_log and dt_bias must be supplied together")
    processed = numeric_policy.gate_mode == "processed"
    if not processed and inputs.A_log is None:
        raise ValueError("raw gates require FP32 A_log and dt_bias")
    if inputs.scale is not None and (
        not isinstance(inputs.scale, Real) or isinstance(inputs.scale, bool)
    ):
        raise ValueError("scale must be a host scalar or None")
    norm = _NORM_POLICIES[numeric_policy.qk_norm]
    # Checkpoint and all metadata are read-only; acceptance belongs to the caller.
    if cached:
        # Optional prefix offsets are checked for zero on device, never read here.
        _cached_verify(
            inputs.q, inputs.k, inputs.v, inputs.gate, inputs.beta,
            inputs.A_log, inputs.dt_bias, state,
            history.values, history.keys, history.gates, out,
            query_start_loc, mapping.state_indices, mapping.replay_indices,
            step.lengths, step.starts, step.buffers, max_steps,
            inputs.scale, null_slot, norm, numeric_policy.epsilon, processed,
            numeric_policy.dot_precision == "tf32", workspace, False,
        )
        return out
    _verify(
        inputs.q, inputs.k, inputs.v, inputs.gate, inputs.beta,
        inputs.A_log, inputs.dt_bias, state,
        history.values, history.keys, history.gates,
        history.value_residual, history.key_residual, None, out,
        query_start_loc, mapping.state_indices, mapping.replay_indices,
        step.lengths, step.starts, None, length, max_steps,
        inputs.scale, norm != 0, null_slot, 1,
        numeric_policy.dot_precision == "tf32",
        workspace, False, norm, numeric_policy.epsilon, processed, False,
    )
    return out


def gdn_replay_verify(inputs: ReplayInputs, state: torch.Tensor,
                      history: ReplayHistory, mapping: ReplayMapping,
                      step: ReplayStep, out: torch.Tensor,
                      workspace: torch.Tensor | None, *,
                      query_start_loc: torch.Tensor | None, max_verify_len: int,
                      numeric_policy: NumericPolicy = NumericPolicy()):
    """Verify without accepting or allocating; only cached history permits fixed-T starts=None."""
    if query_start_loc is None and HistoryFormat(history.format) != HistoryFormat.CACHED_UKG_DOUBLE:
        raise ValueError("compact verify requires query_start_loc; use decode for one token per row")
    return _run_verify(inputs, state, history, mapping, step, out, workspace,
                       query_start_loc, max_verify_len, numeric_policy)


def gdn_replay_decode(inputs: ReplayInputs, state: torch.Tensor,
                      history: ReplayHistory, mapping: ReplayMapping,
                      step: ReplayStep, out: torch.Tensor,
                      workspace: torch.Tensor | None, *,
                      numeric_policy: NumericPolicy = NumericPolicy()):
    """One token per mapping row, with no cu_seqlens allocation."""
    return _run_verify(inputs, state, history, mapping, step, out, workspace,
                       None, 1, numeric_policy)


def gdn_replay_workspace_elements(inputs: ReplayInputs, state: torch.Tensor,
                                  history: ReplayHistory, mapping: ReplayMapping,
                                  max_verify_len: int) -> int:
    """Return FP32 workspace elements (not bytes), using tensor metadata only."""
    fmt, length = _verify_capacity(history)
    cached = fmt == HistoryFormat.CACHED_UKG_DOUBLE
    steps = _host_integer(max_verify_len, "max_verify_len", 1, 8 if cached else 32)
    if inputs.q.ndim != 3 or inputs.v.ndim != 3 or state.ndim != 4:
        raise ValueError("q/v/state must have ranks 3, 3, 4")
    if (mapping.state_indices.ndim != 1 or mapping.replay_indices.ndim != 1
            or mapping.state_indices.shape != mapping.replay_indices.shape):
        raise ValueError("state_indices and replay_indices must be matching vectors")
    if cached:
        tokens, heads, key_dim = inputs.q.shape
        hv, value_dim = history.values.shape[-2:]
        if (steps > length or inputs.k.shape != inputs.q.shape
                or inputs.v.shape != (tokens, hv, value_dim)
                or history.keys.shape[-2:] != (heads, key_dim)
                or state.shape[1:] != (hv, value_dim, key_dim)
                or inputs.gate.shape != (tokens, hv) or inputs.beta.shape != (tokens, hv)):
            raise ValueError("cached input/state/history shapes or max_verify_len do not match")
        return 0
    batch = mapping.state_indices.shape[0]
    head_work = batch * inputs.v.shape[1]
    threshold = 192 if steps == 4 else 64 if steps >= 5 else 96
    if (head_work >= threshold
            and inputs.q.shape[2] == 128 and inputs.v.shape[2] == 128
            and (steps <= 4 or steps in (6, 8)) and length <= 32 and inputs.q.dtype == torch.bfloat16
            and history.values.dtype == torch.bfloat16 and state.dtype == torch.float32):
        return batch * inputs.v.shape[1] * 4 * steps * 128
    return 0


def gdn_replay_materialize(state_layers: torch.Tensor, history: ReplayHistory,
                           work: MaterializeWork, *,
                           numeric_policy: NumericPolicy = NumericPolicy(),
                           null_block_id: int = -1,
                           state_descriptors: torch.Tensor | None = None) -> None:
    """Fold processed history into source/target states; never update metadata."""
    fmt = _history_format(history)
    null_slot = _host_integer(null_block_id, "null_block_id", -(2**31))
    indexed = (work.state_indices, work.replay_indices, work.lengths,
               work.starts, work.buffers, work.metadata_indices, work.enabled)
    if work.items is not None:
        if work.n_items is None or any(t is not None for t in indexed):
            raise ValueError("items requires n_items and forbids indexed fields")
    elif work.n_items is not None or any(t is None for t in indexed[:3]):
        raise ValueError("indexed work requires state_indices, replay_indices, and lengths")
    # Descriptor owners must stay alive; destinations must not alias other sources or history.
    _materialize(
        state_layers, history.values, history.keys, history.gates,
        history.value_residual, history.key_residual, history.beta,
        work.state_indices, work.replay_indices, work.lengths, work.starts,
        work.buffers, work.metadata_indices, work.target_state_indices,
        work.enabled, work.items, work.n_items, state_descriptors,
        _HISTORY_FORMATS[fmt], _NORM_POLICIES[numeric_policy.qk_norm],
        numeric_policy.epsilon, null_slot,
    )
