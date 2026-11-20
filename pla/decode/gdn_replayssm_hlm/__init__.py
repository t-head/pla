# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM cached replay boundary; ordinary stream ordering only, no active PDL."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Optional

import torch

from ..gdn_replayssm._C import cached_verify as _cached_verify
from ..gdn_replayssm.contracts import (
    HistoryFormat,
    MaterializeWork,
    NumericPolicy,
    ReplayHistory,
    ReplayInputs,
    ReplayMapping,
    ReplayStep,
)

__all__ = [
    "fused_recurrent_gated_delta_rule_cached_replay_update",
    "commit_gdn_cached_replay_history_layers",
]

_FLOATING = (torch.float16, torch.bfloat16, torch.float32)


def _integer(value, name, minimum=1, maximum=2**31 - 1):
    if (not isinstance(value, Integral) or isinstance(value, bool)
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be a host integer in [{minimum}, {maximum}]")
    return int(value)


def _boolean(value, name, optional=False):
    if not isinstance(value, bool) and not (optional and value is None):
        raise ValueError(f"{name} must be a bool" + (" or None" if optional else ""))


def _tensor(tensor, name, rank, device, dtypes, *, contiguous=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (not tensor.is_cuda or tensor.device != device or tensor.layout != torch.strided
            or tensor.ndim != rank or tensor.is_conj() or tensor.is_neg()):
        raise ValueError(f"{name} must be a rank-{rank} strided tensor on {device}")
    if tensor.dtype not in dtypes:
        raise ValueError(f"{name} has an unsupported dtype")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous; history/state are never copied")


def _shape(tensor, expected, name):
    if tensor.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")


def _inner(tensor, dimensions, name):
    expected = 1
    for axis in range(tensor.ndim - 1, tensor.ndim - dimensions - 1, -1):
        if tensor.size(axis) > 1 and tensor.stride(axis) != expected:
            raise ValueError(f"{name} must have contiguous inner dimensions")
        expected *= tensor.size(axis)


def _hints(value_dim, block_v, num_warps, stages, *, prefix=""):
    # These validate source-compatible hints, not this backend's launch geometry.
    if block_v is not None:
        tile = _integer(block_v, prefix + "block_v")
        if tile & (tile - 1) or tile > value_dim:
            raise ValueError(prefix + "block_v must be a power of two <= V")
    if num_warps is not None:
        warps = _integer(num_warps, prefix + "num_warps")
        if warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(prefix + "num_warps must be a power of two in [1, 32]")
    if stages is not None:
        _integer(stages, prefix + "pipeline stages")


def _work(items, count, device, rows=None):
    if (items is None) != (count is None):
        raise ValueError("replay_work_items and n_writes must either both be set or both be None")
    if items is None:
        return
    _tensor(items, "replay_work_items", 2, device, (torch.int32,), contiguous=True)
    if items.shape[1] != 4 or (rows is not None and items.shape[0] != rows):
        raise ValueError("replay_work_items must have shape (N, 4)")
    if not isinstance(count, torch.Tensor) or count.ndim > 1 or count.numel() != 1:
        raise ValueError("n_writes must be a device scalar or one-element vector")
    _tensor(count, "n_writes", count.ndim, device, (torch.int32,))


def _readonly_work(items, count, writes):
    # The main replay does not consume work items. Still prevent its writes from
    # corrupting these caller-owned descriptors before the later layered commit.
    def interval(tensor):
        start = tensor.data_ptr()
        span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        return start, start + span * tensor.element_size()

    for read in (items, count):
        if read is None or read.numel() == 0:
            continue
        begin, end = interval(read)
        for write in writes:
            if write.numel() == 0:
                continue
            other_begin, other_end = interval(write)
            if begin < other_end and other_begin < end:
                raise ValueError("replay work metadata must not overlap writable tensors")


def _history(old_u, old_k, old_G, device, history_size, hv, value_dim, key_dim,
             *, layers=None, input_dtype=None):
    layered = layers is not None
    vector_rank, gate_rank = (6, 5) if layered else (5, 4)
    types = _FLOATING if input_dtype is None else (input_dtype,)
    _tensor(old_u, "old_u", vector_rank, device, types, contiguous=True)
    _tensor(old_k, "old_k", vector_rank, device, (old_u.dtype,), contiguous=True)
    _tensor(old_G, "old_G", gate_rank, device, (torch.float32,), contiguous=True)
    slots = old_u.shape[1 if layered else 0]
    heads = _integer(old_k.shape[-2], "H")
    if hv % heads:
        raise ValueError("HV must be divisible by H")
    prefix = (layers, slots) if layered else (slots,)
    _shape(old_u, (*prefix, 2, history_size, hv, value_dim), "old_u")
    _shape(old_k, (*prefix, 2, history_size, heads, key_dim), "old_k")
    _shape(old_G, (*prefix, 2, hv, history_size), "old_G")
    return ReplayHistory(HistoryFormat.CACHED_UKG_DOUBLE, old_u, old_k, old_G)


def fused_recurrent_gated_delta_rule_cached_replay_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    state_indices: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    old_beta: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    history_size: int,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = False,
    replay_work_items: Optional[torch.Tensor] = None,
    n_writes: Optional[torch.Tensor] = None,
    block_v: Optional[int] = None,
    num_warps: Optional[int] = None,
    use_l2_state_cache: Optional[bool] = None,
    use_l2_streaming_inputs: Optional[bool] = None,
    use_l2_shared_inputs: Optional[bool] = None,
    main_num_stages: Optional[int] = None,
    packed_qkv: Optional[torch.Tensor] = None,
    use_all_layer_commit: bool = False,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replay cached inputs; only inline mode folds old history, never caller-owned metadata."""
    from ..gdn_replayssm import gdn_replay_decode, gdn_replay_verify

    del old_beta  # Signature-compatible placeholder; cached updates embed beta.
    _boolean(launch_with_pdl, "launch_with_pdl")
    _boolean(use_qk_l2norm_in_kernel, "use_qk_l2norm_in_kernel")
    _boolean(use_all_layer_commit, "use_all_layer_commit")
    for name, hint in (("use_l2_state_cache", use_l2_state_cache),
                       ("use_l2_streaming_inputs", use_l2_streaming_inputs),
                       ("use_l2_shared_inputs", use_l2_shared_inputs)):
        _boolean(hint, name, optional=True)
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a tensor")
    device = q.device
    _tensor(q, "q", 4, device, _FLOATING)
    if launch_with_pdl and torch.cuda.get_device_capability(device)[0] >= 9:
        raise NotImplementedError("active PDL producers require a supported grid-dependency wait")
    _tensor(k, "k", 4, device, (q.dtype,))
    _tensor(v, "v", 4, device, (q.dtype,))
    n, steps, heads, key_dim = k.shape
    steps = _integer(steps, "T", maximum=8)
    heads = _integer(heads, "H")
    key_dim = _integer(key_dim, "K", maximum=256)
    if key_dim & (key_dim - 1):
        raise ValueError("TRT-LLM cached replay K must be a power of two")
    hv, value_dim = v.shape[-2:]
    _integer(hv, "HV")
    _integer(value_dim, "V")
    if hv % heads:
        raise ValueError("HV must be divisible by H")
    _shape(q, (n, steps, heads, key_dim), "q")
    _shape(v, (n, steps, hv, value_dim), "v")
    history_size = _integer(history_size, "history_size", steps, 16)
    _hints(value_dim, block_v, num_warps, main_num_stages)
    if scale is not None and (not isinstance(scale, Real) or isinstance(scale, bool)
                              or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38):
        raise ValueError("scale must be a finite float32 host scalar or None")
    _tensor(ssm_states, "ssm_states", 4, device, (q.dtype, torch.float32))
    _shape(ssm_states, (ssm_states.shape[0], hv, value_dim, key_dim), "ssm_states")
    _inner(ssm_states, 3, "ssm_states")
    history = _history(old_u, old_k, old_G, device, history_size, hv, value_dim,
                       key_dim, input_dtype=q.dtype)
    if old_k.shape[-2] != heads:
        raise ValueError("old_k heads must match q/k")
    _tensor(state_indices, "state_indices", 1, device, (torch.int32, torch.int64))
    _shape(state_indices, (n,), "state_indices")
    for name, metadata in (("cache_buf_idx", cache_buf_idx),
                           ("prev_num_accepted_tokens", prev_num_accepted_tokens)):
        _tensor(metadata, name, 1, device, (torch.int32, torch.int64))
        if metadata.shape[0] < old_u.shape[0]:
            raise ValueError(f"{name} must cover every replay slot")
    for name, gate in (("g", g), ("beta", beta)):
        _tensor(gate, name, 3, device, _FLOATING)
        _shape(gate, (n, steps, hv), name)
        if gate.stride(2) != 1 or gate.stride(0) != steps * gate.stride(1):
            raise ValueError(f"{name} requires contiguous heads and uniform token-row strides")
    if A_log is not None:
        for name, parameter in (("A_log", A_log), ("dt_bias", dt_bias)):
            _tensor(parameter, name, 1, device, (torch.float32,), contiguous=True)
            _shape(parameter, (hv,), name)
    else:
        dt_bias = None  # As in the target, this parameter is unused without A_log.
    _work(replay_work_items, n_writes, device, rows=n)
    if (replay_work_items is not None) != use_all_layer_commit:
        raise ValueError("partitioned replay requires work items and use_all_layer_commit=True together")
    if output is not None:
        _tensor(output, "output", 4, device, (q.dtype,), contiguous=True)
        _shape(output, (n, steps, hv, value_dim), "output")
    if packed_qkv is not None:
        width = 2 * heads * key_dim + hv * value_dim
        _tensor(packed_qkv, "packed_qkv", 2, device, (q.dtype,), contiguous=True)
        _shape(packed_qkv, (n * steps, width), "packed_qkv")
        q_flat = packed_qkv[:, :heads * key_dim].view(n * steps, heads, key_dim)
        k_flat = packed_qkv[:, heads * key_dim:2 * heads * key_dim].view(n * steps, heads, key_dim)
        v_flat = packed_qkv[:, 2 * heads * key_dim:].view(n * steps, hv, value_dim)
    else:
        q_flat = q.contiguous().view(n * steps, heads, key_dim)
        k_flat = k.contiguous().view(n * steps, heads, key_dim)
        v_flat = v.contiguous().view(n * steps, hv, value_dim)
    if output is None:
        output = q.new_empty((n, steps, hv, value_dim))
    out = output.view(n * steps, hv, value_dim)
    _readonly_work(replay_work_items, n_writes, (old_u, old_k, old_G, output))
    inputs = ReplayInputs(q_flat, k_flat, v_flat, g.view(n * steps, hv),
                          beta.view(n * steps, hv), A_log, dt_bias, scale)
    policy = NumericPolicy(qk_norm="l2_outside_sqrt" if use_qk_l2norm_in_kernel else "none",
                           epsilon=1e-6, gate_mode="processed" if A_log is None else "raw",
                           dot_precision="tf32")
    mapping = ReplayMapping(state_indices, state_indices, null_block_id=-1)
    step = ReplayStep(prev_num_accepted_tokens, buffers=cache_buf_idx)
    if use_all_layer_commit:
        # Work values are deliberately not read by the main replay. The caller's
        # later layered commit consumes the device-resident list/count directly.
        if steps == 1:
            gdn_replay_decode(inputs, ssm_states, history, mapping, step, out, None,
                              numeric_policy=policy)
        else:
            gdn_replay_verify(inputs, ssm_states, history, mapping, step, out, None,
                              query_start_loc=None, max_verify_len=steps, numeric_policy=policy)
    else:
        # One validated host call checks aliases for BOTH launches before either
        # writes. The second launch folds only the old overflowing buffer.
        _cached_verify(
            inputs.q, inputs.k, inputs.v, inputs.gate, inputs.beta,
            inputs.A_log, inputs.dt_bias, ssm_states, old_u, old_k, old_G, out,
            None, state_indices, state_indices, prev_num_accepted_tokens,
            None, cache_buf_idx, steps, scale, -1,
            2 if use_qk_l2norm_in_kernel else 0, 1e-6, A_log is None,
            True, None, True,
        )
    return output


def commit_gdn_cached_replay_history_layers(
    *,
    ssm_states: torch.Tensor,
    ssm_state_descriptors: Optional[torch.Tensor] = None,
    ssm_state_num_layers: Optional[int] = None,
    ssm_state_layer_stride: Optional[int] = None,
    ssm_state_slot_stride: Optional[int] = None,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    replay_work_items: torch.Tensor,
    n_writes: torch.Tensor,
    history_size: int,
    persistent_waves: int = 2,
    commit_block_v: Optional[int] = None,
    commit_num_warps: Optional[int] = None,
    commit_pipeline_stages: Optional[int] = None,
) -> None:
    """Materialize old buffers across layers; caller retains descriptor pointees through completion."""
    from ..gdn_replayssm import gdn_replay_materialize

    if not isinstance(ssm_states, torch.Tensor):
        raise TypeError("ssm_states must be a tensor")
    device = ssm_states.device
    stride_args = (ssm_state_num_layers, ssm_state_layer_stride, ssm_state_slot_stride)
    explicit_strides = any(value is not None for value in stride_args)
    if ssm_state_descriptors is not None and explicit_strides:
        raise ValueError("state descriptors and explicit state strides are mutually exclusive")
    indirect = ssm_state_descriptors is not None or explicit_strides
    _tensor(ssm_states, "ssm_states", 4 if indirect else 5, device, _FLOATING)
    _inner(ssm_states, 3, "ssm_states")
    hv, value_dim, key_dim = ssm_states.shape[-3:]
    _integer(hv, "HV")
    _integer(value_dim, "V")
    key_dim = _integer(key_dim, "K", maximum=256)
    if key_dim & (key_dim - 1):
        raise ValueError("TRT-LLM cached replay K must be a power of two")
    history_size = _integer(history_size, "history_size", maximum=16)
    _integer(persistent_waves, "persistent_waves")
    _hints(value_dim, commit_block_v, commit_num_warps, commit_pipeline_stages, prefix="commit_")
    state_layers = ssm_states
    if ssm_state_descriptors is not None:
        _tensor(ssm_state_descriptors, "ssm_state_descriptors", 2, device,
                (torch.int64,), contiguous=True)
        layers = ssm_state_descriptors.shape[0]
        _shape(ssm_state_descriptors, (layers, 2), "ssm_state_descriptors")
    elif explicit_strides:
        if any(value is None for value in stride_args):
            raise ValueError("all three explicit state stride arguments must be supplied")
        layers = _integer(ssm_state_num_layers, "ssm_state_num_layers")
        layer_stride = _integer(ssm_state_layer_stride, "ssm_state_layer_stride", 0, 2**63 - 1)
        slot_stride = _integer(ssm_state_slot_stride, "ssm_state_slot_stride", 1, 2**63 - 1)
        if layers > 1 and layer_stride == 0:
            raise ValueError("multiple layers require a positive layer stride")
        # as_strided checks the complete view against the backing storage bounds;
        # the native materializer additionally rejects overlapping writable axes.
        state_layers = torch.as_strided(
            ssm_states, (layers, ssm_states.shape[0], hv, value_dim, key_dim),
            (layer_stride, slot_stride, *ssm_states.stride()[-3:]),
        )
    else:
        layers = ssm_states.shape[0]
    history = _history(old_u, old_k, old_G, device, history_size, hv, value_dim,
                       key_dim, layers=layers)
    if replay_work_items is None or n_writes is None:
        raise ValueError("layered commit requires replay_work_items and n_writes")
    _work(replay_work_items, n_writes, device)
    if layers == 0:
        return
    gdn_replay_materialize(
        state_layers, history, MaterializeWork(items=replay_work_items, n_items=n_writes),
        numeric_policy=NumericPolicy(qk_norm="none", gate_mode="processed", dot_precision="tf32"),
        null_block_id=-1, state_descriptors=ssm_state_descriptors,
    )
