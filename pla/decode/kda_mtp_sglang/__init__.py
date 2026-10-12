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

"""``pla.decode.kda_mtp_sglang`` — SGLang KDA MTP verify / commit entry points.

Ports of the two Kimi-K3 KDA speculative-decoding kernels from SGLang, drop-in
against the upstream Python signatures so a backend can swap by importing this
namespace instead:

``fused_kda_decode_mtp_dspark``
    Verify path — matches ``sglang.kernels.ops.kimi_k3.kda_decode_mtp
    .fused_kda_decode_mtp_dspark``. CACHE_RING mode only (all four
    ``replayssm_*`` rings are required, ``intermediate_ssm`` snapshots are
    skipped, no fused output RMSNorm). The persistent recurrent / conv states
    are read-only; per-step raw inputs go to the four ring caches (consumed by
    the commit-time exact fold).

``commit_kda_replayssm_after_verify``
    Commit path — matches ``sglang.kernels.ops.attention.fla
    .kda_replayssm_spec_decode.commit_kda_replayssm_after_verify``. Replays each
    request's accepted prefix into the fp32 checkpoint (in place, one launch for
    all layers) and rolls back the conv state to the last accepted step.

Both are pure CUDA-core delta-rule recurrences (register-resident state,
warp-shuffle reductions); see ``src/decode/kda_mtp_sglang/`` for the device
bodies and ``ppu_compat.cuh`` for the PPU intrinsic shim.
"""

from typing import Optional

# MUST come before importing the extension: it links against libtorch/libc10
# with no RPATH, so those libraries have to be loaded into the process first.
# Without this the import dies with
#     ImportError: libc10.so: cannot open shared object file
# on any host whose LD_LIBRARY_PATH does not include torch/lib.
import torch  # noqa: F401

from ._C import verify_dspark as _verify_dspark
from ._C import fold_all_layers as _fold_all_layers
from ._C import conv_window_scatter as _conv_window_scatter

__all__ = [
    "fused_kda_decode_mtp_dspark",
    "commit_kda_replayssm_after_verify",
]

TILE_K = 128


def fused_kda_decode_mtp_dspark(
    *,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    intermediate_ssm: Optional[torch.Tensor] = None,
    intermediate_state_indices: torch.Tensor,
    intermediate_conv_q: torch.Tensor,
    intermediate_conv_k: torch.Tensor,
    intermediate_conv_v: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float,
    scale: Optional[float] = None,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    onorm_gate: Optional[torch.Tensor] = None,
    onorm_weight: Optional[torch.Tensor] = None,
    onorm_eps: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run Kimi-K3 KDA verify (DSpARK rollback), CACHE_RING mode.

    Keyword-only, matching sglang's entry point. Returns the attention output
    ``out`` (allocated if not supplied). All four ``replayssm_*`` rings must be
    passed; ``onorm_*`` is not implemented in this port.

    Q/K/V and gate inputs may have a token stride (for split mixed-QKV views),
    with contiguous head/channel dimensions. Conv states ``[slot, H*128, 3]``
    and windows ``[scratch, step, H*128, 3]`` accept strided views, including
    sglang's split/transposed conv pool and dense scratch. State, rings and
    output keep their contiguous layouts.
    """
    rings = (replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta)
    if any(r is None for r in rings):
        raise ValueError(
            "pla.decode.kda_mtp_sglang.fused_kda_decode_mtp_dspark implements "
            "CACHE_RING only; pass all four replayssm_* rings.")
    if onorm_gate is not None or onorm_weight is not None or onorm_eps is not None:
        raise NotImplementedError(
            "fused output RMSNorm (onorm_*) is not implemented in the PPU port.")

    if scale is None:
        scale = TILE_K ** -0.5
    if out is None:
        out = torch.empty_like(x_v, memory_format=torch.contiguous_format)

    _verify_dspark(
        x_q, x_k, x_v,
        g, beta,
        w_q.contiguous(),
        w_k.contiguous(),
        w_v.contiguous(),
        cs_q, cs_k, cs_v,
        A_log.contiguous(),
        dt_bias.contiguous(),
        recurrent_state, out,
        intermediate_conv_q, intermediate_conv_k, intermediate_conv_v,
        replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta,
        ssm_state_indices, intermediate_state_indices,
        cu_seqlens, float(scale), float(lower_bound),
    )
    return out


def commit_kda_replayssm_after_verify(
    *,
    spec_state,
    state_batch_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_steps_to_track: Optional[torch.Tensor] = None,
    null_block_id: int = -1,
    conv_state_layout: str = "dim_width",
) -> None:
    """Fold each layer's accepted window into ``spec_state.temporal`` and roll
    back conv.

    Keyword-only, matching sglang's entry point. ``spec_state`` is a
    ``MambaPool.SpeculativeState``-like duck type carrying ``temporal``,
    ``replayssm_rawv/rawk/g/beta``, ``conv``, and ``intermediate_conv_window``.
    ``conv_state_layout="width_dim"`` consumes SGLang's conv pools
    ``[layers, slots, width, dim]`` and scratch directly through transpose views.
    The default preserves the original PLA ``[..., dim, width]`` contract.
    """
    assert conv_state_layout in ("dim_width", "width_dim")
    assert len(spec_state.conv) == len(spec_state.intermediate_conv_window)
    num_k_heads = spec_state.replayssm_rawk.shape[2]
    _fold_all_layers(
        spec_state.temporal,
        spec_state.replayssm_rawv,
        spec_state.replayssm_rawk,
        spec_state.replayssm_g,
        spec_state.replayssm_beta,
        state_batch_indices,
        accept_lens,
        mamba_track_indices,
        mamba_steps_to_track,
        num_k_heads,
        null_block_id,
    )
    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        if conv_state_layout == "width_dim":
            conv_states = conv_states.transpose(-1, -2)
            interm_conv = interm_conv.transpose(-1, -2)
        _conv_window_scatter(
            conv_states, interm_conv, state_batch_indices,
            last_correct_step_indices)
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            _conv_window_scatter(
                conv_states, interm_conv, mamba_track_indices,
                mamba_steps_to_track)
