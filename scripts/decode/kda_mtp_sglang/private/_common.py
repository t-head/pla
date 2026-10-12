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

"""Shared input builders for the H20 KDA ReplaySSM port.

Shapes follow the K3 dspark contract enforced by fused_kda_decode_mtp_dspark's
wrapper and _can_run_dspark_cutedsl_mtp's gate:

  head_q_dim == head_k_dim == head_v_dim == 128
  num_q_heads == num_k_heads == num_v_heads == H
  conv kernel width 4  ->  conv state window 3
  every request contributes exactly draft_token_num = 1 + num_spec tokens
  mixed_qkv / a / b are bf16; conv weights, ssm state, rings g/beta are fp32
"""

from dataclasses import dataclass, field

import torch

HEAD_DIM = 128
CONV_W = 4
CONV_WIN = CONV_W - 1


@dataclass
class Cfg:
    """K3 dspark typical: gamma=7 -> draft_token_num=8, H=12 heads."""

    bs: int = 8
    num_spec: int = 7
    heads: int = 12
    ring_len: int = 16
    pool: int = 0            # 0 -> derived from bs
    seed: int = 1234
    device: str = "cuda"
    lower_bound: float = -8.0
    onorm: bool = True
    layers: int = 1          # commit-side only: KDA layer count

    @property
    def draft_token_num(self) -> int:
        return 1 + self.num_spec

    @property
    def T(self) -> int:
        return self.bs * self.draft_token_num

    @property
    def dim(self) -> int:
        return self.heads * HEAD_DIM

    @property
    def slots(self) -> int:
        return self.pool or (self.bs + 2)


def _bf16(*shape, gen, device, scale=1.0):
    return (torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)
            * scale).to(torch.bfloat16)


def build_verify_inputs(cfg: Cfg) -> dict:
    """Tensors for one fused_kda_decode_mtp_dspark call (CACHE_RING mode)."""
    d, H, T = cfg.device, cfg.heads, cfg.T
    g_ = torch.Generator(device=d).manual_seed(cfg.seed)
    dim, slots, L = cfg.dim, cfg.slots, cfg.ring_len
    dtn = cfg.draft_token_num

    # bounded inputs: the recurrence runs silu/sigmoid/exp chains, and wild
    # magnitudes make the fp32-vs-fp64 comparison meaningless rather than wrong.
    x_q = _bf16(1, T, H, HEAD_DIM, gen=g_, device=d, scale=0.5)
    x_k = _bf16(1, T, H, HEAD_DIM, gen=g_, device=d, scale=0.5)
    x_v = _bf16(1, T, H, HEAD_DIM, gen=g_, device=d, scale=0.5)
    a = _bf16(1, T, H, HEAD_DIM, gen=g_, device=d, scale=0.5)     # g / gate pre-act
    b = _bf16(1, T, H, gen=g_, device=d, scale=1.0)               # beta pre-sigmoid

    conv_w = (torch.randn(3 * dim, CONV_W, generator=g_, device=d,
                          dtype=torch.float32) * 0.3)
    w_q, w_k, w_v = conv_w.split([dim, dim, dim], dim=0)

    # conv pool is [slots, WIN, 3*dim]; the caller splits on -1 then transposes.
    conv_states = (torch.randn(slots, CONV_WIN, 3 * dim, generator=g_, device=d,
                               dtype=torch.float32) * 0.3).to(torch.bfloat16)
    cs_q, cs_k, cs_v = (t.transpose(-1, -2).contiguous()
                        for t in conv_states.split([dim, dim, dim], dim=-1))

    A_log = torch.rand(H, generator=g_, device=d, dtype=torch.float32) * 0.5 + 0.5
    dt_bias = torch.randn(dim, generator=g_, device=d, dtype=torch.float32) * 0.1

    recurrent_state = (torch.randn(slots, H, HEAD_DIM, HEAD_DIM, generator=g_,
                                   device=d, dtype=torch.float32) * 0.1)

    ic = torch.zeros(slots, dtn, 3 * dim, CONV_WIN, device=d, dtype=torch.bfloat16)
    ic_q, ic_k, ic_v = (t.contiguous() for t in ic.split([dim, dim, dim], dim=-2))

    # slot i for request i; the kernel treats slot < 0 as CUDA-graph padding.
    ssm_state_indices = torch.arange(cfg.bs, device=d, dtype=torch.int32)
    intermediate_state_indices = torch.arange(cfg.bs, device=d, dtype=torch.int32)
    cu_seqlens = torch.arange(0, T + 1, dtn, device=d, dtype=torch.int32)

    rings = dict(
        replayssm_rawv=torch.zeros(slots, H, L, HEAD_DIM, device=d, dtype=torch.bfloat16),
        replayssm_rawk=torch.zeros(slots, H, L, HEAD_DIM, device=d, dtype=torch.bfloat16),
        replayssm_g=torch.zeros(slots, H, L, HEAD_DIM, device=d, dtype=torch.float32),
        replayssm_beta=torch.zeros(slots, H, L, device=d, dtype=torch.float32),
    )

    onorm = {}
    if cfg.onorm:
        onorm = dict(
            onorm_gate=_bf16(1, T, H, HEAD_DIM, gen=g_, device=d, scale=0.5),
            onorm_weight=(torch.rand(HEAD_DIM, generator=g_, device=d,
                                     dtype=torch.float32) * 0.5 + 0.75),
            onorm_eps=1e-6,
        )

    return dict(
        x_q=x_q, x_k=x_k, x_v=x_v,
        w_q=w_q, w_k=w_k, w_v=w_v,
        cs_q=cs_q, cs_k=cs_k, cs_v=cs_v,
        g=a, beta=b, A_log=A_log, dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        intermediate_ssm=None,                     # CACHE_RING -> not allocated
        intermediate_state_indices=intermediate_state_indices,
        intermediate_conv_q=ic_q, intermediate_conv_k=ic_k, intermediate_conv_v=ic_v,
        ssm_state_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=cfg.lower_bound,
        scale=HEAD_DIM ** -0.5,
        **rings, **onorm,
    )


def build_commit_inputs(cfg: Cfg, accept_lens=None) -> dict:
    """SpeculativeState-alike + indices for commit_kda_replayssm_after_verify."""
    d, H, L = cfg.device, cfg.heads, cfg.ring_len
    g_ = torch.Generator(device=d).manual_seed(cfg.seed + 7)
    nl, slots, dim, dtn = cfg.layers, cfg.slots, cfg.dim, cfg.draft_token_num

    temporal = (torch.randn(nl, slots, H, HEAD_DIM, HEAD_DIM, generator=g_,
                            device=d, dtype=torch.float32) * 0.1)
    rawv = _bf16(nl, slots, H, L, HEAD_DIM, gen=g_, device=d, scale=0.5)
    rawk = _bf16(nl, slots, H, L, HEAD_DIM, gen=g_, device=d, scale=0.5)
    # gk is a log-decay: strictly negative, so exp(gk) contracts the state.
    gk = -torch.rand(nl, slots, H, L, HEAD_DIM, generator=g_, device=d,
                     dtype=torch.float32) * 0.5
    beta = torch.rand(nl, slots, H, L, generator=g_, device=d, dtype=torch.float32)

    conv = [torch.randn(nl, slots, dim, CONV_WIN, generator=g_, device=d,
                        dtype=torch.bfloat16)]
    interm_conv = [torch.randn(nl, slots, dtn, dim, CONV_WIN, generator=g_,
                               device=d, dtype=torch.bfloat16)]

    if accept_lens is None:
        # spread 1..draft_token_num so the sweep covers full accept and 1-only
        accept_lens = torch.tensor(
            [(i % dtn) + 1 for i in range(cfg.bs)], device=d, dtype=torch.int32)
    state_batch_indices = torch.arange(cfg.bs, device=d, dtype=torch.int32)
    last_correct = (accept_lens.to(torch.int64) - 1)

    return dict(
        temporal=temporal, rawv=rawv, rawk=rawk, gk=gk, beta=beta,
        conv=conv, intermediate_conv_window=interm_conv,
        state_batch_indices=state_batch_indices,
        accept_lens=accept_lens, last_correct_step_indices=last_correct,
    )


@dataclass
class SpecState:
    """Duck-type of MambaPool.SpeculativeState for the commit entry point."""

    temporal: torch.Tensor
    replayssm_rawv: torch.Tensor
    replayssm_rawk: torch.Tensor
    replayssm_g: torch.Tensor
    replayssm_beta: torch.Tensor
    conv: list = field(default_factory=list)
    intermediate_conv_window: list = field(default_factory=list)
    intermediate_ssm: torch.Tensor = None

    @staticmethod
    def from_commit_inputs(ci: dict) -> "SpecState":
        return SpecState(
            temporal=ci["temporal"],
            replayssm_rawv=ci["rawv"], replayssm_rawk=ci["rawk"],
            replayssm_g=ci["gk"], replayssm_beta=ci["beta"],
            conv=ci["conv"], intermediate_conv_window=ci["intermediate_conv_window"],
        )


def gpu_banner() -> str:
    p = torch.cuda.get_device_properties(0)
    return (f"{p.name}  sm_{p.major}{p.minor}  SMs={p.multi_processor_count}  "
            f"smem/block={p.shared_memory_per_block_optin}B")
