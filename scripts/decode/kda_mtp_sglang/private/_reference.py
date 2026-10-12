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

"""fp64 reference implementations -- the correctness oracle for the H20 port.

Both references were derived by reading the kernels line by line, not from a
paper, so the op order matches the kernels' intent exactly. Everything runs in
float64 so the fp32 kernels can be scored against a materially more accurate
answer instead of against each other.

Layout note that trips people up: the fold kernel's `b_h` tile is [BK, BV]
(K rows, V cols) while `temporal` is [..., HV, V, K]. Element (k, v) of the tile
lives at offset v*K + k, i.e. b_h[k, v] == temporal[hv, v, k]. Both references
below carry the state as [V, K] to match `temporal`, and the gate therefore
broadcasts along K.

The shared recurrence, per head, per token:

    h[v, k] *= exp(gk[k])                 # per-K log-decay
    shk[v]   = sum_k h[v, k] * k[k]
    vn[v]    = (v[v] - shk[v]) * beta
    h[v, k] += vn[v] * k[k]               # rank-1 update
    o[v]     = sum_k h[v, k] * q[k]       # read AFTER the update
"""

import math

import torch

HEAD_DIM = 128
CONV_W = 4
CONV_WIN = CONV_W - 1
L2_EPS = 1e-6


def _silu(x):
    return x * torch.sigmoid(x)


def _causal_conv_silu(x_seq, cs, w):
    """Width-4 causal conv + silu over [cs(3) || x], per channel.

    x_seq [T, C] / cs [C, 3] / w [C, 4]  ->  ([T, C] post-silu,
                                              [T, C, 3] post-advance windows)

    The kernel keeps a 3-slot sliding register window seeded from the conv
    state, folds in x_t with the last tap, then shifts. Window snapshot for
    step t is the window AFTER consuming x_t, which is what the conv rollback
    restores.
    """
    T, C = x_seq.shape
    full = torch.cat([cs.transpose(0, 1), x_seq], dim=0)     # [3 + T, C]
    out = torch.empty(T, C, dtype=x_seq.dtype, device=x_seq.device)
    win = torch.empty(T, C, CONV_WIN, dtype=x_seq.dtype, device=x_seq.device)
    for t in range(T):
        acc = torch.zeros(C, dtype=x_seq.dtype, device=x_seq.device)
        for j in range(CONV_W):
            acc = acc + full[t + j] * w[:, j]
        out[t] = _silu(acc)
        for j in range(CONV_WIN):
            win[t, :, j] = full[t + 1 + j]
    return out, win


def verify_reference(vi: dict, cfg) -> dict:
    """Recompute fused_kda_decode_mtp_dspark in fp64.

    Returns the output plus the four ring records and the conv-window snapshots,
    so a single call scores every side effect the kernel is contracted to make.
    """
    dev = vi["x_q"].device
    H, T = cfg.heads, cfg.T
    dtn, L = cfg.draft_token_num, cfg.ring_len
    f64 = dict(dtype=torch.float64, device=dev)

    x_q = vi["x_q"][0].to(torch.float64)          # [T, H, D]
    x_k = vi["x_k"][0].to(torch.float64)
    x_v = vi["x_v"][0].to(torch.float64)
    g_raw = vi["g"][0].to(torch.float64)          # [T, H, D]
    b_raw = vi["beta"][0].to(torch.float64)       # [T, H]
    A_log = vi["A_log"].to(torch.float64)         # [H]
    dt_bias = vi["dt_bias"].to(torch.float64)     # [H*D]
    scale = float(vi["scale"])
    lower_bound = float(vi["lower_bound"])
    cu = vi["cu_seqlens"].tolist()
    slots = vi["ssm_state_indices"].tolist()
    scratch = vi["intermediate_state_indices"].tolist()

    out = torch.zeros(1, T, H, HEAD_DIM, **f64)
    ring_rawv = torch.zeros(cfg.slots, H, L, HEAD_DIM, **f64)
    ring_rawk = torch.zeros(cfg.slots, H, L, HEAD_DIM, **f64)
    ring_gk = torch.zeros(cfg.slots, H, L, HEAD_DIM, **f64)
    ring_beta = torch.zeros(cfg.slots, H, L, **f64)
    ic_q = torch.zeros(cfg.slots, dtn, H * HEAD_DIM, CONV_WIN, **f64)
    ic_k = torch.zeros_like(ic_q)
    ic_v = torch.zeros_like(ic_q)

    for n in range(cfg.bs):
        slot, srow = slots[n], scratch[n]
        if slot < 0:
            continue
        bos, eos = cu[n], cu[n + 1]
        n_tok = eos - bos
        for h in range(H):
            off = h * HEAD_DIM
            sl = slice(off, off + HEAD_DIM)
            wq = vi["w_q"][sl].to(torch.float64)
            wk = vi["w_k"][sl].to(torch.float64)
            wv = vi["w_v"][sl].to(torch.float64)
            csq = vi["cs_q"][slot, sl].to(torch.float64)
            csk = vi["cs_k"][slot, sl].to(torch.float64)
            csv = vi["cs_v"][slot, sl].to(torch.float64)

            q_c, q_w = _causal_conv_silu(x_q[bos:eos, h], csq, wq)
            k_c, k_w = _causal_conv_silu(x_k[bos:eos, h], csk, wk)
            v_c, v_w = _causal_conv_silu(x_v[bos:eos, h], csv, wv)

            ic_q[srow, :n_tok, sl] = q_w
            ic_k[srow, :n_tok, sl] = k_w
            ic_v[srow, :n_tok, sl] = v_w
            ring_rawk[slot, h, :n_tok] = k_c      # post-conv, PRE-L2norm
            ring_rawv[slot, h, :n_tok] = v_c

            # q/k L2 norm; q additionally folds in `scale`
            q_n = q_c / torch.sqrt((q_c * q_c).sum(-1, keepdim=True) + L2_EPS) * scale
            k_n = k_c / torch.sqrt((k_c * k_c).sum(-1, keepdim=True) + L2_EPS)

            beta = torch.sigmoid(b_raw[bos:eos, h])
            ring_beta[slot, h, :n_tok] = beta

            exp_A = torch.exp(A_log[h])
            gk = lower_bound * torch.sigmoid(exp_A * (g_raw[bos:eos, h] + dt_bias[sl]))
            ring_gk[slot, h, :n_tok] = gk
            decay = torch.exp(gk)                                # [T, D]

            hs = vi["recurrent_state"][slot, h].to(torch.float64)  # [V, K]
            for t in range(n_tok):
                hs = hs * decay[t][None, :]
                shk = (hs * k_n[t][None, :]).sum(-1)             # [V]
                vn = (v_c[t] - shk) * beta[t]
                hs = hs + vn[:, None] * k_n[t][None, :]
                out[0, bos + t, h] = (hs * q_n[t][None, :]).sum(-1)

    if "onorm_gate" in vi:
        gate = torch.sigmoid(vi["onorm_gate"][0].to(torch.float64))
        w = vi["onorm_weight"].to(torch.float64)
        eps = float(vi["onorm_eps"])
        rms = torch.rsqrt((out[0] * out[0]).mean(-1, keepdim=True) + eps)
        out = (out[0] * rms * w[None, None, :] * gate).unsqueeze(0)

    return dict(out=out, ring_rawv=ring_rawv, ring_rawk=ring_rawk,
                ring_g=ring_gk, ring_beta=ring_beta,
                ic_q=ic_q, ic_k=ic_k, ic_v=ic_v)


def fold_reference(ci: dict, cfg) -> torch.Tensor:
    """Recompute the exact-fold (commit) in fp64. Returns the folded `temporal`.

    Mirrors kda_replayssm_exact_fold_kernel: division-form L2 norm with eps
    inside the sqrt, per-K gate decay, then decay -> delta -> rank-1.
    """
    temporal = ci["temporal"].to(torch.float64).clone()
    rawv, rawk = ci["rawv"].to(torch.float64), ci["rawk"].to(torch.float64)
    gkc, betac = ci["gk"].to(torch.float64), ci["beta"].to(torch.float64)
    idx = ci["state_batch_indices"].tolist()
    acc = ci["accept_lens"].tolist()
    nl, _, HV = temporal.shape[0], temporal.shape[1], temporal.shape[2]
    H = rawk.shape[2]
    grp = HV // H                                  # v-heads per k-head

    for l in range(nl):
        for n, slot in enumerate(idx):
            if slot < 0 or acc[n] <= 0:
                continue
            for hv in range(HV):
                h = hv // grp
                hs = temporal[l, slot, hv]         # [V, K]
                for t in range(acc[n]):
                    kk = rawk[l, slot, h, t]
                    vv = rawv[l, slot, hv, t].clone()
                    gg = gkc[l, slot, hv, t]
                    bb = betac[l, slot, hv, t]
                    kk = kk / torch.sqrt((kk * kk).sum() + L2_EPS)
                    hs = hs * torch.exp(gg)[None, :]
                    vv = vv - (hs * kk[None, :]).sum(-1)
                    vv = vv * bb
                    hs = hs + vv[:, None] * kk[None, :]
                temporal[l, slot, hv] = hs
    return temporal


def score(got: torch.Tensor, want: torch.Tensor, name: str) -> dict:
    """Relative-error report against the fp64 oracle."""
    g, w = got.double(), want.double()
    denom = w.abs().max().clamp_min(1e-30)
    abs_err = (g - w).abs()
    rel_l2 = ((g - w).norm() / w.norm().clamp_min(1e-30)).item()
    return dict(name=name, max_abs=abs_err.max().item(),
                max_rel=(abs_err.max() / denom).item(), rel_l2=rel_l2,
                finite=bool(torch.isfinite(g).all()))


def fmt(s: dict) -> str:
    return (f"{s['name']:<22} max_abs={s['max_abs']:.3e}  "
            f"max_rel={s['max_rel']:.3e}  rel_l2={s['rel_l2']:.3e}  "
            f"finite={s['finite']}")
