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

"""Kernel1 (prepare) correctness test vs a self-contained torch reference.

Validates the 6 workspace outputs (k_decayed, q_decayed, k_restored, g_total,
INV, Mqk) per (tile, head) against the blueprint math (torch_ref.py chunk loop),
computed here in fp32 to avoid depending on the reference's cuBLAS/inline-CUDA
helpers. Accuracy is judged by err_ratio (bf16 storage + 2-ulp approx).
"""
import torch

CH, D = 16, 128
LOG2E = 1.4426950408889634


def err_ratio(x, y):
    x = x.float(); y = y.float()
    return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)).item()


def l2norm_rows(x):  # x [.,D] fp32
    return x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt()


def ref_prepare(q, k, g, beta, A_log, dt_bias, scale, lower_bound, T, H):
    """Return dict of [nht, ...] reference arrays (non-varlen, N=1)."""
    total_tiles = (T + CH - 1) // CH
    nht = H * total_tiles
    kd = torch.zeros(nht, CH, D); qd = torch.zeros(nht, CH, D)
    kr = torch.zeros(nht, CH, D); gt = torch.zeros(nht, D)
    inv = torch.zeros(nht, CH, CH); mqk = torch.zeros(nht, CH, CH)
    a_exp = torch.exp(A_log.float())                      # [H]
    gate_scale = lower_bound * LOG2E
    for h in range(H):
        for t in range(total_tiles):
            s = t * CH; e = min(s + CH, T); al = e - s
            qc = torch.zeros(CH, D); kc = torch.zeros(CH, D)
            gc = torch.zeros(CH, D); bc = torch.zeros(CH)
            qc[:al] = l2norm_rows(q[s:e, h].float())
            kc[:al] = l2norm_rows(k[s:e, h].float())
            gc[:al] = g[s:e, h].float()
            bc[:al] = beta[s:e, h].float()
            # gate: lower*log2e * sigmoid(exp(A_log)*(g+dt))
            gg = gc + dt_bias[h].float().unsqueeze(0)
            gg = gate_scale * torch.sigmoid(a_exp[h] * gg)
            gg[al:] = 0.0
            gcs = torch.cumsum(gg, 0)                       # cumsum
            g_tot = gcs[-1]                                 # [D]
            e_pos = torch.exp2(gcs); e_neg = torch.exp2(-gcs)
            k_dec = kc * e_pos
            q_dec = qc * e_pos * scale
            k_iv = kc * e_neg
            k_res = k_iv * torch.exp2(g_tot).unsqueeze(0)
            # zero tail rows
            k_dec[al:] = 0; q_dec[al:] = 0; k_iv[al:] = 0; k_res[al:] = 0
            L = k_dec @ k_iv.t()
            Mq = q_dec @ k_iv.t()
            bsig = torch.sigmoid(bc)
            L = torch.tril(L, -1) * bsig.unsqueeze(1)       # per-row beta[i]
            Mq = torch.tril(Mq)
            INV = torch.eye(CH) - L
            L2 = L @ L; INV = INV + INV @ L2
            L4 = L2 @ L2; INV = INV + INV @ L4
            L8 = L4 @ L4; INV = INV + INV @ L8
            idx = h * total_tiles + t
            # store as bf16-rounded to match kernel storage
            kd[idx] = k_dec.bfloat16().float(); qd[idx] = q_dec.bfloat16().float()
            kr[idx] = k_res.bfloat16().float(); gt[idx] = torch.exp2(g_tot)
            inv[idx] = INV.bfloat16().float(); mqk[idx] = Mq.bfloat16().float()
    return dict(kd=kd, qd=qd, kr=kr, gt=gt, inv=inv, mqk=mqk)


def main():
    from pla.prefill.flashkdapro.private import flashkda_prepare
    torch.manual_seed(0)
    T, H = 32, 2
    scale = 1.0 / (D ** 0.5); LOWER = -5.0
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")

    ws = flashkda_prepare(q, k, g, beta, A_log, dt_bias, scale, LOWER)
    torch.cuda.synchronize()
    kd, qd, kr, gt, inv, mqk = [w.cpu() for w in ws]

    ref = ref_prepare(q.cpu(), k.cpu(), g.cpu(), beta.cpu(),
                      A_log.cpu(), dt_bias.cpu(), scale, LOWER, T, H)
    names = ["kd", "qd", "kr", "gt", "inv", "mqk"]
    outs = {"kd": kd, "qd": qd, "kr": kr, "gt": gt, "inv": inv, "mqk": mqk}
    ok = True
    for n in names:
        r = err_ratio(outs[n].reshape(-1), ref[n].reshape(-1))
        status = "OK" if r < 3e-2 else "FAIL"
        if r >= 3e-2: ok = False
        print(f"  {n:4s} err_ratio={r:.4e}  {status}")
    assert ok, "Kernel1 prepare mismatch vs reference"
    print("KERNEL1 OK: prepare matches torch reference within err_ratio<3e-2")


if __name__ == "__main__":
    main()
