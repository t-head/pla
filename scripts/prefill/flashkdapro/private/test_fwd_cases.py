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

"""Broader correctness cases for flashkda_fwd vs torch reference."""
import os
import sys
# test_fwd_e2e lives next to this script, not in the installed package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from test_fwd_e2e import ref_fwd, err_ratio, D, TOL


def run(T, H, use_state, seed=0):
    from pla.prefill.flashkdapro import flashkda_fwd
    torch.manual_seed(seed)
    scale = 1.0 / (D ** 0.5); LOWER = -5.0
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    if use_state:
        h0 = torch.randn(1, H, D, D, dtype=torch.bfloat16, device="cuda") * 0.1
    else:
        h0 = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda")

    out = torch.empty(T, H, D, dtype=torch.bfloat16, device="cuda")
    fin = torch.empty(1, H, D, D, dtype=torch.bfloat16, device="cuda")
    flashkda_fwd(
        q, k, v, g, beta, A_log, dt_bias, out, scale, LOWER,
        h0 if use_state else None, fin)
    torch.cuda.synchronize()
    rout, rfin = ref_fwd(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(),
                         A_log.cpu(), dt_bias.cpu(), scale, LOWER, h0[0].cpu(), T, H)
    ro = err_ratio(out.cpu().reshape(-1), rout.reshape(-1))
    rf = err_ratio(fin[0].cpu().reshape(-1), rfin.reshape(-1))
    tag = f"T={T:<5d} H={H:<3d} state={int(use_state)}"
    ok = ro < TOL and rf < TOL
    print(f"  {tag} | out={ro:.3e} state={rf:.3e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    cases = [
        (256, 4, True), (128, 32, False), (100, 2, True),
        (512, 8, True), (2048, 16, False),
    ]
    allok = all(run(*c) for c in cases)
    assert allok, "some case failed"
    print("ALL CASES OK")


if __name__ == "__main__":
    main()
