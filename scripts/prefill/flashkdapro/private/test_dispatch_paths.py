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

"""Correctness sweep across every dispatch path flashkda_fwd can take.

One case per path, so a dispatch-table change cannot silently skip a kernel:
  dvsplit BV16 (fixed b1 H<=24), dvsplit BV32 (fixed b1 24<H<=64),
  full-DV plain nows (fixed b1 H>64), varlen plain nows (packed b>=8),
  baseline DVT32 (ragged b6 h6, i.e. N*H<40).
Compares pla.prefill.flashkdapro.flashkda_fwd against SGLang's vendored
FLA-Triton chunk_kda baseline with the repository's BF16 tolerances, so the
SGLang Python tree must be importable.

For the same path coverage without the FLA dependency, use the top-level
``e2e_acc_full.py``, which checks against the fp32 torch reference instead.
"""
import math, os, sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # test_fwd_e2e (torch reference)
sys.path.insert(0, os.path.dirname(_HERE))      # bench_vs_fla (FLA wrappers)
import bench_vs_fla as B
from pla.prefill.flashkdapro import flashkda_fwd

D = 128
scale = 1.0 / math.sqrt(D)
LOWER = -5.0
ATOL = 2e-2
RTOL = 1e-2


def mk(T, H):
    g = torch.Generator(device="cuda").manual_seed(1234 + T * 131 + H)
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    gg = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    beta = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", generator=g)
    a = torch.rand(H, dtype=torch.float32, device="cuda", generator=g)
    dt = torch.rand(H, D, dtype=torch.float32, device="cuda", generator=g)
    return q, k, v, gg, beta, a, dt


def fixed_case(tag, H, S, path):
    q, k, v, gg, beta, a, dt = mk(S, H)
    o = torch.empty(S, H, D, dtype=torch.bfloat16, device="cuda")
    s = torch.empty(1, H, D, D, dtype=torch.bfloat16, device="cuda")
    flashkda_fwd(q, k, v, gg, beta, a, dt, o, scale, LOWER, None, s)
    fo, fs = B.run_fla_chunk_kda(q[None], k[None], v[None], gg[None], beta[None],
                                 a, dt, scale, None)
    torch.cuda.synchronize()
    eo = B.err_ratio(o.cpu().reshape(-1), fo[0].cpu().reshape(-1))
    es = B.err_ratio(s[0].cpu().reshape(-1), fs.cpu().reshape(-1))
    torch.testing.assert_close(o, fo[0], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        s, fs, atol=ATOL, rtol=RTOL, check_dtype=False
    )
    ok = True
    print(f"  [{path:16s}] {tag:22s} out={eo:.3e} state={es:.3e} {'OK' if ok else 'FAIL'}")
    assert ok
    return ok


def varlen_case(tag, H, seqlens, path):
    T = sum(seqlens)
    q, k, v, gg, beta, a, dt = mk(T, H)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
                      dtype=torch.int64, device="cuda")
    o = torch.empty(T, H, D, dtype=torch.bfloat16, device="cuda")
    s = torch.empty(len(seqlens), H, D, D, dtype=torch.bfloat16, device="cuda")
    flashkda_fwd(q, k, v, gg, beta, a, dt, o, scale, LOWER, None, s, cu)
    torch.cuda.synchronize()
    # FLA per-sequence reference
    fos = []
    off = 0
    for L in seqlens:
        fo, _ = B.run_fla_chunk_kda(
            q[off:off + L][None], k[off:off + L][None], v[off:off + L][None],
            gg[off:off + L][None], beta[off:off + L][None], a, dt, scale, None)
        fos.append(fo[0]); off += L
    fo = torch.cat(fos, 0)
    eo = B.err_ratio(o.cpu().reshape(-1), fo.cpu().reshape(-1))
    torch.testing.assert_close(o, fo, atol=ATOL, rtol=RTOL)
    ok = True
    print(f"  [{path:16s}] {tag:22s} out={eo:.3e} {'OK' if ok else 'FAIL'}")
    assert ok
    return ok


def main():
    print(
        "Post-cleanup E2E dispatch-path correctness "
        f"(PPU vs SGLang FLA, atol={ATOL}, rtol={RTOL}):"
    )
    fixed_case("fixed-b1-h6-s8192", 6, 8192, "dvsplit BV16")
    fixed_case("fixed-b1-h24-s4096", 24, 4096, "dvsplit BV16")
    fixed_case("fixed-b1-h48-s4096", 48, 4096, "dvsplit BV32")
    fixed_case("fixed-b1-h96-s2048", 96, 2048, "full-DV nows")
    varlen_case("equal-b8-h6-s512", 6, [512] * 8, "varlen nows")
    varlen_case("equal-b32-h12-s1024", 12, [1024] * 32, "varlen nows")
    varlen_case("ragged-b6-h6", 6, [1300, 547, 2048, 963, 271, 3063], "baseline DVT32")
    varlen_case("ragged-b6-h12", 12, [1300, 547, 2048, 963, 271, 3063], "varlen nows")
    print("ALL DISPATCH PATHS OK")


if __name__ == "__main__":
    main()
