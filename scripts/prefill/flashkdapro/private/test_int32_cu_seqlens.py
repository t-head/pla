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

"""Regression for FlashKDA's serving-native int32 cu_seqlens path."""

import torch

from pla.prefill.flashkdapro import flashkda_fwd


def main():
    torch.manual_seed(20260826)
    lengths = [100, 273, 16, 272]
    real_tokens, physical_tokens, heads, dim = sum(lengths), 668, 12, 128
    q = torch.randn(physical_tokens, heads, dim, device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(physical_tokens, heads, device="cuda",
                       dtype=torch.bfloat16)
    a_log = torch.rand(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(heads, dim, device="cuda", dtype=torch.float32)
    h0 = torch.randn(len(lengths), heads, dim, dim, device="cuda",
                     dtype=torch.bfloat16)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu32 = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    cu64 = cu32.to(torch.int64)

    def run(cu, store_final, raw_cu_seqlens=None):
        out = torch.full_like(q, 7)
        final = (torch.empty_like(h0) if store_final else None)
        flashkda_fwd(q, k, v, g, beta, a_log, dt_bias, out,
                     initial_state=h0, final_state=final, cu_seqlens=cu,
                     raw_cu_seqlens=raw_cu_seqlens)
        return out, final

    out32, final32 = run(cu32, True)
    out32_trusted, final32_trusted = run(cu32, True, tuple(offsets))
    out64, final64 = run(cu64, True)
    out32_nostate, _ = run(cu32, False)

    # The same packed batch without physical suffix padding exercises the
    # trusted Host fast path where endpoint == T_total: no zero-tail kernel is
    # launched and K1/K2 write the complete output directly.
    exact_out = torch.full_like(v[:real_tokens], 19)
    exact_final = torch.empty_like(h0)
    flashkda_fwd(
        q[:real_tokens],
        k[:real_tokens],
        v[:real_tokens],
        g[:real_tokens],
        beta[:real_tokens],
        a_log,
        dt_bias,
        exact_out,
        initial_state=h0,
        final_state=exact_final,
        cu_seqlens=cu32,
        raw_cu_seqlens=tuple(offsets),
    )
    torch.cuda.synchronize()

    assert torch.equal(out32, out64), "int32/int64 output mismatch"
    assert torch.equal(out32_trusted, out32), "trusted padding output mismatch"
    assert torch.equal(final32, final64), "int32/int64 final-state mismatch"
    assert torch.equal(final32_trusted, final32), "trusted padding state mismatch"
    assert torch.equal(out32, out32_nostate), "final-state store changed output"
    assert torch.count_nonzero(out32[real_tokens:]).item() == 0
    assert torch.count_nonzero(out32_trusted[real_tokens:]).item() == 0
    assert torch.equal(exact_out, out32[:real_tokens]), "exact fast-path mismatch"
    assert torch.equal(exact_final, final32), "exact fast-path state mismatch"

    overflow = torch.tensor(
        [0, torch.iinfo(torch.int32).max + 1], device="cuda", dtype=torch.int64)
    try:
        run(overflow, False)
    except ValueError as exc:
        assert "int32 range" in str(exc)
    else:
        raise AssertionError("int64 endpoint overflow was not rejected")

    print("INT32 CU_SEQLENS: output/state bit-equal to int64; device-check and "
          "trusted padding tails zero; trusted exact fast path matches; "
          "overflow rejected")


if __name__ == "__main__":
    main()
