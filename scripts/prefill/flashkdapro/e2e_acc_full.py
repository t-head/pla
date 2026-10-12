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

"""FlashKDA prefill end-to-end accuracy: every dispatch path, vs a fp32 torch reference.

This is the accuracy gate for `pla.prefill.flashkdapro`. It is self-contained —
the oracle is the fp32 FlashKDA torch reference in `private/test_fwd_e2e.py`
(`ref_fwd`, dtype/rounding semantics matching upstream `torch_ref.py`), so it
needs nothing beyond torch and the built extension.

Coverage is organised around what actually changes the kernel that runs, since
`flashkda_fwd` picks its K2 variant from the shape alone:

  dvsplit BV16     single sequence, H <= 24
  dvsplit BV32     single sequence, 24 < H <= 64
  full-DV nows     single sequence, H > 64
  varlen nows      packed multi-sequence (equal and ragged lengths)
  baseline DVT32   very small grid (N*H < 40)

plus the orthogonal axes: a partial (non-multiple-of-16) chunk tail, and the
state out-param contract -- zero vs bf16 vs fp32 initial state, a bf16 or fp32
caller-allocated ``final_state``, and ``final_state=None`` (K2 must skip the
state store and still produce an identical output). Varlen cases are checked
against the reference run per sequence and concatenated, which is exactly the
semantics the packed kernel must reproduce.

Run (from the repo root, as a plain script so `import pla` still resolves to the
installed package):

    CUDA_VISIBLE_DEVICES=4 python3 scripts/prefill/flashkdapro/e2e_acc_full.py
"""
import math
import os
import sys

# The fp32 reference lives next to this script, under private/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "private"))

import torch

from pla.prefill.flashkdapro import flashkda_fwd
from test_fwd_e2e import D, TOL, err_ratio, ref_fwd

LOWER = -5.0
SCALE = 1.0 / math.sqrt(D)


def make_inputs(T, H, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device="cuda", generator=g)
    return (
        mk(T, H, D),                                                    # q
        mk(T, H, D),                                                    # k
        mk(T, H, D),                                                    # v
        mk(T, H, D),                                                    # g
        mk(T, H),                                                       # beta
        torch.rand(H, dtype=torch.float32, device="cuda", generator=g),  # A_log
        torch.rand(H, D, dtype=torch.float32, device="cuda", generator=g),  # dt_bias
    )


def _dt(dtype):
    if dtype is None:
        return "none"
    return "f32" if dtype == torch.float32 else "bf16"


def check(tag, path, H, seq_lens, h0_dtype, final_dtypes, seed=20260806,
          pad_tokens=0):
    """Run one case against the fp32 reference; returns a list of pass flags.

    ``h0_dtype`` is ``None`` for a zero initial state, otherwise the
    ``initial_state`` dtype. ``final_dtypes`` lists the caller-allocated
    ``final_state`` dtypes to exercise -- once an initial state is present
    upstream requires the two to match, so that is a single entry. The
    ``final_state=None`` path is always checked as well.
    """
    real_tokens = sum(seq_lens)
    T = real_tokens + pad_tokens
    q, k, v, g, beta, A_log, dt_bias = make_inputs(T, H, seed)
    n_seq = len(seq_lens)
    varlen = n_seq > 1 or pad_tokens > 0

    if h0_dtype is None:
        h0 = None
    else:
        gen = torch.Generator(device="cuda").manual_seed(seed + 1)
        h0 = torch.randn(n_seq, H, D, D, dtype=h0_dtype, device="cuda",
                         generator=gen) * 0.1

    cu = None
    if varlen:
        cu = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(),
                          dtype=torch.int64, device="cuda")

    def run(final_dtype):
        """Allocate the out-params, run, and hand them back."""
        # A nonzero sentinel makes an omitted padded-tail store observable.
        o = torch.full(
            (T, H, D), 7.0, dtype=torch.bfloat16, device="cuda")
        fs = (None if final_dtype is None else
              torch.empty(n_seq, H, D, D, dtype=final_dtype, device="cuda"))
        flashkda_fwd(q, k, v, g, beta, A_log, dt_bias, o,
                     SCALE, LOWER, h0, fs, cu)
        return o, fs

    runs = [(dtype,) + run(dtype) for dtype in final_dtypes]
    out_nostate, _ = run(None)
    torch.cuda.synchronize()

    # Reference: one sequence at a time, then concatenate along the token axis.
    zero_h0 = torch.zeros(H, D, D, dtype=torch.bfloat16)
    ref_outs, ref_states, off = [], [], 0
    for i, L in enumerate(seq_lens):
        sl = slice(off, off + L)
        ro, rs = ref_fwd(q[sl].cpu(), k[sl].cpu(), v[sl].cpu(), g[sl].cpu(),
                         beta[sl].cpu(), A_log.cpu(), dt_bias.cpu(), SCALE, LOWER,
                         h0[i].cpu() if h0 is not None else zero_h0, L, H)
        ref_outs.append(ro)
        ref_states.append(rs)
        off += L
    ref_out = torch.cat(ref_outs, 0).reshape(-1)
    ref_state = torch.stack(ref_states, 0).reshape(-1)

    flags = []
    for final_dtype, out, final in runs:
        e_out = err_ratio(out[:real_tokens].cpu().reshape(-1), ref_out)
        e_st = err_ratio(final.cpu().reshape(-1), ref_state)
        dtype_ok = final.dtype == final_dtype
        tail_zero = (
            pad_tokens == 0 or
            torch.count_nonzero(out[real_tokens:]).item() == 0)
        ok = e_out < TOL and e_st < TOL and dtype_ok and tail_zero
        flags.append(ok)
        note = "" if dtype_ok else f" DTYPE={final.dtype}!={final_dtype}"
        print(f"  [{path:14s}] {tag:26s} {_dt(h0_dtype):4s}->{_dt(final_dtype):4s} "
              f"out={e_out:.3e} state={e_st:.3e} "
              f"pad={pad_tokens} tail_zero={tail_zero} "
              f"{'OK' if ok else 'FAIL'}{note}")

    # final_state=None must skip the state store without perturbing the output.
    nostate_tail_zero = (
        pad_tokens == 0 or
        torch.count_nonzero(out_nostate[real_tokens:]).item() == 0)
    ok = torch.equal(out_nostate, runs[0][1]) and nostate_tail_zero
    flags.append(ok)
    print(f"  [{path:14s}] {tag:26s} {_dt(h0_dtype):4s}->none "
          f"out bit-identical={ok} {'OK' if ok else 'FAIL'}")
    return flags


# (tag, dispatch path, H, seq_lens)
#
# Which K2 variant runs is decided by N, H and the dvsplit T>=256 threshold, not
# by long sequences, so these use the shortest lengths that still select each
# kernel (the fp32 reference is a per-chunk Python loop — keeping T small is what
# keeps this gate quick):
#   dvsplit   N==1 && !varlen && T_total>=256 && H<=64   (BV16 if H<=24 else BV32)
#   full-DV   N==1 && H>64, and any launch dvsplit declines
#   baseline  N*H < 40  (shared-memory column split)
#   varlen    N>1 && N*H >= 40
CASES = [
    # single sequence -> register-resident value-split
    ("b1-h6-s512",           "dvsplit BV16",   6,  [512]),
    ("b1-h24-s256",          "dvsplit BV16",   24, [256]),
    ("b1-h48-s256",          "dvsplit BV32",   48, [256]),
    ("b1-h96-s256",          "full-DV nows",   96, [256]),
    # partial chunk tail (T % 16 != 0) on the split and full-DV paths
    ("b1-h12-s264-tail",     "dvsplit BV16",   12, [264]),
    ("b1-h96-s100-tail",     "full-DV nows",   96, [100]),
    # packed multi-sequence
    ("b8-h12-s128-equal",    "varlen nows",    12, [128] * 8),
    ("b4-h12-ragged-tail",   "varlen nows",    12, [100, 273, 16, 272]),
    # short high-head varlen BV32 promotion.  H96/N3 switches only when an
    # initial state is present; H48/N2 switches in both state modes, so these
    # two cases exercise both the promoted path and its full-DV fallback.
    ("b3-h96-ragged-short",   "short high-H",    96, [48, 64, 80]),
    ("b2-h48-ragged-short",   "short high-H",    48, [97, 121]),
    # N*H < 40 -> shared-memory baseline (column split)
    ("b6-h6-ragged-baseline", "baseline DVT32", 6, [100, 150, 80, 64, 48, 32]),
    ("b2-h4-s256-baseline",   "baseline DVT32", 4, [256, 256]),
]


def main():
    print(f"FlashKDA prefill E2E accuracy vs fp32 torch reference (tol<{TOL:g}):")
    print("  columns: [K2 path] case  initial_state->final_state  errors")
    results = []
    for tag, path, H, seq_lens in CASES:
        # Zero initial state: final_state may be either dtype, since upstream
        # allows an fp32 final state to be filled from a zero initial state.
        results += check(tag, path, H, seq_lens, None,
                         (torch.bfloat16, torch.float32))
        # With an initial state the two dtypes must match, so one each.
        results += check(tag, path, H, seq_lens, torch.bfloat16, (torch.bfloat16,))
        results += check(tag, path, H, seq_lens, torch.float32, (torch.float32,))
    padded_cases = [
        # Same one-token physical alignment that exposed the QLA integration
        # bug, now exercising FlashKDA's caller-owned output buffer.
        ("b1-h12-s2205-pad1", "padded varlen", 12, [2205], 1),
        ("b4-h12-ragged-pad7", "padded varlen", 12,
         [100, 273, 16, 272], 7),
    ]
    for tag, path, H, seq_lens, pad_tokens in padded_cases:
        results += check(tag, path, H, seq_lens, None,
                         (torch.bfloat16, torch.float32),
                         pad_tokens=pad_tokens)
        results += check(tag, path, H, seq_lens, torch.bfloat16,
                         (torch.bfloat16,), pad_tokens=pad_tokens)
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} checks OK")
    assert n_ok == len(results), f"{len(results) - n_ok} check(s) failed"
    print("E2E ACCURACY OK")


if __name__ == "__main__":
    main()
