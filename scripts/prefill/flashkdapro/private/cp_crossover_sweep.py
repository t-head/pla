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

"""CP-vs-serial crossover sweep over the K3 small-batch / low-head / long-T space.

For every shape it reports three numbers:

    serial   PLA_FLASHKDA_CP=0, the plain single-chain recurrence
    auto     the shipped dispatch (PLA_FLASHKDA_CP=1); "off" when the gate
             declines to engage, in which case it is identical to serial
    best     the best of a forced segment-count sweep (PLA_FLASHKDA_CP=2)

`best` is the oracle the dispatch gate is tuned against: wherever
best/serial > 1 and auto == serial the gate is leaving speed on the table, and
wherever auto < serial it is engaging where it should not.

The gate regime matters because the pre_scan transition chain has an exact
early exit on a zero state slice (see kernel2_recurrence_nows.cu): with a fast
decay M_i underflows within a few chunks and pre_scan costs a fraction of a
full pass, with a slow decay it never underflows and pre_scan pays in full.
Real traffic sits between the two, so both are swept and reported as bounds.

Timing rule on PPU: no device events around a single kernel; every number is
the wall clock of whole API calls with one synchronise per batch.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from cp_stage_timing import D, LOWER, DECAYS, inputs, timeit, set_env


def cu_seqlens(T, N, force=False):
    # A single sequence normally runs the fixed-shape path (cu_seqlens=None).
    # `force` builds the 2-entry table instead, which matters because the plain
    # K2 dispatch picks BV32 for varlen but BV16 -- plus the H in {6,12} DK-split
    # leader -- for fixed shapes, so the serial baseline is not the same kernel.
    if N <= 1 and not force:
        return None
    # Equal split; T is the token total so the per-sequence length is T/N.
    assert T % N == 0, (T, N)
    return torch.arange(0, T + 1, T // N, dtype=torch.int64, device="cuda")


def sweep_shape(_C, T, H, N, state, decay, mults, bvs, iters, force_cu=False):
    x = inputs(T, H, state, N, decay)
    cu = cu_seqlens(T, N, force_cu)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"], x["A_log"],
            x["dt_bias"], 1.0 / (D ** 0.5), LOWER)
    # out/final_state are caller-allocated out-params; args[:7] are the tensors and
    # args[7:] the (scale, lower_bound) pair the binding takes after `out`.
    nseq = 1 if cu is None else cu.numel() - 1
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device="cuda")
    fin = torch.empty(nseq, H, D, D, dtype=torch.bfloat16, device="cuda")
    call = lambda: _C.flashkda_fwd(*args[:7], out, *args[7:], x["h0"], fin, cu)

    set_env(CP=0, CP_BV=None, CP_MULT=None, CP_SEG_TILES=None)
    t_serial = timeit(call, iters)

    # Shipped dispatch.  A plan that declines to engage runs the serial path,
    # so it is reported as t_serial rather than re-timed.
    set_env(CP=1, CP_BV=None, CP_MULT=None)
    auto_plan = _C.flashkda_cp_plan(T, H, cu)
    if auto_plan["engage"]:
        t_auto = timeit(call, iters)
        auto_seg = auto_plan["n_seg"]
    else:
        t_auto, auto_seg = t_serial, 0

    best = None
    for bv, mult in itertools.product(bvs, mults):
        set_env(CP=2, CP_BV=bv, CP_MULT=mult or None, CP_SEG_TILES=None)
        if not _C.flashkda_cp_plan(T, H, cu)["engage"]:
            continue
        t = timeit(call, iters)
        if best is None or t < best[0]:
            best = (t, bv, mult, _C.flashkda_cp_plan(T, H, cu)["n_seg"])
    set_env(CP=1, CP_BV=None, CP_MULT=None)

    return dict(T=T, H=H, N=N, state=int(state), decay=decay,
                varlen=int(cu is not None),
                serial=t_serial, auto=t_auto, auto_seg=auto_seg,
                best=best[0] if best else t_serial,
                best_bv=best[1] if best else 0,
                best_mult=best[2] if best else 0,
                best_seg=best[3] if best else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=str, default="12,24")
    ap.add_argument("--tokens", type=str, default="4096,8192,16384,32768")
    ap.add_argument("--seqs", type=str, default="1,2,4")
    ap.add_argument("--decays", type=str, default="fast,slow")
    ap.add_argument("--states", type=str, default="0,1")
    ap.add_argument("--mults", type=str, default="0,4,8,16,24,32")
    ap.add_argument("--bvs", type=str, default="16,32")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--force-cu", action="store_true",
                    help="drive single-sequence shapes through the varlen path")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    from pla.prefill.flashkdapro import _C

    heads = [int(s) for s in a.heads.split(",")]
    tokens = [int(s) for s in a.tokens.split(",")]
    seqs = [int(s) for s in a.seqs.split(",")]
    decays = [s for s in a.decays.split(",")]
    states = [bool(int(s)) for s in a.states.split(",")]
    mults = [int(s) for s in a.mults.split(",")]
    bvs = [int(s) for s in a.bvs.split(",")]

    print(f"{'T':>6} {'H':>3} {'N':>2} {'vl':>2} {'st':>2} {'decay':>5} "
          f"{'serial':>8} {'auto':>8} {'best':>8} "
          f"{'auto/x':>7} {'best/x':>7}  cfg", flush=True)
    rows = []
    for decay, H, T, N, st in itertools.product(decays, heads, tokens, seqs,
                                                states):
        if T % N:
            continue
        r = sweep_shape(_C, T, H, N, st, decay, mults, bvs, a.iters,
                        a.force_cu)
        rows.append(r)
        print(f"{T:>6} {H:>3} {N:>2} {r['varlen']:>2} {r['state']:>2} "
              f"{decay:>5} "
              f"{r['serial']:>8.3f} {r['auto']:>8.3f} {r['best']:>8.3f} "
              f"{r['serial']/r['auto']:>6.2f}x {r['serial']/r['best']:>6.2f}x  "
              f"auto_seg={r['auto_seg']} "
              f"best=bv{r['best_bv']}/mult{r['best_mult']}/seg{r['best_seg']}",
              flush=True)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
