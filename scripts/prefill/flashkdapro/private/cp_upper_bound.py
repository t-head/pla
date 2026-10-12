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

"""CP profitability upper bound probe (zero implementation cost).

Context-parallel (intracard-CP) for KDA prefill works by cutting each sequence's
tile chain into S segments that run concurrently, at the cost of a pre_scan pass
(the affine transition M_seg / bias B_seg per segment) plus a serial merge.
cuLA's SM90 planner charges that re-run at RERUN_RATIO=3 tiles-equivalent.

Before writing the pre_scan/merge kernels we can measure the *ideal* speedup
CP could ever deliver on PPU1.5 for free: feed the very same total token count
to the existing kernel as S independent varlen sequences.  That reproduces
exactly the parallelism CP creates (S x more concurrent (seq,head) chains, each
1/S as long) with none of the pre_scan cost and no carry dependency.

  ideal_speedup(S) = t_serial(B seqs of T) / t_varlen(B*S seqs of T/S)

Interpretation:
  * ideal_speedup(S) <= ~1.0  -> the machine is already full; CP can only lose.
  * ideal_speedup(S) >= ~3    -> CP has headroom even at cuLA's pessimistic
    RERUN_RATIO=3 accounting.
  * in between  -> CP profitable only if its measured re-run factor is below
    the ideal speedup.

The token tensors depend only on (B*T, H), so they are allocated once per
(H, B, T) and only ``cu_seqlens`` changes with S -- the S sweep is therefore
allocation-free and directly comparable.

Run inside the PPU1.5 container, from the repo root:

  CUDA_VISIBLE_DEVICES=4 python3 -u \\
    scripts/prefill/flashkdapro/private/cp_upper_bound.py --json /tmp/cp_ub.json
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from pla.prefill.flashkdapro import flashkda_fwd

D = 128
CHUNK = 16
LOWER_BOUND = -5.0
DEV = "cuda"


def make_tokens(T_total, H, nseq_max, with_state, seed=0):
    """Allocate the token tensors once; they do not depend on the split."""
    torch.manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    return dict(
        q=rnd(T_total, H, D), k=rnd(T_total, H, D), v=rnd(T_total, H, D),
        g=rnd(T_total, H, D), beta=rnd(T_total, H),
        A_log=torch.randn(H, device=DEV, dtype=torch.float32),
        dt_bias=torch.randn(H, D, device=DEV, dtype=torch.float32),
        state_pool=rnd(nseq_max, H, D, D) if with_state else None,
    )


def timed(pool, seq_lens, with_state, warmup, iters, repeats):
    nseq = len(seq_lens)
    cu = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(),
                      dtype=torch.int64, device=DEV)
    kw = dict(q=pool["q"], k=pool["k"], v=pool["v"], g=pool["g"], beta=pool["beta"],
              A_log=pool["A_log"], dt_bias=pool["dt_bias"], cu_seqlens=cu,
              lower_bound=LOWER_BOUND)
    # out is a caller-allocated out-param; the state input stays keyword.
    T_total, H = pool["q"].shape[0], pool["q"].shape[1]
    kw["out"] = torch.empty(T_total, H, D, dtype=torch.bfloat16, device=DEV)
    if with_state:
        kw["initial_state"] = pool["state_pool"][:nseq]
    for _ in range(warmup):
        flashkda_fwd(**kw)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            flashkda_fwd(**kw)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    return min(samples), statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[12, 24])
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--with-state", action="store_true")
    ap.add_argument("--min-seg-tiles", type=int, default=4)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    rows = []
    print(f"{'H':>4} {'B':>3} {'T/seq':>7} {'S':>3} {'nseq':>5} "
          f"{'ms(min)':>9} {'ms(med)':>9} {'ideal':>7}")
    for H in a.heads:
        for B in a.batch:
            for T in a.tokens:
                splits = [s for s in a.splits
                          if T % (s * CHUNK) == 0
                          and (T // s) // CHUNK >= a.min_seg_tiles]
                if not splits:
                    continue
                pool = make_tokens(B * T, H, B * max(splits), a.with_state)
                base = None
                for S in splits:
                    seq_lens = [T // S] * (B * S)
                    mn, md = timed(pool, seq_lens, a.with_state,
                                   a.warmup, a.iters, a.repeats)
                    if S == 1:
                        base = mn
                    speed = base / mn if base else float("nan")
                    print(f"{H:>4} {B:>3} {T:>7} {S:>3} {len(seq_lens):>5} "
                          f"{mn:>9.3f} {md:>9.3f} {speed:>7.3f}", flush=True)
                    rows.append(dict(H=H, B=B, T=T, S=S, nseq=len(seq_lens),
                                     ms_min=mn, ms_med=md, ideal_speedup=speed,
                                     with_state=a.with_state))
                del pool
                torch.cuda.empty_cache()
                if a.json:
                    with open(a.json, "w") as f:
                        json.dump(rows, f, indent=1)
    if a.json:
        print(f"[json] {a.json}")


if __name__ == "__main__":
    main()
