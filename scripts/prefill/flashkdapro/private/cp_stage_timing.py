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

"""Per-pass wall-clock breakdown of the CP pipeline, to explain CP-vs-serial.

CP replaces one long serial chain with three passes:

    K1 (prepare)  ->  pre_scan (2 chains, per segment)  ->  merge  ->  segment K2

`flashkda_cp_stages` stops after the merge, so timing it against the full CP
forward splits the pipeline without any device-side instrumentation:

    t_stages          = K1 + pre_scan + merge
    t_segment         = t_cp - t_stages
    t_prescan+merge   = t_stages - t_K1

t_K1 is measured by running the plain forward with the recurrence's work made
negligible (T split into one segment per CU is not possible for K1, so K1 is
instead taken from the caller-supplied --k1 acu number when given).

Timing rule on PPU: never wrap a single kernel in device events inside one API
call (it can wedge the queue).  Every number here is the wall clock of a whole
API call with exactly one synchronise at the end of a batch of iterations.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

D = 128
LOWER = -5.0

# Gate shift applied to g before the sigmoid: the KDA gate is exp2(cumsum(...)),
# so a shift towards -inf drives the gate to 1 (no decay, the pre_scan transition
# chain never underflows) and a shift of 0 makes the state die inside one chunk
# (the transition chain hits its exact early exit immediately).  "empty" keeps
# uninitialised buffers: cheapest to set up, but the garbage gate values make the
# early exit behave arbitrarily, so it is only a pessimistic bound.
DECAYS = {"fast": 0.0, "mid": -3.0, "slow": -4.5}


def inputs(T, H, use_state, nseq, decay="empty"):
    # torch.empty, not randn: filling ~30MB of bf16 costs orders of magnitude
    # more than the kernel being measured.  Only the one-off setup pays for the
    # realistic-gate variants, never the timed region.
    mk = lambda *s: torch.empty(*s, device="cuda", dtype=torch.bfloat16)
    if decay == "empty":
        q, k, v, g, beta = mk(T, H, D), mk(T, H, D), mk(T, H, D), mk(T, H, D), mk(T, H)
    else:
        rnd = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32)
        q, k, v = rnd(T, H, D).bfloat16(), rnd(T, H, D).bfloat16(), rnd(T, H, D).bfloat16()
        g = (rnd(T, H, D) * 0.3 + DECAYS[decay]).bfloat16()
        beta = rnd(T, H).bfloat16()
    return dict(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=torch.zeros(H, device="cuda"),
        dt_bias=torch.zeros(H, D, device="cuda"),
        h0=(torch.zeros(nseq, H, D, D, device="cuda", dtype=torch.bfloat16)
            if use_state else None),
    )


def timeit(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def set_env(**kw):
    for name, val in kw.items():
        key = "PLA_FLASHKDA_" + name
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument(
        "--seq-lens",
        type=str,
        default="",
        help="exact comma-separated ragged lengths; overrides --tokens/--seqs",
    )
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--force-cu", action="store_true",
                    help="pass a 2-entry cu_seqlens table for N=1 so the "
                         "single-sequence varlen dispatch can be measured")
    ap.add_argument("--mults", type=str, default="0,2,3,5,8,12,16,24")
    ap.add_argument("--bvs", type=str, default="16,32")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--decay", choices=["empty", "fast", "mid", "slow"],
                   default="empty",
                   help="gate regime: 'fast' lets the pre_scan transition chain "
                        "hit its exact early exit at once, 'slow' never lets it "
                        "underflow (worst case), 'empty' is uninitialised input")
    a = ap.parse_args()

    from pla.prefill.flashkdapro import _C

    seq_lens = (
        tuple(int(x) for x in a.seq_lens.split(",") if x)
        if a.seq_lens
        else ()
    )
    if any(x <= 0 for x in seq_lens):
        ap.error("--seq-lens entries must be positive")
    T = sum(seq_lens) if seq_lens else a.tokens
    H = a.heads
    N = len(seq_lens) if seq_lens else a.seqs
    x = inputs(T, H, a.state, N, a.decay)
    if seq_lens:
        prefix = [0]
        for length in seq_lens:
            prefix.append(prefix[-1] + length)
        cu = torch.tensor(prefix, dtype=torch.int64, device="cuda")
    else:
        cu = (torch.arange(0, T + 1, T // N, dtype=torch.int64, device="cuda")
              if N > 1 or a.force_cu else None)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"], x["A_log"],
            x["dt_bias"], 1.0 / (D ** 0.5), LOWER)
    # flashkda_fwd takes out/final_state as caller-allocated out-params; args[:7]
    # are the tensors and args[7:] the (scale, lower_bound) pair that follows
    # `out`.  flashkda_cp_stages keeps the original layout and still takes `args`
    # verbatim.
    nseq = 1 if cu is None else cu.numel() - 1
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device="cuda")
    fin = torch.empty(nseq, H, D, D, dtype=torch.bfloat16, device="cuda")
    fwd = lambda: _C.flashkda_fwd(*args[:7], out, *args[7:], x["h0"], fin, cu)

    set_env(CP=0)
    t_serial = timeit(fwd, a.iters)
    print(f"T={T} H={H} N={N} seq={list(seq_lens) if seq_lens else 'equal'} "
          f"varlen={int(cu is not None)} "
          f"state={int(a.state)} decay={a.decay}  "
          f"serial {t_serial:.3f} ms")

    for bv in [int(s) for s in a.bvs.split(",")]:
        for mult in [int(s) for s in a.mults.split(",")]:
            # mult=0 means "let AUTO pick the segment count".
            set_env(CP=2, CP_BV=bv, CP_MULT=mult or None, CP_SEG_TILES=None)
            plan = _C.flashkda_cp_plan(T, H, cu)
            if not plan["engage"]:
                print(f"  bv={bv} mult={mult or 'auto'}: no engage")
                continue
            t_cp = timeit(fwd, a.iters)
            t_st = timeit(lambda: _C.flashkda_cp_stages(*args, x["h0"], cu),
                          a.iters)
            nseg, maxseg = plan["n_seg"], plan["max_seg_tiles"]
            print(f"  bv={bv:<2d} mult={mult or 'auto':<4} nseg={nseg:<3d} "
                  f"maxseg={maxseg:<5d} | cp {t_cp:7.3f}  "
                  f"K1+pre+mrg {t_st:7.3f}  segment {t_cp - t_st:7.3f}  "
                  f"speedup {t_serial / t_cp:5.2f}x", flush=True)
    set_env(CP=1, CP_BV=None, CP_MULT=None)


if __name__ == "__main__":
    main()
