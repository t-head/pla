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

"""Kimi-K3 CP A/B rows: CP off vs CP auto vs FLA-Triton, on the shapes where the
flashkdapro CP dispatch can actually engage.

Why a separate script.  kimi_k3_e2e_rows.py samples 150 of the ~1296 unique
shapes per tp, which is the right sampling for the production mix but lands on
only 3 of the 29 tp8 (H=12) single-sequence shapes that are long enough for the
CP AUTO gate.  Averaged over 150 shapes the CP effect is invisible, so this
script keeps every shape the gate accepts and drops the rest.

Engagement is queried from the kernel's own planner (``flashkda_cp_plan``), not
re-derived in Python, so the selection can never drift from the dispatch under
test.  The control is the SAME binary on the SAME device with
``PLA_FLASHKDA_CP=0`` -- the env var is read per launch -- which is the only way
to attribute a delta to CP rather than to machine occupancy (the report's gate
panel documents a previous cross-device A/B that produced spurious regressions).

Row schema = kimi_k3_e2e_rows.py plus:
  ppu_cpoff_*     same binary with CP disabled
  cp_gain         ppu_cpoff_median_ms / ppu_median_ms   (>1 = CP faster)
  speedup_cpoff   fla_median_ms / ppu_cpoff_median_ms
  n_seg, max_seg_tiles, cp_engaged, decay

Decay regime is not optional for CP.  pre_scan's transition chain has an exact
early exit on a zero state slice (kernel2_recurrence_nows.cu, ZCHK=8): under a
fast decay M_i underflows within a few chunks and pre_scan costs a fraction of a
pass, under a slow decay it never underflows and pre_scan pays in full.  So CP
cost is data-dependent and uninitialised tensors measure neither bound --
``--decay slow`` and ``--decay fast`` must both be run and reported as bounds.
``--decay empty`` keeps the torch.empty inputs of the production-mix sweeps,
where CP is gated off for all but a handful of shapes and the point of the run is
the null control, not the CP cost.

Run inside hp_ppu15_sglang0.5.12, from the pla tree:
  PYTHONPATH=scripts/prefill/flashkdapro:scripts/prefill/flashkdapro/private:$PWD:$FLA \
  PPU_VISIBLE_DEVICES=0 python3 scripts/prefill/flashkdapro/private/kimi_k3_cp_rows.py \
      --tp 8 --output /tmp/rows_cp_tp8.jsonl
"""
import argparse
import json
import os
import time

import torch

from bench_vs_fla import D, LOWER_BOUND, run_fla_chunk_kda
from cp_stage_timing import DECAYS
from kimi_k3_e2e_rows import build, parse_list, run_ppu, stats, suite_of, time_fn
from pla.prefill.flashkdapro import _C as _flashkda_C

LIST_DIR = os.environ.get("KIMI_K3_LIST_DIR", "/mnt/ssd/hpwu/kimi_k3_lists_gino")


def set_cp(on):
    """PLA_FLASHKDA_CP is re-read on every launch, so this is a live switch."""
    if on:
        os.environ.pop("PLA_FLASHKDA_CP", None)
    else:
        os.environ["PLA_FLASHKDA_CP"] = "0"


def apply_decay(q, k, v, g, beta, A_log, dt_bias, regime):
    """Give the gate a known decay regime, in place.

    Same convention as cp_crossover_sweep / cp_stage_timing so the two sets of
    numbers are comparable: g ~ N(DECAYS[regime], 0.3), A_log and dt_bias zeroed
    (build() leaves them as torch.empty garbage, which would make the gate -- and
    therefore pre_scan's early exit -- undefined).  One-off setup cost only; the
    timed region never sees it.
    """
    if regime == "empty":
        return
    for t in (q, k, v):
        t.normal_(0.0, 1.0)
    beta.normal_(0.0, 1.0)
    g.normal_(0.0, 0.3).add_(DECAYS[regime])
    A_log.zero_()
    dt_bias.zero_()


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--iters-large", type=int, default=0,
                    help="iters used when token_heads > --large-th (0 = same as "
                         "--iters).  Three configs x (warmup+iters) launches per "
                         "row makes the multi-100k-token tail dominate wall time; "
                         "these shapes have ms-scale latency so fewer samples "
                         "still give a tight median.")
    ap.add_argument("--large-th", type=int, default=4_000_000,
                    help="token_heads threshold for --iters-large")
    ap.add_argument("--decay", choices=["empty", "fast", "mid", "slow"],
                    default="empty",
                    help="gate decay regime.  'fast'/'slow' bracket pre_scan's "
                         "exact zero-state early exit and MUST both be reported "
                         "for any CP cost claim; 'empty' keeps torch.empty "
                         "inputs (production-mix null control only).")
    ap.add_argument("--limit", type=int, default=0, help="cap shapes (0 = all engaged)")
    ap.add_argument("--all-shapes", action="store_true",
                    help="keep non-engaging shapes too (control coverage)")
    ap.add_argument("--sample", type=int, default=0,
                    help="representative sampling identical to kimi_k3_e2e_rows.py "
                         "(sort by (N,T), even stride) so rows pair 1:1 with the "
                         "dataset already in the report; 0 = every shape")
    args = ap.parse_args()

    H_of = {1: 96, 2: 48, 4: 24, 8: 12}
    lp = (f"{LIST_DIR}/Kimi-K3_prefill_tp{args.tp}"
          f"_kvlimit_200GB_model_len_128K_kda_op.list")
    shapes = parse_list(lp)
    print(f"[tp{args.tp}] {len(shapes)} unique shapes, H={H_of[args.tp]}", flush=True)

    # Sampling must happen BEFORE the CP filter and must be bit-identical to
    # kimi_k3_e2e_rows.py, otherwise the rows cannot be paired with the ones
    # already embedded in the report.
    if args.sample:
        shapes.sort(key=lambda s: (s[1], s[2]))
        if len(shapes) > args.sample:
            step = len(shapes) / args.sample
            shapes = [shapes[int(i * step)] for i in range(args.sample)]
        print(f"[tp{args.tp}] sampled {len(shapes)} shapes", flush=True)

    # Ask the dispatch itself which shapes CP would take.
    set_cp(True)
    kept = []
    for (H, N, T_total, seq_lens) in shapes:
        cu = None
        if N > 1:
            cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
                              dtype=torch.long, device="cuda")
        plan = _flashkda_C.flashkda_cp_plan(T_total, H, cu)
        if plan["engage"] or args.all_shapes:
            kept.append((H, N, T_total, seq_lens, bool(plan["engage"]),
                         int(plan["n_seg"]), int(plan["max_seg_tiles"])))
    kept.sort(key=lambda s: (s[1], s[2]))
    if args.limit:
        kept = kept[:args.limit]
    print(f"[tp{args.tp}] CP-engaging shapes: {len(kept)}"
          f" (T range {kept[0][2] if kept else 0}..{kept[-1][2] if kept else 0})"
          f" decay={args.decay}",
          flush=True)
    if not kept:
        print("nothing to measure")
        return

    # Warm the context / Triton JIT once.
    try:
        b = build(H_of[args.tp], 2048, (512, 512, 512, 512))
        for _ in range(3):
            run_ppu(*b[:8], None, b[8])
            run_fla_chunk_kda(*b[:8], None, extra={"cu_seqlens": b[8]})
        torch.cuda.synchronize()
        print("warmup done", flush=True)
    except Exception as exc:
        print(f"warmup skip: {exc}", flush=True)

    t0 = time.time()
    gains, sp_on, sp_off = [], [], []
    with open(args.output, "w") as fh:
        for idx, (H, N, T_total, seq_lens, engaged, n_seg, max_seg_tiles) in enumerate(kept):
            suite = suite_of(N, seq_lens)
            S = seq_lens[0] if N == 1 else (T_total // N)
            warm, iters = args.warmup, args.iters
            if args.iters_large and T_total * H > args.large_th:
                warm, iters = 1, args.iters_large
            for mode in ("no_state", "with_state"):
                (q, k, v, g, beta, A_log, dt_bias, scale, cu, is_varlen, _N) = build(
                    H, T_total, seq_lens)
                apply_decay(q, k, v, g, beta, A_log, dt_bias, args.decay)
                h0 = (torch.zeros((N, H, D, D), dtype=torch.float32, device="cuda")
                      if mode == "with_state" else None)
                ck = time.time()
                try:
                    def ppu_call():
                        return run_ppu(q, k, v, g, beta, A_log, dt_bias, scale, h0, cu)

                    def fla_call():
                        return run_fla_chunk_kda(
                            q, k, v, g, beta, A_log, dt_bias, scale, h0,
                            extra={"cu_seqlens": cu} if is_varlen else {})

                    set_cp(True)
                    on_s = time_fn(ppu_call, warm, iters)
                    set_cp(False)
                    off_s = time_fn(ppu_call, warm, iters)
                    set_cp(True)
                    fla_s = time_fn(fla_call, warm, iters)

                    pm, pmean, pp10, pp90, pcv = stats(on_s)
                    om, omean, op10, op90, ocv = stats(off_s)
                    fm, fmean, fp10, fp90, fcv = stats(fla_s)
                    row = {
                        "case_id": f"tp{args.tp}-{suite}-H{H}-N{N}-T{T_total}",
                        "suite": suite, "mode": mode, "H": H, "B": N, "S": S, "D": D,
                        "T_total": T_total, "token_heads": T_total * H,
                        "seq_lens": list(seq_lens) if N <= 8 else [seq_lens[0], "...", seq_lens[-1]],
                        "status": "ok", "elapsed_s": round(time.time() - ck, 3),
                        "ppu_median_ms": pm, "ppu_mean_ms": pmean,
                        "ppu_p10_ms": pp10, "ppu_p90_ms": pp90, "ppu_cv": pcv,
                        "ppu_cpoff_median_ms": om, "ppu_cpoff_mean_ms": omean,
                        "ppu_cpoff_p10_ms": op10, "ppu_cpoff_p90_ms": op90,
                        "ppu_cpoff_cv": ocv,
                        "fla_median_ms": fm, "fla_mean_ms": fmean,
                        "fla_p10_ms": fp10, "fla_p90_ms": fp90, "fla_cv": fcv,
                        "speedup": fm / pm if pm > 0 else 0.0,
                        "speedup_cpoff": fm / om if om > 0 else 0.0,
                        "cp_gain": om / pm if pm > 0 else 0.0,
                        "cp_engaged": engaged, "n_seg": n_seg,
                        "max_seg_tiles": max_seg_tiles, "decay": args.decay,
                        "retested": False, "sample_count": len(on_s),
                        "warmup": warm, "iters": iters,
                        "tp": args.tp,
                    }
                    gains.append(row["cp_gain"])
                    sp_on.append(row["speedup"])
                    sp_off.append(row["speedup_cpoff"])
                except Exception as exc:
                    row = {"case_id": f"tp{args.tp}-{suite}-H{H}-N{N}-T{T_total}",
                           "suite": suite, "mode": mode, "H": H, "B": N, "S": S, "D": D,
                           "T_total": T_total, "token_heads": T_total * H,
                           "seq_lens": list(seq_lens)[:4],
                           "status": "error:" + str(exc)[:80], "speedup": 0.0,
                           "cp_gain": 0.0, "cp_engaged": engaged,
                           "decay": args.decay, "tp": args.tp}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                del q, k, v, g, beta, A_log, dt_bias, h0
            if (idx + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{idx+1}/{len(kept)}] {el/60:.1f}min "
                      f"cp_gain geo={geomean(gains):.4f}", flush=True)
    set_cp(True)
    print(f"[tp{args.tp}] DONE {(time.time()-t0)/60:.1f}min  points={len(gains)}"
          f"  decay={args.decay}", flush=True)
    print(f"  cp_gain      geo={geomean(gains):.4f}  "
          f"min={min(gains):.4f} max={max(gains):.4f}")
    print(f"  vs FLA       CP off geo={geomean(sp_off):.4f} -> CP on geo={geomean(sp_on):.4f}")
    print(f"  losses <1x   off={sum(1 for s in sp_off if s < 1)} "
          f"on={sum(1 for s in sp_on if s < 1)}")


if __name__ == "__main__":
    main()
