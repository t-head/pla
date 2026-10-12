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

"""Kimi-K3 prefill sweep: PPU (current dispatch, default env) vs FLA-Triton, per shape.

For every unique shape found in a Kimi-K3 tp{1,2,4,8} .list file, benchmark:
  ppu = pla.prefill.flashkdapro.flashkda_fwd  (whatever dispatch selects)
  fla = fla.ops.kda.chunk_kda                  (with the same lower-bound gate)

Emits one JSONL row per shape with median latencies + dispatch hit and winner:
  {"tp","H","N","T_total","seq_lens_hash","ppu_ms","fla_ms","winner","fast"}

fast = whether the case hits the flashkdapro register-resident nows fast path
       (mirror of the outer use_nows gate in kernel2_recurrence.cu):
         (N==1 && T>=256) || (N*H>=40) || (varlen && H in {6,12})

Env:
  --tp <int>         : which tp to sweep (H = {96,48,24,12}[tp])
  --list-path <str>  : Kimi-K3 .list file path (default derived from tp)
  --output <str>     : JSONL output path
  --start / --end    : shard within the shape list (for parallel-per-device runs)
  --warmup/--iters/--repeats : bench_fn samples (default 2/5/2 = 10 samples)

Run inside hp_ppu15_flashqla_kda container, from ppu_port dir:
  FLA_SOURCE_PATH=/mnt/ssd/hpwu/flash-linear-attention \
  PYTHONPATH=tests:.:/mnt/ssd/hpwu/flash-linear-attention \
  CUDA_VISIBLE_DEVICES=2 python3 tests/kimi_k3_sweep_all.py --tp 8 --output /tmp/tp8.jsonl
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch

# fla runner + timing helper come from ppu_port's tests/bench_vs_fla.py
from bench_vs_fla import (
    D, LOWER_BOUND, bench_fn, run_fla_chunk_kda,
)
# NOTE: we deliberately do NOT use bench_vs_fla.run_ppu_flashkda — that hits the
# ppu_port live .so whose K2 varlen gate is still the old hardcoded point.  The
# claim under test is the REFACTORED pla.prefill.flashkdapro dispatch, so we
# call it directly (installed wheel; PLA_FLASHKDA_K2_* env vars govern it).
from pla.prefill.flashkdapro import flashkda_fwd as _pla_flashkda_fwd


def run_ppu_flashkda(q, k, v, g, beta, A_log, dt_bias, scale, h0, cu_seqlens=None):
    # out/final_state are caller-allocated out-params; allocate inside the timed
    # region so this stays comparable to the FLA baseline, which allocates too.
    T, H, Dh = q.shape[1], q.shape[2], q.shape[3]
    n_seq = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    o = torch.empty(T, H, Dh, dtype=torch.bfloat16, device=q.device)
    fin = torch.empty(n_seq, H, Dh, Dh, dtype=torch.bfloat16, device=q.device)
    _pla_flashkda_fwd(
        q[0], k[0], v[0], g[0], beta[0], A_log, dt_bias, o, scale, LOWER_BOUND,
        h0.to(torch.bfloat16) if h0 is not None else None,
        fin, cu_seqlens,
    )
    return o.unsqueeze(0), fin

DEV = "cuda"
CASE_RE = re.compile(
    r"num_heads_qk:(\d+),.*?batch_size:(\d+),dtype:bf16,cu_seqlens_q:\[([\d,]+)\]"
)


def parse_list(path):
    """Return list of (H, N, T_total, seq_lens) preserving order but deduping identical rows."""
    seen = set()
    out = []
    with open(path) as fh:
        for line in fh:
            m = CASE_RE.search(line)
            if not m:
                continue
            H = int(m.group(1))
            cu = [int(x) for x in m.group(3).split(",")]
            seq_lens = tuple(cu[i + 1] - cu[i] for i in range(len(cu) - 1))
            T = sum(seq_lens)
            N = len(seq_lens)
            key = (H, N, T, seq_lens)
            if key in seen:
                continue
            seen.add(key)
            out.append((H, N, T, seq_lens))
    return out


def is_fast(H, N, T, is_varlen):
    """Mirror of the outer use_nows AUTO gate in kernel2_recurrence.cu."""
    return ((N == 1 and T >= 256)
            or (N * H >= 40)
            or (is_varlen and H in (6, 12)))


def build_inputs(H, T_total, seq_lens):
    N = len(seq_lens)
    is_varlen = N > 1
    scale = 1.0 / math.sqrt(D)
    if is_varlen:
        cu = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.long, device=DEV)
    else:
        cu = None
    # Uninitialized bf16 tensors -- we're timing, not validating (correctness is
    # verified separately by e2e_kimi_k3.py). torch.randn on ~30MB bf16 tensors
    # dominates wall time otherwise (100-1000x kernel latency).
    q = torch.empty((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    k = torch.empty((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    v = torch.empty((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    g = torch.empty((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    beta = torch.empty((1, T_total, H), dtype=torch.bfloat16, device=DEV)
    A_log = torch.empty(H, dtype=torch.float32, device=DEV)
    dt_bias = torch.empty(H, D, dtype=torch.float32, device=DEV)
    return (q, k, v, g, beta, A_log, dt_bias, scale, cu, is_varlen)


def median_ms(fn, warmup, iters, repeats):
    """bench_fn returns (mean, min, max) across warmup+iters*repeats samples."""
    mean, mn, mx = bench_fn(fn, warmup, iters, repeats)
    return mean


def set_nows(mode):
    """mode: 'default' (unset), 'baseline' (NOWS=0), 'nows' (NOWS=1)."""
    key = "PLA_FLASHKDA_K2_NOWS"
    os.environ.pop(key, None)
    if mode == "baseline":
        os.environ[key] = "0"
    elif mode == "nows":
        os.environ[key] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True, choices=[1, 2, 4, 8])
    ap.add_argument("--list-path", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap number of shapes (0=all)")
    args = ap.parse_args()

    DEFAULT_LIST_DIR = os.environ.get("KIMI_K3_LIST_DIR", "/default/path/to/kimi_k3_lists")
    list_dir = args.list_path or DEFAULT_LIST_DIR
    lp = args.list_path or f"{list_dir}/Kimi-K3_prefill_tp{args.tp}_kvlimit_200GB_model_len_128K_kda_op.list"
    shapes = parse_list(lp)
    print(f"[tp{args.tp}] parsed {len(shapes)} unique shapes from {lp}", flush=True)
    if args.limit:
        shapes = shapes[:args.limit]
    end = len(shapes) if args.end < 0 else args.end
    slice_shapes = shapes[args.start:end]
    print(f"[tp{args.tp}] running shard [{args.start}:{end}] = {len(slice_shapes)} shapes; params w={args.warmup} i={args.iters} r={args.repeats}", flush=True)

    # Warm the CUDA context / allocator / Triton JIT once so the FIRST measured
    # shape is not contaminated by one-time cold-start cost.
    try:
        H0 = slice_shapes[0][0] if slice_shapes else 12
        (wq, wk, wv, wg, wb, wA, wdt, wsc, wcu, _) = build_inputs(H0, 2048, (512, 512, 512, 512))
        for _ in range(3):
            run_ppu_flashkda(wq, wk, wv, wg, wb, wA, wdt, wsc, None, wcu)
            run_fla_chunk_kda(wq, wk, wv, wg, wb, wA, wdt, wsc, None, extra={"cu_seqlens": wcu})
        torch.cuda.synchronize()
        del wq, wk, wv, wg, wb, wA, wdt
        print(f"[tp{args.tp}] warmup done", flush=True)
    except Exception as exc:
        print(f"[tp{args.tp}] warmup skipped: {exc}", flush=True)

    ppu_wins = fla_wins = ties = 0          # default-PPU vs FLA
    dispatch_missed = 0                     # default not within 2% of ppu_best
    missed_fast = missed_base = 0
    worst_regress = 1.0                     # max default_ms/best_ms
    t0 = time.time()

    with open(args.output, "w") as fh:
        for idx, (H, N, T_total, seq_lens) in enumerate(slice_shapes):
            (q, k, v, g, beta, A_log, dt_bias, scale, cu, is_varlen) = build_inputs(
                H, T_total, seq_lens)
            fast = is_fast(H, N, T_total, is_varlen)

            def ppu_call():
                return run_ppu_flashkda(q, k, v, g, beta, A_log, dt_bias, scale, None, cu)
            def fla_call():
                return run_fla_chunk_kda(q, k, v, g, beta, A_log, dt_bias, scale, None,
                                         extra={"cu_seqlens": cu} if is_varlen else {})

            # 3-way PPU: default gate, forced baseline (NOWS=0), forced nows (NOWS=1)
            set_nows("default");  ppu_def = median_ms(ppu_call, args.warmup, args.iters, args.repeats)
            set_nows("baseline"); ppu_base = median_ms(ppu_call, args.warmup, args.iters, args.repeats)
            set_nows("nows");     ppu_nows = median_ms(ppu_call, args.warmup, args.iters, args.repeats)
            set_nows("default")
            fla_ms = median_ms(fla_call, args.warmup, args.iters, args.repeats)

            ppu_best = min(ppu_def, ppu_base, ppu_nows)
            # dispatch is optimal if the default path is within 2% of the best
            # reachable PPU path for this shape.
            regress = ppu_def / ppu_best if ppu_best > 0 else 1.0
            dispatch_optimal = regress <= 1.02
            if not dispatch_optimal:
                dispatch_missed += 1
                if fast: missed_fast += 1
                else:    missed_base += 1
            worst_regress = max(worst_regress, regress)

            if ppu_def < fla_ms * 0.98:
                winner = "ppu"; ppu_wins += 1
            elif fla_ms < ppu_def * 0.98:
                winner = "fla"; fla_wins += 1
            else:
                winner = "tie"; ties += 1

            row = {"tp": args.tp, "H": H, "N": N, "T_total": T_total,
                   "seq_lens_hash": hash(seq_lens) & 0xffffffff,
                   "ppu_def_ms": ppu_def, "ppu_base_ms": ppu_base,
                   "ppu_nows_ms": ppu_nows, "ppu_best_ms": ppu_best,
                   "fla_ms": fla_ms, "winner_vs_fla": winner,
                   "fast": fast, "dispatch_optimal": dispatch_optimal,
                   "regress": regress}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            del q, k, v, g, beta, A_log, dt_bias
            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(slice_shapes) - idx - 1) / rate
                print(f"  [{idx+1}/{len(slice_shapes)}] "
                      f"ppu_vs_fla_wins={ppu_wins}/{fla_wins} "
                      f"dispatch_missed={dispatch_missed} worst_regress={worst_regress:.3f} "
                      f"rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)
    elapsed = time.time() - t0
    print(f"\n[tp{args.tp}] DONE in {elapsed/60:.2f} min ({len(slice_shapes)} shapes)", flush=True)
    print(f"  vs FLA: ppu_wins={ppu_wins} fla_wins={fla_wins} ties={ties}")
    print(f"  dispatch NOT optimal (default >2% slower than best PPU path): "
          f"{dispatch_missed} (fast-bucket={missed_fast}, base-bucket={missed_base})")
    print(f"  worst dispatch regression (default/best): {worst_regress:.3f}x")


if __name__ == "__main__":
    main()
