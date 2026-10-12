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

"""pla.decode.kda_mtp_sglang fold -- OperatorsForge Kimi-K3 MTP case matrix.

Covers the full sglang MTP speculative-decoding space (seqlen_q in {3,4,5,8} x heads in {12,24,48,96}
x batch_size from 1 to ~600), running on the same inputs:

  * the pla CUDA fold (``pla.decode.kda_mtp_sglang._C.fold_all_layers``)
  * the upstream sglang Triton fold (vendored copy ``_triton_fold.py``,
    ``kda_replayssm_exact_fold_kernel``)

Both sides match the upstream fp64 numerical semantics (~1e-7 fp32 difference); this script only measures end-to-end latency: per case
median us + speedup ratio, then prints the geomean speedup + win_rate at the end. The case list is derived from
``OperatorsForge/FlashLinearAttn/GeneralizedUTLists/inference/decode/
Kimi-K3_decode_tp{1,2,4,8}_*_kda_op.list`` (TP1..8 map to H=96/48/24/12).

With large batches the fp32 ``temporal [layers, slots, H, 128, 128]`` would blow up device memory, so the script adaptively
reduces the layer count against a 5x working-set budget (K3 originally has 69 layers, minimum 1).

Usage (inside the PPU server container):
    CUDA_VISIBLE_DEVICES=6 python scripts/decode/kda_mtp_sglang/bench_fold_mtp_cases.py
    CUDA_VISIBLE_DEVICES=6 python scripts/decode/kda_mtp_sglang/bench_fold_mtp_cases.py \\
        --seqlens 3,8 --heads 12,96 --reps 10 --inner 3
    CUDA_VISIBLE_DEVICES=6 python scripts/decode/kda_mtp_sglang/bench_fold_mtp_cases.py \\
        --max-cases 40   # quick pass
"""

# ruff: noqa: E501
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "private"))

import torch

from _common import Cfg, build_commit_inputs
from _harness import Timer, save
from _triton_fold import triton_fold_all_layers

from pla.decode.kda_mtp_sglang import _C as _ext

# Max batch size per head count (OperatorsForge memory envelope, TP1..8).
MAX_BS_MAP = {12: 615, 24: 370, 48: 200, 96: 106}


def log_sample_bs(max_bs: int, n: int = 10) -> list[int]:
    """Sample ~n batch sizes log-spaced from 1 to max_bs, deduped."""
    if max_bs <= 1:
        return [1]
    pts = set()
    for i in range(n):
        v = int(round(math.exp(math.log(max_bs) * i / (n - 1))))
        pts.add(max(1, min(v, max_bs)))
    return sorted(pts)


def cuda_fold(ci, temporal):
    _ext.fold_all_layers(
        temporal, ci["rawv"], ci["rawk"], ci["gk"], ci["beta"],
        ci["state_batch_indices"], ci["accept_lens"], None, None,
        ci["rawk"].shape[2], -1)


def tri_fold(ci, temporal, L):
    triton_fold_all_layers(
        temporal, ci["rawv"], ci["rawk"], ci["gk"], ci["beta"],
        ci["state_batch_indices"], ci["accept_lens"], L, ci["rawk"].shape[2],
        null_block_id=-1, use_qk_l2norm=True)


def build_cases(seqlens, heads_list, max_cases):
    """Build (seqlen_q, heads, bs) case list."""
    cases = []
    for sq in seqlens:
        for h in heads_list:
            max_bs = MAX_BS_MAP.get(h, 100)
            for bs in log_sample_bs(max_bs, n=10):
                cases.append((sq, h, bs))
    if max_cases and len(cases) > max_cases:
        step = max(1, len(cases) // max_cases)
        cases = cases[::step][:max_cases]
    return cases


def main():
    ap = argparse.ArgumentParser(
        description="Compare pla CUDA fold vs sglang Triton fold on the "
                    "OperatorsForge Kimi-K3 MTP case matrix")
    ap.add_argument("--seqlens", default="3,4,5,8",
                    help="Comma-separated seqlen_q values (default: 3,4,5,8)")
    ap.add_argument("--heads", default="12,24,48,96",
                    help="Comma-separated head counts (default: 12,24,48,96)")
    ap.add_argument("--reps", type=int, default=15,
                    help="Timer repetitions (default: 15)")
    ap.add_argument("--inner", type=int, default=5,
                    help="Timer inner iterations (default: 5)")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Limit total cases for quick runs")
    ap.add_argument("--json", default=None,
                    help="Optional path to save the results JSON (default: none)")
    args = ap.parse_args()

    seqlens = [int(x) for x in args.seqlens.split(",")]
    heads_list = [int(x) for x in args.heads.split(",")]

    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  sm_{p.major}{p.minor}  SMs={p.multi_processor_count}")
    print(f"timer:  reps={args.reps} inner={args.inner}")

    cases = build_cases(seqlens, heads_list, args.max_cases)
    print(f"cases:  {len(cases)} (seqlen_q in {seqlens}, heads in {heads_list})")

    timer = Timer(reps=args.reps, inner=args.inner)
    rows = []
    print(f"\n  {'seqlen_q':>8}  {'heads':>5}  {'bs':>5}  {'cuda_us':>9}  "
          f"{'triton_us':>10}  {'speedup':>8}")
    print(f"  {'--------':>8}  {'-----':>5}  {'-----':>5}  {'---------':>9}  "
          f"{'----------':>10}  {'--------':>8}")

    for sq, h, bs in cases:
        # Adaptive layers: cap the working set at ~60 GB. Working set ≈
        # temporal[layers, slots, H, 128, 128] * 3 (orig + 2 clones) + rings
        # ≈ 1.5× temporal; budget as 5× per-layer state.
        MAX_MEM = 60e9
        state_bytes_per_layer = (bs + 2) * h * 128 * 128 * 4
        layers = min(69, max(1, int(MAX_MEM / (5 * state_bytes_per_layer))))

        # num_spec = seqlen_q - 1  =>  draft_token_num = seqlen_q
        cfg = Cfg(bs=bs, num_spec=sq - 1, heads=h, ring_len=16, layers=layers)

        # Uniform accept_lens = seqlen_q (accept-all MTP case).
        accept_lens = torch.full((bs,), sq, device="cuda", dtype=torch.int32)
        ci = build_commit_inputs(cfg, accept_lens=accept_lens)
        L = ci["rawv"].shape[3]

        # Warmup (also proves both backends run cleanly on this shape).
        t_cuda = ci["temporal"].clone()
        t_tri = ci["temporal"].clone()
        cuda_fold(ci, t_cuda)
        tri_fold(ci, t_tri, L)
        torch.cuda.synchronize()

        t_cuda = ci["temporal"].clone()
        rc = timer.run(lambda: cuda_fold(ci, t_cuda))
        t_tri = ci["temporal"].clone()
        rt = timer.run(lambda: tri_fold(ci, t_tri, L))

        cuda_us = rc["median_ms"] * 1e3
        triton_us = rt["median_ms"] * 1e3
        speedup = triton_us / cuda_us if cuda_us > 0 else float("inf")
        rows.append(dict(seqlen_q=sq, heads=h, bs=bs, layers=layers,
                         cuda_us=cuda_us, triton_us=triton_us,
                         speedup=speedup))
        print(f"  {sq:>8}  {h:>5}  {bs:>5}  {cuda_us:>9.1f}  "
              f"{triton_us:>10.1f}  {speedup:>7.2f}x")

        del ci, t_cuda, t_tri, accept_lens
        torch.cuda.empty_cache()

    # ---- summary -----------------------------------------------------------
    speedups = [r["speedup"] for r in rows]
    if speedups:
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        wins = sum(1 for s in speedups if s > 1.0)
        win_rate = wins / len(speedups) * 100.0
        print(f"\n  summary: {len(speedups)} cases")
        print(f"    geomean speedup: {geomean:.3f}x")
        print(f"    win rate:        {wins}/{len(speedups)} ({win_rate:.1f}%)")
    else:
        print("\n  no cases to summarize")
        geomean, win_rate = 0.0, 0.0

    if args.json:
        save(args.json, dict(
            cases=rows,
            summary=dict(geomean_speedup=geomean,
                         win_rate_pct=win_rate,
                         total_cases=len(speedups)),
            config=dict(seqlens=seqlens, heads=heads_list,
                        reps=args.reps, inner=args.inner,
                        layers="adaptive_up_to_69"),
        ))


if __name__ == "__main__":
    main()
