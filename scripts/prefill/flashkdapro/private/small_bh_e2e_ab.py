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

"""Exact-shape PPU default/CP/FLA A/B for small-BH Kimi-K3 tails."""

import argparse
import math
import os
import statistics
import sys

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bench_vs_fla import (  # noqa: E402
    D,
    err_ratio,
    run_fla_chunk_kda,
    run_ppu_flashkda,
)


DECAY_SHIFT = {"fast": 0.0, "mid": -3.0, "slow": -4.5}


def set_cp(mode, bv=None, mult=None, seg_tiles=None):
    values = {
        "PLA_FLASHKDA_CP": mode,
        "PLA_FLASHKDA_CP_BV": bv,
        "PLA_FLASHKDA_CP_MULT": mult,
        "PLA_FLASHKDA_CP_SEG_TILES": seg_tiles,
    }
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def samples(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def ratio(a, b):
    return err_ratio(a.float().cpu().reshape(-1), b.float().cpu().reshape(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", required=True)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--state", action="store_true")
    parser.add_argument("--decay", choices=tuple(DECAY_SHIFT), default="slow")
    parser.add_argument(
        "--input", choices=("deterministic", "random", "empty"),
        default="deterministic",
        help=(
            "deterministic matches kimi_k3_e2e_rows.py and keeps input "
            "initialization outside the timed region"
        ),
    )
    parser.add_argument("--mults", default="3,5,8")
    parser.add_argument("--bvs", default="32")
    parser.add_argument("--seg-tiles", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    seq_lens = tuple(int(x) for x in args.seq_lens.split(",") if x)
    tokens, nseq, heads = sum(seq_lens), len(seq_lens), args.heads
    varlen = nseq > 1
    torch.manual_seed(1)
    rnd = lambda *shape: torch.randn(
        *shape, device="cuda", dtype=torch.float32
    )
    if args.input == "empty":
        mk = lambda *shape: torch.empty(
            *shape, device="cuda", dtype=torch.bfloat16
        )
        q = mk(1, tokens, heads, D)
        k = mk(1, tokens, heads, D)
        v = mk(1, tokens, heads, D)
        g = mk(1, tokens, heads, D)
        beta = mk(1, tokens, heads)
        A_log = torch.empty(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.empty(heads, D, device="cuda", dtype=torch.float32)
    elif args.input == "random":
        q = rnd(1, tokens, heads, D).bfloat16()
        k = rnd(1, tokens, heads, D).bfloat16()
        v = rnd(1, tokens, heads, D).bfloat16()
        g = (
            rnd(1, tokens, heads, D) * 0.3 + DECAY_SHIFT[args.decay]
        ).bfloat16()
        beta = rnd(1, tokens, heads).bfloat16()
        A_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros(heads, D, device="cuda", dtype=torch.float32)
    else:
        shape = (1, tokens, heads, D)
        q = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.full(shape, 0.25, device="cuda", dtype=torch.bfloat16)
        g = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        beta = torch.zeros(
            (1, tokens, heads), device="cuda", dtype=torch.bfloat16
        )
        A_log = torch.zeros(heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros(heads, D, device="cuda", dtype=torch.float32)
    h0 = (
        torch.zeros(nseq, heads, D, D, device="cuda", dtype=torch.float32)
        if args.state
        else None
    )
    cu = None
    if varlen:
        prefix = [0]
        for length in seq_lens:
            prefix.append(prefix[-1] + length)
        cu = torch.tensor(prefix, device="cuda", dtype=torch.int64)
    scale = 1.0 / math.sqrt(D)

    def ppu():
        return run_ppu_flashkda(
            q, k, v, g, beta, A_log, dt_bias, scale, h0, cu
        )

    extra = {"cu_seqlens": cu} if varlen else {}

    set_cp(0)
    serial_out = ppu()
    serial_samples = samples(ppu, args.warmup, args.iters)
    fla_out = run_fla_chunk_kda(
        q, k, v, g, beta, A_log, dt_bias, scale, h0, extra=extra
    )
    fla_samples = samples(
        lambda: run_fla_chunk_kda(
            q, k, v, g, beta, A_log, dt_bias, scale, h0, extra=extra
        ),
        args.warmup,
        args.iters,
    )
    serial_ms = statistics.median(serial_samples)
    fla_ms = statistics.median(fla_samples)
    print(
        f"shape H={heads} N={nseq} T={tokens} seq={list(seq_lens)} "
        f"state={int(args.state)} input={args.input} decay={args.decay}"
    )
    print(
        f"serial_ms={serial_ms:.4f} fla_ms={fla_ms:.4f} "
        f"fla/serial={fla_ms / serial_ms:.4f} "
        f"err_serial_fla_out={ratio(serial_out[0], fla_out[0]):.4e} "
        f"err_serial_fla_state={ratio(serial_out[1], fla_out[1]):.4e}"
    )

    from pla.prefill.flashkdapro import _C

    set_cp(1)
    auto_plan = _C.flashkda_cp_plan(tokens, heads, cu)
    auto_out = ppu()
    auto_samples = samples(ppu, args.warmup, args.iters)
    auto_ms = statistics.median(auto_samples)
    print(
        f"auto engage={int(auto_plan['engage'])} nseg={auto_plan['n_seg']} "
        f"maxseg={auto_plan['max_seg_tiles']} ms={auto_ms:.4f} "
        f"serial/auto={serial_ms / auto_ms:.4f} fla/auto={fla_ms / auto_ms:.4f} "
        f"err_auto_serial_out={ratio(auto_out[0], serial_out[0]):.4e} "
        f"err_auto_serial_state={ratio(auto_out[1], serial_out[1]):.4e}"
    )

    for bv in (int(x) for x in args.bvs.split(",") if x):
        for mult in (int(x) for x in args.mults.split(",") if x):
            set_cp(2, bv=bv, mult=mult, seg_tiles=args.seg_tiles)
            plan = _C.flashkda_cp_plan(tokens, heads, cu)
            candidate_out = ppu()
            candidate_samples = samples(ppu, args.warmup, args.iters)
            candidate_ms = statistics.median(candidate_samples)
            print(
                f"cp bv={bv} mult={mult} engage={int(plan['engage'])} "
                f"nseg={plan['n_seg']} maxseg={plan['max_seg_tiles']} "
                f"ms={candidate_ms:.4f} serial/cp={serial_ms / candidate_ms:.4f} "
                f"fla/cp={fla_ms / candidate_ms:.4f} "
                f"err_cp_serial_out={ratio(candidate_out[0], serial_out[0]):.4e} "
                f"err_cp_serial_state={ratio(candidate_out[1], serial_out[1]):.4e}"
            )

    set_cp(1)


if __name__ == "__main__":
    main()
