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

"""Kimi K3 KDA decode performance comparison -- PPU kernel vs ref kernel (E2E).

Both paths are CUDA kernels already compiled in this repo and share the same pybind entry
``fused_kda_decode_mega_forward``:

  * ``PPU``  : default path (``KDA_DECODE_IMPL`` unset) -- the PPU fused mega kernel.
  * ``ref``  : set ``KDA_DECODE_IMPL=ref`` -- internally re-dispatches to the NVIDIA CUDA reference implementation
              (``fused_kda_decode_ref_forward``), with the same signature/semantics as PPU.

Criteria: no correctness check (fully covered by compare_kda_decode_fusion.py); only end-to-end
       latency is measured. Timed with ``torch.cuda.Event``, taking the median (us).

Sweeps the HxB grid by default: H in {4,6,8,12,16,24,48,64,96} x B in {1,2,4,8,16,32,64}. Override with ``--heads`` and
``--batches``. lower_bound / fuse_output_norm are fixed to vllm test_kda's
main-case settings (lower_bound=-5.0, fuse_output_norm=True).

Usage:
    python scripts/decode/kda_fusion/bench_ppu_vs_ref.py
    python scripts/decode/kda_fusion/bench_ppu_vs_ref.py --iters 200 --json bench.json
    python scripts/decode/kda_fusion/bench_ppu_vs_ref.py --heads 24,96 --batches 1,8,64
"""

# ruff: noqa: E501
import argparse
import json
import os
import statistics
import sys

import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../scripts/decode/kda_fusion
PARENT_DIR = os.path.dirname(SCRIPTS_DIR)                  # .../scripts/decode
# NOTE: do NOT add the repo root to sys.path. The source tree's ``pla/``
# package would shadow the installed one and has no compiled ``_C``.
for _p in (SCRIPTS_DIR, PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse build_inputs / _alloc_paged_cache / D / W / NORM_EPS from compare
try:
    from kda_fusion.compare_kda_decode_fusion import (  # noqa: E402
        D, NORM_EPS, build_inputs, _alloc_paged_cache,
    )
except ModuleNotFoundError:
    from compare_kda_decode_fusion import (  # noqa: E402
        D, NORM_EPS, build_inputs, _alloc_paged_cache,
    )

from pla.decode.kda import fused_kda_decode_mega_forward  # noqa: E402

DEVICE = "cuda"


def _prepare_call(inp, lower_bound, fuse_output_norm):
    conv, state = _alloc_paged_cache(inp, conv_dtype=torch.bfloat16)
    out = torch.empty(
        1, inp["num_seqs"], inp["num_heads"], D,
        dtype=torch.bfloat16, device=DEVICE,
    )
    kwargs = dict(
        x=inp["packed_x"], weight=inp["fused_weight"], bias=None,
        conv_state=conv, raw_g=inp["raw_g"], raw_beta=inp["raw_beta"],
        a_log=inp["A_log"], dt_bias=inp["dt_bias"], state_indices=inp["state_indices"],
        state=state, out=out, lower_bound=lower_bound,
        output_gate=inp["output_gate"] if fuse_output_norm else None,
        norm_weight=inp["norm_weight"] if fuse_output_norm else None,
        norm_eps=NORM_EPS,
    )
    return kwargs


def _bench_impl(impl, kwargs, iters, warmup):
    """impl in {'ppu','ref'}. Set/clear the environment variable before every call -- the PPU kernel internally
    calls ``std::getenv("KDA_DECODE_IMPL")`` and dispatches on the fly."""
    if impl == "ppu":
        os.environ.pop("KDA_DECODE_IMPL", None)
    else:
        os.environ["KDA_DECODE_IMPL"] = "ref"

    for _ in range(warmup):
        fused_kda_decode_mega_forward(**kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fused_kda_decode_mega_forward(**kwargs)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3)  # us

    times.sort()
    med = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    mean = statistics.mean(times)
    return med, mean, p90


def _parse_int_list(s, default):
    if s is None:
        return default
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default=None,
                    help="comma-sep H values (default 4,6,8,12,16,24,48,64,96)")
    ap.add_argument("--batches", default=None,
                    help="comma-sep B values (default 1,2,4,8,16,32,64)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lower-bound", type=float, default=-5.0)
    ap.add_argument("--no-fon", action="store_true",
                    help="disable fuse_output_norm (default: enabled)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    heads = _parse_int_list(args.heads, [4, 6, 8, 12, 16, 24, 48, 64, 96])
    batches = _parse_int_list(args.batches, [1, 2, 4, 8, 16, 32, 64])
    fon = not args.no_fon
    lb = args.lower_bound

    print("=" * 108)
    print("Kimi K3 KDA decode E2E: PPU kernel vs ref kernel (compiled CUDA, same pybind entry)")
    print(f"  grid           : H {heads} x B {batches}, lower_bound={lb}, fuse_output_norm={fon}")
    print(f"  iters={args.iters}, warmup={args.warmup}, timer=cudaEvent(median)")
    print("=" * 108)
    header = (
        f"  {'H':>3} {'B':>3} | "
        f"{'PPU us':>8} {'p90':>7} | {'ref us':>8} {'p90':>7} | "
        f"{'ref/PPU':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for H in heads:
        for B in batches:
            seed = 967 + H + B
            inp = build_inputs(H, B, seed)
            kwargs = _prepare_call(inp, lb, fon)

            ppu_med, ppu_mean, ppu_p90 = _bench_impl("ppu", kwargs, args.iters, args.warmup)
            ref_med, ref_mean, ref_p90 = _bench_impl("ref", kwargs, args.iters, args.warmup)
            speedup = ref_med / ppu_med if ppu_med > 0 else float("nan")

            print(
                f"  {H:>3} {B:>3} | "
                f"{ppu_med:>8.2f} {ppu_p90:>7.2f} | {ref_med:>8.2f} {ref_p90:>7.2f} | "
                f"x{speedup:>6.2f}",
                flush=True,
            )

            rows.append({
                "num_heads": H, "num_seqs": B,
                "lower_bound": lb, "fuse_output_norm": fon,
                "ppu_us": {"median": ppu_med, "mean": ppu_mean, "p90": ppu_p90},
                "ref_us": {"median": ref_med, "mean": ref_mean, "p90": ref_p90},
                "speedup_ref_over_ppu": speedup,
            })

    # summary
    speedups = [r["speedup_ref_over_ppu"] for r in rows if r["speedup_ref_over_ppu"] == r["speedup_ref_over_ppu"]]
    if speedups:
        geomean = statistics.geometric_mean(speedups)
        print("=" * 108)
        print(
            f"summary: {len(speedups)} cells, geomean(ref/PPU)={geomean:.2f}, "
            f"min={min(speedups):.2f}, max={max(speedups):.2f}"
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "cells": rows,
                "iters": args.iters, "warmup": args.warmup,
                "lower_bound": lb, "fuse_output_norm": fon,
            }, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
