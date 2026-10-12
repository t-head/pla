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

"""Split the flashkdapro E2E wall into K1 (prepare) and K2 (recurrence).

CP only shortens the K2 critical path: K1 is already token-parallel and runs at
full occupancy.  So the achievable CP speedup is

    (K1 + K2_serial) / (K1 + (1 + prescan_ratio) * K2_parallel)

and the K1 share decides whether the pre_scan re-run can ever be paid for.
This probe measures K1 alone (the exposed flashkda_prepare entry point) and the
full fused forward, for the same shapes as cp_upper_bound.py.

  CUDA_VISIBLE_DEVICES=4 python3 -u \\
    scripts/prefill/flashkdapro/private/cp_k1_share.py
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from pla.prefill.flashkdapro import _C as ext
from pla.prefill.flashkdapro import flashkda_fwd

D = 128
CHUNK = 16
LOWER_BOUND = -5.0
DEV = "cuda"


def make_tokens(T_total, H, seed=0):
    torch.manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    return dict(q=rnd(T_total, H, D), k=rnd(T_total, H, D), v=rnd(T_total, H, D),
                g=rnd(T_total, H, D), beta=rnd(T_total, H),
                A_log=torch.randn(H, device=DEV, dtype=torch.float32),
                dt_bias=torch.randn(H, D, device=DEV, dtype=torch.float32))


def bench(fn, warmup, iters, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / iters * 1e3)
    return min(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[12, 24])
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    rows = []
    print(f"{'H':>4} {'T':>7} {'S':>3} {'k1_ms':>8} {'e2e_ms':>8} "
          f"{'k2_ms':>8} {'k1_share':>9}")
    for H in a.heads:
        for T in a.tokens:
            p = make_tokens(T, H)
            for S in a.splits:
                if T % (S * CHUNK):
                    continue
                cu = torch.arange(0, T + 1, T // S, dtype=torch.int64, device=DEV)
                out = torch.empty(T, H, D, dtype=torch.bfloat16, device=DEV)
                k1 = bench(lambda: ext.flashkda_prepare(
                    p["q"], p["k"], p["g"], p["beta"], p["A_log"], p["dt_bias"],
                    D ** -0.5, LOWER_BOUND, cu), a.warmup, a.iters, a.repeats)
                e2e = bench(lambda: flashkda_fwd(
                    p["q"], p["k"], p["v"], p["g"], p["beta"], p["A_log"],
                    p["dt_bias"], out=out, lower_bound=LOWER_BOUND, cu_seqlens=cu),
                    a.warmup, a.iters, a.repeats)
                k2 = e2e - k1
                print(f"{H:>4} {T:>7} {S:>3} {k1:>8.3f} {e2e:>8.3f} "
                      f"{k2:>8.3f} {k1 / e2e:>9.3f}", flush=True)
                rows.append(dict(H=H, T=T, S=S, k1_ms=k1, e2e_ms=e2e,
                                 k2_ms=k2, k1_share=k1 / e2e))
            del p
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"[json] {a.json}")


if __name__ == "__main__":
    main()
