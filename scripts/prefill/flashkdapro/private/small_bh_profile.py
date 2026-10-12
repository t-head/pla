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

"""Single-shape FlashKDApro driver for small-BH ACU profiles and A/B timing.

The existing ``cp_profile_one.py`` only builds equal-length sequences.  This
driver accepts exact ragged lengths so the dominant-chain tail seen in Kimi-K3
can be reproduced.  It calls the compiled extension directly with preallocated
outputs; allocation and FLA comparison belong to the E2E harness, not a kernel
profile.
"""

import argparse
import os
import statistics
import time

import torch


D = 128
LOWER_BOUND = -5.0
DECAY_SHIFT = {"fast": 0.0, "mid": -3.0, "slow": -4.5}


def set_env(name, value):
    key = f"PLA_FLASHKDA_{name}"
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(value)


def make_inputs(seq_lens, heads, with_state, state_dtype, decay):
    torch.manual_seed(1)
    tokens = sum(seq_lens)
    rnd = lambda *shape: torch.randn(
        *shape, device="cuda", dtype=torch.float32
    )
    q = rnd(tokens, heads, D).bfloat16()
    k = rnd(tokens, heads, D).bfloat16()
    v = rnd(tokens, heads, D).bfloat16()
    g = (rnd(tokens, heads, D) * 0.3 + DECAY_SHIFT[decay]).bfloat16()
    beta = rnd(tokens, heads).bfloat16()
    initial_state = (
        torch.zeros(
            len(seq_lens), heads, D, D, device="cuda", dtype=state_dtype
        )
        if with_state
        else None
    )
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu = [0]
        for length in seq_lens:
            cu.append(cu[-1] + length)
        cu_seqlens = torch.tensor(cu, device="cuda", dtype=torch.int64)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": torch.zeros(heads, device="cuda", dtype=torch.float32),
        "dt_bias": torch.zeros(
            heads, D, device="cuda", dtype=torch.float32
        ),
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="31801,967")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--state", action="store_true")
    parser.add_argument(
        "--state-dtype", choices=("bf16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--decay", choices=tuple(DECAY_SHIFT), default="slow"
    )
    parser.add_argument("--cp-mode", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--cp-bv", type=int, choices=(16, 32, 128))
    parser.add_argument("--cp-mult", type=int)
    parser.add_argument("--cp-seg-tiles", type=int)
    parser.add_argument("--k2-dvsplit", type=int, choices=(0, 16, 32))
    parser.add_argument("--k2-dksplit", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--k2-dkleader", type=int, choices=(0, 1))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--time", action="store_true", help="print per-call wall-time samples"
    )
    args = parser.parse_args()

    seq_lens = tuple(int(x) for x in args.seq_lens.split(",") if x)
    if not seq_lens or any(x <= 0 for x in seq_lens):
        parser.error("--seq-lens must contain positive comma-separated lengths")

    set_env("CP", args.cp_mode)
    set_env("CP_BV", args.cp_bv)
    set_env("CP_MULT", args.cp_mult)
    set_env("CP_SEG_TILES", args.cp_seg_tiles)
    set_env("K2_DVSPLIT", args.k2_dvsplit)
    set_env("K2_DKSPLIT", args.k2_dksplit)
    set_env("K2_DKLEADER", args.k2_dkleader)

    from pla.prefill.flashkdapro import _C

    state_dtype = (
        torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
    )
    inputs = make_inputs(
        seq_lens, args.heads, args.state, state_dtype, args.decay
    )
    tokens = sum(seq_lens)
    out = torch.empty(
        tokens, args.heads, D, device="cuda", dtype=torch.bfloat16
    )
    final_state = torch.empty(
        len(seq_lens), args.heads, D, D, device="cuda", dtype=state_dtype
    )

    def call():
        _C.flashkda_fwd(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            out,
            1.0 / (D**0.5),
            LOWER_BOUND,
            inputs["initial_state"],
            final_state,
            inputs["cu_seqlens"],
        )

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    plan = _C.flashkda_cp_plan(tokens, args.heads, inputs["cu_seqlens"])
    summary = (
        f"[done] H={args.heads} N={len(seq_lens)} T={tokens} "
        f"seq={list(seq_lens)} state={int(args.state)} "
        f"state_dtype={args.state_dtype} decay={args.decay} "
        f"cp={args.cp_mode} engage={int(plan['engage'])} "
        f"n_seg={plan['n_seg']} bv={plan['bv']} iters={args.iters}"
    )
    if args.time:
        summary += (
            f" median_ms={statistics.median(samples):.4f} "
            f"samples_ms={[round(x, 4) for x in samples]}"
        )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
