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

"""Minimal single-shape driver for acu profiling of flashkdapro forward.

`flashkda_prepare` cannot be used to time Kernel1: it zero-fills and then makes
six ``.contiguous()`` copies of the whole workspace (tens of MB), which dwarfs
the kernel itself.  So the K1/K2 split is obtained from a profiler instead --
this script just issues a fixed number of forward calls for one (H, T, S) shape.

  acu --set full --import-source yes -o /tmp/rep -- python3 \\
      cp_profile_one.py --heads 12 --tokens 32768 --splits 16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from pla.prefill.flashkdapro import flashkda_fwd
from cp_stage_timing import inputs

D = 128
CHUNK = 16
LOWER_BOUND = -5.0
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--decay", choices=["fast", "mid", "slow"],
                    default="slow")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--fixed", action="store_true",
                    help="omit cu_seqlens (requires --splits 1)")
    ap.add_argument("--cp-mode", type=int, choices=[0, 1, 2], default=1)
    ap.add_argument("--mult", type=int, default=0,
                    help="forced segment multiplier; 0 keeps AUTO")
    a = ap.parse_args()

    H, T, S = a.heads, a.tokens, a.splits
    if a.fixed and S != 1:
        ap.error("--fixed requires --splits 1")
    x = inputs(T, H, a.state, S, a.decay)
    cu = (None if a.fixed else
          torch.arange(0, T + 1, T // S, dtype=torch.int64, device=DEV))
    os.environ["PLA_FLASHKDA_CP"] = str(a.cp_mode)
    if a.mult:
        os.environ["PLA_FLASHKDA_CP_MULT"] = str(a.mult)
    else:
        os.environ.pop("PLA_FLASHKDA_CP_MULT", None)

    # out is a caller-allocated out-param; reuse one buffer across the loops so
    # the profile stays a kernel profile rather than an allocator one.
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device=DEV)
    for _ in range(a.warmup):
        flashkda_fwd(x["q"], x["k"], x["v"], x["g"], x["beta"],
                     x["A_log"], x["dt_bias"], out, initial_state=x["h0"],
                     lower_bound=LOWER_BOUND, cu_seqlens=cu)
    torch.cuda.synchronize()
    for _ in range(a.iters):
        flashkda_fwd(x["q"], x["k"], x["v"], x["g"], x["beta"],
                     x["A_log"], x["dt_bias"], out, initial_state=x["h0"],
                     lower_bound=LOWER_BOUND, cu_seqlens=cu)
    torch.cuda.synchronize()
    print(f"[done] H={H} T={T} S={S} varlen={int(cu is not None)} "
          f"decay={a.decay} state={int(a.state)} cp={a.cp_mode} "
          f"mult={a.mult or 'auto'} iters={a.iters}", flush=True)


if __name__ == "__main__":
    main()
