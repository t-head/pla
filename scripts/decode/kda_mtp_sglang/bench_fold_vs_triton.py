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

"""Fold: pla CUDA vs vendored sglang Triton on PPU —— A/B on identical inputs.

Isolates the conv scatter and pulls out the ssm fold, the largest part of commit, so that pla's CUDA
C++ implementation (``pla.decode.kda_mtp_sglang._C.fold_all_layers``) and sglang's upstream
Triton implementation (``_triton_fold.triton_fold_all_layers``) are cross-checked on the same inputs:

  1. each path is compared against the fp64 oracle;
  2. the two paths are diffed against each other;
  3. median latency comparison with 69 layers, mean accept 4.5 and L2 flush.

Usage (inside the PPU server container):
    python scripts/decode/kda_mtp_sglang/bench_fold_vs_triton.py
"""
# ruff: noqa: E501
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "private"))

import torch

from _common import Cfg, build_commit_inputs
from _harness import Timer
from _reference import fmt, fold_reference, score
from _triton_fold import triton_fold_all_layers

# The commit wrapper also runs conv scatter; here we want the fold-only path,
# so we call the underlying extension entry point directly (mirrors the source
# script's `ext.fold_all_layers(...)`).
from pla.decode.kda_mtp_sglang import _C as _ext


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


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count}")

    # ---- correctness: both backends vs fp64 oracle, and vs each other ----
    print("\n-- correctness (fold state) --")
    for bs, layers in ((4, 2), (8, 4)):
        cfg = Cfg(bs=bs, num_spec=7, heads=12, layers=layers)
        ci = build_commit_inputs(cfg)
        L = ci["rawv"].shape[3]
        want = fold_reference(ci, cfg)
        t_cuda = ci["temporal"].clone(); cuda_fold(ci, t_cuda)
        t_tri = ci["temporal"].clone(); tri_fold(ci, t_tri, L)
        torch.cuda.synchronize()
        s_c = score(t_cuda, want, f"cuda  bs={bs}")
        s_t = score(t_tri, want, f"trito bs={bs}")
        s_ct = score(t_cuda, t_tri, f"cuda-vs-triton bs={bs}")
        for s in (s_c, s_t, s_ct):
            print("  " + fmt(s))

    # ---- timing: 69 layers, mean accept 4.5, L2-flushed ----
    print("\n-- latency (69 layers, L2-flushed): cuda vs triton --")
    timer = Timer(reps=20, inner=10)
    print(f"  {'bs':>3}  {'cuda_us':>9}  {'triton_us':>10}  {'speedup':>8}")
    for bs in (1, 4, 8, 16, 32, 64):
        cfg = Cfg(bs=bs, num_spec=7, heads=12, layers=69)
        ci = build_commit_inputs(cfg)
        L = ci["rawv"].shape[3]
        tc = ci["temporal"].clone(); tt = ci["temporal"].clone()
        cuda_fold(ci, tc); tri_fold(ci, tt, L); torch.cuda.synchronize()
        rc = timer.run(lambda: cuda_fold(ci, tc))
        rt = timer.run(lambda: tri_fold(ci, tt, L))
        cu, tr = rc["median_ms"] * 1e3, rt["median_ms"] * 1e3
        print(f"  {bs:>3}  {cu:9.1f}  {tr:10.1f}  {tr / cu:7.2f}x")


if __name__ == "__main__":
    main()
