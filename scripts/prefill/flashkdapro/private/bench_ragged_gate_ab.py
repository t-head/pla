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

"""A/B validation of the generalized varlen dispatch gate on PPU1.5.

Runs against the *existing* ppu_port binary (no rebuild): the register-resident
(nows) path + BV32 dvsplit can be forced per-launch via PLA_FLASHKDA_K2_NOWS /
PLA_FLASHKDA_K2_DVSPLIT (both read by getenv at launch time).  For each shape we
compare:

  default      -> whatever the live hardcoded gate selects
  generalized  -> PLA_FLASHKDA_K2_NOWS=1 PLA_FLASHKDA_K2_DVSPLIT=32  (what my new gate
                  would route these small-batch long-segment ragged cases into)
  FLA          -> fla.ops.kda.chunk_kda (correctness + perf anchor)

This isolates the *routing decision* the generalized gate makes.  Correctness
is judged by err_ratio(ppu, FLA) and gen-vs-default equivalence (same math,
different launch/tiling -> must agree to ~1e-2).

Run inside hp_ppu15_flashqla_kda, tests on cwd sys.path:
  FLA_SOURCE_PATH=/mnt/ssd/hpwu/flash-linear-attention \
  PYTHONPATH=tests:.:$FLA_SOURCE_PATH CUDA_VISIBLE_DEVICES=2 \
  python3 bench_ragged_gate_ab.py
"""
import math
import os
import sys

import torch

# reuse the official bench helpers (same input construction, timing, fla path)
from bench_vs_fla import (
    D,
    LOWER_BOUND,
    bench_fn,
    err_ratio,
    run_fla_chunk_kda,
    run_ppu_flashkda,
)

DEV = "cuda"

WARMUP = 20
ITERS = 60
REPEATS = 3

# (name, seq_lens).  avg_seg = sum/len.  is_varlen = len>1.
SHAPES = [
    # --- newly-generalized region: small batch, long segments (avg_seg >= 512) ---
    ("b2-seg4096", [4096, 4096]),                       # N=2 avg=4096 T=8192
    ("b2-seg2048", [2048, 2048]),                       # N=2 avg=2048 T=4096
    ("b3-seg2048", [2048, 2048, 2048]),                 # N=3 avg=2048 T=6144
    ("b4-seg2048", [2048, 2048, 2048, 2048]),           # N=4 avg=2048 T=8192
    ("b4-seg1024", [1024, 1024, 1024, 1024]),           # N=4 avg=1024 T=4096
    ("b6-official", [1300, 547, 2048, 963, 271, 3063]),  # N=6 avg=1365 T=8192
    ("b6-seg1024", [1024] * 6),                          # N=6 avg=1024 T=6144
    # --- negative controls: short segments (avg_seg < 512) -> should NOT route ---
    ("neg-b4-seg384", [384, 384, 384, 384]),             # N=4 avg=384  T=1536
    ("neg-b8-seg256", [256] * 8),                        # N=8 avg=256  T=2048
    # --- danger zone: high N, short segments (plain grid already large;
    #     N*H>=40 so default already uses nows -> tests dvsplit over-split) ---
    ("hiN-b16-seg512", [512] * 16),                      # N=16 avg=512 T=8192
    ("hiN-b16-seg256", [256] * 16),                      # N=16 avg=256 T=4096
    ("hiN-b32-seg128", [128] * 32),                      # N=32 avg=128 T=4096
    ("hiN-b32-seg256", [256] * 32),                      # N=32 avg=256 T=8192
    # --- over-subscription zone: extreme N (BV32 grid = N*H*4 >> 39 CUs) ---
    ("xN-b64-seg128", [128] * 64),                       # N=64 avg=128 T=8192
    ("xN-b64-seg256", [256] * 64),                       # N=64 avg=256 T=16384
    ("xN-b128-seg64", [64] * 128),                       # N=128 avg=64 T=8192
]

HEADS = [6, 12]


def build_inputs(seq_lens, H):
    T_total = sum(seq_lens)
    N = len(seq_lens)
    scale = 1.0 / math.sqrt(D)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.long, device=DEV,
    )
    torch.manual_seed(1)
    q = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    k = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    v = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    g = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=DEV)
    beta = torch.randn((1, T_total, H), dtype=torch.bfloat16, device=DEV)
    A_log = torch.rand(H, dtype=torch.float32, device=DEV)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=DEV)
    return q, k, v, g, beta, A_log, dt_bias, scale, cu_seqlens, N, T_total


def set_gate(mode):
    for key in ("PLA_FLASHKDA_K2_NOWS", "PLA_FLASHKDA_K2_DVSPLIT"):
        os.environ.pop(key, None)
    if mode == "generalized":
        os.environ["PLA_FLASHKDA_K2_NOWS"] = "1"
        os.environ["PLA_FLASHKDA_K2_DVSPLIT"] = "32"


def median_ms(fn):
    mean, mn, mx = bench_fn(fn, WARMUP, ITERS, REPEATS)
    return mean  # bench_fn returns (mean, min, max); use mean as the stable stat


def main():
    print(f"device: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"torch={torch.__version__}  warmup={WARMUP} iters={ITERS} repeats={REPEATS}")
    hdr = (f"{'shape':<14}{'H':>3}{'N':>3}{'T':>7}{'avgseg':>8}"
           f"{'default':>10}{'general':>10}{'fla':>9}"
           f"{'gen/def':>9}{'fla/gen':>9}"
           f"{'errGvF':>10}{'errDvF':>10}{'errGvD':>9}")
    print(hdr)
    print("-" * len(hdr))

    for name, seq_lens in SHAPES:
        for H in HEADS:
            (q, k, v, g, beta, A_log, dt_bias, scale, cu, N, T) = build_inputs(seq_lens, H)
            avg = T // N

            def call():
                return run_ppu_flashkda(q, k, v, g, beta, A_log, dt_bias, scale,
                                        None, cu)

            # default gate
            set_gate("default")
            o_def, _ = call()
            t_def = median_ms(call)

            # generalized (forced nows + dvsplit32)
            set_gate("generalized")
            o_gen, _ = call()
            t_gen = median_ms(call)

            # fla anchor
            set_gate("default")
            o_fla, _ = run_fla_chunk_kda(q, k, v, g, beta, A_log, dt_bias, scale,
                                         None, extra={"cu_seqlens": cu})
            t_fla = median_ms(lambda: run_fla_chunk_kda(
                q, k, v, g, beta, A_log, dt_bias, scale, None,
                extra={"cu_seqlens": cu}))

            torch.cuda.synchronize()
            gv = o_gen[0].float().cpu().reshape(-1)
            dv = o_def[0].float().cpu().reshape(-1)
            fv = o_fla[0].float().cpu().reshape(-1)
            e_gvf = err_ratio(gv, fv)
            e_dvf = err_ratio(dv, fv)
            e_gvd = err_ratio(gv, dv)

            print(f"{name:<14}{H:>3}{N:>3}{T:>7}{avg:>8}"
                  f"{t_def:>10.4f}{t_gen:>10.4f}{t_fla:>9.4f}"
                  f"{t_def / t_gen:>9.2f}{t_fla / t_gen:>9.2f}"
                  f"{e_gvf:>10.2e}{e_dvf:>10.2e}{e_gvd:>9.2e}")


if __name__ == "__main__":
    main()
