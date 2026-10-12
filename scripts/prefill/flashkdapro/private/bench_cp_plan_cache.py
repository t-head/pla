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

"""OPT-05 wall-clock A/B for uncached device planning vs trusted cached plan."""

from __future__ import annotations

import os
import statistics
import time

import torch

from pla.prefill.flashkdapro import _C
from pla.prefill.flashkdapro import flashkda_fwd


def make_case():
    torch.manual_seed(20260826)
    T, H, D = 8192, 12, 128
    offsets = (0, 4999, 8191)
    q = (torch.randn(T, H, D, device="cuda") * 0.05).bfloat16()
    k = (torch.randn_like(q.float()) * 0.05).bfloat16()
    v = (torch.randn_like(q.float()) * 0.05).bfloat16()
    g = (torch.randn_like(q.float()) * 0.05).bfloat16()
    beta = (torch.randn(T, H, device="cuda") * 0.05).bfloat16()
    A_log = torch.linspace(-0.4, 0.2, H, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(H, D, device="cuda", dtype=torch.float32) * 0.05
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    out = torch.empty_like(v)

    def call(raw_offsets):
        flashkda_fwd(
            q, k, v, g, beta, A_log, dt_bias, out,
            cu_seqlens=cu,
            raw_cu_seqlens=raw_offsets,
            final_state=None,
        )
        return out

    return offsets, call


def measure(call, offsets, cached: bool, warmup: int = 5, repeats: int = 30):
    os.environ["PLA_FLASHKDA_CP"] = "2"
    os.environ["PLA_FLASHKDA_CP_PLAN_CACHE_SIZE"] = "4" if cached else "0"
    _C.flashkda_cp_plan_cache_clear()
    raw = offsets if cached else None
    for _ in range(warmup):
        call(raw)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call(raw)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), dict(_C.flashkda_cp_plan_cache_stats())


def main():
    offsets, call = make_case()

    os.environ["PLA_FLASHKDA_CP"] = "2"
    os.environ["PLA_FLASHKDA_CP_PLAN_CACHE_SIZE"] = "0"
    baseline = call(None).clone()
    os.environ["PLA_FLASHKDA_CP_PLAN_CACHE_SIZE"] = "4"
    _C.flashkda_cp_plan_cache_clear()
    candidate = call(offsets).clone()
    torch.cuda.synchronize()
    if not torch.equal(candidate, baseline):
        raise AssertionError("cached and uncached CP outputs differ")

    b_first, b_stats_first = measure(call, offsets, cached=False)
    c_second, c_stats_second = measure(call, offsets, cached=True)
    c_first, c_stats_first = measure(call, offsets, cached=True)
    b_second, b_stats_second = measure(call, offsets, cached=False)
    b_mean = (b_first + b_second) / 2
    c_mean = (c_first + c_second) / 2
    print(
        f"baseline={b_mean:.4f} ms candidate={c_mean:.4f} ms "
        f"delta={(c_mean / b_mean - 1) * 100:.2f}% "
        f"speedup={b_mean / c_mean:.3f}x "
        f"AB=({b_first:.4f},{c_second:.4f}) "
        f"BA=({c_first:.4f},{b_second:.4f})"
    )
    print("baseline_stats", b_stats_first, b_stats_second)
    print("candidate_stats", c_stats_first, c_stats_second)


if __name__ == "__main__":
    main()
