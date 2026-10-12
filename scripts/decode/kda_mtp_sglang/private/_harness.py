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

"""Timing harness built for comparing kernel variants, not for headline numbers.

Choices that matter for that goal:
  * Wall-clock timing around a batch of `inner` whole-API calls. PPU device
    events can wedge a process when timing a single kernel path; the one device
    synchronization is therefore at the end of each timed batch.
  * An L2 flush before every timed batch. The state/ring working set at small
    bs fits in H20's 60MB L2, so without a flush a rep-loop measures an
    unrealistically warm cache and hides exactly the memory-side changes we
    are trying to evaluate.
  * Median plus P10/P90 over independent reps. A single mean cannot tell a
    real 5% win from clock drift, and this box is shared.
"""

import json
import os
import statistics
import time

import torch

L2_FLUSH_MB = 128


class Timer:
    def __init__(self, reps=30, inner=10, warmup=5, flush_l2=True):
        self.reps, self.inner, self.warmup = reps, inner, warmup
        self.flush_l2 = flush_l2
        self._scratch = (
            torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32,
                        device="cuda")
            if flush_l2 else None
        )

    def _flush(self):
        if self._scratch is not None:
            self._scratch.zero_()

    def run(self, fn) -> dict:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        samples = []
        for _ in range(self.reps):
            self._flush()
            torch.cuda.synchronize()
            beg = time.perf_counter()
            for _ in range(self.inner):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - beg) * 1e3 / self.inner)

        samples.sort()
        n = len(samples)
        return dict(
            median_ms=statistics.median(samples),
            p10_ms=samples[max(0, int(0.10 * n) - 1)],
            p90_ms=samples[min(n - 1, int(0.90 * n))],
            min_ms=samples[0],
            spread_pct=(samples[min(n - 1, int(0.90 * n))] - samples[0])
            / max(samples[0], 1e-9) * 100.0,
            reps=self.reps, inner=self.inner,
        )


def clocks() -> dict:
    """SM clock + throttle reasons; a big spread usually shows up here first."""
    try:
        import subprocess
        q = ("clocks.sm,clocks.max.sm,temperature.gpu,power.draw,"
             "clocks_throttle_reasons.active")
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            text=True, timeout=10).strip().splitlines()[0]
        f = [x.strip() for x in raw.split(",")]
        return dict(sm_mhz=f[0], max_mhz=f[1], temp_c=f[2], power_w=f[3],
                    throttle=f[4])
    except Exception as e:
        return dict(error=str(e))


def save(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(payload)
    payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = torch.cuda.get_device_properties(0)
    payload["device"] = dict(name=p.name, sm=f"{p.major}.{p.minor}",
                             sms=p.multi_processor_count)
    payload["clocks"] = clocks()
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nsaved -> {path}")


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def table(rows, cols, title=""):
    if title:
        print("\n" + title)
    w = [max(len(str(r[c])) for r in rows + [{c: c for c in cols}]) for c in cols]
    print("  " + "  ".join(f"{c:>{w[i]}}" for i, c in enumerate(cols)))
    print("  " + "  ".join("-" * w[i] for i in range(len(cols))))
    for r in rows:
        print("  " + "  ".join(f"{str(r[c]):>{w[i]}}" for i, c in enumerate(cols)))
