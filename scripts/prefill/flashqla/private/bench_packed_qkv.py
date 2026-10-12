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

"""OPT-07 end-to-end A/B for SGLang's split path vs packed QKV PLA."""

from __future__ import annotations

import math
import statistics
import time

import torch

try:
    from .packed_qkv_split import split_qkv
except ImportError:
    from packed_qkv_split import split_qkv

from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

try:
    from pla.prefill.flashqla import gdn_prefill_packed_qkv_fwd
except (ImportError, AttributeError):
    # ``generalized_prefill_replay --gdn-path split`` reuses ``split_qkv``
    # with wheels predating the packed entry point.  The benchmark itself still
    # fails explicitly below when its packed candidate is requested.
    gdn_prefill_packed_qkv_fwd = None


D = 128


def make_case(T: int, offsets: tuple[int, ...], H: int, Hg: int):
    torch.manual_seed(20260826 + T + H)
    width = (2 * Hg + H) * D
    # Non-zero offset and wider token pitch cover post-convolution workspace
    # views while retaining the feature-contiguous framework contract.
    storage = torch.randn(T, width + 64, device="cuda", dtype=torch.bfloat16)
    packed = storage[:, 17:17 + width]
    ba = torch.randn(T, 2 * H, device="cuda", dtype=torch.bfloat16)
    b, a = ba.split(H, dim=-1)
    A_log = torch.linspace(-0.4, 0.2, H, device="cuda", dtype=torch.float32)
    dt_bias = torch.linspace(-0.2, 0.1, H, device="cuda", dtype=torch.float32)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    num_sequences = len(offsets) - 1
    state_indices = torch.arange(
        1, num_sequences + 1, device="cuda", dtype=torch.int32)
    initial_pool = (
        torch.randn(
            num_sequences + 2, H, D, D,
            device="cuda", dtype=torch.float32,
        )
        * 0.01
    )
    baseline_pool = initial_pool.clone()
    candidate_pool = initial_pool.clone()
    common = dict(
        scale=1.0 / math.sqrt(D),
        cu_seqlens=cu,
        raw_cu_seqlens=offsets,
        output_final_state=False,
        auto_cp=False,
    )

    def baseline():
        q, k, v = split_qkv(packed, H, Hg)
        return chunk_gated_delta_rule_fwd(
            q,
            k,
            v,
            None,
            None,
            normalize_qk=True,
            raw_gate_a=a,
            raw_beta_b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=baseline_pool,
            initial_state_indices=state_indices,
            **common,
        )

    def candidate():
        if gdn_prefill_packed_qkv_fwd is None:
            raise RuntimeError(
                "the installed PLA wheel does not expose packed-QKV GDN prefill"
            )
        return gdn_prefill_packed_qkv_fwd(
            packed,
            a,
            b,
            A_log,
            dt_bias,
            cu,
            num_qk_heads=Hg,
            num_v_heads=H,
            scale=common["scale"],
            initial_state=candidate_pool,
            initial_state_indices=state_indices,
            raw_cu_seqlens=offsets,
            output_final_state=False,
            auto_cp=False,
        )

    return baseline, candidate, baseline_pool, candidate_pool


def measure(fn, warmup: int = 5, repeats: int = 30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
        del result
    return statistics.median(samples)


def measure_peak_allocated(fn):
    """Return peak live allocation above the pre-call level."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del result
    return peak


def run_case(name: str, T: int, offsets: tuple[int, ...], H: int, Hg: int):
    baseline, candidate, baseline_pool, candidate_pool = make_case(
        T, offsets, H, Hg)
    ref = baseline()[2]
    actual = candidate()[2]
    if not torch.equal(actual, ref):
        diff = (actual.float() - ref.float()).abs().max().item()
        raise AssertionError(f"{name}: packed output differs; max_abs={diff}")
    if not torch.equal(candidate_pool, baseline_pool):
        diff = (candidate_pool - baseline_pool).abs().max().item()
        raise AssertionError(f"{name}: packed state differs; max_abs={diff}")

    b_first = measure(baseline)
    c_second = measure(candidate)
    c_first = measure(candidate)
    b_second = measure(baseline)
    b_peak = measure_peak_allocated(baseline)
    c_peak = measure_peak_allocated(candidate)
    b_mean = (b_first + b_second) / 2
    c_mean = (c_first + c_second) / 2
    saved_mib = T * (2 * Hg + H) * D * 2 * 2 / (1024 * 1024)
    print(
        f"{name} T={T} real={offsets[-1]} H={H} Hg={Hg} "
        f"baseline={b_mean:.4f} ms candidate={c_mean:.4f} ms "
        f"delta={(c_mean / b_mean - 1) * 100:.2f}% "
        f"speedup={b_mean / c_mean:.3f}x qkv_traffic_saved={saved_mib:.2f} MiB "
        f"peak_alloc=({b_peak / (1024 * 1024):.2f},"
        f"{c_peak / (1024 * 1024):.2f}) MiB "
        f"AB=({b_first:.4f},{c_second:.4f}) "
        f"BA=({c_first:.4f},{b_second:.4f})"
    )


def main():
    run_case("ragged-padded", 668, (0, 127, 384, 661), 32, 16)
    run_case("single-padded", 2206, (0, 2205), 32, 16)


if __name__ == "__main__":
    main()
