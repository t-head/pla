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

"""Canonical FlashQLA CP acceptance against serial and FP32 references."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from e2e_acc_full import run_e2e
from pla.prefill.flashqla.ops import (
    CHUNK_SIZE as OPS_CHUNK_SIZE,
    SUPPORTED_HEAD_CONFIGS,
    _chunk_indices_cache,
    _prepare_chunk_indices,
    _resolve_raw_cu,
)
from pla.prefill.flashqla.private.cp_preprocess import (
    CHUNK_SIZE,
    TC01_TFU_MODEL_ENV,
    TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE,
    _TC02_SINGLE_SEQ_CP_MIN_TOKENS_BY_TOPOLOGY,
    _cp_plan_cache,
    _head_shards_from_compute_capability,
    calc_cp_seqs,
)


@contextmanager
def _legacy_tc01_planner(enabled: bool):
    """Pin legacy tc01 assertions without changing the E2E model under test."""
    had_value = TC01_TFU_MODEL_ENV in os.environ
    old_value = os.environ.get(TC01_TFU_MODEL_ENV)
    if enabled:
        os.environ[TC01_TFU_MODEL_ENV] = "0"
    try:
        yield
    finally:
        if enabled:
            if had_value:
                os.environ[TC01_TFU_MODEL_ENV] = old_value
            else:
                os.environ.pop(TC01_TFU_MODEL_ENV, None)


CASES = (
    # seq_len, dtype, h0, gate mode, expected fallback
    (8192, torch.float16, False, "fast", "zero"),
    (8193, torch.bfloat16, True, "fast", "zero"),
    (8192, torch.float16, True, "slow", "nonzero"),
    (8193, torch.bfloat16, True, "slow", "nonzero"),
)

PLANNER_CASES = (
    # sequence lengths, value heads, chunk size, expected auto-CP decision
    ((17,), 16, 32, False),
    # Single-sequence length boundaries, including H=16 segment-size changes.
    ((1535,), 4, 32, False),
    ((1536,), 4, 32, True),
    ((1535,), 6, 32, False),
    ((1536,), 6, 32, True),
    ((2047,), 8, 32, False),
    ((2048,), 8, 32, True),
    ((895,), 12, 32, False),
    ((896,), 12, 32, True),
    ((3071,), 16, 32, False),
    ((3072,), 16, 32, True),
    ((4416,), 16, 32, True),
    ((4448,), 16, 32, True),
    ((5823,), 16, 32, True),
    ((5824,), 16, 32, True),
    ((3583,), 24, 32, False),
    ((3584,), 24, 32, True),
    ((13999,), 32, 32, False),
    ((14000,), 32, 32, True),
    ((196607,), 48, 32, False),
    ((196608,), 48, 32, True),
    ((262144,), 64, 32, False),
    # Meeting the token threshold alone is insufficient when no split occurs.
    ((1024,), 16, 1024, False),
    # Packed multi-sequence occupancy boundaries.
    ((1024, 1024), 6, 32, False),
    ((1024, 1024), 16, 32, False),
    ((2048, 2048), 16, 32, False),
    # Post-epilogue tc02 low-grid islands: complete-API same-card sweeps show
    # that balanced CP now amortizes its preprocessing and shortens the fused
    # register chain. Keep representatives for every promoted topology.
    ((1647, 2716), 8, 32, True),
    ((315, 2077, 2265), 8, 32, True),
    ((508, 3356), 16, 32, True),
    ((6397, 1795), 24, 32, True),
    # Below the promoted H=8 total-token boundary, preserve the existing
    # profitable generic CP plan instead of replacing its segment geometry.
    # Likewise, do not replace the coarse H=24/Be<=1.2 plan.
    ((1235, 2716), 8, 32, True),
    ((6957, 1235), 24, 32, True),
    # Dense packed grids that older raw-wave rules enabled but production-like
    # slow-decay API E2E showed to be slower after CP preprocessing.
    ((2048,) * 7, 6, 32, False),
    ((2048,) * 16, 6, 32, False),
    ((4096,) * 9, 6, 32, False),
    ((4096,) * 10, 6, 32, False),
    ((4096,) * 18, 6, 32, False),
    ((6144,) * 20, 6, 32, False),
    ((4096,) * 26, 6, 32, False),
    ((4096,) * 27, 6, 32, False),
    ((16384,) * 13, 6, 32, False),
    ((16384,) * 10, 6, 32, False),
    ((16384,) * 27, 6, 32, False),
    ((16384,) * 32, 6, 32, False),
    ((16384,) * 8, 16, 32, False),
    # Profitable and rejected production slow-decay/SWA boundaries.  Neighboring
    # cases pin the strict/equal 2x path rules, uneven Be<2 exception, Be*H<=24
    # long-sequence rule, and the final Be*H<=30 range.
    ((2048, 512), 4, 32, True),
    ((2048, 2048), 4, 32, True),
    ((3072, 3072), 4, 32, True),
    ((2048, 512), 6, 32, True),
    ((2048, 2048), 6, 32, False),
    ((3072, 3072), 6, 32, True),
    ((3072,) * 3, 6, 32, False),
    ((4096,) * 3, 6, 32, True),
    ((4096,) * 4, 6, 32, True),
    ((4096, 4096, 1024, 1024), 6, 32, True),
    ((8192,) * 4, 6, 32, True),
    ((8192,) * 8, 6, 32, False),
    ((8192,) * 2, 12, 32, True),
    ((8192,) * 4, 12, 32, False),
    ((4096, 1024), 12, 32, True),
    ((3072, 768), 8, 32, True),
    ((4096, 4096), 8, 32, True),
    ((8192, 2048), 16, 32, True),
    ((8192,) * 2, 16, 32, False),
    ((2783, 18349, 11636), 16, 32, True),
    ((8192, 2048), 24, 32, True),
    ((5963, 2229), 24, 32, False),
    ((994, 6144, 1054), 24, 32, False),
    ((994, 6145, 1053), 24, 32, True),
    ((7950, 242), 24, 32, True),
    ((16384, 4096), 24, 32, True),
    ((16384, 8192), 24, 32, True),
    ((16384, 4096), 32, 32, False),
)

GROUPED_PLANNER_CASES = (
    # Same-card public-API ABBA floors are topology-specific. Pin L-1/L/L+1
    # explicitly rather than deriving expectations from the production table.
    ((447,), 4, 2, 32, False),
    ((448,), 4, 2, 32, True),
    ((449,), 4, 2, 32, True),
    ((415,), 6, 2, 32, False),
    ((416,), 6, 2, 32, True),
    ((417,), 6, 2, 32, True),
    ((423,), 8, 2, 32, False),
    ((424,), 8, 2, 32, True),
    ((425,), 8, 2, 32, True),
    ((423,), 8, 4, 32, False),
    ((424,), 8, 4, 32, True),
    ((425,), 8, 4, 32, True),
    ((527,), 12, 4, 32, False),
    ((528,), 12, 4, 32, True),
    ((529,), 12, 4, 32, True),
    ((1408,), 16, 2, 32, False),
    ((1409,), 16, 2, 32, True),
    ((1410,), 16, 2, 32, True),
    ((1408,), 16, 4, 32, False),
    ((1409,), 16, 4, 32, True),
    ((1410,), 16, 4, 32, True),
    ((1408,), 16, 8, 32, False),
    ((1409,), 16, 8, 32, True),
    ((1410,), 16, 8, 32, True),
    ((2048,), 24, 8, 32, False),
    ((2049,), 24, 8, 32, True),
    ((2050,), 24, 8, 32, True),
    ((12519,), 32, 4, 32, False),
    ((12520,), 32, 4, 32, True),
    ((12521,), 32, 4, 32, True),
    ((12519,), 32, 8, 32, False),
    ((12520,), 32, 8, 32, True),
    ((12521,), 32, 8, 32, True),
    ((9439,), 32, 16, 32, False),
    ((9440,), 32, 16, 32, True),
    ((9441,), 32, 16, 32, True),
    ((2079,), 48, 16, 32, False),
    ((2080,), 48, 16, 32, True),
    ((2081,), 48, 16, 32, True),
    ((4863,), 64, 8, 32, False),
    ((4864,), 64, 8, 32, True),
    ((4865,), 64, 8, 32, True),
    ((5119,), 64, 16, 32, False),
    ((5120,), 64, 16, 32, True),
    ((5121,), 64, 16, 32, True),
    ((3455,), 128, 16, 32, False),
    ((3456,), 128, 16, 32, True),
    ((3457,), 128, 16, 32, True),
    # The floor is monotonic; the calibrated H32/Hg16 interval uses seven
    # balanced segments throughout these neighboring lengths.
    ((10239,), 32, 16, 32, True),
    ((10240,), 32, 16, 32, True),
    ((10241,), 32, 16, 32, True),
    # CS64 is outside the topology map and retains the legacy H-only floor.
    ((10240,), 32, 16, 64, False),
    ((13999,), 32, 16, 64, False),
    ((14000,), 32, 16, 64, True),
)

GROUPED_SEGMENT_CASES = (
    ((10239,), 32, 16, 7),
    ((10240,), 32, 16, 7),
    ((10241,), 32, 16, 7),
)

SEGMENT_CASES = (
    # lengths, value heads, expected number of CP segments
    ((6277, 1915), 8, 9),
    ((1647, 2716), 8, 8),
    ((2882, 2716), 8, 9),
    ((315, 2077, 2265), 8, 9),
    ((508, 3356), 16, 4),
    ((6397, 1795), 24, 9),
    ((4096, 1024), 12, 5),
    # Doubling would leave fewer than 39 fused_fwd tasks, so keep the base plan.
    ((3072, 768), 8, 8),
    ((7950, 242), 24, 7),
)

# tc01 has 64 CUs and launches two DV CTA shards per value head.  These cases
# pin the verified fixed-segment/wave planner independently of tc02 above.
TC01_PLANNER_CASES = (
    ((17,), 16, 32, False),
    ((2527,), 4, 32, True),
    ((2528,), 4, 32, True),
    ((3039,), 6, 32, True),
    ((3040,), 6, 32, True),
    ((9215,), 8, 32, True),
    ((9216,), 8, 32, True),
    ((15900, 484), 8, 32, True),
    ((27826, 4942), 8, 32, True),
    ((1988, 13107, 1289), 8, 32, True),
    ((32767,), 12, 32, True),
    ((32768,), 12, 32, True),
    ((131072,), 16, 32, True),
    ((4095,), 32, 32, False),
    ((4096,), 32, 32, False),
    ((16383,), 64, 32, False),
    ((16384,), 64, 32, False),
    ((1024,), 16, 1024, False),
    # tc01 H=16 single-sequence CP starts at 1824 tokens.  The shorter tail on
    # the same 57-chunk boundary does not amortize preprocessing.
    ((1823,), 16, 32, False),
    ((1824,), 16, 32, True),
    # Exact 2.5 raw batch-head waves round up to three without enough CP work
    # reduction; keep these packed occupancy holes serial.
    ((2048,) * 5, 16, 32, False),
    ((2048,) * 10, 8, 32, False),
    # Preserve the measured long, low-fill H=16/B=5 exception.  The first case
    # is exactly 8*total_chunks == 33*max_chunks; one extra chunk crosses back
    # into the serial occupancy hole.
    ((6144, 4800, 4800, 4800, 4800), 16, 32, True),
    ((6144, 4832, 4800, 4800, 4800), 16, 32, False),
    # H=8/B=2 stays split with five local segments; the sixth segment crosses
    # the measured overhead bound while the longest sequence is at most 48
    # chunks.
    ((896, 1440), 8, 32, True),
    ((1120, 1440), 8, 32, False),
    # Pin both sides of the measured H=8/B=5 local-cap transition.
    ((5597, 5273, 5750, 6593, 6482), 8, 32, False),
    ((7173, 5913, 6448, 7392, 5842), 8, 32, True),
    # H=24/B=3 needs at least a 6/5 reduction in wave-weighted chunk work.
    ((4846, 3995, 4356), 24, 32, False),
    ((10434, 9830, 10719), 24, 32, True),
    # A very short fourth sequence needs the same work reduction.  Preserve
    # both a low-fill positive case and the neighboring >22-chunk tail.
    ((508, 3356, 3659, 4196), 24, 32, False),
    ((11925, 9830, 10719, 294), 24, 32, True),
    ((896, 5913, 6448, 7392), 24, 32, True),
    ((1024, 1024), 6, 32, True),
    ((1024, 1024), 16, 32, False),
    ((2048, 2048), 16, 32, False),
    ((2048,) * 7, 6, 32, False),
    ((2048,) * 16, 6, 32, False),
    ((4096,) * 9, 6, 32, False),
    ((4096,) * 10, 6, 32, False),
    ((4096,) * 18, 6, 32, False),
    ((6144,) * 20, 6, 32, False),
    ((4096,) * 26, 6, 32, False),
    ((4096,) * 27, 6, 32, False),
    ((16384,) * 13, 6, 32, True),
    ((16384,) * 10, 6, 32, False),
    ((16384,) * 27, 6, 32, True),
    ((16384,) * 32, 6, 32, False),
    ((16384,) * 8, 16, 32, False),
    ((8192, 2048), 4, 32, True),
    ((8192, 2048), 6, 32, True),
    ((8192, 2048), 8, 32, True),
    ((16384, 4096), 6, 32, True),
    ((16384, 8192, 4096), 6, 32, True),
    ((16384, 8192, 4096), 8, 32, True),
    ((3180, 20971, 8617), 8, 32, True),
    ((3379, 22281, 7108), 8, 32, True),
    ((32768, 8192, 8192, 8192), 8, 32, True),
    ((32768, 26624), 8, 32, True),
)


def main() -> int:
    all_ok = True
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    head_shards = _head_shards_from_compute_capability(
        int(props.major), int(props.minor))
    is_tc01 = head_shards == 2
    print(
        f"  [planner model] capability={props.major}.{props.minor} "
        f"CUs={props.multi_processor_count} "
        f"architecture={'tc01' if is_tc01 else 'tc02'}"
    )
    calibrated_topologies = frozenset(
        (heads, key_heads)
        for heads, key_heads, chunk_size in
        _TC02_SINGLE_SEQ_CP_MIN_TOKENS_BY_TOPOLOGY
        if chunk_size == TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE
    )
    topology_coverage_ok = calibrated_topologies == SUPPORTED_HEAD_CONFIGS
    calibrated_chunk_size_ok = (
        CHUNK_SIZE == TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE and
        OPS_CHUNK_SIZE == TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE
    )
    all_ok &= topology_coverage_ok
    all_ok &= calibrated_chunk_size_ok
    print(
        "  [CP topology map] supported-config coverage "
        f"{'PASS' if topology_coverage_ok else 'FAIL'}; production CS "
        f"{'PASS' if calibrated_chunk_size_ok else 'FAIL'}"
    )
    # Equal-valued tensors with different identities must share a plan, while an
    # in-place value update must build a different plan.  This directly guards
    # the serving pattern that motivated the cache and its stale-metadata risk.
    _cp_plan_cache.clear()
    cp_cu_a = torch.tensor([0, 8192], dtype=torch.int32, device="cuda")
    cp_cu_b = cp_cu_a.clone()
    plan_a = calc_cp_seqs(cp_cu_a, 32, 16)
    plan_b = calc_cp_seqs(cp_cu_b, 32, 16)
    cp_cache_ok = (plan_a is plan_b and len(_cp_plan_cache) == 1)
    all_ok &= cp_cache_ok
    print(f"  [CP plan cache] value-key reuse {'PASS' if cp_cache_ok else 'FAIL'}")
    cp_cu_b[1] = 4096
    plan_c = calc_cp_seqs(cp_cu_b, 32, 16)
    cp_mutation_ok = plan_c is not plan_b
    all_ok &= cp_mutation_ok
    print(
        f"  [CP plan cache] in-place mutation "
        f"{'PASS' if cp_mutation_ok else 'FAIL'}")
    _cp_plan_cache.clear()
    cp_cu_grouped = torch.tensor(
        [0, 10240], dtype=torch.int32, device="cuda"
    )
    grouped_key_heads = (None, 4, 8, 16)
    with _legacy_tc01_planner(is_tc01):
        grouped_plans = [
            calc_cp_seqs(cp_cu_grouped, 32, 32, num_k_heads=key_heads)
            for key_heads in grouped_key_heads
        ]
    expected_grouped_decisions = (
        [False, False, False, False] if is_tc01
        else [False, False, False, True]
    )
    grouped_segments = [
        plan[1].numel() - 1 if plan[0] else 1 for plan in grouped_plans
    ]
    expected_grouped_segments = (
        [1, 1, 1, 1] if is_tc01 else [1, 1, 1, 7]
    )
    cp_grouped_cache_ok = (
        [plan[0] for plan in grouped_plans] == expected_grouped_decisions and
        grouped_segments == expected_grouped_segments and
        len({id(plan) for plan in grouped_plans}) == 4 and
        len(_cp_plan_cache) == 4
    )
    all_ok &= cp_grouped_cache_ok
    print(
        "  [CP plan cache] grouped-head key isolation "
        f"{'PASS' if cp_grouped_cache_ok else 'FAIL'}"
    )
    _cp_plan_cache.clear()
    _chunk_indices_cache.clear()
    cu_a = torch.tensor([0, 513, 1538], dtype=torch.int32, device="cuda")
    cu_b = cu_a.clone()
    indices_a = _prepare_chunk_indices(cu_a, 32)
    indices_b = _prepare_chunk_indices(cu_b, 32)
    indices_c = _prepare_chunk_indices(cu_b, 64)
    cache_ok = (indices_a is indices_b and indices_c is not indices_a and
                len(_chunk_indices_cache) == 2)
    all_ok &= cache_ok
    print(f"  [chunk cache] value-key reuse {'PASS' if cache_ok else 'FAIL'}")
    cached_raw, _ = _resolve_raw_cu(cu_b, 32)
    cu_b[1] = 257
    mutated_raw, _ = _resolve_raw_cu(cu_b, 32)
    indices_d = _prepare_chunk_indices(cu_b, 32)
    mutation_ok = (indices_d is not indices_b and
                   not torch.equal(indices_d, indices_b) and
                   cached_raw == (0, 513, 1538) and
                   mutated_raw == (0, 257, 1538))
    all_ok &= mutation_ok
    print(f"  [chunk cache] in-place mutation {'PASS' if mutation_ok else 'FAIL'}")
    with torch.inference_mode():
        cu_inference = torch.tensor(
            [0, 513, 1538], dtype=torch.int32, device="cuda")
        raw_inference = (0, 513, 1538)
        indices_e = _prepare_chunk_indices(
            cu_inference, 32, raw_inference)
        indices_f = _prepare_chunk_indices(
            cu_inference, 32, raw_inference)
    inference_ok = indices_e is indices_f
    all_ok &= inference_ok
    print(
        f"  [chunk cache] inference metadata "
        f"{'PASS' if inference_ok else 'FAIL'}")
    _chunk_indices_cache.clear()

    planner_cases = TC01_PLANNER_CASES if is_tc01 else PLANNER_CASES
    planner_passed = 0
    with _legacy_tc01_planner(is_tc01):
        for lengths, heads, chunk_size, expected in planner_cases:
            cu = torch.tensor(
                [0] + torch.tensor(lengths).cumsum(0).tolist(),
                dtype=torch.int32,
                device="cuda",
            )
            actual, _, _, _, _ = calc_cp_seqs(cu, chunk_size, heads)
            ok = actual == expected
            planner_passed += int(ok)
            all_ok &= ok
            print(
                f"  [planner] lengths={lengths!s:<20} H={heads:>3} "
                f"chunk={chunk_size:>4} CP={actual} expected={expected} "
                f"{'PASS' if ok else 'FAIL'}"
            )

    grouped_planner_passed = 0
    grouped_planner_cases = () if is_tc01 else GROUPED_PLANNER_CASES
    for lengths, heads, key_heads, chunk_size, expected in grouped_planner_cases:
        cu = torch.tensor(
            [0] + torch.tensor(lengths).cumsum(0).tolist(),
            dtype=torch.int32,
            device="cuda",
        )
        actual, _, _, _, _ = calc_cp_seqs(
            cu, chunk_size, heads, num_k_heads=key_heads
        )
        ok = actual == expected
        grouped_planner_passed += int(ok)
        all_ok &= ok
        print(
            f"  [grouped planner] lengths={lengths!s:<12} H={heads:>3} "
            f"Hg={key_heads:>3} chunk={chunk_size:>4} "
            f"CP={actual} expected={expected} "
            f"{'PASS' if ok else 'FAIL'}"
        )

    segment_cases = () if is_tc01 else SEGMENT_CASES
    segment_passed = 0
    for lengths, heads, expected_segments in segment_cases:
        cu = torch.tensor(
            [0] + torch.tensor(lengths).cumsum(0).tolist(),
            dtype=torch.int32,
            device="cuda",
        )
        use_cp, cp_cu, _, _, _ = calc_cp_seqs(cu, 32, heads)
        actual_segments = cp_cu.numel() - 1 if use_cp else len(lengths)
        ok = use_cp and actual_segments == expected_segments
        segment_passed += int(ok)
        all_ok &= ok
        print(
            f"  [segments] lengths={lengths!s:<20} H={heads:>3} "
            f"segments={actual_segments} expected={expected_segments} "
            f"{'PASS' if ok else 'FAIL'}"
        )

    grouped_segment_passed = 0
    grouped_segment_cases = () if is_tc01 else GROUPED_SEGMENT_CASES
    for lengths, heads, key_heads, expected_segments in grouped_segment_cases:
        cu = torch.tensor(
            [0] + torch.tensor(lengths).cumsum(0).tolist(),
            dtype=torch.int32,
            device="cuda",
        )
        use_cp, cp_cu, _, _, _ = calc_cp_seqs(
            cu, 32, heads, num_k_heads=key_heads
        )
        actual_segments = cp_cu.numel() - 1 if use_cp else len(lengths)
        ok = use_cp and actual_segments == expected_segments
        grouped_segment_passed += int(ok)
        all_ok &= ok
        print(
            f"  [grouped segments] lengths={lengths!s:<12} H={heads:>3} "
            f"Hg={key_heads:>3} segments={actual_segments} "
            f"expected={expected_segments} {'PASS' if ok else 'FAIL'}"
        )

    total = 0
    passed = 0
    test_h, test_hg = (6, 2) if is_tc01 else (16, 8)
    for seed in (42, 123):
        for seq_len, dtype, use_h0, gate_mode, expected_fallback in CASES:
            ok = run_e2e(
                seq_len,
                H=test_h,
                Hg=test_hg,
                use_h0=use_h0,
                fast_decay=gate_mode == "fast",
                slow_decay=gate_mode == "slow",
                dtype=dtype,
                compare_no_cp=True,
                expected_fallback=expected_fallback,
                seed=seed,
            )
            total += 1
            passed += int(ok)
            all_ok &= ok

    print(f"CP PLANNER: {planner_passed}/{len(planner_cases)}")
    print(
        "CP GROUPED PLANNER: "
        f"{grouped_planner_passed}/{len(grouped_planner_cases)}"
    )
    print(f"CP SEGMENTS: {segment_passed}/{len(segment_cases)}")
    print(
        "CP GROUPED SEGMENTS: "
        f"{grouped_segment_passed}/{len(grouped_segment_cases)}"
    )
    print(f"CP NUMERICS: {passed}/{total}")
    print(f"CP ACCEPTANCE: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
