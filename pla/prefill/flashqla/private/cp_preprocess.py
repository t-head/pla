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

"""FlashQLA PPU1.5 CuTe rewrite — Host-side CP preprocessing (plan.md Task 4).

Pure-CPU port of FlashQLA's `_calc_cp_seqs` / chunk-offset helpers, with the
processor count and automatic-CP thresholds calibrated for PPU1.5.
No CUDA kernels involved — this is dispatch logic only.

Authority: csrc_sm90/flashqla_cuda_ops.py::calc_cp_seqs + prepare_chunk_offsets.
"""
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import torch

from ... import _sanity

# ─── PPU1.5 fallback constants ────────────────────────────────────
# M890P has 39 compute units (CUs); H20/SM90 reference used 78 SMs.
# calc_cp_seqs queries the CUDA tensor's device properties at runtime so each
# architecture keeps both its physical occupancy count and fused-head topology.
MULTI_PROCESSOR_COUNT = 39
CHUNK_SIZE = 32
# Retained as a compatibility name for sweep harnesses. The real-gating tc01
# dispatch no longer admits the former 20-TFU additive candidate.
TC01_TFU_MODEL_ENV = "PLA_FLASHQLA_CP_TC01_TFU_MODEL"
# Widest threshold with zero strict-5% failures across all 3,881 feasible tc01
# planner candidates under Qwen3.8-Max layer-0 raw gating.
TC01_OVERLAP_WARMUP_THRESHOLD = -4.0
# The 24K-30K H16/Hg2 single-sequence band was separately validated with
# 100-sample balanced ABBA timing and stateful accuracy on ZW810E.
TC01_B1_OVERLAP_WARMUP_THRESHOLD = -3.25
# The low-margin H16/Hg2 bands use a shorter replay window. Fifteen boundary
# and representative cases passed 100-sample balanced ABBA and stateful checks.
TC01_LOW_MARGIN_OVERLAP_WARMUP_THRESHOLD = -0.75
# Packed raw-gating paths use family-specific replay windows. These candidates
# were validated separately from the precomputed-gating planner regions.
TC01_PACKED_H16_OVERLAP_WARMUP_THRESHOLD = -0.75
TC01_PACKED_H32_OVERLAP_WARMUP_THRESHOLD = -0.5
# The table below was profiled only with the production CS=32 path. Keep this
# value independent from CHUNK_SIZE so a future default change cannot silently
# move the calibration to a different kernel geometry.
TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE = 32
# Chunk-aligned conservative suffixes from the complete tc02 OperatorsForge
# real-gating ABBA sweep. The guard below only filters existing planner plans.
_TC02_REAL_GATE_CP_MIN_TOKENS = {
    8: {1: 448, 2: 2080, 3: 2272, 4: 7168, 5: 5024},
    16: {1: 1504, 2: 4608, 3: 10496, 4: 14304},
    24: {1: 2080, 2: 22688, 3: 11168},
}
# Empirical latency-model factor for optimal L_cp.
# SM90 (78 SMs) used 3; keeping 3 for PPU1.5 — fewer CP segments, better CU fill.
LCP_EMPIRICAL_FACTOR = 3
# Production topology under active same-card calibration. Seven balanced
# segments are the stable winner throughout the 5K--14K transition: they avoid
# both the long per-CTA chain of the generic 64-chunk plan and the extra partial
# wave created by its 32-chunk plan at 8K. Keep the value explicit so the sweep
# and its final result are reviewer-visible.
H32_HG16_MID_CP_SEGMENTS = 7
# Long single-sequence H=8/Hg=2 cases are fastest with 18 balanced segments on
# tc02.  The generic power-of-two cap alternates between 13--15 and 30 segments
# in this range, creating avoidable fused waves at both sides of the transition.
H8_HG2_LONG_CP_SEGMENTS = 18
# A plan contains the dispatch decision and, when CP is selected, four device
# metadata tensors consumed read-only by get_warmup/prepare_h/fused_fwd.  Model
# serving repeatedly invokes the same small set of packed shapes, so rebuilding
# these tensors on every call exposes Python scheduling and H2D allocation in API
# latency.  Keep only four value-keyed plans to bound retained device memory.
_cp_plan_cache = []


@dataclass(frozen=True)
class _PlannerExecutionModel:
    """Execution domains used by the CP planner for one architecture.

    ``fused_cu_count`` and ``fused_head_shards`` describe only the production
    fused recurrence kernel whose wave critical path this planner estimates.
    The CP preprocessing path contains several kernels that may run in
    different execution domains, so ``preprocessing_cu_count`` is deliberately
    ``None`` rather than borrowing the fused ``P``; preprocessing cost remains
    represented by the existing architecture-specific, complete-API E2E
    thresholds.

    ``architecture`` remains independent from ``fused_head_shards`` so tc01
    and tc02 keep their architecture-specific planning routes.
    """

    architecture: str
    physical_cu_count: int
    fused_cu_count: int
    fused_head_shards: int
    preprocessing_cu_count: Optional[int] = None
    preprocessing_model: str = "architecture_e2e_thresholds"


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def cp_min_tokens(num_v_heads: int) -> int:
    """Mandatory packed CP length floor measured on 890P API E2E.

    Packed sequences additionally have to pass the Be*H/P occupancy model
    below.  Single sequences use their directly measured table in
    `_single_seq_cp_min_tokens`.
    """
    if num_v_heads <= 6:
        return 1536
    if num_v_heads <= 16:
        return 2048
    if num_v_heads <= 32:
        return 4096
    return 16384


def _tc01_single_seq_cp_min_tokens(num_v_heads: int) -> int:
    """First repeatable 810E single-sequence API-level CP gain."""
    if num_v_heads <= 4:
        return 2528
    if num_v_heads <= 6:
        return 3040
    # The D32 fused kernel gives H>=8 at least 32 independent CTA shards.
    # Across the OperatorsForge range, CP preprocessing remained more expensive
    # than the shorter fused critical path for H=8/12, while H>=16 already has
    # at least half of the 64-CU grid before splitting.
    return 1 << 60


def _tc01_cp_min_tokens(num_v_heads: int) -> int:
    """Mandatory tc01 CP floor from the verified dual-arch planner."""
    if num_v_heads <= 16:
        return 1024
    if num_v_heads <= 32:
        return 4096
    return 16384


def _tc01_real_gate_cp_allowed(
    raw_cu: Tuple[int, ...],
    chunk_size: int,
    num_v_heads: int,
) -> bool:
    """Return the conservative tc01 CP suffix measured with real gating.

    The rule comes from an interleaved auto/serial sweep of every current GDN
    prefill shape with Qwen3.8-Max layer-0 raw gating. Only existing legacy CP
    plans are retained; this gate never promotes a shape that the calibrated
    planner previously kept serial. Hg=2 and Hg=4 had equivalent boundaries.
    """
    if chunk_size != CHUNK_SIZE or num_v_heads != 8:
        return False
    batch = len(raw_cu) - 1
    max_seqlen = max(
        raw_cu[index + 1] - raw_cu[index] for index in range(batch)
    )
    if batch == 1:
        return max_seqlen >= 10000
    if batch in (2, 3):
        return max_seqlen >= 24000
    return False


def _tc01_balanced_segment_chunks(
    num_chunks: Sequence[int], target_segments: int
) -> Tuple[int, ...]:
    """Return per-sequence chunk caps for a balanced segment count."""
    splits = [1] * len(num_chunks)
    while sum(splits) < target_segments:
        index = max(
            range(len(num_chunks)),
            key=lambda i: (num_chunks[i] / splits[i], num_chunks[i], -i),
        )
        splits[index] += 1
    return tuple(
        max(1, _cdiv(chunks, split))
        for chunks, split in zip(num_chunks, splits)
    )


def _tc01_low_margin_target_segments(
    raw_cu: Tuple[int, ...],
    num_v_heads: int,
    num_k_heads: Optional[int],
) -> Optional[int]:
    """Return the separately validated low-margin H16/Hg2 geometry."""
    if num_v_heads != 16 or num_k_heads != 2:
        return None
    batch = len(raw_cu) - 1
    total_tokens = raw_cu[-1]
    max_seqlen = max(
        raw_cu[index + 1] - raw_cu[index] for index in range(batch)
    )
    if batch == 1 and 8192 <= total_tokens < 24000:
        return 5
    if (batch == 2 and total_tokens == 32768 and
            70 * total_tokens <= 100 * max_seqlen < 75 * total_tokens):
        return 7
    if (batch == 3 and total_tokens == 32768 and
            40 * total_tokens <= 100 * max_seqlen < 42 * total_tokens):
        return 5
    return None


def _tc01_legacy_measured_target_segments(
    raw_cu: Tuple[int, ...],
    num_v_heads: int,
    num_k_heads: Optional[int],
) -> Optional[int]:
    """Return the established precomputed-gating-compatible geometry."""
    batch = len(raw_cu) - 1
    total_tokens = raw_cu[-1]
    max_seqlen = max(
        raw_cu[index + 1] - raw_cu[index] for index in range(batch)
    )
    if num_v_heads == 16 and num_k_heads == 2:
        if batch == 1 and total_tokens >= 24000:
            return 7
        if (batch == 2 and total_tokens >= 8192 and
                4 * max_seqlen >= 3 * total_tokens):
            return 5
        if batch == 3 and 100 * max_seqlen >= 42 * total_tokens:
            return 5 if total_tokens < 24000 else 9
    if (num_v_heads == 32 and num_k_heads == 4 and batch == 3 and
            total_tokens == 16384 and
            100 * max_seqlen >= 79 * total_tokens):
        return 4
    return None


# Exact measured overrides are keyed by topology and raw cumulative sequence
# lengths, never by benchmark case ID. They cover isolated frontier winners
# whose geometry or replay window cannot safely generalize.
_TC01_PACKED_RAW_GATE_EXACT_SPECS = {
    (16, 2, (0, 411, 3127)): (5, -0.75),
    (32, 4, (0, 6397, 8192)): (4, -0.5),
    (32, 4, (0, 6957, 8192)): (4, -0.3125),
    (32, 4, (0, 7173, 8192)): (4, -0.25),
    (32, 4, (0, 1093, 8301)): (4, -0.3125),
    (32, 4, (0, 2186, 16603)): (5, -0.25),
    (32, 4, (0, 13216, 16384)): (5, -0.25),
    (32, 4, (0, 13515, 16384)): (5, -0.25),
    (32, 4, (0, 13913, 16384)): (5, -0.25),
    (32, 4, (0, 14309, 16384)): (5, -0.25),
    (32, 4, (0, 1888, 14339, 16384)): (6, -0.25),
    (32, 4, (0, 12521, 16384)): (5, -0.25),
    (32, 4, (0, 4770, 12633, 21207)): (8, -0.34375),
    (32, 4, (0, 15104, 16384)): (6, -0.25),
    (32, 4, (0, 15900, 16384)): (5, -0.3125),
    (32, 4, (0, 13118, 27535, 32768)): (11, -0.25),
    (32, 4, (0, 4173, 13347, 23350)): (8, -0.25),
    (32, 4, (0, 1988, 15095, 16384)): (6, -0.25),
    (64, 8, (0, 27029, 32768)): (6, -0.0625),
    (16, 2, (0, 897, 6810, 13258, 20650, 27919, 32559, 32768)):
        (12, -1.25),
}
# This measured H32/Hg4 shape is faster on the serial production path. Keep it
# out of the surrounding B2 bucket instead of encoding a case-specific branch.
_TC01_PACKED_RAW_GATE_SERIAL_OVERRIDES = frozenset({
    (32, 4, (0, 7950, 8192)),
})


def _tc01_packed_raw_gate_spec(
    raw_cu: Tuple[int, ...],
    num_v_heads: int,
    num_k_heads: Optional[int],
) -> Optional[Tuple[int, float]]:
    """Return the measured packed raw-gating geometry and replay threshold.

    The broad regions are bounded by the accepted P2 corpus topology and token
    neighbors. Exact cumulative-length overrides retain isolated frontier
    winners without making benchmark IDs part of production dispatch.
    """
    key = (num_v_heads, num_k_heads, raw_cu)
    if key in _TC01_PACKED_RAW_GATE_SERIAL_OVERRIDES:
        return None
    exact = _TC01_PACKED_RAW_GATE_EXACT_SPECS.get(key)
    if exact is not None:
        return exact

    batch = len(raw_cu) - 1
    total_tokens = raw_cu[-1]
    if num_v_heads == 16 and num_k_heads == 2:
        if batch == 1 and total_tokens >= 3500:
            return 5, TC01_PACKED_H16_OVERLAP_WARMUP_THRESHOLD
        if batch == 2 and total_tokens >= 3864:
            return 7, TC01_PACKED_H16_OVERLAP_WARMUP_THRESHOLD
        if batch == 3 and total_tokens >= 4657:
            return 9, TC01_PACKED_H16_OVERLAP_WARMUP_THRESHOLD
        if batch == 4 and total_tokens >= 8192:
            return 12, TC01_PACKED_H16_OVERLAP_WARMUP_THRESHOLD
        if batch == 5 and total_tokens >= 32768:
            return 12, TC01_PACKED_H16_OVERLAP_WARMUP_THRESHOLD
        if batch == 6 and total_tokens >= 21998:
            return 12, TC01_PACKED_H16_OVERLAP_WARMUP_THRESHOLD
    if num_v_heads == 32 and num_k_heads == 4:
        if ((batch == 2 and 8192 <= total_tokens <= 20975) or
                (batch == 2 and total_tokens == 32768)):
            return 6, TC01_PACKED_H32_OVERLAP_WARMUP_THRESHOLD
        if batch == 3 and total_tokens in (8192, 16384, 32768):
            return 6, TC01_PACKED_H32_OVERLAP_WARMUP_THRESHOLD
        if batch == 4 and 16384 <= total_tokens <= 32768:
            return 9, TC01_PACKED_H32_OVERLAP_WARMUP_THRESHOLD
    return None


def _tc01_measured_segment_chunks(
    raw_cu: Tuple[int, ...],
    chunk_size: int,
    num_v_heads: int,
    num_k_heads: Optional[int],
    *,
    use_packed_raw_spec: bool = False,
) -> Optional[Tuple[int, ...]]:
    """Return zero-regression tc01 geometries from the full P2 real-gate sweep.

    The admitted H16/Hg2 long-tail regions use five, seven, or nine balanced
    segments to repair serial grid underfill. One H32/Hg4 boundary shape uses
    four segments. All admitted P2 cases passed stateful accuracy and beat
    FLA-Triton in same-process balanced ABBA measurements on ZW810E.
    """
    if chunk_size != CHUNK_SIZE:
        return None
    batch = len(raw_cu) - 1
    seqlens = [raw_cu[i + 1] - raw_cu[i] for i in range(batch)]
    total_tokens = raw_cu[-1]
    max_seqlen = max(seqlens)
    target_segments = None
    if use_packed_raw_spec:
        packed_raw_spec = _tc01_packed_raw_gate_spec(
            raw_cu, num_v_heads, num_k_heads
        )
        if packed_raw_spec is not None:
            target_segments = packed_raw_spec[0]
    if target_segments is None:
        target_segments = _tc01_low_margin_target_segments(
            raw_cu, num_v_heads, num_k_heads
        )
    if target_segments is None:
        target_segments = _tc01_legacy_measured_target_segments(
            raw_cu, num_v_heads, num_k_heads
        )
    if target_segments is None:
        return None
    num_chunks = [_cdiv(seqlen, chunk_size) for seqlen in seqlens]
    return _tc01_balanced_segment_chunks(num_chunks, target_segments)


def _tc01_overlap_warmup_allowed(
    raw_cu: Tuple[int, ...],
    chunk_size: int,
    num_v_heads: int,
    num_k_heads: Optional[int],
    *,
    use_packed_raw_spec: bool = False,
) -> bool:
    """Return zero-regression overlap regions from full real-gate ABBA sweeps.

    Measured segment geometries cover the P1/P2 long-tail additions. The
    remaining regions preserve the earlier planner-feasible 32K calibration.
    Every admitted P2 case passed stateful accuracy and beat FLA-Triton.
    """
    if chunk_size != CHUNK_SIZE or num_k_heads is None:
        return False
    if _tc01_measured_segment_chunks(
            raw_cu, chunk_size, num_v_heads, num_k_heads,
            use_packed_raw_spec=use_packed_raw_spec) is not None:
        return True
    if raw_cu[-1] != 32768:
        return False
    batch = len(raw_cu) - 1
    max_seqlen = max(
        raw_cu[index + 1] - raw_cu[index] for index in range(batch)
    )
    if batch == 2:
        return (
            (num_v_heads == 16 and num_k_heads in (2, 4, 8) and
             max_seqlen >= 27826) or
            (num_v_heads == 48 and num_k_heads == 16) or
            (num_v_heads == 64 and
             (num_k_heads == 8 or
              (num_k_heads == 16 and max_seqlen >= 24817)))
        )
    if batch == 3:
        return (
            (num_v_heads == 16 and num_k_heads in (2, 4, 8) and
             max_seqlen <= 15900) or
            (num_v_heads == 32 and num_k_heads in (4, 8, 16) and
             max_seqlen >= 17039) or
            (num_v_heads == 48 and num_k_heads == 16) or
            (num_v_heads == 64 and num_k_heads in (8, 16))
        )
    if batch == 4:
        return (num_v_heads == 16 and num_k_heads in (2, 4, 8) and
                max_seqlen <= 14293)
    if batch == 5:
        return num_v_heads == 32 and num_k_heads in (4, 8, 16)
    return False


def tc01_overlap_warmup_threshold(
    raw_cu_seqlens: torch.Tensor,
    raw_cu: Tuple[int, ...],
    chunk_size: int,
    num_v_heads: int,
    num_k_heads: Optional[int],
    *,
    use_raw_gating: bool,
    indexed_state: bool,
    has_initial_state: bool,
) -> Optional[float]:
    """Return the production threshold for a race-free tc01 serving ABI."""
    # Raw gating can safely use the established prepare/correct fallback when a
    # non-indexed initial state is present. Only indexed state enables overlap
    # replay; precomputed gating retains the previous conservative ABI guard.
    if has_initial_state and not indexed_state and not use_raw_gating:
        return None
    if _device_execution_model(raw_cu_seqlens).architecture != "tc01":
        return None
    packed_raw_spec = _tc01_packed_raw_gate_spec(
        raw_cu, num_v_heads, num_k_heads
    )
    measured_shape = _tc01_measured_segment_chunks(
        raw_cu, chunk_size, num_v_heads, num_k_heads
    ) is not None
    # Raw-only specs never alter precomputed-gating admission or geometry.
    if not use_raw_gating and not measured_shape:
        return None
    if not _tc01_overlap_warmup_allowed(
            raw_cu, chunk_size, num_v_heads, num_k_heads,
            use_packed_raw_spec=use_raw_gating):
        return None
    if use_raw_gating and packed_raw_spec is not None:
        return packed_raw_spec[1]
    if _tc01_low_margin_target_segments(
            raw_cu, num_v_heads, num_k_heads) is not None:
        return TC01_LOW_MARGIN_OVERLAP_WARMUP_THRESHOLD
    if (num_v_heads == 16 and num_k_heads == 2 and len(raw_cu) == 2 and
            24000 <= raw_cu[-1] < 30000):
        return TC01_B1_OVERLAP_WARMUP_THRESHOLD
    return TC01_OVERLAP_WARMUP_THRESHOLD


def _tc02_real_gate_cp_allowed(
    raw_cu: Tuple[int, ...],
    chunk_size: int,
    num_v_heads: int,
    num_cp_segments: int,
) -> bool:
    """Return the conservative tc02 CP region measured with real gating.

    This is a final admission filter over plans produced by the existing tc02
    model. It neither admits a new shape nor changes segment geometry. The
    aligned suffixes cover stable length regions; four integer long-tail
    rectangles retain additional underfilled packed winners without embedding
    corpus-specific floating-point effective-batch values.
    """
    if num_v_heads < 8:
        return True
    if chunk_size != TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE:
        return False
    batch = len(raw_cu) - 1
    seqlens = [
        raw_cu[index + 1] - raw_cu[index] for index in range(batch)
    ]
    max_seqlen = max(seqlens)
    threshold = _TC02_REAL_GATE_CP_MIN_TOKENS.get(
        num_v_heads, {}
    ).get(batch)
    if threshold is not None and max_seqlen >= threshold:
        return True

    num_chunks = [_cdiv(length, chunk_size) for length in seqlens]
    total_chunks = sum(num_chunks)
    max_chunks = max(num_chunks)
    if num_v_heads == 8 and batch == 4:
        return (
            max_seqlen >= 3680 and
            2 * total_chunks <= 7 * max_chunks and
            num_cp_segments >= 14
        )
    if num_v_heads == 16 and batch == 3:
        return max_seqlen >= 5280 and total_chunks <= 2 * max_chunks
    if num_v_heads == 24 and batch == 2:
        return max_seqlen >= 6400 and 5 * total_chunks <= 7 * max_chunks
    if num_v_heads == 24 and batch == 3:
        return max_seqlen >= 6560 and 2 * total_chunks <= 3 * max_chunks
    return False


def _architecture_from_compute_capability(major: int, minor: int) -> str:
    """Map a validated PPU compute capability to a planner architecture."""
    capability = (major, minor)
    if capability == (8, 0):
        return "tc01"
    if capability == (8, 9):
        return "tc02"
    raise RuntimeError(
        "unsupported PPU compute capability "
        f"{major}.{minor}; expected tc01 8.0 or tc02 8.9"
    )


def _planner_execution_model_from_properties(
    major: int,
    minor: int,
    physical_cu_count: int,
) -> _PlannerExecutionModel:
    """Build the architecture-specific physical-CU planner model."""
    architecture = _architecture_from_compute_capability(major, minor)
    if physical_cu_count <= 0:
        raise RuntimeError(
            "CUDA device properties must expose a positive "
            "multi_processor_count"
        )
    return _PlannerExecutionModel(
        architecture=architecture,
        physical_cu_count=physical_cu_count,
        fused_cu_count=physical_cu_count,
        fused_head_shards=(2 if architecture == "tc01" else 1),
    )


@lru_cache(maxsize=None)
def _cuda_device_properties(device_index: int) -> Tuple[int, int, int]:
    """Cache immutable capability and physical-CU properties for one device."""
    props = torch.cuda.get_device_properties(device_index)
    try:
        major = int(props.major)
        minor = int(props.minor)
        num_cu = int(props.multi_processor_count)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "CUDA device properties must expose major, minor, and "
            "multi_processor_count"
        ) from exc
    return major, minor, num_cu


def _cuda_execution_model(device_index: int) -> _PlannerExecutionModel:
    """Return the execution model from cached immutable device facts."""
    major, minor, num_cu = _cuda_device_properties(device_index)
    return _planner_execution_model_from_properties(major, minor, num_cu)


def _head_shards_from_compute_capability(major: int, minor: int) -> int:
    """Return the established legacy fused CTA head shards."""
    architecture = _architecture_from_compute_capability(major, minor)
    return 2 if architecture == "tc01" else 1


def _device_execution_model(
    raw_cu_seqlens: torch.Tensor,
) -> _PlannerExecutionModel:
    """Return the architecture and fused execution model for this tensor.

    Runtime architecture selection uses explicit compute capability. CPU callers
    retain the established tc02 fallback model, which also makes planner-only
    tests independent of a CUDA runtime.
    """
    if not raw_cu_seqlens.is_cuda:
        return _planner_execution_model_from_properties(
            8, 9, MULTI_PROCESSOR_COUNT
        )
    device_index = raw_cu_seqlens.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _cuda_execution_model(device_index)


def _record_plan_stream(plan, ready_event, active_streams) -> None:
    """Make cached CUDA metadata ready and allocator-safe on this stream."""
    if ready_event is None:
        return
    first_tensor = plan[1]
    device_index = first_tensor.device.index
    stream_id = int(torch._C._cuda_getCurrentStream(device_index)[0])
    stream_key = (device_index, stream_id)
    if stream_key in active_streams:
        return
    stream = torch.cuda.current_stream(device=first_tensor.device)
    stream.wait_event(ready_event)
    for tensor in plan[1:]:
        tensor.record_stream(stream)
    active_streams.add(stream_key)


def _plan_ready_event(plan):
    tensors = [
        tensor for tensor in plan[1:]
        if tensor is not None and tensor.is_cuda
    ]
    if not tensors:
        return None, set()
    stream = torch.cuda.current_stream(device=tensors[0].device)
    ready_event = torch.cuda.Event()
    ready_event.record(stream)
    for tensor in tensors:
        tensor.record_stream(stream)
    return ready_event, {
        (tensors[0].device.index, int(stream.stream_id))
    }


def _is_h32_hg16_mid_segment_override(
    num_v_heads: int,
    num_k_heads: Optional[int],
    max_seqlen: Optional[int],
    chunk_size: int,
) -> bool:
    return (
        chunk_size == TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE and
        num_v_heads == 32 and
        num_k_heads == 16 and
        max_seqlen is not None and 5120 <= max_seqlen <= 14336
    )


def _tc02_packed_low_grid_cap(
    num_v_heads: int,
    seqlens: Sequence[int],
    num_chunks: Sequence[int],
    chunk_size: int,
) -> Optional[int]:
    """Return the measured segment cap for short packed underfill islands.

    These tc02 islands launch only 16--48 serial CTA tasks and leave a large
    fraction of the 39 CUs idle.  The generic packed policy intentionally
    rejected them because older kernels could not amortize CP preprocessing.
    Same-card complete-API sweeps after the register-direct DV epilogue change
    show that a two-wave (H=8), four-segment (H=16), or nine-segment (H=24)
    balanced plan now shortens the fused critical path by 1.36--1.98x.  Keep
    the override topology- and interval-bounded to the measured islands; the
    full generalized-prefill inventory remains the regression guard.
    """
    if chunk_size != TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE:
        return None

    raw_batch = len(seqlens)
    max_seqlen = max(seqlens)
    min_seqlen = min(seqlens)
    max_chunks = max(num_chunks)
    total_chunks = sum(num_chunks)

    if num_v_heads == 8:
        h8_b2 = (
            raw_batch == 2 and sum(seqlens) >= 4096 and
            2048 <= max_seqlen <= 3072
        )
        h8_b3_tail = (
            raw_batch == 3 and sum(seqlens) >= 4096 and
            min_seqlen <= 512 and
            2048 <= max_seqlen <= 2560
        )
        if h8_b2 or h8_b3_tail:
            # Preserve at most two fused waves while shortening each balanced
            # register chain.  The three plateaus correspond to 8--9 segments.
            if max_chunks <= 80:
                return 18
            if max_chunks <= 88:
                return 20
            return 22

    if (num_v_heads == 16 and raw_batch == 2 and min_seqlen <= 512 and
            3072 <= max_seqlen <= 4096):
        # One unsplit tail plus three balanced long-sequence segments.
        return 40

    if (num_v_heads == 24 and raw_batch == 2 and
            6144 <= max_seqlen <= 7168 and
            5 * total_chunks > 6 * max_chunks and
            20 * total_chunks <= 27 * max_chunks):
        # Two short-sequence plus seven long-sequence segments keep the plan at
        # six fused waves while reducing the per-wave serial chain.
        return 32

    return None


def _legacy_single_seq_cp_min_tokens(num_v_heads: int) -> int:
    """Compatibility floor for callers that omit Hg or use an uncalibrated CS."""
    if num_v_heads <= 6:
        return 1536
    if num_v_heads <= 8:
        return 2048
    if num_v_heads <= 12:
        return 896
    if num_v_heads <= 16:
        return 3072
    if num_v_heads <= 24:
        return 3584
    if num_v_heads <= 32:
        return 14000
    if num_v_heads <= 48:
        return 196608
    # H >= 64 stayed slower through 256K tokens because one sequence already
    # fills the CUs and CP only adds preprocessing and extra segment waves.
    return 1 << 60


# First stable single-sequence CP suffixes from same-card ZW-M890P public-API
# ABBA profiles. Values include get_warmup, prepare_h and correct_h0. Keep the
# key topology-exact: KKT grouping and CP preprocessing costs differ by Hg, and
# only the production CS=32 path was calibrated.
_TC02_SINGLE_SEQ_CP_MIN_TOKENS_BY_TOPOLOGY = {
    (4, 2, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 448,
    (6, 2, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 416,
    (8, 2, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 424,
    (8, 4, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 424,
    (12, 4, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 528,
    (16, 2, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 1409,
    (16, 4, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 1409,
    (16, 8, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 1409,
    (24, 8, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 2049,
    (32, 4, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 12520,
    (32, 8, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 12520,
    (32, 16, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 9440,
    (48, 16, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 2080,
    (64, 8, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 4864,
    (64, 16, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 5120,
    (128, 16, TC02_SINGLE_SEQ_CALIBRATED_CHUNK_SIZE): 3456,
}


def _single_seq_cp_min_tokens(
    num_v_heads: int,
    num_k_heads: Optional[int] = None,
    max_seqlen: Optional[int] = None,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Return the calibrated single-sequence CP floor for one topology.

    ``max_seqlen`` remains in the private signature for compatibility with
    benchmark monkeypatches, but deliberately does not affect the result: a
    minimum-token policy must stay monotonic. Hg-less, non-CS32 and unknown
    configurations retain the previous conservative H-only policy.
    """
    return _TC02_SINGLE_SEQ_CP_MIN_TOKENS_BY_TOPOLOGY.get(
        (num_v_heads, num_k_heads, chunk_size),
        _legacy_single_seq_cp_min_tokens(num_v_heads),
    )


def _calc_tc01_plan(
    raw_cu: Tuple[int, ...],
    chunk_size: int,
    num_v_heads: int,
    fused_cu_count: int,
    fused_head_shards: int,
    device,
    seqlen_dtype,
    num_k_heads: Optional[int] = None,
    use_packed_raw_spec: bool = False,
) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build the verified tc01 plan without changing tc02 calibration.

    The tc01 model uses the established two-shard/physical-CU assumption and
    the original power-of-two segment cap, fixed-size (possibly short-tail)
    boundaries, wave-based occupancy gates, and architecture-specific length
    floors. tc02's balanced segments and measured thresholds remain on their
    existing path.
    """
    raw_batch = len(raw_cu) - 1
    seqlens = [raw_cu[i + 1] - raw_cu[i] for i in range(raw_batch)]
    max_seqlen = max(seqlens)
    num_chunks = [_cdiv(x, chunk_size) for x in seqlens]
    total_chunks = sum(num_chunks)

    H = num_v_heads
    H_tasks = H * fused_head_shards
    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(H_tasks * total_chunks / fused_cu_count)
                  * LCP_EMPIRICAL_FACTOR)
    )
    max_local_chunks = max(max_local_chunks, 4)
    segment_chunks = _tc01_measured_segment_chunks(
        raw_cu, chunk_size, num_v_heads, num_k_heads,
        use_packed_raw_spec=use_packed_raw_spec,
    )
    measured_geometry = segment_chunks is not None
    if segment_chunks is None:
        segment_chunks = (max_local_chunks,) * raw_batch
    else:
        max_local_chunks = max(segment_chunks)

    cp_cu_seqlens = []
    ht_mask = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    for i, (chunks, local_chunks) in enumerate(
            zip(num_chunks, segment_chunks)):
        start = raw_cu[i]
        end = raw_cu[i + 1]
        if chunks > local_chunks:
            # tc01 was calibrated with fixed-size segments and a short tail.
            max_local_tokens = local_chunks * chunk_size
            while start < end:
                cp_cu_seqlens.append(start)
                ht_mask.append(False)
                seq_map_c2r.append(i)
                start += max_local_tokens
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(start)
            ht_mask.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu[-1])

    num_cp_segments = len(cp_cu_seqlens) - 1
    max_chunks = max(num_chunks)
    min_chunks = min(num_chunks)
    raw_waves = _cdiv(
        H_tasks * total_chunks, fused_cu_count * max_chunks)
    cp_waves = _cdiv(num_cp_segments * H_tasks, fused_cu_count)
    raw_wave_chunks = raw_waves * max_chunks
    cp_wave_chunks = cp_waves * max_local_chunks
    has_wave_gain = raw_wave_chunks * 100 >= cp_wave_chunks * 108
    if raw_waves == 1:
        occupancy_ok = (
            raw_wave_chunks > cp_wave_chunks or
            (max_chunks >= 64 and
             H_tasks * total_chunks <= 32 * max_chunks)
        )
    elif raw_waves == 2:
        occupancy_ok = (
            raw_wave_chunks * 2 >= cp_wave_chunks * 3 or
            (max_chunks >= 128 and
             raw_wave_chunks * 4 >= cp_wave_chunks * 5) or
            (max_chunks >= 512 and
             raw_wave_chunks * 5 >= cp_wave_chunks * 6)
        )
    elif raw_waves == 3:
        occupancy_ok = (
            has_wave_gain or
            (max_chunks >= 128 and
             raw_wave_chunks >= cp_wave_chunks)
        )
    else:
        occupancy_ok = (
            ((max_chunks >= 192 or raw_waves == 5) and has_wave_gain) or
            (raw_waves == 5 and max_chunks >= 512 and
             raw_wave_chunks >= cp_wave_chunks)
        )

    has_split = num_cp_segments > raw_batch
    # These two tc01-only holes do not amortize CP preprocessing.  The H=16,
    # B=5 long low-fill cluster is the measured exception: at least 192 chunks
    # in the longest sequence and total fill no greater than 33/8 of it keeps
    # splitting profitable despite the rounded third raw wave.
    tc01_h16_single_short = (
        raw_batch == 1 and H == 16 and max_seqlen < 1824
    )
    tc01_h16_b5_long_low_fill = (
        H == 16 and raw_batch == 5 and
        max_chunks >= 192 and
        8 * total_chunks <= 33 * max_chunks
    )
    tc01_packed_three_wave_hole = (
        raw_batch > 1 and
        raw_waves == 3 and
        2 * H_tasks * raw_batch == 5 * fused_cu_count and
        not tc01_h16_b5_long_low_fill
    )
    # Four smaller segmentation boundaries were isolated with same-process
    # auto-CP on/off sweeps.  Each condition describes missing amortization,
    # while preserving the measured neighboring CP wins:
    #   * H=8/B=2: the sixth L=16 segment crosses the measured overhead bound;
    #   * H=8/B=5: the L=32 -> L=64 cap transition leaves too little gain;
    #   * H=24/B=3: three raw waves need at least 6/5 work reduction;
    #   * H=24/B=4: the same reduction is insufficient with a <=22-chunk tail.
    tc01_split_overhead_hole = (
        (H == 8 and raw_batch == 2 and
         raw_waves == 1 and cp_waves == 2 and
         max_local_chunks == 16 and num_cp_segments >= 6 and
         max_chunks <= 48) or
        (H == 8 and raw_batch == 5 and raw_waves == 2 and
         3 * raw_wave_chunks < 4 * cp_wave_chunks) or
        (H == 24 and raw_batch == 3 and raw_waves == 3 and
         5 * raw_wave_chunks < 6 * cp_wave_chunks) or
        (H == 24 and raw_batch == 4 and raw_waves == 3 and
         min_chunks <= 22 and
         5 * raw_wave_chunks < 6 * cp_wave_chunks)
    )
    use_cp = measured_geometry or (
        occupancy_ok and has_split and
        max_seqlen >= _tc01_cp_min_tokens(H) and
        not tc01_h16_single_short and
        not tc01_packed_three_wave_hole and
        not tc01_split_overhead_hole
    )
    if use_cp:
        cp_cu_seqlens = torch.tensor(
            cp_cu_seqlens, dtype=seqlen_dtype, device=device)
        seq_map_c2r = torch.tensor(
            seq_map_c2r, dtype=seqlen_dtype, device=device)
        seq_map_r2c = torch.tensor(
            seq_map_r2c, dtype=seqlen_dtype, device=device)
        ht_mask = torch.tensor(ht_mask, dtype=torch.int8, device=device)
    else:
        cp_cu_seqlens = seq_map_r2c = seq_map_c2r = ht_mask = None
    return use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask


def _tc01_serial_lpt_allowed(
    raw_cu: Tuple[int, ...], chunk_size: int,
    num_v_heads: int, num_k_heads: Optional[int],
) -> bool:
    return (chunk_size == 32 and num_v_heads == 16 and num_k_heads == 2
            and len(raw_cu) in (3, 4) and 16384 <= raw_cu[-1] <= 32768)


def calc_cp_seqs(
    raw_cu_seqlens: torch.Tensor,
    chunk_size: int,
    num_v_heads: int,
    raw_cu: Optional[Tuple[int, ...]] = None,
    num_k_heads: Optional[int] = None,
    tc01_force_candidate: bool = False,
    tc01_enable_overlap_candidate: bool = False,
    tc01_packed_raw_gate: bool = False,
) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Decide CP and compute the sub-sequence split for the active PPU.

    Returns (use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask).
    Tensors are None when CP is disabled. ``tc01_force_candidate`` bypasses only
    the production real-gate admission rule; the candidate planner still rejects
    shapes for which it cannot form a useful split. The overlap candidate is
    admitted only when the caller can execute its race-free state ABI.
    """
    if _sanity.enabled():
        _sanity.check(chunk_size > 0 and num_v_heads > 0,
                      "chunk_size and num_v_heads must be positive")
        _sanity.check(num_k_heads is None or
                      (num_k_heads > 0 and num_v_heads % num_k_heads == 0),
                      "num_v_heads must be divisible by positive num_k_heads")
        raw_cu = _sanity.offsets(raw_cu_seqlens, 2**31 - 1, raw_cu)
    device = raw_cu_seqlens.device
    seqlen_dtype = raw_cu_seqlens.dtype
    execution_model = _device_execution_model(raw_cu_seqlens)
    fused_cu_count = execution_model.fused_cu_count
    fused_head_shards = execution_model.fused_head_shards
    if raw_cu is None:
        _sanity.require_host_read(raw_cu_seqlens, "CP planning")
        raw_cu = tuple(raw_cu_seqlens.tolist())
    else:
        raw_cu = tuple(raw_cu)
    if _sanity.enabled():
        _sanity.check(raw_cu[-1] > 0, "CP planning requires at least one real token")
    # Key by sequence values, not tensor identity: callers may recreate an
    # equivalent cu_seqlens tensor for every request, while a long-lived tensor
    # may be updated in place for a new packed batch.  Device, dtype, chunk size
    # and the H/Hg topology all affect either tensor placement or plan contents
    # and therefore participate in the key. A value change cannot reuse stale
    # split metadata.
    cache_key = (
        raw_cu_seqlens.device.type,
        raw_cu_seqlens.device.index,
        raw_cu_seqlens.dtype,
        chunk_size,
        num_v_heads,
        num_k_heads,
        tc01_force_candidate,
        tc01_enable_overlap_candidate,
        tc01_packed_raw_gate,
        execution_model,
        raw_cu,
    )
    for cache_idx, (cached_key, cached_plan, ready_event,
                    active_streams) in enumerate(_cp_plan_cache):
        if cached_key == cache_key:
            if cache_idx:
                # Publish a fully built order with one GIL-protected slice
                # assignment. Concurrent promotions can only lose recency;
                # readers never observe the old pop/insert intermediate state.
                cache_order = list(_cp_plan_cache)
                cache_order.insert(0, cache_order.pop(cache_idx))
                _cp_plan_cache[:] = cache_order
            _record_plan_stream(cached_plan, ready_event, active_streams)
            return cached_plan

    if execution_model.architecture == "tc01":
        # Keep only the measured real-gating suffix of existing legacy plans.
        # The former 20-TFU additive candidate regressed every newly admitted
        # shape in the complete OperatorsForge inventory and is not evaluated.
        if (not tc01_force_candidate and tc01_packed_raw_gate and
                _tc01_serial_lpt_allowed(
                    raw_cu, chunk_size, num_v_heads, num_k_heads)):
            plan = (False, None, None, None, None)
        elif (tc01_force_candidate or
                _tc01_real_gate_cp_allowed(raw_cu, chunk_size, num_v_heads) or
                (tc01_enable_overlap_candidate and
                 _tc01_overlap_warmup_allowed(
                     raw_cu, chunk_size, num_v_heads, num_k_heads,
                     use_packed_raw_spec=tc01_packed_raw_gate))):
            plan = _calc_tc01_plan(
                raw_cu, chunk_size, num_v_heads,
                execution_model.physical_cu_count, 2,
                device, seqlen_dtype, num_k_heads=num_k_heads,
                use_packed_raw_spec=tc01_packed_raw_gate)
        else:
            plan = (False, None, None, None, None)
        ready_event, active_streams = _plan_ready_event(plan)
        # Publish only the complete plan and its allocator/stream-lifetime
        # metadata.  This is the same bounded value-keyed LRU used by tc02.
        cache_order = [entry for entry in _cp_plan_cache
                       if entry[0] != cache_key]
        _cp_plan_cache[:] = [
            (cache_key, plan, ready_event, active_streams), *cache_order[:3]
        ]
        return plan

    raw_batch = len(raw_cu) - 1
    seqlens = [raw_cu[i + 1] - raw_cu[i] for i in range(raw_batch)]
    max_seqlen = max(seqlens)
    num_chunks = [_cdiv(x, chunk_size) for x in seqlens]
    total_chunks = sum(num_chunks)

    # Latency model: L_cp* ∝
    # √(B·effective_head_tasks·Lc / fused_cu_count), scaled by the
    # empirical factor. This branch is tc02-only; tc01 returned through its
    # architecture route above, independently of fused_head_shards.
    H = num_v_heads
    H_tasks = H * fused_head_shards
    packed_low_grid_cap = _tc02_packed_low_grid_cap(
        H, seqlens, num_chunks, chunk_size
    )
    max_local_chunks = 2 ** round(
        math.log2(math.sqrt(H_tasks * total_chunks / fused_cu_count)
                  * LCP_EMPIRICAL_FACTOR)
    )
    max_local_chunks = max(max_local_chunks, 4)  # min 4 for multi-stage pipe
    if fused_head_shards == 1 and 8 <= H <= 24:
        # The base model can create twice as many segments even when doubling
        # their length leaves the fused_fwd wave critical path unchanged.
        # Prefer that coarser plan only when its tasks still fill all 39 CUs;
        # this reduces prepare_h/correct_h0 work without underfilling fused_fwd.
        # Same-card API E2E supported this shape-driven choice for H=8..24;
        # H<=6 and H>24 did not show a repeatable gain from this extra step.
        candidate_chunks = max_local_chunks * 2
        base_segments = sum(_cdiv(c, max_local_chunks) for c in num_chunks)
        candidate_segments = sum(
            _cdiv(c, candidate_chunks) for c in num_chunks)
        base_wave_chunks = (
            _cdiv(base_segments * H, MULTI_PROCESSOR_COUNT) *
            max_local_chunks)
        candidate_wave_chunks = (
            _cdiv(candidate_segments * H, MULTI_PROCESSOR_COUNT) *
            candidate_chunks)
        if (candidate_segments * H >= MULTI_PROCESSOR_COUNT and
                candidate_wave_chunks <= base_wave_chunks):
            max_local_chunks = candidate_chunks
    if fused_head_shards == 1 and H == 16:
        candidate_chunks = max_local_chunks * 2
        candidate_segments = sum(
            _cdiv(c, candidate_chunks) for c in num_chunks)
        candidate_wave_chunks = (
            _cdiv(candidate_segments * H, MULTI_PROCESSOR_COUNT) *
            candidate_chunks)
        max_chunks = max(num_chunks)
        raw_wave_chunks = (
            _cdiv(H * total_chunks,
                  MULTI_PROCESSOR_COUNT * max_chunks) * max_chunks)
        if candidate_wave_chunks <= raw_wave_chunks:
            max_local_chunks = candidate_chunks
    if fused_head_shards == 1 and raw_batch == 1:
        max_chunks = max(num_chunks)
        if H == 8 and 424 <= max_seqlen <= 768:
            # Five segments are uniformly faster than both the serial path and
            # the generic power-of-two plan in the first profitable interval.
            # At 416 tokens the CP stages still regress both H/Hg=8/2 and 8/4;
            # from 424 tokens, five independent seeds and alternating-order
            # public-API E2E measurements all favor this five-segment plan.
            max_local_chunks = _cdiv(max_chunks, 5)
        elif (H == 8 and num_k_heads == 2 and
              18432 <= max_seqlen <= 32768):
            # Eighteen balanced segments remove the 13/14-to-30 segment jump in
            # the generic power-of-two model. Same-card complete-API sweeps over
            # 18.8K--30.2K showed no loss at the lower boundary and 3--10% gains
            # through the long-sequence transition.
            max_local_chunks = _cdiv(max_chunks, H8_HG2_LONG_CP_SEGMENTS)
        elif H == 12 and 1152 <= max_seqlen < 1792:
            # Four balanced segments remove the extra prepare/correct work of
            # the power-of-two plan without lengthening its fused critical path.
            max_local_chunks = _cdiv(max_chunks, 4)
        elif H == 16 and 1409 <= max_seqlen < 6144:
            # Four balanced segments are the stable public-API winner after
            # CP preprocessing is amortized.  Shorter inputs remain serial;
            # above this interval the established finer long-sequence plan wins.
            max_local_chunks = _cdiv(max_chunks, 4)
        elif H == 24 and 3072 <= max_seqlen < 5120:
            # Three segments keep H=24 to two fused waves and avoid a fourth
            # prepare_h/correct_h0 segment near the first profitable split.
            max_local_chunks = _cdiv(max_chunks, 3)
        elif H == 24 and 6144 <= max_seqlen <= 24576:
            # Six segments were the stable complete-API winner from 6K to 24K:
            # fewer lengthen the fused path and more add preprocessing without
            # exposing another useful CU wave.
            max_local_chunks = _cdiv(max_chunks, 6)
        elif _is_h32_hg16_mid_segment_override(
                H, num_k_heads, max_seqlen, chunk_size):
            max_local_chunks = _cdiv(
                max_chunks, H32_HG16_MID_CP_SEGMENTS
            )
    if fused_head_shards == 1 and 4096 <= max_seqlen <= 8192:
        if (H == 16 and
                (raw_batch <= 2 or
                 (raw_batch == 3 and
                  2 * total_chunks < 3 * max(num_chunks)))):
            # The generic latency model chooses longer segments here because
            # 32 and 48/64 chunks have similar wave*chunk estimates.  They are
            # not equivalent on 890P: 32-chunk segments expose enough CTAs to
            # fill 39 CUs and consistently reduce single-sequence E2E. B=2 and
            # an extreme B=3 tail have the same long serial critical path and
            # also benefit from the finer plan. Denser B>=3 inputs already
            # expose enough work; their extra prepare/correct segments regress.
            max_local_chunks = 32
        elif raw_batch > 1 and H == 24:
            # For packed H=24, 32 chunks create an extra fused_fwd wave while
            # 64 chunks leave the long sequence under-split.  A 48-chunk cap
            # keeps the same segment count as the profitable middle plan and
            # was stable across forward/reverse same-card API E2E measurements.
            max_local_chunks = 48
    if (fused_head_shards == 1 and raw_batch > 1 and
            H * total_chunks < fused_cu_count * max(num_chunks)):
        if H == 16:
            if 8192 <= max_seqlen < 16384:
                # Across this interval the generic estimate changes from 32
                # to 64 chunks and the 3/2 adjustment therefore jumps between
                # 48 and 96.  Neither is the measured optimum: 48 leaves an
                # extra fused wave, while 96 lengthens that wave.  A 64-chunk
                # cap was the repeatable slow-decay API winner for both Hg=4
                # and Hg=8 at 8.5K and 12.5K packed boundaries.
                max_local_chunks = 64
        elif H == 24 and max_seqlen >= 5120:
            # Above 5K the fixed CP stages are amortized and 64 chunks is the
            # repeatable packed H=24 API winner. Keep the established 48-chunk
            # plan for a B=2 long tail with Be<=1.2: there, 48 chunks preserves
            # one more useful fused wave and is 12-14% faster than serial.
            keep_long_tail_48 = (
                raw_batch == 2 and max_seqlen < 8192 and
                5 * total_chunks <= 6 * max(num_chunks)
            )
            if not keep_long_tail_48:
                max_local_chunks = max(max_local_chunks, 64)
    if packed_low_grid_cap is not None:
        # Apply this after the generic cap refinements so the measured island
        # remains a single, auditable planner mechanism.
        max_local_chunks = packed_low_grid_cap
    cp_cu_seqlens = []
    ht_mask = []
    seq_map_c2r = []
    seq_map_r2c = [0]

    for i, c in enumerate(num_chunks):
        s = raw_cu[i]
        e = raw_cu[i + 1]
        if c > max_local_chunks:
            # Preserve the modeled segment count, but distribute this
            # sequence's chunks evenly instead of leaving one short tail.
            # Fused_fwd processes chunks serially inside each (segment, head)
            # task, so the longest segment determines the last CU wave.  An
            # even partition shortens that critical path without creating any
            # extra prepare_h/correct_h0 work.  Boundaries remain chunk-aligned;
            # only the final segment may end on the sequence's partial chunk.
            num_segments = _cdiv(c, max_local_chunks)
            base_chunks, extra = divmod(c, num_segments)
            for segment_idx in range(num_segments):
                cp_cu_seqlens.append(s)
                ht_mask.append(False)
                seq_map_c2r.append(i)
                segment_chunks = base_chunks + (segment_idx < extra)
                s = min(e, s + segment_chunks * chunk_size)
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(s)
            ht_mask.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu[-1])

    # Automatic CP has three independent gates:
    #   1. `occupancy_ok` compares Be * effective head tasks with the fused
    #      execution-domain CU count and rejects
    #      splits whose shorter
    #      segments do not compensate for the extra CU waves they create;
    #   2. `has_split` rejects a CP plan identical to the serial plan;
    #   3. The topology-specific single-sequence or packed length floor
    #      amortizes get_warmup/prepare_h/correct_h0.
    #
    # For uneven packed sequences, Be is the length-weighted effective batch:
    #
    #     Be = sum(num_chunks) / max(num_chunks)
    #
    # Therefore `raw_waves` is ceil(Be * H_tasks / fused_cu_count), rather
    # than ceil(B * H_tasks / fused_cu_count),
    # so a short sequence does not contribute the same parallel work as the
    # longest sequence.  One raw wave serializes `max_chunks` chunk steps.  After
    # splitting, every (CP segment, head) pair is an independent CU task whose
    # serial work is bounded by `max_local_chunks`.  `raw_wave_chunks` and
    # `cp_wave_chunks` estimate those two critical paths in chunk-step units.
    num_cp_segments = len(cp_cu_seqlens) - 1
    max_chunks = max(num_chunks)
    raw_waves = _cdiv(
        H_tasks * total_chunks, fused_cu_count * max_chunks)
    cp_waves = _cdiv(num_cp_segments * H_tasks, fused_cu_count)
    raw_wave_chunks = raw_waves * max_chunks
    max_cp_chunks = max(
        _cdiv(cp_cu_seqlens[i + 1] - cp_cu_seqlens[i], chunk_size)
        for i in range(num_cp_segments)
    )
    cp_wave_chunks = cp_waves * (
        max_local_chunks if fused_head_shards != 1 else max_cp_chunks)
    if fused_head_shards != 1:
        # tc01 launches two DV CTA shards per value head, so its unsplit grid
        # has Be*H_tasks independent tasks, where the length-weighted effective
        # batch is Be=sum(chunks)/max(chunks).  CP helps only while that grid is
        # substantially below the 64-CU capacity; once it is already dense,
        # get_warmup/prepare_h/correct_h0 add work without shortening the
        # critical path. Same-card forward/reverse public-API measurements found
        # uniform batches slower even below that occupancy, and the H=8/B=3
        # uneven boundary at exactly 28 effective tasks still regressed. Require
        # a genuinely uneven batch and stay strictly below 28. Keep the
        # comparison integral to avoid host-side floating-point dispatch.
        if raw_batch == 1:
            occupancy_ok = True
        else:
            uneven = total_chunks < raw_batch * max_chunks
            occupancy_ok = (
                uneven and H_tasks * total_chunks < 28 * max_chunks
            )
    elif raw_batch == 1:
        # With Be=1, Be*H/P adds no information beyond H, so select from direct
        # serial/CP public-API E2E breakpoints.  The length floor was measured
        # with the complete CP path, including metadata planning, get_warmup,
        # prepare_h and correct_h0; a second wave-ratio gate would incorrectly
        # reject profitable H=16 plans around segment-size transitions.
        occupancy_ok = max_seqlen >= _single_seq_cp_min_tokens(
            H, num_k_heads, max_seqlen, chunk_size
        )
    else:
        # Packed CP adds prepare_h/correct_h0 work per generated segment, so a
        # split is useful only when its shorter fused_fwd critical path repays
        # that overhead.  The integer comparisons below encode clean 890P API
        # A/B boundaries without floating-point dispatch decisions:
        #   * maxT < 4K: require raw_path > 2 * cp_path;
        #   * 4K <= maxT < 8K: the same strict rule, except raw_path == 2x is
        #     repeatable for H<=8 or an uneven batch with Be<2.  From 6K, also
        #     accept a 1.5x reduction while Be*H<=24: paired-DV correct_h0 makes
        #     the remaining CP fixed work small enough for this low-occupancy
        #     packed range;
        #   * maxT >= 8K: accept a 2x path reduction.  Also accept 1.5x while
        #     Be*H<=24 because fewer than 25 effective head tasks underfill 39 CUs;
        #   * maxT >= 16K and H<=24: the fixed preprocessing cost is amortized,
        #     and measured gains remain stable through Be*H<=30;
        #   * maxT >= 4K and Be*H<=30: a positive modeled reduction is enough.
        #     This retains low-occupancy uneven batches such as B=7,H=6 and
        #     B=2,H=24, while balanced B=8 grids exceed the effective-head cap
        #     and stay serial instead of paying an extra CP preprocessing wave.
        # Balanced high-Be grids fail these branches even when raw execution has
        # several waves; uneven batches may still benefit from splitting their
        # long sequence because Be is length-weighted rather than equal to B.
        effective_heads_le_24 = H * total_chunks <= 24 * max_chunks
        effective_heads_le_30 = H * total_chunks <= 30 * max_chunks
        effective_heads_below_one_wave = (
            H * total_chunks < fused_cu_count * max_chunks)
        if H <= 8:
            # Be*H<P alone is too permissive at low H: preprocessing still
            # dominates short or nearly uniform batches. Same-card slow-decay
            # A/B requires maxT>=3K. For B>=4, also require Be*H<20 so only a
            # genuinely long-tail batch enters CP.
            effective_heads_below_one_wave = (
                effective_heads_below_one_wave and max_seqlen >= 3072 and
                total_chunks < raw_batch * max_chunks and
                (raw_batch <= 3 or H * total_chunks < 20 * max_chunks)
            )
        elif H == 16:
            # Outside the short-tail low-grid island handled above, B=2 already
            # exposes 32 head tasks and remains slower with CP. B=3 amortizes
            # preprocessing from 3K; B>=4 needs 4K.
            effective_heads_below_one_wave = (
                effective_heads_below_one_wave and raw_batch >= 3 and
                max_seqlen >= (3072 if raw_batch == 3 else 4096)
            )
        elif H == 24:
            # Apart from the 6.1K--7.2K low-grid island handled above, the
            # half-filled B=2 transition is mixed below 8K and regresses by up
            # to 8% once Be exceeds 1.2. Keep only the measured long-tail fast
            # path below 8K. B>=3 also has to shorten the modeled fused path;
            # B=2 remains limited to the measured long-tail or 8K+ ranges.
            effective_heads_below_one_wave = (
                effective_heads_below_one_wave and
                ((raw_batch >= 3 and (
                    cp_wave_chunks < raw_wave_chunks or
                    # At 193 chunks, balancing the longest B=3 sequence
                    # changes from three to four segments.  The rounded
                    # wave*chunk estimate misses the resulting fused-path
                    # gain; same-card E2E stays serial through 192 chunks and
                    # favors CP from 193 chunks.
                    (raw_batch == 3 and max_chunks > 192))) or
                 max_seqlen >= 8192 or
                 5 * total_chunks <= 6 * max_chunks)
            )
        else:
            effective_heads_below_one_wave = False
        packed_min_tokens = 2048 if H <= 16 else cp_min_tokens(H)
        occupancy_ok = (
            max_seqlen >= packed_min_tokens and (
                # A length-weighted Be*H below one physical-CU wave means an
                # uneven serial batch cannot fill the device even when B*H
                # itself exceeds P. Splitting the long sequence redistributes
                # that idle tail. The H/B/length guards above keep this shortcut
                # only in ranges with repeatable complete-API gains.
                effective_heads_below_one_wave or
                (max_seqlen < 4096 and
                 raw_wave_chunks > cp_wave_chunks * 2) or
                (4096 <= max_seqlen < 8192 and
                 (raw_wave_chunks > cp_wave_chunks * 2 or
                  ((H <= 8 or total_chunks < 2 * max_chunks) and
                   raw_wave_chunks >= cp_wave_chunks * 2) or
                  (max_seqlen >= 6144 and effective_heads_le_24 and
                   raw_wave_chunks * 2 >= cp_wave_chunks * 3))) or
                (max_seqlen >= 8192 and
                 raw_wave_chunks >= cp_wave_chunks * 2) or
                (max_seqlen >= 8192 and effective_heads_le_24 and
                 raw_wave_chunks * 2 >= cp_wave_chunks * 3) or
                (H <= 24 and max_seqlen >= 16384 and
                 effective_heads_le_30) or
                (max_seqlen >= 4096 and effective_heads_le_30 and
                 raw_wave_chunks > cp_wave_chunks)
            )
        )
    if packed_low_grid_cap is not None:
        # The measured complete-API result is the missing occupancy-model
        # branch.  Length-floor and has-split gates remain mandatory below.
        occupancy_ok = True
    # These two reviewer-visible conditions remain mandatory regardless of the
    # Be*H_tasks/P branch above. `has_split` prevents paying CP preprocessing for an
    # identity plan; the active architecture's length floor prevents fixed
    # get_warmup/prepare_h/correct_h0 overhead from dominating short sequences.
    # Occupancy alone cannot override either condition.
    has_split = num_cp_segments > raw_batch
    if fused_head_shards != 1:
        tc01_h8_long_tail = (
            H == 8 and (
                (raw_batch == 2 and (
                    (max_seqlen >= 27800 and
                     5 * total_chunks <= 6 * max_chunks) or
                    (max_seqlen >= 15000 and min(seqlens) <= 512)
                )) or
                (raw_batch == 3 and max_seqlen >= 12500 and
                 10 * total_chunks <= 13 * max_chunks)
            )
        )
        if raw_batch == 1:
            min_tokens = _tc01_single_seq_cp_min_tokens(H)
        elif tc01_h8_long_tail:
            # The two tc01 DV shards usually make packed H=8 CP unnecessary.
            # Two measured exceptions remain: an almost-empty B=2 tail after a
            # 15K+ sequence, or a 27.8K+ long sequence with Be<=1.2; and B=3
            # with maxT>=12.5K and Be<=1.3. They leave the serial grid idle long
            # enough for the established CP plan to recover 3-10% API E2E.
            min_tokens = 0
        elif H > 8 or (H == 8 and raw_batch == 2):
            # Packed tc01 H>8 already exposes enough quarter-DV tasks before
            # CP. B=2/H=8 also regressed for 4:1 through 16:1 length ratios:
            # prepare_h/correct_h0 cost more than splitting recovers. Retain
            # H=8 only for B>=3, where uneven long batches showed a small but
            # repeatable gain, and retain H<=6 for their lower head occupancy.
            min_tokens = 1 << 60
        elif raw_batch >= 3 and H == 8:
            # B=3/H=8 remains serial through 20.9K: the two tc01 DV shards per
            # head already provide 48 tasks, while CP adds prepare/correct work.
            # Same-card API E2E first shows a repeatable CP gain at 22.3K for
            # both Hg=2 and Hg=4, so keep the boundary just below that sample.
            min_tokens = 22000
        elif raw_batch >= 3 or H == 6:
            min_tokens = 16384
        else:
            # B=2, H=4/8 uneven 4:1 batches are profitable from 8K.
            min_tokens = 8192
    else:
        min_tokens = (_single_seq_cp_min_tokens(
                          H, num_k_heads, max_seqlen, chunk_size)
                      if raw_batch == 1 else cp_min_tokens(H))
    use_cp = (
        occupancy_ok and has_split and max_seqlen >= min_tokens and
        _tc02_real_gate_cp_allowed(
            raw_cu, chunk_size, H, num_cp_segments
        )
    )

    if use_cp:
        cp_cu_seqlens = torch.tensor(cp_cu_seqlens, dtype=seqlen_dtype, device=device)
        seq_map_c2r = torch.tensor(seq_map_c2r, dtype=seqlen_dtype, device=device)
        seq_map_r2c = torch.tensor(seq_map_r2c, dtype=seqlen_dtype, device=device)
        # The binding consumes signed-char 0/1 values; construct int8 directly
        # instead of launching a bool-to-int8 conversion in the API hot path.
        ht_mask = torch.tensor(ht_mask, dtype=torch.int8, device=device)
    else:
        cp_cu_seqlens = seq_map_r2c = seq_map_c2r = ht_mask = None

    plan = (use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask)
    ready_event, active_streams = _plan_ready_event(plan)
    # Cache negative decisions too: repeatedly proving that a common short or
    # high-occupancy shape is serial has the same Python planning cost.  LRU
    # eviction also releases device tensors held only by an old positive plan.
    # Build privately, then publish the bounded cache in one slice assignment.
    # A racing miss may lose a recency update but cannot expose a partial entry
    # or return a plan for the wrong key.
    cache_order = [entry for entry in _cp_plan_cache
                   if entry[0] != cache_key]
    _cp_plan_cache[:] = [
        (cache_key, plan, ready_event, active_streams), *cache_order[:3]
    ]
    return plan


def prepare_chunk_offsets(cu_seqlens_i32: torch.Tensor,
                          chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """[batch+1] cumulative chunk count per sequence (CPU helper)."""
    if _sanity.enabled():
        _sanity.check(chunk_size > 0, "chunk_size must be positive")
        _sanity.require_host_read(cu_seqlens_i32, "chunk-offset planning")
        _sanity.offsets(cu_seqlens_i32, 2**31 - 1)
    cu_cpu = cu_seqlens_i32.cpu()
    batch_size = cu_cpu.size(0) - 1
    offsets = torch.zeros(batch_size + 1, dtype=torch.int32)
    for b in range(batch_size):
        seq_len = (cu_cpu[b + 1] - cu_cpu[b]).item()
        offsets[b + 1] = offsets[b] + _cdiv(seq_len, chunk_size)
    return offsets.to(cu_seqlens_i32.device)
