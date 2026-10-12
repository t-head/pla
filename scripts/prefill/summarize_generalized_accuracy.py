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

"""Validate and summarize sharded generalized-prefill accuracy JSONL files.

This is intentionally a standard-library-only post-processing gate.  It can run
on a host without torch, a GPU, or an installed PLA wheel after the device jobs
have copied their JSONL shards back.

Example::

    python scripts/prefill/summarize_generalized_accuracy.py \
      --expected-count 26266 \
      --expect-reference fla \
      --expect-tol 0.02 \
      --expect-inventory-file-count 20 \
      --expect-inventory-record-count 26266 \
      --expect-list-manifest-sha256 "$LIST_MANIFEST_SHA256" \
      --expect-gdn-path packed \
      --expect-gdn-state-abi indexed_pool \
      --expect-gdn-gating-abi raw \
      --expect-gdn-pla-extension-sha256 "$GDN_EXTENSION_SHA256" \
      --expect-kda-trusted-host-offsets true \
      --expect-kda-state-abi indexed_fp32_pool \
      --expect-kda-pla-extension-sha256 "$KDA_EXTENSION_SHA256" \
      /tmp/accuracy-shard*.jsonl

The gate rejects malformed rows, duplicate/missing/out-of-range indices, rows
that are not successful accuracy results, errors at or above the independently
specified tolerance, and run-identity/ABI metadata mismatches.  Optional strong
metrics catch errors that a whole-tensor relative norm can dilute.  P99 uses
the nearest-rank definition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


class AccuracyGateError(ValueError):
    """Raised when input shards cannot form a valid full accuracy run."""


@dataclass(frozen=True)
class MetricSummary:
    count: int
    out_max: float
    out_p99: float
    state_max: float
    state_p99: float


@dataclass(frozen=True)
class AccuracySummary:
    expected_count: int
    input_count: int
    overall: MetricSummary
    by_family: dict[str, MetricSummary]
    by_source: dict[tuple[str, str], MetricSummary]
    gdn_paths: dict[str, int]
    gdn_state_abis: dict[str, int]
    gdn_gating_abis: dict[str, int]
    kda_trusted_host_offsets: dict[str, int]
    kda_state_abis: dict[str, int]
    pla_extension_sha256s: dict[str, dict[str, int]]
    reference: str
    tolerance: float
    inventory_file_count: int
    inventory_record_count: int
    list_manifest_sha256: str
    strong_metric_maxima: dict[str, float]


STRONG_REL_METRICS = ("out_head_rel", "state_seq_head_rel")
STRONG_ABS_METRICS = ("out_max_abs", "state_max_abs")
KDA_STATE_ABIS = ("indexed_fp32_pool", "compact_bf16_final_state")
GDN_STATE_ABIS = ("legacy", "indexed_pool")
GDN_GATING_ABIS = ("activated", "raw")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _location(path: Path, line_number: int) -> str:
    return f"{path}:{line_number}"


def _read_rows(paths: Sequence[Path]) -> dict[int, tuple[dict[str, Any], str]]:
    if not paths:
        raise AccuracyGateError("no input JSONL files were supplied")

    rows: dict[int, tuple[dict[str, Any], str]] = {}
    for path in paths:
        if not path.is_file():
            raise AccuracyGateError(f"input is not a regular file: {path}")
        for line_number, raw in enumerate(path.read_text().splitlines(), 1):
            location = _location(path, line_number)
            if not raw.strip():
                raise AccuracyGateError(f"empty JSONL row at {location}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AccuracyGateError(
                    f"invalid JSON at {location}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise AccuracyGateError(
                    f"JSON row at {location} must be an object"
                )
            index = row.get("index")
            if type(index) is not int:  # bool is an int subclass; reject it.
                raise AccuracyGateError(
                    f"index at {location} must be an integer, got {index!r}"
                )
            if index in rows:
                previous = rows[index][1]
                raise AccuracyGateError(
                    f"duplicate index {index}: {previous} and {location}"
                )
            rows[index] = (row, location)
    return rows


def _finite_nonnegative_metric(row: dict[str, Any], key: str, location: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AccuracyGateError(
            f"{key} at {location} must be a finite non-negative number, "
            f"got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AccuracyGateError(
            f"{key} at {location} must be a finite non-negative number, "
            f"got {value!r}"
        )
    return result


def _strict_int(row: dict[str, Any], key: str, location: str) -> int:
    value = row.get(key)
    if type(value) is not int:  # reject bool and numeric strings
        raise AccuracyGateError(
            f"{key} at {location} must be an integer, got {value!r}"
        )
    return value


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AccuracyGateError(
            f"{label} must be a lowercase 64-character SHA256, got {value!r}"
        )
    return value


def _require_below_tol(
    value: float, key: str, tolerance: float, location: str,
) -> None:
    if value >= tolerance:
        raise AccuracyGateError(
            f"{key} at {location} must be < tol={tolerance!r}, got {value!r}"
        )


def _nearest_rank_p99(values: Sequence[float]) -> float:
    if not values:
        raise AccuracyGateError("cannot calculate p99 for an empty group")
    ordered = sorted(values)
    rank = max(1, math.ceil(0.99 * len(ordered)))
    return ordered[rank - 1]


def _metric_summary(samples: Sequence[tuple[float, float]]) -> MetricSummary:
    if not samples:
        raise AccuracyGateError("cannot summarize an empty group")
    out_values = [sample[0] for sample in samples]
    state_values = [sample[1] for sample in samples]
    return MetricSummary(
        count=len(samples),
        out_max=max(out_values),
        out_p99=_nearest_rank_p99(out_values),
        state_max=max(state_values),
        state_p99=_nearest_rank_p99(state_values),
    )


def summarize_accuracy(
    paths: Sequence[Path],
    expected_count: int,
    *,
    expected_reference: str,
    expected_tolerance: float,
    expected_inventory_file_count: int,
    expected_inventory_record_count: int,
    expected_list_manifest_sha256: str,
    expected_gdn_pla_extension_sha256: str | None = None,
    expected_kda_pla_extension_sha256: str | None = None,
    expected_gdn_path: str | None = None,
    expected_gdn_state_abi: str | None = None,
    expected_gdn_gating_abi: str | None = None,
    expected_kda_trusted_host_offsets: bool | None = None,
    expected_kda_state_abi: str | None = None,
    require_strong_metrics: bool = False,
) -> AccuracySummary:
    """Validate ``paths`` and return deterministic aggregate statistics.

    Run-identity arguments are mandatory: every shard row must come from the
    same oracle, tolerance, inventory, exact case-list manifest, and expected
    family-specific PLA extension binary.
    GDN path/state/gating expectations apply only to GDN rows.  The indexed
    GDN state ABI additionally requires ``unmapped_state_unchanged`` to be the
    JSON boolean true.  Trusted-host-offset and KDA state-ABI expectations
    apply only to KDA rows; the former requires a JSON boolean, not a truthy
    integer or string.  The indexed KDA state ABI also requires
    ``unmapped_state_unchanged`` to be the JSON boolean true.  When
    ``require_strong_metrics`` is true, every row must also contain scalar
    ``out_head_rel`` and ``state_seq_head_rel`` values below the tolerance plus
    finite, non-negative ``out_max_abs`` and ``state_max_abs`` values.
    """

    if expected_count < 0:
        raise AccuracyGateError("expected_count must be non-negative")
    if expected_reference not in ("fla", "serial"):
        raise AccuracyGateError(
            "expected_reference must be 'fla' or 'serial', got "
            f"{expected_reference!r}"
        )
    if (
        isinstance(expected_tolerance, bool)
        or not isinstance(expected_tolerance, (int, float))
        or not math.isfinite(float(expected_tolerance))
        or float(expected_tolerance) <= 0.0
    ):
        raise AccuracyGateError(
            "expected_tolerance must be a finite positive number, got "
            f"{expected_tolerance!r}"
        )
    expected_tolerance = float(expected_tolerance)
    if type(expected_inventory_file_count) is not int or expected_inventory_file_count <= 0:
        raise AccuracyGateError(
            "expected_inventory_file_count must be a positive integer, got "
            f"{expected_inventory_file_count!r}"
        )
    if type(expected_inventory_record_count) is not int or expected_inventory_record_count <= 0:
        raise AccuracyGateError(
            "expected_inventory_record_count must be a positive integer, got "
            f"{expected_inventory_record_count!r}"
        )
    if expected_inventory_record_count != expected_count:
        raise AccuracyGateError(
            "expected_inventory_record_count must equal expected_count for a "
            f"full gate, got {expected_inventory_record_count} and {expected_count}"
        )
    expected_list_manifest_sha256 = _validate_sha256(
        expected_list_manifest_sha256, "expected_list_manifest_sha256"
    )
    if expected_gdn_pla_extension_sha256 is not None:
        expected_gdn_pla_extension_sha256 = _validate_sha256(
            expected_gdn_pla_extension_sha256,
            "expected_gdn_pla_extension_sha256",
        )
    if expected_kda_pla_extension_sha256 is not None:
        expected_kda_pla_extension_sha256 = _validate_sha256(
            expected_kda_pla_extension_sha256,
            "expected_kda_pla_extension_sha256",
        )
    if expected_gdn_path == "":
        raise AccuracyGateError("expected_gdn_path must not be empty")
    if (
        expected_gdn_state_abi is not None
        and expected_gdn_state_abi not in GDN_STATE_ABIS
    ):
        raise AccuracyGateError(
            f"expected_gdn_state_abi must be one of {GDN_STATE_ABIS}, got "
            f"{expected_gdn_state_abi!r}"
        )
    if (
        expected_gdn_gating_abi is not None
        and expected_gdn_gating_abi not in GDN_GATING_ABIS
    ):
        raise AccuracyGateError(
            f"expected_gdn_gating_abi must be one of {GDN_GATING_ABIS}, got "
            f"{expected_gdn_gating_abi!r}"
        )
    if (
        expected_kda_state_abi is not None
        and expected_kda_state_abi not in KDA_STATE_ABIS
    ):
        raise AccuracyGateError(
            f"expected_kda_state_abi must be one of {KDA_STATE_ABIS}, got "
            f"{expected_kda_state_abi!r}"
        )

    indexed_rows = _read_rows(paths)
    expected_indices = set(range(expected_count))
    actual_indices = set(indexed_rows)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise AccuracyGateError(
            "index coverage mismatch: "
            f"expected=0..{expected_count - 1} records={expected_count}, "
            f"actual_records={len(actual_indices)}, "
            f"missing_count={len(missing)} missing_first={missing[:20]}, "
            f"extra_count={len(extra)} extra_first={extra[:20]}"
        )

    overall_samples: list[tuple[float, float]] = []
    family_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    source_samples: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    gdn_paths: Counter[str] = Counter()
    gdn_state_abis: Counter[str] = Counter()
    gdn_gating_abis: Counter[str] = Counter()
    kda_host_offsets: Counter[str] = Counter()
    kda_state_abis: Counter[str] = Counter()
    pla_extension_sha256s: dict[str, Counter[str]] = defaultdict(Counter)
    strong_metric_maxima = {
        key: 0.0 for key in (*STRONG_REL_METRICS, *STRONG_ABS_METRICS)
    } if require_strong_metrics else {}
    saw_expected_gdn_metadata = False
    saw_expected_kda_metadata = False

    for index in range(expected_count):
        row, location = indexed_rows[index]
        if row.get("mode") != "accuracy":
            raise AccuracyGateError(
                f"row index {index} at {location} is not an accuracy result: "
                f"mode={row.get('mode')!r}"
            )
        if row.get("reference") != expected_reference:
            raise AccuracyGateError(
                f"reference mismatch at {location}: expected "
                f"{expected_reference!r}, got {row.get('reference')!r}"
            )
        row_tolerance = _finite_nonnegative_metric(row, "tol", location)
        if row_tolerance != expected_tolerance:
            raise AccuracyGateError(
                f"tol mismatch at {location}: expected "
                f"{expected_tolerance!r}, got {row_tolerance!r}"
            )
        inventory_file_count = _strict_int(
            row, "inventory_file_count", location
        )
        if inventory_file_count != expected_inventory_file_count:
            raise AccuracyGateError(
                f"inventory_file_count mismatch at {location}: expected "
                f"{expected_inventory_file_count}, got {inventory_file_count}"
            )
        inventory_record_count = _strict_int(
            row, "inventory_record_count", location
        )
        if inventory_record_count != expected_inventory_record_count:
            raise AccuracyGateError(
                f"inventory_record_count mismatch at {location}: expected "
                f"{expected_inventory_record_count}, got {inventory_record_count}"
            )
        manifest_sha256 = _validate_sha256(
            row.get("list_manifest_sha256"),
            f"list_manifest_sha256 at {location}",
        )
        if manifest_sha256 != expected_list_manifest_sha256:
            raise AccuracyGateError(
                f"list_manifest_sha256 mismatch at {location}: expected "
                f"{expected_list_manifest_sha256!r}, got {manifest_sha256!r}"
            )
        if row.get("pass") is not True:
            detail = row.get("error")
            suffix = f" error={detail!r}" if detail is not None else ""
            raise AccuracyGateError(
                f"row index {index} at {location} did not pass:{suffix}"
            )

        family = row.get("family")
        if family not in ("gdn", "kda"):
            raise AccuracyGateError(
                f"family at {location} must be 'gdn' or 'kda', got {family!r}"
            )
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise AccuracyGateError(
                f"source at {location} must be a non-empty string, got {source!r}"
            )
        extension_sha256 = _validate_sha256(
            row.get("pla_extension_sha256"),
            f"pla_extension_sha256 at {location}",
        )
        expected_extension_sha256 = (
            expected_gdn_pla_extension_sha256
            if family == "gdn"
            else expected_kda_pla_extension_sha256
        )
        if expected_extension_sha256 is None:
            raise AccuracyGateError(
                f"expected {family} pla_extension_sha256 was not supplied"
            )
        if extension_sha256 != expected_extension_sha256:
            raise AccuracyGateError(
                f"pla_extension_sha256 mismatch at {location}: expected "
                f"{expected_extension_sha256!r}, got {extension_sha256!r}"
            )
        pla_extension_sha256s[family][extension_sha256] += 1

        out_rel = _finite_nonnegative_metric(row, "out_rel", location)
        state_rel = _finite_nonnegative_metric(row, "state_rel", location)
        # Never trust the producer's pass bit or its chosen tolerance.  The
        # aggregation host independently applies the expected threshold.
        _require_below_tol(out_rel, "out_rel", expected_tolerance, location)
        _require_below_tol(state_rel, "state_rel", expected_tolerance, location)
        if require_strong_metrics:
            for key in STRONG_REL_METRICS:
                value = _finite_nonnegative_metric(row, key, location)
                _require_below_tol(value, key, expected_tolerance, location)
                strong_metric_maxima[key] = max(strong_metric_maxima[key], value)
            for key in STRONG_ABS_METRICS:
                value = _finite_nonnegative_metric(row, key, location)
                strong_metric_maxima[key] = max(strong_metric_maxima[key], value)
        sample = (out_rel, state_rel)
        overall_samples.append(sample)
        family_samples[family].append(sample)
        source_samples[(family, source)].append(sample)

        if family == "gdn":
            path_value = row.get("gdn_path", "<missing>")
            gdn_paths[str(path_value)] += 1
            if expected_gdn_path is not None:
                saw_expected_gdn_metadata = True
                if row.get("gdn_path") != expected_gdn_path:
                    raise AccuracyGateError(
                        f"gdn_path mismatch at {location}: "
                        f"expected {expected_gdn_path!r}, "
                        f"got {row.get('gdn_path')!r}"
                    )
            state_abi = row.get("gdn_state_abi", "<missing>")
            gdn_state_abis[str(state_abi)] += 1
            if expected_gdn_state_abi is not None:
                if not isinstance(state_abi, str):
                    raise AccuracyGateError(
                        f"gdn_state_abi at {location} must be a string, got "
                        f"{state_abi!r}"
                    )
                if state_abi != expected_gdn_state_abi:
                    raise AccuracyGateError(
                        f"gdn_state_abi mismatch at {location}: expected "
                        f"{expected_gdn_state_abi!r}, got {state_abi!r}"
                    )
            gating_abi = row.get("gdn_gating_abi", "<missing>")
            gdn_gating_abis[str(gating_abi)] += 1
            if expected_gdn_gating_abi is not None:
                if not isinstance(gating_abi, str):
                    raise AccuracyGateError(
                        f"gdn_gating_abi at {location} must be a string, got "
                        f"{gating_abi!r}"
                    )
                if gating_abi != expected_gdn_gating_abi:
                    raise AccuracyGateError(
                        f"gdn_gating_abi mismatch at {location}: expected "
                        f"{expected_gdn_gating_abi!r}, got {gating_abi!r}"
                    )
            if expected_gdn_state_abi == "indexed_pool":
                unmapped_unchanged = row.get("unmapped_state_unchanged")
                if unmapped_unchanged is not True:
                    raise AccuracyGateError(
                        "unmapped_state_unchanged at "
                        f"{location} must be the JSON boolean true for GDN "
                        f"indexed_pool, got {unmapped_unchanged!r}"
                    )
        else:
            host_value = row.get("kda_trusted_host_offsets", "<missing>")
            if type(host_value) is bool:
                host_label = str(host_value).lower()
            else:
                host_label = str(host_value)
            kda_host_offsets[host_label] += 1
            if expected_kda_trusted_host_offsets is not None:
                saw_expected_kda_metadata = True
                if type(host_value) is not bool or host_value is not expected_kda_trusted_host_offsets:
                    raise AccuracyGateError(
                        "kda_trusted_host_offsets mismatch at "
                        f"{location}: expected "
                        f"{expected_kda_trusted_host_offsets!r}, got {host_value!r}"
                    )
            state_abi = row.get("kda_state_abi", "<missing>")
            kda_state_abis[str(state_abi)] += 1
            if expected_kda_state_abi is not None:
                if state_abi != expected_kda_state_abi:
                    raise AccuracyGateError(
                        f"kda_state_abi mismatch at {location}: expected "
                        f"{expected_kda_state_abi!r}, got {state_abi!r}"
                    )
            if expected_kda_state_abi == "indexed_fp32_pool":
                unmapped_unchanged = row.get("unmapped_state_unchanged")
                if unmapped_unchanged is not True:
                    raise AccuracyGateError(
                        "unmapped_state_unchanged at "
                        f"{location} must be the JSON boolean true for "
                        "indexed_fp32_pool, got "
                        f"{unmapped_unchanged!r}"
                    )

    if expected_count == 0:
        raise AccuracyGateError("an accuracy gate cannot summarize zero records")
    if expected_gdn_path is not None and not saw_expected_gdn_metadata:
        raise AccuracyGateError(
            "--expect-gdn-path was supplied, but the input contains no GDN rows"
        )
    if expected_gdn_state_abi is not None and not gdn_state_abis:
        raise AccuracyGateError(
            "--expect-gdn-state-abi was supplied, but the input contains no "
            "GDN rows"
        )
    if expected_gdn_gating_abi is not None and not gdn_gating_abis:
        raise AccuracyGateError(
            "--expect-gdn-gating-abi was supplied, but the input contains no "
            "GDN rows"
        )
    if (
        expected_kda_trusted_host_offsets is not None
        and not saw_expected_kda_metadata
    ):
        raise AccuracyGateError(
            "--expect-kda-trusted-host-offsets was supplied, but the input "
            "contains no KDA rows"
        )
    if expected_kda_state_abi is not None and not kda_state_abis:
        raise AccuracyGateError(
            "--expect-kda-state-abi was supplied, but the input contains no "
            "KDA rows"
        )

    return AccuracySummary(
        expected_count=expected_count,
        input_count=len(paths),
        overall=_metric_summary(overall_samples),
        by_family={
            family: _metric_summary(samples)
            for family, samples in sorted(family_samples.items())
        },
        by_source={
            key: _metric_summary(samples)
            for key, samples in sorted(source_samples.items())
        },
        gdn_paths=dict(sorted(gdn_paths.items())),
        gdn_state_abis=dict(sorted(gdn_state_abis.items())),
        gdn_gating_abis=dict(sorted(gdn_gating_abis.items())),
        kda_trusted_host_offsets=dict(sorted(kda_host_offsets.items())),
        kda_state_abis=dict(sorted(kda_state_abis.items())),
        pla_extension_sha256s={
            family: dict(sorted(hashes.items()))
            for family, hashes in sorted(pla_extension_sha256s.items())
        },
        reference=expected_reference,
        tolerance=expected_tolerance,
        inventory_file_count=expected_inventory_file_count,
        inventory_record_count=expected_inventory_record_count,
        list_manifest_sha256=expected_list_manifest_sha256,
        strong_metric_maxima=strong_metric_maxima,
    )


def _format_metrics(label: str, metrics: MetricSummary) -> str:
    return (
        f"{label}: count={metrics.count} "
        f"out_rel[p99={metrics.out_p99:.6e} max={metrics.out_max:.6e}] "
        f"state_rel[p99={metrics.state_p99:.6e} max={metrics.state_max:.6e}]"
    )


def format_summary(summary: AccuracySummary) -> str:
    lines = [
        f"PASS accuracy gate: records={summary.expected_count} "
        f"input_files={summary.input_count} indices=0..{summary.expected_count - 1}",
        f"run_identity: reference={summary.reference} tol={summary.tolerance!r} "
        f"inventory_files={summary.inventory_file_count} "
        f"inventory_records={summary.inventory_record_count} "
        f"list_manifest_sha256={summary.list_manifest_sha256}",
        _format_metrics("overall", summary.overall),
        "families:",
    ]
    for family, metrics in summary.by_family.items():
        lines.append(_format_metrics(f"  {family}", metrics))
    lines.append("sources:")
    for (family, source), metrics in summary.by_source.items():
        lines.append(_format_metrics(f"  {family}/{source}", metrics))
    lines.extend((
        "abi_metadata:",
        f"  gdn_path={summary.gdn_paths}",
        f"  gdn_state_abi={summary.gdn_state_abis}",
        f"  gdn_gating_abi={summary.gdn_gating_abis}",
        "  kda_trusted_host_offsets="
        f"{summary.kda_trusted_host_offsets}",
        f"  kda_state_abi={summary.kda_state_abis}",
        f"  pla_extension_sha256={summary.pla_extension_sha256s}",
    ))
    if summary.strong_metric_maxima:
        lines.append(
            "strong_metric_maxima: "
            + " ".join(
                f"{key}={value:.6e}"
                for key, value in summary.strong_metric_maxima.items()
            )
        )
    return "\n".join(lines)


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate and summarize sharded generalized-prefill "
            "accuracy JSONL results without importing torch."
        )
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="accuracy JSONL shards")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--expect-reference", choices=("fla", "serial"), required=True,
        help="require this independent-or-serial oracle on every row",
    )
    parser.add_argument(
        "--expect-tol", type=float, required=True,
        help="require this tolerance and independently apply it to every row",
    )
    parser.add_argument(
        "--expect-inventory-file-count", type=int, required=True,
        help="require this case-list file count on every row",
    )
    parser.add_argument(
        "--expect-inventory-record-count", type=int, required=True,
        help="require this full inventory record count on every row",
    )
    parser.add_argument(
        "--expect-list-manifest-sha256", required=True,
        help="require this exact lowercase SHA256 case-list manifest identity",
    )
    parser.add_argument(
        "--expect-gdn-pla-extension-sha256",
        help=(
            "require this exact lowercase SHA256 in pla_extension_sha256 on "
            "every GDN row; required when the input contains GDN rows"
        ),
    )
    parser.add_argument(
        "--expect-kda-pla-extension-sha256",
        help=(
            "require this exact lowercase SHA256 in pla_extension_sha256 on "
            "every KDA row; required when the input contains KDA rows"
        ),
    )
    parser.add_argument(
        "--expect-gdn-path",
        help="require this gdn_path value on every GDN row",
    )
    parser.add_argument(
        "--expect-gdn-state-abi",
        choices=GDN_STATE_ABIS,
        help=(
            "require this state ABI on every GDN row; indexed_pool also "
            "requires unmapped_state_unchanged=true"
        ),
    )
    parser.add_argument(
        "--expect-gdn-gating-abi",
        choices=GDN_GATING_ABIS,
        help="require this gating ABI on every GDN row",
    )
    parser.add_argument(
        "--expect-kda-trusted-host-offsets",
        type=_parse_bool,
        metavar="{true,false}",
        help=(
            "require this JSON boolean in kda_trusted_host_offsets on every "
            "KDA row"
        ),
    )
    parser.add_argument(
        "--expect-kda-state-abi",
        choices=KDA_STATE_ABIS,
        help=(
            "require this serving-state ABI on every KDA row; indexed state "
            "also requires unmapped_state_unchanged=true"
        ),
    )
    parser.add_argument(
        "--require-strong-metrics", action="store_true",
        help=(
            "require per-head/sequence relative metrics below tolerance and "
            "finite non-negative output/state max-absolute metrics"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = summarize_accuracy(
            args.inputs,
            args.expected_count,
            expected_reference=args.expect_reference,
            expected_tolerance=args.expect_tol,
            expected_inventory_file_count=args.expect_inventory_file_count,
            expected_inventory_record_count=args.expect_inventory_record_count,
            expected_list_manifest_sha256=args.expect_list_manifest_sha256,
            expected_gdn_pla_extension_sha256=(
                args.expect_gdn_pla_extension_sha256
            ),
            expected_kda_pla_extension_sha256=(
                args.expect_kda_pla_extension_sha256
            ),
            expected_gdn_path=args.expect_gdn_path,
            expected_gdn_state_abi=args.expect_gdn_state_abi,
            expected_gdn_gating_abi=args.expect_gdn_gating_abi,
            expected_kda_trusted_host_offsets=(
                args.expect_kda_trusted_host_offsets
            ),
            expected_kda_state_abi=args.expect_kda_state_abi,
            require_strong_metrics=args.require_strong_metrics,
        )
    except (AccuracyGateError, OSError) as exc:
        print(f"FAIL accuracy gate: {exc}", file=sys.stderr)
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
