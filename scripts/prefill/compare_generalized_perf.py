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

"""Compare two sharded ``generalized_prefill_replay.py --mode perf`` runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Sequence


class PerfComparisonError(ValueError):
    """Raised when inputs cannot form a valid performance comparison."""


IDENTITY_FIELDS = (
    "family", "source", "line", "shape_id", "H", "Hg", "N", "T",
    "warmup", "repeat", "record_first_call", "timing_method",
    "first_call_timing_method", "seed", "shard", "num_shards",
    "selection_sha256",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def parse_bool(text: str) -> bool:
    value = text.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def _location(side: str, path: Path, line_number: int) -> str:
    return f"{side} {path}:{line_number}"


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PerfComparisonError(
            f"{label} must be a lowercase 64-character SHA256, got {value!r}"
        )
    return value


def _validate_identity_metadata(row: dict[str, Any], location: str) -> None:
    missing = [field for field in IDENTITY_FIELDS if field not in row]
    if missing:
        raise PerfComparisonError(
            f"missing identity metadata at {location}: {missing}"
        )
    if row["family"] not in ("gdn", "kda"):
        raise PerfComparisonError(
            f"family at {location} must be 'gdn' or 'kda', "
            f"got {row['family']!r}"
        )
    for field in (
        "source", "shape_id", "timing_method", "first_call_timing_method",
        "selection_sha256",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise PerfComparisonError(
                f"{field} at {location} must be a non-empty string, "
                f"got {row[field]!r}"
            )
    for field in (
        "line", "H", "Hg", "N", "T", "warmup", "repeat", "seed",
        "shard", "num_shards",
    ):
        if type(row[field]) is not int:
            raise PerfComparisonError(
                f"{field} at {location} must be an integer, got {row[field]!r}"
            )
    if row["warmup"] < 0 or row["repeat"] <= 0:
        raise PerfComparisonError(
            f"invalid warmup/repeat at {location}: "
            f"{row['warmup']}/{row['repeat']}"
        )
    if row["num_shards"] <= 0 or not 0 <= row["shard"] < row["num_shards"]:
        raise PerfComparisonError(
            f"invalid shard metadata at {location}: "
            f"{row['shard']}/{row['num_shards']}"
        )
    if type(row["record_first_call"]) is not bool:
        raise PerfComparisonError(
            "record_first_call at "
            f"{location} must be a boolean, got {row['record_first_call']!r}"
        )


def _finite_positive_ms(row: dict[str, Any], field: str, location: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerfComparisonError(
            f"{field} at {location} must be a finite positive number, "
            f"got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PerfComparisonError(
            f"{field} at {location} must be a finite positive number, "
            f"got {value!r}"
        )
    return result


def _validate_expected_abi(
    row: dict[str, Any],
    location: str,
    *,
    expected_gdn_path: str | None,
    expected_gdn_state_abi: str | None,
    expected_gdn_gating_abi: str | None,
    expected_kda_trusted_host_offsets: bool | None,
    expected_kda_cp_plan_cache_size_env: str | None,
    expected_kda_state_abi: str | None,
    expected_kda_beta_abi: str | None,
) -> None:
    if row["family"] == "gdn":
        for field, expected in (
            ("gdn_path", expected_gdn_path),
            ("gdn_state_abi", expected_gdn_state_abi),
            ("gdn_gating_abi", expected_gdn_gating_abi),
        ):
            if expected is None:
                continue
            actual = row.get(field)
            if type(actual) is not str or actual != expected:
                raise PerfComparisonError(
                    f"{field} mismatch at {location}: "
                    f"expected {expected!r}, got {actual!r}"
                )
    if row["family"] != "kda":
        return
    if expected_kda_trusted_host_offsets is not None:
        actual = row.get("kda_trusted_host_offsets")
        if (
            type(actual) is not bool
            or actual is not expected_kda_trusted_host_offsets
        ):
            raise PerfComparisonError(
                f"kda_trusted_host_offsets mismatch at {location}: "
                f"expected {expected_kda_trusted_host_offsets!r}, got {actual!r}"
            )
    if expected_kda_cp_plan_cache_size_env is not None:
        actual = row.get("kda_cp_plan_cache_size_env")
        if (
            type(actual) is not str
            or actual != expected_kda_cp_plan_cache_size_env
        ):
            raise PerfComparisonError(
                f"kda_cp_plan_cache_size_env mismatch at {location}: "
                f"expected {expected_kda_cp_plan_cache_size_env!r}, "
                f"got {actual!r}"
            )
    for field, expected in (
        ("kda_state_abi", expected_kda_state_abi),
        ("kda_beta_abi", expected_kda_beta_abi),
    ):
        if expected is None:
            continue
        actual = row.get(field)
        if type(actual) is not str or actual != expected:
            raise PerfComparisonError(
                f"{field} mismatch at {location}: "
                f"expected {expected!r}, got {actual!r}"
            )


def _validate_expected_snapshot(
    row: dict[str, Any],
    location: str,
    *,
    expected_inventory_file_count: int | None,
    expected_inventory_record_count: int | None,
    expected_list_manifest_sha256: str | None,
) -> None:
    for field, expected in (
        ("inventory_file_count", expected_inventory_file_count),
        ("inventory_record_count", expected_inventory_record_count),
    ):
        if expected is None:
            continue
        actual = row.get(field)
        if type(actual) is not int or actual != expected:
            raise PerfComparisonError(
                f"{field} mismatch at {location}: "
                f"expected {expected!r}, got {actual!r}"
            )
    if expected_list_manifest_sha256 is not None:
        actual = row.get("list_manifest_sha256")
        if type(actual) is not str or actual != expected_list_manifest_sha256:
            raise PerfComparisonError(
                f"list_manifest_sha256 mismatch at {location}: "
                f"expected {expected_list_manifest_sha256!r}, got {actual!r}"
            )


def _validate_perf_row(
    row: dict[str, Any],
    location: str,
    *,
    expected_gdn_path: str | None,
    expected_gdn_state_abi: str | None,
    expected_gdn_gating_abi: str | None,
    expected_kda_trusted_host_offsets: bool | None,
    expected_kda_cp_plan_cache_size_env: str | None,
    expected_kda_state_abi: str | None,
    expected_kda_beta_abi: str | None,
    expected_inventory_file_count: int | None,
    expected_inventory_record_count: int | None,
    expected_list_manifest_sha256: str | None,
    expected_gdn_pla_extension_sha256: str | None,
    expected_kda_pla_extension_sha256: str | None,
    require_first_call_ms: bool,
) -> tuple[float, float | None]:
    if row.get("mode") != "perf":
        raise PerfComparisonError(
            f"row at {location} is not a perf result: mode={row.get('mode')!r}"
        )
    if row.get("pass") is not True:
        suffix = (
            f" error={row['error']!r}" if row.get("error") is not None else ""
        )
        raise PerfComparisonError(f"row at {location} did not pass:{suffix}")
    _validate_identity_metadata(row, location)
    extension_sha256 = _validate_sha256(
        row.get("pla_extension_sha256"),
        f"pla_extension_sha256 at {location}",
    )
    expected_extension_sha256 = (
        expected_gdn_pla_extension_sha256
        if row["family"] == "gdn"
        else expected_kda_pla_extension_sha256
    )
    if expected_extension_sha256 is None:
        raise PerfComparisonError(
            f"expected {row['family']} pla_extension_sha256 was not supplied "
            f"for {location}"
        )
    if extension_sha256 != expected_extension_sha256:
        raise PerfComparisonError(
            f"pla_extension_sha256 mismatch at {location}: expected "
            f"{expected_extension_sha256!r}, got {extension_sha256!r}"
        )
    ms = _finite_positive_ms(row, "ms", location)
    record_first_call = row["record_first_call"]
    if require_first_call_ms and not record_first_call:
        raise PerfComparisonError(
            f"first_call_ms is required at {location}, but "
            "record_first_call is false"
        )
    first_call_ms = None
    if record_first_call:
        first_call_ms = _finite_positive_ms(row, "first_call_ms", location)
    elif "first_call_ms" in row:
        raise PerfComparisonError(
            f"first_call_ms is present at {location}, but "
            "record_first_call is false"
        )
    _validate_expected_abi(
        row,
        location,
        expected_gdn_path=expected_gdn_path,
        expected_gdn_state_abi=expected_gdn_state_abi,
        expected_gdn_gating_abi=expected_gdn_gating_abi,
        expected_kda_trusted_host_offsets=expected_kda_trusted_host_offsets,
        expected_kda_cp_plan_cache_size_env=(
            expected_kda_cp_plan_cache_size_env
        ),
        expected_kda_state_abi=expected_kda_state_abi,
        expected_kda_beta_abi=expected_kda_beta_abi,
    )
    _validate_expected_snapshot(
        row,
        location,
        expected_inventory_file_count=expected_inventory_file_count,
        expected_inventory_record_count=expected_inventory_record_count,
        expected_list_manifest_sha256=expected_list_manifest_sha256,
    )
    return ms, first_call_ms


def _require_same_identity(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    context: str,
) -> None:
    mismatches = {
        field: (before[field], after[field])
        for field in IDENTITY_FIELDS
        if before[field] != after[field]
    }
    if mismatches:
        raise PerfComparisonError(
            f"identity metadata mismatch for {context}: {mismatches}"
        )


def read_rows(
    paths: Sequence[Path],
    aggregate_runs: bool = False,
    *,
    side: str = "input",
    expected_gdn_path: str | None = None,
    expected_gdn_state_abi: str | None = None,
    expected_gdn_gating_abi: str | None = None,
    expected_kda_trusted_host_offsets: bool | None = None,
    expected_kda_cp_plan_cache_size_env: str | None = None,
    expected_kda_state_abi: str | None = None,
    expected_kda_beta_abi: str | None = None,
    expected_inventory_file_count: int | None = None,
    expected_inventory_record_count: int | None = None,
    expected_list_manifest_sha256: str | None = None,
    expected_gdn_pla_extension_sha256: str | None = None,
    expected_kda_pla_extension_sha256: str | None = None,
    require_first_call_ms: bool = False,
) -> dict[int, dict[str, Any]]:
    samples: dict[
        int, list[tuple[dict[str, Any], str, float, float | None]]
    ] = {}
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
    saw_gdn = saw_kda = False
    for path in paths:
        if not path.is_file():
            raise PerfComparisonError(f"{side} input is not a regular file: {path}")
        for line_number, raw in enumerate(path.read_text().splitlines(), 1):
            location = _location(side, path, line_number)
            if not raw.strip():
                raise PerfComparisonError(f"empty JSONL row at {location}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PerfComparisonError(
                    f"invalid JSON at {location}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise PerfComparisonError(
                    f"JSON row at {location} must be an object"
                )
            index = row.get("index")
            if type(index) is not int:
                raise PerfComparisonError(
                    f"index at {location} must be an integer, got {index!r}"
                )
            ms, first_call_ms = _validate_perf_row(
                row,
                location,
                expected_gdn_path=expected_gdn_path,
                expected_gdn_state_abi=expected_gdn_state_abi,
                expected_gdn_gating_abi=expected_gdn_gating_abi,
                expected_kda_trusted_host_offsets=(
                    expected_kda_trusted_host_offsets
                ),
                expected_kda_cp_plan_cache_size_env=(
                    expected_kda_cp_plan_cache_size_env
                ),
                expected_kda_state_abi=expected_kda_state_abi,
                expected_kda_beta_abi=expected_kda_beta_abi,
                expected_inventory_file_count=expected_inventory_file_count,
                expected_inventory_record_count=expected_inventory_record_count,
                expected_list_manifest_sha256=expected_list_manifest_sha256,
                expected_gdn_pla_extension_sha256=(
                    expected_gdn_pla_extension_sha256
                ),
                expected_kda_pla_extension_sha256=(
                    expected_kda_pla_extension_sha256
                ),
                require_first_call_ms=require_first_call_ms,
            )
            saw_gdn |= row["family"] == "gdn"
            saw_kda |= row["family"] == "kda"
            if index in samples and not aggregate_runs:
                previous = samples[index][0][1]
                raise PerfComparisonError(
                    f"duplicate index {index}: {previous} and {location}"
                )
            samples.setdefault(index, []).append(
                (row, location, ms, first_call_ms)
            )
    if not samples:
        raise PerfComparisonError(f"{side} record set is empty")
    if (
        expected_gdn_path is not None
        or expected_gdn_state_abi is not None
        or expected_gdn_gating_abi is not None
    ) and not saw_gdn:
        raise PerfComparisonError(
            f"expected {side} GDN metadata, but {side} contains no GDN rows"
        )
    if (
        expected_kda_trusted_host_offsets is not None
        or expected_kda_cp_plan_cache_size_env is not None
        or expected_kda_state_abi is not None
        or expected_kda_beta_abi is not None
    ) and not saw_kda:
        raise PerfComparisonError(
            f"expected {side} KDA metadata, but {side} contains no KDA rows"
        )

    rows: dict[int, dict[str, Any]] = {}
    for index, index_samples in samples.items():
        first, first_location, first_ms, first_first_call_ms = index_samples[0]
        for sample, sample_location, _, _ in index_samples[1:]:
            _require_same_identity(
                first,
                sample,
                context=(
                    f"{side} duplicate index {index} "
                    f"({first_location} vs {sample_location})"
                ),
            )
        row = dict(first)
        row["ms"] = first_ms
        if first_first_call_ms is not None:
            row["first_call_ms"] = first_first_call_ms
        if len(index_samples) > 1:
            row["pass"] = True
            row["ms"] = statistics.median(sample[2] for sample in index_samples)
            if first_first_call_ms is not None:
                row["first_call_ms"] = statistics.median(
                    sample[3] for sample in index_samples
                    if sample[3] is not None
                )
            row["aggregate_runs"] = len(index_samples)
        rows[index] = row
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[round((len(ordered) - 1) * fraction)]


def _validate_expected_coverage(
    indices: set[int], expected_count: int | None,
) -> None:
    if expected_count is None:
        return
    if expected_count <= 0:
        raise PerfComparisonError("--expected-count must be positive")
    expected = set(range(expected_count))
    if indices != expected:
        missing = sorted(expected - indices)
        extra = sorted(indices - expected)
        raise PerfComparisonError(
            "index coverage mismatch: "
            f"expected=0..{expected_count - 1} records={expected_count}, "
            f"actual_records={len(indices)}, "
            f"missing_count={len(missing)} missing_first={missing[:20]}, "
            f"extra_count={len(extra)} extra_first={extra[:20]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--expected-count", type=int,
        help="require the complete global index set 0..N-1",
    )
    parser.add_argument("--relative-limit", type=float, default=0.03)
    parser.add_argument("--absolute-limit-ms", type=float, default=0.003)
    parser.add_argument(
        "--gate-first-call", action="store_true",
        help=(
            "also fail on per-case pre-warmup wall-clock regressions; without "
            "this flag first-call results are validated and reported only"
        ),
    )
    parser.add_argument(
        "--aggregate-runs", action="store_true",
        help="median duplicate indices across independent input runs",
    )
    parser.add_argument(
        "--require-first-call-ms", action="store_true",
        help=(
            "require every row to contain a finite positive pre-warmup "
            "first_call_ms measurement"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--regression-index-file", type=Path,
        help="write the initially flagged global indices, one per line",
    )
    parser.add_argument(
        "--expect-baseline-gdn-path",
        choices=("legacy", "split", "packed"),
    )
    parser.add_argument(
        "--expect-candidate-gdn-path",
        choices=("legacy", "split", "packed"),
    )
    parser.add_argument(
        "--expect-baseline-gdn-state-abi",
        help="require the exact baseline gdn_state_abi string",
    )
    parser.add_argument(
        "--expect-candidate-gdn-state-abi",
        help="require the exact candidate gdn_state_abi string",
    )
    parser.add_argument(
        "--expect-baseline-gdn-gating-abi",
        help="require the exact baseline gdn_gating_abi string",
    )
    parser.add_argument(
        "--expect-candidate-gdn-gating-abi",
        help="require the exact candidate gdn_gating_abi string",
    )
    parser.add_argument(
        "--expect-baseline-kda-trusted-host-offsets", type=parse_bool,
        metavar="true|false",
    )
    parser.add_argument(
        "--expect-candidate-kda-trusted-host-offsets", type=parse_bool,
        metavar="true|false",
    )
    parser.add_argument(
        "--expect-baseline-kda-cp-plan-cache-size-env",
        help="require the exact baseline kda_cp_plan_cache_size_env string",
    )
    parser.add_argument(
        "--expect-candidate-kda-cp-plan-cache-size-env",
        help="require the exact candidate kda_cp_plan_cache_size_env string",
    )
    parser.add_argument(
        "--expect-baseline-kda-state-abi",
        help="require the exact baseline kda_state_abi string",
    )
    parser.add_argument(
        "--expect-candidate-kda-state-abi",
        help="require the exact candidate kda_state_abi string",
    )
    parser.add_argument(
        "--expect-baseline-kda-beta-abi",
        help="require the exact baseline kda_beta_abi string",
    )
    parser.add_argument(
        "--expect-candidate-kda-beta-abi",
        help="require the exact candidate kda_beta_abi string",
    )
    parser.add_argument(
        "--expect-inventory-file-count", type=int,
        help="require this case-list file count in every raw row",
    )
    parser.add_argument(
        "--expect-inventory-record-count", type=int,
        help="require this case-list record count in every raw row",
    )
    parser.add_argument(
        "--expect-list-manifest-sha256",
        help="require this exact case-list manifest SHA-256 in every raw row",
    )
    parser.add_argument(
        "--expect-baseline-gdn-pla-extension-sha256",
        help=(
            "require this exact PLA GDN extension SHA256 on every baseline "
            "GDN row; required when baseline contains GDN rows"
        ),
    )
    parser.add_argument(
        "--expect-candidate-gdn-pla-extension-sha256",
        help=(
            "require this exact PLA GDN extension SHA256 on every candidate "
            "GDN row; required when candidate contains GDN rows"
        ),
    )
    parser.add_argument(
        "--expect-baseline-kda-pla-extension-sha256",
        help=(
            "require this exact PLA KDA extension SHA256 on every baseline "
            "KDA row; required when baseline contains KDA rows"
        ),
    )
    parser.add_argument(
        "--expect-candidate-kda-pla-extension-sha256",
        help=(
            "require this exact PLA KDA extension SHA256 on every candidate "
            "KDA row; required when candidate contains KDA rows"
        ),
    )
    args = parser.parse_args(argv)

    baseline = read_rows(
        args.baseline,
        args.aggregate_runs,
        side="baseline",
        expected_gdn_path=args.expect_baseline_gdn_path,
        expected_gdn_state_abi=args.expect_baseline_gdn_state_abi,
        expected_gdn_gating_abi=args.expect_baseline_gdn_gating_abi,
        expected_kda_trusted_host_offsets=(
            args.expect_baseline_kda_trusted_host_offsets
        ),
        expected_kda_cp_plan_cache_size_env=(
            args.expect_baseline_kda_cp_plan_cache_size_env
        ),
        expected_kda_state_abi=args.expect_baseline_kda_state_abi,
        expected_kda_beta_abi=args.expect_baseline_kda_beta_abi,
        expected_inventory_file_count=args.expect_inventory_file_count,
        expected_inventory_record_count=args.expect_inventory_record_count,
        expected_list_manifest_sha256=args.expect_list_manifest_sha256,
        expected_gdn_pla_extension_sha256=(
            args.expect_baseline_gdn_pla_extension_sha256
        ),
        expected_kda_pla_extension_sha256=(
            args.expect_baseline_kda_pla_extension_sha256
        ),
        require_first_call_ms=args.require_first_call_ms,
    )
    candidate = read_rows(
        args.candidate,
        args.aggregate_runs,
        side="candidate",
        expected_gdn_path=args.expect_candidate_gdn_path,
        expected_gdn_state_abi=args.expect_candidate_gdn_state_abi,
        expected_gdn_gating_abi=args.expect_candidate_gdn_gating_abi,
        expected_kda_trusted_host_offsets=(
            args.expect_candidate_kda_trusted_host_offsets
        ),
        expected_kda_cp_plan_cache_size_env=(
            args.expect_candidate_kda_cp_plan_cache_size_env
        ),
        expected_kda_state_abi=args.expect_candidate_kda_state_abi,
        expected_kda_beta_abi=args.expect_candidate_kda_beta_abi,
        expected_inventory_file_count=args.expect_inventory_file_count,
        expected_inventory_record_count=args.expect_inventory_record_count,
        expected_list_manifest_sha256=args.expect_list_manifest_sha256,
        expected_gdn_pla_extension_sha256=(
            args.expect_candidate_gdn_pla_extension_sha256
        ),
        expected_kda_pla_extension_sha256=(
            args.expect_candidate_kda_pla_extension_sha256
        ),
        require_first_call_ms=args.require_first_call_ms,
    )
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise PerfComparisonError(
            f"record sets differ: missing={missing[:10]} extra={extra[:10]}"
        )
    _validate_expected_coverage(set(baseline), args.expected_count)

    for index in baseline:
        baseline_runs = int(baseline[index].get("aggregate_runs", 1))
        candidate_runs = int(candidate[index].get("aggregate_runs", 1))
        if baseline_runs != candidate_runs:
            raise PerfComparisonError(
                f"aggregate sample-count mismatch for index {index}: "
                f"baseline={baseline_runs} candidate={candidate_runs}"
            )

    details = []
    for index in sorted(baseline):
        before, after = baseline[index], candidate[index]
        _require_same_identity(
            before, after, context=f"baseline/candidate index {index}"
        )
        base_ms, candidate_ms = float(before["ms"]), float(after["ms"])
        ratio = candidate_ms / base_ms
        regression = (
            ratio > 1.0 + args.relative_limit
            and candidate_ms - base_ms > args.absolute_limit_ms
        )
        first_call_ratio = (
            float(after["first_call_ms"]) / float(before["first_call_ms"])
            if before.get("first_call_ms") is not None
            and after.get("first_call_ms") is not None
            else None
        )
        first_call_regression = bool(
            first_call_ratio is not None
            and first_call_ratio > 1.0 + args.relative_limit
            and float(after["first_call_ms"])
            - float(before["first_call_ms"]) > args.absolute_limit_ms
        )
        details.append({
            "index": index,
            "family": after["family"],
            "source": after["source"],
            "line": after["line"],
            "H": after["H"],
            "Hg": after["Hg"],
            "N": after["N"],
            "T": after["T"],
            "baseline_pla_extension_sha256": before["pla_extension_sha256"],
            "candidate_pla_extension_sha256": after["pla_extension_sha256"],
            "baseline_ms": before.get("ms"),
            "candidate_ms": after.get("ms"),
            "ratio": ratio,
            "baseline_first_call_ms": before.get("first_call_ms"),
            "candidate_first_call_ms": after.get("first_call_ms"),
            "first_call_ratio": first_call_ratio,
            "first_call_regression": first_call_regression,
            "regression": regression,
        })

    regressions = [row for row in details if row["regression"]]
    first_call_regressions = [
        row for row in details if row["first_call_regression"]
    ]
    print(
        f"records={len(details)} regressions={len(regressions)} "
        f"first_call_regressions={len(first_call_regressions)} "
        f"first_call_gate={'on' if args.gate_first_call else 'report-only'} "
        f"limit={args.relative_limit:.1%}+{args.absolute_limit_ms:.4f}ms"
    )
    for family in ("gdn", "kda"):
        family_rows = [row for row in details if row["family"] == family]
        if not family_rows:
            continue
        ratios = [row["ratio"] for row in family_rows if math.isfinite(row["ratio"])]
        timed_rows = [
            row for row in family_rows
            if row["baseline_ms"] is not None and row["candidate_ms"] is not None
        ]
        baseline_total = sum(float(row["baseline_ms"]) for row in timed_rows)
        candidate_total = sum(float(row["candidate_ms"]) for row in timed_rows)
        aggregate = (
            candidate_total / baseline_total
            if len(timed_rows) == len(family_rows) and baseline_total
            else float("inf")
        )
        print(
            f"{family}: n={len(family_rows)} median={statistics.median(ratios):.4f}x "
            f"p95={percentile(ratios, .95):.4f}x max={max(ratios):.4f}x "
            f"aggregate={aggregate:.6f}x "
            f"regressions={sum(row['regression'] for row in family_rows)}"
        )
        first_call_rows = [
            row for row in family_rows if row["first_call_ratio"] is not None
        ]
        if first_call_rows:
            first_ratios = [float(row["first_call_ratio"]) for row in first_call_rows]
            baseline_first_total = sum(
                float(row["baseline_first_call_ms"]) for row in first_call_rows
            )
            candidate_first_total = sum(
                float(row["candidate_first_call_ms"]) for row in first_call_rows
            )
            print(
                f"{family}-first-call: n={len(first_call_rows)} "
                f"median={statistics.median(first_ratios):.4f}x "
                f"p95={percentile(first_ratios, .95):.4f}x "
                f"max={max(first_ratios):.4f}x "
                f"aggregate={candidate_first_total / baseline_first_total:.6f}x"
            )
    for row in sorted(regressions, key=lambda item: item["ratio"], reverse=True)[:20]:
        before = row["baseline_ms"]
        after = row["candidate_ms"]
        timing = (
            f"{float(before):.6f}->{float(after):.6f}ms"
            if before is not None and after is not None
            else f"{before}->{after}ms"
        )
        print(
            f"  index={row['index']} {row['family']} H={row['H']}/{row['Hg']} "
            f"N={row['N']} T={row['T']} {timing} ({row['ratio']:.4f}x)"
        )
    if args.output:
        args.output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in details)
        )
    if args.regression_index_file:
        args.regression_index_file.write_text(
            "".join(f"{row['index']}\n" for row in regressions)
        )
    failed = bool(regressions) or (
        args.gate_first_call and bool(first_call_regressions)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
