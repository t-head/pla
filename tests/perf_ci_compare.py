#!/usr/bin/env python3
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

"""Kernel cycle comparison of PLA vs SGLang-Triton-FLA and the performance regression threshold verdict.

The data comes from the acu result CSVs collected by the two rounds of test_op_forge_perf.py:
  - Triton round (TRITON_PLA=1) -> *_triton_*_acu_result.csv
  - PLA CUDA round (default)     -> *_cuda_*_acu_result.csv
The row order of both CSVs is strictly aligned with the same caselist, so joining by "case name" is enough.

Thresholds are tiered by the batch_size of each case in the caselist:
  - batch_size <  200 -> small case, tolerates 5% regression
  - batch_size >= 200 -> large case, tolerates 2% regression
The criterion is the within-run ratio pla_cycle / triton_cycle and does not depend on a historical baseline,
so machine generation changes / compiler upgrades do not invalidate the thresholds.

The output CSV is named like perf_ci_<patch_id>_<timestamp>.csv and lands in --output-dir,
which in CI usually points to the red-zone network drive directory. The exit code is 1 if any case is judged REGRESSION.

Usage::

    python perf_ci_compare.py \
        --caselist caselist/decode/Qwen3.7-Max_decode_tp1_..._gdn_op.list \
        --pla-csv    <cuda round CSV or its directory> \
        --triton-csv <triton round CSV or its directory> \
        --run-type ci --output-dir /ppusw/share/.../pla_perf
"""
from pathlib import Path
import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime

# ============================================================
# Default thresholds: aligned with the requirements (small case 5%, large case 2%)
# ============================================================
DEFAULT_SMALL_BATCH_THRESHOLD = 200
DEFAULT_SMALL_TOL_PCT = 5.0
DEFAULT_LARGE_TOL_PCT = 2.0

# Column names in the acu result CSV (written by batch_read_cycle_from_nculog)
COL_CASE = "case name"
COL_CYCLE = "pmu cycles"

# patch/change identifiers the CI platform may provide, probed in priority order; fall back to git short SHA when all are missing
PATCH_ID_ENV_CANDIDATES = (
    "PLA_PERF_PATCH_ID",
    "AONE_CI_PATCH_ID",
    "AONE_CI_CHANGE_ID",
    "AONE_CI_CR_ID",
    "AONE_CI_COMMIT_ID",
    "CI_COMMIT_SHORT_SHA",
)

# kernel / caselist go in the first two columns: cycles of the same shape under different kernels (e.g. k_last and
# k_last_packed) are not comparable; with only the casename column, the record names two cases report to ES
# would collide exactly and the trend chart would merge two curves into one. These two columns make the CSV self-describing (entries
# can be told apart when several are combined in the summary table), and PLAPerfResultParser uses them to build the ES casename.
OUTPUT_HEADERS = [
    "kernel",
    "caselist",
    "casename",
    "batch_size",
    "pla_cycle",
    "triton_fla_cycle",
    "pla_vs_triton_ratio",
    "speedup",
    "size_class",
    "tolerance_pct",
    "verdict",
]

VERDICT_PASS = "PASS"
VERDICT_REGRESSION = "REGRESSION"
VERDICT_MISSING = "MISSING"
VERDICT_SKIP = "SKIP"

# k_last / k_last_packed only consume decode cases with seqlen_q<=1 (or prefill cases with explicit cu_seqlens);
# test_op_forge_perf.py filters out the other lines before collection. The comparison side must use the same criterion,
# otherwise seqlen_q>1 lines in decode caselists would be misjudged as MISSING.
SEQLEN_FILTERED_KERNELS = ("k_last", "k_last_packed")


def is_kernel_filtered(casename, kernel):
    """Whether the case is filtered out by kernel in the perf collection script (produces no kernel launch)."""
    if kernel == "prefill":
        # prefill is exactly the opposite: only varlen cases with explicit cu_seqlens are consumed, other lines are filtered
        return "cu_seqlens" not in casename
    if kernel not in SEQLEN_FILTERED_KERNELS:
        return False
    if "cu_seqlens" in casename:
        return False
    match = re.search(r"seqlen_q\s*:\s*(\d+)", casename)
    return match is not None and int(match.group(1)) > 1


def resolve_patch_id(explicit=None, repo_root=None):
    """Determine the patch identifier in the CSV file name.

    Priority: --patch-id > CI environment variables > pla repo git short SHA > 'nopatch'.
    """
    if explicit:
        return _sanitize(explicit)
    for key in PATCH_ID_ENV_CANDIDATES:
        val = os.environ.get(key)
        if val and val.strip():
            return _sanitize(val.strip())
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        )
        if out.returncode == 0:
            sha = out.stdout.decode().strip()
            if sha:
                return _sanitize(sha)
    except (OSError, subprocess.SubprocessError):
        pass
    print("WARNING: cannot determine the patch ID (no CI environment variable and git unavailable), using 'nopatch'")
    return "nopatch"


def _sanitize(text):
    """Strip unsafe characters from the file name so a patch ID containing '/' and the like does not break the path."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))


# Fixed template segments in caselist file names that do not help distinguish them: the 26 decode caselists still have
# pairwise distinct names without it, and the casename is already long, so strip this noise before building the ES record name.
_CASELIST_NOISE = "_kvlimit_200GB_model_len_128K"


def caselist_label(path):
    """Compress a caselist path into a compact label (strip directory, .list suffix and fixed template segments).

    E.g.: caselist/decode/Kimi-Linear-48B-A3B_decode_tp1_kvlimit_200GB_model_len_128K_kda_op.list
      -> Kimi-Linear-48B-A3B_decode_tp1_kda_op
    """
    if not path:
        return ""
    return Path(path).stem.replace(_CASELIST_NOISE, "")


def parse_batch_size(casename):
    """Take batch_size from the case description; return None if unavailable.

    OpForge lines of decode / prefill / KDA all carry a batch_size:<int> field, e.g.
    [GDN]:[D][1]gdn_attn:seqlen_q:1,...,batch_size:1,dtype:bf16.
    """
    match = re.search(r"batch_size\s*:\s*(\d+)", casename)
    if match:
        return int(match.group(1))
    return None


def classify(batch_size, small_batch_threshold, small_tol, large_tol):
    """Return (size_class, tolerance_pct) by batch scale.

    When batch parsing fails the case is treated as large (stricter threshold): better a false alarm than a missed regression.
    """
    if batch_size is None:
        return "large(batch?)", large_tol
    if batch_size < small_batch_threshold:
        return "small", small_tol
    return "large", large_tol


def _resolve_csv(path, kind):
    """--pla-csv / --triton-csv accept a file or a directory; for a directory the latest one matching kind is taken.

    Only result CSVs with kind in the file name are accepted; if none matches, return None and never settle for
    an arbitrary .csv in the directory: both rounds land in the same directory, and when one side fails (on an acu segfault
    the CSV write is skipped) that would read the other side's results as this side's, comparing pla against itself,
    with every ratio exactly 1.0000 and all PASS -- a total collection failure disguised as identical performance.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(p.glob(f"*{kind}*acu_result.csv"),
                            key=lambda f: f.stat().st_mtime)
        if candidates:
            picked = candidates[-1]
            if len(candidates) > 1:
                print(f"NOTE: {p} has {len(candidates)} {kind} CSVs, taking the latest {picked.name}")
            return picked
        others = sorted(p.glob("*acu_result.csv"))
        if others:
            print(f"ERROR: {p} has no {kind}-side result CSV, but has "
                  f"{', '.join(f.name for f in others)}; collection on this side did not succeed, "
                  f"so no cross-side fallback is done to avoid a fake ratio identically equal to 1")
    return None


def load_cycles(csv_path):
    """Read an acu result CSV and return {case name: cycle} (keeping first-appearance order)."""
    cycles = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or COL_CASE not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing the '{COL_CASE}' column, actual columns: {reader.fieldnames}")
        if COL_CYCLE not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing the '{COL_CYCLE}' column, actual columns: {reader.fieldnames}")
        for row in reader:
            case = (row.get(COL_CASE) or "").strip()
            if not case:
                continue
            try:
                cycle = float(row.get(COL_CYCLE))
            except (TypeError, ValueError):
                cycle = None
            # only the first row of a same-name case counts, so historical rows in append mode do not override this run
            if case not in cycles:
                cycles[case] = cycle
    return cycles


def read_caselist(path):
    """Read the caselist, skipping empty lines and # comment lines, preserving the original order."""
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cases.append(stripped)
    return cases


def compare(cases, pla_cycles, triton_cycles, small_batch_threshold, small_tol, large_tol,
            kernel=None, caselist=None):
    """Compare case by case and return (rows, stats).

    kernel / caselist are only written verbatim into each row as identifier columns and do not affect the verdict (kernel is also used
    for SKIP detection in is_kernel_filtered); caselist may be a path or a label, both are compressed into a compact label.
    """
    rows = []
    tag = dict(kernel=kernel or "", caselist=caselist_label(caselist))
    stats = {"pass": 0, "regression": 0, "missing": 0, "skip": 0}
    for case in cases:
        batch_size = parse_batch_size(case)
        size_class, tol = classify(batch_size, small_batch_threshold, small_tol, large_tol)
        pla = pla_cycles.get(case)
        triton = triton_cycles.get(case)

        if pla is None and triton is None and is_kernel_filtered(case, kernel):
            # the collection side does not run such cases at all; mark SKIP to distinguish them from real collection failures
            stats["skip"] += 1
            rows.append({
                **tag,
                "casename": case,
                "batch_size": batch_size if batch_size is not None else "",
                "pla_cycle": "",
                "triton_fla_cycle": "",
                "pla_vs_triton_ratio": "",
                "speedup": "",
                "size_class": size_class,
                "tolerance_pct": tol,
                "verdict": VERDICT_SKIP,
            })
            continue

        if not pla or not triton or pla <= 0 or triton <= 0:
            # without data from one side no verdict is possible; record MISSING but not as a fail, so a few cases'
            # profiling failures do not turn the whole pipeline red (real regressions show up on cases with data)
            stats["missing"] += 1
            rows.append({
                **tag,
                "casename": case,
                "batch_size": batch_size if batch_size is not None else "",
                "pla_cycle": pla if pla else "",
                "triton_fla_cycle": triton if triton else "",
                "pla_vs_triton_ratio": "",
                "speedup": "",
                "size_class": size_class,
                "tolerance_pct": tol,
                "verdict": VERDICT_MISSING,
            })
            continue

        ratio = pla / triton
        regressed = ratio > (1.0 + tol / 100.0)
        if regressed:
            stats["regression"] += 1
        else:
            stats["pass"] += 1
        rows.append({
            **tag,
            "casename": case,
            "batch_size": batch_size if batch_size is not None else "",
            "pla_cycle": f"{pla:.2f}",
            "triton_fla_cycle": f"{triton:.2f}",
            "pla_vs_triton_ratio": f"{ratio:.4f}",
            "speedup": f"{triton / pla:.4f}",
            "size_class": size_class,
            "tolerance_pct": tol,
            "verdict": VERDICT_REGRESSION if regressed else VERDICT_PASS,
        })
    return rows, stats


def write_csv(rows, output_dir, run_type, patch_id, timestamp, tag=None):
    """Write the comparison CSV; the file name contains run type / patch ID / timestamp."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = ["perf", _sanitize(run_type)]
    if tag:
        parts.append(_sanitize(tag))
    parts += [patch_id, timestamp]
    out_path = out_dir / ("_".join(parts) + ".csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="PLA vs SGLang Triton FLA cycle comparison and performance regression threshold verdict")
    parser.add_argument("--caselist", required=True,
                        help="caselist path used for this collection, for case order and batch_size")
    parser.add_argument("--pla-csv", required=True,
                        help="acu result CSV of the PLA CUDA round (or its directory, taking the latest one)")
    parser.add_argument("--triton-csv", required=True,
                        help="acu result CSV of the SGLang Triton round (or its directory, taking the latest one)")
    parser.add_argument("--kernel", default=None,
                        choices=["cuda", "k_last", "k_last_packed", "k_last_mtp", "prefill"],
                        help="kernel used for this collection; identifies cases filtered by seqlen_q / cu_seqlens on the collection side (marked SKIP)")
    parser.add_argument("--run-type", default="ci", choices=["ci", "daily", "weekly"],
                        help="run type, written into the CSV file name (default ci)")
    parser.add_argument("--output-dir", default="perf_reports",
                        help="output directory of the comparison CSV (points to the red-zone network drive in CI)")
    parser.add_argument("--patch-id", default=None,
                        help="patch identifier; when omitted, probe CI environment variables and then the pla repo git short SHA")
    parser.add_argument("--tag", default=None,
                        help="tag appended to the file name, e.g. a model-tp prefix or kernel name, to tell apart multiple runs in the same batch")
    parser.add_argument("--small-batch-threshold", type=int, default=DEFAULT_SMALL_BATCH_THRESHOLD,
                        help=f"batch boundary between small/large cases (default {DEFAULT_SMALL_BATCH_THRESHOLD}; below it counts as small)")
    parser.add_argument("--small-tol", type=float, default=DEFAULT_SMALL_TOL_PCT,
                        help=f"regression percentage tolerated for small cases (default {DEFAULT_SMALL_TOL_PCT})")
    parser.add_argument("--large-tol", type=float, default=DEFAULT_LARGE_TOL_PCT,
                        help=f"regression percentage tolerated for large cases (default {DEFAULT_LARGE_TOL_PCT})")
    parser.add_argument("--allow-regression", action="store_true",
                        help="report only, never fail (for first baseline collection or tuning)")
    args = parser.parse_args()

    pla_csv = _resolve_csv(args.pla_csv, "cuda")
    triton_csv = _resolve_csv(args.triton_csv, "triton")
    if pla_csv is None:
        print(f"ERROR: PLA-side CSV not found: {args.pla_csv}")
        return 2
    if triton_csv is None:
        print(f"ERROR: Triton-side CSV not found: {args.triton_csv}")
        return 2
    if not Path(args.caselist).is_file():
        print(f"ERROR: caselist does not exist: {args.caselist}")
        return 2

    print(f"*--- PLA vs SGLang-Triton-FLA perf compare ---*")
    print(f"  caselist    : {args.caselist}")
    print(f"  pla    csv  : {pla_csv}")
    print(f"  triton csv  : {triton_csv}")
    print(f"  threshold   : batch<{args.small_batch_threshold} -> {args.small_tol}%, "
          f"batch>={args.small_batch_threshold} -> {args.large_tol}%")

    cases = read_caselist(args.caselist)
    try:
        pla_cycles = load_cycles(pla_csv)
        triton_cycles = load_cycles(triton_csv)
    except (OSError, ValueError) as exc:
        print(f"ERROR: failed to read CSV: {exc}")
        return 2

    rows, stats = compare(cases, pla_cycles, triton_cycles,
                          args.small_batch_threshold, args.small_tol, args.large_tol,
                          kernel=args.kernel, caselist=args.caselist)
    if not rows:
        print(f"ERROR: caselist {args.caselist} yielded no cases")
        return 2

    patch_id = resolve_patch_id(args.patch_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = write_csv(rows, args.output_dir, args.run_type, patch_id, timestamp, args.tag)

    print(f"\n  cases={len(rows)} pass={stats['pass']} "
          f"regression={stats['regression']} missing={stats['missing']} skip={stats['skip']}")
    print(f"  report -> {out_path}")

    if stats["regression"]:
        print("\n  Performance Regression detected:")
        for row in rows:
            if row["verdict"] == VERDICT_REGRESSION:
                print(f"    [batch={row['batch_size']} {row['size_class']} "
                      f"tol={row['tolerance_pct']}%] ratio={row['pla_vs_triton_ratio']} "
                      f"pla={row['pla_cycle']} triton={row['triton_fla_cycle']}")
                print(f"      {row['casename']}")
    if stats["missing"]:
        print(f"\n  WARNING: {stats['missing']} cases are missing cycle data from one side and were not judged")

    if stats["regression"] and not args.allow_regression:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
