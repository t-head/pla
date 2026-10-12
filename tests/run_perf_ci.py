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

"""PLA perf CI driver: runs two rounds of acu collection per caselist, then applies the threshold verdict.

A full comparison needs two profiling rounds, because acu can only filter one kernel per --kernel-name:
  - round 1 TRITON_PLA=1 -> collect SGLang Triton FLA kernel cycles
  - round 2 TRITON_PLA=0 -> collect PLA CUDA kernel cycles
Each round is further split by --chunk-size before being fed to acu (application replay cannot handle too many launches, see
the note on DEFAULT_CHUNK_SIZE); chunk results are merged into merged_<side>_acu_result.csv,
then handed to perf_ci_compare.py for the report and the 5%/2% threshold verdict.
--repeat N repeats both rounds above N times and takes the per-case median cycle, to suppress the sampling jitter
of small-batch cases (see the note on _median_cycles).
Chunking only counts "lines this kernel will actually run": the caselist mixes shapes for other kernels, which the collection side
silently skips; blindly splitting by raw line count yields chunks that collect nothing at all (see split_caselist).

Exit codes: 0 all pass; 1 performance regression (can be masked with --allow-regression); 2 collection/comparison failure
(acu segfault, missing result CSV, MISSING over the ratio) -- such failures are not masked by --allow-regression,
because they mean the data was not collected, not that performance got worse.

Coverage is not described by this script: each caselist corresponds to one xtest perf case file
(test/cases/pla_test/pla_perf/), and which ones run is decided by the plan's tag selection. This script only
runs a single caselist -- case granularity rather than the whole list, so a hung entry only kills itself.

Usage::

    python tests/run_perf_ci.py --kernel k_last \
        --caselist caselist/decode/Qwen3.7-Max_decode_tp1_..._gdn_op.list \
        --run-type ci --output-dir /ppusw/share/.../pla_perf
"""
from pathlib import Path
import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from perf_ci_compare import (  # noqa: E402  (sys.path must be patched first)
    DEFAULT_LARGE_TOL_PCT,
    DEFAULT_SMALL_BATCH_THRESHOLD,
    DEFAULT_SMALL_TOL_PCT,
    caselist_label,
    compare,
    is_kernel_filtered,
    load_cycles,
    read_caselist,
    resolve_patch_id,
    write_csv,
)

# test_op_forge_perf.py builds acu commands with relative paths internally (python -u test_op_forge.py ...),
# so it must run from the tests/ directory, otherwise the child process launched by acu cannot find test_op_forge.py
PERF_SCRIPT = "test_op_forge_perf.py"

# Maximum number of cases fed to one acu call.
# --replay-mode application replays the whole application several times (determined by the number of metric groups, 4 passes measured);
# with too many launches, asight segfaults in ApplicationRelayMatcher::GetItemPointerInSharedMemory on the 2nd pass,
# after which acu reports "No kernels were profiled" and drops the whole round of data.
# Measured on 890: 63 cases (441 launches) are fine, 127 cases (889 launches) always crash. 48 leaves margin.
DEFAULT_CHUNK_SIZE = 48
RESULT_CASE_COL = "case name"


def run_one_pass(caselist, kernel, output_dir, use_triton, timeout, extra_args=None):
    """Run one round of test_op_forge_perf.py. Returns True if the child process exit code is 0."""
    env = dict(os.environ)
    env["TRITON_PLA"] = "1" if use_triton else "0"
    # the perf script's temporary record CSV defaults to a personal network drive path, which may not exist in CI containers;
    # redirect it to this run's directory so a failing mkdir does not kill the whole round
    env.setdefault("PLA_PERF_RECORD_DIR", str(Path(output_dir) / "_record"))
    cmd = [
        sys.executable, "-u", PERF_SCRIPT,
        f"--caselist={caselist}",
        f"--kernel={kernel}",
        "--daily",
        f"--output={output_dir}",
    ]
    if extra_args:
        cmd += extra_args
    side = "triton" if use_triton else "pla-cuda"
    print(f"    [{side}] {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=str(_TESTS_DIR), env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    ERROR: [{side}] collection timed out ({timeout}s)")
        return False
    if proc.returncode != 0:
        print(f"    ERROR: [{side}] collection failed, returncode={proc.returncode}")
        return False
    _archive_acu_log(caselist, output_dir, side)
    return True


def _archive_acu_log(caselist, output_dir, side):
    """Archive this round's raw acu log into the run directory.

    In test_op_forge_perf.py the log name is determined only by the caselist, so the two rounds overwrite each other;
    archive a copy with a side suffix, to check afterwards which kernel launches the cycles were taken from.
    """
    stem = Path(caselist).name.replace(".", "_").replace("-", "_")
    src = _TESTS_DIR / "logs" / f"case_{stem}.log"
    if not src.is_file():
        return
    try:
        shutil.copy2(src, Path(output_dir) / f"acu_{side}_{stem}.log")
    except OSError as exc:
        print(f"    WARNING: failed to archive acu log: {exc}")


def runnable_lines(caselist, kernel):
    """Read the lines of the caselist this kernel will actually run (dropping empty lines, # comments and shapes filtered by kernel).

    The line count is not the number of cases that will run: the collection side test_op_forge.py silently continues on
    shapes that do not belong to this kernel (e.g. k_last only runs seqlen_q=1; the seqlen_q>1 lines are k_last_mtp
    draft token shapes), neither erroring nor counting them. The criterion directly reuses perf_ci_compare's
    is_kernel_filtered, so the chunking side and the comparison side share one source -- the comparison side already marks these lines SKIP with it.
    """
    lines = []
    for raw in Path(caselist).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if is_kernel_filtered(line, kernel):
            continue
        lines.append(line)
    return lines


def split_caselist(caselist, chunk_size, out_dir, kernel):
    """Split the caselist into several sub-caselists and return the list of sub-file paths.

    chunk_size is the launch budget of a single acu call, so it must count "lines this kernel will actually run":
    blindly splitting by raw line count yields chunks made entirely of filtered lines, where acu captures no kernel at all,
    no result CSV is produced, and the whole item is then judged a collection failure -- the chunks already collected are discarded too, with zero report records.
    Measured on 890: 144 of the 200 lines of a decode caselist have seqlen_q>1; of the 5 chunks cut with chunk-size=48,
    3 had no runnable line at all, giving exit code 2 and no CSV whatsoever.
    Returns an empty list when nothing will run, which the caller treats as SKIP.
    Sub-file names must keep the original stem: the collection side parses model and tp from <model>_<stage>_tp<N> in the file name,
    and a rename that loses this segment gets skipped as an invalid caselist.
    """
    src = Path(caselist)
    lines = runnable_lines(src, kernel)
    if not lines:
        return []
    if chunk_size <= 0 or len(lines) <= chunk_size:
        return [src]
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for i in range(0, len(lines), chunk_size):
        dst = out_dir / f"{src.stem}_chunk{i // chunk_size:03d}{src.suffix}"
        dst.write_text("\n".join(lines[i:i + chunk_size]) + "\n", encoding="utf-8")
        chunks.append(dst)
    return chunks


def _merge_result_csv(csv_files, dest):
    """Merge the per-chunk acu result CSVs into one and return dest (None when there is no data row at all).

    Result file names are only accurate to the second and the collection side writes in append mode, so two chunks finishing
    in the same second end up in the same file with an extra header line in the middle, which is detected by the first column and dropped when merging.
    """
    rows, fieldnames = [], None
    for path in csv_files:
        with open(path, newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                case = (row.get(RESULT_CASE_COL) or "").strip()
                if not case or case == RESULT_CASE_COL:
                    continue
                rows.append(row)
    if not rows or fieldnames is None:
        return None
    with open(dest, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return dest


def run_pass_chunked(caselist, kernel, item_dir, use_triton, timeout, chunk_size,
                     extra_args=None):
    """Run one round of collection chunk by chunk and return (merged CSV path, successful chunks, total chunks).

    One crashed chunk does not bring down the whole round: its cases are missing from the merged result and are marked MISSING in the comparison stage.
    Chunk CSVs go into the chunks_<side>/ subdirectory and the merged product into item_dir, avoiding _resolve_csv's directory scan.
    """
    side = "triton" if use_triton else "pla-cuda"
    kind = "triton" if use_triton else "cuda"
    chunk_dir = item_dir / f"chunks_{kind}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_caselist(caselist, chunk_size, chunk_dir, kernel)
    pattern = f"*{kind}*acu_result.csv"

    def snapshot():
        # two chunks in the same second append to the same file; comparing only the file name set would miss it, so include the size
        return {(f.name, f.stat().st_size) for f in chunk_dir.glob(pattern)}

    ok_chunks = 0
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"    [{side}] chunk {i}/{len(chunks)}: {chunk.name}")
        before = snapshot()
        if not run_one_pass(str(chunk), kernel, chunk_dir, use_triton, timeout, extra_args):
            continue
        if snapshot() == before:
            print(f"    WARNING: [{side}] chunk {i} produced no result CSV (acu captured no kernel)")
            continue
        ok_chunks += 1
    if not ok_chunks:
        return None, 0, len(chunks)
    merged = _merge_result_csv(
        sorted(chunk_dir.glob(pattern), key=lambda f: f.stat().st_mtime),
        item_dir / f"merged_{kind}_acu_result.csv")
    return merged, ok_chunks, len(chunks)


def _median_cycles(runs):
    """Combine the {case: cycle} of multiple collection rounds into one per-case median.

    The measured single-round cycle jitter of small-batch cases is larger than the threshold itself: on 890, a padded
    varlen case with batch=2 collected three rounds in a row lands at ratio 0.94 / 1.03 / 1.12, a 12% spread, while the small-case
    threshold is only 5%, so single-round regression verdicts give random false alarms (the 2 regressions flagged across all 1344 cases
    both had a three-round median of 0.97 on retest). The median rather than the mean keeps individual outlier rounds from skewing the result.
    """
    if len(runs) == 1:
        return runs[0]
    merged = {}
    for run in runs:
        for case, cycle in run.items():
            merged.setdefault(case, []).append(cycle)
    return {case: statistics.median(vals) for case, vals in merged.items()}


def _tag_for(caselist, kernel):
    """Build the distinguishing tag in the CSV file name from the caselist file name and kernel."""
    return f"{caselist_label(caselist)}_{kernel}"


def main():
    parser = argparse.ArgumentParser(
        description="PLA perf CI driver: two rounds of acu collection + PLA vs Triton threshold verdict")
    parser.add_argument("--caselist", required=True,
                        help="caselist path to run (use together with --kernel)")
    parser.add_argument("--kernel", default="k_last",
                        choices=["cuda", "k_last", "k_last_packed", "k_last_mtp", "prefill"],
                        help="kernel used by this caselist (default k_last)")
    # defaults of run-type / output-dir come from environment variables instead of being spliced into script_args by the xtest case:
    # the xtest loader rejects import statements in case files (including `import os`), and if one appears the whole
    # case fails to load. So the two dynamic values "the same case reused by ci/weekly" and "report directory probed by run_perf_ci.sh"
    # must be read from the environment by this script itself, keeping the case purely declarative.
    parser.add_argument("--run-type",
                        default=os.environ.get("PLA_PERF_RUN_TYPE", "ci"),
                        choices=["ci", "daily", "weekly"],
                        help="run type, written into the CSV file name (defaults to $PLA_PERF_RUN_TYPE, then ci)")
    parser.add_argument("--output-dir",
                        default=os.environ.get("PLA_PERF_REPORT_DIR_RESOLVED", "perf_reports"),
                        help="output directory of the comparison CSV (defaults to $PLA_PERF_REPORT_DIR_RESOLVED, "
                             "which run_perf_ci.sh exports after probing that the network drive is writable)")
    parser.add_argument("--work-dir", default=None,
                        help="directory for the raw acu CSVs of both rounds (default ./logs/perf_ci_<timestamp>)")
    parser.add_argument("--patch-id", default=None, help="patch identifier; auto-detected when omitted")
    parser.add_argument("--case-timeout", type=int, default=7200,
                        help="timeout in seconds for one round of acu collection (default 7200)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"case limit per acu call, 0 disables chunking (default {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--repeat", type=int, default=1,
                        help="number of collection rounds per side, taking the per-case median cycle (default 1). "
                             "single-round jitter of small-batch cases can reach 10%%; 3 is recommended for threshold verdicts")
    parser.add_argument("--max-missing-pct", type=float, default=5.0,
                        help="allowed ratio of MISSING cases; above it the run is treated as a collection failure (default 5%%)")
    parser.add_argument("--small-batch-threshold", type=int, default=DEFAULT_SMALL_BATCH_THRESHOLD,
                        help=f"batch boundary between small/large cases (default {DEFAULT_SMALL_BATCH_THRESHOLD})")
    parser.add_argument("--small-tol", type=float, default=DEFAULT_SMALL_TOL_PCT,
                        help=f"regression percentage tolerated for small cases (default {DEFAULT_SMALL_TOL_PCT})")
    parser.add_argument("--large-tol", type=float, default=DEFAULT_LARGE_TOL_PCT,
                        help=f"regression percentage tolerated for large cases (default {DEFAULT_LARGE_TOL_PCT})")
    parser.add_argument("--allow-regression", action="store_true",
                        help="report only, never fail on performance regressions (for first baseline collection or tuning); "
                             "collection/comparison failures are not masked and still return 2")
    args = parser.parse_args()

    items = [(args.kernel, args.caselist)]

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path(args.work_dir) if args.work_dir else _REPO_ROOT / "logs" / f"perf_ci_{run_stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    patch_id = resolve_patch_id(args.patch_id, repo_root=_REPO_ROOT)

    print(f"*--- PLA perf CI ({args.run_type}) ---*")
    # The two lines below are PLAPerfResultParser's fallback channel for kernel / caselist (same-name columns in the
    # comparison CSV take precedence): when no CSV was written it still needs these two values to name fallback records distinctly,
    # and the case file is purely declarative and cannot put them into context.config. When changing the format, update
    # the two regexes _STDOUT_KERNEL / _STDOUT_CASELIST in pla_perf_collector.py accordingly.
    print(f"  kernel    : {args.kernel}")
    print(f"  caselist  : {caselist_label(args.caselist)}")
    print(f"  items     : {len(items)}")
    print(f"  patch id  : {patch_id}")
    print(f"  work dir  : {work_dir}")
    print(f"  report dir: {args.output_dir}")
    print(f"  threshold : batch<{args.small_batch_threshold} -> {args.small_tol}%, "
          f"batch>={args.small_batch_threshold} -> {args.large_tol}%")
    print(f"  chunk     : {args.chunk_size or 'off'} cases per acu call")
    print(f"  repeat    : {args.repeat} rounds per side" + (" (per-case median)" if args.repeat > 1 else ""))

    all_rows = []
    total = {"pass": 0, "regression": 0, "missing": 0, "skip": 0}
    failed_items = []
    # record collection/comparison failures separately: an acu segfault means "no data collected", not "performance got worse",
    # and must not be masked by --allow-regression, otherwise the baseline collection round would let failures through too
    broken_items = []
    skipped_items = []

    for idx, (kernel, caselist) in enumerate(items, 1):
        caselist_path = caselist if Path(caselist).is_absolute() else str(_REPO_ROOT / caselist)
        tag = _tag_for(caselist, kernel)
        print(f"\n----[ {idx}/{len(items)} ] {tag} ----")
        if not Path(caselist_path).is_file():
            print(f"    SKIP: caselist does not exist: {caselist_path}")
            skipped_items.append((tag, "caselist missing"))
            continue

        # lines filtered by kernel are silently skipped on the collection side; without printing the count, the fact that
        # "a 200-line caselist actually runs only 56 entries" is completely invisible in the log
        runnable = runnable_lines(caselist_path, kernel)
        total_lines = len([ln for ln in Path(caselist_path).read_text(encoding="utf-8").splitlines()
                           if ln.strip() and not ln.strip().startswith("#")])
        print(f"    caselist  : {len(runnable)}/{total_lines} entries run under {kernel}"
              f" (the other {total_lines - len(runnable)} do not belong to this kernel and are marked SKIP in the report)")
        if not runnable:
            print(f"    SKIP: no case runs under {kernel}, nothing to collect")
            skipped_items.append((tag, f"no runnable case for {kernel}"))
            continue

        item_dir = work_dir / tag
        item_dir.mkdir(parents=True, exist_ok=True)

        # each round runs triton first, then cuda, each collecting one merged_<side>_acu_result.csv chunk by chunk;
        # with repeat>1 each round lands in a round<N>/ subdirectory, and the per-case median cycle is compared at the end
        pla_runs, triton_runs, collect_err = [], [], None
        for rnd in range(1, args.repeat + 1):
            round_dir = item_dir / f"round{rnd}" if args.repeat > 1 else item_dir
            round_dir.mkdir(parents=True, exist_ok=True)
            if args.repeat > 1:
                print(f"    -- round {rnd}/{args.repeat} --")
            triton_csv, ok_t, n_t = run_pass_chunked(
                caselist_path, kernel, round_dir, True, args.case_timeout, args.chunk_size)
            time.sleep(1)
            pla_csv, ok_p, n_p = run_pass_chunked(
                caselist_path, kernel, round_dir, False, args.case_timeout, args.chunk_size)
            if triton_csv is None or pla_csv is None:
                collect_err = (f"collection produced no result CSV"
                               f" (triton {ok_t}/{n_t} chunks, pla {ok_p}/{n_p} chunks)")
                break
            if ok_t != n_t or ok_p != n_p:
                # when acu segfaults in application replay it does not return a nonzero exit code, it just skips writing the result
                # CSV; this must be treated as a failure, otherwise this chunk's cases silently become MISSING,
                # the report is still all green, and a collection failure passes as "no regression".
                collect_err = (f"some chunks failed to collect (triton {ok_t}/{n_t} chunks, pla {ok_p}/{n_p} chunks), "
                               f"the cases in those chunks have no valid cycle data")
                break
            try:
                triton_runs.append(load_cycles(triton_csv))
                pla_runs.append(load_cycles(pla_csv))
            except (OSError, ValueError) as exc:
                collect_err = f"failed to read result CSV: {exc}"
                break
        if collect_err:
            print(f"    ERROR: {collect_err}")
            broken_items.append((tag, "collect failed"))
            continue

        cases = read_caselist(caselist_path)
        try:
            rows, stats = compare(
                cases, _median_cycles(pla_runs), _median_cycles(triton_runs),
                args.small_batch_threshold, args.small_tol, args.large_tol,
                kernel=kernel, caselist=caselist)
        except ValueError as exc:
            print(f"    ERROR: comparison failed: {exc}")
            broken_items.append((tag, "compare failed"))
            continue

        print(f"    cases={len(rows)} pass={stats['pass']} "
              f"regression={stats['regression']} missing={stats['missing']} skip={stats['skip']}")
        for row in rows:
            if row["verdict"] == "REGRESSION":
                print(f"      REGRESSION [batch={row['batch_size']} tol={row['tolerance_pct']}%] "
                      f"ratio={row['pla_vs_triton_ratio']} {row['casename']}")
        for key in total:
            total[key] += stats[key]
        # one CSV per item, for locating the specific model/tp; also merged into the summary table
        write_csv(rows, args.output_dir, args.run_type, patch_id, run_stamp, tag)
        all_rows += rows
        if stats["regression"]:
            failed_items.append((tag, f"{stats['regression']} regression"))
        # all chunks produced CSVs but cases fail to line up on a large scale -- also a collection problem (acu crashed only on
        # some cases, or the kernel whitelist missed a variant); keep a ratio threshold as a safety net. A few cases having
        # no launch because their own shape is unsupported is normal, so missing>0 alone is not a failure.
        missing_pct = 100.0 * stats["missing"] / len(rows) if rows else 0.0
        if missing_pct > args.max_missing_pct:
            print(f"    ERROR: MISSING ratio {missing_pct:.1f}% exceeds "
                  f"{args.max_missing_pct}%, cycle data from both sides is largely missing")
            broken_items.append((tag, f"{stats['missing']}/{len(rows)} missing"))

    if all_rows:
        summary_path = write_csv(all_rows, args.output_dir, args.run_type,
                                 patch_id, run_stamp, "summary")
        print(f"\n  summary report -> {summary_path}")

    print(f"\n*--- perf CI summary ({args.run_type}) ---*")
    print(f"  cases  : total={len(all_rows)} pass={total['pass']} "
          f"regression={total['regression']} missing={total['missing']} skip={total['skip']}")
    if skipped_items:
        print(f"  skipped: {len(skipped_items)}")
        for tag, reason in skipped_items:
            print(f"    - {tag}: {reason}")
    if broken_items:
        print(f"  broken : {len(broken_items)} (collection/comparison failures, not masked by --allow-regression)")
        for tag, reason in broken_items:
            print(f"    - {tag}: {reason}")
    if failed_items:
        print(f"  failed : {len(failed_items)}")
        for tag, reason in failed_items:
            print(f"    - {tag}: {reason}")

    if broken_items:
        print("  ERROR: collection/comparison failures present, the results of this run are not trustworthy")
        return 2
    if not all_rows:
        print("  ERROR: no case produced valid comparison data")
        return 2
    if failed_items and not args.allow_regression:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
