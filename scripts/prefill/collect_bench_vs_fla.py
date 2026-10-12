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

"""Collect PLA-vs-Triton prefill numbers for the report shape sets.

The two ``bench_vs_fla.py`` scripts already define the measurement recipe
(inputs, reference call, CUDA-event timing). Their CLIs only express B=1 sweeps
plus one fixed ragged batch, so this driver imports their helpers directly.

Two shape sources are supported:

* The curated sets used by ``reports/perf_report_prefill.html``: the GDN
  13-shape standard set, and KDA TP 1/2/4/8 local head counts over five
  uniform lengths plus one ragged batch.
* Real OperatorsForge caselists under ``caselist/prefill`` (``--caselist``).
  Shapes are parsed with the repo's own ``tests/utils.py`` parser so they match
  what the caselist test runners execute. A full list is 1296 records, so
  ``--select stratified`` (default) keeps the heaviest record per
  (batch, total-token) bucket instead of replaying everything.

Usage:
    CUDA_VISIBLE_DEVICES=0 \
      python3 scripts/prefill/collect_bench_vs_fla.py \
        --family all --output reports/prefill_bench_20260911.json

    CUDA_VISIBLE_DEVICES=0 \
      python3 scripts/prefill/collect_bench_vs_fla.py \
        --caselist caselist/prefill --caselist-filter GLM-5.3-Flash \
        --output reports/prefill_caselist.json
"""

import argparse
import importlib.util
import json
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAD_DIM = 128

# (num_v_heads, num_k_heads, batch, seq_len_per_sequence)
GDN_CASES = [
    (16, 8, 1, 2048),
    (16, 8, 1, 4096),
    (16, 8, 1, 8192),
    (16, 8, 4, 2048),
    (16, 8, 4, 4096),
    (16, 8, 8, 2048),
    (32, 16, 1, 2048),
    (32, 16, 1, 4096),
    (32, 16, 1, 8192),
    (32, 16, 1, 16384),
    (32, 16, 4, 2048),
    (32, 16, 4, 4096),
    (32, 16, 8, 2048),
]

# tp -> local head count for the KDA production config (96 heads total).
KDA_TP_HEADS = [(1, 96), (2, 48), (4, 24), (8, 12)]
KDA_UNIFORM_LENS = [512, 1024, 2048, 4096, 8192]
KDA_RAGGED = [4096, 137, 2048, 63]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_bench_module(family):
    path = Path(__file__).resolve().parent / family / "bench_vs_fla.py"
    return _load_module(path, f"_bench_{family}")


_CASELIST_UTILS = None


def _caselist_utils():
    """Reuse the repo's caselist parser so shapes match the test runners."""
    global _CASELIST_UTILS
    if _CASELIST_UTILS is None:
        _CASELIST_UTILS = _load_module(
            REPO_ROOT / "tests" / "utils.py", "_pla_caselist_utils"
        )
    return _CASELIST_UTILS


_TP_RE = re.compile(r"_tp(\d+)[_.]")


def _model_and_tp(path):
    """Split ``<model>_prefill_tp<N>_...list`` into its model name and TP degree."""
    name = Path(path).name
    model = name.split("_prefill_")[0]
    match = _TP_RE.search(name)
    return model, int(match.group(1)) if match else None


def read_caselist(path):
    """Parse one caselist into measurable shape dicts, skipping what we cannot run."""
    utils = _caselist_utils()
    model, tp = _model_and_tp(path)
    cases, skipped = [], []
    for case in utils.read_cmds_from_file(str(path)):
        raw = case.get("_raw_line", "")
        try:
            offsets = utils.resolve_cu_seqlens(case)
        except ValueError as exc:
            skipped.append(f"bad cu_seqlens: {exc}")
            continue
        if offsets is None:
            skipped.append("no cu_seqlens (not a varlen prefill record)")
            continue
        dk = int(case.get("head_dim_qk", HEAD_DIM))
        dv = int(case.get("head_dim_v", HEAD_DIM))
        if dk != HEAD_DIM or dv != HEAD_DIM:
            skipped.append(f"head_dim ({dk},{dv}) != {HEAD_DIM}")
            continue
        heads_v = case.get("num_heads_v")
        heads_qk = case.get("num_heads_qk")
        if heads_v is None or heads_qk is None:
            skipped.append("missing num_heads_v/num_heads_qk")
            continue
        seq_lens = [
            int(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1)
        ]
        cases.append(
            {
                "family": "kda" if utils.detect_is_kda_in_text(raw) else "gdn",
                "source": Path(path).name,
                "model": model,
                "tp": tp,
                "num_heads": int(heads_qk),
                "num_v_heads": int(heads_v),
                "num_k_heads": int(heads_qk),
                "batch": len(seq_lens),
                "seq_lens": seq_lens,
                "ragged": len(set(seq_lens)) > 1,
                "total_tokens": sum(seq_lens),
            }
        )
    return cases, skipped


# Bucket edges for stratified selection. Batch buckets separate the single
# sequence path from small/medium/large ragged batches; token buckets span the
# 24..32768 range the lists cover.
_BATCH_EDGES = (1, 8, 64)
_TOKEN_EDGES = (1024, 4096, 16384)


def _bucket(value, edges):
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def select_stratified(cases, per_bucket):
    """Keep the heaviest records per (batch, total-token) bucket.

    Selection is a pure function of the parsed shapes, so the same list always
    yields the same subset regardless of file or discovery order.
    """
    groups = {}
    for case in cases:
        key = (
            _bucket(case["batch"], _BATCH_EDGES),
            _bucket(case["total_tokens"], _TOKEN_EDGES),
        )
        groups.setdefault(key, []).append(case)
    picked = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda c: (-c["total_tokens"], -c["batch"], c["seq_lens"]),
        )
        picked.extend(ranked[:per_bucket])
    return picked


def _git_commit(override=None):
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(args):
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "device_name": props.name,
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "git_commit": _git_commit(args.commit),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "head_dim": HEAD_DIM,
        "gdn_gate": f"relative L2 <= {args.tol}",
        "kda_gate": f"assert_close(atol={args.atol}, rtol={args.rtol})",
    }


def _assert_close_passes(out_pla, out_ref, state_pla, state_ref, atol, rtol):
    """The gate flashkdapro/bench_vs_fla.py applies: element-wise assert_close."""
    try:
        torch.testing.assert_close(out_pla, out_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(state_pla, state_ref, atol=atol, rtol=rtol)
    except AssertionError:
        return False
    return True


def _measure(module, inputs, pool_init, used_index, args, *, gate):
    """Run both paths once for accuracy, then time each with CUDA events.

    ``gate`` selects the tolerance check each upstream bench script uses:
    flashqla gates on the relative L2 error, flashkdapro on ``assert_close``.
    """
    out_ref, pool_ref = module.run_once(inputs, pool_init, use_pla=False)
    out_pla, pool_pla = module.run_once(inputs, pool_init, use_pla=True)

    used = inputs[used_index].to(torch.long)
    o_err = module.rel_err(out_pla, out_ref)
    state_err = module.rel_err(pool_pla[used], pool_ref[used])
    if gate == "l2":
        within_tol = max(o_err, state_err) <= args.tol
    else:
        within_tol = _assert_close_passes(
            out_pla, out_ref, pool_pla[used], pool_ref[used], args.atol, args.rtol
        )

    triton_ms = module.timed_ms(
        lambda: module.run_once(inputs, pool_init, use_pla=False),
        args.warmup,
        args.repeat,
    )
    pla_ms = module.timed_ms(
        lambda: module.run_once(inputs, pool_init, use_pla=True),
        args.warmup,
        args.repeat,
    )
    return o_err, state_err, triton_ms, pla_ms, within_tol


def _measure_recheck(module, inputs, pool_init, used_index, args, *, gate, blocks):
    """Re-time one case with alternating backend order per block.

    The plain ``_measure`` runs the whole Triton block before the whole PLA
    block, so clock drift or cache state can systematically favour whichever
    side runs second. Alternating which side goes first in each block cancels
    that bias, and the block-to-block spread exposes an unstable measurement.
    Accuracy is re-verified with each family's own gate, exactly as in
    ``_measure``.
    """
    out_ref, pool_ref = module.run_once(inputs, pool_init, use_pla=False)
    out_pla, pool_pla = module.run_once(inputs, pool_init, use_pla=True)

    used = inputs[used_index].to(torch.long)
    o_err = module.rel_err(out_pla, out_ref)
    state_err = module.rel_err(pool_pla[used], pool_ref[used])
    if gate == "l2":
        within_tol = max(o_err, state_err) <= args.tol
    else:
        within_tol = _assert_close_passes(
            out_pla, out_ref, pool_pla[used], pool_ref[used], args.atol, args.rtol
        )

    # Warm both sides up front so no block carries cold-start cost.
    module.timed_ms(
        lambda: module.run_once(inputs, pool_init, use_pla=False), args.warmup, 1
    )
    module.timed_ms(
        lambda: module.run_once(inputs, pool_init, use_pla=True), args.warmup, 1
    )
    triton_blocks, pla_blocks = [], []
    for b in range(blocks):
        order = (False, True) if b % 2 == 0 else (True, False)
        for use_pla in order:
            ms = module.timed_ms(
                lambda: module.run_once(inputs, pool_init, use_pla=use_pla),
                0,
                args.repeat,
            )
            (pla_blocks if use_pla else triton_blocks).append(ms)
    return o_err, state_err, triton_blocks, pla_blocks, within_tol


def collect_recheck(args, dtype):
    """Re-measure prior results: every case PLA lost, plus a win control sample.

    Input is one or more JSON files written by an earlier run of this script
    (a directory means every ``*.json`` inside). Cases at or below
    ``--recheck-threshold`` are all re-measured; ``--recheck-wins`` adds a
    deterministic random sample of winning cases so we can tell a stable
    measurement process from one where every ratio moves.
    """
    paths = []
    for raw in args.recheck:
        p = Path(raw)
        paths.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    if not paths:
        raise SystemExit(f"no JSON found under {args.recheck}")
    prior = []
    for p in paths:
        prior.extend(json.loads(p.read_text())["records"])

    slow = [
        r for r in prior
        if "speedup" in r and r["speedup"] <= args.recheck_threshold
    ]
    picks = [dict(r, recheck_group="slow") for r in slow]
    if args.recheck_wins:
        wins = [r for r in prior if r.get("speedup", 0) > args.recheck_threshold]
        control = random.Random(0).sample(wins, min(args.recheck_wins, len(wins)))
        picks += [dict(r, recheck_group="control") for r in control]
    print(
        f"\n# recheck: {len(slow)} slow cases + "
        f"{len(picks) - len(slow)} control wins from {len(paths)} file(s)",
        flush=True,
    )

    modules, records = {}, []
    for case in picks:
        module = modules.get(case["family"])
        if module is None:
            module = _load_bench_module(_BENCH_DIR[case["family"]])
            modules[case["family"]] = module
        record = dict(case)
        label = (
            f"{case['model']} tp{case['tp']} H={case['num_heads']} "
            f"B={case['batch']} T={case['total_tokens']}"
        )
        inputs = None
        try:
            torch.manual_seed(0)
            inputs, used_index, gate = _build_inputs(module, case, dtype)
            o_err, state_err, t_blocks, p_blocks, within_tol = _measure_recheck(
                module, inputs, inputs[-1], used_index, args,
                gate=gate, blocks=args.recheck_blocks,
            )
            triton_ms = statistics.median(t_blocks)
            pla_ms = statistics.median(p_blocks)
            record.update(
                {
                    "triton_ms": triton_ms,
                    "pla_ms": pla_ms,
                    "prior_speedup": case["speedup"],
                    "speedup": triton_ms / pla_ms,
                    "output_rel_err": o_err,
                    "state_rel_err": state_err,
                    "within_tol": within_tol,
                    "block_spread_triton": max(t_blocks) / min(t_blocks),
                    "block_spread_pla": max(p_blocks) / min(p_blocks),
                }
            )
            verdict = (
                "confirmed" if (case["speedup"] <= 1.0) == (triton_ms / pla_ms <= 1.0)
                else "FLIPPED"
            )
            _print_row(record, label)
            print(
                f"{'':>44} prior {case['speedup']:.3f}x -> {record['speedup']:.3f}x "
                f"[{verdict}] spread T {record['block_spread_triton']:.3f} "
                f"P {record['block_spread_pla']:.3f}",
                flush=True,
            )
        except Exception as exc:  # one shape must not kill the sweep
            record.update(
                {"within_tol": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"{label:>44} ERROR {record['error']}", flush=True)
        finally:
            del inputs
            torch.cuda.empty_cache()
        records.append(record)

    slow_out = [r for r in records if r["recheck_group"] == "slow" and "speedup" in r]
    confirmed = sum(1 for r in slow_out if r["speedup"] <= 1.0)
    ctrl_out = [r for r in records if r["recheck_group"] == "control" and "speedup" in r]
    held = sum(1 for r in ctrl_out if r["speedup"] > 1.0)
    print(
        f"\n# recheck summary: slow confirmed {confirmed}/{len(slow_out)}, "
        f"flipped {len(slow_out) - confirmed}/{len(slow_out)}; "
        f"control wins held {held}/{len(ctrl_out)}",
        flush=True,
    )
    return records


def collect_gdn(args, dtype):
    module = _load_bench_module("flashqla")
    records = []
    for num_v_heads, num_k_heads, batch, seq_len in GDN_CASES:
        seq_lens = [seq_len] * batch
        torch.manual_seed(0)
        inputs = module.make_inputs(
            seq_lens,
            num_k_heads,
            num_v_heads,
            dtype,
            "cuda",
            num_slots=len(seq_lens) + 1,
        )
        o_err, state_err, triton_ms, pla_ms, within_tol = _measure(
            module, inputs, inputs[-1], 6, args, gate="l2"
        )
        record = {
            "family": "gdn",
            "batch": batch,
            "num_v_heads": num_v_heads,
            "num_k_heads": num_k_heads,
            "seq_len": seq_len,
            "total_tokens": sum(seq_lens),
            "triton_ms": triton_ms,
            "pla_ms": pla_ms,
            "speedup": triton_ms / pla_ms,
            "output_rel_err": o_err,
            "state_rel_err": state_err,
            "within_tol": within_tol,
        }
        records.append(record)
        _print_row(record, f"B={batch} Hv={num_v_heads} T={seq_len}")
        del inputs
        torch.cuda.empty_cache()
    return records


def collect_kda(args, dtype):
    module = _load_bench_module("flashkdapro")
    records = []
    for tp, num_heads in KDA_TP_HEADS:
        cases = [[length] for length in KDA_UNIFORM_LENS] + [list(KDA_RAGGED)]
        for seq_lens in cases:
            torch.manual_seed(0)
            inputs = module.make_inputs(
                seq_lens,
                num_heads,
                dtype,
                "cuda",
                num_slots=len(seq_lens) + 2,
            )
            o_err, state_err, triton_ms, pla_ms, within_tol = _measure(
                module, inputs, inputs[-1], 9, args, gate="assert_close"
            )
            record = {
                "family": "kda",
                "tp": tp,
                "num_heads": num_heads,
                "batch": len(seq_lens),
                "seq_lens": seq_lens,
                "ragged": len(set(seq_lens)) > 1,
                "total_tokens": sum(seq_lens),
                "triton_ms": triton_ms,
                "pla_ms": pla_ms,
                "speedup": triton_ms / pla_ms,
                "output_rel_err": o_err,
                "state_rel_err": state_err,
                "within_tol": within_tol,
            }
            records.append(record)
            label = ",".join(str(length) for length in seq_lens)
            _print_row(record, f"tp{tp} H={num_heads} [{label}]")
            del inputs
            torch.cuda.empty_cache()
    return records


_BENCH_DIR = {"gdn": "flashqla", "kda": "flashkdapro"}


def _build_inputs(module, case, dtype):
    """Build inputs with each bench script's own signature.

    Returns the inputs plus the index of the slot-indices tensor and the
    accuracy gate, both of which differ between the two families.
    """
    seq_lens = case["seq_lens"]
    if case["family"] == "kda":
        inputs = module.make_inputs(
            seq_lens,
            case["num_heads"],
            dtype,
            "cuda",
            num_slots=len(seq_lens) + 2,
        )
        return inputs, 9, "assert_close"
    inputs = module.make_inputs(
        seq_lens,
        case["num_k_heads"],
        case["num_v_heads"],
        dtype,
        "cuda",
        num_slots=len(seq_lens) + 1,
    )
    return inputs, 6, "l2"


def _discover_caselists(root, patterns):
    root = Path(root)
    files = sorted(root.rglob("*.list")) if root.is_dir() else [root]
    if patterns:
        wanted = tuple(p.strip() for p in patterns.split(",") if p.strip())
        files = [f for f in files if any(w in f.name for w in wanted)]
    return files


def collect_caselist(args, dtype):
    """Replay selected records from real OperatorsForge prefill caselists."""
    paths = _discover_caselists(args.caselist, args.caselist_filter)
    if not paths:
        raise SystemExit(f"no caselist matched under {args.caselist}")

    modules, records = {}, []
    for path in paths:
        cases, skipped = read_caselist(path)
        if args.select == "all":
            selected = cases
        else:
            selected = select_stratified(cases, args.per_bucket)
        if args.limit:
            selected = selected[: args.limit]
        print(
            f"\n# {path.name}: {len(cases)} parsed, {len(skipped)} skipped, "
            f"{len(selected)} selected ({args.select})",
            flush=True,
        )
        for case in selected:
            module = modules.get(case["family"])
            if module is None:
                module = _load_bench_module(_BENCH_DIR[case["family"]])
                modules[case["family"]] = module
            record = dict(case)
            label = (
                f"{case['model']} tp{case['tp']} H={case['num_heads']} "
                f"B={case['batch']} T={case['total_tokens']}"
            )
            inputs = None
            try:
                torch.manual_seed(0)
                inputs, used_index, gate = _build_inputs(module, case, dtype)
                o_err, state_err, triton_ms, pla_ms, within_tol = _measure(
                    module, inputs, inputs[-1], used_index, args, gate=gate
                )
                record.update(
                    {
                        "triton_ms": triton_ms,
                        "pla_ms": pla_ms,
                        "speedup": triton_ms / pla_ms,
                        "output_rel_err": o_err,
                        "state_rel_err": state_err,
                        "within_tol": within_tol,
                    }
                )
                _print_row(record, label)
            except Exception as exc:  # one shape must not kill the sweep
                record.update(
                    {"within_tol": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"{label:>44} ERROR {record['error']}", flush=True)
            finally:
                del inputs
                torch.cuda.empty_cache()
            records.append(record)
    return records


def _print_row(record, label):
    print(
        f"{label:>44} {record['output_rel_err']:>10.3e} "
        f"{record['state_rel_err']:>10.3e} {record['triton_ms']:>10.3f} "
        f"{record['pla_ms']:>10.3f} {record['speedup']:>7.3f}x"
        f"{'' if record['within_tol'] else '  TOL-FAIL'}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["gdn", "kda", "all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--tol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Record this commit id; use when the tree is not a git checkout.",
    )
    parser.add_argument(
        "--caselist",
        type=Path,
        default=None,
        help="Replay a caselist file or directory instead of the curated sets.",
    )
    parser.add_argument(
        "--caselist-filter",
        type=str,
        default=None,
        help="Comma-separated substrings; keep only caselists whose name matches.",
    )
    parser.add_argument(
        "--select",
        choices=["stratified", "all"],
        default="stratified",
        help="Replay every record, or the heaviest per (batch, token) bucket.",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=1,
        help="Records kept per bucket when --select stratified (default: 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the records taken from each caselist (smoke tests).",
    )
    parser.add_argument(
        "--recheck",
        nargs="+",
        default=None,
        metavar="JSON",
        help="Re-measure prior results instead of collecting new shapes: every "
        "case at or below --recheck-threshold, plus --recheck-wins controls.",
    )
    parser.add_argument(
        "--recheck-threshold",
        type=float,
        default=1.0,
        help="Speedup at or below which a prior record is re-measured.",
    )
    parser.add_argument(
        "--recheck-wins",
        type=int,
        default=0,
        help="Also re-measure this many randomly sampled winning cases.",
    )
    parser.add_argument(
        "--recheck-blocks",
        type=int,
        default=8,
        help="Alternating timing blocks per case in --recheck mode.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    header = (
        f"{'case':>44} {'o_err':>10} {'state_err':>10} "
        f"{'triton_ms':>10} {'pla_ms':>10} {'speedup':>8}"
    )
    meta = provenance(args)
    print(json.dumps(meta, indent=2))
    print(header)
    print("-" * len(header))

    records = []
    if args.recheck is not None:
        records += collect_recheck(args, dtype)
    elif args.caselist is not None:
        records += collect_caselist(args, dtype)
    else:
        if args.family in ("gdn", "all"):
            records += collect_gdn(args, dtype)
        if args.family in ("kda", "all"):
            records += collect_kda(args, dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"provenance": meta, "records": records}, indent=2) + "\n"
    )
    print(f"\nwrote {len(records)} records to {args.output}")

    errors = [r for r in records if r.get("error")]
    failures = [r for r in records if not r["within_tol"] and not r.get("error")]
    if errors:
        print(f"ERRORED: {len(errors)} case(s) could not run")
        for record in errors:
            print(f"  {record.get('source', '')} B={record['batch']} "
                  f"T={record['total_tokens']}: {record['error']}")
    if failures:
        print(f"FAILED: {len(failures)} case(s) outside tolerance")
    if errors or failures:
        raise SystemExit(1)
    print("All cases within tolerance.")


if __name__ == "__main__":
    main()
