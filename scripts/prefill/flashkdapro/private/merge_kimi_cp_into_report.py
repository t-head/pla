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

"""Inject the intracard-CP A/B dataset into flashkda-e2e-vs-fla.html.

The report already carries two Kimi-K3 datasets:
    blob["kimi_rows"]        gate = is_varlen && H in {6,12}
    blob["kimi_rows_after"]  gate = is_varlen && H <= 24
Both were measured on a different host, so this script does NOT compare against
them for the CP claim.  The CP rows carry their own same-binary, same-device
control (PLA_FLASHKDA_CP=0), produced by kimi_k3_cp_rows.py, and the old dataset
is only used for a machine-consistency cross-check.

Injected keys:
    blob["cp_rows"]        production mix, 150-shape sampling per tp, every shape
    blob["cp_focus_rows"]  every tp8 shape the CP AUTO gate accepts (no sampling)
    blob["cp_summary"]     aggregates: mix on/off, engaged vs not, focus, xcheck

Usage:
  python3 merge_kimi_cp_into_report.py --mix rows/mix_tp*.jsonl \
      --focus rows/cp_tp8.jsonl --html .../flashkda-e2e-vs-fla.html \
      --meta-commit e5aa9a2 --meta-date 2026-08-19
"""
import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def load(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if line.strip().startswith("{"):
                    out.append(json.loads(line))
    return out


CENSUS_TOTAL = re.compile(r"\[tp(\d+)\] (\d+) unique shapes, H=(\d+)")
CENSUS_KEEP = re.compile(r"\[tp(\d+)\] CP-engaging shapes: (\d+) \(T range (\d+)\.\.(\d+)\)")


def census(paths):
    """Planner census straight from the sweep logs.

    The tp4/tp2/tp1 focus runs produce zero rows -- that IS the result (CP_MAX_H
    = 12), and it would be invisible if we only shipped rows.  Reading it out of
    the log keeps the "does not engage" claim measured rather than asserted.
    """
    out = {}
    for p in paths:
        try:
            with open(p, errors="replace") as fh:
                txt = fh.read()
        except OSError:
            continue
        for tp, total, H in CENSUS_TOTAL.findall(txt):
            e = out.setdefault(int(tp), {"tp": int(tp), "H": int(H),
                                         "total_shapes": int(total)})
            e["H"], e["total_shapes"] = int(H), int(total)
        for tp, keep, t0, t1 in CENSUS_KEEP.findall(txt):
            e = out.setdefault(int(tp), {"tp": int(tp)})
            e["engaged_shapes"] = int(keep)
            e["engaged_T_min"], e["engaged_T_max"] = int(t0), int(t1)
        for tp in out:
            out[tp].setdefault("engaged_shapes", 0)
    return [out[k] for k in sorted(out)]


def sp_agg(rows, field):
    """Aggregate a speedup-like field over ok rows."""
    xs = [r[field] for r in rows
          if r.get("status") == "ok" and r.get(field, 0) > 0]
    if not xs:
        return {}
    return {
        "count": len(xs), "geomean": geomean(xs), "median": statistics.median(xs),
        "min": min(xs), "max": max(xs),
        "win_count": sum(1 for x in xs if x > 1.0),
        "win_rate": sum(1 for x in xs if x > 1.0) / len(xs),
        "loss_count": sum(1 for x in xs if x < 1.0),
    }


def bucket(rows, label):
    """One row of the summary table: CP gain + vs-FLA before/after for a subset."""
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {}
    g = sp_agg(ok, "cp_gain")
    on = sp_agg(ok, "speedup")
    off = sp_agg(ok, "speedup_cpoff")
    return {
        "label": label, "count": len(ok),
        "gain_geo": g.get("geomean", 0.0), "gain_med": g.get("median", 0.0),
        "gain_min": g.get("min", 0.0), "gain_max": g.get("max", 0.0),
        "sp_off_geo": off.get("geomean", 0.0), "sp_on_geo": on.get("geomean", 0.0),
        "sp_off_min": off.get("min", 0.0), "sp_on_min": on.get("min", 0.0),
        "losses_off": off.get("loss_count", 0), "losses_on": on.get("loss_count", 0),
        "engaged_count": sum(1 for r in ok if r.get("cp_engaged")),
        "ppu_off_med_ms": statistics.median([r["ppu_cpoff_median_ms"] for r in ok]),
        "ppu_on_med_ms": statistics.median([r["ppu_median_ms"] for r in ok]),
        "fla_med_ms": statistics.median([r["fla_median_ms"] for r in ok]),
    }


def key_of(r):
    sl = r.get("seq_lens") or []
    return (r["tp"], r["H"], r["B"], r["T_total"], r["mode"], tuple(sl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", nargs="+", required=True)
    ap.add_argument("--focus", nargs="+", default=[])
    ap.add_argument("--census-logs", nargs="+", default=[],
                    help="focus-run logs ONLY (cp_tp*.log).  The mix runs pass "
                         "--all-shapes, so their 'CP-engaging shapes' line counts "
                         "kept shapes and would corrupt the census.")
    ap.add_argument("--html", required=True)
    ap.add_argument("--meta-commit", default="")
    ap.add_argument("--meta-branch", default="develop_flashkda_cp")
    ap.add_argument("--meta-date", default="")
    ap.add_argument("--meta-host", default="")
    ap.add_argument("--allow-tp", nargs="+", type=int, default=None,
                    help="Restrict both mix and focus to these tp values "
                         "(e.g. --allow-tp 4 8).  Defaults to all tps present.")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="Drop rows with B > this value (applied after tp filter).")
    args = ap.parse_args()

    def _apply_filters(rows, label):
        n0 = len(rows)
        if args.allow_tp is not None:
            allowed = set(args.allow_tp)
            rows = [r for r in rows if r.get("tp") in allowed]
        if args.max_batch is not None:
            rows = [r for r in rows if r.get("B", 0) <= args.max_batch]
        if len(rows) != n0:
            print(f"  filter {label}: {n0} -> {len(rows)} rows "
                  f"(allow_tp={args.allow_tp}, max_batch={args.max_batch})")
        return rows

    mix = _apply_filters(load(args.mix), "mix")
    focus = _apply_filters(load(args.focus), "focus") if args.focus else []
    print(f"mix rows: {len(mix)}   focus rows: {len(focus)}")

    with open(args.html, encoding="utf-8") as fh:
        html = fh.read()
    m = re.search(r'(<script id="report-data"[^>]*>)(.*?)(</script>)', html, re.S)
    blob = json.loads(m.group(2))

    ok_mix = [r for r in mix if r.get("status") == "ok"]
    engaged = [r for r in ok_mix if r.get("cp_engaged")]
    plain = [r for r in ok_mix if not r.get("cp_engaged")]

    by_tp = []
    for tp in sorted({r["tp"] for r in ok_mix}):
        b = bucket([r for r in ok_mix if r["tp"] == tp], f"tp{tp}")
        b["tp"] = tp
        b["H"] = next(r["H"] for r in ok_mix if r["tp"] == tp)
        by_tp.append(b)

    buckets = [bucket(engaged, "CP engaged"), bucket(plain, "CP not engaged")]
    buckets = [b for b in buckets if b]

    focus_ok = [r for r in focus if r.get("status") == "ok"]
    # CP cost is bracketed by the gate decay regime (pre_scan's exact zero-state
    # early exit), so the regimes must never be pooled -- the geomean of a
    # slow+fast mixture is not a bound on anything.  Slow first: it is the
    # pessimistic bound, where pre_scan pays in full.
    ORDER = {"slow": 0, "mid": 1, "fast": 2, "empty": 3}
    focus_by_decay = []
    for dc in sorted({r.get("decay", "empty") for r in focus_ok},
                     key=lambda d: ORDER.get(d, 9)):
        sub = [r for r in focus_ok if r.get("decay", "empty") == dc]
        focus_by_decay.append({
            "decay": dc,
            "overall": bucket(sub, f"tp8 engaged · decay={dc}"),
            "by_mode": [dict(bucket([r for r in sub if r["mode"] == md], md),
                             mode=md) for md in ("no_state", "with_state")
                        if any(r["mode"] == md for r in sub)],
            "biggest_gains": sorted(sub, key=lambda r: -r.get("cp_gain", 0))[:12],
        })
    bounds = {d["decay"]: d["overall"].get("gain_geo", 0.0) for d in focus_by_decay}
    focus_summary = {
        "by_decay": focus_by_decay,
        "regimes": [d["decay"] for d in focus_by_decay],
        "gain_bounds": bounds,
        "note": ("gain reported as a slow|fast pair; slow = pre_scan pays a full "
                 "pass, fast = the transition chain underflows and exits early"),
    }

    # Machine-consistency cross-check: the CP-off leg against the dataset already
    # in the report (different host, so a spread is expected -- this only proves
    # the two machines are in the same ballpark, it is NOT the CP claim).
    prev = {key_of(r): r for r in (blob.get("kimi_rows_after") or [])
            if r.get("status") == "ok"}
    ratios, paired = [], 0
    for r in ok_mix:
        p = prev.get(key_of(r))
        if not p:
            continue
        paired += 1
        if r["ppu_cpoff_median_ms"] > 0:
            ratios.append(p["ppu_median_ms"] / r["ppu_cpoff_median_ms"])
    xcheck = {
        "paired": paired, "geo_ratio": geomean(ratios),
        "median_ratio": statistics.median(ratios) if ratios else 0.0,
        "min_ratio": min(ratios) if ratios else 0.0,
        "max_ratio": max(ratios) if ratios else 0.0,
        "note": ("previous host ppu_median / this host CP-off ppu_median; "
                 ">1 means this machine is faster"),
    }

    summary = {
        "meta": {
            "source": ("Kimi-K3 prefill case lists (tp1/2/4/8); mix uses the same "
                       "150-shape representative sampling as the datasets above, "
                       "focus uses every tp8 shape the CP AUTO gate accepts"),
            "control": ("same binary, same device, PLA_FLASHKDA_CP=0 "
                        "(read per launch) -- no cross-device comparison"),
            "pla_branch": args.meta_branch, "pla_commit": args.meta_commit,
            "created_at": args.meta_date, "host": args.meta_host,
            "allow_tp": args.allow_tp, "max_batch": args.max_batch,
        },
        "mix": {
            "row_count": len(mix), "ok_count": len(ok_mix),
            "engaged_count": len(engaged), "plain_count": len(plain),
            "overall": bucket(ok_mix, "all"),
            "by_tp": by_tp, "buckets": buckets,
        },
        "focus": focus_summary,
        "census": census(args.census_logs),
        "xcheck": xcheck,
    }

    blob["cp_rows"] = mix
    blob["cp_focus_rows"] = focus
    blob["cp_summary"] = summary

    print("\n=== mix (production sampling) ===")
    o = summary["mix"]["overall"]
    print(f"  rows={o['count']} engaged={len(engaged)}  "
          f"gain geo={o['gain_geo']:.4f} [{o['gain_min']:.3f}..{o['gain_max']:.3f}]")
    print(f"  vs FLA geo: off {o['sp_off_geo']:.3f} -> on {o['sp_on_geo']:.3f}   "
          f"losses {o['losses_off']} -> {o['losses_on']}")
    for b in by_tp:
        print(f"   tp{b['tp']} H={b['H']}: n={b['count']} engaged={b['engaged_count']} "
              f"gain geo={b['gain_geo']:.4f} [{b['gain_min']:.3f}..{b['gain_max']:.3f}] "
              f"sp {b['sp_off_geo']:.3f}->{b['sp_on_geo']:.3f} "
              f"losses {b['losses_off']}->{b['losses_on']}")
    for b in buckets:
        print(f"   {b['label']}: n={b['count']} gain geo={b['gain_geo']:.4f} "
              f"[{b['gain_min']:.3f}..{b['gain_max']:.3f}] "
              f"sp {b['sp_off_geo']:.3f}->{b['sp_on_geo']:.3f}")
    if focus_ok:
        print("\n=== focus (every CP-engaging tp8 shape, per decay regime) ===")
        for d in focus_by_decay:
            f = d["overall"]
            print(f"  decay={d['decay']:<5} rows={f['count']} "
                  f"gain geo={f['gain_geo']:.4f} med={f['gain_med']:.4f} "
                  f"[{f['gain_min']:.3f}..{f['gain_max']:.3f}]  "
                  f"vs FLA {f['sp_off_geo']:.3f}->{f['sp_on_geo']:.3f}  "
                  f"losses {f['losses_off']}->{f['losses_on']}")
    if summary["census"]:
        print("\n=== planner census (from focus logs) ===")
        for c in summary["census"]:
            rng = (f" T {c.get('engaged_T_min')}..{c.get('engaged_T_max')}"
                   if c.get("engaged_shapes") else "")
            print(f"  tp{c['tp']} H={c.get('H')}: {c.get('engaged_shapes')}"
                  f" / {c.get('total_shapes')} shapes engage CP{rng}")
    print(f"\n=== xcheck vs previous host ===\n  paired={xcheck['paired']} "
          f"geo={xcheck['geo_ratio']:.3f} median={xcheck['median_ratio']:.3f} "
          f"[{xcheck['min_ratio']:.3f}..{xcheck['max_ratio']:.3f}]")

    new_json = json.dumps(blob, ensure_ascii=False, separators=(",", ":"))
    html = html[:m.start(2)] + new_json + html[m.end(2):]
    with open(args.html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\ninjected cp_rows({len(mix)}) + cp_focus_rows({len(focus)}) "
          f"+ cp_summary; HTML {len(html)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
