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

"""Inject the "after gate widening" Kimi-K3 dataset + before/after comparison
into flashkda-e2e-vs-fla.html.

The report already carries (from merge_kimi_rows_into_report.py):
    blob["kimi_rows"]     -- 1200 points, gate = is_varlen && H in {6,12}
    blob["kimi_summary"]
This script adds:
    blob["kimi_rows_after"]     -- same shapes, gate = is_varlen && H <= 24
    blob["kimi_summary_after"]
    blob["kimi_delta"]          -- per-(case,mode) before/after pairing + aggregates

Rows are paired on (tp, H, N, T_total, mode, seq_lens) so the comparison is
apples-to-apples (the sweep sampling is deterministic).

Usage:
  python3 merge_kimi_after_into_report.py --rows rows_after_tp*.jsonl \
      --html .../flashkda-e2e-vs-fla.html --meta-commit 6c23a25
"""
import argparse, json, math, re, statistics, sys
from collections import defaultdict


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def agg(rows):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("speedup", 0) > 0]
    if not ok:
        return {}
    sp = [r["speedup"] for r in ok]
    return {
        "count": len(ok), "geomean_speedup": geomean(sp),
        "median_speedup": statistics.median(sp),
        "win_count": sum(1 for s in sp if s > 1.0),
        "win_rate": sum(1 for s in sp if s > 1.0) / len(sp),
        "min_speedup": min(sp), "max_speedup": max(sp),
        "ppu_median_of_medians_ms": statistics.median([r["ppu_median_ms"] for r in ok]),
        "fla_median_of_medians_ms": statistics.median([r["fla_median_ms"] for r in ok]),
    }


def grouped(rows, field):
    g = defaultdict(list)
    for r in rows:
        g[r.get(field)].append(r)
    out = []
    for k in sorted(g.keys(), key=lambda x: (x is None, x)):
        a = agg(g[k])
        if a:
            a[field] = k
            out.append(a)
    return out


def key_of(r):
    return (r["tp"], r["H"], r["B"], r["T_total"], r["mode"], tuple(r.get("seq_lens") or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--meta-commit", default="")
    ap.add_argument("--meta-cr", default="29320733")
    ap.add_argument("--meta-date", default="")
    args = ap.parse_args()

    after = []
    for p in args.rows:
        with open(p) as fh:
            for line in fh:
                if line.strip().startswith("{"):
                    after.append(json.loads(line))
    print(f"after rows: {len(after)}")

    with open(args.html, encoding="utf-8") as fh:
        html = fh.read()
    m = re.search(r'(<script id="report-data"[^>]*>)(.*?)(</script>)', html, re.S)
    blob = json.loads(m.group(2))
    before = blob.get("kimi_rows") or []
    print(f"before rows in report: {len(before)}")

    bidx = {key_of(r): r for r in before if r.get("status") == "ok"}
    pairs, unmatched = [], 0
    for r in after:
        if r.get("status") != "ok":
            continue
        b = bidx.get(key_of(r))
        if not b:
            unmatched += 1
            continue
        pairs.append({
            "tp": r["tp"], "H": r["H"], "N": r["B"], "T_total": r["T_total"],
            "mode": r["mode"], "seq_lens": r.get("seq_lens"),
            "before_ppu_ms": b["ppu_median_ms"], "after_ppu_ms": r["ppu_median_ms"],
            "before_sp": b["speedup"], "after_sp": r["speedup"],
            "fla_ms": r["fla_median_ms"],
            "gain": b["ppu_median_ms"] / r["ppu_median_ms"] if r["ppu_median_ms"] else 0.0,
        })
    print(f"paired: {len(pairs)}  unmatched: {unmatched}")

    summary_after = {
        "meta": {
            "source": "Kimi-K3 prefill case lists (tp1/2/4/8), same deterministic sampling as the before run",
            "gate": "is_varlen && H <= 24  (widened from H in {6,12})",
            "pla_commit": args.meta_commit, "code_review": args.meta_cr,
            "created_at": args.meta_date,
        },
        "overall": agg(after), "by_suite": grouped(after, "suite"),
        "by_mode": grouped(after, "mode"), "by_tp": grouped(after, "tp"),
        "row_count": len(after), "ok_count": sum(1 for r in after if r.get("status") == "ok"),
        "case_count": len({r["case_id"] for r in after}),
    }

    # delta aggregates
    def dagg(ps):
        if not ps:
            return {}
        g = [p["gain"] for p in ps]
        fixed = sum(1 for p in ps if p["before_sp"] < 1.0 <= p["after_sp"])
        broke = sum(1 for p in ps if p["after_sp"] < 1.0 <= p["before_sp"])
        return {
            "count": len(ps), "geomean_gain": geomean(g),
            "median_gain": statistics.median(g), "min_gain": min(g), "max_gain": max(g),
            "before_geo_sp": geomean([p["before_sp"] for p in ps]),
            "after_geo_sp": geomean([p["after_sp"] for p in ps]),
            "before_losses": sum(1 for p in ps if p["before_sp"] < 1.0),
            "after_losses": sum(1 for p in ps if p["after_sp"] < 1.0),
            "fixed": fixed, "newly_broken": broke,
        }

    by_tp_delta = []
    for tp in sorted({p["tp"] for p in pairs}):
        d = dagg([p for p in pairs if p["tp"] == tp])
        d["tp"] = tp
        by_tp_delta.append(d)

    delta = {
        "overall": dagg(pairs), "by_tp": by_tp_delta,
        "pairs": pairs,
        "biggest_gains": sorted(pairs, key=lambda p: -p["gain"])[:12],
        "regressions": sorted([p for p in pairs if p["gain"] < 0.98], key=lambda p: p["gain"])[:12],
        "still_losing": sorted([p for p in pairs if p["after_sp"] < 1.0], key=lambda p: p["after_sp"]),
    }

    blob["kimi_rows_after"] = after
    blob["kimi_summary_after"] = summary_after
    blob["kimi_delta"] = delta

    print("\n=== BEFORE vs AFTER (overall) ===")
    o = delta["overall"]
    print(f"  paired={o['count']}  geo gain={o['geomean_gain']:.4f}x  median gain={o['median_gain']:.4f}x")
    print(f"  geo speedup vs FLA: {o['before_geo_sp']:.3f}x -> {o['after_geo_sp']:.3f}x")
    print(f"  losses: {o['before_losses']} -> {o['after_losses']}  (fixed {o['fixed']}, newly broken {o['newly_broken']})")
    print("--- by tp ---")
    for d in by_tp_delta:
        print(f"  tp{d['tp']}: gain geo={d['geomean_gain']:.4f}x med={d['median_gain']:.4f}x "
              f"[{d['min_gain']:.3f}..{d['max_gain']:.3f}] sp {d['before_geo_sp']:.3f}->{d['after_geo_sp']:.3f} "
              f"losses {d['before_losses']}->{d['after_losses']} fixed={d['fixed']} broken={d['newly_broken']}")

    new_json = json.dumps(blob, ensure_ascii=False, separators=(",", ":"))
    html = html[:m.start(2)] + new_json + html[m.end(2):]
    with open(args.html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\ninjected kimi_rows_after({len(after)}) + kimi_delta({len(pairs)} pairs); HTML {len(html)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
