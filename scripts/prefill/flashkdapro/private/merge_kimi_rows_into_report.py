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

"""Merge a Kimi-K3 E2E sweep (rows JSONL) into flashkda-e2e-vs-fla.html.

The report is a self-contained HTML whose data lives in
  <script id="report-data" type="application/json">{"summary":..., "rows":[...]}</script>

This script ADDS a new dataset without destroying the original 2026-07-31 one:
  * keeps the original rows under blob["rows"] untouched
  * adds blob["kimi_rows"]  = the Kimi-K3 rows
  * adds blob["kimi_summary"] = aggregates (overall / by_suite / by_mode / by_tp)
  * injects a new "Kimi-K3 production case" panel + tab-ish section into the HTML body
So the original report stays valid and the new data is a clearly-labelled addition.

Usage:
  python3 merge_kimi_rows_into_report.py \
      --rows rows_tp8.jsonl rows_tp4.jsonl rows_tp2.jsonl rows_tp1.jsonl \
      --html .../flashkda-e2e-vs-fla.html \
      --meta-ppu-sha <sha> --meta-commit 3f436ec --meta-cr 29320733
"""
import argparse, json, math, os, re, statistics, sys
from collections import defaultdict


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def agg(rows, keyfn=None):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("speedup", 0) > 0]
    if not ok:
        return {}
    sp = [r["speedup"] for r in ok]
    out = {
        "count": len(ok),
        "geomean_speedup": geomean(sp),
        "median_speedup": statistics.median(sp),
        "win_count": sum(1 for s in sp if s > 1.0),
        "win_rate": sum(1 for s in sp if s > 1.0) / len(sp),
        "min_speedup": min(sp),
        "max_speedup": max(sp),
        "ppu_median_of_medians_ms": statistics.median([r["ppu_median_ms"] for r in ok]),
        "fla_median_of_medians_ms": statistics.median([r["fla_median_ms"] for r in ok]),
    }
    return out


def grouped(rows, field):
    g = defaultdict(list)
    for r in rows:
        g[r.get(field)].append(r)
    out = []
    for k in sorted(g.keys(), key=lambda x: (str(type(x)), x if x is not None else "")):
        a = agg(g[k])
        if a:
            a[field] = k
            out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--meta-ppu-sha", default="")
    ap.add_argument("--meta-commit", default="")
    ap.add_argument("--meta-cr", default="")
    ap.add_argument("--meta-date", default="")
    args = ap.parse_args()

    rows = []
    for p in args.rows:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("{"):
                    rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {len(args.rows)} files")
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"  ok={len(ok)} failed={len(rows)-len(ok)}")

    summary = {
        "meta": {
            "source": "Kimi-K3 prefill case lists (tp1/2/4/8)",
            "pla_commit": args.meta_commit,
            "code_review": args.meta_cr,
            "ppu_extension_sha256": args.meta_ppu_sha,
            "created_at": args.meta_date,
            "note": "pla.prefill.flashkdapro default dispatch vs FLA-Triton chunk_kda",
        },
        "overall": agg(rows),
        "by_suite": grouped(ok, "suite"),
        "by_mode": grouped(ok, "mode"),
        "by_tp": grouped(ok, "tp"),
        "case_count": len({r["case_id"] for r in rows}),
        "row_count": len(rows),
        "ok_count": len(ok),
    }
    sl = sorted([r for r in ok if r.get("speedup", 0) > 0], key=lambda r: r["speedup"])
    summary["slowest_relative"] = sl[:8]
    summary["fastest_relative"] = sl[-8:][::-1]

    print("\n=== OVERALL ===")
    o = summary["overall"]
    print(f"  n={o.get('count')} geomean={o.get('geomean_speedup'):.3f}x "
          f"median={o.get('median_speedup'):.3f}x win_rate={o.get('win_rate'):.1%} "
          f"range={o.get('min_speedup'):.2f}x..{o.get('max_speedup'):.2f}x")
    for grp in ("by_tp", "by_suite", "by_mode"):
        print(f"--- {grp} ---")
        for a in summary[grp]:
            key = [v for k, v in a.items() if k in ("tp", "suite", "mode")][0]
            print(f"  {key}: n={a['count']} geo={a['geomean_speedup']:.3f}x "
                  f"med={a['median_speedup']:.3f}x win={a['win_rate']:.0%} "
                  f"min={a['min_speedup']:.2f}x max={a['max_speedup']:.2f}x")

    # ---- inject into HTML ----
    with open(args.html, encoding="utf-8") as fh:
        html = fh.read()
    m = re.search(r'(<script id="report-data"[^>]*>)(.*?)(</script>)', html, re.S)
    if not m:
        print("ERROR: report-data script not found", file=sys.stderr)
        return 1
    blob = json.loads(m.group(2))
    blob["kimi_rows"] = rows
    blob["kimi_summary"] = summary
    new_json = json.dumps(blob, ensure_ascii=False, separators=(",", ":"))
    html = html[:m.start(2)] + new_json + html[m.end(2):]

    with open(args.html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\ninjected kimi_rows({len(rows)}) + kimi_summary into {args.html}")
    print(f"new HTML size: {len(html)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
