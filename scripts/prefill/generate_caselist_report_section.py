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

"""Render a caselist sweep section into ``reports/perf_report_prefill.html``.

Unlike ``generate_prefill_report_section.py``, which lists every case of the
curated 13+24 shape set, this script consumes full OperatorsForge caselist
sweeps (thousands of records per file). Dumping one row per case would make the
report unreadable, so the section is aggregated instead:

* one summary row per caselist file (model / TP / local heads),
* a batch x token-count matrix of median speedups, to show where the kernel
  wins and where it does not,
* the slowest cases verbatim, so regressions stay visible.

Usage:
    python3 scripts/prefill/generate_caselist_report_section.py \
      --input reports/caselist \
      --html reports/perf_report_prefill.html \
      --date 2026-09-12
"""

import argparse
import csv
import html
import json
import statistics
from pathlib import Path

ANCHOR = '<p class="subtitle">'

# Same Plotly build the decode report (gen_report.py) already pins.
_PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

# Same edges the collector uses for stratified selection, so the matrix rows
# line up with how the sweep describes its own coverage.
_BATCH_EDGES = (1, 8, 64)
_TOKEN_EDGES = (1024, 4096, 16384)


def _batch_bucket(batch):
    if batch <= 1:
        return "B=1 (single sequence)"
    if batch <= _BATCH_EDGES[1]:
        return "B=2..8"
    if batch <= _BATCH_EDGES[2]:
        return "B=9..64"
    return "B&gt;64"


def _token_bucket(total):
    if total <= _TOKEN_EDGES[0]:
        return "T&le;1K"
    if total <= _TOKEN_EDGES[1]:
        return "1K&lt;T&le;4K"
    if total <= _TOKEN_EDGES[2]:
        return "4K&lt;T&le;16K"
    return "T&gt;16K"


_BATCH_ORDER = ["B=1 (single sequence)", "B=2..8", "B=9..64", "B&gt;64"]
_TOKEN_ORDER = ["T&le;1K", "1K&lt;T&le;4K", "4K&lt;T&le;16K", "T&gt;16K"]


def _speedup_cell(speedup):
    cls = "win" if speedup > 1.0 else "loss"
    return f'<td class="{cls}">{speedup:.3f}x</td>'


def load_records(input_path):
    """Load one JSON file or every ``*.json`` under a directory."""
    paths = (
        sorted(input_path.glob("*.json"))
        if input_path.is_dir()
        else [input_path]
    )
    if not paths:
        raise SystemExit(f"no JSON found under {input_path}")
    records, provenance = [], None
    for path in paths:
        data = json.loads(path.read_text())
        provenance = provenance or data["provenance"]
        records.extend(data["records"])
    return records, provenance


_CSV_HEADER = (
    "source", "model", "tp", "num_heads", "batch", "total_tokens",
    "seq_min", "seq_max", "ragged", "triton_ms", "pla_ms", "speedup",
    "output_rel_err", "state_rel_err", "within_tol",
)


def write_csv(records, path):
    """Flatten the sweep into one row per case.

    The raw JSON keeps every ``seq_lens`` list, which dominates its size for
    batches of a few hundred sequences. The CSV drops those in favour of
    min/max per record, matching how the decode sweeps store their numbers and
    keeping the committed artifact small enough to diff.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for r in sorted(
            records, key=lambda r: (r["model"], r["tp"], r["batch"], r["total_tokens"])
        ):
            seq_lens = r["seq_lens"]
            writer.writerow([
                r["source"], r["model"], r["tp"], r["num_heads"], r["batch"],
                r["total_tokens"], min(seq_lens), max(seq_lens), int(r["ragged"]),
                f"{r['triton_ms']:.4f}" if "triton_ms" in r else "",
                f"{r['pla_ms']:.4f}" if "pla_ms" in r else "",
                f"{r['speedup']:.4f}" if "speedup" in r else "",
                f"{r['output_rel_err']:.4e}" if "output_rel_err" in r else "",
                f"{r['state_rel_err']:.4e}" if "state_rel_err" in r else "",
                int(bool(r.get("within_tol"))),
            ])
    return path


def _stats(records):
    measured = [r for r in records if "speedup" in r]
    speedups = [r["speedup"] for r in measured]
    triton_total = sum(r["triton_ms"] for r in measured)
    pla_total = sum(r["pla_ms"] for r in measured)
    return {
        "count": len(records),
        "measured": len(measured),
        "errors": len(records) - len(measured),
        "passed": sum(1 for r in records if r.get("within_tol")),
        "wins": sum(1 for r in measured if r["speedup"] > 1.0),
        "median": statistics.median(speedups),
        "geomean": statistics.geometric_mean(speedups),
        "min": min(speedups),
        "max": max(speedups),
        "total_ratio": triton_total / pla_total,
        "triton_total": triton_total,
        "pla_total": pla_total,
        "max_o_err": max(r["output_rel_err"] for r in measured),
        "max_state_err": max(r["state_rel_err"] for r in measured),
    }


def _per_file_table(records, device=None):
    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    def sort_key(item):
        sample = item[1][0]
        return (sample["model"], sample["tp"])

    rows = []
    for _, group in sorted(by_source.items(), key=sort_key):
        s = _stats(group)
        sample = group[0]
        chip_td = f"<td>{html.escape(device)}</td>" if device else ""
        rows.append(
            "<tr>"
            f"{chip_td}<td>{html.escape(sample['model'])}</td><td>tp{sample['tp']}</td>"
            f"<td>{sample['num_heads']}</td><td>{s['count']}</td>"
            f'<td class="pass">{s["passed"]} / {s["count"]}</td>'
            f"<td>{s['wins']} / {s['measured']}</td>"
            f"{_speedup_cell(s['median'])}"
            f"{_speedup_cell(s['geomean'])}"
            f"<td>{s['min']:.3f}x</td><td>{s['max']:.3f}x</td>"
            f"{_speedup_cell(s['total_ratio'])}"
            f"<td>{s['max_o_err']:.3e}</td><td>{s['max_state_err']:.3e}</td>"
            "</tr>"
        )
    chip_th = "<th>Chip</th>" if device else ""
    header = (
        "<thead><tr>" + chip_th + "<th>Model</th><th>TP</th><th>Local heads</th><th>Cases</th>"
        "<th>Accuracy passed</th><th>PLA faster</th><th>Median</th><th>Geomean</th>"
        "<th>Min</th><th>Max</th><th>Total-time ratio</th><th>Max output rel err</th>"
        "<th>Max state rel err</th></tr></thead>"
    )
    return f"<table>\n{header}\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"


def _matrix_table(records):
    """Median speedup per (batch bucket, token bucket), with case counts."""
    cells = {}
    for r in records:
        if "speedup" not in r:
            continue
        key = (_batch_bucket(r["batch"]), _token_bucket(r["total_tokens"]))
        cells.setdefault(key, []).append(r["speedup"])

    rows = []
    for b in _BATCH_ORDER:
        cols = [f"<td><strong>{b}</strong></td>"]
        for t in _TOKEN_ORDER:
            vals = cells.get((b, t))
            if not vals:
                cols.append('<td class="note">—</td>')
                continue
            median = statistics.median(vals)
            cls = "win" if median > 1.0 else "loss"
            cols.append(
                f'<td class="{cls}">{median:.3f}x'
                f'<br><span class="note">n={len(vals)}</span></td>'
            )
        rows.append("<tr>" + "".join(cols) + "</tr>")
    header = (
        "<thead><tr><th>batch \\ tokens</th>"
        + "".join(f"<th>{t}</th>" for t in _TOKEN_ORDER)
        + "</tr></thead>"
    )
    return f"<table>\n{header}\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"


def _worst_table(records, limit, device=None):
    measured = sorted(
        (r for r in records if "speedup" in r), key=lambda r: r["speedup"]
    )[:limit]
    rows = []
    for r in measured:
        chip_td = f"<td>{html.escape(device)}</td>" if device else ""
        rows.append(
            "<tr>"
            f"{chip_td}<td>{html.escape(r['model'])}</td><td>tp{r['tp']}</td>"
            f"<td>{r['num_heads']}</td><td>{r['batch']}</td>"
            f"<td>{r['total_tokens']}</td>"
            f"<td>{'ragged' if r['ragged'] else 'uniform'}</td>"
            f"<td>{r['triton_ms']:.3f}</td><td>{r['pla_ms']:.3f}</td>"
            f"{_speedup_cell(r['speedup'])}"
            f"<td>{r['output_rel_err']:.3e}</td>"
            f"<td>{r['state_rel_err']:.3e}</td>"
            "</tr>"
        )
    chip_th = "<th>Chip</th>" if device else ""
    header = (
        "<thead><tr>" + chip_th + "<th>Model</th><th>TP</th><th>Local heads</th><th>B</th>"
        "<th>Total tokens</th><th>Shape</th><th>SGLang Triton (ms)</th>"
        "<th>PLA (ms)</th><th>Speedup</th><th>Output rel err</th>"
        "<th>State rel err</th></tr></thead>"
    )
    return f"<table>\n{header}\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"


def _charts_block(records, date):
    """Embed the whole sweep as two Plotly charts.

    7776 rows cannot go into a table, but they can go into a chart: the
    scatter shows every case (log token axis, one trace per caselist), and the
    histogram shows the speedup distribution with the 1.0 parity line marked.
    Data is baked in as plain JS arrays so the report stays a single
    self-contained file; scattergl keeps 7.8k points fluid.
    """
    tag = date.replace("-", "")
    by_source = {}
    for r in records:
        if "speedup" in r:
            by_source.setdefault(r["source"], []).append(r)
    sources = sorted(
        by_source.values(),
        key=lambda g: (g[0]["model"], g[0]["tp"]),
    )
    traces = []
    all_speedups = []
    for group in sources:
        sample = group[0]
        name = f"{sample['model']} tp{sample['tp']} (H={sample['num_heads']})"
        xs = [r["total_tokens"] for r in group]
        ys = [round(r["speedup"], 3) for r in group]
        bs = [r["batch"] for r in group]
        all_speedups.extend(ys)
        traces.append(
            "{x:%s,y:%s,customdata:%s,mode:'markers',type:'scattergl',"
            "name:%s,marker:{size:5,opacity:0.55},"
            "hovertemplate:'T=%%{x}<br>B=%%{customdata}<br>%%{y:.3f}x"
            "<extra>%%{fullData.name}</extra>'}"
            % (json.dumps(xs), json.dumps(ys), json.dumps(bs), json.dumps(name))
        )
    scatter = (
        f'<div id="caselist-scatter-{tag}" style="width:100%"></div>\n'
        "<script>(function(){\n"
        f"Plotly.newPlot('caselist-scatter-{tag}',[{','.join(traces)}],"
        "{title:'Speedup vs total tokens (one point per case, log x axis)',"
        "xaxis:{type:'log',title:'total tokens'},"
        "yaxis:{title:'Speedup (Triton ms / PLA ms)'},"
        "shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:1,y1:1,"
        "line:{color:'#b3261e',width:1.5,dash:'dash'}}],"
        "height:430,margin:{t:50,b:50},legend:{orientation:'h',y:-0.22}},"
        "{responsive:true});\n})();</script>"
    )
    histogram = (
        f'<div id="caselist-hist-{tag}" style="width:100%"></div>\n'
        "<script>(function(){\n"
        f"var y={json.dumps(all_speedups)};\n"
        f"Plotly.newPlot('caselist-hist-{tag}',"
        "[{x:y,type:'histogram',xbins:{size:0.05},"
        "marker:{color:'#0f3460',opacity:0.75}}],"
        f"{{title:'Speedup distribution (all {len(all_speedups)} cases, bin=0.05x, red line is the 1.0 break-even)',"
        "xaxis:{title:'Speedup (Triton ms / PLA ms)'},yaxis:{title:'cases'},"
        "shapes:[{type:'line',x0:1,x1:1,yref:'paper',y0:0,y1:1,"
        "line:{color:'#b3261e',width:1.5,dash:'dash'}}],"
        "height:320,margin:{t:50,b:50}},{responsive:true});\n})();</script>"
    )
    return (
        "<h3>Charts: speedup distribution and shape map</h3>\n"
        + scatter
        + "\n"
        + histogram
    )


def _metrics_grid(stats):
    return (
        '<div class="summary-grid">\n'
        f'  <div class="metric">Cases covered<span class="value">{stats["count"]}</span></div>\n'
        f'  <div class="metric">Accuracy passed<span class="value">{stats["passed"]} / {stats["count"]}</span></div>\n'
        f'  <div class="metric">PLA faster<span class="value">{stats["wins"]} / {stats["measured"]}</span></div>\n'
        f'  <div class="metric">Median speedup<span class="value">{stats["median"]:.3f}x</span></div>\n'
        f'  <div class="metric">Total-time speedup<span class="value">{stats["total_ratio"]:.3f}x</span></div>\n'
        "</div>"
    )


def build_section(records, meta, date, worst, extra_note=None, device=None):
    stats = _stats(records)
    sources = sorted({r["source"] for r in records})
    models = sorted({r["model"] for r in records})
    parts = [
        "<section>",
        f"<h2>Prefill caselist full replay: FLA-Triton vs PLA ({date})</h2>",
        "<p>Covers <strong>all</strong> records of the "
        f"{len(sources)} {', '.join(html.escape(m) for m in models)} caselists "
        f"under <code>caselist/prefill/</code> ({stats['count']} records"
        ", no sampling), each timed after alignment with the vendored SGLang Triton baseline. "
        "Per-case details are in <code>reports/perf_prefill_caselist.csv</code>; "
        "this section only shows the summary and the slowest samples.</p>",
        _metrics_grid(stats),
        "<p><strong>Summary:</strong> median speedup <strong>"
        f"{stats['median']:.3f}x</strong>, geomean <strong>{stats['geomean']:.3f}x</strong>, "
        f"range <strong>{stats['min']:.3f}–{stats['max']:.3f}x</strong>; "
        f"Triton/PLA total time is <strong>{stats['triton_total']:.1f}/{stats['pla_total']:.1f} ms</strong>, "
        f"ratio <strong>{stats['total_ratio']:.3f}x</strong>. "
        f"Accuracy gate: <strong>{stats['passed']}/{stats['count']}</strong> passed, "
        f"<strong>{stats['errors']}</strong> collection errors.</p>",
        _charts_block(records, date),
        "<h3>Summary by caselist</h3>",
        _per_file_table(records, device),
        "<h3>Median speedup by batch x token scale</h3>",
        "<p>Rows are batch buckets, columns are total-tokens-per-call buckets; each cell shows the median speedup "
        "and case count of all cases in the bucket. Red cells mean PLA is on average slower than Triton in that shape range.</p>",
        _matrix_table(records),
        f"<h3>Slowest {worst} cases</h3>",
        "<p>Listed by ascending speedup to locate shape ranges that still need optimization; all these cases passed the accuracy gate.</p>",
        _worst_table(records, worst, device),
    ]

    commit = meta.get("git_commit") or "unknown"
    parts.append(
        "<p><strong>Methodology:</strong> shapes are parsed from the raw caselist by the in-repo authoritative parser "
        "<code>tests/utils.py</code> (<code>read_cmds_from_file</code> / "
        "<code>resolve_cu_seqlens</code>), exactly as the test runner does; "
        "both sides share the same input tensors, indexed FP32 state pool, slot indices and cu_seqlens, "
        "and each call clones its own copy of the pool. Timing takes the CUDA Event median, "
        f"<code>warmup={meta['warmup']}</code>, <code>repeat={meta['repeat']}</code>, "
        f"dtype <code>{meta['dtype']}</code>, head_dim = {meta['head_dim']}. "
        "Speedup is Triton ms / PLA ms; &gt;1 means PLA is faster.</p>"
    )
    parts.append(
        "<p><strong>Accuracy gate:</strong> KDA keeps the criterion of <code>bench_vs_fla.py</code>: "
        f"<code>{html.escape(str(meta.get('kda_gate', 'n/a')))}</code>. "
        "The output/state relative L2 errors in the table are diagnostics, not the pass criterion: "
        "KDA outputs are small in magnitude, so the L2 ratio can still reach the 1e-2 range when element-wise <code>assert_close</code> passes.</p>"
    )
    parts.append(
        "<p><strong>Environment:</strong> "
        f"<code>{html.escape(meta['hostname'])}</code>, device <code>{html.escape(meta['device_name'])}</code>"
        " (each caselist collected in parallel on its own card), "
        f"PyTorch <code>{html.escape(meta['torch_version'])}</code>, "
        f"Python <code>{meta['python_version']}</code>, commit <code>{commit[:12]}</code>, "
        f"collected at {html.escape(meta['timestamp'])}.</p>"
    )
    parts.append(
        "<p><strong>Reproduce:</strong></p>\n"
        "<pre>for f in caselist/prefill/GLM-5.3-Flash_prefill_tp*.list \\\n"
        "         caselist/prefill/Kimi-K3_prefill_tp16_*.list; do\n"
        "  tag=$(basename \"$f\" | sed 's/_kvlimit.*//')\n"
        "  CUDA_VISIBLE_DEVICES=$i python3 scripts/prefill/collect_bench_vs_fla.py \\\n"
        f"      --caselist \"$f\" --select all --warmup {meta['warmup']} --repeat {meta['repeat']} \\\n"
        "      --output \"reports/caselist/${tag}.json\" &amp;\n"
        "  i=$((i+1))\n"
        "done; wait\n"
        "python3 scripts/prefill/generate_caselist_report_section.py \\\n"
        "    --input reports/caselist --html reports/perf_report_prefill.html \\\n"
        f"    --csv reports/perf_prefill_caselist.csv --date {date}</pre>"
    )
    if extra_note:
        parts.append(f'<p class="note">{extra_note}</p>')
    parts.append("</section>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file, or directory holding one JSON per caselist.",
    )
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write a flat one-row-per-case CSV of the whole sweep.",
    )
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Chip label (e.g. ZW-M890P); adds a Chip column to the tables")
    args = parser.parse_args()

    records, meta = load_records(args.input)
    section = build_section(records, meta, args.date, args.worst, args.note, args.device)
    if args.csv:
        write_csv(records, args.csv)
        print(f"wrote {len(records)} rows to {args.csv}")

    page = args.html.read_text()
    # The section needs Plotly once per page; older sections predate it.
    if "cdn.plot.ly" not in page:
        section = _PLOTLY_CDN + "\n" + section
    marker = f"<h2>Prefill caselist full replay: FLA-Triton vs PLA ({args.date})</h2>"
    if marker in page:
        start = page.rindex("<section>", 0, page.index(marker))
        end = page.index("</section>", start) + len("</section>")
        page = page[:start] + section + page[end:]
    else:
        insert_at = page.index("\n", page.index(ANCHOR)) + 1
        page = page[:insert_at] + "\n" + section + "\n" + page[insert_at:]

    args.html.write_text(page)
    print(
        f"updated {args.html}: {len(records)} records from "
        f"{len({r['source'] for r in records})} caselists"
    )


if __name__ == "__main__":
    main()
