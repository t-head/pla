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

"""Render a dated section into ``reports/perf_report_prefill.html``.

``reports/perf_report_prefill.html`` is a hand-maintained log that accumulates
one section per measurement round, newest first. This script takes the JSON
produced by ``scripts/prefill/collect_bench_vs_fla.py`` and inserts (or replaces)
the section for that run, leaving every historical section untouched.

Usage:
    python3 scripts/prefill/generate_prefill_report_section.py \
      --input reports/prefill_bench_20260911.json \
      --html reports/perf_report_prefill.html \
      --date 2026-09-11
"""

import argparse
import html
import json
import statistics
from pathlib import Path

ANCHOR = '<p class="subtitle">'


def _speedup_cell(speedup):
    cls = "win" if speedup > 1.0 else "loss"
    return f'<td class="{cls}">{speedup:.3f}x</td>'


def _verdict_cell(within_tol):
    return (
        '<td class="pass">PASS</td>'
        if within_tol
        else '<td class="loss">FAIL</td>'
    )


def _stats(records):
    speedups = [r["speedup"] for r in records]
    triton_total = sum(r["triton_ms"] for r in records)
    pla_total = sum(r["pla_ms"] for r in records)
    return {
        "count": len(records),
        "passed": sum(1 for r in records if r["within_tol"]),
        "wins": sum(1 for r in records if r["speedup"] > 1.0),
        "median": statistics.median(speedups),
        "geomean": statistics.geometric_mean(speedups),
        "min": min(speedups),
        "max": max(speedups),
        "total_ratio": triton_total / pla_total,
        "triton_total": triton_total,
        "pla_total": pla_total,
        "max_o_err": max(r["output_rel_err"] for r in records),
        "max_state_err": max(r["state_rel_err"] for r in records),
    }


def _gdn_table(records):
    rows = []
    for r in records:
        rows.append(
            "<tr>"
            f"<td>{r['batch']}</td><td>{r['num_v_heads']}</td>"
            f"<td>{r['num_k_heads']}</td><td>{r['seq_len']}</td>"
            f"<td>{r['total_tokens']}</td>"
            f"<td>{r['triton_ms']:.3f}</td><td>{r['pla_ms']:.3f}</td>"
            f"{_speedup_cell(r['speedup'])}"
            f"<td>{r['output_rel_err']:.3e}</td>"
            f"<td>{r['state_rel_err']:.3e}</td>"
            f"{_verdict_cell(r['within_tol'])}"
            "</tr>"
        )
    header = (
        "<thead><tr><th>B</th><th>V Heads</th><th>K Heads</th><th>T / seq</th>"
        "<th>Total tokens</th><th>FLA-Triton (ms)</th><th>PLA (ms)</th>"
        "<th>Speedup</th><th>Output rel err</th><th>State rel err</th>"
        "<th>Accuracy</th></tr></thead>"
    )
    return f"<table>\n{header}\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"


def _kda_table(records):
    rows = []
    for r in records:
        label = ",".join(str(length) for length in r["seq_lens"])
        shape = "ragged" if r["ragged"] else "uniform"
        rows.append(
            "<tr>"
            f"<td>tp{r['tp']}</td><td>{r['num_heads']}</td>"
            f"<td>{r['batch']}</td><td><code>{html.escape(label)}</code></td>"
            f"<td>{shape}</td><td>{r['total_tokens']}</td>"
            f"<td>{r['triton_ms']:.3f}</td><td>{r['pla_ms']:.3f}</td>"
            f"{_speedup_cell(r['speedup'])}"
            f"<td>{r['output_rel_err']:.3e}</td>"
            f"<td>{r['state_rel_err']:.3e}</td>"
            f"{_verdict_cell(r['within_tol'])}"
            "</tr>"
        )
    header = (
        "<thead><tr><th>TP</th><th>Local heads</th><th>B</th><th>seq_lens</th>"
        "<th>Shape</th><th>Total tokens</th><th>SGLang Triton (ms)</th>"
        "<th>PLA (ms)</th><th>Speedup</th><th>Output rel err</th>"
        "<th>State rel err</th><th>Accuracy</th></tr></thead>"
    )
    return f"<table>\n{header}\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"


def _metrics_grid(stats):
    return (
        '<div class="summary-grid">\n'
        f'  <div class="metric">Cases covered<span class="value">{stats["count"]}</span></div>\n'
        f'  <div class="metric">Accuracy passed<span class="value">{stats["passed"]} / {stats["count"]}</span></div>\n'
        f'  <div class="metric">PLA faster<span class="value">{stats["wins"]} / {stats["count"]}</span></div>\n'
        f'  <div class="metric">Median speedup<span class="value">{stats["median"]:.3f}x</span></div>\n'
        f'  <div class="metric">Total-time speedup<span class="value">{stats["total_ratio"]:.3f}x</span></div>\n'
        "</div>"
    )


def _family_block(title, records, table_html):
    stats = _stats(records)
    return (
        f"<h3>{title}</h3>\n"
        + _metrics_grid(stats)
        + "\n<p><strong>Summary:</strong> median speedup <strong>"
        f"{stats['median']:.3f}x</strong>, geomean <strong>{stats['geomean']:.3f}x</strong>, "
        f"range <strong>{stats['min']:.3f}–{stats['max']:.3f}x</strong>; "
        f"Triton/PLA total time is <strong>{stats['triton_total']:.3f}/{stats['pla_total']:.3f} ms</strong>, "
        f"ratio <strong>{stats['total_ratio']:.3f}x</strong>. Max output/state relative L2 error is "
        f"<code>{stats['max_o_err']:.3e}</code>/<code>{stats['max_state_err']:.3e}</code>.</p>\n"
        + table_html
    )


def build_section(data, date, extra_note=None):
    meta = data["provenance"]
    records = data["records"]
    gdn = [r for r in records if r["family"] == "gdn"]
    kda = [r for r in records if r["family"] == "kda"]

    parts = [
        "<section>",
        f"<h2>Prefill performance refresh: FLA-Triton vs PLA ({date})</h2>",
    ]
    if gdn:
        parts.append(
            _family_block(
                "GDN 13-shape standard set (FlashQLA)",
                gdn,
                _gdn_table(gdn),
            )
        )
    if kda:
        parts.append(
            _family_block(
                "KDA 24-case set (FlashKDAPro, TP 1/2/4/8)",
                kda,
                _kda_table(kda),
            )
        )

    commit = meta.get("git_commit") or "unknown"
    parts.append(
        "<p><strong>Methodology:</strong> both sides share the same input tensors, indexed FP32 state pool, "
        "slot indices and cu_seqlens, and each call clones its own copy of the pool; "
        "PLA calls its public entry directly, and the baseline is the SGLang Triton implementation vendored under "
        "<code>scripts/prefill/private/ref</code>. Timing takes the CUDA Event median, "
        f"<code>warmup={meta['warmup']}</code>, <code>repeat={meta['repeat']}</code>, "
        f"dtype <code>{meta['dtype']}</code>, head_dim = {meta['head_dim']}. "
        "Speedup is Triton ms / PLA ms; &gt;1 means PLA is faster.</p>"
    )
    parts.append(
        "<p><strong>Accuracy gate:</strong> each family keeps the original criterion of its own <code>bench_vs_fla.py</code>: "
        f"GDN uses <code>{html.escape(str(meta.get('gdn_gate', 'n/a')))}</code>, "
        f"KDA uses <code>{html.escape(str(meta.get('kda_gate', 'n/a')))}</code>. "
        "The output/state relative L2 errors in the table are diagnostics, not the KDA pass criterion: "
        "KDA outputs are small in magnitude, so the L2 ratio can still reach the 1e-2 range when element-wise <code>assert_close</code> passes.</p>"
    )
    parts.append(
        "<p><strong>Environment:</strong> "
        f"<code>{html.escape(meta['hostname'])}</code>, device <code>{html.escape(meta['device_name'])}</code>"
        " (single card, <code>CUDA_VISIBLE_DEVICES=0</code>), "
        f"PyTorch <code>{html.escape(meta['torch_version'])}</code>, "
        f"Python <code>{meta['python_version']}</code>, commit <code>{commit[:12]}</code>, "
        f"collected at {html.escape(meta['timestamp'])}.</p>"
    )
    parts.append(
        "<p><strong>Reproduce:</strong></p>\n"
        "<pre>CUDA_VISIBLE_DEVICES=0 python3 scripts/prefill/collect_bench_vs_fla.py \\\n"
        f"    --family all --warmup {meta['warmup']} --repeat {meta['repeat']} \\\n"
        "    --output reports/prefill_bench.json\n"
        "python3 scripts/prefill/generate_prefill_report_section.py \\\n"
        "    --input reports/prefill_bench.json --html reports/perf_report_prefill.html \\\n"
        f"    --date {date}</pre>"
    )
    if extra_note:
        parts.append(f'<p class="note">{extra_note}</p>')
    parts.append("</section>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note rendered at the end of the section.",
    )
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    section = build_section(data, args.date, args.note)

    page = args.html.read_text()
    marker = f"<h2>Prefill performance refresh: FLA-Triton vs PLA ({args.date})</h2>"
    if marker in page:
        start = page.rindex("<section>", 0, page.index(marker))
        end = page.index("</section>", start) + len("</section>")
        page = page[:start] + section + page[end:]
    else:
        insert_at = page.index("\n", page.index(ANCHOR)) + 1
        page = page[:insert_at] + "\n" + section + "\n" + page[insert_at:]

    args.html.write_text(page)
    print(f"updated {args.html} with section for {args.date}")


if __name__ == "__main__":
    main()
