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

"""Generate a data-embedded PLA vs FLA-Triton decode performance report.

The output HTML embeds all data as JSON and uses Plotly.js (CDN) for visualization.
It can be opened directly in a browser (file://) without any web server.

Usage:
    python3 gen_report.py                          # default: reads from reports/
    python3 gen_report.py --report-dir reports/    # explicit dir
    python3 gen_report.py --output reports/perf_report_890p.html
    python3 gen_report.py --manifest reports/run_manifest.json
"""
import argparse
import csv
import html
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from html import escape as escape_html


KERNELS = [
    ("k_last",        "perf_k_last.csv",        "k_last (decode, T=1)"),
    ("k_last_mtp",    "perf_k_last_mtp.csv",    "k_last_mtp (MTP verify)"),
    ("k_last_packed", "perf_k_last_packed.csv",  "k_last_packed (packed QKV)"),
]

# Keep in sync with the complete decode list of the released Kimi-K3; reports with missing rows are rejected.
EXPECTED_RAW_ROWS = 6931
WIN_RELATIVE_THRESHOLD = 0.03
WIN_ABSOLUTE_THRESHOLD_US = 3.0

CSV_HEADER = [
    "file", "B", "H", "HV", "K", "V", "dtype", "is_kda",
    "cuda_cycles", "triton_cycles", "cycle_ratio",
    "cuda_time_us", "triton_time_us",
    "cuda_tensor_pct", "triton_tensor_pct",
    "cuda_dram_pct", "triton_dram_pct",
    "cuda_host_us", "triton_host_us", "host_ratio",
    "cuda_e2e_us", "triton_e2e_us", "e2e_ratio",
]


def collapse_min_per_workload(rows):
    """Collapse repeated measurements of the same workload into one row.

    The caselist repeats each batch_size several times (different seqlen_q),
    but decode kernels ignore seqlen_q, so these are just repeated timings of
    the SAME workload. A cold-start / DVFS-not-ramped invocation can leave one
    repeat with an abnormally high time (outlier below the ratio=1 line).

    We group by (file,B,H,HV,K,V,dtype) and take the MIN independently per
    timing column (each side keeps its own best/least-noisy measurement),
    then recompute the ratios. Non-timing fields are kept from the first row.
    """
    groups = defaultdict(list)
    order = []
    for r in rows:
        key = (r.get("file"), r.get("B"), r.get("H"), r.get("HV"),
               r.get("K"), r.get("V"), r.get("dtype"))
        if key not in groups:
            order.append(key)
        groups[key].append(r)

    def fmin(rs, col):
        vals = []
        for r in rs:
            try:
                v = float(r.get(col))
            except (TypeError, ValueError):
                continue
            if v > 0:
                vals.append(v)
        return min(vals) if vals else None

    collapsed = []
    for key in order:
        rs = groups[key]
        base = dict(rs[0])
        for cuda_col, tri_col, ratio_col in (
            ("cuda_time_us", "triton_time_us", "cycle_ratio"),
            ("cuda_host_us", "triton_host_us", "host_ratio"),
            ("cuda_e2e_us", "triton_e2e_us", "e2e_ratio"),
        ):
            cmin = fmin(rs, cuda_col)
            tmin = fmin(rs, tri_col)
            if cmin is not None:
                base[cuda_col] = f"{cmin:.1f}"
            if tmin is not None:
                base[tri_col] = f"{tmin:.1f}"
            if cmin and tmin:
                base[ratio_col] = f"{tmin / cmin:.3f}"
        collapsed.append(base)
    return collapsed


def _positive_finite(value, field, path, line_number):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "{}:{}: {} must be numeric, got {!r}".format(
                path, line_number, field, value
            )
        )
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            "{}:{}: {} must be positive and finite, got {!r}".format(
                path, line_number, field, value
            )
        )
    return number


def read_csv_data(path, expected_rows=EXPECTED_RAW_ROWS):
    """Read and strictly validate one complete decode result CSV."""
    if not os.path.isfile(path):
        raise ValueError("required CSV is missing: {}".format(path))

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(
                "{}: CSV header mismatch\nexpected: {}\nactual:   {}".format(
                    path, CSV_HEADER, reader.fieldnames
                )
            )
        for line_number, row in enumerate(reader, 2):
            if (None in row or set(row) != set(CSV_HEADER)
                    or any(row[field] is None for field in CSV_HEADER)):
                raise ValueError(
                    "{}:{}: malformed row or unexpected field count".format(
                        path, line_number
                    )
                )
            missing = [field for field in CSV_HEADER if not str(row[field]).strip()]
            if missing:
                raise ValueError(
                    "{}:{}: empty required fields: {}".format(
                        path, line_number, ", ".join(missing)
                    )
                )
            if row["file"] == "file":
                raise ValueError(
                    "{}:{}: repeated CSV header in data".format(path, line_number)
                )
            for field in ("B", "H", "HV", "K", "V"):
                try:
                    value = int(row[field])
                except ValueError:
                    raise ValueError(
                        "{}:{}: {} must be an integer, got {!r}".format(
                            path, line_number, field, row[field]
                        )
                    )
                if value <= 0:
                    raise ValueError(
                        "{}:{}: {} must be positive, got {!r}".format(
                            path, line_number, field, row[field]
                        )
                    )
            if row["is_kda"].lower() not in ("true", "false"):
                raise ValueError(
                    "{}:{}: is_kda must be True or False, got {!r}".format(
                        path, line_number, row["is_kda"]
                    )
                )
            _positive_finite(row["cuda_e2e_us"], "cuda_e2e_us", path, line_number)
            _positive_finite(row["triton_e2e_us"], "triton_e2e_us", path, line_number)
            _positive_finite(row["e2e_ratio"], "e2e_ratio", path, line_number)
            rows.append(row)

    if len(rows) != expected_rows:
        raise ValueError(
            "{}: expected exactly {} raw rows, found {}".format(
                path, expected_rows, len(rows)
            )
        )
    return rows


def _manifest_int_map(manifest, field):
    value = manifest.get(field)
    if not isinstance(value, dict):
        raise ValueError("manifest field {!r} must be an object".format(field))
    expected_keys = {kernel_id for kernel_id, _, _ in KERNELS}
    if set(value) != expected_keys:
        raise ValueError(
            "manifest field {!r} must contain exactly {}".format(
                field, sorted(expected_keys)
            )
        )
    parsed = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                "manifest {}.{} must be an integer".format(field, key)
            )
        parsed[key] = item
    return parsed


def read_manifest(path):
    """Read and validate the completion contract for a publishable report."""
    if not os.path.isfile(path):
        raise ValueError("required run manifest is missing: {}".format(path))
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError("could not read manifest {}: {}".format(path, exc))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object: {}".format(path))
    if manifest.get("status") != "completed":
        raise ValueError(
            "manifest status must be 'completed', got {!r}".format(
                manifest.get("status")
            )
        )

    expected = _manifest_int_map(manifest, "expected_rows_by_kernel")
    actual = _manifest_int_map(manifest, "result_rows_by_kernel")
    skipped = _manifest_int_map(manifest, "skipped_rows_by_kernel")
    required = {kernel_id: EXPECTED_RAW_ROWS for kernel_id, _, _ in KERNELS}
    if expected != required:
        raise ValueError(
            "manifest expected_rows_by_kernel must be {}, got {}".format(
                required, expected
            )
        )
    if actual != required:
        raise ValueError(
            "manifest result_rows_by_kernel must be {}, got {}".format(
                required, actual
            )
        )
    if any(skipped.values()) or manifest.get("skipped_total_rows") != 0:
        raise ValueError(
            "manifest reports skipped rows: per-kernel={}, total={!r}".format(
                skipped, manifest.get("skipped_total_rows")
            )
        )
    if manifest.get("raw_case_count") != EXPECTED_RAW_ROWS:
        raise ValueError(
            "manifest raw_case_count must be {}, got {!r}".format(
                EXPECTED_RAW_ROWS, manifest.get("raw_case_count")
            )
        )
    expected_total = EXPECTED_RAW_ROWS * len(KERNELS)
    for field in ("expected_total_rows", "result_total_rows"):
        if manifest.get(field) != expected_total:
            raise ValueError(
                "manifest {} must be {}, got {!r}".format(
                    field, expected_total, manifest.get(field)
                )
            )
    return manifest


def format_provenance(manifest, manifest_path, counts):
    """Build escaped provenance/count HTML before interpolating the template."""
    labels = {kernel_id: title for kernel_id, _, title in KERNELS}
    count_rows = []
    for kernel_id, _, _ in KERNELS:
        item = counts[kernel_id]
        count_rows.append(
            "<tr><td>{}</td><td>{:,}</td><td>{:,}</td><td>{:,}</td></tr>".format(
                html.escape(labels[kernel_id]),
                item["raw"], item["expected"], item["collapsed"],
            )
        )
    raw_total = sum(item["raw"] for item in counts.values())
    expected_total = sum(item["expected"] for item in counts.values())
    collapsed_total = sum(item["collapsed"] for item in counts.values())
    count_rows.append(
        "<tr><td><b>All kernels</b></td><td><b>{:,}</b></td>"
        "<td><b>{:,}</b></td><td><b>{:,}</b></td></tr>".format(
            raw_total, expected_total, collapsed_total
        )
    )

    def compact(value):
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return str(value)

    provenance_fields = [
        ("Manifest", os.path.abspath(manifest_path)),
        ("Status", manifest.get("status")),
        ("Created", manifest.get("created_at", "not recorded")),
        ("Finished", manifest.get("finished_at", "not recorded")),
        ("Cards", manifest.get("cards", "not recorded")),
        ("Mode", manifest.get("mode", "not recorded")),
        ("Warmup / iterations", "{} / {}".format(
            manifest.get("warmup", "?"), manifest.get("iters", "?")
        )),
        ("Max memory per case", "{} GB".format(manifest.get("max_mem_gb", "?"))),
        ("Caselists / raw cases", "{} / {}".format(
            manifest.get("case_file_count", "?"), manifest.get("raw_case_count", "?")
        )),
        ("Method", manifest.get("method", "not recorded")),
        ("Filters", manifest.get("filters", "not recorded")),
    ]
    provenance_rows = "".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(label), html.escape(compact(value))
        )
        for label, value in provenance_fields
    )
    manifest_text = html.escape(json.dumps(manifest, indent=2, sort_keys=True))
    return """
<section class="provenance">
  <h2>Coverage and provenance</h2>
  <p>All raw records are validated before duplicate semantic workloads are
     min-collapsed for visualization. No skipped case is publishable.</p>
  <table>
    <tr><th>Kernel</th><th>Raw rows</th><th>Expected rows</th><th>Collapsed workloads</th></tr>
    {count_rows}
  </table>
  <table class="kv-table">{provenance_rows}</table>
  <details><summary>Complete run manifest</summary><pre>{manifest_text}</pre></details>
</section>
""".format(
        count_rows="".join(count_rows),
        provenance_rows=provenance_rows,
        manifest_text=manifest_text,
    )


def shorten_case_name(name):
    """Shorten long case names for legend."""
    name = name.replace("_decode_", " ").replace("_kvlimit_200GB_model_len_128K_", " ")
    name = name.replace("_op.list", "").replace("gdn", "GDN").replace("kda", "KDA")
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Generate a validated PLA vs FLA-Triton decode report"
    )
    parser.add_argument("--report-dir", type=str, default="reports",
                        help="Directory containing merged CSV files (default: reports/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: <report-dir>/perf_report_890p.html)")
    parser.add_argument("--device", type=str, default=None,
                        help="device name shown in the report; defaults to the candidate device info from the manifest")
    parser.add_argument(
        "--manifest", type=str, default=None,
        help=("Completed run manifest (default: "
              "<report-dir>/run_manifest.json)"),
    )
    args = parser.parse_args()

    report_dir = args.report_dir
    output_path = args.output or os.path.join(report_dir, "perf_report_890p.html")
    manifest_path = args.manifest or os.path.join(report_dir, "run_manifest.json")

    # A report is publishable only when both the run completion contract and
    # all three full raw datasets validate.
    try:
        manifest = read_manifest(manifest_path)
        all_data = {}
        counts = {}
        for kernel_id, csv_file, _ in KERNELS:
            path = os.path.join(report_dir, csv_file)
            raw_rows = read_csv_data(path)
            raw_n = len(raw_rows)
            rows = collapse_min_per_workload(raw_rows)
            all_data[kernel_id] = rows
            counts[kernel_id] = {
                "raw": raw_n,
                "expected": manifest["expected_rows_by_kernel"][kernel_id],
                "collapsed": len(rows),
            }
            print(
                "  {}: {} raw rows -> {} workloads (min-collapsed)".format(
                    path, counts[kernel_id]["raw"], len(rows)
                )
            )
    except ValueError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1

    # Serialize data as JSON for embedding
    data_json = json.dumps(
        all_data, allow_nan=False, separators=(",", ":")
    ).replace("</", "<\\/")
    kernels_json = json.dumps(KERNELS, separators=(",", ":"))
    provenance_html = format_provenance(manifest, manifest_path, counts)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = args.device or manifest.get("provenance", {}).get("candidate", {}).get("gpu")
    device_suffix = " — " + escape_html(device) if device else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PLA vs FLA-Triton Decode E2E Performance{device_suffix}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #1a1a2e; margin-bottom: 5px; }}
h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 40px; }}
.subtitle {{ color: #666; margin-bottom: 20px; }}
.method-note {{ background: #eef6ff; border-left: 4px solid #0f3460; padding: 10px 14px;
                border-radius: 3px; color: #26374a; }}
.chart-container {{ background: white; border-radius: 8px; padding: 15px;
                    margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
th {{ background: #0f3460; color: white; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.cuda-win {{ color: #2ca02c; font-weight: bold; }}
.triton-win {{ color: #d62728; font-weight: bold; }}
.tabs {{ display: flex; gap: 4px; margin-bottom: 15px; }}
.tab {{ padding: 8px 16px; border: 1px solid #ddd; border-radius: 4px 4px 0 0;
        cursor: pointer; background: #f0f0f0; }}
.tab.active {{ background: white; border-bottom-color: white; font-weight: bold; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.filter-bar {{ margin-bottom: 15px; padding: 10px; background: white; border-radius: 6px;
               box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.filter-bar label {{ margin-right: 15px; font-size: 13px; }}
.filter-bar select {{ padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; }}
.provenance {{ background: white; border-radius: 8px; padding: 4px 15px 15px;
               margin: 25px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.kv-table th {{ width: 210px; text-align: left; }}
.kv-table td {{ text-align: left; overflow-wrap: anywhere; }}
details {{ margin-top: 10px; }}
summary {{ cursor: pointer; font-weight: 600; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f8fa;
       border-radius: 5px; padding: 12px; font-size: 12px; }}
</style>
</head>
<body>
<h1>PLA vs FLA-Triton Decode Performance{device_suffix}</h1>
<p class="subtitle">Generated: {gen_time} | Default metric: end-to-end latency |
Y-axis = FLA-Triton / PLA | <b>Ratio &gt; 1 = PLA faster</b></p>
<p class="method-note"><b>Winner rule:</b> a result is a win only when it clears
both gates: more than 3% relative improvement and more than 3 &micro;s absolute
improvement. Everything else is a tie.</p>

{provenance_html}

<div class="filter-bar">
  <label>Metric:
    <select id="metricSelect" onchange="renderAll()">
      <option value="e2e" selected>End-to-End (host+device+sync)</option>
      <option value="device">Device Kernel Time</option>
      <option value="host">Host Time (launch overhead)</option>
    </select>
  </label>
  <label>Kernel:
    <select id="kernelSelect" onchange="renderAll()">
      <option value="all">All Kernels</option>
    </select>
  </label>
  <label>Model:
    <select id="modelSelect" onchange="renderAll()">
      <option value="all">All Models</option>
    </select>
  </label>
  <label>TP:
    <select id="tpSelect" onchange="renderAll()">
      <option value="all">All TP</option>
    </select>
  </label>
</div>

<div id="summaryTable"></div>
<div id="chartArea"></div>
<div id="dataTable"></div>

<script>
// Embedded data
var ALL_DATA = {data_json};
var KERNELS = {kernels_json};
var WIN_RELATIVE_THRESHOLD = {WIN_RELATIVE_THRESHOLD};
var WIN_ABSOLUTE_THRESHOLD_US = {WIN_ABSOLUTE_THRESHOLD_US};

var COLORS = [
  '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
  '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
  '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5'
];

function getMetricCols(metric) {{
  if (metric === 'host') return ['cuda_host_us', 'triton_host_us', 'host_ratio'];
  if (metric === 'e2e') return ['cuda_e2e_us', 'triton_e2e_us', 'e2e_ratio'];
  return ['cuda_time_us', 'triton_time_us', 'cycle_ratio'];
}}

function extractModel(filename) {{
  // Extract model name from caselist filename
  // e.g. "Qwen3.5-Max_decode_tp1_..." -> "Qwen3.5-Max"
  var m = filename.match(/^(.+?)_decode_tp/);
  return m ? m[1] : filename;
}}

function extractTP(filename) {{
  var m = filename.match(/_decode_(tp\\d+)_/);
  return m ? m[1] : 'unknown';
}}

function initFilters() {{
  var ks = document.getElementById('kernelSelect');
  KERNELS.forEach(function(k) {{
    if (ALL_DATA[k[0]]) {{
      var opt = document.createElement('option');
      opt.value = k[0]; opt.text = k[2];
      ks.appendChild(opt);
    }}
  }});
  // Extract unique models and TPs from caselist file names
  var models = new Set();
  var tps = new Set();
  for (var kid in ALL_DATA) {{
    ALL_DATA[kid].forEach(function(r) {{
      models.add(extractModel(r.file));
      tps.add(extractTP(r.file));
    }});
  }}
  var ms = document.getElementById('modelSelect');
  Array.from(models).sort().forEach(function(m) {{
    var opt = document.createElement('option');
    opt.value = m; opt.text = m;
    ms.appendChild(opt);
  }});
  var ts = document.getElementById('tpSelect');
  Array.from(tps).sort(function(a,b) {{
    return parseInt(a.replace('tp','')) - parseInt(b.replace('tp',''));
  }}).forEach(function(t) {{
    var opt = document.createElement('option');
    opt.value = t; opt.text = t.toUpperCase();
    ts.appendChild(opt);
  }});
}}

function winnerFor(plaTime, flaTime) {{
  var absoluteDelta = flaTime - plaTime;
  if (absoluteDelta > WIN_ABSOLUTE_THRESHOLD_US &&
      flaTime / plaTime > 1.0 + WIN_RELATIVE_THRESHOLD) return 'pla';
  if (-absoluteDelta > WIN_ABSOLUTE_THRESHOLD_US &&
      plaTime / flaTime > 1.0 + WIN_RELATIVE_THRESHOLD) return 'fla';
  return 'tie';
}}

function median(values) {{
  var sorted = values.slice().sort(function(a, b) {{ return a - b; }});
  var middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}}

function summarizeRows(rows, cols, label) {{
  var plaCol = cols[0], flaCol = cols[1];
  var pairs = rows.map(function(r) {{
    return [parseFloat(r[plaCol]), parseFloat(r[flaCol])];
  }}).filter(function(pair) {{
    return isFinite(pair[0]) && pair[0] > 0 && isFinite(pair[1]) && pair[1] > 0;
  }});
  if (pairs.length === 0) return null;
  var ratios = pairs.map(function(pair) {{ return pair[1] / pair[0]; }});
  var plaTotal = pairs.reduce(function(total, pair) {{ return total + pair[0]; }}, 0);
  var flaTotal = pairs.reduce(function(total, pair) {{ return total + pair[1]; }}, 0);
  var geomean = Math.exp(ratios.reduce(function(total, ratio) {{
    return total + Math.log(ratio);
  }}, 0) / ratios.length);
  var plaWins = 0, flaWins = 0, ties = 0;
  pairs.forEach(function(pair) {{
    var winner = winnerFor(pair[0], pair[1]);
    if (winner === 'pla') plaWins += 1;
    else if (winner === 'fla') flaWins += 1;
    else ties += 1;
  }});
  return {{
    kernel: label,
    aggregate: flaTotal / plaTotal,
    geomean: geomean,
    median: median(ratios),
    min: Math.min.apply(null, ratios),
    max: Math.max.apply(null, ratios),
    plaWins: plaWins,
    flaWins: flaWins,
    ties: ties,
    total: pairs.length
  }};
}}

function computeSummary(metric, kernelFilter, modelFilter, tpFilter) {{
  var cols = getMetricCols(metric);
  var summary = [];
  var combined = [];
  KERNELS.forEach(function(kinfo) {{
    var kid = kinfo[0], title = kinfo[2];
    if (kernelFilter !== 'all' && kernelFilter !== kid) return;
    var rows = ALL_DATA[kid];
    if (!rows || rows.length === 0) return;
    var filtered = rows.filter(function(r) {{ return matchFilters(r, modelFilter, tpFilter); }});
    var stats = summarizeRows(filtered, cols, title);
    if (stats) summary.push(stats);
    combined = combined.concat(filtered);
  }});
  if (summary.length > 1) {{
    summary.unshift(summarizeRows(combined, cols, 'All selected kernels'));
  }}
  return summary;
}}

function renderSummary(metric, kernelFilter, modelFilter, tpFilter) {{
  var summary = computeSummary(metric, kernelFilter, modelFilter, tpFilter);
  var metricLabel = metric === 'host' ? 'Host Time Ratio' : metric === 'e2e' ? 'E2E Ratio' : 'Device Kernel Ratio';
  var html = '<h2>Summary (' + metricLabel + ', FLA-Triton / PLA)</h2>';
  html += '<p>Aggregate ratio = sum(FLA-Triton time) / sum(PLA time); geomean and median use per-workload ratios.</p>';
  html += '<table><tr><th>Kernel</th><th>Aggregate Ratio</th><th>Geomean</th>';
  html += '<th>Median</th><th>Min</th><th>Max</th><th>PLA Wins</th><th>Ties</th>';
  html += '<th>FLA-Triton Wins</th></tr>';
  summary.forEach(function(s) {{
    var plaClass = s.plaWins > s.total/2 ? 'cuda-win' : '';
    var flaClass = s.flaWins > s.total/2 ? 'triton-win' : '';
    html += '<tr><td><b>' + s.kernel + '</b></td>';
    html += '<td>' + s.aggregate.toFixed(3) + '</td><td>' + s.geomean.toFixed(3) + '</td>';
    html += '<td>' + s.median.toFixed(3) + '</td><td>' + s.min.toFixed(3) + '</td>';
    html += '<td>' + s.max.toFixed(3) + '</td>';
    html += '<td class="' + plaClass + '">' + s.plaWins + '/' + s.total + '</td>';
    html += '<td>' + s.ties + '/' + s.total + '</td>';
    html += '<td class="' + flaClass + '">' + s.flaWins + '/' + s.total + '</td></tr>';
  }});
  html += '</table>';
  document.getElementById('summaryTable').innerHTML = html;
}}

function renderCharts(metric, kernelFilter, modelFilter, tpFilter) {{
  var cols = getMetricCols(metric);
  var cudaCol = cols[0];
  var tritonCol = cols[1];
  var area = document.getElementById('chartArea');
  area.innerHTML = '';
  if (typeof Plotly === 'undefined') {{
    area.innerHTML = '<div class="chart-container"><p>Interactive charts require access to cdn.plot.ly; the validated summary, provenance, and full detail tables remain available offline.</p></div>';
    return;
  }}

  KERNELS.forEach(function(kinfo) {{
    var kid = kinfo[0], title = kinfo[2];
    if (kernelFilter !== 'all' && kernelFilter !== kid) return;
    var rows = ALL_DATA[kid];
    if (!rows || rows.length === 0) return;

    // Group by case file
    var byFile = {{}};
    rows.forEach(function(r) {{
      if (!matchFilters(r, modelFilter, tpFilter)) return;
      if (!byFile[r.file]) byFile[r.file] = [];
      byFile[r.file].push(r);
    }});

    var traces = [];
    var files = Object.keys(byFile).sort();
    files.forEach(function(f, i) {{
      var pts = byFile[f].sort(function(a,b){{ return parseInt(a.B) - parseInt(b.B); }});
      var xs = pts.map(function(p){{ return parseInt(p.B); }});
      var ys = pts.map(function(p){{
        return parseFloat(p[tritonCol]) / parseFloat(p[cudaCol]);
      }});
      var hoverText = pts.map(function(p) {{
        return 'B=' + p.B + ' | PLA: ' + parseFloat(p[cudaCol]).toFixed(1) +
               'us, FLA-Triton: ' + parseFloat(p[tritonCol]).toFixed(1) + 'us';
      }});
      var shortName = f.replace(/_decode_/,' ').replace(/_kvlimit_200GB_model_len_128K_/,' ')
                       .replace(/_op.list/,'').replace('gdn','GDN').replace('kda','KDA');
      traces.push({{
        x: xs, y: ys, mode: 'lines+markers',
        name: shortName,
        text: hoverText, hoverinfo: 'text+name',
        line: {{ color: COLORS[i % COLORS.length], width: 2 }},
        marker: {{ size: 5 }}
      }});
    }});

    // Add ratio=1.0 reference line
    traces.push({{
      x: [1, 10000], y: [1.0, 1.0],
      mode: 'lines', name: 'ratio=1 (tie)',
      line: {{ color: '#999', width: 1, dash: 'dash' }},
      showlegend: true
    }});

    var metricLabel = metric === 'host' ? 'Host Time' : metric === 'e2e' ? 'E2E Time' : 'Device Kernel Time';
    var div = document.createElement('div');
    div.className = 'chart-container';
    div.id = kid + '_chart';
    div.style.height = '500px';
    area.appendChild(div);

    Plotly.newPlot(div.id, traces, {{
      title: {{ text: title + ' — ' + metricLabel + ' Ratio (FLA-Triton/PLA)', font: {{ size: 16 }} }},
      xaxis: {{ title: 'Batch Size (B)', type: 'log' }},
      yaxis: {{ title: 'Ratio (>1 = PLA faster)', zeroline: false }},
      legend: {{ orientation: 'h', y: -0.25, font: {{ size: 11 }} }},
      hovermode: 'x unified',
      margin: {{ l: 65, r: 30, t: 55, b: 110 }},
      shapes: [{{ type: 'line', x0: 0, x1: 1, y0: 1, y1: 1, xref: 'paper',
                  line: {{ color: '#e74c3c', width: 1.5, dash: 'dot' }} }}]
    }}, {{ responsive: true }});
  }});
}}

function renderDataTable(metric, kernelFilter, modelFilter, tpFilter) {{
  var cols = getMetricCols(metric);
  var cudaCol = cols[0], tritonCol = cols[1];
  var area = document.getElementById('dataTable');
  var html = '<h2>Detailed Data</h2>';

  KERNELS.forEach(function(kinfo) {{
    var kid = kinfo[0], title = kinfo[2];
    if (kernelFilter !== 'all' && kernelFilter !== kid) return;
    var rows = ALL_DATA[kid];
    if (!rows || rows.length === 0) return;

    var filtered = rows.filter(function(r){{ return matchFilters(r, modelFilter, tpFilter); }});
    if (filtered.length === 0) return;

    html += '<h3>' + title + ' (' + filtered.length + ' rows)</h3>';
    html += '<table><tr><th>File</th><th>B</th><th>H</th><th>HV</th><th>K</th><th>V</th>';
    html += '<th>PLA (us)</th><th>FLA-Triton (us)</th><th>FLA/PLA</th><th>Winner</th></tr>';
    filtered.sort(function(a,b){{ return parseInt(a.B) - parseInt(b.B); }});
    filtered.forEach(function(r) {{
      var plaTime = parseFloat(r[cudaCol]);
      var flaTime = parseFloat(r[tritonCol]);
      var ratio = flaTime / plaTime;
      var winnerCode = winnerFor(plaTime, flaTime);
      var winner = winnerCode === 'pla' ? '<span class=\\"cuda-win\\">PLA</span>' :
                   winnerCode === 'fla' ? '<span class=\\"triton-win\\">FLA-Triton</span>' : 'Tie';
      var shortFile = r.file.replace(/_kvlimit_200GB_model_len_128K_/,' ').replace(/_op.list/,'');
      html += '<tr><td style=\\"text-align:left;font-size:11px\\">' + shortFile + '</td>';
      html += '<td>' + r.B + '</td><td>' + r.H + '</td><td>' + r.HV + '</td><td>' + r.K + '</td><td>' + r.V + '</td>';
      html += '<td>' + plaTime.toFixed(1) + '</td>';
      html += '<td>' + flaTime.toFixed(1) + '</td>';
      html += '<td><b>' + ratio.toFixed(3) + '</b></td>';
      html += '<td>' + winner + '</td></tr>';
    }});
    html += '</table>';
  }});
  area.innerHTML = html;
}}

function matchFilters(row, modelFilter, tpFilter) {{
  if (modelFilter !== 'all' && extractModel(row.file) !== modelFilter) return false;
  if (tpFilter !== 'all' && extractTP(row.file) !== tpFilter) return false;
  return true;
}}

function renderAll() {{
  var metric = document.getElementById('metricSelect').value;
  var kernelFilter = document.getElementById('kernelSelect').value;
  var modelFilter = document.getElementById('modelSelect').value;
  var tpFilter = document.getElementById('tpSelect').value;
  renderSummary(metric, kernelFilter, modelFilter, tpFilter);
  renderCharts(metric, kernelFilter, modelFilter, tpFilter);
  renderDataTable(metric, kernelFilter, modelFilter, tpFilter);
}}

// Init
initFilters();
renderAll();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nGenerated: {output_path}")
    print(f"Open in browser: file://{os.path.abspath(output_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
