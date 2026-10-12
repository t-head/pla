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

"""Generate a self-contained decode performance HTML report from CSV files.

This is the decode counterpart used for the per-device sweep reports
(perf_report_890p.html / perf_report_810e.html).
It embeds all data as JSON and uses Plotly.js (CDN) for visualization, so the
output can be opened directly in a browser (file://) without any web server.

Usage:
    python3 scripts/decode/gen_decode_report.py --report-dir reports/
    python3 scripts/decode/gen_decode_report.py \
        --report-dir .qoder/tmp/report_810e \
        --device "ZW810E (PPU1.0/1.1, 11.165.159.10)" \
        --output reports/perf_report_810e.html

The report directory must contain perf_k_last.csv, perf_k_last_mtp.csv and
perf_k_last_packed.csv (missing files are skipped, not fatal).
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime


KERNELS = [
    ("k_last",        "perf_k_last.csv",        "k_last (decode, T=1)"),
    ("k_last_mtp",    "perf_k_last_mtp.csv",    "k_last_mtp (MTP verify)"),
    ("k_last_packed", "perf_k_last_packed.csv",  "k_last_packed (packed QKV)"),
]

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


def read_csv_data(path):
    """Read CSV and return list of dicts (missing file -> empty list)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("file") == "file":
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML decode perf report from CSVs")
    parser.add_argument("--report-dir", type=str, default="reports",
                        help="Directory containing merged CSV files (default: reports/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: <report-dir>/perf_report_890p.html)")
    parser.add_argument("--device", type=str, default=None,
                        help=('Device label shown in the report title, e.g. '
                              '"ZW-M890P (PPU1.5, 30.70.218.2)". The first '
                              "whitespace-separated token is used as the short "
                              "name in the subtitle."))
    args = parser.parse_args()

    report_dir = args.report_dir
    output_path = args.output or os.path.join(report_dir, "perf_report_890p.html")

    # Load all data
    all_data = {}
    for kernel_id, csv_file, title in KERNELS:
        path = os.path.join(report_dir, csv_file)
        rows = read_csv_data(path)
        if rows:
            raw_n = len(rows)
            rows = collapse_min_per_workload(rows)
            all_data[kernel_id] = rows
            print(f"  {path}: {raw_n} rows -> {len(rows)} workloads (min-collapsed)")
        else:
            print(f"  {path}: (not found or empty)")

    if not all_data:
        print("ERROR: No data found. Run perf_compare_parallel.sh first.")
        sys.exit(1)

    # Serialize data as JSON for embedding
    data_json = json.dumps(all_data)
    kernels_json = json.dumps(KERNELS)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if args.device:
        device_full = args.device
        device_short = args.device.split()[0]
        h1_html = f"PLA Kernel Performance Report — {device_full}"
        subtitle_prefix = (f"Decode full sweep | Device: <b>{device_short}</b> | ")
    else:
        h1_html = "PLA Kernel Performance Report"
        subtitle_prefix = ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PLA Kernel Performance: CUDA vs Triton</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #1a1a2e; margin-bottom: 5px; }}
h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 40px; }}
.subtitle {{ color: #666; margin-bottom: 20px; }}
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
</style>
</head>
<body>
<h1>{h1_html}</h1>
<p class="subtitle">{subtitle_prefix}Generated: {gen_time} | Y-axis = Triton_time / CUDA_time |
<b>Ratio &gt; 1 = CUDA faster</b>, Ratio &lt; 1 = Triton faster</p>

<div class="filter-bar">
  <label>Metric:
    <select id="metricSelect" onchange="renderAll()">
      <option value="device">Device Kernel Time</option>
      <option value="host">Host Time (launch overhead)</option>
      <option value="e2e">End-to-End (host+device+sync)</option>
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

function computeSummary(metric, modelFilter, tpFilter) {{
  var cols = getMetricCols(metric);
  var ratioCol = cols[2];
  var cudaCol = cols[0];
  var tritonCol = cols[1];
  var summary = [];
  KERNELS.forEach(function(kinfo) {{
    var kid = kinfo[0], title = kinfo[2];
    var rows = ALL_DATA[kid];
    if (!rows || rows.length === 0) return;
    var filtered = rows.filter(function(r) {{ return matchFilters(r, modelFilter, tpFilter); }});
    var ratios = filtered.map(function(r) {{ return parseFloat(r[ratioCol]) || 0; }}).filter(function(v){{ return v > 0; }});
    if (ratios.length === 0) return;
    var mean = ratios.reduce(function(a,b){{ return a+b; }},0) / ratios.length;
    var min = Math.min.apply(null, ratios);
    var max = Math.max.apply(null, ratios);
    // ±3% dead-band: near-ties (0.97..1.03) are within measurement noise and
    // counted as "Tie", not as a win for either side.
    var cudaWins = ratios.filter(function(r){{ return r > 1.03; }}).length;
    var tritonWins = ratios.filter(function(r){{ return r < 0.97; }}).length;
    var ties = ratios.length - cudaWins - tritonWins;
    summary.push({{ kernel: title, mean: mean, min: min, max: max,
                    cudaWins: cudaWins, tritonWins: tritonWins, ties: ties, total: ratios.length }});
  }});
  return summary;
}}

function renderSummary(metric, modelFilter, tpFilter) {{
  var summary = computeSummary(metric, modelFilter, tpFilter);
  var metricLabel = metric === 'host' ? 'Host Time Ratio' : metric === 'e2e' ? 'E2E Ratio' : 'Device Kernel Ratio';
  var html = '<h2>Summary (' + metricLabel + ', &plusmn;3% dead-band)</h2><table><tr><th>Kernel</th><th>Mean Ratio</th><th>Min</th><th>Max</th><th>CUDA Wins (ratio&gt;1.03)</th><th>Tie (&plusmn;3%)</th><th>Triton Wins (ratio&lt;0.97)</th></tr>';
  summary.forEach(function(s) {{
    var cwClass = s.cudaWins > s.total/2 ? 'cuda-win' : '';
    var twClass = s.tritonWins > s.total/2 ? 'triton-win' : '';
    html += '<tr><td><b>' + s.kernel + '</b></td><td>' + s.mean.toFixed(3) + '</td><td>' + s.min.toFixed(3) + '</td><td>' + s.max.toFixed(3) + '</td>';
    html += '<td class="' + cwClass + '">' + s.cudaWins + '/' + s.total + '</td>';
    html += '<td>' + s.ties + '/' + s.total + '</td>';
    html += '<td class="' + twClass + '">' + s.tritonWins + '/' + s.total + '</td></tr>';
  }});
  html += '</table>';
  document.getElementById('summaryTable').innerHTML = html;
}}

function renderCharts(metric, kernelFilter, modelFilter, tpFilter) {{
  var cols = getMetricCols(metric);
  var ratioCol = cols[2];
  var cudaCol = cols[0];
  var tritonCol = cols[1];
  var area = document.getElementById('chartArea');
  area.innerHTML = '';

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
      var ys = pts.map(function(p){{ return parseFloat(p[ratioCol]) || 0; }});
      var hoverText = pts.map(function(p) {{
        return 'B=' + p.B + ' | CUDA: ' + parseFloat(p[cudaCol]).toFixed(1) + 'us, Triton: ' + parseFloat(p[tritonCol]).toFixed(1) + 'us';
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
      title: {{ text: title + ' — ' + metricLabel + ' Ratio (Triton/CUDA)', font: {{ size: 16 }} }},
      xaxis: {{ title: 'Batch Size (B)', type: 'log' }},
      yaxis: {{ title: 'Ratio (>1 = CUDA faster)', zeroline: false }},
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
  var ratioCol = cols[2], cudaCol = cols[0], tritonCol = cols[1];
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
    html += '<th>CUDA (us)</th><th>Triton (us)</th><th>Ratio</th><th>Winner</th></tr>';
    filtered.sort(function(a,b){{ return parseInt(a.B) - parseInt(b.B); }});
    filtered.forEach(function(r) {{
      var ratio = parseFloat(r[ratioCol]) || 0;
      var winner = ratio > 1.0 ? '<span class=\\"cuda-win\\">CUDA</span>' : '<span class=\\"triton-win\\">Triton</span>';
      var shortFile = r.file.replace(/_kvlimit_200GB_model_len_128K_/,' ').replace(/_op.list/,'');
      html += '<tr><td style=\\"text-align:left;font-size:11px\\">' + shortFile + '</td>';
      html += '<td>' + r.B + '</td><td>' + r.H + '</td><td>' + r.HV + '</td><td>' + r.K + '</td><td>' + r.V + '</td>';
      html += '<td>' + parseFloat(r[cudaCol]).toFixed(1) + '</td>';
      html += '<td>' + parseFloat(r[tritonCol]).toFixed(1) + '</td>';
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
  renderSummary(metric, modelFilter, tpFilter);
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
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nGenerated: {output_path}")
    print(f"Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
