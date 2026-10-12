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

"""Generate a self-contained PLA-vs-FLA prefill performance report.

The input directories are produced by ``run_generalized_perf_shards.sh``.
Both directions are required so every displayed timing is the geometric mean
of a baseline-first and candidate-first run on the same device assignment.
No workload is collapsed: repeated OperatorsForge records remain visible and
the report separately states the number of unique ``shape_id`` values.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


META_FIELDS = (
    "index", "source", "line", "shape_id", "H", "Hg", "N", "T",
    "max_seq", "min_seq", "imbalance", "physical_device", "device_name",
    "device_uuid", "device_pci", "compute_capability", "multi_processor_count",
)

EXPECTED_QWEN_FILES = {
    "Qwen3.7-Max_prefill_tp1_kvlimit_200GB_model_len_128K_gdn_op.list": {
        "tp": "tp1", "H": 128, "Hg": 16, "start": 0,
        "sha256": "361de7b21005125c73aaa1972f6054dc72b5cd8ebb5c99f54409ad1ed323326f",
    },
    "Qwen3.7-Max_prefill_tp2_kvlimit_200GB_model_len_128K_gdn_op.list": {
        "tp": "tp2", "H": 64, "Hg": 8, "start": 1344,
        "sha256": "59bf67ea64b927080052ab5e02d9ae4d18c4520ab49c3f98b248c44e3fd3d4e9",
    },
    "Qwen3.7-Max_prefill_tp4_kvlimit_200GB_model_len_128K_gdn_op.list": {
        "tp": "tp4", "H": 32, "Hg": 4, "start": 2688,
        "sha256": "98c1541d9739f3176def16c6343dc4b582ff4ab1e4051241ec5f1e15bfad878b",
    },
    "Qwen3.7-Max_prefill_tp8_kvlimit_200GB_model_len_128K_gdn_op.list": {
        "tp": "tp8", "H": 16, "Hg": 2, "start": 4032,
        "sha256": "acfb86d477569ee3f5b4e12209110a4fba621319b9c27b1931f6195a1a5e7c2d",
    },
}


def _manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate((path / "manifest.txt").read_text().splitlines(), 1):
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"malformed manifest line {path}:{line_number}")
        key, value = raw.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate/empty manifest key {key!r} in {path}")
        result[key] = value
    if result.get("exit_status") != "0":
        raise ValueError(f"incomplete run manifest: {path}")
    return result


def _rows(path: Path, label: str, shard_count: int) -> dict[int, dict[str, Any]]:
    files = sorted(path.glob(f"{label}-shard*.jsonl"))
    file_shards: dict[int, Path] = {}
    for item in files:
        match = re.fullmatch(rf"{re.escape(label)}-shard(\d+)\.jsonl", item.name)
        if not match:
            raise ValueError(f"malformed shard filename: {item}")
        shard = int(match.group(1))
        if shard in file_shards:
            raise ValueError(f"duplicate shard {shard} for {label} in {path}")
        file_shards[shard] = item
    if set(file_shards) != set(range(shard_count)):
        raise ValueError(
            f"{label} shard set mismatch in {path}: "
            f"{sorted(file_shards)} != {list(range(shard_count))}"
        )
    rows: dict[int, dict[str, Any]] = {}
    for shard, item in sorted(file_shards.items()):
        for line_number, raw in enumerate(item.read_text().splitlines(), 1):
            row = json.loads(raw)
            index = int(row["index"])
            if index in rows:
                raise ValueError(f"duplicate index {index} in {item}:{line_number}")
            if not row.get("pass"):
                raise ValueError(f"failed benchmark row {index}: {row.get('error')}")
            if row.get("mode") != "perf" or row.get("perf_timing") != "e2e":
                raise ValueError(
                    f"row {index} is not an E2E perf sample: "
                    f"mode={row.get('mode')!r}, timing={row.get('perf_timing')!r}"
                )
            if row.get("family") != "gdn":
                raise ValueError(f"row {index} is not GDN: {row.get('family')!r}")
            if row.get("shard") != shard or index % shard_count != shard:
                raise ValueError(
                    f"row {index} has wrong shard in {item}: "
                    f"{row.get('shard')!r}, expected {shard}"
                )
            ms = float(row["ms"])
            if not math.isfinite(ms) or ms <= 0:
                raise ValueError(f"invalid timing for index {index}: {ms}")
            rows[index] = row
    return rows


def _assert_same_metadata(rows: list[dict[str, Any]]) -> None:
    first = rows[0]
    for row in rows[1:]:
        for field in META_FIELDS:
            if row.get(field) != first.get(field):
                raise ValueError(
                    f"metadata mismatch at index {first['index']} for {field}: "
                    f"{first.get(field)!r} != {row.get(field)!r}"
                )


def _validate_qwen_record(row: dict[str, Any]) -> None:
    index = row.get("index")
    source = row.get("source")
    if type(index) is not int or not 0 <= index < 5376:
        raise ValueError(f"invalid canonical index: {index!r}")
    if source not in EXPECTED_QWEN_FILES:
        raise ValueError(f"unexpected Qwen3.7 source for index {index}: {source!r}")
    expected = EXPECTED_QWEN_FILES[source]
    if not expected["start"] <= index < expected["start"] + 1344:
        raise ValueError(f"index {index} is outside the canonical block for {source}")
    for field in ("line", "H", "Hg", "N", "T", "max_seq", "min_seq"):
        if type(row.get(field)) is not int or row[field] <= 0:
            raise ValueError(f"invalid {field} for index {index}: {row.get(field)!r}")
    if (row["H"], row["Hg"]) != (expected["H"], expected["Hg"]):
        raise ValueError(
            f"head geometry mismatch for index {index}: "
            f"{row['H']}/{row['Hg']} != {expected['H']}/{expected['Hg']}"
        )
    if row["H"] % row["Hg"] or not row["min_seq"] <= row["max_seq"] <= row["T"]:
        raise ValueError(f"invalid shape metadata for index {index}")
    imbalance = row.get("imbalance")
    expected_imbalance = row["max_seq"] / (row["T"] / row["N"])
    if (
        not isinstance(imbalance, (int, float))
        or not math.isfinite(float(imbalance))
        or not math.isclose(float(imbalance), expected_imbalance, rel_tol=1e-12)
    ):
        raise ValueError(f"invalid imbalance for index {index}: {imbalance!r}")
    shape_id = row.get("shape_id")
    if not isinstance(shape_id, str) or not re.fullmatch(r"[0-9a-f]{16}", shape_id):
        raise ValueError(f"invalid shape_id for index {index}: {shape_id!r}")
    if type(row.get("physical_device")) is not int or row["physical_device"] not in range(8):
        raise ValueError(f"invalid physical device for index {index}")
    if row.get("device_name") != "ZW-M890P" or row.get("compute_capability") != "8.9":
        raise ValueError(f"benchmark row {index} was not produced on PPU1.5")
    if row.get("multi_processor_count") != 39:
        raise ValueError(f"unexpected CU count for index {index}")
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        str(row.get("device_uuid", "")),
    ):
        raise ValueError(f"invalid device UUID for index {index}")
    if not re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.0", str(row.get("device_pci", ""))):
        raise ValueError(f"invalid PCI identity for index {index}")


def _validate_list_provenance(provenance: dict[str, Any]) -> None:
    supplied = provenance.get("qwen37_list_files")
    if not isinstance(supplied, list):
        raise ValueError("provenance must contain qwen37_list_files")
    normalized = {}
    for item in supplied:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("malformed qwen37_list_files provenance")
        name = item["name"]
        if name in normalized:
            raise ValueError(f"duplicate list provenance for {name}")
        normalized[name] = item
    if set(normalized) != set(EXPECTED_QWEN_FILES):
        raise ValueError("Qwen3.7 list provenance does not name the canonical four files")
    for name, expected in EXPECTED_QWEN_FILES.items():
        item = normalized[name]
        if item.get("sha256") != expected["sha256"] or item.get("records") != 1344:
            raise ValueError(f"Qwen3.7 list provenance mismatch for {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_provenance_binding(
    provenance_path: Path, manifests: dict[str, dict[str, str]],
) -> None:
    actual_sha256 = _sha256(provenance_path)
    for name, manifest in manifests.items():
        recorded_sha256 = manifest.get("provenance_sha256")
        if recorded_sha256 != actual_sha256:
            raise ValueError(
                f"{name} provenance SHA mismatch: manifest records "
                f"{recorded_sha256!r}, but {provenance_path} is {actual_sha256}"
            )


def _artifact_inventory(path: Path) -> list[dict[str, Any]]:
    artifacts = []
    for item in sorted(path.iterdir()):
        if not item.is_file() or not (
            item.name == "manifest.txt" or item.suffix in (".jsonl", ".log")
        ):
            continue
        entry: dict[str, Any] = {
            "name": item.name,
            "bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        if item.suffix == ".jsonl":
            entry["rows"] = sum(1 for raw in item.read_text().splitlines() if raw)
        artifacts.append(entry)
    return artifacts


def _tp(source: str) -> str:
    match = re.search(r"_prefill_(tp\d+)_", source)
    return match.group(1) if match else "unknown"


def _winner(pla_ms: float, fla_ms: float) -> str:
    ratio = fla_ms / pla_ms
    if ratio > 1.03 and fla_ms - pla_ms > 0.003:
        return "PLA"
    if ratio < 0.97 and pla_ms - fla_ms > 0.003:
        return "FLA"
    return "Tie"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speedups = [float(row["speedup"]) for row in rows]
    pla_total = sum(float(row["pla_ms"]) for row in rows)
    fla_total = sum(float(row["fla_ms"]) for row in rows)
    return {
        "records": len(rows),
        "unique_shapes": len({row["shape_id"] for row in rows}),
        "aggregate": fla_total / pla_total,
        "geomean": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
        "median": statistics.median(speedups),
        "p05": _percentile(speedups, 0.05),
        "p95": _percentile(speedups, 0.95),
        "pla_wins": sum(row["winner"] == "PLA" for row in rows),
        "ties": sum(row["winner"] == "Tie" for row in rows),
        "fla_wins": sum(row["winner"] == "FLA" for row in rows),
    }


def build_rows(
    bf_dir: Path, cf_dir: Path, expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    manifests = {"baseline_first": _manifest(bf_dir), "candidate_first": _manifest(cf_dir)}
    for name, manifest in manifests.items():
        if manifest.get("baseline_impl") != "fla" or manifest.get("candidate_impl") != "pla":
            raise ValueError(f"{name} does not describe FLA baseline and PLA candidate")
        if manifest.get("family") != "gdn" or manifest.get("perf_timing") != "e2e":
            raise ValueError(f"{name} is not a GDN E2E run")
    if manifests["baseline_first"].get("order_mode") != "baseline-first":
        raise ValueError("baseline-first directory has the wrong order_mode")
    if manifests["candidate_first"].get("order_mode") != "candidate-first":
        raise ValueError("candidate-first directory has the wrong order_mode")
    for field in (
        "list_dir", "runner", "baseline_site", "candidate_site", "cards",
        "warmup", "repeat", "index_file", "family", "baseline_impl",
        "candidate_impl", "perf_timing", "provenance_json", "provenance_sha256",
    ):
        if manifests["baseline_first"].get(field) != manifests["candidate_first"].get(field):
            raise ValueError(f"BF/CF manifest mismatch for {field}")

    try:
        cards = [int(value) for value in manifests["baseline_first"]["cards"].split(",")]
    except (KeyError, ValueError) as error:
        raise ValueError("invalid card list in perf manifest") from error
    if not cards or len(cards) != len(set(cards)) or any(not 0 <= card < 8 for card in cards):
        raise ValueError(f"invalid/duplicate PPU card list: {cards}")

    bf_fla = _rows(bf_dir, "baseline", len(cards))
    bf_pla = _rows(bf_dir, "candidate", len(cards))
    cf_fla = _rows(cf_dir, "baseline", len(cards))
    cf_pla = _rows(cf_dir, "candidate", len(cards))
    keys = bf_fla.keys()
    if not (keys == bf_pla.keys() == cf_fla.keys() == cf_pla.keys()):
        raise ValueError("BF/CF record sets differ")
    if len(keys) != expected_count:
        raise ValueError(f"expected {expected_count} records, found {len(keys)}")

    result: list[dict[str, Any]] = []
    for index in sorted(keys):
        samples = [bf_fla[index], bf_pla[index], cf_fla[index], cf_pla[index]]
        _assert_same_metadata(samples)
        if expected_count == 5376:
            _validate_qwen_record(samples[0])
            expected_card = cards[index % len(cards)]
            if samples[0]["physical_device"] != expected_card:
                raise ValueError(
                    f"index {index} ran on card {samples[0]['physical_device']}, "
                    f"expected {expected_card}"
                )
        for row, implementation in (
            (bf_fla[index], "fla"), (bf_pla[index], "pla"),
            (cf_fla[index], "fla"), (cf_pla[index], "pla"),
        ):
            if row.get("implementation") != implementation:
                raise ValueError(
                    f"implementation mismatch at index {index}: "
                    f"{row.get('implementation')} != {implementation}"
                )
        pla_bf, pla_cf = float(bf_pla[index]["ms"]), float(cf_pla[index]["ms"])
        fla_bf, fla_cf = float(bf_fla[index]["ms"]), float(cf_fla[index]["ms"])
        pla_ms = math.sqrt(pla_bf * pla_cf)
        fla_ms = math.sqrt(fla_bf * fla_cf)
        base = {field: bf_pla[index][field] for field in META_FIELDS}
        base.update({
            "tp": _tp(str(base["source"])),
            "pla_bf_ms": pla_bf,
            "pla_cf_ms": pla_cf,
            "fla_bf_ms": fla_bf,
            "fla_cf_ms": fla_cf,
            "pla_ms": pla_ms,
            "fla_ms": fla_ms,
            "speedup": fla_ms / pla_ms,
            "winner": _winner(pla_ms, fla_ms),
        })
        result.append(base)
    if expected_count == 5376:
        expected_tp_counts = {"tp1": 1344, "tp2": 1344, "tp4": 1344, "tp8": 1344}
        actual_tp_counts = {
            tp: sum(row["tp"] == tp for row in result) for tp in expected_tp_counts
        }
        if actual_tp_counts != expected_tp_counts:
            raise ValueError(
                f"Qwen3.7 TP coverage mismatch: {actual_tp_counts} != "
                f"{expected_tp_counts}"
            )
        if set(keys) != set(range(5376)):
            raise ValueError("Qwen3.7 index coverage is not exactly 0..5375")
        if len({row["shape_id"] for row in result}) != 5184:
            raise ValueError("Qwen3.7 unique-shape coverage is not exactly 5,184")
    return result, manifests


def render(
    rows: list[dict[str, Any]], manifests: dict[str, dict[str, str]],
    provenance: dict[str, Any], output: Path,
) -> None:
    summaries = {"all": _summary(rows)}
    for tp in sorted({str(row["tp"]) for row in rows}):
        summaries[tp] = _summary([row for row in rows if row["tp"] == tp])

    payload = json.dumps(rows, separators=(",", ":")).replace("<", "\\u003c")
    summary_payload = json.dumps(summaries, separators=(",", ":")).replace("<", "\\u003c")
    provenance = dict(provenance)
    provenance["baseline_first_manifest"] = manifests["baseline_first"]
    provenance["candidate_first_manifest"] = manifests["candidate_first"]
    provenance_html = html.escape(json.dumps(provenance, indent=2, sort_keys=True))

    template = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Qwen3.7 Prefill Performance: PLA vs FLA-Triton</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box} body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa;color:#222}
h1{color:#1a1a2e;margin-bottom:5px} h2{color:#16213e;border-bottom:2px solid #0f3460;padding-bottom:8px;margin-top:36px}.subtitle{color:#666;margin-bottom:20px}
.card,.chart{background:#fff;border-radius:8px;padding:15px;margin-bottom:22px;box-shadow:0 2px 8px rgba(0,0,0,.1)}
.filters{display:flex;gap:18px;flex-wrap:wrap}.filters label{font-size:13px}.filters select{padding:5px 9px;border:1px solid #ccc;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:7px 9px;text-align:center}th{background:#0f3460;color:#fff;position:sticky;top:0}tr:nth-child(even){background:#f8f9fa}.pla{color:#2ca02c;font-weight:700}.fla{color:#d62728;font-weight:700}.tie{color:#666}
.scroll{max-height:680px;overflow:auto}.note{font-size:12px;color:#666}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.stat{background:#f7f9fc;border-radius:6px;padding:12px}.stat b{display:block;font-size:20px;color:#0f3460}pre{white-space:pre-wrap;word-break:break-word;background:#f5f5f5;padding:12px;border-radius:6px;font-size:11px}
</style></head><body>
<h1>Qwen3.7 Prefill Performance Report</h1>
<p class="subtitle">FLA-Triton E2E / PLA E2E; <b>ratio &gt; 1 = PLA faster</b>. Each timing is the geometric mean of baseline-first and candidate-first same-card runs.</p>
<div class="card filters"><label>TP <select id="tp"><option value="all">All TP</option></select></label><label>Batch <select id="batch"><option value="all">All batch sizes</option></select></label><label>Winner <select id="winner"><option value="all">All</option><option>PLA</option><option>Tie</option><option>FLA</option></select></label></div>
<div id="summary" class="card"></div>
<div id="ratioChart" class="chart"></div><div id="scatterChart" class="chart"></div><div id="boxChart" class="chart"></div>
<h2>Workload details</h2><p class="note">All records are embedded. The table renders at most 1,500 filtered rows for browser responsiveness.</p><div class="scroll"><table><thead><tr><th>Index</th><th>TP</th><th>N</th><th>T</th><th>Max seq</th><th>Imbalance</th><th>PLA ms</th><th>FLA ms</th><th>FLA/PLA</th><th>Winner</th></tr></thead><tbody id="details"></tbody></table></div>
<h2>Provenance</h2><pre>__PROVENANCE__</pre>
<script>
const DATA=__DATA__, SUMMARIES=__SUMMARIES__;
const tp=document.getElementById('tp'), batch=document.getElementById('batch'), winner=document.getElementById('winner');
[...new Set(DATA.map(x=>x.tp))].sort().forEach(x=>tp.add(new Option(x.toUpperCase(),x)));
[...new Set(DATA.map(x=>x.N))].sort((a,b)=>a-b).forEach(x=>batch.add(new Option(String(x),String(x))));
function selected(){return DATA.filter(x=>(tp.value==='all'||x.tp===tp.value)&&(batch.value==='all'||String(x.N)===batch.value)&&(winner.value==='all'||x.winner===winner.value));}
function calc(rows){const r=rows.map(x=>x.speedup).sort((a,b)=>a-b), sumP=rows.reduce((a,x)=>a+x.pla_ms,0),sumF=rows.reduce((a,x)=>a+x.fla_ms,0),q=p=>r[Math.round((r.length-1)*p)],n=r.length,median=n%2?r[(n-1)/2]:(r[n/2-1]+r[n/2])/2;return {records:rows.length,unique:new Set(rows.map(x=>x.shape_id)).size,aggregate:sumF/sumP,median:median,p05:q(.05),p95:q(.95),pla:rows.filter(x=>x.winner==='PLA').length,tie:rows.filter(x=>x.winner==='Tie').length,fla:rows.filter(x=>x.winner==='FLA').length};}
function render(){const rows=selected();if(rows.length===0){document.getElementById('summary').innerHTML='<p>No workloads match these filters.</p>';['ratioChart','scatterChart','boxChart'].forEach(id=>{if(typeof Plotly==='undefined')document.getElementById(id).innerHTML='';else Plotly.purge(id);});document.getElementById('details').innerHTML='';return;}const s=calc(rows);document.getElementById('summary').innerHTML=`<div class="stats"><div class="stat"><b>${s.records}</b>records (${s.unique} unique)</div><div class="stat"><b>${s.aggregate.toFixed(4)}x</b>aggregate speedup</div><div class="stat"><b>${s.median.toFixed(4)}x</b>median speedup</div><div class="stat"><b>${s.p05.toFixed(4)}–${s.p95.toFixed(4)}</b>p05–p95</div><div class="stat"><b class="pla">${s.pla}</b>PLA wins</div><div class="stat"><b>${s.tie}</b>ties</div><div class="stat"><b class="fla">${s.fla}</b>FLA wins</div></div><p class="note">Win threshold: relative 3% and absolute 0.003 ms; both conditions must hold.</p>`;
document.getElementById('details').innerHTML=rows.slice(0,1500).map(x=>`<tr><td>${x.index}</td><td>${x.tp.toUpperCase()}</td><td>${x.N}</td><td>${x.T}</td><td>${x.max_seq}</td><td>${x.imbalance.toFixed(3)}</td><td>${x.pla_ms.toFixed(6)}</td><td>${x.fla_ms.toFixed(6)}</td><td>${x.speedup.toFixed(4)}</td><td class="${x.winner.toLowerCase()}">${x.winner}</td></tr>`).join('');
if(typeof Plotly==='undefined'){['ratioChart','scatterChart','boxChart'].forEach(id=>document.getElementById(id).innerHTML='<p class="note">Charts require access to cdn.plot.ly; all summary and detail data remain available offline.</p>');return;}
const byTp=[...new Set(rows.map(x=>x.tp))].sort();const traces=byTp.map((name,i)=>{const a=rows.filter(x=>x.tp===name).sort((x,y)=>x.T-y.T);return{x:a.map(x=>x.T),y:a.map(x=>x.speedup),mode:'markers',name:name.toUpperCase(),text:a.map(x=>`N=${x.N}, max=${x.max_seq}<br>PLA=${x.pla_ms.toFixed(4)} ms<br>FLA=${x.fla_ms.toFixed(4)} ms`)}});Plotly.newPlot('ratioChart',traces,{title:'E2E speedup by total tokens',xaxis:{title:'Total tokens',type:'log'},yaxis:{title:'FLA / PLA'},shapes:[{type:'line',x0:1,x1:1e6,y0:1,y1:1,line:{dash:'dash',color:'#777'}}],height:480},{responsive:true});
Plotly.newPlot('scatterChart',[{x:rows.map(x=>x.pla_ms),y:rows.map(x=>x.fla_ms),mode:'markers',type:'scattergl',marker:{size:5,color:rows.map(x=>x.speedup),colorscale:'RdYlGn'},text:rows.map(x=>`${x.tp.toUpperCase()} N=${x.N} T=${x.T}`)}],{title:'PLA vs FLA-Triton E2E',xaxis:{title:'PLA ms',type:'log'},yaxis:{title:'FLA-Triton ms',type:'log'},height:500},{responsive:true});
Plotly.newPlot('boxChart',byTp.map(name=>({y:rows.filter(x=>x.tp===name).map(x=>x.speedup),name:name.toUpperCase(),type:'box',boxpoints:false})),{title:'Speedup distribution by TP',yaxis:{title:'FLA / PLA'},height:430},{responsive:true});
}
[tp,batch,winner].forEach(x=>x.addEventListener('change',render));render();
</script></body></html>'''
    document = template.replace("__DATA__", payload)
    document = document.replace("__SUMMARIES__", summary_payload)
    document = document.replace("__PROVENANCE__", provenance_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-first", type=Path, required=True)
    parser.add_argument("--candidate-first", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=5376)
    parser.add_argument("--provenance-json", type=Path)
    args = parser.parse_args()
    provenance = {}
    if args.provenance_json:
        provenance = json.loads(args.provenance_json.read_text())
    if args.expected_count == 5376:
        _validate_list_provenance(provenance)
    rows, manifests = build_rows(
        args.baseline_first, args.candidate_first, args.expected_count,
    )
    if args.provenance_json:
        _validate_provenance_binding(args.provenance_json, manifests)
    record_digest = hashlib.sha256()
    for row in rows:
        canonical = {field: row[field] for field in META_FIELDS}
        record_digest.update(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        )
        record_digest.update(b"\n")
    provenance = dict(provenance)
    provenance["record_set_sha256"] = record_digest.hexdigest()
    provenance["benchmark_artifacts"] = {
        "baseline_first": _artifact_inventory(args.baseline_first),
        "candidate_first": _artifact_inventory(args.candidate_first),
    }
    render(rows, manifests, provenance, args.output)
    summary = _summary(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
