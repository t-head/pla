#!/bin/bash
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

# Launch perf_compare.py across selected idle GPUs, then merge CSVs.
#
# On PPU hosts, CUDA_VISIBLE_DEVICES is mandatory: ppu-smi output is not parsed
# here, so the script fails closed instead of guessing that devices 0-7 are idle.
# Usage:
#   bash scripts/decode/perf_compare_parallel.sh [max-mem-gb] [gpu-util-threshold]
#
# Examples:
#   bash scripts/decode/perf_compare_parallel.sh              # default: 8GB, util<30%
#   bash scripts/decode/perf_compare_parallel.sh 8 50         # util threshold 50%
#   CUDA_VISIBLE_DEVICES=0,2,5 bash scripts/decode/perf_compare_parallel.sh  # force specific GPUs

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# scripts/decode/ -> repo root
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"
# The per-case runner is executed by absolute script path, so Python otherwise
# puts scripts/decode (not the repository root) on sys.path.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MAX_MEM_GB="${1:-8}"
UTIL_THRESHOLD="${2:-30}"   # GPU utilization % below which considered idle
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
REPORT_DIR="${REPORT_DIR:-reports}"
# Publishes the 890P report by default; for an 810E run set REPORT_HTML=perf_report_810e.html.
REPORT_HTML="${REPORT_HTML:-perf_report_890p.html}"
case "$REPORT_HTML" in
    ''|*/*|.|..) echo "ERROR: REPORT_HTML must be a file name." >&2; exit 1 ;;
esac
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
# 2026-09-15: the non-power-of-two fix passed 890P/810E correctness, five A/B rounds and a separate Kimi re-test.
# The lists are checked by non-empty, non-comment lines: 31 GDN/KDA files, 6931 raw records.
# The fixed pla.decode must be built/installed before running; old wheels are still forbidden from running Kimi.
# If memory filtering drops rows, publishing the full report is still forbidden; the completeness assertion below is not relaxed.
EXPECTED_CASELIST_COUNT=31
EXPECTED_RAW_CASE_COUNT=6931
RUN_PROVENANCE_JSON="${RUN_PROVENANCE_JSON:-}"

if [ -z "$PYTHON" ]; then
    echo "ERROR: python3/python not found; set PYTHON explicitly." >&2
    exit 1
fi
case "$WARMUP" in
    ''|*[!0-9]*) echo "ERROR: WARMUP must be a non-negative integer." >&2; exit 1 ;;
esac
case "$ITERS" in
    ''|*[!0-9]*) echo "ERROR: ITERS must be a positive integer." >&2; exit 1 ;;
esac
if [ "$ITERS" -eq 0 ]; then
    echo "ERROR: ITERS must be greater than zero." >&2
    exit 1
fi
case "$UTIL_THRESHOLD" in
    ''|*[!0-9]*) echo "ERROR: gpu-util-threshold must be an integer." >&2; exit 1 ;;
esac

# ============================================================================
# Auto-detect idle GPUs
# ============================================================================
detect_free_gpus() {
    local threshold=$1
    local free_gpus=()

    # If CUDA_VISIBLE_DEVICES is explicitly set, respect it
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        case "$CUDA_VISIBLE_DEVICES" in
            *[!0-9,]*|,*|*,|*,,*)
                echo "ERROR: CUDA_VISIBLE_DEVICES must be a comma-separated list of numeric device indices." >&2
                return 1
                ;;
        esac
        IFS=',' read -ra FORCED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
        local seen=","
        local gpu
        for gpu in "${FORCED_GPUS[@]}"; do
            case "$seen" in
                *",${gpu},"*)
                    echo "ERROR: duplicate device ${gpu} in CUDA_VISIBLE_DEVICES." >&2
                    return 1
                    ;;
            esac
            seen="${seen}${gpu},"
        done
        echo "${FORCED_GPUS[@]}"
        return
    fi

    if command -v ppu-smi >/dev/null 2>&1; then
        echo "ERROR: PPU host detected; set CUDA_VISIBLE_DEVICES explicitly after checking ppu-smi." >&2
        return 1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: neither explicit CUDA_VISIBLE_DEVICES nor a supported GPU status tool is available." >&2
        return 1
    fi

    # Query nvidia-smi for GPU utilization and memory usage
    # Format: index, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
    local smi_output
    if ! smi_output=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                          --format=csv,noheader,nounits 2>/dev/null); then
        echo "ERROR: nvidia-smi query failed; refusing to guess device availability." >&2
        return 1
    fi

    while IFS=', ' read -r idx util mem_used mem_total; do
        # Remove whitespace
        idx=$(echo "$idx" | tr -d ' ')
        util=$(echo "$util" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')
        mem_total=$(echo "$mem_total" | tr -d ' ')

        # Consider GPU idle if utilization < threshold AND memory usage < 10%
        local mem_pct=0
        if [ "$mem_total" -gt 0 ] 2>/dev/null; then
            mem_pct=$((mem_used * 100 / mem_total))
        fi

        if [ "$util" -lt "$threshold" ] 2>/dev/null && [ "$mem_pct" -lt 10 ]; then
            free_gpus+=("$idx")
        fi
    done <<< "$smi_output"

    if [ ${#free_gpus[@]} -eq 0 ]; then
        echo "ERROR: no GPU satisfies util<${threshold}% and memory<10%; refusing a relaxed fallback." >&2
        return 1
    fi

    echo "${free_gpus[@]}"
}

# Detect available GPUs without masking a detector failure behind read(1).
if ! FREE_GPU_OUTPUT="$(detect_free_gpus "$UTIL_THRESHOLD")"; then
    exit 1
fi
read -ra FREE_GPUS <<< "$FREE_GPU_OUTPUT"
NUM_GPUS=${#FREE_GPUS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: no available GPU selected." >&2
    exit 1
fi

# Collect the audited GDN/KDA decode inventory. Test lists and operators not
# implemented by this harness must never be parsed through its GDN defaults.
CASELISTS=()
for f in caselist/decode/*.list; do
    [ -s "$f" ] || continue
    case "$(basename "$f")" in
        test_*|*mamba2*|*lightning*)
            echo "  [SKIP unsupported op] $(basename "$f")"
            continue
            ;;
        *_gdn_op.list|*_kda_op.list)
            CASELISTS+=("$f")
            ;;
        *)
            echo "  [SKIP unaudited list] $(basename "$f")"
            ;;
    esac
done

if [ "${#CASELISTS[@]}" -ne "$EXPECTED_CASELIST_COUNT" ]; then
    echo "ERROR: expected ${EXPECTED_CASELIST_COUNT} audited GDN/KDA caselists, found ${#CASELISTS[@]}." >&2
    echo "Refusing to publish a partial or silently expanded report." >&2
    exit 1
fi

echo "=== Parallel perf_compare: ${#CASELISTS[@]} caselists across $NUM_GPUS idle GPUs ==="
echo "Idle GPUs detected: ${FREE_GPUS[*]} (util threshold: <${UTIL_THRESHOLD}%)"
echo "Max mem: ${MAX_MEM_GB} GB, warmup: $WARMUP, timed iterations: $ITERS"
echo ""

mkdir -p "$REPORT_DIR"
MANIFEST_PATH="$REPORT_DIR/run_manifest.json"
METHOD="minimum of ${ITERS} synchronized E2E samples (median retained in runner output but not CSV); ordinary gate (caselist lower-bound/ragged metadata is not replayed); k_last T1; k_last_mtp T4 with existing linear-parent harness; k_last_packed T1; PLA measured before FLA-Triton; runner timed region unchanged"
CARDS_CSV="$(IFS=,; echo "${FREE_GPUS[*]}")"

"$PYTHON" - "$MANIFEST_PATH" "$CARDS_CSV" "$MAX_MEM_GB" "$WARMUP" "$ITERS" "$METHOD" "$EXPECTED_RAW_CASE_COUNT" "$RUN_PROVENANCE_JSON" "${CASELISTS[@]}" <<'PY'
import datetime
import json
import sys

(
    manifest_path, cards_csv, max_mem_gb, warmup, iters, method,
    expected_raw_case_count, provenance_path,
) = sys.argv[1:9]
case_files = sys.argv[9:]


def raw_case_count(path):
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


files = [
    {"path": path, "raw_case_count": raw_case_count(path)}
    for path in case_files
]
raw_total = sum(item["raw_case_count"] for item in files)
expected_raw_case_count = int(expected_raw_case_count)
if raw_total != expected_raw_case_count:
    raise SystemExit(
        "audited raw case count changed: expected {}, found {}".format(
            expected_raw_case_count, raw_total
        )
    )
expected_rows = {
    "k_last": raw_total,
    "k_last_mtp": raw_total,
    "k_last_packed": raw_total,
}
manifest = {
    "schema_version": 1,
    "status": "running",
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "case_file_count": len(files),
    "raw_case_count": raw_total,
    "case_files": files,
    "expected_rows_by_kernel": expected_rows,
    "expected_total_rows": sum(expected_rows.values()),
    "cards": [int(card) for card in cards_csv.split(",")],
    "max_mem_gb": float(max_mem_gb),
    "warmup": int(warmup),
    "iters": int(iters),
    "mode": "events",
    "method": method,
    "filters": {
        "included": ["*_gdn_op.list", "*_kda_op.list"],
        "excluded": ["test_*", "*mamba2*", "*lightning*"],
        "restored_after_validation": ["Kimi-K3_*"],
    },
    "nonpow2_head_validation": {
        "architectures": ["ZW-M890P", "PPU-ZW810E"],
        "evidence": "reports/nonpow2_20260915/acceptance.json",
        "requires_rebuilt_decode_extension": True,
    },
}
if provenance_path:
    with open(provenance_path, "r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not isinstance(provenance, dict):
        raise SystemExit("RUN_PROVENANCE_JSON must contain a JSON object")
    manifest["provenance"] = provenance
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

update_manifest_status() {
    local status=$1
    "$PYTHON" - "$MANIFEST_PATH" "$status" <<'PY'
import datetime
import json
import sys

path, status = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
manifest["status"] = status
manifest["finished_at"] = datetime.datetime.now().astimezone().isoformat()
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

update_manifest_results() {
    "$PYTHON" - "$MANIFEST_PATH" "$@" <<'PY'
import json
import sys

path = sys.argv[1]
actual = {}
for item in sys.argv[2:]:
    kernel, value = item.split("=", 1)
    actual[kernel] = int(value)

with open(path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
expected = manifest["expected_rows_by_kernel"]
if set(actual) != set(expected):
    raise SystemExit("result kernels do not match manifest expectation")
for kernel, rows in actual.items():
    if rows > expected[kernel]:
        raise SystemExit(
            "{} produced {} rows, more than expected {}".format(
                kernel, rows, expected[kernel]
            )
        )
skipped = {
    kernel: expected[kernel] - actual[kernel]
    for kernel in sorted(expected)
}
manifest["result_rows_by_kernel"] = actual
manifest["result_total_rows"] = sum(actual.values())
manifest["skipped_rows_by_kernel"] = skipped
manifest["skipped_total_rows"] = sum(skipped.values())
with open(path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

FINAL_ARTIFACTS=(
    "$REPORT_DIR/perf_k_last.csv"
    "$REPORT_DIR/perf_k_last_mtp.csv"
    "$REPORT_DIR/perf_k_last_packed.csv"
    "$REPORT_DIR/triton_wins.list"
    "$REPORT_DIR/$REPORT_HTML"
)
rm -f "${FINAL_ARTIFACTS[@]}"

PIDS=()
RUN_COMPLETE=0
terminate_process_tree() {
    local parent_pid=$1
    local child_pid
    if command -v pgrep >/dev/null 2>&1; then
        for child_pid in $(pgrep -P "$parent_pid" 2>/dev/null || true); do
            terminate_process_tree "$child_pid"
        done
    fi
    kill -TERM "$parent_pid" 2>/dev/null || true
}

cleanup_on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    if [ "$RUN_COMPLETE" -ne 1 ]; then
        [ "$rc" -ne 0 ] || rc=1
        for pid in "${PIDS[@]:-}"; do
            [ -n "$pid" ] || continue
            terminate_process_tree "$pid"
        done
        for pid in "${PIDS[@]:-}"; do
            [ -n "$pid" ] || continue
            wait "$pid" 2>/dev/null || true
        done
        rm -f "${FINAL_ARTIFACTS[@]}"
        update_manifest_status "failed" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Create per-GPU output dirs and clean old CSVs
for i in $(seq 0 $((NUM_GPUS-1))); do
    mkdir -p "$REPORT_DIR/gpu${FREE_GPUS[$i]}"
    rm -f "$REPORT_DIR/gpu${FREE_GPUS[$i]}/perf_"*.csv
    rm -f "$REPORT_DIR/gpu${FREE_GPUS[$i]}/triton_wins.list"
done

# Assign caselists round-robin to available GPUs
declare -a GPU_FILES
for i in $(seq 0 $((NUM_GPUS-1))); do
    GPU_FILES[$i]=""
done

idx=0
for f in "${CASELISTS[@]}"; do
    slot=$((idx % NUM_GPUS))
    GPU_FILES[$slot]="${GPU_FILES[$slot]} $f"
    idx=$((idx + 1))
done

# Print assignments
for i in $(seq 0 $((NUM_GPUS-1))); do
    g=${FREE_GPUS[$i]}
    count=$(echo ${GPU_FILES[$i]} | wc -w)
    echo "  GPU $g: $count files"
    for f in ${GPU_FILES[$i]}; do
        echo "    - $(basename $f)"
    done
done
echo ""

# Launch one background process per GPU; each loops over its assigned caselists.
# A failing worker signals the parent so no merge/HTML can be published.
PARENT_PID=$$
GPU_INDICES_USED=()
for i in $(seq 0 $((NUM_GPUS-1))); do
    files="${GPU_FILES[$i]}"
    [ -z "$files" ] && continue

    g=${FREE_GPUS[$i]}
    outdir="$REPORT_DIR/gpu$g"
    logfile="$outdir/run.log"

    (
        set -e
        for casefile in $files; do
            echo "[GPU $g] Starting $(basename $casefile) ..."
            if CUDA_VISIBLE_DEVICES=$g "$PYTHON" scripts/decode/perf_compare.py \
                    --caselist "$casefile" \
                    --output-dir "$outdir" \
                    --warmup "$WARMUP" \
                    --iters "$ITERS" \
                    --mtp-tokens 4 \
                    --mode events \
                    --max-mem-gb "$MAX_MEM_GB" \
                    --stop-on-fail \
                    2>&1; then
                :
            else
                rc=$?
                echo "[GPU $g] ERROR: $(basename "$casefile") failed with exit code $rc"
                kill -TERM "$PARENT_PID" 2>/dev/null || true
                exit "$rc"
            fi
        done
        echo "[GPU $g] Done."
    ) > "$logfile" 2>&1 &

    PIDS+=($!)
    GPU_INDICES_USED+=($g)
    echo "Launched GPU $g (PID $!) with $(echo $files | wc -w) caselists"
done

echo ""
echo "All ${#PIDS[@]} GPU workers launched. PIDs: ${PIDS[*]}"
echo "Logs: $REPORT_DIR/gpu*/run.log"
echo ""

# Wait for all
echo "Waiting for all GPUs to finish..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL+1))
done

if [ "$FAIL" -gt 0 ]; then
    echo "ERROR: $FAIL GPU worker(s) failed; final report will not be generated." >&2
    exit 1
fi

# Merge per-GPU CSVs into single files
echo ""
echo "Merging CSVs..."
MERGED_ROW_COUNTS=()
for kernel in k_last k_last_mtp k_last_packed; do
    merged="$REPORT_DIR/perf_${kernel}.csv"
    header_written=false
    for g in "${GPU_INDICES_USED[@]}"; do
        csv="$REPORT_DIR/gpu$g/perf_${kernel}.csv"
        [ -f "$csv" ] || continue
        if [ "$header_written" = false ]; then
            cat "$csv" > "$merged"
            header_written=true
        else
            tail -n +2 "$csv" >> "$merged"
        fi
    done
    if [ "$header_written" = true ]; then
        rows=$(($(wc -l < "$merged") - 1))
        if [ "$rows" -le 0 ]; then
            echo "ERROR: $merged has no result rows." >&2
            exit 1
        fi
        if [ "$rows" -gt "$EXPECTED_RAW_CASE_COUNT" ]; then
            echo "ERROR: $merged has $rows rows, above expected $EXPECTED_RAW_CASE_COUNT." >&2
            exit 1
        fi
        skipped=$((EXPECTED_RAW_CASE_COUNT - rows))
        MERGED_ROW_COUNTS+=("${kernel}=${rows}")
        echo "  $merged ($rows rows, $skipped memory-filtered/missing rows)"
    else
        echo "ERROR: no worker produced perf_${kernel}.csv." >&2
        exit 1
    fi
done
update_manifest_results "${MERGED_ROW_COUNTS[@]}"
for item in "${MERGED_ROW_COUNTS[@]}"; do
    rows=${item#*=}
    if [ "$rows" -ne "$EXPECTED_RAW_CASE_COUNT" ]; then
        echo "ERROR: full decode report requires exactly ${EXPECTED_RAW_CASE_COUNT} rows per kernel; ${item%%=*} produced $rows." >&2
        exit 1
    fi
done

# Merge triton_wins.list
merged_wins="$REPORT_DIR/triton_wins.list"
echo "# Cases where Triton kernel has better performance than CUDA" > "$merged_wins"
echo "# Generated by perf_compare_parallel.sh on $(date '+%Y-%m-%d %H:%M:%S')" >> "$merged_wins"
win_count=0
for g in "${GPU_INDICES_USED[@]}"; do
    wf="$REPORT_DIR/gpu$g/triton_wins.list"
    [ -f "$wf" ] || continue
    while IFS= read -r line; do
        [[ "$line" =~ ^#.*$ ]] && continue
        [ -z "$line" ] && continue
        echo "$line" >> "$merged_wins"
        win_count=$((win_count+1))
    done < "$wf"
done
echo "  $merged_wins ($win_count Triton-wins cases)"

# Generate HTML report with embedded data
echo ""
echo "Generating HTML report..."
update_manifest_status "completed"
"$PYTHON" gen_report.py --report-dir "$REPORT_DIR" --output "$REPORT_DIR/$REPORT_HTML"
RUN_COMPLETE=1

echo ""
echo "=== Done ==="
echo "Report: $REPORT_DIR/$REPORT_HTML"
echo "Manifest: $MANIFEST_PATH"
echo "Open in browser: file://$(cd "$REPORT_DIR" && pwd)/$REPORT_HTML"
