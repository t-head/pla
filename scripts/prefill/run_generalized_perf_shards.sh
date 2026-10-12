#!/usr/bin/env bash
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

set -euo pipefail

# Optional rerun controls:
#   INDEX_FILE=/path/to/indices.txt
#   ORDER_MODE=alternating|baseline-first|candidate-first
#   FAMILY=gdn|kda|all
# Optional isolated ABI controls (defaults preserve the previous runner path):
#   BASELINE_GDN_PATH=legacy|split|packed
#   CANDIDATE_GDN_PATH=legacy|split|packed
#   BASELINE_KDA_TRUSTED_HOST_OFFSETS=0|1
#   CANDIDATE_KDA_TRUSTED_HOST_OFFSETS=0|1
#   BASELINE_KDA_INDEXED_STATE=0|1
#   CANDIDATE_KDA_INDEXED_STATE=0|1
#   BASELINE_KDA_CP_PLAN_CACHE_SIZE=0..16  (unset means inherit/default)
#   CANDIDATE_KDA_CP_PLAN_CACHE_SIZE=0..16 (unset means inherit/default)
#   EXPECTED_FILE_COUNT=<positive integer>
#   EXPECTED_COUNT=<positive integer>
#   RECORD_FIRST_CALL=0|1 (record the sequential pre-warmup call for every case)

if [[ $# -lt 6 || $# -gt 8 ]]; then
    echo "usage: $0 LIST_DIR RUNNER OUTPUT_DIR BASELINE_SITE CANDIDATE_SITE CARDS_CSV [WARMUP] [REPEAT]" >&2
    exit 2
fi

list_dir=$1
runner=$2
output_dir=$3
baseline_site=$4
candidate_site=$5
cards_csv=$6
warmup=${7:-2}
repeat=${8:-5}
order_mode=${ORDER_MODE:-alternating}
index_file=${INDEX_FILE:-}
family=${FAMILY:-all}
baseline_gdn_path=${BASELINE_GDN_PATH:-legacy}
candidate_gdn_path=${CANDIDATE_GDN_PATH:-legacy}
baseline_kda_trusted=${BASELINE_KDA_TRUSTED_HOST_OFFSETS:-0}
candidate_kda_trusted=${CANDIDATE_KDA_TRUSTED_HOST_OFFSETS:-0}
baseline_kda_indexed=${BASELINE_KDA_INDEXED_STATE:-0}
candidate_kda_indexed=${CANDIDATE_KDA_INDEXED_STATE:-0}
baseline_kda_cache_size=${BASELINE_KDA_CP_PLAN_CACHE_SIZE:-}
candidate_kda_cache_size=${CANDIDATE_KDA_CP_PLAN_CACHE_SIZE:-}
expected_file_count=${EXPECTED_FILE_COUNT:-}
expected_count=${EXPECTED_COUNT:-}
record_first_call=${RECORD_FIRST_CALL:-0}

if [[ ! "$warmup" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARMUP must be a positive integer for steady-state perf" >&2
    exit 2
fi
if [[ ! "$repeat" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEAT must be a positive integer" >&2
    exit 2
fi

case "$order_mode" in
    alternating|baseline-first|candidate-first) ;;
    *)
        echo "invalid ORDER_MODE: $order_mode" >&2
        exit 2
        ;;
esac
case "$family" in
    gdn|kda|all) ;;
    *)
        echo "invalid FAMILY: $family" >&2
        exit 2
        ;;
esac
for side_gdn_path in "$baseline_gdn_path" "$candidate_gdn_path"; do
    case "$side_gdn_path" in
        legacy|split|packed) ;;
        *)
            echo "invalid per-side GDN path: $side_gdn_path" >&2
            exit 2
            ;;
    esac
done
for side_flag in \
    "$baseline_kda_trusted" "$candidate_kda_trusted" \
    "$baseline_kda_indexed" "$candidate_kda_indexed" \
    "$record_first_call"; do
    case "$side_flag" in
        0|1) ;;
        *)
            echo "per-side KDA ABI flags must be 0 or 1: $side_flag" >&2
            exit 2
            ;;
    esac
done
for expected in "$expected_file_count" "$expected_count"; do
    if [[ -z "$expected" || ! "$expected" =~ ^[1-9][0-9]*$ ]]; then
        echo "EXPECTED_FILE_COUNT and EXPECTED_COUNT must be positive integers" >&2
        exit 2
    fi
done
for side_cache_size in "$baseline_kda_cache_size" "$candidate_kda_cache_size"; do
    if [[ -n "$side_cache_size" ]]; then
        if [[ ! "$side_cache_size" =~ ^[0-9]+$ ]] ||
           (( 10#$side_cache_size > 16 )); then
            echo "per-side KDA plan-cache size must be empty or 0..16: $side_cache_size" >&2
            exit 2
        fi
    fi
done

IFS=, read -r -a cards <<< "$cards_csv"
if [[ ${#cards[@]} -eq 0 ]]; then
    echo "no cards supplied" >&2
    exit 2
fi
for ((i = 0; i < ${#cards[@]}; ++i)); do
    if [[ ! "${cards[$i]}" =~ ^[0-9]+$ ]]; then
        echo "card ids must be non-negative integers: ${cards[$i]}" >&2
        exit 2
    fi
    for ((j = 0; j < i; ++j)); do
        if [[ "${cards[$i]}" == "${cards[$j]}" ]]; then
            echo "duplicate card id is not allowed: ${cards[$i]}" >&2
            exit 2
        fi
    done
done

for required in "$list_dir" "$runner" "$baseline_site" "$candidate_site"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required path: $required" >&2
        exit 2
    fi
done
if [[ ! -f "$list_dir/SHA256SUMS" ]]; then
    echo "missing required case-list manifest: $list_dir/SHA256SUMS" >&2
    exit 2
fi
if [[ -n "$index_file" && ! -f "$index_file" ]]; then
    echo "missing INDEX_FILE: $index_file" >&2
    exit 2
fi

if [[ -e "$output_dir" ]]; then
    echo "refusing to reuse output directory: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"

if ! (cd "$list_dir" && sha256sum -c SHA256SUMS) \
    > "$output_dir/list-verification.txt" 2>&1; then
    echo "case-list SHA256 verification failed" >&2
    exit 1
fi

{
    printf 'started=%s\n' "$(date --iso-8601=seconds)"
    printf 'list_dir=%s\n' "$list_dir"
    printf 'runner=%s\n' "$runner"
    printf 'baseline_site=%s\n' "$baseline_site"
    printf 'candidate_site=%s\n' "$candidate_site"
    printf 'cards=%s\n' "$cards_csv"
    printf 'warmup=%s\n' "$warmup"
    printf 'repeat=%s\n' "$repeat"
    printf 'order_mode=%s\n' "$order_mode"
    printf 'index_file=%s\n' "$index_file"
    printf 'family=%s\n' "$family"
    printf 'baseline_gdn_path=%s\n' "$baseline_gdn_path"
    printf 'candidate_gdn_path=%s\n' "$candidate_gdn_path"
    printf 'baseline_kda_trusted_host_offsets=%s\n' "$baseline_kda_trusted"
    printf 'candidate_kda_trusted_host_offsets=%s\n' "$candidate_kda_trusted"
    printf 'baseline_kda_indexed_state=%s\n' "$baseline_kda_indexed"
    printf 'candidate_kda_indexed_state=%s\n' "$candidate_kda_indexed"
    printf 'baseline_kda_cp_plan_cache_size=%s\n' "${baseline_kda_cache_size:-inherit}"
    printf 'candidate_kda_cp_plan_cache_size=%s\n' "${candidate_kda_cache_size:-inherit}"
    printf 'expected_file_count=%s\n' "$expected_file_count"
    printf 'expected_count=%s\n' "$expected_count"
    printf 'record_first_call=%s\n' "$record_first_call"
    printf 'list_manifest_sha256=%s\n' "$(sha256sum "$list_dir/SHA256SUMS" | awk '{print $1}')"
    printf 'runner_sha256=%s\n' "$(sha256sum "$runner" | awk '{print $1}')"
} > "$output_dir/manifest.txt"

run_one() {
    local label=$1
    local site=$2
    local card=$3
    local shard=$4
    local shards=$5
    local gdn_path
    local kda_trusted
    local kda_indexed
    local kda_cache_size

    case "$label" in
        baseline)
            gdn_path=$baseline_gdn_path
            kda_trusted=$baseline_kda_trusted
            kda_indexed=$baseline_kda_indexed
            kda_cache_size=$baseline_kda_cache_size
            ;;
        candidate)
            gdn_path=$candidate_gdn_path
            kda_trusted=$candidate_kda_trusted
            kda_indexed=$candidate_kda_indexed
            kda_cache_size=$candidate_kda_cache_size
            ;;
        *)
            echo "invalid run label: $label" >&2
            return 2
            ;;
    esac

    local -a selection_args=()
    if [[ -n "$index_file" ]]; then
        selection_args+=(--index-file "$index_file")
    fi
    local -a abi_args=(--gdn-path "$gdn_path")
    if [[ "$kda_trusted" == 1 ]]; then
        abi_args+=(--kda-trusted-host-offsets)
    fi
    if [[ "$kda_indexed" == 1 ]]; then
        abi_args+=(--kda-indexed-state)
    fi
    local -a inventory_args=(
        --expected-list-files "$expected_file_count"
        --expected-records "$expected_count"
    )
    inventory_args+=(
        --expect-list-manifest-sha256
        "$(sha256sum "$list_dir/SHA256SUMS" | awk '{print $1}')"
    )
    local -a timing_args=()
    if [[ "$record_first_call" == 1 ]]; then
        timing_args+=(--record-first-call)
    fi
    local -a side_env=()
    if [[ -n "$kda_cache_size" ]]; then
        side_env+=("PLA_FLASHKDA_CP_PLAN_CACHE_SIZE=$kda_cache_size")
    fi

    env \
        -u PLA_FLASHKDA_CP \
        -u PLA_FLASHKDA_CP_BV \
        -u PLA_FLASHKDA_CP_SEG_TILES \
        -u PLA_FLASHKDA_CP_MAX_H \
        -u PLA_FLASHKDA_CP_BALANCE \
        -u PLA_FLASHKDA_CP_MULT \
        -u PLA_FLASHKDA_CP_VERBOSE \
        -u PLA_FLASHKDA_CP_PLAN_CACHE_SIZE \
        -u PLA_FLASHKDA_K1_CPC \
        -u PLA_FLASHKDA_K1_THREADS \
        -u PLA_FLASHKDA_K2_DKLEADER \
        -u PLA_FLASHKDA_K2_DKSPLIT \
        -u PLA_FLASHKDA_K2_DVSPLIT \
        -u PLA_FLASHKDA_K2_NOWS \
        -u PPU_LIB_SHOW_PARAMS \
        -u FF_FORCE_1BLK \
        -u FF_FORCE_2BLK \
        -u FF_PAD_BYTES \
        "CUDA_VISIBLE_DEVICES=$card" \
        PYTHONNOUSERSITE=1 \
        "PYTHONPATH=$site${PYTHONPATH:+:$PYTHONPATH}" \
        "${side_env[@]}" \
        python "$runner" \
        --list-dir "$list_dir" \
        --family "$family" \
        --mode perf \
        --output "$output_dir/${label}-shard${shard}.jsonl" \
        --num-shards "$shards" \
        --shard-index "$shard" \
        --warmup "$warmup" \
        --repeat "$repeat" \
        --kda-exact-b1-cu \
        "${inventory_args[@]}" \
        "${abi_args[@]}" \
        "${timing_args[@]}" \
        "${selection_args[@]}" \
        > "$output_dir/${label}-shard${shard}.log" 2>&1
}

run_shard() {
    local shard=$1
    local card=$2
    local shards=$3

    if [[ "$order_mode" == baseline-first ]] ||
       [[ "$order_mode" == alternating && $((shard % 2)) -eq 0 ]]; then
        run_one baseline "$baseline_site" "$card" "$shard" "$shards"
        run_one candidate "$candidate_site" "$card" "$shard" "$shards"
    else
        run_one candidate "$candidate_site" "$card" "$shard" "$shards"
        run_one baseline "$baseline_site" "$card" "$shard" "$shards"
    fi
}

pids=()
for shard in "${!cards[@]}"; do
    run_shard "$shard" "${cards[$shard]}" "${#cards[@]}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

printf 'finished=%s\n' "$(date --iso-8601=seconds)" >> "$output_dir/manifest.txt"
printf 'exit_status=%s\n' "$failed" >> "$output_dir/manifest.txt"
exit "$failed"
