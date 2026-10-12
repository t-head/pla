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

# Run an auditable multi-device generalized-prefill accuracy sweep.
#
# Optional controls:
#   FAMILY=all|gdn|kda                  (default: all)
#   REFERENCE=fla|serial                (default: fla)
#   GDN_PATH=legacy|split|packed        (default: legacy)
#   KDA_EXACT_B1_CU=0|1                (default: 1)
#   KDA_TRUSTED_HOST_OFFSETS=0|1        (default: 0)
#   KDA_INDEXED_STATE=0|1               (default: 0)
#   EXPECTED_FILE_COUNT=<positive int>   (required)
#   EXPECTED_COUNT=<positive integer>   (required for strict summary)
#   EXPECTED_GDN_PLA_EXTENSION_SHA256=<64 lowercase hex> (required if GDN runs)
#   EXPECTED_KDA_PLA_EXTENSION_SHA256=<64 lowercase hex> (required if KDA runs)

if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo "usage: $0 LIST_DIR RUNNER SUMMARY OUTPUT_DIR SITE FLA_SOURCE CARDS_CSV [TOL]" >&2
    exit 2
fi

list_dir=$1
runner=$2
summary=$3
output_dir=$4
site=$5
fla_source=$6
cards_csv=$7
tol=${8:-0.02}

family=${FAMILY:-all}
reference=${REFERENCE:-fla}
gdn_path=${GDN_PATH:-legacy}
kda_exact_b1=${KDA_EXACT_B1_CU:-1}
kda_trusted=${KDA_TRUSTED_HOST_OFFSETS:-0}
kda_indexed=${KDA_INDEXED_STATE:-0}
expected_count=${EXPECTED_COUNT:-}
expected_file_count=${EXPECTED_FILE_COUNT:-}
expected_gdn_extension_sha256=${EXPECTED_GDN_PLA_EXTENSION_SHA256:-}
expected_kda_extension_sha256=${EXPECTED_KDA_PLA_EXTENSION_SHA256:-}

case "$family" in
    all|gdn|kda) ;;
    *) echo "invalid FAMILY: $family" >&2; exit 2 ;;
esac
case "$reference" in
    fla|serial) ;;
    *) echo "invalid REFERENCE: $reference" >&2; exit 2 ;;
esac
case "$gdn_path" in
    legacy|split|packed) ;;
    *) echo "invalid GDN_PATH: $gdn_path" >&2; exit 2 ;;
esac
for value in "$kda_exact_b1" "$kda_trusted" "$kda_indexed"; do
    case "$value" in
        0|1) ;;
        *) echo "KDA flags must be 0 or 1: $value" >&2; exit 2 ;;
    esac
done
if [[ -z "$expected_count" || ! "$expected_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_COUNT must be a positive integer" >&2
    exit 2
fi
if [[ -z "$expected_file_count" || ! "$expected_file_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXPECTED_FILE_COUNT must be a positive integer" >&2
    exit 2
fi
sha256_re='^[0-9a-f]{64}$'
if [[ "$family" != kda && ! "$expected_gdn_extension_sha256" =~ $sha256_re ]]; then
    echo "EXPECTED_GDN_PLA_EXTENSION_SHA256 must be a lowercase 64-character SHA256 when GDN runs" >&2
    exit 2
fi
if [[ "$family" != gdn && ! "$expected_kda_extension_sha256" =~ $sha256_re ]]; then
    echo "EXPECTED_KDA_PLA_EXTENSION_SHA256 must be a lowercase 64-character SHA256 when KDA runs" >&2
    exit 2
fi

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
for required in "$list_dir" "$runner" "$summary" "$site"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required path: $required" >&2
        exit 2
    fi
done
if [[ ! -f "$list_dir/SHA256SUMS" ]]; then
    echo "missing required case-list manifest: $list_dir/SHA256SUMS" >&2
    exit 2
fi
if [[ "$reference" == fla && ! -d "$fla_source" ]]; then
    echo "missing FLA source path: $fla_source" >&2
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
    printf 'summary=%s\n' "$summary"
    printf 'site=%s\n' "$site"
    printf 'fla_source=%s\n' "$fla_source"
    printf 'cards=%s\n' "$cards_csv"
    printf 'tol=%s\n' "$tol"
    printf 'family=%s\n' "$family"
    printf 'reference=%s\n' "$reference"
    printf 'gdn_path=%s\n' "$gdn_path"
    printf 'kda_exact_b1_cu=%s\n' "$kda_exact_b1"
    printf 'kda_trusted_host_offsets=%s\n' "$kda_trusted"
    printf 'kda_indexed_state=%s\n' "$kda_indexed"
    printf 'expected_count=%s\n' "$expected_count"
    printf 'expected_file_count=%s\n' "$expected_file_count"
    printf 'expected_gdn_pla_extension_sha256=%s\n' "${expected_gdn_extension_sha256:-not-selected}"
    printf 'expected_kda_pla_extension_sha256=%s\n' "${expected_kda_extension_sha256:-not-selected}"
    printf 'list_manifest_sha256=%s\n' "$(sha256sum "$list_dir/SHA256SUMS" | awk '{print $1}')"
    printf 'runner_sha256=%s\n' "$(sha256sum "$runner" | awk '{print $1}')"
} > "$output_dir/manifest.txt"

run_shard() {
    local shard=$1
    local card=$2
    local shards=$3
    local -a abi_args=(--gdn-path "$gdn_path")
    local -a inventory_args=(
        --expected-list-files "$expected_file_count"
        --expected-records "$expected_count"
    )
    local -a python_path=("PYTHONPATH=$site${PYTHONPATH:+:$PYTHONPATH}")
    local -a fla_env=()

    if [[ "$kda_exact_b1" == 1 ]]; then
        abi_args+=(--kda-exact-b1-cu)
    fi
    if [[ "$kda_trusted" == 1 ]]; then
        abi_args+=(--kda-trusted-host-offsets)
    fi
    if [[ "$kda_indexed" == 1 ]]; then
        abi_args+=(--kda-indexed-state)
    fi
    if [[ "$reference" == fla ]]; then
        python_path=("PYTHONPATH=$site:$fla_source${PYTHONPATH:+:$PYTHONPATH}")
        fla_env=("FLA_SOURCE_PATH=$fla_source")
    fi
    inventory_args+=(
        --expect-list-manifest-sha256
        "$(sha256sum "$list_dir/SHA256SUMS" | awk '{print $1}')"
    )

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
        "${python_path[@]}" \
        "${fla_env[@]}" \
        python "$runner" \
        --list-dir "$list_dir" \
        --family "$family" \
        --mode accuracy \
        --reference "$reference" \
        --tol "$tol" \
        --num-shards "$shards" \
        --shard-index "$shard" \
        --output "$output_dir/accuracy-shard${shard}.jsonl" \
        "${inventory_args[@]}" \
        "${abi_args[@]}" \
        > "$output_dir/accuracy-shard${shard}.log" 2>&1
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

summary_args=(
    --expected-count "$expected_count"
    --expect-reference "$reference"
    --expect-tol "$tol"
    --expect-inventory-record-count "$expected_count"
    --expect-inventory-file-count "$expected_file_count"
    --require-strong-metrics
)
summary_args+=(
    --expect-list-manifest-sha256
    "$(sha256sum "$list_dir/SHA256SUMS" | awk '{print $1}')"
)
if [[ "$family" != kda ]]; then
    summary_args+=(--expect-gdn-path "$gdn_path")
    summary_args+=(--expect-gdn-state-abi "$([[ "$gdn_path" == legacy ]] && printf legacy || printf indexed_pool)")
    summary_args+=(--expect-gdn-gating-abi "$([[ "$gdn_path" == legacy ]] && printf activated || printf raw)")
    summary_args+=(--expect-gdn-pla-extension-sha256 "$expected_gdn_extension_sha256")
fi
if [[ "$family" != gdn ]]; then
    summary_args+=(--expect-kda-trusted-host-offsets "$([[ "$kda_trusted" == 1 ]] && printf true || printf false)")
    summary_args+=(--expect-kda-state-abi "$([[ "$kda_indexed" == 1 ]] && printf indexed_fp32_pool || printf compact_bf16_final_state)")
    summary_args+=(--expect-kda-pla-extension-sha256 "$expected_kda_extension_sha256")
fi

if [[ "$failed" == 0 ]]; then
    if ! python "$summary" \
        "${summary_args[@]}" \
        "$output_dir"/accuracy-shard*.jsonl \
        > "$output_dir/summary.txt" 2>&1; then
        failed=1
    fi
fi

printf 'finished=%s\n' "$(date --iso-8601=seconds)" >> "$output_dir/manifest.txt"
printf 'exit_status=%s\n' "$failed" >> "$output_dir/manifest.txt"
exit "$failed"
