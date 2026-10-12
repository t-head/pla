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

"""Generate per-instantiation .cu files for k_last and k_last_packed kernels.

Modes:
  (default/full): generate ALL valid combos
  --default-only: generate only BV=64,WN=8,SN=4,HV_PER_CTA=1 (fastest build)
  --tuned-csv FILE: generate only combos found in CSV + default combo
"""

import argparse
import csv
import itertools
import os
from collections import defaultdict

GENERATED_BSD3_HEADER = """\
/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
GENERATED_DIR = os.path.join(PROJECT_DIR, "src", "decode", "k-last", "generated")

BV_VALUES = [16, 32, 64, 128]
WARP_NUM_VALUES = [4, 8]
STAGE_NUM_VALUES = [4, 5, 6, 8]
HV_PER_CTA_VALUES = [1]

DEFAULT_COMBO = (64, 8, 4, 1)

SCALAR_TYPES = [("__half", "Half", "fp16"), ("__ppu_bfloat16", "BFloat16", "bf16")]
ALOG_TYPES = [("float", "Float", "f32"), ("__half", "Half", "fp16"), ("__ppu_bfloat16", "BFloat16", "bf16")]
STATE_TYPES = [("float", "f32"), ("__ppu_bfloat16", "bf16")]


def get_valid_combinations():
    combos = []
    for bv, wn, sn, hv in itertools.product(
        BV_VALUES, WARP_NUM_VALUES, STAGE_NUM_VALUES, HV_PER_CTA_VALUES
    ):
        if sn <= bv // wn:
            combos.append((bv, wn, sn, hv))
    return combos


def _extract_device_name(csv_path):
    """Extract (device_name, mode) from a filename like tuned_result_ZW-M890L.csv.

    A trailing _chain/_tree suffix marks a mode-split table for the same device
    (tuned_result_PPU-ZW810E_chain.csv -> ("PPU-ZW810E", "chain")).
    """
    base = os.path.splitext(os.path.basename(csv_path))[0]
    prefix = "tuned_result_"
    if base.startswith(prefix) and len(base) > len(prefix):
        base = base[len(prefix):]
    mode = None
    for suffix in ("_chain", "_tree"):
        if base.endswith(suffix):
            mode = suffix[1:]
            base = base[:-len(suffix)]
            break
    return base, mode


def _load_single_csv(csv_path, device_name, mode=None):
    combos = set()
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = (int(row["BV"]), int(row["WARP_NUM"]), int(row["STAGE_NUM"]), int(row["HV_PER_CTA"]))
            combos.add(c)
            records.append({
                "device_name": row.get("device_name", device_name),
                "kernel": row["kernel"], "B": int(row["B"]),
                "H": int(row["H"]), "HV": int(row["HV"]),
                "K": int(row["K"]), "V": int(row["V"]),
                "state_dtype": row.get("state_dtype", "fp32"),
                "mode": row.get("mode") or mode,
                "BV": c[0], "WARP_NUM": c[1], "STAGE_NUM": c[2], "HV_PER_CTA": c[3],
            })
    return combos, records


def load_tuned_csv(csv_path):
    """Load tuned CSV(s). csv_path can be a single file or a glob pattern."""
    import glob as globmod
    paths = sorted(globmod.glob(csv_path))
    if not paths:
        raise FileNotFoundError(f"No files matching: {csv_path}")
    all_combos = set()
    all_records = []
    for p in paths:
        device_name, mode = _extract_device_name(p)
        combos, records = _load_single_csv(p, device_name, mode)
        all_combos.update(combos)
        all_records.extend(records)
    return all_combos, all_records


# ---------------------------------------------------------------------------
# Wrapper naming
# ---------------------------------------------------------------------------

def wrapper_name_k_last(bv, wn, sn, hv, alog_short, scalar_short, mtp, state_short="f32"):
    mtp_str = "mtp" if mtp else "dec"
    suffix = f"_state{state_short}" if state_short != "f32" else ""
    return f"launch_k_last_bv{bv}_wn{wn}_sn{sn}_hv{hv}_{alog_short}_{scalar_short}_{mtp_str}{suffix}"


def wrapper_name_k_last_packed(bv, wn, sn, hv, alog_short, scalar_short, state_short="f32"):
    suffix = f"_state{state_short}" if state_short != "f32" else ""
    return f"launch_k_last_packed_bv{bv}_wn{wn}_sn{sn}_hv{hv}_{alog_short}_{scalar_short}{suffix}"


# ---------------------------------------------------------------------------
# Param signatures
# ---------------------------------------------------------------------------

K_LAST_WRAPPER_PARAMS = """\
    const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream"""

K_LAST_PACKED_WRAPPER_PARAMS = """\
    const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream"""


# ---------------------------------------------------------------------------
# .cu file generators
# ---------------------------------------------------------------------------

def generate_k_last_cu(bv, wn, sn, hv):
    return GENERATED_BSD3_HEADER + "\n".join(
        _generate_k_last_state_cu(bv, wn, sn, hv, state_t, state_short)
        for state_t, state_short in STATE_TYPES
    )


def _generate_k_last_state_cu(bv, wn, sn, hv, state_t, state_short):
    lines = ['#include "../fused_sigmoid_gating_k_last_kernel.cuh"', '',
             f'// Wrappers: BV={bv}, WARP_NUM={wn}, STAGE_NUM={sn}, HV_PER_CTA={hv}', '']
    for alog_t, _, alog_short in ALOG_TYPES:
        for scalar_t, _, scalar_short in SCALAR_TYPES:
            for mtp in [False, True]:
                mtp_str = "true" if mtp else "false"
                fname = wrapper_name_k_last(bv, wn, sn, hv, alog_short, scalar_short, mtp, state_short)
                targs_gdn = f"{alog_t}, {scalar_t}, {wn}, {bv}, {hv}, {sn}, false, false, {mtp_str}, true, {state_t}"
                targs_kda = f"{alog_t}, {scalar_t}, {wn}, {bv}, {hv}, {sn}, true, false, {mtp_str}, true, {state_t}"
                targs_kda_safe = f"{alog_t}, {scalar_t}, {wn}, {bv}, {hv}, {sn}, true, true, {mtp_str}, true, {state_t}"
                # The kernel derives dt_bias's type from IS_KDA && MTP_VERIFY: fp32 for
                # KDA MTP verify, the activation dtype everywhere else (see dt_bias_t
                # in the kernel header).
                dtb_kda = "float" if mtp else scalar_t
                def kernel_args(dtb_t):
                    return [
                        f"        reinterpret_cast<const {alog_t}*>(A_log),",
                        f"        reinterpret_cast<const {scalar_t}*>(a),",
                        f"        reinterpret_cast<const {dtb_t}*>(dt_bias),",
                        f"        softplus_beta_or_lower_bound, softplus_threshold,",
                        f"        reinterpret_cast<const {scalar_t}*>(q),",
                        f"        reinterpret_cast<const {scalar_t}*>(k),",
                        f"        reinterpret_cast<const {scalar_t}*>(v),",
                        f"        reinterpret_cast<const {scalar_t}*>(b),",
                        f"        reinterpret_cast<{scalar_t}*>(o),",
                        f"        reinterpret_cast<{state_t}*>(h0_source), cu_seqlens,",
                        f"        reinterpret_cast<{state_t}*>(intermediate_states_buffer),",
                        f"        h0_indices, intermediate_state_indices,",
                        f"        cache_steps, retrieve_parent_token_ptr,",
                        f"        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,",
                        f"        scale, T_max,",
                        f"        stride_a, stride_q, stride_k, stride_v, stride_b,",
                        f"        h0_slot_stride,",
                        f"        B, H, HV, K, V,",
                        f"        cache_intermediate_states, has_eagle_tree,",
                        f"        use_qk_l2norm, disable_state_update,",
                        f"        num_accepted_tokens, stride_h0_indices_seq, is_sglang",
                        f"        );",
                    ]
                lines += [
                    f"void {fname}(",
                    f"    {K_LAST_WRAPPER_PARAMS}",
                    ") {",
                    f"    constexpr int WARP_SIZE = 32;",
                    f"    auto block = dim3(WARP_SIZE, {wn});",
                    f"    if (is_kda && use_lower_bound) {{",
                    f"        fused_sigmoid_gating_delta_kernel<{targs_kda_safe}>",
                    f"        <<<grid, block, shared_mem_size, stream>>>(",
                ] + kernel_args(dtb_kda) + [
                    f"    }} else if (is_kda) {{",
                    f"        fused_sigmoid_gating_delta_kernel<{targs_kda}>",
                    f"        <<<grid, block, shared_mem_size, stream>>>(",
                ] + kernel_args(dtb_kda) + [
                    f"    }} else {{",
                    f"        fused_sigmoid_gating_delta_kernel<{targs_gdn}>",
                    f"        <<<grid, block, shared_mem_size, stream>>>(",
                ] + kernel_args(scalar_t) + [
                    f"    }}",
                    "}", "",
                ]
    return "\n".join(lines) + "\n"


def generate_k_last_packed_cu(bv, wn, sn, hv):
    lines = ['#include "../fused_sigmoid_gating_k_last_packed_kernel.cuh"', '',
             f'// Wrappers: BV={bv}, WARP_NUM={wn}, STAGE_NUM={sn}, HV_PER_CTA={hv}', '']
    for state_t, state_short in STATE_TYPES:
        for alog_t, _, alog_short in ALOG_TYPES:
            for scalar_t, _, scalar_short in SCALAR_TYPES:
                fname = wrapper_name_k_last_packed(bv, wn, sn, hv, alog_short, scalar_short, state_short)
                # Template args: alog_t, scalar_t, WARP_NUM, BV, HV_PER_CTA, STAGE_NUM, IS_KDA, use_initial_state, state_t
                targs_gdn = f"{alog_t}, {scalar_t}, {wn}, {bv}, {hv}, {sn}, false, true, {state_t}"
                targs_kda = f"{alog_t}, {scalar_t}, {wn}, {bv}, {hv}, {sn}, true, true, {state_t}"
                h0_cast = f"reinterpret_cast<{state_t}*>(h0_source)"
                packed_args = [
                    f"        reinterpret_cast<const {alog_t}*>(A_log),",
                    f"        reinterpret_cast<const {scalar_t}*>(a),",
                    f"        reinterpret_cast<const {scalar_t}*>(dt_bias),",
                    f"        softplus_beta, softplus_threshold,",
                    f"        reinterpret_cast<const {scalar_t}*>(mixed_qkv),",
                    f"        reinterpret_cast<const {scalar_t}*>(b),",
                    f"        reinterpret_cast<{scalar_t}*>(o),",
                    f"        {h0_cast}, cu_seqlens, h0_indices,",
                    f"        scale, T_max,",
                    f"        stride_a, stride_qkv, stride_b,",
                    f"        h0_slot_stride,",
                    f"        B, H, HV, K, V,",
                    f"        use_qk_l2norm, is_sglang",
                    f"    );",
                ]
                lines += [
                    f"void {fname}(",
                    f"    {K_LAST_PACKED_WRAPPER_PARAMS}",
                    ") {",
                    f"    constexpr int WARP_SIZE = 32;",
                    f"    auto block = dim3(WARP_SIZE, {wn});",
                    f"    if (is_kda) {{",
                    f"    fused_sigmoid_gating_delta_kernel_packed<{targs_kda}>",
                    f"        <<<grid, block, shared_mem_size, stream>>>(",
                ] + packed_args + [
                    f"    }} else {{",
                    f"    fused_sigmoid_gating_delta_kernel_packed<{targs_gdn}>",
                    f"        <<<grid, block, shared_mem_size, stream>>>(",
                ] + packed_args + [
                    f"    }}",
                    "}", "",
                ]
    return GENERATED_BSD3_HEADER + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Dispatch headers (wrapper declarations + runtime dispatcher)
# ---------------------------------------------------------------------------

def _dtype_dispatch_block(combos_with_names, is_packed):
    """Generate dtype if/else chain returning a launcher function pointer."""
    lines = []
    alog_enums = {"f32": "at::ScalarType::Float", "fp16": "at::ScalarType::Half", "bf16": "at::ScalarType::BFloat16"}
    scalar_enums = {"fp16": "at::ScalarType::Half", "bf16": "at::ScalarType::BFloat16"}
    first_dtype = True
    for (_, _, alog_short) in ALOG_TYPES:
        for (_, _, scalar_short) in SCALAR_TYPES:
            prefix = "if" if first_dtype else "} else if"
            first_dtype = False
            lines.append(f"        {prefix} (alog_type == {alog_enums[alog_short]} && scalar_type == {scalar_enums[scalar_short]}) {{")
            if is_packed:
                name = combos_with_names[(alog_short, scalar_short)]
                lines.append(f"            return {name};")
            else:
                dec_name = combos_with_names[(alog_short, scalar_short, False)]
                mtp_name = combos_with_names[(alog_short, scalar_short, True)]
                lines.append(f"            return mtp_verify ? {mtp_name} : {dec_name};")
    lines.append("        }")
    return lines


def generate_dispatch_header_k_last(combos):
    lines = ["#pragma once", "", "#include <hggc_runtime.h>", "#include <cstdint>", ""]

    for bv, wn, sn, hv in combos:
        for (_, state_short), (_, _, alog_short), (_, _, scalar_short), mtp in itertools.product(
            STATE_TYPES, ALOG_TYPES, SCALAR_TYPES, [False, True]
        ):
            fname = wrapper_name_k_last(bv, wn, sn, hv, alog_short, scalar_short, mtp, state_short)
            lines += [f"void {fname}(", f"    {K_LAST_WRAPPER_PARAMS});"]

    lines += ["",
              f"using k_last_launcher_t = void(*)(",
              f"    {K_LAST_WRAPPER_PARAMS});",
              "",
              "inline k_last_launcher_t get_k_last_launcher_dispatch(",
              "    int bv, int wn, int sn, int hv_per_cta,",
              "    at::ScalarType scalar_type, at::ScalarType alog_type, bool mtp_verify,",
              "    at::ScalarType state_dtype = at::ScalarType::Float) {"]

    state_enums = {"f32": "at::ScalarType::Float", "bf16": "at::ScalarType::BFloat16"}
    for i, (bv, wn, sn, hv) in enumerate(combos):
        cond = "if" if i == 0 else "} else if"
        lines.append(f"    {cond} (bv == {bv} && wn == {wn} && sn == {sn} && hv_per_cta == {hv}) {{")
        for _, state_short in STATE_TYPES:
            lines.append(f"        if (state_dtype == {state_enums[state_short]}) {{")
            name_map = {}
            for _, _, as_ in ALOG_TYPES:
                for _, _, ss_ in SCALAR_TYPES:
                    for mtp in [False, True]:
                        name_map[(as_, ss_, mtp)] = wrapper_name_k_last(bv, wn, sn, hv, as_, ss_, mtp, state_short)
            lines += ["    " + line for line in _dtype_dispatch_block(name_map, is_packed=False)]
            lines.append("        }")

    lines += ["    }", "    return nullptr;", "}"]
    return GENERATED_BSD3_HEADER + "\n".join(lines) + "\n"


def generate_dispatch_header_k_last_packed(combos):
    lines = ["#pragma once", "", "#include <hggc_runtime.h>", "#include <cstdint>", ""]

    # Declare all wrapper functions (both fp32 and bf16 state variants)
    for bv, wn, sn, hv in combos:
        for state_t, state_short in STATE_TYPES:
            for _, _, alog_short in ALOG_TYPES:
                for _, _, scalar_short in SCALAR_TYPES:
                    fname = wrapper_name_k_last_packed(bv, wn, sn, hv, alog_short, scalar_short, state_short)
                    lines += [f"void {fname}(", f"    {K_LAST_PACKED_WRAPPER_PARAMS});"]

    lines += ["",
              f"using k_last_packed_launcher_t = void(*)(",
              f"    {K_LAST_PACKED_WRAPPER_PARAMS});",
              "",
              "inline k_last_packed_launcher_t get_k_last_packed_launcher_dispatch(",
              "    int bv, int wn, int sn, int hv_per_cta,",
              "    at::ScalarType scalar_type, at::ScalarType alog_type,",
              "    at::ScalarType state_dtype = at::ScalarType::Float) {"]

    alog_enums = {"f32": "at::ScalarType::Float", "fp16": "at::ScalarType::Half", "bf16": "at::ScalarType::BFloat16"}
    scalar_enums = {"fp16": "at::ScalarType::Half", "bf16": "at::ScalarType::BFloat16"}
    state_enums = {"f32": "at::ScalarType::Float", "bf16": "at::ScalarType::BFloat16"}

    for i, (bv, wn, sn, hv) in enumerate(combos):
        cond = "if" if i == 0 else "} else if"
        lines.append(f"    {cond} (bv == {bv} && wn == {wn} && sn == {sn} && hv_per_cta == {hv}) {{")
        first_dtype = True
        for _, state_short in STATE_TYPES:
            for _, _, alog_short in ALOG_TYPES:
                for _, _, scalar_short in SCALAR_TYPES:
                    prefix = "if" if first_dtype else "} else if"
                    first_dtype = False
                    cond_str = (f"alog_type == {alog_enums[alog_short]} && "
                                f"scalar_type == {scalar_enums[scalar_short]} && "
                                f"state_dtype == {state_enums[state_short]}")
                    fname = wrapper_name_k_last_packed(bv, wn, sn, hv, alog_short, scalar_short, state_short)
                    lines.append(f"        {prefix} ({cond_str}) {{")
                    lines.append(f"            return {fname};")
        lines.append("        }")

    lines += ["    }", "    return nullptr;", "}"]
    return GENERATED_BSD3_HEADER + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tuned dispatch header (hash map: runtime params → best template combo)
# ---------------------------------------------------------------------------

def _bucket_table_entries(recs, batch_interval=16):
    """Build the B/interval-indexed table rows for one bucket series."""
    recs = sorted(recs, key=lambda x: x["B"])
    max_b = max(r["B"] for r in recs)
    array_size = max_b // batch_interval + 1
    entries = []
    rec_idx = 0
    for i in range(array_size):
        b_val = i * batch_interval
        while rec_idx < len(recs) - 1 and recs[rec_idx + 1]["B"] <= b_val:
            rec_idx += 1
        r = recs[rec_idx]
        entries.append(f"{{{r['BV']}, {r['WARP_NUM']}, {r['STAGE_NUM']}, {r['HV_PER_CTA']}}}")
    return entries


def _emit_bucket_table(result, arr_name, entries, indent="        "):
    result.append(f"{indent}static constexpr TunedParams {arr_name}[] = {{")
    for row_start in range(0, len(entries), 8):
        row = ", ".join(entries[row_start:row_start+8])
        result.append(f"{indent}    {row},")
    result.append(f"{indent}}};")


def _generate_device_lookup(records_for_device, kernel_type, func_name, batch_interval=16):
    """Generate array-based lookup: index = B / batch_interval.

    k_last_mtp records may carry a chain/tree mode; the lookup then takes a
    `tree` flag and picks the matching table. A shape with records in only one
    mode serves both branches with that table.
    """
    groups = defaultdict(list)
    for r in records_for_device:
        if r["kernel"] == kernel_type:
            state_dtype = r.get("state_dtype", "fp32") if kernel_type == "k_last_packed" else "any"
            mode = r.get("mode") if kernel_type == "k_last_mtp" else None
            groups[(r["H"], r["HV"], r["K"], r["V"], state_dtype, mode)].append(r)

    if not groups:
        return ["    return DEFAULT_PARAMS;"]

    # Collapse the mode dimension into per-shape dicts for k_last_mtp.
    shapes = defaultdict(dict)
    for (h, hv, k, v, state_dtype, mode), recs in groups.items():
        shapes[(h, hv, k, v, state_dtype)][mode] = recs

    result = []
    first_shape = True
    for (h, hv, k, v, state_dtype), mode_recs in sorted(shapes.items()):
        cond = "if" if first_shape else "} else if"
        first_shape = False
        state_cond = ""
        if kernel_type == "k_last_packed":
            state_cond = " && bf16_state" if state_dtype == "bf16" else " && !bf16_state"
        result.append(f"    {cond} (H == {h} && HV == {hv} && K == {k} && V == {v}{state_cond}) {{")

        chain_recs = mode_recs.get("chain") or mode_recs.get(None)
        tree_recs = mode_recs.get("tree") or mode_recs.get(None)
        result.append(f"        int idx = B / {batch_interval};")
        if kernel_type != "k_last_mtp" or chain_recs is tree_recs:
            recs = chain_recs if chain_recs is not None else tree_recs
            entries = _bucket_table_entries(recs, batch_interval)
            arr_name = f"table_{kernel_type}_{h}_{hv}_{k}_{v}"
            _emit_bucket_table(result, arr_name, entries)
            result.append(f"        if (idx >= {len(entries)}) idx = {len(entries) - 1};")
            result.append(f"        return {arr_name}[idx];")
        else:
            for label, recs in (("chain", chain_recs), ("tree", tree_recs)):
                if recs is None:
                    continue
                entries = _bucket_table_entries(recs, batch_interval)
                arr_name = f"table_{kernel_type}_{label}_{h}_{hv}_{k}_{v}"
                _emit_bucket_table(result, arr_name, entries)
                if label == "tree":
                    result.append(f"        if (tree) {{")
                else:
                    result.append(f"        {'if (!tree) ' if tree_recs is not None else ''}{{")
                result.append(f"            if (idx >= {len(entries)}) idx = {len(entries) - 1};")
                result.append(f"            return {arr_name}[idx];")
                result.append("        }")
            # One mode missing: fall through to the other table.
            if chain_recs is None or tree_recs is None:
                recs = chain_recs or tree_recs
                label = "chain" if chain_recs is not None else "tree"
                arr_name = f"table_{kernel_type}_{label}_{h}_{hv}_{k}_{v}"
                result.append(f"        if (idx >= {len(_bucket_table_entries(recs, batch_interval))}) idx = {len(_bucket_table_entries(recs, batch_interval)) - 1};")
                result.append(f"        return {arr_name}[idx];")

    result += ["    }", "    return DEFAULT_PARAMS;"]
    return result


def generate_tuned_dispatch(records):
    lines = ["#pragma once", "",
             "#include <string>",
             "#include <hggc_runtime.h>",
             "",
             "struct TunedParams { int bv, wn, sn, hv_per_cta; };",
             "",
             f"static constexpr TunedParams DEFAULT_PARAMS = {{{DEFAULT_COMBO[0]}, {DEFAULT_COMBO[1]}, {DEFAULT_COMBO[2]}, {DEFAULT_COMBO[3]}}};",
             ""]

    # Group records by device_name
    devices = defaultdict(list)
    for r in records:
        devices[r.get("device_name", "")].append(r)

    device_names = sorted(devices.keys())

    # Generate per-device lookup functions
    for dev_name in device_names:
        safe_name = dev_name.replace("-", "_").replace(" ", "_").replace(".", "_")
        dev_records = devices[dev_name]

        for kernel_type, func_suffix in [("k_last", "k_last"),
                                          ("k_last_mtp", "k_last_mtp"),
                                          ("k_last_packed", "k_last_packed")]:
            fname = f"get_tuned_{func_suffix}_{safe_name}"
            state_arg = ", bool bf16_state" if kernel_type == "k_last_packed" else ""
            has_mode = any(r.get("mode") for r in dev_records if r["kernel"] == kernel_type)
            tree_arg = "" if kernel_type != "k_last_mtp" else (", bool tree" if has_mode else ", bool /*tree*/")
            lines.append(f"static inline TunedParams {fname}(int B, int H, int HV, int K, int V{state_arg}{tree_arg}) {{")
            lines += _generate_device_lookup(dev_records, kernel_type, fname)
            lines += ["}", ""]

    # Generate runtime dispatch that picks the right device function
    lines += [
        "static inline std::string _get_cuda_device_name() {",
        "    int dev; hggcGetDevice(&dev);",
        "    hggcDeviceProp prop; hggcGetDeviceProperties(&prop, dev);",
        "    return std::string(prop.name);",
        "}",
        "",
        "static inline const std::string& _cached_device_name() {",
        "    static std::string name = _get_cuda_device_name();",
        "    return name;",
        "}",
        "",
    ]

    for kernel_type, func_name in [("k_last", "get_tuned_k_last"),
                                    ("k_last_mtp", "get_tuned_k_last_mtp"),
                                    ("k_last_packed", "get_tuned_k_last_packed")]:
        state_arg = ", bool bf16_state" if kernel_type == "k_last_packed" else ""
        tree_arg = ", bool tree" if kernel_type == "k_last_mtp" else ""
        lines.append(f"inline TunedParams {func_name}(int B, int H, int HV, int K, int V{state_arg}{tree_arg}) {{")
        lines.append(f"    const std::string& dev = _cached_device_name();")
        first = True
        for dev_name in device_names:
            safe_name = dev_name.replace("-", "_").replace(" ", "_").replace(".", "_")
            target = f"get_tuned_{kernel_type}_{safe_name}"
            cond = "if" if first else "} else if"
            first = False
            lines.append(f'    {cond} (dev == "{dev_name}") {{')
            state_call = ", bf16_state" if kernel_type == "k_last_packed" else ""
            tree_call = ", tree" if kernel_type == "k_last_mtp" else ""
            lines.append(f"        return {target}(B, H, HV, K, V{state_call}{tree_call});")
        lines += ["    }", "    return DEFAULT_PARAMS;", "}", ""]

    return GENERATED_BSD3_HEADER + "\n".join(lines) + "\n"


def generate_default_tuned_dispatch():
    lines = ["#pragma once", "",
             "struct TunedParams { int bv, wn, sn, hv_per_cta; };",
             "",
             f"static constexpr TunedParams DEFAULT_PARAMS = {{{DEFAULT_COMBO[0]}, {DEFAULT_COMBO[1]}, {DEFAULT_COMBO[2]}, {DEFAULT_COMBO[3]}}};",
             ""]
    for fn in ["get_tuned_k_last", "get_tuned_k_last_mtp", "get_tuned_k_last_packed"]:
        state_arg = ", bool bf16_state" if fn == "get_tuned_k_last_packed" else ""
        tree_arg = ", bool tree" if fn == "get_tuned_k_last_mtp" else ""
        lines += [f"inline TunedParams {fn}(int B, int H, int HV, int K, int V{state_arg}{tree_arg}) {{",
                  "    return DEFAULT_PARAMS;", "}", ""]
    return GENERATED_BSD3_HEADER + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate kernel instantiation files")
    parser.add_argument("--default-only", action="store_true",
                        help="Only default combo BV=64,WN=8,SN=4,HV_PER_CTA=1")
    parser.add_argument("--tuned-csv", type=str, default=None,
                        help="Path to tuned_result.csv")
    args = parser.parse_args()

    os.makedirs(GENERATED_DIR, exist_ok=True)
    for f in os.listdir(GENERATED_DIR):
        os.remove(os.path.join(GENERATED_DIR, f))

    if args.default_only:
        combos = [DEFAULT_COMBO]
        tuned_header = generate_default_tuned_dispatch()
    elif args.tuned_csv:
        tuned_combos, records = load_tuned_csv(args.tuned_csv)
        tuned_combos.add(DEFAULT_COMBO)
        all_valid = get_valid_combinations()
        combos = [c for c in all_valid if c in tuned_combos]
        tuned_header = generate_tuned_dispatch(records)
    else:
        combos = get_valid_combinations()
        tuned_header = generate_default_tuned_dispatch()

    print(f"Generating {len(combos)} combos × 2 kernels = {len(combos) * 2} .cu files")

    for bv, wn, sn, hv in combos:
        with open(os.path.join(GENERATED_DIR, f"k_last_bv{bv}_wn{wn}_sn{sn}_hv{hv}.cu"), "w") as f:
            f.write(generate_k_last_cu(bv, wn, sn, hv))
        with open(os.path.join(GENERATED_DIR, f"k_last_packed_bv{bv}_wn{wn}_sn{sn}_hv{hv}.cu"), "w") as f:
            f.write(generate_k_last_packed_cu(bv, wn, sn, hv))

    with open(os.path.join(GENERATED_DIR, "fused_sigmoid_gating_dispatch_k_last.h"), "w") as f:
        f.write(generate_dispatch_header_k_last(combos))
    with open(os.path.join(GENERATED_DIR, "fused_sigmoid_gating_dispatch_k_last_packed.h"), "w") as f:
        f.write(generate_dispatch_header_k_last_packed(combos))
    with open(os.path.join(GENERATED_DIR, "tuned_dispatch.h"), "w") as f:
        f.write(tuned_header)

    print(f"Generated in {GENERATED_DIR}: {len(combos)} k_last + {len(combos)} k_last_packed + 3 headers")


if __name__ == "__main__":
    main()
