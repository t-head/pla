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

from pathlib import Path
import os
import sys
import argparse
import csv
import re
import shlex
import subprocess
import traceback
from datetime import datetime

import pandas as pd
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Ensure tests dir on sys.path so utils.py is importable when launched from any cwd
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# helper functions are centralized in tests/utils.py
from utils import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    normalize_seqlen_q,
    parse_pla_string_re,
    replace_seqlen_q,
    read_cmds_raw_from_file as read_cmds_from_file,
    resolve_cu_seqlens,
    detect_is_kda_in_path,
    detect_is_kda_in_text,
    DEFAULT_MTP_EAGLE_T,
    DEFAULT_MTP_EAGLE_TOPK,
    DEFAULT_MTP_DRAFT_TOKEN_NUMS,
    mtp_subcase_specs,
    select_eagle_boundary_indices,
)


# ============================================================
# prefill kernel whitelist
# ============================================================
# decode launches only 1 target kernel per call, prefill does not: both sides launch a chain of kernels
# (cumsum -> kkt/solve -> w,u -> state recurrence -> output); the whole chain must be captured
# and summed per case, capturing only one of them badly underestimates. Following the approach of
# run_pmu_collection in OperatorsForge/tools/FAOpBench.py: acu takes a "regex:<A>|<B>|..." whitelist; the command goes through a shell and the regex
# contains '|', so the whole thing must be wrapped in single quotes again.
#
# pla side = the CuTe kernels in src/prefill/flashqla/kernels/ (correct_h0 /
# get_warmup_chunks only launch on the CP path; missing them does not affect the sum).
# With normalize_qk=True (production semantics) cumsum becomes the fused qk_norm kernel -- pla folds the two
# L2 norm passes into the cumsum launch without extra kernels, so both cumsum variants must be listed.
# fused_fwd / kkt_solve each have several specialized variants (indexed / runtime / pair / quad / oct);
# which one runs depends on the shape and whether varlen is padded; missing one captures only the cumsum launch,
# so the list includes every .so symbol from nm -C -- better too many than too few.
PREFILL_KERNELS_PLA = (
    "chunk_local_cumsum_cute_kernel",
    "chunk_local_cumsum_qk_norm_cute_kernel",
    "get_warmup_chunks_cute_kernel",
    "kkt_solve_cute_kernel_tc01",
    "kkt_solve_cute_kernel_tc02",
    "kkt_solve_oct_tc01_kernel",
    "kkt_solve_quad_tc01_kernel",
    "kkt_solve_pair_cute_kernel",
    "prepare_h_cp_cute_kernel",
    "correct_h0_cute_kernel",
    "fused_fwd_cp_cute_indexed_kernel",
    "fused_fwd_cp_cute_runtime_kernel",
    "fused_fwd_cp_cute_kernel",
)
# triton side = the call chain of sglang.srt.layers.attention.fla.chunk.chunk_gated_delta_rule:
#   l2norm.l2norm_fwd_kernel / _kernel1                (use_qk_l2norm_in_kernel=True
#                                                       launches once each for q/k, 2 launches total)
#   cumsum.chunk_local_cumsum_scalar_kernel        (g is a [B,T,H] scalar)
#   chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel
#   kda.recompute_w_u_fwd_kernel                   (reused by chunk_fwd_intra)
#   chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
#   chunk_o.chunk_fwd_kernel_o
PREFILL_KERNELS_TRITON = (
    # longer names first: alternation is leftmost-first; if kernel1 came after kernel,
    # "l2norm_fwd_kernel1" would first hit the prefix and rely on \b backtracking; the reverse order is simpler.
    "l2norm_fwd_kernel1",
    "l2norm_fwd_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_local_cumsum_vector_kernel",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
)


def prefill_kernels(is_triton):
    """Return (acu --kernel-name argument, target_kernels regex fragment)."""
    names = PREFILL_KERNELS_TRITON if is_triton else PREFILL_KERNELS_PLA
    alternation = "|".join(names)
    return f"'regex:{alternation}'", alternation


def run_cmd(cmd: str, timeout=3600, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    print(f"Run command: {cmd}, timeout: {timeout}")
    try:
        ret = subprocess.run(args=cmd, timeout=timeout, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8")
        if stdout == subprocess.PIPE:
            for line in ret.stdout.splitlines() + ret.stderr.splitlines():
                print(line)
        if ret.returncode != 0:
            print(f"Run command failed!")
            return False
        else:
            print(f"Run command succeed!")
            return True
    except Exception as e:
        print(traceback.format_exc())
        return False


def read_cycle_from_nculog(filename, target_kernels=None):
    # kernel_pattern = r"(.*)deep_gemm(.*)"
    # kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    kernel_pattern = r"(.*)Device\s+\d+"
    cycles_pattern = r"__cycles_elapsed.max"
    time_pattern = r"time_duration"
    global tc_pattern
    # tc_pattern = "we_pipe_tensor_cycles_active"
    tc_pattern = "_tensor_(.*)avg.pct_of_peak_sustained"
    compute_pattern = r"cu__throughput\.avg\.pct_of_peak_sustained_elapsed"
    dram_pattern = r"ppu__dram_throughput\.avg\.pct_of_peak_sustained_elapsed"
    kernel_list = []
    cycles_list = []
    tc_list = []
    compute_list = []
    dram_list = []
    time_list = []

    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            line = line.strip()

            # 1. filter: skip empty lines and lines with system notices (e.g. [Asight])
            if not line or "[Asight]" in line or "deprecated" in line:
                continue

            # 2. parse: try to extract the data
            try:
                parts = line.split()
                if not parts:
                    continue
                last_word = parts[-1]  # last word of the line (usually the value)

                if re.search(kernel_pattern, line):
                    kernel_list.append(line)

                if re.search(cycles_pattern, line):
                    # only append if it converts to a number
                    try:
                        cycles_list.append(int(last_word))
                    except (ValueError, TypeError):
                        cycles_list.append(0)

                if re.search(tc_pattern, line):
                    tc_list.append(_safe_float(last_word))

                if re.search(compute_pattern, line):
                    compute_list.append(_safe_float(last_word))

                if re.search(dram_pattern, line):
                    dram_list.append(_safe_float(last_word))

                if re.search(time_pattern, line):
                    fwd_time = _safe_float(last_word)
                    if "ns" in line:  # ppu return ns, gpu return us
                        fwd_time = fwd_time / 1000
                    time_list.append(round(fwd_time, 4))

            except (ValueError, IndexError):
                # if a keyword matched but the last token is not a number (e.g. a system warning),
                # float() conversion fails; just ignore this line and continue with the next one.
                continue

    if (len(kernel_list) != len(cycles_list)) or (len(kernel_list) != len(tc_list)) \
            or (len(kernel_list) != len(compute_list)) or (len(kernel_list) != len(dram_list)):
        print(f"assert len(kernel_list){len(kernel_list)} == len(cycles_list){len(cycles_list)} == len(tc_list){len(tc_list)} == len(compute_list)"
              f"{len(compute_list)} == len(dram_list){len(dram_list)} failed!!")
        return 0, 0, [], 0, 0, 0

    op_cycles = dict()
    fwd_cycle_sum = 0
    fwd_tc_sum = 0
    fwd_compute_sum = 0
    fwd_dram_sum = 0
    fwd_time = 0

    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        op_cycles[op] = cycle
        fwd_cycle_sum += cycle
        fwd_time += time_list[i]
        # Only update tc/compute/dram if we have valid (non-zero) values
        if tc_list[i] != 0:
            fwd_tc_sum = tc_list[i]
        if compute_list[i] != 0:
            fwd_compute_sum = compute_list[i]
        if dram_list[i] != 0:
            fwd_dram_sum = dram_list[i]
        # Special case: gemv kernel may have 0 tc but still valid
        if "gemv" in op.lower():
            if tc_list[i] != 0:
                fwd_tc_sum = tc_list[i]
            if compute_list[i] != 0:
                fwd_compute_sum = compute_list[i]
            if dram_list[i] != 0:
                fwd_dram_sum = dram_list[i]
    # calculate statistics data
    if fwd_cycle_sum != 0:
        # fwd unit case
        return fwd_cycle_sum, fwd_tc_sum, op_cycles, fwd_compute_sum, fwd_dram_sum, fwd_time
    else:
        print("Not valid CSV file!")
        return 0, 0, [], 0, 0, 0


def map_kernel_id_to_case_id(filename, fa_cases, backend="pla", target_kernels=None):
    fwd_pattern = re.compile(r"--- start test case (\d+)")
    if target_kernels is None:
        target_kernels = "fused_sigmoid_gating_delta_rule_update_kernel_fast|fused_sigmoid_gating_delta_rule_cuda_kernel"
    kernel_pattern = re.compile(rf'"({target_kernels})" - (\d+)')
    current_case = None
    kernel_to_case = {}
    line_idx = -1
    with open(filename, newline='') as log_file:
        for line in log_file.read().split("\n"):
            line_idx += 1
            # check whether a new FWD case starts
            fwd_match = fwd_pattern.search(line)
            if fwd_match:
                current_case = int(fwd_match.group(1))
                continue  # note: do not use elif, since the same line may have no kernel

            # find device_kernel lines
            kernel_match = kernel_pattern.search(line)
            if kernel_match and current_case is not None:
                kernel_id = int(kernel_match.groups()[-1])
                kernel_to_case[kernel_id] = current_case

            if "terminated" in line:
                break
        # found end
    print(line_idx)
    print(kernel_to_case)
    return line_idx, kernel_to_case


def batch_read_cycle_from_nculog(filename, fa_cases, backend='pla', is_ppu=True, bwd=False, target_kernels=None):
    '''
    type1: get_mla_metadata_kernel, flash_sparse_decode_fwd_kernel/flash_fwd_splitkv_mla_kernel, mla_combine_kernel
    type2: flash_sparse_prefill_fwd_kernel
    type3: fused_sigmoid_gating_delta_rule_update_kernel_fast/fused_sigmoid_gating_delta_rule_cuda_kernel
           fused_sigmoid_gating_delta_kernel/fused_sigmoid_gating_delta_kernel_packed
    '''
    # match kernel lines with the exact target_kernels names + the Device marker
    # actual ACU log format: "fused_sigmoid_gating_delta_kernel (1,4,8)x(32,1,1), Device 5" -- the kernel name is unquoted
    if target_kernels is None:
        target_kernels_str = "fused_sigmoid_gating_delta_rule_update_kernel_fast|fused_sigmoid_gating_delta_rule_cuda_kernel"
    else:
        target_kernels_str = target_kernels
    kernel_pattern = rf'\b({target_kernels_str})\b.*Device\s+\d+'
    # anchor for any kernel line: kernel instance request lines in the ACU log all end with "Device <id>"
    # e.g.: fused_sigmoid_gating_delta_rule_update_kernel (1,4,8)x(32,1,1), Device 5
    any_kernel_pattern = r'.+Device\s+\d+'
    duration_pattern = "__time_duration.sum"
    cycles_pattern = "__cycles_elapsed.max"
    tc_pattern = "_tensor_(.*)avg.pct_of_peak_sustained"  # cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
    l2_pattern = "hit_rate.pct"
    compute_pattern = r"cu__throughput\.avg\.pct_of_peak_sustained_elapsed"
    dram_pattern = r"ppu__dram_throughput\.avg\.pct_of_peak_sustained_elapsed"

    kernel_list = []
    duration_list = []
    cycles_list = []
    tc_list = []
    l2_list = []
    compute_list = []
    dram_list = []
    start_idx, kernel_to_case = map_kernel_id_to_case_id(filename, fa_cases, backend, target_kernels=target_kernels)
    # parse by kernel block: on any kernel line -> decide whether it is a target;
    # metrics are only collected inside target blocks. This way mtp mode can take the cycle
    # of only the cuda kernel or only the triton kernel without mixing them.
    in_target_block = False
    with open(filename, newline='') as log_file:
        for current_line_num, line in enumerate(log_file):
            if current_line_num < start_idx:  # skip the first idx lines (0 ~ idx-1)
                continue
            line = line.rstrip('\n\r')
            # 1) is this a kernel line (any kernel)
            if re.search(any_kernel_pattern, line) and "cycles_elapsed" not in line \
                    and "time_duration" not in line and "throughput" not in line \
                    and "hit_rate" not in line and "_tensor_" not in line:
                if re.search(kernel_pattern, line, re.IGNORECASE):
                    in_target_block = True
                    kernel_list.append(line.strip())
                else:
                    in_target_block = False
                continue
            # 2) metrics outside a target kernel block are always skipped
            if not in_target_block:
                continue
            if re.search(cycles_pattern, line):
                cycles_list.append(_safe_int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(_safe_float(line.strip().split()[-1]))
            if re.search(l2_pattern, line):
                l2_list.append(_safe_float(line.strip().split()[-1]))
            if re.search(compute_pattern, line):
                compute_list.append(_safe_float(line.strip().split()[-1]))
            if re.search(dram_pattern, line):
                dram_list.append(_safe_float(line.strip().split()[-1]))
            if re.search(duration_pattern, line):
                fwd_time = _safe_float(line.strip().split()[-1])
                if "ns" in line:  # ppu return ns, gpu return us
                    fwd_time = fwd_time / 1000
                duration_list.append(round(fwd_time, 4))

    # No assert. If the lengths differ (common when the ACU log format does not match expectations),
    # truncate to the shortest list and warn, so the whole flow does not abort with an AssertionError.
    list_lengths = {
        "kernel": len(kernel_list),
        "cycles": len(cycles_list),
        "tc": len(tc_list),
        "l2": len(l2_list),
        "compute": len(compute_list),
        "dram": len(dram_list),
        "duration": len(duration_list),
    }
    if len(set(list_lengths.values())) != 1:
        print(f"WARNING: list length mismatch in batch_read_cycle_from_nculog: {list_lengths}")
        min_len = min(list_lengths.values())
        kernel_list = kernel_list[:min_len]
        cycles_list = cycles_list[:min_len]
        tc_list = tc_list[:min_len]
        l2_list = l2_list[:min_len]
        compute_list = compute_list[:min_len]
        dram_list = dram_list[:min_len]
        duration_list = duration_list[:min_len]
        if min_len == 0:
            print("WARNING: kernel_list is empty after truncation; returning empty result.")
            return []

    # Group by case before aggregating, instead of "the i-th kernel launch pairs with the i-th case".
    # On decode one case launches only one target kernel, so both are equivalent; on prefill one call launches
    # several kernels (prepare / recurrence / merge ...), which must be grouped and summed to line up. When a case in the middle
    # fails functionally and produces no launch, sequential pairing would also shift later cases' cycles onto earlier cases.
    #
    # The aggregation criteria match read_cycle_from_nculog in OperatorsForge/tools/FAOpBench.py:
    #   cycle = sum of ce cycles of all matched kernels in the case
    #   utilization = cycle-weighted average (otherwise small kernels skew the overall utilization)
    # When kernel_to_case is empty (no "<kernel>" - <id> markers captured in the log), fall back to sequential pairing,
    # keeping the behavior from before this change.
    grouped = {}
    for i in range(len(kernel_list)):
        case_no = kernel_to_case.get(i, i + 1)
        agg = grouped.setdefault(case_no, {
            "cycle": 0, "duration": 0.0, "tc": 0.0,
            "l2": 0.0, "compute": 0.0, "dram": 0.0, "launches": 0,
        })
        cycle = cycles_list[i]
        agg["cycle"] += cycle
        agg["duration"] += duration_list[i]
        agg["tc"] += tc_list[i] * cycle
        agg["l2"] += l2_list[i] * cycle
        agg["compute"] += compute_list[i] * cycle
        agg["dram"] += dram_list[i] * cycle
        agg["launches"] += 1

    output_list = list()
    for case_no in sorted(grouped):
        agg = grouped[case_no]
        idx = case_no - 1
        if idx < 0 or idx >= len(fa_cases):
            print(f"WARNING: kernel belongs to case index {case_no}, out of caselist range "
                  f"({len(fa_cases)}), dropping this group of data")
            continue
        cycle = agg["cycle"]
        weight = cycle if cycle else 1
        output_list.append({
            "case name": fa_cases[idx].strip(),
            "op time in us": round(agg["duration"], 4),
            "pmu cycles": cycle,
            "pmu flops util": round(agg["tc"] / weight, 2),
            "pmu compute util": round(agg["compute"] / weight, 2),
            "pmu dram util": round(agg["dram"] / weight, 2),
            "pmu l2 util": round(agg["l2"] / weight, 2),
            "kernel launches": agg["launches"],
        })

    # return statistics data
    return output_list


def check_perf(test_perf_result, row, caselist_path, record_file):
    triton_path = Path("/ppusw/share/likaixu.lkx/fla_0308/triton_kernel")

    file_name = os.path.basename(caselist_path)
    parts = file_name.split('_')
    model_version = parts[0].split('-')[0]
    tp = ""
    for p in parts:
        if p.startswith('tp'):
            tp = p
            break
    log = model_version + "-" + tp
    search_file = next(triton_path.glob(f"*{log}*.csv"), None)

    current_case = row[0].strip("'")
    current_cycle = float(row[2])
    baseline_cycle = None

    if search_file and search_file.exists():
        with open(search_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for csv_row in reader:
                if csv_row['casename'].strip("'") == current_case:
                    baseline_cycle = float(csv_row['cycle'])
                    break
    else:
        print(f"Error: baseline file not found {log}")
        return False

    if baseline_cycle is None:
        print(f"Warning: Case '{current_case}' does not exist in the baseline file")
        return False

    ratio = (baseline_cycle / current_cycle) * 100
    ratio_str = f"{ratio:.2f}%"

    with open(record_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([current_case, baseline_cycle, current_cycle, ratio_str])

    if current_cycle > baseline_cycle * 1.1:
        print(f"Performance Regression! {current_case}: {baseline_cycle} -> {current_cycle} (>10%)")
        test_perf_result.append(current_case)
        return False

    print(f"perf check pass: {current_case} | Ratio: {ratio_str}")
    return True


def main():
    parser = argparse.ArgumentParser(description="GDN caselist test")
    parser.add_argument("--caselist", type=str, help="caselist file path")
    parser.add_argument("--case_idx", type=str, help="line index in caselist file path")
    parser.add_argument("--format", type=str,
                        help="input case format:[GDN]:[D][4]gdn_attn:seqlen_q:1,head_dim_qk:128,head_dim_v:128,"
                             "num_heads_qk:16,num_heads_v:32,hidden_act:silu,conv_kdim:4,batch_size:4,dtype:bf16,tp:4.")
    parser.add_argument("--tp", type=int, help="must give tp size")
    parser.add_argument("--kernel", type=str, default="cuda",
                        choices=["cuda", "k_last", "k_last_packed", "k_last_mtp", "prefill"],
                        help="kernel type to use: cuda (default), k_last, k_last_packed, k_last_mtp, "
                             "prefill (runs test_op_forge_prefill.py, sums over a multi-kernel whitelist)")
    parser.add_argument("--is_kda", action="store_true",
                        help="force KDA mode and pass it through to test_op_forge.py. When omitted it is auto-detected from Kimi/[KDA]/kda_attn in the caselist path/lines.")
    parser.add_argument('--output', default="output", type=str, required=False,
                        help='the output storing cycles of DG cases')
    parser.add_argument('--ischeckin', default=False, type=bool, required=False,
                        help='set True for checkin perf test')
    parser.add_argument('--none_consecutive', default=False, action="store_true", required=False,
                        help='specify if run test_op_forge.py consecutively or not')
    parser.add_argument('--daily', action='store_true', required=False,
                        help='run in daily mode, output is a directory for this test run')
    parser.add_argument('--mtp-topks', type=str, default="1,2,3,4,5",
                        help='topk list covered in k_last_mtp mode, comma separated (default 1,2,3,4,5). topk=1 is single_chain, >1 is eagle_tree.')
    parser.add_argument('--mtp-draft-tokens', type=str, default="4,16",
                        help='draft_token_num(T) list covered in k_last_mtp mode, comma separated (default 4,16). '
                             'T and topk are independent and are cross-expanded; T<3 is raised to 3.')
    args = parser.parse_args()
    # parse --mtp-topks / --mtp-draft-tokens into integer lists
    try:
        args.mtp_topks = [int(x) for x in args.mtp_topks.split(',') if x.strip()]
    except ValueError:
        print(f"ERROR: bad --mtp-topks format: {args.mtp_topks}")
        return
    try:
        args.mtp_draft_tokens = [int(x) for x in args.mtp_draft_tokens.split(',') if x.strip()]
    except ValueError:
        print(f"ERROR: bad --mtp-draft-tokens format: {args.mtp_draft_tokens}")
        return
    print(f"*--- fused_sigmoid_gating_delta_rule_update perf test ---*")
    print(f"DEBUG args: caselist={args.caselist}, kernel={args.kernel}, none_consecutive={args.none_consecutive}, daily={args.daily}, is_kda={args.is_kda}")

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    # Defaults to the personal network drive directory (local manual runs); that path may not exist in CI containers,
    # where PLA_PERF_RECORD_DIR overrides it to this run's directory; falls back to ./logs when mkdir fails
    record_dir = Path(os.environ.get("PLA_PERF_RECORD_DIR",
                                     "/ppusw/share/likaixu.lkx/pla_perf_temp"))
    try:
        record_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        record_dir = Path("./logs/perf_record")
        print(f"WARNING: cannot create record directory ({exc}), falling back to {record_dir}")
        record_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caselist_filename = os.path.basename(args.caselist)

    # Extract model name and tp from caselist filename
    # Pattern: {model}_decode_tp{tp}... or {model}_mtp_tp{tp}...
    # e.g., Qwen3-Next-80B-A3B_decode_tp1_kvlimit_200GB_model_len_128K_gdn_op.list
    # e.g., Qwen3-Next-80B-A3B_mtp_tp1_gdn_op.list
    decode_match = re.search(r"(.+)_(decode|mtp|prefill)_tp(\d+)", caselist_filename)
    if decode_match:
        model_name = decode_match.group(1)  # the part before decode/mtp is the model name
        mode_str = decode_match.group(2)    # decode or mtp
        tp_size = decode_match.group(3)     # the number after tp is the tp size
        # when running the MTP kernel on a reused decode caselist, mark the output name explicitly as mtp
        if args.kernel == "k_last_mtp" and mode_str == "decode":
            mode_str = "mtp"
        model_tp_prefix = f"{model_name}-{mode_str}_tp{tp_size}"
    else:
        # Fallback if pattern doesn't match
        model_tp_prefix = caselist_filename.replace('.list', '').replace('.', '_')
        tp_size = "unknown"

    current_record_csv = record_dir / f"perf_{model_tp_prefix}_{timestamp}.csv"

    with open(current_record_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['casename', 'triton_cycle', 'new_cuda_cycle', 'cycle_compare'])

    if args.caselist is not None and args.none_consecutive == False:
        match = re.search(r"tp(\d+)", args.caselist)
        if match:
            tp_size = int(match.group(1))
        elif args.tp is not None:
            tp_size = int(args.tp)
        else:
            print("no tp_size value (no tp\\d+ in the caselist file name, and --tp not given)")
            return
        os.environ["ASIGHT_FEATURE_PC_SAMPLING"] = "0"
        os.environ["ASIGHT_FEATURE_PM_SAMPLING"] = "0"
        os.environ["ASIGHT_FEATURE_ICNLINK_SAMPLING"] = "0"
        os.environ["ASIGHT_COMPUTE_PROFILER_DISABLE_SW_PRE_PASS"] = "1"
        headers = ["casename", "time(us)", "cycle", "tc efficiency", "hbm efficiency", "dtype", "result", "cmd", "detail"]
        log_file = os.path.basename(args.caselist).replace(".", "_").replace("-", "_")
        log_file = f"./logs/case_{log_file}.log"

        if int(os.environ.get("TRITON_PLA", 0)) != 0:  # means TRITON_PLA=1, CUDA_PLA=0
            if args.kernel == "prefill":
                kernel_name, _ = prefill_kernels(is_triton=True)
                suffix = "_triton_prefill_acu_result.csv"
            elif args.kernel == "k_last":
                kernel_name = 'fused_sigmoid_gating_delta_rule_update_kernel'
                suffix = "_triton_k_last_acu_result.csv"
            elif args.kernel == "k_last_mtp":
                kernel_name = 'fused_sigmoid_gating_delta_rule_update_kernel'
                suffix = "_triton_k_last_mtp_acu_result.csv"
            elif args.kernel == "k_last_packed":
                kernel_name = 'fused_recurrent_gated_delta_rule_packed_decode_kernel'
                suffix = "_triton_k_last_packed_acu_result.csv"
            else:
                kernel_name = 'fused_sigmoid_gating_delta_rule_update_kernel_fast'
                suffix = "_triton_pla_acu_result.csv"
        elif args.kernel == "k_last":
            kernel_name = 'fused_sigmoid_gating_delta_kernel'
            suffix = "_cuda_k_last_acu_result.csv"
        elif args.kernel == "k_last_mtp":
            kernel_name = 'fused_sigmoid_gating_delta_kernel'
            suffix = "_cuda_k_last_mtp_acu_result.csv"
        elif args.kernel == "k_last_packed":
            kernel_name = 'fused_sigmoid_gating_delta_kernel_packed'
            suffix = "_cuda_k_last_packed_acu_result.csv"
        elif args.kernel == "prefill":
            kernel_name, _ = prefill_kernels(is_triton=False)
            suffix = "_cuda_prefill_acu_result.csv"
        else:  # DEFAULT: TRITON_PLA=0, CUDA_PLA=1
            kernel_name = 'fused_sigmoid_gating_delta_rule_cuda_kernel'
            suffix = "_cuda_pla_acu_result.csv"
        # Handle daily mode: if daily=True, output is a directory for this test run
        if args.daily:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            result_file_name = output_path / f"{model_tp_prefix}_{timestamp}{suffix}"
        else:
            result_file_name = os.path.basename(args.caselist).replace(".", "_").replace("-", "_") + suffix
        lines = read_cmds_from_file(args.caselist)
        if args.kernel == "prefill":
            # the prefill kernel only consumes varlen cases with explicit cu_seqlens (in test_op_forge_prefill.py,
            # _case_to_e2e_params judges the other lines as "no cu_seqlens" fail directly);
            # filter before collection so empty runs do not misalign case indices with kernel launches.
            lines = [ln for ln in lines
                     if resolve_cu_seqlens(parse_pla_string_re(ln)) is not None]
        elif args.kernel in ("k_last", "k_last_packed"):
            filtered_lines = []
            for line in lines:
                parsed = parse_pla_string_re(line)
                cu_seqlens = resolve_cu_seqlens(parsed)
                if cu_seqlens is not None and args.kernel != "k_last":
                    raise ValueError("explicit cu_seqlens is only supported by the k_last prefill test")
                if cu_seqlens is not None or int(parsed.get("seqlen_q", 1)) <= 1:
                    filtered_lines.append(line)
            lines = filtered_lines
        # for non-MTP kernels force seqlen_q to 1 for plain decode cases; prefill cases with explicit cu_seqlens keep their values
        if args.kernel != "prefill":
            lines = [normalize_seqlen_q(ln, args.kernel) for ln in lines]
        # MTP mode: reuse the decode caselist, expanding each original case into several sub-items.
        # T (draft_token_num) and topk are independent (following sglang):
        #   topk=1 -> single_chain (linear verify); topk>1 -> eagle_tree
        #   T is always >= 3; seqlen_q=1 in the caselist is meaningless for MTP, cross-expand with draft_token_nums
        # The expansion order must match test_op_forge.py, so reuse the single source of truth
        # utils.mtp_subcase_specs to guarantee the k-th kernel instance recorded by ACU maps strictly 1:1 to lines[k-1]
        if args.kernel == "k_last_mtp":
            expanded_lines = []
            mtp_topks = getattr(args, 'mtp_topks', None)
            mtp_draft_tokens = getattr(args, 'mtp_draft_tokens', None)
            # parse everything first, computing the eagle boundary cases with the same function/order as the functional side
            parsed_cases = [parse_pla_string_re(ln) for ln in lines]
            eagle_boundary = select_eagle_boundary_indices(parsed_cases)
            print(f"MTP: eagle tree runs only boundary case indices {sorted(eagle_boundary)} "
                  f"(of {len(lines)} cases); single_chain runs on all")
            for idx, line in enumerate(lines):
                parsed = parsed_cases[idx]
                specs = mtp_subcase_specs(
                    seqlen_q=parsed.get('seqlen_q', 1),
                    has_explicit_topk=('topk' in parsed),
                    explicit_topk=parsed.get('topk'),
                    topks=mtp_topks,
                    draft_token_nums=mtp_draft_tokens,
                    include_eagle=(idx in eagle_boundary),
                )
                base = line.rstrip('.')
                has_topk_in_line = 'topk:' in line
                for T, topk, _label in specs:
                    sub = replace_seqlen_q(base, T)
                    # do not append again when the line already has an explicit topk (specs already respect it)
                    expanded_lines.append(f"{sub}." if has_topk_in_line else f"{sub},topk:{int(topk)}.")
            lines = expanded_lines
            print(f"DEBUG MTP: expanded -> {len(lines)} sub-cases "
                  f"(topks={mtp_topks or 'default'}, draft_token_nums={mtp_draft_tokens or DEFAULT_MTP_DRAFT_TOKEN_NUMS})")
        metrics_string = (
            "ppu__time_duration.sum,"
            "ce__cycles_elapsed.max,"
            "cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
            "cu__throughput.avg.pct_of_peak_sustained_elapsed,"
            "l2__requests_hit_rate.pct,"
            "ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
        )
        script = "test_op_forge.py"
        # prefill uses a separate runner (the real CuTe prefill kernel, not decode's k_last);
        # it takes no --kernel / --is_kda, only --caselist; --perf tells it to skip the golden comparison
        # (golden is a torch float reference implementation, very slow on long sequences, and perf collection needs no accuracy).
        if args.kernel == "prefill":
            script = "test_op_forge_prefill.py"
        # pass is_kda through: forced by the --is_kda CLI, or auto-detected from Kimi/[KDA]/kda_attn in the caselist path/lines
        auto_is_kda = (args.is_kda or detect_is_kda_in_path(args.caselist)
                       or any(detect_is_kda_in_text(ln) for ln in lines))
        forge_extra = " --is_kda" if auto_is_kda else ""
        if auto_is_kda and not args.is_kda:
            print(f"DEBUG: auto-detected is_kda=True from caselist (path/content), forging --is_kda flag")
        # pass --mtp-topks / --mtp-draft-tokens through: guarantees the sub-items expanded by the child test_op_forge.py
        # match the perf side's expanded_lines strictly in count and order
        # otherwise the ACU kernel instance count != len(lines) -> batch_read_cycle_from_nculog index out of range
        forge_extra_topks = ""
        if args.kernel == "k_last_mtp":
            mtp_topks_str = ",".join(str(int(k)) for k in getattr(args, 'mtp_topks', [1, 2, 3, 4, 5]))
            mtp_dtns_str = ",".join(str(int(t)) for t in getattr(args, 'mtp_draft_tokens', DEFAULT_MTP_DRAFT_TOKEN_NUMS))
            forge_extra_topks = f" --mtp-topks={mtp_topks_str} --mtp-draft-tokens={mtp_dtns_str}"
        if args.kernel == "prefill":
            cmd = (
                f'acu --clock-control none --replay-mode application '
                f'--kernel-name {kernel_name} '
                f'--metrics="{metrics_string}" '
                f'--page=details '
                f'python -u {script} --caselist={args.caselist} --perf '
                f'2>&1 | tee {log_file}'
            )
        else:
            cmd = (
                f'acu --clock-control none --replay-mode application '
                f'--kernel-name {kernel_name} '
                f'--metrics="{metrics_string}" '
                f'--page=details '
                f'python -u {script} --caselist={args.caselist} --kernel={args.kernel}{forge_extra}{forge_extra_topks} '  # -u for real-time output
                f'2>&1 | tee {log_file}'
            )
        # NOTE: --replay-mode application makes the cycle include the context overhead of surrounding kernels
        # for an exact single-kernel cycle use: --replay-mode kernel (but profiling time grows significantly)
        if args.kernel == "prefill":
            _, target_kernels = prefill_kernels(
                is_triton=int(os.environ.get("TRITON_PLA", 0)) != 0)
        elif args.kernel == "k_last" or args.kernel == "k_last_mtp":
            if int(os.environ.get("TRITON_PLA", 0)) != 0:
                target_kernels = 'fused_sigmoid_gating_delta_rule_update_kernel'
            else:
                target_kernels = 'fused_sigmoid_gating_delta_kernel'
        elif args.kernel == "k_last_packed":
            if int(os.environ.get("TRITON_PLA", 0)) != 0:
                target_kernels = 'fused_recurrent_gated_delta_rule_packed_decode_kernel'
            else:
                target_kernels = 'fused_sigmoid_gating_delta_kernel_packed'
        else:
            target_kernels = None
        if run_cmd(cmd, timeout=None):
            # filename, fa_cases, backend, is_ppu, bwd
            output_list = batch_read_cycle_from_nculog(log_file, lines, target_kernels=target_kernels)
            if not output_list:
                print(f"WARNING: No kernels were profiled for {args.caselist}, skip writing CSV.")
                return
            with open(result_file_name, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=output_list[0].keys())
                writer.writeheader()
                writer.writerows(output_list)
        else:
            # if ret.returncode != 0:
            print("ERROR: failed to run cmd, please check!!")
            exit(1)

    elif args.caselist is not None and args.none_consecutive == True:
        match = re.search(r"tp(\d+)", args.caselist)
        if match:
            tp_size = int(match.group(1))
        elif args.tp is not None:
            tp_size = int(args.tp)
        else:
            print("no tp_size value (no tp\\d+ in the caselist file name, and --tp not given)")
            return

        caselist = read_cmds_from_file(args.caselist)
        i = 1
        temp_perf_fail = []
        for case in caselist:
            print(f"*--- start test case {i} ---*")
            torch.cuda.empty_cache()
            headers = ["casename", "time(us)", "cycle", "tc efficiency", "compute efficiency", "dram efficiency",
                       "dtype", "result", "cmd", "detail"]
            case_id_match = re.search(r"\[D\]\[(\d+)\]", case)
            case_id = case_id_match.group(1) if case_id_match else i

            log_file = f"./logs/case_{case_id}.log"
            output_file = output_file_base
            metrics_string = (
                "ppu__time_duration.sum,"
                "ce__cycles_elapsed.max,"
                "cu__we_pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
                "cu__throughput.avg.pct_of_peak_sustained_elapsed,"
                "l2__requests_hit_rate.pct,"
                "ppu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
            )
            script = "test_op_forge.py"
            case = case.rstrip(".") + f",tp:{tp_size}."
            # force seqlen_q=1 for plain decode cases; prefill cases with explicit cu_seqlens keep their values
            case = normalize_seqlen_q(case, args.kernel)
            format_arg = shlex.quote(case)
            # pass is_kda through: forced by the --is_kda CLI, or auto-detected from Kimi/[KDA]/kda_attn in the caselist path/case line
            auto_is_kda = (args.is_kda or detect_is_kda_in_path(args.caselist)
                           or detect_is_kda_in_text(case))
            forge_extra = " --is_kda" if auto_is_kda else ""
            if int(os.environ.get("TRITON_PLA", 0)) != 0:  # means TRITON_PLA=1, CUDA_PLA=0
                if args.kernel == "k_last" or args.kernel == "k_last_mtp" or args.kernel == "k_last_packed":
                    kernel_name = 'fused_sigmoid_gating_delta_rule_update_kernel'
                else:
                    kernel_name = 'fused_sigmoid_gating_delta_rule_update_kernel_fast'
            elif args.kernel == "k_last" or args.kernel == "k_last_mtp":
                kernel_name = 'fused_sigmoid_gating_delta_kernel'
            elif args.kernel == "k_last_packed":
                kernel_name = 'fused_sigmoid_gating_delta_kernel_packed'
            else:  # DEFAULT: TRITON_PLA=0, CUDA_PLA=1
                kernel_name = 'fused_sigmoid_gating_delta_rule_cuda_kernel'
            cmd = (
                f'acu --clock-control none '
                f'--kernel-name {kernel_name} '
                f'--metrics="{metrics_string}" '
                f'--page=details '
                f'python -u {script} --format={format_arg} --kernel={args.kernel}{forge_extra} '  # -u for real-time output
                f'2>&1 | tee {log_file}'
            )
            run_cmd(cmd)

            if args.kernel == "k_last" or args.kernel == "k_last_mtp":
                if int(os.environ.get("TRITON_PLA", 0)) != 0:
                    target_kernels = 'fused_sigmoid_gating_delta_rule_update_kernel_fast'
                else:
                    target_kernels = 'fused_sigmoid_gating_delta_kernel'
            elif args.kernel == "k_last_packed":
                if int(os.environ.get("TRITON_PLA", 0)) != 0:
                    target_kernels = 'fused_sigmoid_gating_delta_rule_update_kernel_fast'
                else:
                    target_kernels = 'fused_sigmoid_gating_delta_kernel_packed'
            else:
                target_kernels = None
            cycle, tc, detail, compute, dram, time = 0, 0, "", 0, 0, 0
            cycle, tc, detail, compute, dram, time = read_cycle_from_nculog(log_file, target_kernels=target_kernels)
            row = [f"'{case.replace(',', '_')}'", str(time), str(cycle), str(tc), str(compute), str(dram), str(detail)]
            perf_result = True
            if args.ischeckin:
                perf_check = check_perf(temp_perf_fail, row, args.caselist, current_record_csv)
                if perf_check == False:
                    perf_result = False

            output_csv = f"{output_file}.csv"
            if not os.path.exists(output_csv):
                with open(output_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

            with open(output_csv, "a+") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                print("write result succeed")

            i = i + 1
    else:
        print("Input error: please provide a caselist file")
        exit(1)

    if args.ischeckin == True and perf_result == False:
        print(temp_perf_fail)
        exit(1)


if __name__ == "__main__":
    main()
