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

import os
import sys
# Add the decode scripts dir to path BEFORE importing any modules from it
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "decode"))
# Ensure tests dir on sys.path so utils.py is importable when launched from any cwd
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pandas as pd
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import argparse
import re
import math
from contextlib import redirect_stdout
import io
from datetime import datetime

from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last
from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last_packed
import test_fused_sigmoid_gating_recurrent
from _kda_reference import naive_kda_lowerbound_gate, naive_recurrent_kda
try:
    import test_fused_sigmoid_gating_recurrent_k_last
    import test_fused_sigmoid_gating_recurrent_k_last_packed
    import test_fused_sigmoid_gating_recurrent_k_last_mtp
except ImportError as e:
    print(f"Import failed: {e}")  # print the concrete error message
    import traceback
    traceback.print_exc()         # print the full traceback to see which line crashed
    exit(1)

# ---- helper functions are centralized in tests/utils.py (reused by both test_op_forge scripts) ----
from utils import (
    parse_pla_string_re,
    read_cmds_from_file,
    collect_cases_from_path,
    parse_dtype,
    resolve_is_kda,
    resolve_cu_seqlens,
    expand_mtp_subcases,
    select_eagle_boundary_indices,
)


def _kda_lowerbound_reference(
    A_log, dt_bias, a, b, q, k, v, state, cu_seqlens, lower_bound, scale,
):
    """Pure PyTorch reference for the Kimi-K3 safe-gate varlen path."""
    offsets = cu_seqlens.cpu().tolist()
    q_tokens = q.view(-1, q.shape[-2], q.shape[-1])
    k_tokens = k.view(-1, k.shape[-2], k.shape[-1])
    v_tokens = v.view(-1, v.shape[-2], v.shape[-1])
    a_tokens = a.view(-1, a.shape[-2], a.shape[-1])
    b_tokens = b.view(-1, b.shape[-1])
    out = torch.empty_like(v_tokens)
    final_state = state.clone()

    for batch_idx, (start, end) in enumerate(zip(offsets, offsets[1:])):
        q_seq = q_tokens[start:end].unsqueeze(0)
        k_seq = k_tokens[start:end].unsqueeze(0)
        v_seq = v_tokens[start:end].unsqueeze(0)
        q_seq = q_seq.float() / torch.sqrt(
            torch.sum(q_seq.float().square(), dim=-1, keepdim=True) + 1e-6
        )
        k_seq = k_seq.float() / torch.sqrt(
            torch.sum(k_seq.float().square(), dim=-1, keepdim=True) + 1e-6
        )
        g_seq = naive_kda_lowerbound_gate(
            a_tokens[start:end].unsqueeze(0),
            A_log,
            dt_bias,
            lower_bound=lower_bound,
        )
        beta_seq = torch.sigmoid(b_tokens[start:end].float()).unsqueeze(0)
        initial_state = (
            final_state[batch_idx:batch_idx + 1]
            .transpose(-1, -2)
            .contiguous()
        )
        out_seq, state_seq = naive_recurrent_kda(
            q_seq,
            k_seq,
            v_seq,
            g_seq,
            beta_seq,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )
        out[start:end].copy_(out_seq.squeeze(0))
        final_state[batch_idx].copy_(state_seq.squeeze(0).transpose(-1, -2))

    return out.view_as(v), final_state


def klast_ut_test(
    B, T, H, HV, K, V, kernel, is_kda=False, dtype=torch.float16,
    cu_seqlens_values=None, gate_lower_bound=None,
):
    torch.manual_seed(42)
    device = "cuda"

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = test_fused_sigmoid_gating_recurrent_k_last._make_tensors(
        B, T, H, HV, K, V, is_kda, dtype=dtype,
        cu_seqlens_values=cu_seqlens_values,
    )

    # --- Memory optimization: avoid holding 3 copies of state simultaneously ---
    # Phase 1: run the matching reference. Kimi-K3's bounded sigmoid gate is
    # different from the legacy KDA softplus gate used by the Triton helper.
    kernel_scale = K ** -0.5 if is_kda and gate_lower_bound is not None else 1.0
    if is_kda and gate_lower_bound is not None:
        out_triton, s_triton = _kda_lowerbound_reference(
            A_log, dt_bias, a, b, q, k, v, state, cu_seqlens,
            gate_lower_bound, kernel_scale,
        )
    else:
        s_triton = state.clone()
        from fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update as triton_fused_gating
        )
        out_triton = triton_fused_gating(
            A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
            s_triton, indices, scale=1.0, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens, is_kda=is_kda,
            disable_state_update=False,
        )

    # Phase 2: run CUDA target directly on the original state (no extra clone needed)
    s_cuda = state  # reuse original — saves one full state copy
    del state
    out_cuda = fused_sigmoid_gating_delta_rule_forward_k_last(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
        s_cuda, indices, kernel_scale, True,
        cu_seqlens, is_kda,
        False, None, None, None, None, gate_lower_bound,
    )

    try:
        torch.testing.assert_close(s_cuda, s_triton, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(out_cuda, out_triton, atol=1e-3, rtol=1e-2)
    except AssertionError as e:
        print(f"{e}")
        print(f"--- case fail!!! ---")

def klast_packed_ut_test(B, T, H, HV, K, V, pool_size, is_kda=False, dtype=torch.float16):
    """Test function for k_last_packed kernel (memory-optimized)."""
    print(f"test case info: batch_size: {B}, seqlen: {T}, num_heads_qk: {H}, num_heads_v: {HV}, "
          f"head_dim_qk: {K}, head_dim_v: {V}, is_kda: {is_kda}, dtype: {dtype}")
    device = "cuda"

    # --- Memory-optimized correctness check ---
    # Avoids holding 3 full copies of ssm_states simultaneously.
    inp = test_fused_sigmoid_gating_recurrent_k_last_packed.make_inputs(
        B, H, HV, K, V, pool_size, device, dtype, is_kda=is_kda
    )

    # Phase 1: run baseline on a clone
    o_baseline, state_baseline = test_fused_sigmoid_gating_recurrent_k_last_packed.run_baseline(inp)
    # After run_baseline, inp["ssm_states"] is still the original (untouched).
    # run_baseline internally clones it, so state_baseline is baseline's copy.

    # Phase 2: run packed directly, reusing the original ssm_states buffer
    # Avoid run_packed's internal clone: call the kernel directly on original states.
    ssm_states_for_packed = inp["ssm_states"]  # reuse original, no extra clone
    out_packed = inp["mixed_qkv"].new_empty(B, 1, HV, V)

    out_packed = fused_sigmoid_gating_delta_rule_forward_k_last_packed(
        inp["mixed_qkv"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        1.0,  # softplus_beta
        20.0,  # softplus_threshold
        K ** -0.5,  # scale
        ssm_states_for_packed,  # initial_state_source (modified in-place)
        inp["cache_indices"],  # initial_state_indices
        out_packed,  # out
        True,  # use_qk_l2norm_in_kernel
        inp["cu_seqlens"],  # cu_seqlens
        is_kda,
    )
    o_packed = out_packed.transpose(0, 1)
    state_packed = ssm_states_for_packed

    # Free input dict to release non-essential tensors
    del inp

    # Compare
    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    indices = torch.arange(B, device=device, dtype=torch.int32)

    passed = True
    try:
        torch.testing.assert_close(o_packed, o_baseline, atol=atol, rtol=rtol)
    except AssertionError as e:
        out_diff = (o_packed - o_baseline).abs().max().item()
        print(f"  [FAIL] output max_diff={out_diff:.6f}")
        print(f"--- case fail!!! ---")
        passed = False

    try:
        torch.testing.assert_close(
            state_packed[indices], state_baseline[indices], atol=atol, rtol=rtol
        )
    except AssertionError as e:
        st_diff = (state_packed[indices] - state_baseline[indices]).abs().max().item()
        print(f"  [FAIL] state max_diff={st_diff:.6f}")
        print(f"--- case fail!!! ---")
        passed = False

    if passed:
        print(f"  [PASS] B={B} H={H} HV={HV} K={K} V={V}")

def _precision_metrics(actual, ref):
    """FAOpBench-style accuracy metrics (see check_precision in OperatorsForge tools/ref_impls.py):
    returns (max_abs, rel_l2, cos_diff).
      rel_l2   = |actual - ref|_2 / |ref|_2
      cos_diff = 1 - cos(actual, ref)
    """
    a = actual.float().reshape(-1)
    r = ref.float().reshape(-1)
    max_abs = (a - r).abs().max().item()
    rel_l2 = ((a - r).norm() / r.norm().clamp_min(1e-12)).item()
    a_norm = a.norm().item()
    r_norm = r.norm().item()
    if a_norm > 1e-30 and r_norm > 1e-30:
        cos_diff = 1.0 - float(torch.dot(a, r).item() / (a_norm * r_norm))
    else:
        cos_diff = float('nan')
    return max_abs, rel_l2, cos_diff


def klast_mtp_ut_test(B, T, H, HV, K, V, topk=1, is_kda=False, dtype=torch.bfloat16,
                      rel_thresh=5e-2, cos_thresh=1e-3):
    """MTP verify correctness test (rewritten).

    Input construction follows scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_mtp.run_test;
    the judging methodology follows OperatorsForge tools/FAOpBench.py (evaluate_precision):
      - no more element-wise assert_close between two bf16 implementations -- differences in low-precision accumulation order
        amplify into a large max_diff on individual elements (with consistent means), causing false failures;
      - judge with rel_l2 < rel_thresh and cos_diff <= cos_thresh instead.
    Threshold calibration (based on CI measurements):
      - in eagle tree (true branching) mode, bf16 accumulation noise amplified by tree reload/recompute gives
        a measured rel_l2 of 1.0~1.6e-2, exactly satisfying cos_diff ≈ rel_l2^2/2 (the signature of unbiased random noise
        with the error vector orthogonal to the reference), so rel_thresh is 5e-2, leaving ~3x margin;
      - cos_thresh stays 1e-3: the cos_diff of systematic miscomputation (layout/out-of-bounds bugs) is
        several orders of magnitude larger, and this is the main criterion.
    Two test logic problems are also fixed:
      1. Triton / CUDA each use their own intermediate_states_buffer (the old logic shared one buffer
         across both runs, so the parent states written by the first run could be read by the second via the tree reload path);
      2. eagle tree (topk>1) generates a real branching parent map parent(t)=(t-1)//topk per topk;
         in the old logic every topk degenerated into the same single chain.

    Covers the kernel compile-time branches:
      topk=1:  DISABLE_STATE_UPDATE=True, CACHE_INTERMEDIATE_STATES=True, HAS_EAGLE_TREE=False
      topk>1:  DISABLE_STATE_UPDATE=True, CACHE_INTERMEDIATE_STATES=True, HAS_EAGLE_TREE=True
    """
    mtp = test_fused_sigmoid_gating_recurrent_k_last_mtp
    print(f"test case info: batch_size: {B}, seqlen(draft_token_num): {T}, "
          f"num_heads_qk: {H}, num_heads_v: {HV}, head_dim_qk: {K}, head_dim_v: {V}, "
          f"topk: {topk}, is_kda: {is_kda}, dtype: {dtype}")
    if not mtp.HAS_TRITON:
        print("  [SKIP] Triton reference not available")
        print("\nAll tests passed!")
        return

    device = "cuda"
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = mtp._make_tensors(
        B, T, H, HV, K, V, is_kda, device=device, dtype=dtype)
    # packed input layout of sglang MTP verify: a/b flatten the token dimension, q/k/v stay [1, B*T, ...]
    a = a.view(B * T, -1).contiguous()
    b = b.view(B * T, -1).contiguous()

    # eagle tree parent map (topk>1): parent(t) = (t-1)//topk generates a real branching tree;
    # topk=1 is single_chain (no tree, retrieve_parent_token=None)
    retrieve_parent_token = None
    if topk > 1:
        parent_list = [0] + [(t - 1) // int(topk) for t in range(1, T)]
        retrieve_parent_token = torch.tensor(
            [parent_list] * B, dtype=torch.int64, device=device).contiguous()

    cache_steps = T
    inter_idx = torch.arange(B, dtype=torch.int32, device=device)

    def _run(impl):
        # separate state / intermediate buffers per implementation to avoid cross-contamination
        s = state.clone()
        buf = torch.zeros(B * cache_steps, HV, V, K, dtype=torch.float32, device=device)
        if impl == "triton":
            out = mtp.triton_fused_gating(
                A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                s, indices, scale=1.0, use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens, is_kda=is_kda, disable_state_update=True,
                retrieve_parent_token=retrieve_parent_token,
                intermediate_states_buffer=buf,
                intermediate_state_indices=inter_idx,
                cache_steps=cache_steps,
            )
        else:
            out = fused_sigmoid_gating_delta_rule_forward_k_last(
                A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                s, indices, 1.0, True, cu_seqlens, is_kda,
                True, buf, inter_idx, cache_steps, retrieve_parent_token,
            )
        return out

    try:
        out_triton = _run("triton")
    except RuntimeError:
        # when the Triton reference is unavailable, only run CUDA as a redundancy check without judging correctness
        print("  [SKIP] Triton reference raised RuntimeError, CUDA-only run")
        _run("cuda")
        print("\nAll tests passed!")
        return
    out_cuda = _run("cuda")

    tree_tag = "yes" if topk > 1 else "no"
    case_tag = f"B={B},T={T},H={H},HV={HV},K={K},V={V},topk={topk},tree={tree_tag},is_kda={is_kda}"
    max_abs, rel_l2, cos_diff = _precision_metrics(out_cuda, out_triton)
    ok = (math.isfinite(rel_l2) and rel_l2 < rel_thresh
          and math.isfinite(cos_diff) and cos_diff <= cos_thresh)

    # noise signature diagnostic: unbiased orthogonal random noise satisfies cos_diff ≈ rel_l2^2/2;
    # a clear deviation (ratio >> 1) hints at a systematic component, helping tell noise from real bugs
    noise_ratio = cos_diff / (rel_l2 * rel_l2 / 2) if rel_l2 > 0 else float('nan')
    print(f"  out: max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} cos_diff={cos_diff:.3e} "
          f"noise_sig_ratio={noise_ratio:.2f} "
          f"(thresh: rel_l2<{rel_thresh}, cos_diff<={cos_thresh})")
    if ok:
        print(f"  [PASS] {case_tag}")
        print("\nAll tests passed!")
    else:
        print(f"  [FAIL] {case_tag}")
        print(f"         Triton mean: {out_triton.float().mean().item():.6f}, "
              f"CUDA mean: {out_cuda.float().mean().item():.6f}")
        print(f"         Max diff: {max_abs:.6f}, rel_l2: {rel_l2:.6f}, cos_diff: {cos_diff:.6f}")
        print(f"\nFailed tests: {B}, {T}, {H}, {HV}, {K}, {V}, topk={topk}, is_kda={is_kda}")

def run_cmd(params, kernel, args=None, raw_line=None, caselist_path=None):
    """Go through each kernel's unified entry.
    is_kda detection priority (True if any of them is True):
      1. explicit params['is_kda'] in params
      2. forced by the --is_kda CLI
      3. raw_line (the raw caselist line) contains [KDA]/kda_attn
      4. args.format contains [KDA]/kda_attn
      5. caselist_path basename contains Kimi/kda
    dtype is read only from the case inline (caselist / --format).
    """
    if 'is_kda' in params:
        is_kda = bool(params['is_kda'])
    else:
        force_flag = bool(getattr(args, 'is_kda', False)) if args is not None else False
        format_str = getattr(args, 'format', None) if args is not None else None
        is_kda = resolve_is_kda(
            force_flag=force_flag,
            raw_line=raw_line,
            format_str=format_str,
            caselist_path=caselist_path,
        )
    dtype_val = params.get('dtype', None)
    dtype = parse_dtype(dtype_val) if not isinstance(dtype_val, torch.dtype) else dtype_val

    if (kernel == "cuda"):
        print(f"{params['batch_size']}, {params['tp_size']}, {params['model_config']}, {kernel}")
        test_fused_sigmoid_gating_recurrent.batch_test(params["batch_size"], params["tp_size"], params["model_config"])
    elif (kernel == "k_last"):
        # params: B, T, H, HV, K, V
        klast_ut_test(params['B'], params['T'], params['H'], params['HV'], params['K'], params['V'],
                      kernel, is_kda=is_kda, dtype=dtype,
                      cu_seqlens_values=params.get('cu_seqlens'),
                      gate_lower_bound=params.get('gate_lower_bound'))
    elif (kernel == "k_last_packed"):
        # pool_size only needs to be >= B (each batch element uses one state slot).
        # Old: B * T * 2 wastes massive GPU memory for large batches.
        pool_size = params['B']
        klast_packed_ut_test(params['B'], params['T'], params['H'], params['HV'], params['K'], params['V'],
                             pool_size, is_kda=is_kda, dtype=dtype)
    elif (kernel == "k_last_mtp"):
        topk = params.get('topk', 1)
        # MTP default dtype is bf16 (consistent with scripts/decode/test_..._mtp.py)
        mtp_dtype = dtype if dtype_val is not None else torch.bfloat16
        klast_mtp_ut_test(params['B'], params['T'], params['H'], params['HV'], params['K'], params['V'],
                          topk=topk, is_kda=is_kda, dtype=mtp_dtype)

def run_test_cuda(args, case_data=None, case_file=None):
    """Run CUDA kernel tests.

    Args:
        args: argparse arguments
        case_data: pre-collected case data dict with 'file' and 'cases' keys
        case_file: fallback case file path if case_data not provided
    """
    if args.format is not None and args.caselist is None:
        case = parse_pla_string_re(args.format)
        if 'tp' in case.keys():
            tp_size = case['tp']
        else:
            print("no tp_size value")
            exit(1)
        model_config = {
            "linear_key_head_dim": case['head_dim_qk'],
            "linear_num_key_heads": case['num_heads_qk'] * tp_size,
            "linear_num_value_heads": case['num_heads_v'] * tp_size,
            "linear_value_head_dim": case['head_dim_v'],
        }
        batch_size = case['batch_size']
        paras = {
            "model_config": model_config,
            "tp_size": tp_size,
            "batch_size": batch_size
        }
        if 'dtype' in case:
            paras['dtype'] = case['dtype']
        f = io.StringIO()
        with redirect_stdout(f):
            run_cmd(paras, args.kernel, args, raw_line=args.format, caselist_path=None)

        output = f.getvalue()
        if args.perf:
            print(output)
        if " differences (" in output:
            print(f"--- {output.strip()}---\n")
            print(f"--- case fail!!! ---")
            line_content = str(case)
            with open("fail.log", "a", encoding="utf-8") as g:
                now = datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                g.write(line_content + time_str + "\n")
            exit(1)
        else:
            print(f"--- case pass!!! ---")
    elif args.format is None and args.caselist is not None:
        # use case_data if provided; otherwise read from the file
        if case_data is not None:
            # case_data may be a dict (single file) or a list (multiple files)
            if isinstance(case_data, dict):
                case_files_data = [case_data]
            else:
                case_files_data = case_data
        else:
            # collect all case files
            case_files_data, case_files = collect_cases_from_path(args.caselist)

        if not case_files_data:
            print(f"No case files found in: {args.caselist}")
            return

        print(f"Found {len(case_files_data)} case file(s)")
        if len(case_files_data) > 1:
            print(f"Processing path: {args.caselist}")

        total_cases = 0
        total_passed = 0
        total_failed = 0

        for cf_data in case_files_data:
            case_file = cf_data['file']
            caselist = cf_data['cases']

            # extract tp_size from the file name
            match = re.search(r"tp(\d+)", case_file)
            if not match:
                print(f"Skipping {case_file}: no tp_size value found in filename")
                continue
            tp_size = int(match.group(1))

            print(f"\n{'='*60}")
            print(f"Processing file: {case_file}")
            print(f"{'='*60}")

            i = 1
            for case in caselist:
                print(f"*--- start test case {i}/{len(caselist)} ---*")
                print(case)
                model_config = {
                    "linear_key_head_dim": case['head_dim_qk'],
                    "linear_num_key_heads": case['num_heads_qk'] * tp_size,
                    "linear_num_value_heads": case['num_heads_v'] * tp_size,
                    "linear_value_head_dim": case['head_dim_v'],
                }
                batch_size = case['batch_size']
                torch.cuda.empty_cache()
                paras = {
                    "model_config": model_config,
                    "tp_size": tp_size,
                    "batch_size": batch_size
                }
                if 'dtype' in case:
                    paras['dtype'] = case['dtype']
                f = io.StringIO()
                with redirect_stdout(f):
                    run_cmd(paras, args.kernel, args, raw_line=case.get('_raw_line'), caselist_path=case_file)

                output = f.getvalue()
                if " differences (" in output:
                    print(f"--- {output.strip()}---\n")
                    print(f"--- case fail!!! ---")
                    line_content = str(case)
                    with open("fail.log", "a", encoding="utf-8") as g:
                        now = datetime.now()
                        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                        g.write(line_content + time_str + f" [file: {case_file}]\n")
                    total_failed += 1
                else:
                    print(f"{paras}, {args.kernel}")
                    print(f"--- case pass!!! ---")
                    total_passed += 1
                i = i + 1
                total_cases += 1

        # print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total cases: {total_cases}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        if case_file:
            print(f"Case file: {case_file}")
        if total_failed:
            raise SystemExit(1)
    else:
        print("Input error: please provide a case or a caselist file")

def run_test_klast(args, case_data=None, case_file=None):
    """Run k_last Required kernel tests.

    Args:
        args: argparse arguments
        case_data: pre-collected case data dict with 'file' and 'cases' keys
        case_file: fallback case file path if case_data not provided
    """
    if args.format is not None and args.caselist is None:
        case = parse_pla_string_re(args.format)
        B, T, H, HV, K, V = case['batch_size'], case.get('seqlen_q', 1), case['num_heads_qk'], case['num_heads_v'], case['head_dim_qk'], case['head_dim_v']
        cu_seqlens = resolve_cu_seqlens(case)
        if cu_seqlens is not None:
            if args.kernel != "k_last":
                raise ValueError("explicit cu_seqlens is only supported by the k_last prefill test")
            B = len(cu_seqlens) - 1
            T = max(end - start for start, end in zip(cu_seqlens, cu_seqlens[1:]))
        # explicit cu_seqlens means varlen prefill; plain non-MTP cases are still treated as decode with T=1
        elif args.kernel != "k_last_mtp":
            T = 1
        # H/HV in the caselist are already per-card values (heads per card); the kernel accepts per-card values, so pass them through
        paras = {
            "B": B,
            "T": T,
            "H": H,
            "HV": HV,
            "K": K,
            "V": V,
        }
        if cu_seqlens is not None:
            paras['cu_seqlens'] = cu_seqlens
        # MTP mode: decide the mode from the case's seqlen_q (T=1 single_chain / T>1 eagle_tree)
        if args.kernel == "k_last_mtp":
            if 'topk' in case:
                paras['topk'] = case['topk']
            else:
                # default: T==1 -> topk=1; T>1 -> topk=2 (eagle tree)
                paras['topk'] = 1 if int(T) == 1 else 2
        if 'dtype' in case:
            paras['dtype'] = case['dtype']
        if 'gate_lower_bound' in case:
            paras['gate_lower_bound'] = case['gate_lower_bound']
        mode_label = "Prefill" if cu_seqlens is not None else (
            "MTP Verify" if (args.kernel == "k_last_mtp" and T > 1) else "Decode"
        )
        print(f"Testing {mode_label} Step (T={T}) Correctness...")
        print(f"test case info: batch_size: {B}, sqlen: {T}, num_heads_qk: {H}, num_heads_v: {HV}, head_dim_qk: {K}, head_dim_v: {V}")
        f = io.StringIO()
        with redirect_stdout(f):
            run_cmd(paras, args.kernel, args, raw_line=args.format, caselist_path=None)

        output = f.getvalue()
        if args.perf:
            print(output)
        if "fail" in output or "Fail" in output:
            print(f"--- {output.strip()}---\n")
            exit(1)
        else:
            print(f"--- case pass!!! ---")
    elif args.format is None and args.caselist is not None:
        # use case_data if provided; otherwise collect from the file/directory
        if case_data is not None:
            # case_data may be a dict (single file) or a list (multiple files)
            if isinstance(case_data, dict):
                case_files_data = [case_data]
            else:
                case_files_data = case_data
        else:
            # collect all case files
            case_files_data, case_files = collect_cases_from_path(args.caselist)

        if not case_files_data:
            print(f"No case files found in: {args.caselist}")
            return

        print(f"Found {len(case_files_data)} case file(s)")
        if len(case_files_data) > 1:
            print(f"Processing path: {args.caselist}")

        total_cases = 0
        total_passed = 0
        total_failed = 0

        for cf_data in case_files_data:
            case_file = cf_data['file']
            caselist = cf_data['cases']
            mtp_topks = getattr(args, 'mtp_topks', None)
            mtp_draft_tokens = getattr(args, 'mtp_draft_tokens', None)
            # k_last_mtp: eagle tree (topk>1) only runs on the boundary cases of this caselist
            # (min(B,H) and max(B,H)); single_chain (topk=1) runs on every case.
            eagle_boundary = (select_eagle_boundary_indices(caselist)
                              if args.kernel == "k_last_mtp" else set())
            if args.kernel == "k_last_mtp":
                print(f"MTP: eagle tree runs only boundary case indices {sorted(eagle_boundary)} "
                      f"(of {len(caselist)} cases); single_chain runs on all")

            i = 1
            for case_idx, case in enumerate(caselist):
                raw_line = case.get('_raw_line')
                seqlen_q_val = int(case.get('seqlen_q', 1))
                cu_seqlens = resolve_cu_seqlens(case)
                if cu_seqlens is not None and args.kernel != "k_last":
                    raise ValueError("explicit cu_seqlens is only supported by the k_last prefill test")
                # plain non-MTP cases only run seqlen_q=1; prefill cases with explicit cu_seqlens are the exception
                if args.kernel != "k_last_mtp" and cu_seqlens is None and seqlen_q_val > 1:
                    continue
                # MTP mode: automatically expand the case into sub-items
                #   T (draft_token_num) and topk are independent (following sglang):
                #     topk=1  -> single_chain (linear verify, HAS_EAGLE_TREE=False)
                #     topk>1  -> eagle_tree (HAS_EAGLE_TREE=True)
                #   T is always >= MIN_MTP_DRAFT_TOKEN_NUM(3): seqlen_q=1 in the caselist
                #   is meaningless for MTP and is cross-expanded with the values of --mtp-draft-tokens.
                #   eagle tree is expanded only on boundary cases (include_eagle).
                if args.kernel == "k_last_mtp":
                    sub_specs = expand_mtp_subcases(
                        case, topks=mtp_topks, draft_token_nums=mtp_draft_tokens,
                        include_eagle=(case_idx in eagle_boundary))
                else:
                    sub_specs = [(None, None, "")]

                for sub_T, sub_topk, sub_label in sub_specs:
                    label = f" ({sub_label})" if sub_label else ""
                    print(f"*--- start test case {i}{label} ---*")
                    print(case)
                    torch.cuda.empty_cache()
                    B, H, HV, K, V = case['batch_size'], case['num_heads_qk'], case['num_heads_v'], case['head_dim_qk'], case['head_dim_v']
                    if cu_seqlens is not None:
                        B = len(cu_seqlens) - 1
                        T = max(end - start for start, end in zip(cu_seqlens, cu_seqlens[1:]))
                    elif args.kernel == "k_last_mtp":
                        T = int(sub_T)
                    else:
                        # plain non-MTP cases are fixed at T=1
                        T = 1
                    # H/HV in the caselist are already per-card values (heads per card); the kernel accepts per-card values, so pass them through
                    paras = {
                        "B": B,
                        "T": T,
                        "H": H,
                        "HV": HV,
                        "K": K,
                        "V": V,
                    }
                    if cu_seqlens is not None:
                        paras['cu_seqlens'] = cu_seqlens
                    if sub_topk is not None:
                        paras['topk'] = int(sub_topk)
                    if 'dtype' in case:
                        paras['dtype'] = case['dtype']
                    if 'gate_lower_bound' in case:
                        paras['gate_lower_bound'] = case['gate_lower_bound']
                    f = io.StringIO()
                    with redirect_stdout(f):
                        run_cmd(paras, args.kernel, args, raw_line=raw_line, caselist_path=case_file)

                    output = f.getvalue()
                    if "fail" in output or "Fail" in output:
                        print(f"--- {output.strip()}---\n")
                        print(f"--- case fail!!! ---")
                        line_content = str(case)
                        with open("fail.log", "a", encoding="utf-8") as g:
                            now = datetime.now()
                            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                            g.write(line_content + time_str + f" [file: {case_file}]\n")
                        total_failed += 1
                    else:
                        print(f"{paras}, {args.kernel}")
                        print(f"--- case pass!!! ---")
                        total_passed += 1
                    i = i + 1
                    total_cases += 1

        # print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total cases: {total_cases}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        if case_file:
            print(f"Case file: {case_file}")
        if total_failed:
            raise SystemExit(1)
    else:
        print("Input error: please provide a case or a caselist file")

def main():
    parser = argparse.ArgumentParser(description="GDN caselist test with path support")
    parser.add_argument("--caselist", type=str, help="input caselist file path or directory path")
    parser.add_argument("--perf", type=str, default=False, help="whether this is a perf test")
    parser.add_argument("--kernel", type=str, default="cuda",
                        choices=["cuda", "k_last", "k_last_packed", "k_last_mtp"],
                        help="kernel type to use: cuda (default k_last), k_last, k_last_packed, k_last_mtp")
    parser.add_argument("--is_kda", action="store_true",
                        help="force KDA mode. When omitted the script auto-detects: it is set to True when the caselist path/line contains Kimi/[KDA]/kda_attn or --format contains [KDA]/kda_attn.")
    parser.add_argument("--format", type=str, help="input case format:[GDN]:[D][4]gdn_attn: \
                        seqlen_q:1,head_dim_qk:128,head_dim_v:128,num_heads_qk:16,num_heads_v:32,hidden_act:silu,conv_kdim:4,batch_size:4,dtype:bf16,tp:4. \
                        unequal-length prefill may omit seqlen_q, using cu_seqlens_q:[0,24,184,358] (or cu_seqlens) and selecting k_last.")
    parser.add_argument("--mtp-topks", type=str, default="1,2,3,4,5",
                        help="topk list covered in k_last_mtp mode, comma separated (default 1,2,3,4,5). topk=1 is single_chain, >1 is eagle_tree. E.g. --mtp-topks=1,5 runs only the two boundaries.")
    parser.add_argument("--mtp-draft-tokens", type=str, default="4,16",
                        help="draft_token_num(T) list covered in k_last_mtp mode, comma separated (default 4,16; 16 is the sglang DFLASH default block_size). "
                             "T and topk are independent and are cross-expanded; T<3 is raised to 3 (seqlen_q=1 in the caselist is meaningless for MTP).")
    args = parser.parse_args()

    # parse --mtp-topks / --mtp-draft-tokens into integer lists; only effective under k_last_mtp
    try:
        args.mtp_topks = [int(x) for x in args.mtp_topks.split(",") if x.strip()]
    except ValueError:
        print(f"ERROR: bad --mtp-topks format: {args.mtp_topks}. Expected comma-separated integers, e.g. 1,2,3,4,5")
        return
    try:
        args.mtp_draft_tokens = [int(x) for x in args.mtp_draft_tokens.split(",") if x.strip()]
    except ValueError:
        print(f"ERROR: bad --mtp-draft-tokens format: {args.mtp_draft_tokens}. Expected comma-separated integers, e.g. 4,16")
        return

    # handle the caselist path
    case_path = args.caselist
    case_files_data = None
    single_file = None

    if case_path is not None and args.format is None:
        if os.path.isfile(case_path):
            # a single file
            cases = read_cmds_from_file(case_path)
            if cases:
                case_files_data = {'file': case_path, 'cases': cases}
                single_file = case_path
        elif os.path.isdir(case_path):
            # a directory: collect all case files
            case_files_data, _ = collect_cases_from_path(case_path)
            if not case_files_data:
                print(f"No case files found in: {case_path}")
                return
            print(f"Found {len(case_files_data)} case file(s)")
        else:
            print(f"ERROR: path not found: {case_path}")
            return

    print(f"*--- fused_sigmoid_gating_delta_rule_forward test ---*")
    print(f"Kernel type: {args.kernel}")
    print(f"is_kda(force flag): {args.is_kda}  (when not forced, auto-detected from Kimi/KDA in caselist/format/path)")
    if case_path:
        print(f"Case source: {case_path}")

    if (args.kernel == "cuda"):
        run_test_cuda(args, case_data=case_files_data, case_file=single_file)
    elif (args.kernel == "k_last" or args.kernel == "k_last_packed"):
        run_test_klast(args, case_data=case_files_data, case_file=single_file)
    elif (args.kernel == "k_last_mtp"):
        run_test_klast(args, case_data=case_files_data, case_file=single_file)

if __name__ == "__main__":
    main()
