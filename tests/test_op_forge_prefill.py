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

"""Prefill caselist test runner for PLA GDN kernels.

Unlike decode, prefill goes through the standalone CuTe prefill kernel
``pla.prefill.flashqla.chunk_gated_delta_rule_fwd``, **not** the decode
k_last kernel. This script used to mis-route to test_op_forge.run_test_klast (decode
k_last), triggering `too many resources requested for launch` at T>=158 -- that is the
launch limit of the decode kernel and has nothing to do with the real prefill kernel.

This script calls tests/prefill_utils.py in the same directory (without modifying anything under scripts/):
it internally calls the real prefill kernel chunk_gated_delta_rule_fwd and uses the vendored
FlashQLA torch reference (ref_gdr) as the golden for cos/rel comparison. Caselist lines are
parsed by parse_pla_string_re into (cu_seqlens, H=num_heads_v, Hg=num_heads_qk, dtype);
single segments go through run_single, multi-segment packed ones through run_packed.

Example:
    python tests/test_op_forge_prefill.py \
        --caselist caselist/prefill/Qwen3.7-Max_prefill_tp2_..._gdn_op.list
"""

import argparse
import os
import re
import sys

# Ensure tests dir, scripts dir, and the prefill script dir are importable.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
_SCRIPTS_DIR = os.path.join(_TESTS_DIR, "..", "scripts")
if _SCRIPTS_DIR not in sys.path and os.path.isdir(_SCRIPTS_DIR):
    sys.path.insert(0, _SCRIPTS_DIR)

import torch  # noqa: E402  (must precede the extension import chain)

from utils import (  # noqa: E402
    collect_cases_from_path,
    read_cmds_from_file,
    resolve_cu_seqlens,
)

# Prefill validation tools in this directory (nothing under scripts/ is modified):
# they internally call the real prefill kernel chunk_gated_delta_rule_fwd + the ref_gdr golden.
import prefill_utils  # noqa: E402

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
}

# The caselist only describes the activation dtype; the recurrent state pool dtype is an independent dimension,
# chosen by --state-dtype: fp32 is the historical contract, bf16 halves the resident state memory.
_STATE_DTYPE_CHOICES = {
    "fp32": [torch.float32],
    "bf16": [torch.bfloat16],
    "both": [torch.float32, torch.bfloat16],
}


def _state_tag(state_dtype):
    return "fp32" if state_dtype == torch.float32 else "bf16"


def _filter_prefill_files(case_files_data, model=None, tp=None):
    """Keep only prefill caselist files, optionally filtering by model/tp."""
    filtered = []
    for cf_data in case_files_data:
        base_lower = os.path.basename(cf_data["file"]).lower()
        if "prefill" not in base_lower:
            continue
        if model is not None and model.lower() not in base_lower:
            continue
        if tp is not None:
            match = re.search(r"[\W_]?tp(\d+)[\W_]", base_lower)
            if not match or int(match.group(1)) != int(tp):
                continue
        filtered.append(cf_data)
    return filtered


def _case_to_e2e_params(case):
    """Map an already-parsed case dict into run_e2e(seq_len, H, Hg, dtype) kwargs.

    NOTE: read_cmds_from_file / collect_cases_from_path already return dicts parsed by
    parse_pla_string_re (with '_raw_line'); do not parse them again.

    Mapping (OpForge naming -> run_e2e / kernel naming):
      num_heads_v  -> H   (value heads)
      num_heads_qk -> Hg  (grouped query/key heads, GVA)
      cu_seqlens_q:[0, T] -> seq_len = T (single-sequence prefill)
      head_dim_qk / head_dim_v must equal 128 (kernel is specialised to DK=DV=128)
    Returns (params_dict, err_str). On any problem err_str is set and params is None.
    """
    if not isinstance(case, dict):
        return None, f"expected parsed case dict, got {type(case).__name__}"

    cu = resolve_cu_seqlens(case)
    if cu is None:
        return None, "no cu_seqlens (not a varlen prefill case)"
    # Single segment [0,T] -> run_e2e; multi-segment packed [0,t1,t2,...] -> multi_sequence.run_case.
    # lengths = adjacent offset differences (per-sequence lengths); the kernel and ref_gdr both natively support multi-segment cu_seqlens.
    lengths = [int(cu[i + 1] - cu[i]) for i in range(len(cu) - 1)]
    seq_len = lengths[0] if len(lengths) == 1 else None

    H = case.get("num_heads_v")
    Hg = case.get("num_heads_qk")
    if H is None or Hg is None:
        return None, f"missing num_heads_v/num_heads_qk in {case}"

    dk = int(case.get("head_dim_qk", 128))
    dv = int(case.get("head_dim_v", 128))
    if dk != prefill_utils.DK or dv != prefill_utils.DV:
        return None, (f"head_dim ({dk},{dv}) != kernel ({prefill_utils.DK},"
                      f"{prefill_utils.DV}); prefill kernel is specialised to 128")

    dtype_s = str(case.get("dtype", "bf16")).lower()
    dtype = _DTYPE_MAP.get(dtype_s)
    if dtype is None:
        return None, f"unsupported dtype: {dtype_s}"

    return dict(seq_len=seq_len, lengths=lengths, H=int(H), Hg=int(Hg),
                dtype=dtype), None


def _run_one(case, state_dtypes, backend="pla", check=True):
    """Run a single prefill case (already-parsed dict); print pass/fail markers.

    ``state_dtypes`` is the list of recurrent state pool dtypes to run; each dtype is a
    separate kernel call, and the case passes only if all of them pass.
    """
    params, err = _case_to_e2e_params(case)
    if err is not None:
        # Not a runnable prefill case for this kernel: mark fail with reason so the
        # parser records it (and does not silently drop the case).
        print(f"skip/err: {err}")
        print(f"--- case fail!!! ---")
        return False

    lengths = params["lengths"]
    multi = len(lengths) > 1
    print(f"test case info: "
          f"{'lengths' if multi else 'seq_len(T)'}: "
          f"{lengths if multi else params['seq_len']}, "
          f"num_heads_v(H): {params['H']}, num_heads_qk(Hg): {params['Hg']}, "
          f"dtype: {params['dtype']}, "
          f"state: {'/'.join(_state_tag(d) for d in state_dtypes)}")
    ok = True
    for state_dtype in state_dtypes:
        try:
            if multi:
                # Multi-sequence packed prefill: call the same prefill kernel with the full multi-segment cu_seqlens.
                sub_ok = prefill_utils.run_packed(
                    lengths, params["H"], params["Hg"], dtype=params["dtype"],
                    backend=backend, check=check, state_dtype=state_dtype)
            else:
                sub_ok = prefill_utils.run_single(
                    params["seq_len"], params["H"], params["Hg"],
                    dtype=params["dtype"], backend=backend, check=check,
                    state_dtype=state_dtype)
        except Exception as e:  # noqa: BLE001 — one case must not kill the whole run
            import traceback
            traceback.print_exc()
            sub_ok = False
        ok = ok and bool(sub_ok)

    if ok:
        print(f"--- case pass!!! ---")
    else:
        print(f"--- case fail!!! ---")
    return bool(ok)


def main():
    parser = argparse.ArgumentParser(
        description="Run PLA prefill caselist tests (GDN, real prefill kernel).")
    parser.add_argument("--caselist", type=str, required=True,
                        help="Path to a prefill caselist .list file or a directory.")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional model name substring filter, e.g. Qwen3.7-Max.")
    parser.add_argument("--tp", type=int, default=None,
                        help="Optional tensor-parallel size filter, e.g. 1/2/4/8.")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Run at most N cases per file (quick smoke test).")
    parser.add_argument("--state-dtype", choices=sorted(_STATE_DTYPE_CHOICES),
                        default="fp32",
                        help="Recurrent state pool dtype: fp32 (default), bf16, "
                             "or both (each case runs once per dtype).")
    # Accepted for CLI compatibility with the old wrapper; prefill is GDN-only here.
    parser.add_argument("--is-kda", action="store_true",
                        help="(unused) FlashQLA prefill path is GDN; kept for CLI compat.")
    parser.add_argument("--perf", action="store_true",
                        help="perf collection mode: only launch the kernel, no golden accuracy (the golden is a torch float "
                             "reference implementation, very slow for long sequences). With env TRITON_PLA=1 it runs the sglang "
                             "triton implementation instead, collecting baseline cycles for the perf comparison.")
    cli_args = parser.parse_args()

    # perf collection runs two rounds: TRITON_PLA=1 runs the triton baseline, unset/=0 runs the pla CuTe kernel.
    # Keep the switch semantics consistent with the decode-side test_op_forge_perf.py.
    backend = "triton" if int(os.environ.get("TRITON_PLA", 0)) != 0 else "pla"
    check = not cli_args.perf
    if cli_args.perf:
        print(f"PERF MODE: backend={backend}, golden check disabled")

    if os.path.isfile(cli_args.caselist):
        cases = read_cmds_from_file(cli_args.caselist)
        if not cases:
            print(f"No cases found in: {cli_args.caselist}")
            return
        case_files_data = [{"file": cli_args.caselist, "cases": cases}]
    elif os.path.isdir(cli_args.caselist):
        case_files_data, _ = collect_cases_from_path(cli_args.caselist)
        if not case_files_data:
            print(f"No caselist files found in: {cli_args.caselist}")
            return
    else:
        print(f"ERROR: path not found: {cli_args.caselist}")
        return

    case_files_data = _filter_prefill_files(
        case_files_data, model=cli_args.model, tp=cli_args.tp)
    if not case_files_data:
        print("No matching prefill caselist files after filtering.")
        return

    print(f"Found {len(case_files_data)} prefill caselist file(s) to run:")
    for cf_data in case_files_data:
        print(f"  - {cf_data['file']} ({len(cf_data['cases'])} cases)")

    if cli_args.max_cases is not None:
        for cf_data in case_files_data:
            n0 = len(cf_data["cases"])
            cf_data["cases"] = cf_data["cases"][: cli_args.max_cases]
            print(f"  [max-cases] {cf_data['file']}: {len(cf_data['cases'])}/{n0}")

    state_dtypes = _STATE_DTYPE_CHOICES[cli_args.state_dtype]
    print(f"\n*--- FlashQLA prefill test (chunk_gated_delta_rule_fwd) ---*")
    print(f"Case source: {cli_args.caselist}")
    print(f"State dtype: {cli_args.state_dtype}")

    total = passed = failed = 0
    for cf_data in case_files_data:
        case_file = cf_data["file"]
        for i, case in enumerate(cf_data["cases"], 1):
            # case is already a dict parsed by parse_pla_string_re (with '_raw_line')
            raw_line = case.get("_raw_line", str(case)) if isinstance(case, dict) else str(case)
            print(f"\n*--- start test case {i} ---*")
            print(raw_line)
            ok = _run_one(case, state_dtypes, backend=backend, check=check)
            total += 1
            if ok:
                passed += 1
            else:
                failed += 1
                with open("fail.log", "a", encoding="utf-8") as g:
                    from datetime import datetime
                    g.write(f"{raw_line} {datetime.now():%Y-%m-%d %H:%M:%S} "
                            f"[file: {case_file}]\n")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Total cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    # A failing sub-case must exit non-zero: the case pipeline only has an ExitCodeChecker,
    # so status depends entirely on the exit code. Previously main fell through returning None (exit code always 0), so
    # however many of the 1344 sub-cases failed, the case level still reported PASSED and the plan reported failed: 0,
    # with failures visible only in the failures count of the merged junit (run 67817843: 379 fail
    # yet XTEST_EXIT=0). The pattern follows decode-side test_op_forge.py:528/714.
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
