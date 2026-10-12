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

"""KDA decode mega-fusion correctness test (xtest entry) -- caselist driven + per-sub-case report.

Path under test: ``pla.decode.kda.fused_kda_decode_mega_forward``
(conv1d_update + packed KDA recurrence + gated RMSNorm fused into a single launch).

## Relation to scripts/decode/kda_fusion/compare_kda_decode_fusion.py

The reference implementation, tolerances and input construction are **all reused** from that script, not rewritten here:

    build_inputs / vllm_reference / torch_reference_fp16 / run_kernel
    VLLM_OUT_ATOL=1e-2  VLLM_OUT_RTOL=1e-2
    VLLM_STATE_ATOL=1e-3 VLLM_STATE_RTOL=1e-3  conv bit-exact

Rationale: those criteria were checked field by field against vllm ``tests/models/kimi_k3/test_kda.py``;
a copy would inevitably drift into two standards over time. This file is only responsible for "which shapes to run" and
"how to report the results to xtest".

## Which shapes to run (following scripts/prefill/flashkdapro/e2e_kimi_k3.py)

Instead of the hand-written K3_CASES table in the script, the real serving batch distribution is taken
from OpForge's decode caselist:

  * batch -- parse every ``batch_size`` from ``--caselist``, take ``--max-batches``
    representative values by quantile (always including min and max), ensuring at least one in each of the <200 and >=200 tiers: these two tiers are
    exactly the perf threshold boundary, and functional coverage follows them so a perf regression never lands on a shape that was never tested.
  * head -- **not** taken from the caselist ``num_heads_qk``. The mega kernel entry supports
    H in {6,12,24,48,96} (the TORCH_CHECK in src/decode/k-last/fused_kda_decode_mega.cu),
    while the Kimi-Linear-48B caselist is tp1=32 / tp2=16; feeding those in would only hit the TORCH_CHECK
    "H must be 6, 12, 24, 48, or 96", which is a caselist/kernel support-set mismatch, not a regression.
    So the heads are the support set itself (which matches the per-card head counts of Kimi-K3 tp1/2/4/8/16).
  * gate mode -- both (lower_bound=-5.0, fuse_output_norm=True) and (None, False)
    are run: the former uses the lower-bound sigmoid gate + fused RMSNorm, the latter the softplus gate
    without fused norm; they are two different branches in the kernel.

Note that head and batch form the Cartesian product of two independent axes, creating large combinations
that do not exist in the caselist. One recurrent state is B*H*128*128*4 bytes; the largest with default parameters, H96xB457,
is 2.7 GiB, still within budget together with the triton reference chain's intermediates; but raising ``--max-batches``
or switching to a caselist with larger batches would exceed it. So combinations first pass a ``--max-state-gib``
budget gate **at generation time** (instead of failing at OOM -- that is a test budget
problem, not a kernel regression); filtered combinations are listed in the header.

## Output

Strictly follows the tests/case_report.py protocol, printing ``*--- start test case i/N (label) ---*`` and
``--- case pass!!! ---``/``--- case fail!!! ---``; acompute.PLAResultParser turns each
(H, B, gate) combination into one ES record and one JUnit testcase row.

Usage (inside the container, from the repo root; must be run as a plain script, otherwise the source tree's pla/ shadows the installed package):

    python tests/test_kda_decode_mega.py
    python tests/test_kda_decode_mega.py --caselist caselist/decode/Kimi-Linear-48B-A3B_decode_tp2_kvlimit_200GB_model_len_128K_kda_op.list
    python tests/test_kda_decode_mega.py --dtype fp16 --max-batches 2
"""
import argparse
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
# NOTE: do not add the repo root to sys.path -- the pla/ in the source tree has no compiled _C
# and would shadow the installed pla package entirely. Only add tests/ and the needed scripts subdirectories.
for _p in (_TESTS_DIR,
           os.path.join(_REPO_ROOT, "scripts", "decode"),
           os.path.join(_REPO_ROOT, "scripts", "decode", "kda_fusion")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from case_report import CaseReporter  # noqa: E402
from utils import read_cmds_from_file  # noqa: E402

# Reference implementation / tolerances / input construction all come from here, see the file header
import compare_kda_decode_fusion as ref  # noqa: E402

DEFAULT_CASELIST = ("caselist/decode/"
                    "Kimi-Linear-48B-A3B_decode_tp1_kvlimit_200GB_model_len_128K_kda_op.list")
# Support set of the mega kernel, equal to the per-card head counts of Kimi-K3 tp1/tp2/tp4/tp8/tp16
SUPPORTED_HEADS = (6, 12, 24, 48, 96)
# Tier boundary of the perf threshold: batch<200 tolerates 5%, >=200 tolerates 2%
BATCH_SPLIT = 200
# (lower_bound, fuse_output_norm)
GATE_MODES = ((-5.0, True), (None, False))
# Bytes of one recurrent state = B * H * head_dim * head_dim * sizeof(fp32)
_STATE_BYTES_PER_BH = 128 * 128 * 4


def state_gib(H, B):
    return H * B * _STATE_BYTES_PER_BH / (1 << 30)


def collect_batches(caselist_path, k):
    """Take representative batch_size values from the caselist: quantile sampling + at least one per threshold tier."""
    cases = read_cmds_from_file(caselist_path)
    values = sorted({int(c["batch_size"]) for c in cases if "batch_size" in c})
    if not values:
        raise ValueError(f"no batch_size parsed from the caselist: {caselist_path}")
    if k >= len(values):
        return values
    # take k evenly spaced values (including both ends)
    picks = {values[round(i * (len(values) - 1) / (k - 1))] for i in range(k)} if k > 1 \
        else {values[-1]}
    small = [v for v in values if v < BATCH_SPLIT]
    large = [v for v in values if v >= BATCH_SPLIT]
    if small and not any(v < BATCH_SPLIT for v in picks):
        picks.add(small[len(small) // 2])
    if large and not any(v >= BATCH_SPLIT for v in picks):
        picks.add(large[len(large) // 2])
    return sorted(picks)


def run_one(H, B, lower_bound, fuse_output_norm, use_fp16):
    """Run one (H, B, gate) combination and return (ok, list of info lines, failure summary)."""
    out_dtype = torch.float16 if use_fp16 else torch.bfloat16
    if use_fp16:
        out_atol = state_atol = ref.FP16_ATOL
        out_rtol = state_rtol = ref.FP16_RTOL
        conv_judge = ref.fp16_allclose          # conv allows fp16 rounding
    else:
        out_atol, out_rtol = ref.VLLM_OUT_ATOL, ref.VLLM_OUT_RTOL
        state_atol, state_rtol = ref.VLLM_STATE_ATOL, ref.VLLM_STATE_RTOL
        conv_judge = ref.conv_exact             # conv bit-exact

    inp = ref.build_inputs(H, B, seed=967 + H + B)
    if use_fp16:
        inp = ref._to_fp16(inp)
        expected_out, ref_state, ref_conv = ref.torch_reference_fp16(
            inp, lower_bound, fuse_output_norm)
    else:
        expected_out, ref_state, ref_conv = ref.vllm_reference(
            inp, lower_bound, fuse_output_norm)

    ppu_out, ppu_state, ppu_conv = ref.run_kernel(
        ref.fused_kda_decode_ppu_forward, inp, lower_bound, fuse_output_norm,
        out_dtype=out_dtype)

    o_abs, o_rel = ref.diff(ppu_out, expected_out)
    s_abs, s_rel = ref.diff(ppu_state, ref_state)
    c_abs, _ = ref.diff(ppu_conv, ref_conv)

    judge_out = ref._allclose(ppu_out, expected_out, atol=out_atol, rtol=out_rtol)
    judge_state = ref._allclose(ppu_state, ref_state, atol=state_atol, rtol=state_rtol)
    judge_conv = conv_judge(ppu_conv, ref_conv)
    ok = judge_out and judge_state and judge_conv

    lines = [
        # batch_size: this format lets the parser's _extract_params put B into the sub-case name
        f"batch_size:{B}, num_heads:{H}, lower_bound:{lower_bound}, "
        f"fuse_output_norm:{fuse_output_norm}, dtype:{'fp16' if use_fp16 else 'bf16'}",
        f"Max diff out abs={o_abs:.3e} rel={o_rel:.3e} "
        f"(atol={out_atol:.0e} rtol={out_rtol:.0e}) -> {judge_out}",
        f"Max diff state abs={s_abs:.3e} rel={s_rel:.3e} "
        f"(atol={state_atol:.0e} rtol={state_rtol:.0e}) -> {judge_state}",
        f"Max diff conv abs={c_abs:.3e} "
        f"({'exact' if not use_fp16 else 'fp16 tol'}) -> {judge_conv}",
    ]
    detail = ""
    if not ok:
        detail = (f"H={H} B={B} out_ok={judge_out} state_ok={judge_state} "
                  f"conv_ok={judge_conv} o_abs={o_abs:.3e} s_abs={s_abs:.3e} "
                  f"c_abs={c_abs:.3e}")
    return ok, lines, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--caselist", default=DEFAULT_CASELIST,
                    help="decode KDA caselist used for the batch distribution (relative to the repo root)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                    help="bf16 uses the inlined vllm triton reference chain (default); "
                         "fp16 uses the torch-naive f32 golden -- a per-channel python loop, "
                         "orders of magnitude slower; use it with --max-batches 1 --heads 12")
    ap.add_argument("--max-batches", type=int, default=6,
                    help="how many batch representative values to take from the caselist (default 6)")
    ap.add_argument("--heads", default=",".join(str(h) for h in SUPPORTED_HEADS),
                    help=f"heads per card; must be a subset of {SUPPORTED_HEADS}")
    ap.add_argument("--max-state-gib", type=float, default=6.0,
                    help="recurrent state budget (GiB) of a single (H,B) combination; combinations over it are "
                         "not generated. head and batch form a Cartesian product, creating large combinations "
                         "absent from the caselist; the largest with default parameters is 2.7 GiB and is not filtered, "
                         "this gate is a safeguard for raising --max-batches or switching to a larger caselist "
                         "(default 6.0)")
    args = ap.parse_args()

    caselist_path = args.caselist if os.path.isabs(args.caselist) \
        else os.path.join(_REPO_ROOT, args.caselist)
    if not os.path.isfile(caselist_path):
        print(f"ERROR: caselist does not exist: {caselist_path}", file=sys.stderr)
        return 2

    heads = [int(h) for h in args.heads.split(",") if h.strip()]
    bad = [h for h in heads if h not in SUPPORTED_HEADS]
    if bad:
        print(f"ERROR: the mega kernel entry supports H={SUPPORTED_HEADS}, not {bad}",
              file=sys.stderr)
        return 2

    batches = collect_batches(caselist_path, args.max_batches)
    use_fp16 = args.dtype == "fp16"
    combos, skipped = [], []
    for H in heads:
        for B in batches:
            gib = state_gib(H, B)
            if gib > args.max_state_gib:
                skipped.append((H, B, gib))
                continue
            for (lb, fon) in GATE_MODES:
                combos.append((H, B, lb, fon))
    if not combos:
        print(f"ERROR: every (H,B) combination exceeds --max-state-gib={args.max_state_gib}",
              file=sys.stderr)
        return 2

    rep = CaseReporter("kda_decode_mega", caselist_path, total=len(combos))
    rep.header()
    ref_desc = ("torch-naive f32 golden (fp16 cast)" if use_fp16 else
                "inline vllm triton chain (conv1d_update -> packed KDA recurrence)")
    print(f"ref: {ref_desc}")
    print(f"batches from caselist: {batches}")
    print(f"heads: {heads}   gate modes: {list(GATE_MODES)}")
    if skipped:
        # not a fail: the state of these combinations does not fit, and they are not in the caselist anyway
        print(f"skipped {len(skipped)} (H,B) combos over "
              f"--max-state-gib={args.max_state_gib}: "
              + ", ".join(f"H{h}xB{b}={g:.1f}GiB" for h, b, g in skipped))

    for H, B, lb, fon in combos:
        gate = "lb5_norm" if fon else "softplus"
        rep.start(f"H{H}_B{B}_{gate}")
        if rep.aborted:
            rep.fail("skipped: an earlier case already brought the device down")
            continue
        with rep.guard():
            torch.cuda.empty_cache()
            ok, lines, detail = run_one(H, B, lb, fon, use_fp16)
            for line in lines:
                rep.info(line)
            rep.ok() if ok else rep.fail(detail)

    return rep.summary()


if __name__ == "__main__":
    sys.exit(main())
