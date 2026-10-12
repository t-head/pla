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

"""FlashKDA prefill correctness test (xtest entry) -- full dispatch paths + real caselist shapes.

Path under test: ``pla.prefill.flashkdapro.flashkda_fwd`` (the CuTe kernel chain of Kimi-K3 KDA
prefill). The oracle is the fp32 torch reference ``ref_fwd`` in ``scripts/prefill/flashkdapro/private/test_fwd_e2e.py``,
and the criterion is ``err_ratio < TOL``.

## Relation to the two manual scripts (reused only, never rewritten)

  * ``scripts/prefill/flashkdapro/e2e_acc_full.py``
    provides the dispatch path matrix ``CASES`` and the single-case body ``check()``: initial_state /
    final_state dtype combinations, the bit-identical contract of ``final_state=None``, chunk tails that are not a
    multiple of 16, and all-zero padded tails. The K2 variants of flashkda_fwd are **selected by shape**
    (dvsplit BV16/BV32, full-DV nows, varlen nows, baseline DVT32); this table is the shortest set of shapes
    that hits every variant once.
  * ``scripts/prefill/flashkdapro/e2e_kimi_k3.py``
    provides caselist parsing and representative shape selection: ``parse_list()`` takes
    ``num_heads_qk`` / ``gate_lower_bound`` / ``cu_seqlens_q`` from the ``.list``, ``pick_acc_cases()``
    keeps only small cases where every sequence is <= max_t (the fp32 reference is a per-chunk python loop, long cases
    never finish inside the accuracy gate), and ``pro_fast()`` replicates the use_nows gating of kernel2_recurrence.cu
    so the report shows whether a shape hits the pro fast path or the baseline.

## Two stages

  * ``dispatch``: runs the path matrix of e2e_acc_full, making sure every K2 variant and state contract is hit.
  * ``caselist``: runs the real shapes of ``caselist/prefill/Kimi-K3_prefill_tp{1,2,4,8}_*kda_op.list``
    (tp1/2/4/8 map to per-card H=96/48/24/12), taking ``--per-tp`` of them per tp.

## Output

Markers are printed per sub-case following the tests/case_report.py protocol; acompute.PLAResultParser turns each
shape into one ES record and one JUnit testcase row; the manual scripts' original prints are kept as info lines
inside the sub-case, and go straight into the ES detail field on failure.

Usage (inside the container, from the repo root; must be run as a plain script, otherwise the source tree's pla/ shadows the installed package):

    python tests/test_kda_prefill_flashkdapro.py
    python tests/test_kda_prefill_flashkdapro.py --phase caselist --tp 8,4 --per-tp 4
    python tests/test_kda_prefill_flashkdapro.py --acc-max-t 512
"""
import argparse
import glob
import io
import os
import sys
from contextlib import redirect_stdout

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
# Same as other pla test scripts: do not add the repo root to sys.path; the source tree's pla/ has no build artifacts
# and would shadow the installed package entirely.
_PRO_DIR = os.path.join(_REPO_ROOT, "scripts", "prefill", "flashkdapro")
for _p in (_TESTS_DIR, _PRO_DIR, os.path.join(_PRO_DIR, "private")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from case_report import CaseReporter  # noqa: E402

import e2e_acc_full as acc  # noqa: E402  dispatch matrix + check()
import e2e_kimi_k3 as k3  # noqa: E402   caselist parsing + representative shape selection

DEFAULT_LIST_DIR = "caselist/prefill"
DEFAULT_TPS = (1, 2, 4, 8)

# (h0 dtype, list of final_state dtypes to examine) -- semantics in the docstring of e2e_acc_full.check:
# a zero initial state allows final_state to be bf16 or fp32; with an initial state upstream requires both dtypes to match.
H0_MODES = (
    (None, (torch.bfloat16, torch.float32)),
    (torch.bfloat16, (torch.bfloat16,)),
    (torch.float32, (torch.float32,)),
)


def _dt_tag(dtype):
    if dtype is None:
        return "zero"
    return "f32" if dtype == torch.float32 else "bf16"


def _run_captured(fn, *args, **kwargs):
    """Run the reference script's body once and capture its prints as info lines.

    The manual scripts "print as they go"; calling them directly mixes their lines between the case markers, and the
    numbers on failure never reach the detail field. Capturing and re-emitting keeps the original info and allows building detail.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        ret = fn(*args, **kwargs)
    return ret, buf.getvalue().rstrip("\n")


def case_dispatch(rep, tag, path, H, seq_lens, h0_dtype, final_dtypes, pad_tokens):
    flags, text = _run_captured(
        acc.check, tag, path, H, seq_lens, h0_dtype, final_dtypes,
        pad_tokens=pad_tokens)
    rep.info(f"K2 path:{path}, num_heads:{H}, batch_size:{len(seq_lens)}, "
             f"seq_lens:{seq_lens}, pad_tokens:{pad_tokens}, "
             f"initial_state:{_dt_tag(h0_dtype)}")
    for line in text.splitlines():
        rep.info(line)
    n_ok = sum(1 for f in flags if f)
    if n_ok == len(flags):
        return []
    bad = [l.strip() for l in text.splitlines() if "FAIL" in l]
    return [f"{n_ok}/{len(flags)} checks OK"] + bad[:4]


def case_from_caselist(rep, tp, c):
    ok, text = _run_captured(k3.run_acc, tp, c)
    fast = "FAST" if k3.pro_fast(c) else "base"
    rep.info(f"tp:{tp}, num_heads:{c['H']}, batch_size:{c['N']}, "
             f"T_total:{c['T_total']}, gate_lower_bound:{c['lower']}, "
             f"dispatch:{fast}")
    for line in text.splitlines():
        rep.info(line)
    if ok:
        return []
    return [f"tp{tp} H{c['H']} N{c['N']} T{c['T_total']} accuracy fail: {text.strip()}"]


def build_caselist_plan(list_dir, tps, per_tp, acc_max_t):
    """Walk the prefill caselists per tp and pick the representative shapes of each tp."""
    plan, sources = [], []
    for tp in tps:
        pattern = os.path.join(list_dir, k3.LIST_GLOB.format(tp=tp))
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: no matching caselist: {pattern}")
            continue
        path = matches[0]
        sources.append(path)
        cases = k3.parse_list(path)
        picked = k3.pick_acc_cases(cases, acc_max_t, per_tp)
        if not picked:
            print(f"WARNING: {os.path.basename(path)} has no case where every sequence is "
                  f"<= {acc_max_t}, skipping")
            continue
        for c in picked:
            label = f"tp{tp}_H{c['H']}_N{c['N']}_T{c['T_total']}"
            plan.append((label, case_from_caselist, (tp, c)))
    return plan, sources


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["all", "dispatch", "caselist"], default="all")
    ap.add_argument("--list-dir", default=DEFAULT_LIST_DIR,
                    help="prefill caselist directory (relative to the repo root)")
    ap.add_argument("--tp", default=",".join(str(t) for t in DEFAULT_TPS),
                    help="tp degrees to walk (per-card H=96/48/24/12)")
    ap.add_argument("--per-tp", type=int, default=2,
                    help="how many representative shapes to take from each tp caselist (default 2)")
    ap.add_argument("--acc-max-t", type=int, default=1024,
                    help="per-sequence length cap of the caselist stage: the fp32 reference is a per-chunk "
                         "python loop, long cases never finish inside the accuracy gate (default 1024)")
    args = ap.parse_args()

    list_dir = args.list_dir if os.path.isabs(args.list_dir) \
        else os.path.join(_REPO_ROOT, args.list_dir)
    tps = [int(t) for t in args.tp.split(",") if t.strip()]

    plan, sources = [], []
    if args.phase in ("all", "dispatch"):
        for tag, path, H, seq_lens in acc.CASES:
            for h0_dtype, final_dtypes in H0_MODES:
                plan.append((f"{tag}_{_dt_tag(h0_dtype)}", case_dispatch,
                             (tag, path, H, seq_lens, h0_dtype, final_dtypes, 0)))
        # padded tail: only examined with zero and bf16 initial states, consistent with the manual script
        for tag, path, H, seq_lens, pad_tokens in (
                ("b1-h12-s2205-pad1", "padded varlen", 12, [2205], 1),
                ("b4-h12-ragged-pad7", "padded varlen", 12, [100, 273, 16, 272], 7)):
            for h0_dtype, final_dtypes in H0_MODES[:2]:
                plan.append((f"{tag}_{_dt_tag(h0_dtype)}", case_dispatch,
                             (tag, path, H, seq_lens, h0_dtype, final_dtypes,
                              pad_tokens)))
    if args.phase in ("all", "caselist"):
        cl_plan, sources = build_caselist_plan(list_dir, tps, args.per_tp,
                                               args.acc_max_t)
        if not cl_plan and args.phase == "caselist":
            print(f"ERROR: no usable Kimi-K3 prefill caselist under {list_dir}",
                  file=sys.stderr)
            return 2
        plan += cl_plan

    # Case file tags the sub-case names with their source; in dispatch-only mode there is no caselist, so tag the script itself
    case_file = sources[0] if sources else "tests/test_kda_prefill_flashkdapro.py"
    rep = CaseReporter("kda_prefill_flashkdapro", case_file, total=len(plan))
    rep.header()
    print(f"oracle: fp32 torch ref_fwd (private/test_fwd_e2e.py), "
          f"judge err_ratio < {acc.TOL:g}")
    print(f"phase: {args.phase}")
    if sources:
        print(f"caselists: {[os.path.basename(s) for s in sources]}")
        print(f"per-tp picks: {args.per_tp}, acc_max_t: {args.acc_max_t}")

    for label, fn, fn_args in plan:
        rep.start(label)
        if rep.aborted:
            rep.fail("skipped: an earlier case already brought the device down")
            continue
        with rep.guard():
            torch.cuda.empty_cache()
            failures = fn(rep, *fn_args)
            if failures:
                rep.fail("; ".join(failures[:5]))
            else:
                rep.ok()

    return rep.summary()


if __name__ == "__main__":
    sys.exit(main())
