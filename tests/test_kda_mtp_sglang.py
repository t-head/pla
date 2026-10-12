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

"""KDA MTP (SGLang spec-decode) correctness test (xtest entry) -- caselist driven + per-sub-case report.

The two entries under test (the verify / commit stages of Kimi-K3 KDA MTP):

  * ``pla.decode.kda_mtp_sglang.fused_kda_decode_mtp_dspark``
    verify (CACHE_RING): computes the attention output and writes back 4 ``replayssm_*`` rings and 3 sets of
    conv-window snapshots for the commit stage to consume. ``recurrent_state`` must be read-only.
  * ``pla.decode.kda_mtp_sglang.commit_kda_replayssm_after_verify``
    commit exact-fold: replays each request's accepted prefix into the fp32 checkpoint (in place,
    one launch covering all layers), and rolls the conv state back according to ``last_correct_step_indices``
    to the last accepted step (a pure copy, required to be byte-for-byte equal).

## Relation to scripts/decode/kda_mtp_sglang/test_correctness.py

The fp64/fp32 oracle (``verify_reference`` / ``fold_reference``), input construction (``Cfg`` /
``build_verify_inputs`` / ``build_commit_inputs``), the conv scatter reference
(``conv_scatter_reference``) and the tolerances TOL_BF16=6e-3 / TOL_FP32=5e-6 are **all reused** from that
script; this file does not rewrite the reference -- those criteria were checked item by item against the H20 original, and a second copy would inevitably drift.

## Which shapes to run (following scripts/prefill/flashkdapro/e2e_kimi_k3.py)

The original script's shapes are hand-written ((4,4),(8,12)) and ((4,2),(8,4)); here they come from OpForge's
decode KDA caselist to reflect the real serving distribution:

  * batch -- parse every ``batch_size`` of the caselist, take ``--max-bs-cases`` representative values by quantile,
    then truncate by ``--bs-cap``. Truncation is necessary: verify allocates bs*(1+num_spec) tokens
    and bs+2 fp32 state slots (H*128*128*4 bytes/slot) at once, and the fp64 oracle is a per-step
    python loop, so bs in the hundreds would both OOM and never finish.
  * head -- take the caselist ``num_heads_qk`` (heads per card) and add the typical K3 dspark value
    12. The MTP kernel only constrains head_dim==128 (kda_mtp_verify.cu); H is not limited by an instantiation set,
    so the real caselist values can be used directly, unlike mega fusion which has to pick from its supported set.
  * layer -- an axis only on the fold side, taken from ``--layers`` (default 2,4); production has 69 layers, but fold
    correctness does not depend on the layer count (each layer replays independently); it only affects how many layers one launch covers.

## Output

Markers are printed per sub-case following the tests/case_report.py protocol; acompute.PLAResultParser turns
each of verify/fold/padding/track into its own ES record and JUnit testcase row.

Usage (inside the container, from the repo root; must be run as a plain script):

    python tests/test_kda_mtp_sglang.py
    python tests/test_kda_mtp_sglang.py --phase fold --layers 2,4,69
    python tests/test_kda_mtp_sglang.py --caselist caselist/decode/Kimi-Linear-48B-A3B_decode_tp2_kvlimit_200GB_model_len_128K_kda_op.list
"""
import argparse
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
# Same as test_kda_decode_mega.py: never add the repo root to sys.path, otherwise the source tree's pla/
# shadows the installed package with its build artifacts.
_MTP_DIR = os.path.join(_REPO_ROOT, "scripts", "decode", "kda_mtp_sglang")
for _p in (_TESTS_DIR, _MTP_DIR, os.path.join(_MTP_DIR, "private")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from case_report import CaseReporter  # noqa: E402
from utils import read_cmds_from_file  # noqa: E402

from _common import Cfg, build_commit_inputs, build_verify_inputs  # noqa: E402
from _reference import fmt, fold_reference, score, verify_reference  # noqa: E402

# run_cuda / run_commit / conv_scatter_reference / tolerances all come from the manual script, see the file header
import test_correctness as ref  # noqa: E402

DEFAULT_CASELIST = ("caselist/decode/"
                    "Kimi-Linear-48B-A3B_decode_tp1_kvlimit_200GB_model_len_128K_kda_op.list")
K3_TYPICAL_HEADS = 12       # Kimi-K3 dspark heads per card
NUM_SPEC = 7                # gamma=7 -> draft_token_num=8, same as production


def collect_bs(caselist_path, k, cap):
    """Take k representative values from the caselist batch_size distribution and truncate by cap (see the file header)."""
    cases = read_cmds_from_file(caselist_path)
    values = sorted({int(c["batch_size"]) for c in cases if "batch_size" in c})
    values = [v for v in values if 0 < v <= cap]
    if not values:
        raise ValueError(
            f"no case with batch_size<={cap} in the caselist: {caselist_path}")
    if k >= len(values):
        return values
    if k <= 1:
        return [values[-1]]
    return sorted({values[round(i * (len(values) - 1) / (k - 1))] for i in range(k)})


def collect_heads(caselist_path):
    """Heads per card from the caselist, plus the typical K3 value 12."""
    cases = read_cmds_from_file(caselist_path)
    heads = {int(c["num_heads_qk"]) for c in cases if "num_heads_qk" in c}
    heads.add(K3_TYPICAL_HEADS)
    return sorted(heads)


def _judge(rep, got, want, name, tol, failures):
    """Compare a group of tensors, one info line; record into failures if out of tolerance."""
    s = score(got, want, name)
    ok = s["finite"] and s["max_rel"] <= tol
    rep.info(f"  Max diff {fmt(s)} tol={tol:.0e} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{name} max_rel={s['max_rel']:.3e}>tol={tol:.0e}")
    return ok


def _judge_equal(rep, got, want, name, failures):
    """Byte-for-byte equality (the conv rollback is a pure copy; any difference is a bug)."""
    ok = torch.equal(got, want)
    rep.info(f"  {name} byte-identical={ok} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{name} not byte-identical")
    return ok


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def case_verify(rep, bs, heads):
    cfg = Cfg(bs=bs, num_spec=NUM_SPEC, heads=heads, onorm=False)
    vi = build_verify_inputs(cfg)
    out = ref.run_cuda(vi, cfg)
    torch.cuda.synchronize()
    expect = verify_reference(vi, cfg)
    n, b = cfg.bs, cfg.draft_token_num

    rep.info(f"batch_size:{bs}, num_heads:{heads}, num_spec:{NUM_SPEC}, "
             f"draft_token_num:{b}, T:{cfg.T}")
    failures = []
    _judge(rep, out, expect["out"], "out (bf16)", ref.TOL_BF16, failures)
    _judge(rep, vi["replayssm_rawv"][:n, :, :b], expect["ring_rawv"][:n, :, :b],
           "ring_rawv", ref.TOL_BF16, failures)
    _judge(rep, vi["replayssm_rawk"][:n, :, :b], expect["ring_rawk"][:n, :, :b],
           "ring_rawk", ref.TOL_BF16, failures)
    _judge(rep, vi["replayssm_g"][:n, :, :b], expect["ring_g"][:n, :, :b],
           "ring_g", ref.TOL_FP32, failures)
    _judge(rep, vi["replayssm_beta"][:n, :, :b], expect["ring_beta"][:n, :, :b],
           "ring_beta", ref.TOL_FP32, failures)
    _judge(rep, vi["intermediate_conv_q"][:n, :b], expect["ic_q"][:n, :b],
           "conv_win_q", ref.TOL_BF16, failures)
    _judge(rep, vi["intermediate_conv_v"][:n, :b], expect["ic_v"][:n, :b],
           "conv_win_v", ref.TOL_BF16, failures)

    # recurrent_state is read-only: the verify stage must never modify the checkpoint in place, otherwise commit's
    # exact-fold would replay from an already polluted starting point.
    rs = vi["recurrent_state"]
    before = rs.clone()
    ref.run_cuda(vi, cfg)
    torch.cuda.synchronize()
    _judge_equal(rep, rs, before, "recurrent_state read-only", failures)
    return failures


def case_verify_padding(rep, heads):
    """Padding requests with slot=-1: their rows must be all zero and must not read/write wild addresses."""
    cfg = Cfg(bs=4, num_spec=NUM_SPEC, heads=heads, onorm=False)
    vi = build_verify_inputs(cfg)
    vi["ssm_state_indices"] = vi["ssm_state_indices"].clone()
    vi["ssm_state_indices"][-1] = -1
    pb = int(vi["cu_seqlens"][cfg.bs - 1])
    pe = int(vi["cu_seqlens"][cfg.bs])
    out = ref.run_cuda(vi, cfg)
    torch.cuda.synchronize()
    ok = bool((out[0, pb:pe] == 0).all())
    rep.info(f"batch_size:4, num_heads:{heads}, padded rows [{pb},{pe}) all-zero={ok}")
    return [] if ok else [f"padded rows not zero (rows {pb}..{pe})"]


# ---------------------------------------------------------------------------
# fold (commit)
# ---------------------------------------------------------------------------
def case_fold(rep, bs, heads, layers):
    cfg = Cfg(bs=bs, num_spec=NUM_SPEC, heads=heads, layers=layers)
    ci = build_commit_inputs(cfg)
    want = fold_reference(ci, cfg)
    conv_want = ref.conv_scatter_reference(ci)
    got = ref.run_commit(ci, ci["temporal"].clone())

    rep.info(f"batch_size:{bs}, num_heads:{heads}, layers:{layers}, "
             f"num_spec:{NUM_SPEC}")
    failures = []
    _judge(rep, got.temporal, want, f"fold bs={bs} L={layers}",
           ref.TOL_FP32, failures)
    _judge_equal(rep, got.conv[0], conv_want, "conv rollback", failures)
    return failures


def case_fold_track(rep, heads):
    """HAS_TRACK path: snapshot the intermediate state of a given accepted step into a spare slot."""
    cfg = Cfg(bs=2, num_spec=NUM_SPEC, heads=heads, layers=2)
    accept = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
    ci = build_commit_inputs(cfg, accept_lens=accept)
    track_idx = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    track_step = torch.tensor([1, 3], device="cuda", dtype=torch.int32)

    want = fold_reference(ci, cfg)
    partial_ci = dict(ci)
    partial_ci["temporal"] = ci["temporal"].clone()
    partial_ci["accept_lens"] = track_step + 1
    partial = fold_reference(partial_ci, cfg)
    for n, dst in enumerate(track_idx.tolist()):
        want[:, dst] = partial[:, n]
    conv_want = ref.conv_scatter_reference(ci, track_idx, track_step)

    got = ref.run_commit(ci, ci["temporal"].clone(), track_idx, track_step)
    rep.info(f"batch_size:2, num_heads:{heads}, layers:2, "
             f"accept_lens:[3,5], track_step:[1,3]")
    failures = []
    _judge(rep, got.temporal, want, "fold HAS_TRACK", ref.TOL_FP32, failures)
    _judge_equal(rep, got.conv[0], conv_want, "track conv", failures)
    return failures


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--caselist", default=DEFAULT_CASELIST,
                    help="decode KDA caselist used for the batch/head distribution (relative to the repo root)")
    ap.add_argument("--phase", choices=["all", "verify", "fold"], default="all")
    ap.add_argument("--max-bs-cases", type=int, default=3,
                    help="how many batch representative values to take from the caselist (default 3)")
    ap.add_argument("--bs-cap", type=int, default=64,
                    help="batch cap: the fp64 oracle is a per-step python loop, "
                         "anything larger both OOMs and never finishes (default 64)")
    ap.add_argument("--layers", default="2,4",
                    help="number of KDA layers on the fold side (default 2,4; production has 69, correctness does not depend on it)")
    args = ap.parse_args()

    caselist_path = args.caselist if os.path.isabs(args.caselist) \
        else os.path.join(_REPO_ROOT, args.caselist)
    if not os.path.isfile(caselist_path):
        print(f"ERROR: caselist does not exist: {caselist_path}", file=sys.stderr)
        return 2

    bs_list = collect_bs(caselist_path, args.max_bs_cases, args.bs_cap)
    heads_list = collect_heads(caselist_path)
    layers_list = [int(x) for x in args.layers.split(",") if x.strip()]

    # List all cases first so the reporter can print the correct i/N
    plan = []
    if args.phase in ("all", "verify"):
        for h in heads_list:
            for bs in bs_list:
                plan.append(("verify", f"verify_H{h}_B{bs}", case_verify, (bs, h)))
        plan.append(("verify", f"verify_padding_H{heads_list[-1]}",
                     case_verify_padding, (heads_list[-1],)))
    if args.phase in ("all", "fold"):
        for h in heads_list:
            for bs in bs_list:
                for L in layers_list:
                    plan.append(("fold", f"fold_H{h}_B{bs}_L{L}",
                                 case_fold, (bs, h, L)))
        plan.append(("fold", f"fold_track_H{heads_list[-1]}",
                     case_fold_track, (heads_list[-1],)))

    rep = CaseReporter("kda_mtp_sglang", caselist_path, total=len(plan))
    rep.header()
    print(f"bs from caselist (cap {args.bs_cap}): {bs_list}")
    print(f"heads from caselist (+K3 typical {K3_TYPICAL_HEADS}): {heads_list}")
    print(f"fold layers: {layers_list}")
    print(f"tolerance: bf16 {ref.TOL_BF16:.0e}, fp32 {ref.TOL_FP32:.0e}")

    for _phase, label, fn, fn_args in plan:
        rep.start(label)
        if rep.aborted:
            rep.fail("skipped: an earlier case already brought the device down")
            continue
        with rep.guard():
            torch.cuda.empty_cache()
            failures = fn(rep, *fn_args)
            if failures:
                rep.fail("; ".join(failures[:6]))
            else:
                rep.ok()

    return rep.summary()


if __name__ == "__main__":
    sys.exit(main())
