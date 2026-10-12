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

"""pla.decode.kda_mtp_sglang correctness test -- both the verify and fold paths are checked against an fp64 oracle.

Kimi-K3 KDA MTP has two entries:

  * ``fused_kda_decode_mtp_dspark`` -- the verify (CACHE_RING) path. Besides returning the attention
    output, it writes back 4 ``replayssm_*`` rings and 3 conv-window snapshots for the
    commit stage to consume; this test compares each of these side effects.
  * ``commit_kda_replayssm_after_verify`` -- the commit-stage exact fold: replays each
    request's accepted prefix into the fp32 checkpoint (in place, one launch covering all
    layers), and rolls the conv state back to the last accepted step according to ``last_correct_step_indices``;
    the conv rollback is a pure copy and byte-for-byte equivalent.

This script matches the original H20 test:

  1. verify: ``bs=4/H=4`` and ``bs=8/H=12``, bf16 output / rings ``rawv/rawk``
     TOL_BF16=6e-3; fp32 rings ``g/beta`` and conv windows TOL_FP32=5e-6; plus one
     read-only assertion on ``recurrent_state`` and an all-zero assertion for ``slot=-1`` padding.
  2. fold: ``bs=4/L=2`` and ``bs=8/L=4``, TOL_FP32=5e-6, with a byte-for-byte equality check of one conv rollback
     and the HAS_TRACK path (copying the intermediate state of a given step into a replica slot).

Usage (inside the PPU server container):
    python scripts/decode/kda_mtp_sglang/test_correctness.py
"""

# ruff: noqa: E501
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "private"))

import torch

from _common import Cfg, SpecState, build_commit_inputs, build_verify_inputs, gpu_banner
from _harness import Timer
from _reference import fmt, fold_reference, score, verify_reference

from pla.decode.kda_mtp_sglang import (
    commit_kda_replayssm_after_verify,
    fused_kda_decode_mtp_dspark,
)

TOL_BF16 = 6e-3
TOL_FP32 = 5e-6
FAILED = []


def check(got, want, name, tol):
    s = score(got, want, name)
    ok = s["finite"] and s["max_rel"] <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {fmt(s)}")
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def run_cuda(vi, cfg, out=None):
    return fused_kda_decode_mtp_dspark(
        x_q=vi["x_q"], x_k=vi["x_k"], x_v=vi["x_v"],
        w_q=vi["w_q"], w_k=vi["w_k"], w_v=vi["w_v"],
        cs_q=vi["cs_q"], cs_k=vi["cs_k"], cs_v=vi["cs_v"],
        g=vi["g"], beta=vi["beta"], A_log=vi["A_log"], dt_bias=vi["dt_bias"],
        recurrent_state=vi["recurrent_state"],
        intermediate_state_indices=vi["intermediate_state_indices"],
        intermediate_conv_q=vi["intermediate_conv_q"],
        intermediate_conv_k=vi["intermediate_conv_k"],
        intermediate_conv_v=vi["intermediate_conv_v"],
        ssm_state_indices=vi["ssm_state_indices"],
        cu_seqlens=vi["cu_seqlens"],
        lower_bound=float(vi["lower_bound"]),
        scale=float(vi["scale"]),
        replayssm_rawv=vi["replayssm_rawv"], replayssm_rawk=vi["replayssm_rawk"],
        replayssm_g=vi["replayssm_g"], replayssm_beta=vi["replayssm_beta"],
        out=out,
    )


def run_verify_tests():
    print("\n=== verify (fused_kda_decode_mtp_dspark) ===")
    for bs, heads in ((4, 4), (8, 12)):
        cfg = Cfg(bs=bs, num_spec=7, heads=heads, onorm=False)
        vi = build_verify_inputs(cfg)
        out = run_cuda(vi, cfg)
        torch.cuda.synchronize()
        ref = verify_reference(vi, cfg)
        n, b = cfg.bs, cfg.draft_token_num
        print(f"-- bs={bs} H={heads}")
        check(out, ref["out"], "out (bf16)", TOL_BF16)
        check(vi["replayssm_rawv"][:n, :, :b], ref["ring_rawv"][:n, :, :b],
              "ring_rawv", TOL_BF16)
        check(vi["replayssm_rawk"][:n, :, :b], ref["ring_rawk"][:n, :, :b],
              "ring_rawk", TOL_BF16)
        check(vi["replayssm_g"][:n, :, :b], ref["ring_g"][:n, :, :b],
              "ring_g", TOL_FP32)
        check(vi["replayssm_beta"][:n, :, :b], ref["ring_beta"][:n, :, :b],
              "ring_beta", TOL_FP32)
        check(vi["intermediate_conv_q"][:n, :b], ref["ic_q"][:n, :b],
              "conv_win_q", TOL_BF16)
        check(vi["intermediate_conv_v"][:n, :b], ref["ic_v"][:n, :b],
              "conv_win_v", TOL_BF16)
        rs = vi["recurrent_state"]
        before = rs.clone()
        run_cuda(vi, cfg)
        torch.cuda.synchronize()
        ok = torch.equal(before, rs)
        print(f"  [{'PASS' if ok else 'FAIL'}] recurrent_state read-only")
        if not ok:
            FAILED.append("read-only")

    print("\n-- padding (slot = -1)")
    cfg = Cfg(bs=4, num_spec=7, heads=12, onorm=False)
    vi = build_verify_inputs(cfg)
    vi["ssm_state_indices"] = vi["ssm_state_indices"].clone()
    vi["ssm_state_indices"][-1] = -1
    pb = int(vi["cu_seqlens"][cfg.bs - 1]); pe = int(vi["cu_seqlens"][cfg.bs])
    out = run_cuda(vi, cfg)
    torch.cuda.synchronize()
    ok = bool((out[0, pb:pe] == 0).all())
    print(f"  [{'PASS' if ok else 'FAIL'}] padded rows all-zero")
    if not ok:
        FAILED.append("padding")

    print("\n-- verify kernel latency (median over reps, L2-flushed)")
    timer = Timer(reps=25, inner=20)
    for bs in (1, 8, 64):
        cfg = Cfg(bs=bs, num_spec=7, heads=12, onorm=False)
        vi = build_verify_inputs(cfg)
        ob = torch.empty_like(vi["x_v"])
        run_cuda(vi, cfg, ob)
        torch.cuda.synchronize()
        r = timer.run(lambda: run_cuda(vi, cfg, ob))
        print(f"  bs={bs:<3} ppu={r['median_ms']*1e3:8.2f}us  "
              f"p10={r['p10_ms']*1e3:7.2f}  p90={r['p90_ms']*1e3:7.2f}  "
              f"spread={r['spread_pct']:.1f}%")


# ---------------------------------------------------------------------------
# fold
# ---------------------------------------------------------------------------


def conv_scatter_reference(ci, track_idx=None, track_step=None):
    """dst[l, dst_idx[i]] = src[l, i, step_idx[i]]  for every valid request."""
    conv = ci["conv"][0].clone()
    src = ci["intermediate_conv_window"][0]
    dst_idx = ci["state_batch_indices"].tolist()
    step_idx = ci["last_correct_step_indices"].tolist()
    nl = conv.shape[0]
    for l in range(nl):
        for i, (d_i, step) in enumerate(zip(dst_idx, step_idx)):
            if step < 0 or d_i < 0:
                continue
            conv[l, d_i] = src[l, i, step]
        if track_idx is not None and track_step is not None:
            for i, (d_i, step) in enumerate(zip(track_idx.tolist(),
                                                track_step.tolist())):
                if step < 0 or d_i < 0:
                    continue
                conv[l, d_i] = src[l, i, step]
    return conv


def run_commit(ci, temporal, track_idx=None, track_step=None):
    st = SpecState.from_commit_inputs(ci)
    st.temporal = temporal
    commit_kda_replayssm_after_verify(
        spec_state=st,
        state_batch_indices=ci["state_batch_indices"],
        accept_lens=ci["accept_lens"],
        last_correct_step_indices=ci["last_correct_step_indices"],
        mamba_track_indices=track_idx,
        mamba_steps_to_track=track_step,
        null_block_id=-1,
    )
    torch.cuda.synchronize()
    return st


def run_fold_tests():
    print("\n=== fold (commit_kda_replayssm_after_verify) ===")
    for bs, layers in ((4, 2), (8, 4)):
        cfg = Cfg(bs=bs, num_spec=7, heads=12, layers=layers)
        ci = build_commit_inputs(cfg)
        want = fold_reference(ci, cfg)
        conv_want = conv_scatter_reference(ci)

        cu = run_commit(ci, ci["temporal"].clone())

        s_cu = score(cu.temporal, want, f"fold  bs={bs} L={layers}")
        ok = s_cu["finite"] and s_cu["max_rel"] <= TOL_FP32
        print(f"  [{'PASS' if ok else 'FAIL'}] {fmt(s_cu)}")
        if not ok:
            FAILED.append(s_cu["name"])
        same_conv = torch.equal(cu.conv[0], conv_want)
        print(f"  [{'PASS' if same_conv else 'FAIL'}] conv rollback byte-identical")
        if not same_conv:
            FAILED.append("conv scatter")

    # Production HAS_TRACK path: snapshot the recurrence after the requested
    # accepted step into spare slots, and apply the matching conv scatter.
    cfg = Cfg(bs=2, num_spec=7, heads=12, layers=2)
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
    conv_want = conv_scatter_reference(ci, track_idx, track_step)
    cu = run_commit(ci, ci["temporal"].clone(), track_idx, track_step)
    s_cu = score(cu.temporal, want, "fold  HAS_TRACK")
    ok = s_cu["finite"] and s_cu["max_rel"] <= TOL_FP32
    print(f"  [{'PASS' if ok else 'FAIL'}] {fmt(s_cu)}")
    if not ok:
        FAILED.append(s_cu["name"])
    same_conv = torch.equal(cu.conv[0], conv_want)
    print(f"  [{'PASS' if same_conv else 'FAIL'}] track conv byte-identical")
    if not same_conv:
        FAILED.append("track conv scatter")

    print("\n-- fold latency (69 layers, mean accept 4.5, L2-flushed) --")
    timer = Timer(reps=20, inner=10)
    for bs in (1, 4, 8, 16, 32, 64):
        cfg = Cfg(bs=bs, num_spec=7, heads=12, layers=69)
        ci = build_commit_inputs(cfg)
        st = SpecState.from_commit_inputs(ci)
        st.temporal = ci["temporal"].clone()
        kw = dict(spec_state=st, state_batch_indices=ci["state_batch_indices"],
                  accept_lens=ci["accept_lens"],
                  last_correct_step_indices=ci["last_correct_step_indices"],
                  null_block_id=-1)
        commit_kda_replayssm_after_verify(**kw)
        torch.cuda.synchronize()
        r = timer.run(lambda: commit_kda_replayssm_after_verify(**kw))
        print(f"  bs={bs:<3} ppu={r['median_ms']*1e3:8.1f}us  "
              f"p10={r['p10_ms']*1e3:7.1f}  p90={r['p90_ms']*1e3:7.1f}  "
              f"spread={r['spread_pct']:.1f}%")


def main():
    print(gpu_banner())
    print("torch", torch.__version__)
    run_verify_tests()
    run_fold_tests()
    print("\nRESULT:", "FAILED: " + ", ".join(FAILED) if FAILED else "all PASS")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
