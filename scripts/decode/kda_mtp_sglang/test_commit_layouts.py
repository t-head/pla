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

"""Compare public commit on PLA and SGLang conv layouts, including graph replay.

CUDA_VISIBLE_DEVICES=0 python scripts/decode/kda_mtp_sglang/test_commit_layouts.py
"""

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "private"))

import torch

from _common import Cfg, SpecState, build_commit_inputs
from _reference import fold_reference, score
from pla.decode.kda_mtp_sglang import _C, commit_kda_replayssm_after_verify


def run_large_case(batch, layers, grouped=False, heads=8):
    cfg = Cfg(bs=batch, pool=2 * batch + 8, heads=heads, num_spec=15, layers=layers)
    ci = build_commit_inputs(cfg)
    direct = SpecState.from_commit_inputs(ci)
    if grouped:
        direct.replayssm_rawk = direct.replayssm_rawk[:, :, :4].contiguous()
    slots = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int64)
    slots[-2:] = torch.tensor([0, -1], device="cuda")
    accepted = torch.arange(batch, device="cuda", dtype=torch.int64) % 17
    kwargs = dict(state_batch_indices=slots, accept_lens=accepted,
                  last_correct_step_indices=accepted - 1, null_block_id=0)
    active = (slots > 0) & (accepted > 0)
    kwargs["last_correct_step_indices"] = torch.where(active, accepted - 1, -1)
    initial = direct.temporal.clone()
    initial_conv = direct.conv[0].clone()
    expected_conv = initial_conv.clone()
    requests = torch.arange(batch, device="cuda")[active]
    expected_conv[:, slots[active]] = direct.intermediate_conv_window[0][
        :, requests, accepted[active] - 1]
    reference = copy.deepcopy(direct)
    disabled = torch.full_like(slots, -1)
    commit_kda_replayssm_after_verify(
        spec_state=reference, mamba_track_indices=disabled,
        mamba_steps_to_track=disabled, **kwargs)

    def reset():
        direct.temporal.copy_(initial)
        direct.conv[0].copy_(initial_conv)

    def check():
        torch.cuda.synchronize()
        error = (direct.temporal - reference.temporal).abs().max().item()
        limit = reference.temporal.abs().max().item() * 5e-6
        assert torch.isfinite(direct.temporal).all() and error <= limit, (error, limit)
        assert torch.equal(direct.conv[0], reference.conv[0])
        assert torch.equal(direct.conv[0], expected_conv), "conv indexing oracle"
        untouched = torch.ones(cfg.slots, dtype=torch.bool, device="cuda")
        untouched[slots[active]] = False
        assert torch.equal(direct.temporal[:, untouched], initial[:, untouched])
        assert torch.equal(direct.conv[0][:, untouched], initial_conv[:, untouched])

    commit_kda_replayssm_after_verify(spec_state=direct, **kwargs)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit_kda_replayssm_after_verify(spec_state=direct, **kwargs)
    for _ in range(2):
        reset()
        graph.replay()
        check()

    for name in ("replayssm_rawk", "replayssm_g"):
        original = getattr(direct, name)
        unaligned = torch.empty(original.numel() + 1, device="cuda", dtype=original.dtype)[1:]
        unaligned = unaligned.view_as(original).copy_(original)
        setattr(direct, name, unaligned)
        reset()
        commit_kda_replayssm_after_verify(spec_state=direct, **kwargs)
        check()
        setattr(direct, name, original)

    for tensors in (direct.conv, direct.intermediate_conv_window):
        original = tensors[0]
        tensors[0] = torch.empty(
            original.numel() + 1, device="cuda", dtype=original.dtype
        )[1:].view_as(original).copy_(original)
        reset()
        commit_kda_replayssm_after_verify(spec_state=direct, **kwargs)
        check()
        tensors[0] = original
    original = direct.intermediate_conv_window[0]
    storage = torch.full((*original.shape[:-1], original.shape[-1] * 2),
                         7, device="cuda", dtype=original.dtype)
    direct.intermediate_conv_window[0] = storage[..., ::2].copy_(original)
    reset()
    commit_kda_replayssm_after_verify(spec_state=direct, **kwargs)
    check()
    assert torch.equal(storage[..., ::2], original), "conv input mutated"
    assert (storage[..., 1::2] == 7).all(), "conv spare storage mutated"
    direct.intermediate_conv_window[0] = original

    tracks = torch.where(active, torch.arange(batch, device="cuda") + batch + 1, -1)
    reset()
    commit_kda_replayssm_after_verify(
        spec_state=direct, mamba_track_indices=tracks,
        mamba_steps_to_track=kwargs["last_correct_step_indices"], **kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(direct.temporal[:, slots[active]],
                               reference.temporal[:, slots[active]], rtol=0, atol=0)
    torch.testing.assert_close(direct.temporal[:, tracks[active]],
                               reference.temporal[:, slots[active]], rtol=0, atol=0)
    assert torch.equal(direct.conv[0][:, tracks[active]], reference.conv[0][:, slots[active]])
    print(f"PASS B={batch} H={heads} layers={layers} grouped={grouped}: accept 0..16, "
          "slots/null, graph, tracking, alignment fallback, strided conv, conv oracle")


def run_ring_case(ring_len, index_dtype):
    cfg = Cfg(bs=4, pool=8, heads=4, num_spec=ring_len - 1,
              ring_len=ring_len, layers=2)
    accepted = torch.tensor([0, 1, max(1, ring_len // 2), ring_len],
                            device="cuda", dtype=index_dtype)
    ci = build_commit_inputs(cfg, accept_lens=accepted)
    ci["state_batch_indices"] = torch.tensor([3, 2, 1, 0], device="cuda",
                                             dtype=index_dtype)
    ci["last_correct_step_indices"] = (accepted - 1).to(index_dtype)
    track_idx = torch.tensor([-1, 5, 6, 7], device="cuda", dtype=index_dtype)
    track_step = torch.where(accepted > 0, (accepted - 1) // 2, -1)
    initial = SpecState.from_commit_inputs(ci)
    direct = copy.deepcopy(initial)
    expected = fold_reference(ci, cfg)
    partial_ci = dict(ci, accept_lens=track_step + 1)
    partial = fold_reference(partial_ci, cfg)
    for slot, dst in zip(ci["state_batch_indices"].tolist()[1:],
                         track_idx.tolist()[1:]):
        expected[:, dst] = partial[:, slot]
    expected_conv = initial.conv[0].clone()
    for i in range(1, cfg.bs):
        expected_conv[:, ci["state_batch_indices"][i]] = (
            initial.intermediate_conv_window[0][:, i, accepted[i] - 1])
        expected_conv[:, track_idx[i]] = (
            initial.intermediate_conv_window[0][:, i, track_step[i]])
    kwargs = dict(state_batch_indices=ci["state_batch_indices"],
                  accept_lens=accepted,
                  last_correct_step_indices=ci["last_correct_step_indices"],
                  mamba_track_indices=track_idx, mamba_steps_to_track=track_step,
                  null_block_id=-1)

    def check():
        torch.cuda.synchronize()
        error = score(direct.temporal, expected, "ring fold")
        assert error["finite"] and error["max_rel"] <= 5e-6, error
        assert torch.equal(direct.conv[0], expected_conv), "ring conv oracle"
        assert torch.equal(direct.temporal[:, [3, 4]], initial.temporal[:, [3, 4]])
        assert torch.equal(direct.conv[0][:, [3, 4]], initial.conv[0][:, [3, 4]])

    commit_kda_replayssm_after_verify(spec_state=direct, **kwargs)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit_kda_replayssm_after_verify(spec_state=direct, **kwargs)
    for _ in range(2):
        direct.temporal.copy_(initial.temporal)
        direct.conv[0].copy_(initial.conv[0])
        graph.replay()
        check()
    print(f"PASS ring={ring_len} indices={index_dtype}: FP64 oracle, full/zero "
          "accept, tracking, untouched slots, graph replay")


def run_conv_case(dtype, width, index_dtype):
    gen = torch.Generator(device="cuda").manual_seed(1234)
    src = torch.randn(2, 4, 3, 129, width, device="cuda", dtype=dtype, generator=gen)
    initial = torch.randn(2, 6, 129, width, device="cuda", dtype=dtype, generator=gen)
    dst_idx = torch.tensor([4, -1, 1, 2], device="cuda", dtype=index_dtype)
    step_idx = torch.tensor([2, 0, 1, -1], device="cuda", dtype=index_dtype)
    expected = initial.clone()
    expected[:, 4] = src[:, 0, 2]
    expected[:, 1] = src[:, 2, 1]
    dst = initial.clone()
    _C.conv_window_scatter(dst, src, dst_idx, step_idx)
    torch.cuda.synchronize()
    assert torch.equal(dst, expected), "conv dtype/window oracle"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _C.conv_window_scatter(dst, src, dst_idx, step_idx)
    for _ in range(2):
        dst.copy_(initial)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(dst, expected), "conv graph dtype/window oracle"
    print(f"PASS conv dtype={dtype} width={width} indices={index_dtype}: "
          "byte-exact oracle, masked requests, odd channels, graph replay")


def main():
    for ring_len in range(1, 33):
        for index_dtype in (torch.int32, torch.int64):
            run_ring_case(ring_len, index_dtype)
    for dtype in (torch.bfloat16, torch.float32):
        for width in range(1, 5):
            for index_dtype in (torch.int32, torch.int64):
                run_conv_case(dtype, width, index_dtype)
    for heads, steps in ((8, 6), (12, 8)):
        ci = build_commit_inputs(Cfg(bs=4, pool=8, heads=heads, num_spec=steps - 1, layers=2))
        original = SpecState.from_commit_inputs(ci)
        direct = copy.deepcopy(original)
        direct.conv = [t.transpose(-1, -2).contiguous() for t in direct.conv]
        direct.intermediate_conv_window = [
            t.transpose(-1, -2).contiguous() for t in direct.intermediate_conv_window
        ]
        kwargs = dict(
            state_batch_indices=torch.tensor([2, 1, 3, -1], device="cuda", dtype=torch.int32),
            accept_lens=torch.tensor([1, steps, steps // 2, 0], device="cuda", dtype=torch.int32),
            last_correct_step_indices=torch.tensor([0, steps - 1, steps // 2 - 1, -1], device="cuda", dtype=torch.int32),
            mamba_track_indices=torch.tensor([5, -1, 6, -1], device="cuda", dtype=torch.int32),
            mamba_steps_to_track=torch.tensor([0, -1, 0, -1], device="cuda", dtype=torch.int32),
        )
        initial = copy.deepcopy(direct)
        commit_kda_replayssm_after_verify(spec_state=original, **kwargs)
        commit_kda_replayssm_after_verify(spec_state=direct, conv_state_layout="width_dim", **kwargs)
        torch.testing.assert_close(direct.temporal, original.temporal, rtol=0, atol=0)
        torch.testing.assert_close(direct.conv[0].transpose(-1, -2), original.conv[0], rtol=0, atol=0)
        torch.testing.assert_close(direct.temporal[:, [0, 4, 7]], initial.temporal[:, [0, 4, 7]], rtol=0, atol=0)
        torch.testing.assert_close(direct.conv[0][:, [0, 4, 7]], initial.conv[0][:, [0, 4, 7]], rtol=0, atol=0)
        expected = copy.deepcopy(direct)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit_kda_replayssm_after_verify(spec_state=direct, conv_state_layout="width_dim", **kwargs)
        for _ in range(2):
            direct.temporal.copy_(initial.temporal)
            direct.conv[0].copy_(initial.conv[0])
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(direct.temporal, expected.temporal, rtol=0, atol=0)
            torch.testing.assert_close(direct.conv[0], expected.conv[0], rtol=0, atol=0)
        unaligned = copy.copy(initial)
        unaligned.temporal = torch.empty(
            initial.temporal.numel() + 1, device="cuda", dtype=torch.float32
        )[1:].view_as(initial.temporal)
        try:
            commit_kda_replayssm_after_verify(
                spec_state=unaligned, conv_state_layout="width_dim", **kwargs
            )
        except RuntimeError as error:
            assert "16-byte aligned" in str(error), error
        else:
            raise AssertionError("unaligned float4 state was not rejected")
        print(f"PASS H={heads} T={steps}: exact fold/conv, padding, tracking, graph")
    run_large_case(512, 1)
    run_large_case(256, 2, grouped=True)
    run_large_case(512, 1, heads=9)
    print("RESULT: all PASS")


if __name__ == "__main__":
    main()
