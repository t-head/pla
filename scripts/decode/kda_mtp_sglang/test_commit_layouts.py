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
from pla.decode.kda_mtp_sglang import commit_kda_replayssm_after_verify


def main():
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
    print("RESULT: all PASS")


if __name__ == "__main__":
    main()
