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

"""Check native sglang split views against the contiguous verify interface.

Run after building pla.decode.kda_mtp_sglang on each supported PPU product:
    python scripts/decode/kda_mtp_sglang/test_verify_layouts.py

Addressing is the only changed operation, so outputs and every cache must match
bit-for-bit. This complements test_correctness.py's independent fp64 oracle.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "private"))

import torch

from _common import Cfg, build_verify_inputs, gpu_banner
from pla.decode.kda_mtp_sglang import fused_kda_decode_mtp_dspark


def clone_inputs(inputs):
    return {key: value.clone(memory_format=torch.contiguous_format)
            if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()}


def run_case(batch, heads, steps):
    cfg = Cfg(bs=batch, heads=heads, num_spec=steps - 1, onorm=False)
    direct = build_verify_inputs(cfg)
    dim = cfg.dim
    # Nontrivial positive slot/scratch mappings, plus one graph-padding request.
    direct["ssm_state_indices"] = torch.arange(
        batch, 0, -1, dtype=torch.int32, device="cuda")
    direct["ssm_state_indices"][-1] = -1
    direct["intermediate_state_indices"] = torch.arange(
        batch, 0, -1, dtype=torch.int32, device="cuda")

    # Keep Q/K/V in one mixed projection, with gaps between successive tokens.
    mixed = torch.cat([direct[f"x_{name}"].reshape(cfg.T, dim)
                       for name in "qkv"], dim=-1)
    for name, part in zip("qkv", mixed.split(dim, dim=-1)):
        direct[f"x_{name}"] = part.view(1, cfg.T, heads, 128)
    # Gate and beta permit a token stride as well.
    gate = torch.zeros(1, cfg.T * 2, heads, 128, dtype=torch.bfloat16, device="cuda")
    beta = torch.zeros(1, cfg.T * 2, heads, dtype=torch.bfloat16, device="cuda")
    gate[:, ::2].copy_(direct["g"])
    beta[:, ::2].copy_(direct["beta"])
    direct["g"], direct["beta"] = gate[:, ::2], beta[:, ::2]

    # sglang stores conv pools [slot, window, Q|K|V].
    conv = torch.cat([direct[f"cs_{name}"].transpose(-1, -2)
                      for name in "qkv"], dim=-1)
    for name, part in zip("qkv", conv.split(dim, dim=-1)):
        direct[f"cs_{name}"] = part.transpose(-1, -2)

    # Scratch has spare rows and steps, including untouched sentinels. Slicing
    # to the current step count retains max-step capacity in the row stride.
    window = torch.full((cfg.slots, steps + 2, 3, 3 * dim), 7,
                        dtype=torch.bfloat16, device="cuda")
    for name, part in zip("qkv", window[:, :steps].split(dim, dim=-1)):
        direct[f"intermediate_conv_{name}"] = part.transpose(-1, -2)
    contiguous = clone_inputs(direct)
    readonly = [mixed, gate, beta, conv, direct["recurrent_state"]]
    before = [value.clone() for value in readonly]
    expected_window = window.clone()

    expected_out = fused_kda_decode_mtp_dspark(**contiguous)
    direct_out = fused_kda_decode_mtp_dspark(**direct)
    torch.cuda.synchronize()
    assert torch.equal(direct_out, expected_out), "strided output differs"
    assert torch.count_nonzero(direct_out[:, -steps:]).item() == 0, "padding output"
    assert direct_out.is_contiguous(), "default output layout"
    for name, part in zip("qkv", expected_window[:, :steps].split(dim, dim=-1)):
        part.transpose(-1, -2).copy_(contiguous[f"intermediate_conv_{name}"])
    assert torch.equal(window, expected_window), "window or untouched scratch differs"
    for name in ("rawv", "rawk", "g", "beta"):
        key = f"replayssm_{name}"
        assert torch.equal(direct[key], contiguous[key]), f"{key} differs"
    for old, value in zip(before, readonly):
        assert torch.equal(old, value), "verify mutated an input or persistent state"

    # Replay uses the same native views and must preserve padding/spare storage.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_kda_decode_mtp_dspark(**direct, out=direct_out)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(direct_out, expected_out), "graph output differs"
    assert torch.equal(window, expected_window), "graph scratch differs"
    for name in ("rawv", "rawk", "g", "beta"):
        key = f"replayssm_{name}"
        assert torch.equal(direct[key], contiguous[key]), f"graph {key} differs"
    for old, value in zip(before, readonly):
        assert torch.equal(old, value), "graph mutated an input or persistent state"
    print(f"PASS B={batch} H={heads} T={steps}: exact outputs/rings/windows, "
          "read-only inputs/state, padding/spare scratch, graph replay")


def main():
    print(gpu_banner())
    # Default policy exercises VSPLIT=2 and VSPLIT=1 respectively.
    run_case(4, 4, 4)
    run_case(32, 12, 5)
    print("RESULT: all PASS")


if __name__ == "__main__":
    main()
