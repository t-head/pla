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

"""Canonical packed multi-sequence FlashQLA acceptance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from private import ref_gdr
from e2e_acc_full import REF_COS_MIN, tensor_metrics
from pla.prefill.flashqla import chunk_gated_delta_rule_fwd
from pla.prefill.flashqla.private.cp_preprocess import (
    _head_shards_from_compute_capability,
    calc_cp_seqs,
)


def run_case(lengths, h, hg, dtype, gate_mode, expected_cp, seed):
    torch.manual_seed(seed)
    total = sum(lengths)
    q = F.normalize(torch.randn(1, total, hg, 128, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(1, total, hg, 128, device="cuda"), dim=-1).to(dtype)
    v = (torch.randn(1, total, h, 128, device="cuda") * 0.1).to(dtype)
    if gate_mode == "slow":
        g = torch.full(
            (1, total, h), -1e-3, device="cuda", dtype=torch.float32)
    else:
        g = F.logsigmoid(
            torch.randn(1, total, h, device="cuda", dtype=torch.float32))
    beta = torch.rand(1, total, h, device="cuda", dtype=torch.float32).sigmoid()
    h0 = (torch.randn(len(lengths), h, 128, 128, device="cuda") * 0.02).float()
    cu = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device="cuda", dtype=torch.int32,
    )
    use_cp, _, _, _, _ = calc_cp_seqs(cu, 32, h)
    planner_ok = use_cp == expected_cp

    _, packed_a, packed_o, _, packed_h = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, initial_state=h0, cu_seqlens=cu, auto_cp=True)
    _, no_cp_a, no_cp_o, _, no_cp_h = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, initial_state=h0, cu_seqlens=cu, auto_cp=False)
    _, ref_o, _, _, ref_h = ref_gdr.chunk_gated_delta_rule_fwd(
        q.float(), k.float(), v.float(), g, beta,
        cu_seqlens=cu, initial_state=h0, chunk_size=32,
    )

    separate_a, separate_o, separate_h = [], [], []
    start = 0
    for index, length in enumerate(lengths):
        end = start + length
        one_cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        _, one_a, one_o, _, one_h = chunk_gated_delta_rule_fwd(
            q[:, start:end], k[:, start:end], v[:, start:end],
            g[:, start:end], beta[:, start:end],
            initial_state=h0[index:index + 1], cu_seqlens=one_cu, auto_cp=False,
        )
        separate_a.append(one_a)
        separate_o.append(one_o)
        separate_h.append(one_h)
        start = end
    separate_a = torch.cat(separate_a, dim=1)
    separate_o = torch.cat(separate_o, dim=1)
    separate_h = torch.cat(separate_h, dim=0)

    rel_max = 1e-2 if dtype == torch.bfloat16 else 5e-3
    metrics = {
        "o_ref": tensor_metrics(packed_o, ref_o),
        "h_ref": tensor_metrics(packed_h, ref_h),
        "a_no_cp": tensor_metrics(packed_a, no_cp_a),
        "o_no_cp": tensor_metrics(packed_o, no_cp_o),
        "h_no_cp": tensor_metrics(packed_h, no_cp_h),
        "a_serial": tensor_metrics(packed_a, separate_a),
        "o_serial": tensor_metrics(packed_o, separate_o),
        "h_serial": tensor_metrics(packed_h, separate_h),
    }
    ok = planner_ok and all(
        finite and cos > REF_COS_MIN and rel < rel_max
        for cos, rel, finite in metrics.values()
    )
    summary = " ".join(
        f"{name}={cos:.5f}/{rel:.2e}"
        for name, (cos, rel, _) in metrics.items()
    )
    print(
        f"  lengths={lengths} H={h} Hg={hg} "
        f"dtype={str(dtype).replace('torch.', '', 1)} gate={gate_mode} "
        f"cp={use_cp}/{expected_cp} "
        f"{summary} {'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_padded_case(lengths, h, hg, dtype, padding, seed):
    """Compare a physical padded call with its real-token-only equivalent."""
    torch.manual_seed(seed)
    real_tokens = sum(lengths)
    physical_tokens = real_tokens + padding
    q = F.normalize(
        torch.randn(1, physical_tokens, hg, 128, device="cuda"), dim=-1
    ).to(dtype)
    k = F.normalize(
        torch.randn(1, physical_tokens, hg, 128, device="cuda"), dim=-1
    ).to(dtype)
    v = (torch.randn(1, physical_tokens, h, 128, device="cuda") * 0.1).to(dtype)
    g = F.logsigmoid(
        torch.randn(1, physical_tokens, h, device="cuda", dtype=torch.float32))
    beta = torch.rand(
        1, physical_tokens, h, device="cuda", dtype=torch.float32).sigmoid()
    h0 = (torch.randn(len(lengths), h, 128, 128, device="cuda") * 0.02).float()
    cu = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device="cuda", dtype=torch.int32,
    )
    raw_cu = tuple([0] + torch.tensor(lengths).cumsum(0).tolist())

    rel_max = 1e-2 if dtype == torch.bfloat16 else 5e-3
    all_ok = True
    summaries = []
    for auto_cp in (False, True):
        padded_g, padded_a, padded_o, _, padded_h = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, initial_state=h0, cu_seqlens=cu,
            auto_cp=auto_cp, raw_cu_seqlens=raw_cu)
        real_g, real_a, real_o, _, real_h = chunk_gated_delta_rule_fwd(
            q[:, :real_tokens].contiguous(),
            k[:, :real_tokens].contiguous(),
            v[:, :real_tokens].contiguous(),
            g[:, :real_tokens].contiguous(),
            beta[:, :real_tokens].contiguous(),
            initial_state=h0, cu_seqlens=cu, auto_cp=auto_cp,
            raw_cu_seqlens=raw_cu)
        _, ref_o, _, _, ref_h = ref_gdr.chunk_gated_delta_rule_fwd(
            q[:, :real_tokens].float(), k[:, :real_tokens].float(),
            v[:, :real_tokens].float(), g[:, :real_tokens],
            beta[:, :real_tokens], cu_seqlens=cu, initial_state=h0,
            chunk_size=32)

        metrics = {
            "g_real": tensor_metrics(padded_g[:, :real_tokens], real_g),
            "a_real": tensor_metrics(padded_a[:, :real_tokens], real_a),
            "o_real": tensor_metrics(padded_o[:, :real_tokens], real_o),
            "h_real": tensor_metrics(padded_h, real_h),
            "o_ref": tensor_metrics(padded_o[:, :real_tokens], ref_o),
            "h_ref": tensor_metrics(padded_h, ref_h),
        }
        numerical_ok = all(
            finite and cos > REF_COS_MIN and rel < rel_max
            for cos, rel, finite in metrics.values())
        shape_ok = padded_o.shape == (1, physical_tokens, h, 128)
        tail_zero = torch.count_nonzero(padded_o[:, real_tokens:]).item() == 0
        ok = numerical_ok and shape_ok and tail_zero
        all_ok &= ok
        summaries.append(
            f"cp={auto_cp} o_real={metrics['o_real'][1]:.2e} "
            f"o_ref={metrics['o_ref'][1]:.2e} tail_zero={tail_zero} "
            f"{'PASS' if ok else 'FAIL'}")

    print(
        f"  PADDED lengths={lengths} physical={physical_tokens} "
        f"real={real_tokens} pad={padding} H={h} Hg={hg} "
        f"dtype={str(dtype).replace('torch.', '', 1)} "
        + " | ".join(summaries))
    return all_ok


def main():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    is_tc01 = _head_shards_from_compute_capability(
        int(props.major), int(props.minor)) == 2
    cases = (
        ((64, 96), 16, 8, torch.bfloat16, "normal", False, 42),
        ((513, 1025), 16, 8, torch.float16, "normal", False, 123),
        # The verified tc01 planner splits this underfilled long tail; tc02's
        # newer occupancy model keeps it serial.
        ((2048, 128), 16, 8, torch.bfloat16, "normal", is_tc01, 2026),
        # Both architectures split this uneven H=8 packed batch.
        ((509, 3356, 3659, 668), 8, 4,
         torch.bfloat16, "normal", True, 2027),
        ((1024, 1024), 6, 2, torch.bfloat16, "normal", is_tc01, 42),
        ((1025, 2049), 6, 2, torch.bfloat16, "slow", True, 123),
        ((3041, 769), 6, 2, torch.bfloat16, "slow", True, 321),
        # This uneven long H=6 pair is beyond tc01's measured packed CP
        # amortization point and exercises the positive planner path.
        ((16384, 4096), 6, 2, torch.bfloat16, "slow", True, 456),
    )
    passed = sum(run_case(*case) for case in cases)
    padded_cases = (
        # Exact SGLang alignment failure: 2205 real tokens in a 2206-token
        # physical tensor. Batch=1 specifically covers the KKT implicit-grid
        # path that used to derive one extra chunk from the physical length.
        ((2205,), 16, 8, torch.bfloat16, 1, 2206),
        # Ragged packed input with a larger suffix verifies that padding cannot
        # leak across sequence/state boundaries.
        ((100, 273, 16, 272), 12, 4, torch.bfloat16, 7, 668),
    )
    padded_passed = sum(run_padded_case(*case) for case in padded_cases)
    total_passed = passed + padded_passed
    total_cases = len(cases) + len(padded_cases)
    print(f"MULTI-SEQUENCE ACCEPTANCE: {total_passed}/{total_cases}")
    return 0 if total_passed == total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
