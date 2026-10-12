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

"""Accuracy regression for OPT-09 raw GDN gating fused into QLA cumsum."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pla.prefill.flashqla import (
    chunk_gated_delta_rule_fwd,
    gdn_prefill_packed_qkv_fwd,
)


D = 128
H = 16
HG = 8


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-12)
    )


def run_case(name: str, physical_tokens: int, offsets: tuple[int, ...], auto_cp: bool):
    torch.manual_seed(20260826)
    q = (torch.randn(1, physical_tokens, HG, D, device="cuda") * 0.1).bfloat16()
    k = (torch.randn_like(q.float()) * 0.1).bfloat16()
    v = (torch.randn(1, physical_tokens, H, D, device="cuda") * 0.1).bfloat16()
    # Match Qwen3.5's MergedColumnParallelLinear output: b/a are adjacent
    # column slices and contiguous only in the head dimension.
    raw_ba = (
        torch.randn(physical_tokens, 2 * H, device="cuda") * 0.1
    ).bfloat16()
    b, a = raw_ba.split(H, dim=-1)
    A_log = torch.linspace(-0.4, 0.2, H, device="cuda", dtype=torch.float32)
    dt_bias = torch.linspace(-0.2, 0.1, H, device="cuda", dtype=torch.float32)
    g = -A_log.exp()[None, :] * F.softplus(a.float() + dt_bias[None, :])
    beta = b.float().sigmoid()
    g = g.unsqueeze(0).contiguous()
    beta = beta.unsqueeze(0).contiguous()
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    indices = torch.arange(len(offsets) - 1, device="cuda", dtype=torch.int32)
    initial = torch.randn(
        len(offsets) - 1, H, D, D, device="cuda", dtype=torch.float32
    ) * 0.01
    baseline_pool = initial.clone()
    candidate_pool = initial.clone()
    packed_pool = initial.clone()

    qk_width = HG * D
    packed_width = 2 * qk_width + H * D
    # Exercise both a non-zero storage offset and a token pitch larger than the
    # logical Q|K|V row, as produced by a sliced post-convolution workspace.
    packed_storage = torch.empty(
        physical_tokens, packed_width + 64,
        device="cuda", dtype=q.dtype)
    packed = packed_storage[:, 17:17 + packed_width]
    packed[:, :qk_width].copy_(q.view(physical_tokens, qk_width))
    packed[:, qk_width:2 * qk_width].copy_(
        k.view(physical_tokens, qk_width))
    packed[:, 2 * qk_width:].copy_(
        v.view(physical_tokens, H * D))

    baseline = chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=1.0 / math.sqrt(D),
        initial_state=baseline_pool,
        initial_state_indices=indices,
        cu_seqlens=cu,
        raw_cu_seqlens=offsets,
        output_final_state=False,
        auto_cp=auto_cp,
        normalize_qk=True,
    )[2]
    candidate = chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        None,
        None,
        scale=1.0 / math.sqrt(D),
        initial_state=candidate_pool,
        initial_state_indices=indices,
        cu_seqlens=cu,
        raw_cu_seqlens=offsets,
        output_final_state=False,
        auto_cp=auto_cp,
        normalize_qk=True,
        raw_gate_a=a,
        raw_beta_b=b,
        A_log=A_log,
        dt_bias=dt_bias,
    )[2]
    packed_output = gdn_prefill_packed_qkv_fwd(
        packed,
        a,
        b,
        A_log,
        dt_bias,
        cu,
        num_qk_heads=HG,
        num_v_heads=H,
        scale=1.0 / math.sqrt(D),
        initial_state=packed_pool,
        initial_state_indices=indices,
        raw_cu_seqlens=offsets,
        output_final_state=False,
        auto_cp=auto_cp,
    )[2]

    real_tokens = offsets[-1]
    output_rel = relative_error(candidate[:, :real_tokens], baseline[:, :real_tokens])
    state_rel = relative_error(candidate_pool[indices], baseline_pool[indices])
    padding_zero = bool(torch.count_nonzero(candidate[:, real_tokens:]).item() == 0)
    packed_output_equal = torch.equal(packed_output, candidate)
    packed_state_equal = torch.equal(packed_pool, candidate_pool)
    if (output_rel >= 2e-3 or state_rel >= 2e-3 or not padding_zero or
            not packed_output_equal or not packed_state_equal):
        raise AssertionError(
            f"{name} auto_cp={auto_cp}: output_rel={output_rel:.6g} "
            f"state_rel={state_rel:.6g} padding_zero={padding_zero} "
            f"packed_output_equal={packed_output_equal} "
            f"packed_state_equal={packed_state_equal}"
        )
    print(
        f"PASS {name} auto_cp={auto_cp} output_rel={output_rel:.6g} "
        f"state_rel={state_rel:.6g} padding_zero={padding_zero} "
        f"packed_equal=True stride={packed.stride()}"
    )


def run_legacy_packed_case(dtype: torch.dtype, auto_cp: bool):
    """Cover non-indexed state ABI and H12's automatic-CP region."""
    physical_tokens, real_tokens = 2206, 2205
    H_legacy, HG_legacy = 12, 4
    offsets = (0, real_tokens)
    torch.manual_seed(20260827 + int(auto_cp))
    q = (torch.randn(
        1, physical_tokens, HG_legacy, D, device="cuda") * 0.1).to(dtype)
    k = (torch.randn_like(q.float()) * 0.1).to(dtype)
    v = (torch.randn(
        1, physical_tokens, H_legacy, D, device="cuda") * 0.1).to(dtype)
    ba = (torch.randn(
        physical_tokens, 2 * H_legacy, device="cuda") * 0.1).to(dtype)
    b, a = ba.split(H_legacy, dim=-1)
    A_log = torch.linspace(
        -0.4, 0.2, H_legacy, device="cuda", dtype=torch.float32)
    dt_bias = torch.linspace(
        -0.2, 0.1, H_legacy, device="cuda", dtype=torch.float32)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    qk_width = HG_legacy * D
    packed_width = 2 * qk_width + H_legacy * D
    storage = torch.empty(
        physical_tokens, packed_width + 64, device="cuda", dtype=dtype)
    packed = storage[:, 17:17 + packed_width]
    packed[:, :qk_width].copy_(q.view(physical_tokens, qk_width))
    packed[:, qk_width:2 * qk_width].copy_(
        k.view(physical_tokens, qk_width))
    packed[:, 2 * qk_width:].copy_(
        v.view(physical_tokens, H_legacy * D))

    split = chunk_gated_delta_rule_fwd(
        q, k, v, None, None,
        cu_seqlens=cu,
        raw_cu_seqlens=offsets,
        output_final_state=True,
        auto_cp=auto_cp,
        normalize_qk=True,
        raw_gate_a=a,
        raw_beta_b=b,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    direct = gdn_prefill_packed_qkv_fwd(
        packed, a, b, A_log, dt_bias, cu,
        num_qk_heads=HG_legacy,
        num_v_heads=H_legacy,
        raw_cu_seqlens=offsets,
        output_final_state=True,
        auto_cp=auto_cp,
    )
    output_equal = torch.equal(direct[2], split[2])
    state_equal = torch.equal(direct[4], split[4])
    padding_zero = bool(
        torch.count_nonzero(direct[2][:, real_tokens:]).item() == 0)
    if not output_equal or not state_equal or not padding_zero:
        raise AssertionError(
            f"legacy-packed dtype={dtype} auto_cp={auto_cp}: "
            f"output_equal={output_equal} state_equal={state_equal} "
            f"padding_zero={padding_zero}"
        )
    print(
        f"PASS legacy-packed dtype={dtype} auto_cp={auto_cp} "
        f"packed_equal=True padding_zero=True stride={packed.stride()}"
    )


def main():
    cases = (
        ("single-padded", 2206, (0, 2205)),
        ("ragged-padded", 668, (0, 127, 384, 661)),
    )
    for name, physical_tokens, offsets in cases:
        for auto_cp in (False, True):
            run_case(name, physical_tokens, offsets, auto_cp)
    for dtype in (torch.float16, torch.bfloat16):
        for auto_cp in (False, True):
            run_legacy_packed_case(dtype, auto_cp)


if __name__ == "__main__":
    main()
