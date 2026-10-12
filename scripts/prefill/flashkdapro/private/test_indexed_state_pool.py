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

"""OPT-06 regression: direct indexed bf16/fp32 KDA state-pool I/O."""

import os

import torch

from pla.prefill.flashkdapro import flashkda_fwd


def _set_dispatch(cp: str, nows: str) -> None:
    os.environ["PLA_FLASHKDA_CP"] = cp
    os.environ["PLA_FLASHKDA_K2_NOWS"] = nows


def _make_inputs(lengths, heads=6, padding=1):
    real_tokens = sum(lengths)
    tokens = real_tokens + padding
    q = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta_storage = torch.randn(
        tokens, heads + 3, device="cuda", dtype=torch.bfloat16
    )
    beta = beta_storage[:, :heads]
    a_log = torch.rand(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(heads, 128, device="cuda", dtype=torch.float32)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    return q, k, v, g, beta, a_log, dt_bias, cu, real_tokens, tuple(offsets)


def _run_case(name, lengths, state_dtype, *, cp, nows):
    _set_dispatch(cp, nows)
    inputs = _make_inputs(lengths)
    q, k, v, g, beta, a_log, dt_bias, cu, real_tokens, raw_offsets = inputs
    nseq, heads = len(lengths), q.shape[1]
    initial = torch.randn(
        nseq, heads, 128, 128, device="cuda", dtype=state_dtype
    )

    compact_out = torch.full_like(v, 11)
    compact_final = torch.empty_like(initial)
    flashkda_fwd(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        compact_out,
        initial_state=initial,
        final_state=compact_final,
        cu_seqlens=cu,
    )

    contiguous_out = torch.full_like(v, 17)
    contiguous_final = torch.empty_like(initial)
    flashkda_fwd(
        q,
        k,
        v,
        g,
        beta.contiguous(),
        a_log,
        dt_bias,
        contiguous_out,
        initial_state=initial,
        final_state=contiguous_final,
        cu_seqlens=cu,
    )

    slots = nseq + 4
    indices = torch.tensor(
        [slots - 1 - 2 * i for i in range(nseq)],
        device="cuda",
        dtype=torch.int32,
    )
    pool = torch.randn(
        slots, heads, 128, 128, device="cuda", dtype=state_dtype
    )
    pool[indices.to(torch.int64)] = initial
    pool_before = pool.clone()
    indexed_out = torch.full_like(v, 13)
    flashkda_fwd(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        indexed_out,
        cu_seqlens=cu,
        state_pool=pool,
        state_indices=indices,
        raw_cu_seqlens=raw_offsets,
    )
    torch.cuda.synchronize()

    selected = indices.to(torch.int64)
    assert torch.equal(indexed_out, compact_out), f"{name}: output mismatch"
    assert torch.equal(pool[selected], compact_final), f"{name}: state mismatch"
    assert torch.equal(contiguous_out, compact_out), (
        f"{name}: strided/contiguous beta output mismatch"
    )
    assert torch.equal(contiguous_final, compact_final), (
        f"{name}: strided/contiguous beta state mismatch"
    )
    assert torch.count_nonzero(indexed_out[real_tokens:]).item() == 0, (
        f"{name}: padded output is not zero"
    )
    selected_set = set(indices.cpu().tolist())
    untouched = [slot for slot in range(slots) if slot not in selected_set]
    assert torch.equal(pool[untouched], pool_before[untouched]), (
        f"{name}: unmapped pool slot changed"
    )
    print(f"PASS {name} dtype={state_dtype} indices={indices.tolist()}")


def main():
    torch.manual_seed(20260826)
    modes = (
        ("baseline", [257, 193], "0", "0"),
        ("nows", [257, 193], "0", "1"),
        ("cp", [1100, 900], "2", "1"),
    )
    for name, lengths, cp, nows in modes:
        for state_dtype in (torch.bfloat16, torch.float32):
            _run_case(name, lengths, state_dtype, cp=cp, nows=nows)
    print("OPT-06 INDEXED STATE POOL: 6/6 passed")


if __name__ == "__main__":
    main()
