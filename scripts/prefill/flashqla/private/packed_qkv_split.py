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

"""Fused Triton split for a post-convolution packed Q|K|V buffer.

This module intentionally has no PLA API dependency.  It is shared by the
focused OPT-07 benchmark and generalized replay, including split baselines run
against wheels that predate the public packed-QKV entry point.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


D = 128


@triton.jit
def _split_qkv_kernel(
    q,
    k,
    v,
    packed,
    packed_stride_t: tl.constexpr,
    packed_stride_d: tl.constexpr,
    q_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, block)
    qk_dim: tl.constexpr = q_dim + k_dim
    total: tl.constexpr = qk_dim + v_dim
    values = tl.load(
        packed + token * packed_stride_t + offsets * packed_stride_d,
        mask=offsets < total,
    )
    tl.store(q + token * q_dim + offsets, values, mask=offsets < q_dim)
    k_offsets = offsets - q_dim
    tl.store(
        k + token * k_dim + k_offsets,
        values,
        mask=(offsets >= q_dim) & (offsets < qk_dim),
    )
    v_offsets = offsets - qk_dim
    tl.store(
        v + token * v_dim + v_offsets,
        values,
        mask=(offsets >= qk_dim) & (offsets < total),
    )


def split_qkv(packed: torch.Tensor, H: int, Hg: int):
    T = packed.size(0)
    q = torch.empty((1, T, Hg, D), device=packed.device, dtype=packed.dtype)
    k = torch.empty_like(q)
    v = torch.empty((1, T, H, D), device=packed.device, dtype=packed.dtype)
    q_dim = Hg * D
    v_dim = H * D
    _split_qkv_kernel[(T,)](
        q,
        k,
        v,
        packed,
        packed.stride(0),
        packed.stride(1),
        q_dim,
        q_dim,
        v_dim,
        block=triton.next_power_of_2(2 * q_dim + v_dim),
        num_warps=8,
        num_stages=3,
    )
    return q, k, v
