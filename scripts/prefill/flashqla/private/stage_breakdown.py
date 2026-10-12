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

"""Measure each CUDA stage used by the public FlashQLA forward API."""

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

from pla.prefill.flashqla import chunk_gated_delta_rule_fwd
from pla.prefill.flashqla import ops


STAGES = (
    "chunk_local_cumsum",
    "kkt_solve",
    "get_warmup_chunks",
    "prepare_h",
    "correct_h0",
    "fused_fwd",
)


def make_case(
    batch: int,
    heads: int,
    kv_heads: int,
    tokens: int,
    cu_values: list[int] | None,
    dtype: torch.dtype,
    constant_inputs: bool,
):
    torch.manual_seed(0)
    if cu_values is None:
        total = batch * tokens
        cu_values = list(range(0, total + 1, tokens))
    else:
        total = cu_values[-1]
    if constant_inputs:
        q = torch.full(
            (1, total, kv_heads, 128), 1.0 / math.sqrt(128),
            dtype=dtype, device="cuda",
        )
        k = torch.full_like(q, 1.0 / math.sqrt(128))
        v = torch.full(
            (1, total, heads, 128), 0.01, dtype=dtype, device="cuda",
        )
        g = torch.full(
            (1, total, heads), -0.6931471805599453,
            dtype=torch.float32, device="cuda",
        )
        beta = torch.full_like(g, 0.5)
    else:
        q = F.normalize(
            (torch.randn(1, total, kv_heads, 128, device="cuda") * 0.1).float(),
            p=2, dim=-1).to(dtype)
        k = F.normalize(
            (torch.randn(1, total, kv_heads, 128, device="cuda") * 0.1).float(),
            p=2, dim=-1).to(dtype)
        v = (torch.randn(1, total, heads, 128, device="cuda") * 0.1).to(dtype)
        g = (
            F.logsigmoid(torch.randn(1, total, heads, device="cuda").float())
            / 16
        )
        swa_mask = torch.zeros(heads, dtype=torch.bool, device="cuda")
        swa_mask[: math.ceil(0.75 * heads)] = True
        swa_mask = swa_mask[torch.randperm(heads, device="cuda")]
        g[:, :, ~swa_mask] = 0.0
        beta = torch.randn(1, total, heads, device="cuda").float().sigmoid()
    cu = torch.tensor(cu_values, dtype=torch.int32, device="cuda")
    return q, k, v, g, beta, cu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument(
        "--cu-seqlens",
        help="comma-separated cumulative lengths; overrides --batch/--tokens",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--constant-inputs", action="store_true")
    parser.add_argument("--auto-cp", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    cu_values = (
        [int(value) for value in args.cu_seqlens.split(",")]
        if args.cu_seqlens else None
    )
    q, k, v, g, beta, cu = make_case(
        args.batch,
        args.heads,
        args.kv_heads or args.heads // 2,
        args.tokens,
        cu_values,
        dtype,
        args.constant_inputs,
    )
    call_args = (q, k, v, g, beta)
    call_kwargs = {
        "scale": 1.0 / math.sqrt(128),
        "cu_seqlens": cu,
        "output_final_state": True,
        "auto_cp": args.auto_cp,
    }
    for _ in range(args.warmup):
        chunk_gated_delta_rule_fwd(*call_args, **call_kwargs)
    torch.cuda.synchronize()

    samples = {name: [] for name in STAGES}
    originals = {}
    for name in STAGES:
        original = getattr(ops.flashqla_cute, name)
        originals[name] = original

        def wrapper(*fn_args, _name=name, _fn=original, **fn_kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _fn(*fn_args, **fn_kwargs)
            end.record()
            end.synchronize()
            samples[_name].append(start.elapsed_time(end))
            return result

        setattr(ops.flashqla_cute, name, wrapper)

    try:
        for _ in range(args.repeat):
            chunk_gated_delta_rule_fwd(*call_args, **call_kwargs)
    finally:
        for name, original in originals.items():
            setattr(ops.flashqla_cute, name, original)

    total = 0.0
    for name in STAGES:
        values = samples[name]
        if values:
            value = statistics.median(values)
            total += value
            print(f"{name},{value:.6f}")
    print(f"cuda_stage_sum,{total:.6f}")


if __name__ == "__main__":
    main()
