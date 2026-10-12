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

"""Correctness + performance check: PLA FlashQLA vs Triton for GDN prefill.

Both paths receive identical inputs, including the indexed state pool. The PLA
path calls its public API directly; the Triton baseline is kept under
``scripts/prefill/private/ref``.

Usage:
    CUDA_VISIBLE_DEVICES=0 \
      python3 scripts/prefill/flashqla/bench_vs_fla.py \
        --num-v-heads 32 --num-k-heads 16 --seq-lens 1024,4096 --ragged
"""

import argparse
import functools
import math
from pathlib import Path
import statistics
import sys

import torch

from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

HEAD_DIM = 128


@functools.lru_cache(maxsize=1)
def _load_triton_chunk_gated_delta_rule():
    ref_root = str(Path(__file__).resolve().parents[1] / "private")
    if ref_root not in sys.path:
        sys.path.insert(0, ref_root)
    from ref.chunk import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def make_inputs(seq_lens, num_k_heads, num_v_heads, dtype, device, num_slots):
    """Build one packed extend batch plus a fresh ssm_state pool."""
    total = sum(seq_lens)
    q = torch.randn(1, total, num_k_heads, HEAD_DIM, device=device, dtype=dtype) * 0.1
    k = torch.randn(1, total, num_k_heads, HEAD_DIM, device=device, dtype=dtype) * 0.1
    v = torch.randn(1, total, num_v_heads, HEAD_DIM, device=device, dtype=dtype) * 0.1
    # Match the FlashQLA/OperatorsForge prefill workload: slow decay on 75% of
    # the heads and no decay on the remaining heads.
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total, num_v_heads, device=device, dtype=torch.float32)
    ) / 16
    swa_mask = torch.zeros(num_v_heads, dtype=torch.bool, device=device)
    swa_mask[: math.ceil(0.75 * num_v_heads)] = True
    swa_mask = swa_mask[torch.randperm(num_v_heads, device=device)]
    g[:, :, ~swa_mask] = 0.0
    beta = torch.randn(
        1, total, num_v_heads, device=device, dtype=torch.float32
    ).sigmoid()

    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cache_indices = torch.arange(len(seq_lens), dtype=torch.int32, device=device)

    # Pool layout is [slots, HV, V, K] (K-contiguous), fp32 by default.
    pool = (
        torch.randn(
            num_slots,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )
    return q, k, v, g, beta, query_start_loc, cache_indices, pool


def run_once(inputs, pool_init, *, use_pla):
    q, k, v, g, beta, query_start_loc, cache_indices, _ = inputs
    pool = pool_init.clone()
    if use_pla:
        _, _, o, _, _ = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            output_final_state=False,
            normalize_qk=True,
        )
        return o, pool

    o, _, _ = _load_triton_chunk_gated_delta_rule()(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=pool,
        initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return o, pool


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def timed_ms(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="512,1024,2048,4096,8192",
        help="Comma-separated single-sequence lengths to sweep.",
    )
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="Also run a multi-sequence ragged batch.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--tol",
        type=float,
        default=2e-2,
        help="Max relative error vs the Triton reference.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda"

    cases = [[int(s)] for s in args.seq_lens.split(",") if s]
    if args.ragged:
        cases.append([4096, 137, 2048, 63])

    print(
        f"H(v_heads)={args.num_v_heads} Hg(k_heads)={args.num_k_heads} "
        f"D={HEAD_DIM} dtype={args.dtype}"
    )
    header = (
        f"{'seq_lens':>26} {'o_err':>10} {'state_err':>10} "
        f"{'triton_ms':>10} {'pla_ms':>10} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    failures = []
    for seq_lens in cases:
        torch.manual_seed(0)
        inputs = make_inputs(
            seq_lens,
            args.num_k_heads,
            args.num_v_heads,
            dtype,
            device,
            num_slots=len(seq_lens) + 1,
        )
        pool_init = inputs[-1]

        out_ref, pool_ref = run_once(inputs, pool_init, use_pla=False)
        out_pla, pool_pla = run_once(inputs, pool_init, use_pla=True)

        o_err = rel_err(out_pla, out_ref)
        # Only the slots this batch actually owns are written.
        used = inputs[6].to(torch.long)
        state_err = rel_err(pool_pla[used], pool_ref[used])

        t_triton = timed_ms(
            lambda: run_once(inputs, pool_init, use_pla=False), args.warmup, args.repeat
        )
        t_pla = timed_ms(
            lambda: run_once(inputs, pool_init, use_pla=True), args.warmup, args.repeat
        )

        label = ",".join(str(s) for s in seq_lens)
        if len(label) > 25:
            label = label[:22] + "..."
        print(
            f"{label:>26} {o_err:>10.3e} {state_err:>10.3e} "
            f"{t_triton:>10.3f} {t_pla:>10.3f} {t_triton / t_pla:>7.2f}x"
        )
        if o_err > args.tol or state_err > args.tol:
            failures.append((label, o_err, state_err))

    if failures:
        print("\nFAILED (relative error above tolerance):")
        for label, o_err, state_err in failures:
            print(f"  seq_lens={label}: o_err={o_err:.3e} state_err={state_err:.3e}")
        raise SystemExit(1)
    print("\nAll cases within tolerance.")


if __name__ == "__main__":
    main()
