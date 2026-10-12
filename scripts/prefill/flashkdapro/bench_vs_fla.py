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

"""Correctness + performance check: PLA FlashKDA vs SGLang Triton.

The Triton path uses the SGLang ``chunk_kda`` reference kept under
``scripts/prefill/private/ref`` with the same runtime tensor semantics: raw q/k
and gate logits, an indexed fp32 state pool, packed sequence offsets, and
in-kernel q/k L2 norm.
The PLA path consumes the same source tensors and state slots through
``pla.prefill.flashkdapro.flashkda_fwd``.

Usage:
    CUDA_VISIBLE_DEVICES=0 \
      python3 scripts/prefill/flashkdapro/bench_vs_fla.py \
        --num-heads 12 --seq-lens 1024,4096,8192 --ragged
"""

import argparse
import functools
import math
from pathlib import Path
import statistics
import sys

import torch

from pla.prefill.flashkdapro import flashkda_fwd

DEV = "cuda"
D = 128
HEAD_DIM = D
LOWER_BOUND = -5.0


@functools.lru_cache(maxsize=1)
def _load_triton_chunk_kda():
    ref_root = str(Path(__file__).resolve().parents[1] / "private")
    if ref_root not in sys.path:
        sys.path.insert(0, ref_root)
    from ref.kda import chunk_kda

    return chunk_kda


def err_ratio(actual, expected):
    """Relative L2 error retained for the private benchmark helpers."""
    actual = actual.float()
    expected = expected.float()
    return (
        (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    ).item()


def runtime_gate(g, A_log, dt_bias, lower):
    """Reference form of SGLang's lower-bound KDA gate activation."""
    head_count = g.shape[-2]
    gate_scale = torch.exp(A_log.float()).view(1, 1, head_count, 1)
    bias = dt_bias.float().view(1, 1, head_count, D)
    return lower * torch.sigmoid(g.float() * gate_scale + bias * gate_scale)


# Backward-compatible name used by older private validation scripts.
blueprint_gate = runtime_gate


def run_fla_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    h0,
    *,
    vendored_gate=False,
    external_gate=None,
    extra=None,
    pre_normalized=False,
):
    """Compatibility helper backed by the private SGLang Triton reference.

    Several private sweeps import this historical function name. Keep the
    signature, but use the copied kernel and indexed state-pool contract
    instead of importing an arbitrary installed upstream ``fla`` package.
    """
    if external_gate is not None:
        vendored_gate = external_gate
    extra = dict(extra or {})
    cu_seqlens = extra.pop("cu_seqlens", None)
    if extra:
        names = ", ".join(sorted(extra))
        raise TypeError(f"unsupported SGLang KDA arguments: {names}")

    n_seq = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    head_count = q.shape[-2]
    if h0 is None:
        pool = torch.zeros(
            n_seq,
            head_count,
            D,
            D,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        pool = h0.float().clone()
    state_indices = torch.arange(n_seq, dtype=torch.int32, device=q.device)
    runtime_cu = None if cu_seqlens is None else cu_seqlens.to(torch.int32)

    if vendored_gate:
        gate_arg = runtime_gate(g, A_log, dt_bias, LOWER_BOUND)
        gate_kwargs = {}
    else:
        gate_arg = g
        gate_kwargs = {
            "A_log": A_log,
            "dt_bias": dt_bias,
            "lower_bound": LOWER_BOUND,
        }

    # PLA takes beta logits and applies sigmoid internally. SGLang's runtime
    # entry takes the already activated beta emitted by the model gating path.
    beta_activated = beta.float().sigmoid()
    # SGLang reuses ``v`` as its output buffer. Clone it so callers keep the
    # source tensor and repeated comparisons always start from identical data.
    output = _load_triton_chunk_kda()(
        q=q,
        k=k,
        v=v.clone(),
        g=gate_arg,
        beta=beta_activated,
        scale=scale,
        initial_state=pool,
        initial_state_indices=state_indices,
        use_qk_l2norm_in_kernel=not pre_normalized,
        cu_seqlens=runtime_cu,
        **gate_kwargs,
    )
    return output, pool.index_select(0, state_indices.to(torch.long))


def run_ppu_flashkda(
    q, k, v, g, beta, A_log, dt_bias, scale, h0, cu_seqlens=None
):
    """Compatibility helper for the existing private FlashKDA sweeps."""
    total, head_count, head_dim = q.shape[1:]
    n_seq = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    output = torch.empty(
        total, head_count, head_dim, dtype=torch.bfloat16, device=q.device
    )
    final_state = torch.empty(
        n_seq,
        head_count,
        head_dim,
        head_dim,
        device=q.device,
        dtype=h0.dtype if h0 is not None else torch.bfloat16,
    )
    flashkda_fwd(
        q[0],
        k[0],
        v[0],
        g[0],
        beta[0],
        A_log,
        dt_bias,
        output,
        scale,
        LOWER_BOUND,
        h0,
        final_state,
        cu_seqlens,
    )
    return output.unsqueeze(0), final_state


def bench_fn(fn, warmup, iters, repeats):
    """Legacy mean/min/max timer retained for private sweep imports."""
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        samples.extend(start.elapsed_time(end) for start, end in zip(starts, ends))

    return statistics.mean(samples), min(samples), max(samples)


def make_inputs(seq_lens, num_heads, dtype, device, num_slots):
    """Build one packed KDA extend batch plus an indexed state pool."""
    total = sum(seq_lens)
    shape = (1, total, num_heads, HEAD_DIM)
    q = torch.randn(shape, device=device, dtype=dtype) * 0.1
    k = torch.randn(shape, device=device, dtype=dtype) * 0.1
    v = torch.randn(shape, device=device, dtype=dtype) * 0.1
    g = torch.randn(shape, device=device, dtype=dtype) * 0.1
    beta_logits = torch.randn(
        1, total, num_heads, device=device, dtype=dtype
    )
    beta = beta_logits.float().sigmoid()
    A_log = torch.rand(num_heads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(
        num_heads, HEAD_DIM, dtype=torch.float32, device=device
    )

    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    cache_indices = torch.arange(len(seq_lens), dtype=torch.int32, device=device)
    pool = (
        torch.randn(
            num_slots,
            num_heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    return (
        q,
        k,
        v,
        g,
        beta_logits,
        beta,
        A_log,
        dt_bias,
        query_start_loc,
        cache_indices,
        pool,
    )


def run_once(inputs, pool_init, *, use_pla):
    (
        q,
        k,
        v,
        g,
        beta_logits,
        beta,
        A_log,
        dt_bias,
        query_start_loc,
        cache_indices,
        _,
    ) = inputs
    pool = pool_init.clone()
    # SGLang KDA writes its output into the storage supplied as ``v``. Both
    # paths therefore get a fresh copy, and the copy cost is timed for both.
    v_work = v.clone()

    if use_pla:
        # The benchmark uses the first N slots, matching cache_indices=arange.
        # A view lets PLA consume and write the same cloned runtime state pool.
        state_slots = pool.narrow(0, 0, cache_indices.numel())
        output = torch.empty_like(v_work[0])
        flashkda_fwd(
            q[0],
            k[0],
            v_work[0],
            g[0],
            beta_logits[0],
            A_log,
            dt_bias,
            output,
            HEAD_DIM**-0.5,
            LOWER_BOUND,
            state_slots,
            state_slots,
            query_start_loc,
        )
        return output.unsqueeze(0), pool

    output = _load_triton_chunk_kda()(
        q=q,
        k=k,
        v=v_work,
        g=g,
        beta=beta,
        scale=HEAD_DIM**-0.5,
        initial_state=pool,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
    )
    return output, pool


def rel_err(actual, expected):
    return err_ratio(actual, expected)


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
    parser.add_argument("--num-heads", "--H", dest="num_heads", type=int, default=12)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="512,1024,2048,4096,8192",
        help="Comma-separated single-sequence lengths to sweep.",
    )
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="Also run the production-style multi-sequence ragged batch.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    cases = [[int(value)] for value in args.seq_lens.split(",") if value]
    if args.ragged:
        cases.append([4096, 137, 2048, 63])

    print(f"H={args.num_heads} D={HEAD_DIM} dtype={args.dtype}")
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
            args.num_heads,
            dtype,
            DEV,
            num_slots=len(seq_lens) + 2,
        )
        pool_init = inputs[-1]

        out_ref, pool_ref = run_once(inputs, pool_init, use_pla=False)
        out_pla, pool_pla = run_once(inputs, pool_init, use_pla=True)
        output_error = rel_err(out_pla, out_ref)
        used = inputs[9].to(torch.long)
        state_error = rel_err(pool_pla[used], pool_ref[used])

        triton_ms = timed_ms(
            lambda: run_once(inputs, pool_init, use_pla=False),
            args.warmup,
            args.repeat,
        )
        pla_ms = timed_ms(
            lambda: run_once(inputs, pool_init, use_pla=True),
            args.warmup,
            args.repeat,
        )

        label = ",".join(str(length) for length in seq_lens)
        if len(label) > 25:
            label = label[:22] + "..."
        print(
            f"{label:>26} {output_error:>10.3e} {state_error:>10.3e} "
            f"{triton_ms:>10.3f} {pla_ms:>10.3f} "
            f"{triton_ms / pla_ms:>7.2f}x"
        )
        try:
            torch.testing.assert_close(
                out_pla, out_ref, atol=args.atol, rtol=args.rtol
            )
            torch.testing.assert_close(
                pool_pla[used], pool_ref[used], atol=args.atol, rtol=args.rtol
            )
        except AssertionError as error:
            failures.append((label, output_error, state_error, error))

    if failures:
        print("\nFAILED (torch.assert_close tolerance exceeded):")
        for label, output_error, state_error, _ in failures:
            print(
                f"  seq_lens={label}: o_err={output_error:.3e} "
                f"state_err={state_error:.3e}"
            )
        raise SystemExit(1)
    print("\nAll cases within tolerance.")


if __name__ == "__main__":
    main()
