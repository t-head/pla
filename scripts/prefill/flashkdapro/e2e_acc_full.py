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

"""FlashKDA prefill end-to-end accuracy: every dispatch path, vs a fp32 torch reference.

This is the accuracy gate for `pla.prefill.flashkdapro`. It is self-contained —
the oracle is the fp32 FlashKDA torch reference in `private/test_fwd_e2e.py`
(`ref_fwd`, dtype/rounding semantics matching upstream `torch_ref.py`), so it
needs nothing beyond torch and the built extension.

Coverage is organised around what actually changes the kernel that runs, since
`flashkda_fwd` picks its K2 variant from the shape alone:

  dvsplit BV16     single sequence, H <= 24
  dvsplit BV32     single sequence, 24 < H <= 64
  full-DV nows     single sequence, H > 64
  varlen nows      packed multi-sequence (equal and ragged lengths)
  baseline DVT32   very small grid (N*H < 40)

plus the orthogonal axes: a partial (non-multiple-of-16) chunk tail, and the
state out-param contract -- zero vs bf16 vs fp32 initial state, a bf16 or fp32
caller-allocated ``final_state``, and ``final_state=None`` (K2 must skip the
state store and still produce an identical output). Varlen cases are checked
against the reference run per sequence and concatenated, which is exactly the
semantics the packed kernel must reproduce.

Run (from the repo root, as a plain script so `import pla` still resolves to the
installed package):

    CUDA_VISIBLE_DEVICES=4 python3 scripts/prefill/flashkdapro/e2e_acc_full.py
"""
import argparse
import math
import os
import sys
import time

# The fp32 reference lives next to this script, under private/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "private"))

import torch

from test_fwd_e2e import D, TOL, err_ratio, ref_fwd, ref_fwd_batched, reference_threads

LOWER = -5.0
SCALE = 1.0 / math.sqrt(D)


def make_inputs(T, H, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    mk = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device=device, generator=g)
    return (
        mk(T, H, D),                                                    # q
        mk(T, H, D),                                                    # k
        mk(T, H, D),                                                    # v
        mk(T, H, D),                                                    # g
        mk(T, H),                                                       # beta
        torch.rand(H, dtype=torch.float32, device=device, generator=g),  # A_log
        torch.rand(H, D, dtype=torch.float32, device=device, generator=g),  # dt_bias
    )


def _dt(dtype):
    if dtype is None:
        return "none"
    return "f32" if dtype == torch.float32 else "bf16"


def check(tag, path, H, seq_lens, h0_dtype, final_dtypes, seed=20260806,
          pad_tokens=0):
    """Run one case against the fp32 reference; returns a list of pass flags.

    ``h0_dtype`` is ``None`` for a zero initial state, otherwise the
    ``initial_state`` dtype. ``final_dtypes`` lists the caller-allocated
    ``final_state`` dtypes to exercise -- once an initial state is present
    upstream requires the two to match, so that is a single entry. The
    ``final_state=None`` path is always checked as well.
    """
    from pla.prefill.flashkdapro import flashkda_fwd

    started = time.perf_counter()

    def progress(stage):
        # stderr remains visible while the xtest entry captures stdout.
        print(f"dispatch_progress case={tag} initial={_dt(h0_dtype)} "
              f"stage={stage} elapsed_seconds={time.perf_counter() - started:.3f}",
              file=sys.stderr, flush=True)

    progress("kernel_start")
    real_tokens = sum(seq_lens)
    T = real_tokens + pad_tokens
    q, k, v, g, beta, A_log, dt_bias = make_inputs(T, H, seed)
    n_seq = len(seq_lens)
    varlen = n_seq > 1 or pad_tokens > 0

    if h0_dtype is None:
        h0 = None
    else:
        gen = torch.Generator(device="cuda").manual_seed(seed + 1)
        h0 = torch.randn(n_seq, H, D, D, dtype=h0_dtype, device="cuda",
                         generator=gen) * 0.1

    cu = None
    if varlen:
        cu = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(),
                          dtype=torch.int64, device="cuda")

    def run(final_dtype):
        """Allocate the out-params, run, and hand them back."""
        # A nonzero sentinel makes an omitted padded-tail store observable.
        o = torch.full(
            (T, H, D), 7.0, dtype=torch.bfloat16, device="cuda")
        fs = (None if final_dtype is None else
              torch.empty(n_seq, H, D, D, dtype=final_dtype, device="cuda"))
        flashkda_fwd(q, k, v, g, beta, A_log, dt_bias, o,
                     SCALE, LOWER, h0, fs, cu)
        return o, fs

    setup_seconds = time.perf_counter() - started
    kernel_started = time.perf_counter()
    runs = [(dtype,) + run(dtype) for dtype in final_dtypes]
    out_nostate, _ = run(None)
    torch.cuda.synchronize()
    kernel_seconds = time.perf_counter() - kernel_started
    progress("kernel_done")

    # Copy each input/parameter once, then batch independent heads on the CPU.
    progress("copy_start")
    copy_started = time.perf_counter()
    cpu_inputs = [x.cpu() for x in (q, k, v, g, beta, A_log, dt_bias)]
    cpu_h0 = h0.cpu() if h0 is not None else None
    cpu_runs = [(dtype, out.cpu(), final.cpu()) for dtype, out, final in runs]
    cpu_nostate = out_nostate.cpu()
    copy_seconds = time.perf_counter() - copy_started
    progress("copy_done")
    progress("reference_start")
    reference_started = time.perf_counter()
    with reference_threads():
        cpu_threads = torch.get_num_threads()
        zero_h0 = torch.zeros(H, D, D, dtype=torch.bfloat16)
        ref_outs, ref_states, off = [], [], 0
        for i, L in enumerate(seq_lens):
            sl = slice(off, off + L)
            ro, rs = ref_fwd_batched(*(x[sl] for x in cpu_inputs[:5]),
                                     *cpu_inputs[5:], SCALE, LOWER,
                                     cpu_h0[i] if cpu_h0 is not None else zero_h0, L, H)
            ref_outs.append(ro)
            ref_states.append(rs)
            off += L
        ref_out = torch.cat(ref_outs, 0).reshape(-1)
        ref_state = torch.stack(ref_states, 0).reshape(-1)
        reference_seconds = time.perf_counter() - reference_started
        progress("reference_done")
        judgement_started = time.perf_counter()

        flags = []
        for final_dtype, out, final in cpu_runs:
            e_out = err_ratio(out[:real_tokens].reshape(-1), ref_out)
            e_st = err_ratio(final.reshape(-1), ref_state)
            dtype_ok = final.dtype == final_dtype
            tail_zero = (
                pad_tokens == 0 or
                torch.count_nonzero(out[real_tokens:]).item() == 0)
            ok = e_out < TOL and e_st < TOL and dtype_ok and tail_zero
            flags.append(ok)
            note = "" if dtype_ok else f" DTYPE={final.dtype}!={final_dtype}"
            print(f"  [{path:14s}] {tag:26s} {_dt(h0_dtype):4s}->{_dt(final_dtype):4s} "
                  f"out={e_out:.3e} state={e_st:.3e} "
                  f"pad={pad_tokens} tail_zero={tail_zero} "
                  f"{'OK' if ok else 'FAIL'}{note}")

        # final_state=None must skip the state store without perturbing the output.
        nostate_tail_zero = (
            pad_tokens == 0 or
            torch.count_nonzero(cpu_nostate[real_tokens:]).item() == 0)
        ok = torch.equal(cpu_nostate, cpu_runs[0][1]) and nostate_tail_zero
        flags.append(ok)
        print(f"  [{path:14s}] {tag:26s} {_dt(h0_dtype):4s}->none "
              f"out bit-identical={ok} {'OK' if ok else 'FAIL'}")
        judgement_seconds = time.perf_counter() - judgement_started
        progress("judgement_done")
    total_seconds = time.perf_counter() - started
    print(f"timing setup_seconds={setup_seconds:.3f} kernel_seconds={kernel_seconds:.3f} "
          f"copy_seconds={copy_seconds:.3f} reference_seconds={reference_seconds:.3f} "
          f"judgement_seconds={judgement_seconds:.3f} total_seconds={total_seconds:.3f} "
          f"cpu_threads={cpu_threads}")
    return flags


# (tag, dispatch path, H, seq_lens)
#
# Which K2 variant runs is decided by N, H and the dvsplit T>=256 threshold, not
# by long sequences, so these use the shortest lengths that still select each
# kernel (the fp32 reference is a per-chunk Python loop — keeping T small is what
# keeps this gate quick):
#   dvsplit   N==1 && !varlen && T_total>=256 && H<=64   (BV16 if H<=24 else BV32)
#   full-DV   N==1 && H>64, and any launch dvsplit declines
#   baseline  N*H < 40  (shared-memory column split)
#   varlen    N>1 && N*H >= 40
CASES = [
    # single sequence -> register-resident value-split
    ("b1-h6-s512",           "dvsplit BV16",   6,  [512]),
    ("b1-h24-s256",          "dvsplit BV16",   24, [256]),
    ("b1-h48-s256",          "dvsplit BV32",   48, [256]),
    ("b1-h96-s256",          "full-DV nows",   96, [256]),
    # partial chunk tail (T % 16 != 0) on the split and full-DV paths
    ("b1-h12-s264-tail",     "dvsplit BV16",   12, [264]),
    ("b1-h96-s100-tail",     "full-DV nows",   96, [100]),
    # packed multi-sequence
    ("b8-h12-s128-equal",    "varlen nows",    12, [128] * 8),
    ("b4-h12-ragged-tail",   "varlen nows",    12, [100, 273, 16, 272]),
    # short high-head varlen BV32 promotion.  H96/N3 switches only when an
    # initial state is present; H48/N2 switches in both state modes, so these
    # two cases exercise both the promoted path and its full-DV fallback.
    ("b3-h96-ragged-short",   "short high-H",    96, [48, 64, 80]),
    ("b2-h48-ragged-short",   "short high-H",    48, [97, 121]),
    # N*H < 40 -> shared-memory baseline (column split)
    ("b6-h6-ragged-baseline", "baseline DVT32", 6, [100, 150, 80, 64, 48, 32]),
    ("b2-h4-s256-baseline",   "baseline DVT32", 4, [256, 256]),
]


PADDED_CASES = (
    ("b1-h12-s2205-pad1", "padded varlen", 12, [2205], 1),
    ("b4-h12-ragged-pad7", "padded varlen", 12, [100, 273, 16, 272], 7),
)


def benchmark_reference(baseline_threads=None):
    """CPU-only comparison of both oracles on the unchanged 40-case dispatch matrix."""
    baseline_threads = torch.get_num_threads() if baseline_threads is None else baseline_threads
    scalar_seconds, batched_seconds, count = 0.0, 0.0, 0
    matrix = [(tag, H, lengths, 0, (None, torch.bfloat16, torch.float32))
              for tag, _, H, lengths in CASES]
    matrix += [(tag, H, lengths, pad, (None, torch.bfloat16))
               for tag, _, H, lengths, pad in PADDED_CASES]
    with reference_threads(), torch.no_grad():
        print(f"CPU reference benchmark: torch={torch.__version__} "
              f"baseline_threads={baseline_threads} batched_threads={torch.get_num_threads()}")
        for tag, H, lengths, pad, dtypes in matrix:
            inputs = make_inputs(sum(lengths) + pad, H, 20260806, device="cpu")
            for dtype in dtypes:
                gen = torch.Generator().manual_seed(20260807)
                h0 = (torch.zeros(len(lengths), H, D, D, dtype=torch.bfloat16) if dtype is None
                      else torch.randn(len(lengths), H, D, D, dtype=dtype, generator=gen) * 0.1)
                off, slow, fast = 0, 0.0, 0.0
                for i, length in enumerate(lengths):
                    sl = slice(off, off + length)
                    args = (*(x[sl] for x in inputs[:5]), *inputs[5:],
                            SCALE, LOWER, h0[i], length, H)
                    before = h0[i].clone()
                    with reference_threads(baseline_threads):
                        start = time.perf_counter()
                        expected = ref_fwd(*args)
                        slow += time.perf_counter() - start
                    start = time.perf_counter()
                    actual = ref_fwd_batched(*args)
                    fast += time.perf_counter() - start
                    for name, got, want in zip(("output", "state"), actual, expected):
                        torch.testing.assert_close(got, want, rtol=0, atol=0,
                                                   msg=f"{tag}/{_dt(dtype)}/{i}/{name}")
                    assert torch.equal(h0[i], before), "reference mutated initial state"
                    off += length
                count += 1
                scalar_seconds += slow
                batched_seconds += fast
                print(f"{tag}_{_dt(dtype)}: exact_match=True serial_seconds={slow:.4f} "
                      f"batched_seconds={fast:.4f}", flush=True)
    print(f"reference_cases={count} serial_threads={baseline_threads} serial_seconds={scalar_seconds:.3f} "
          f"batched_seconds={batched_seconds:.3f} speedup={scalar_seconds / batched_seconds:.2f}x")
    return scalar_seconds, batched_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-reference", action="store_true",
                        help="Compare CPU oracles on all 40 cases without loading PLA or using a PPU")
    parser.add_argument("--baseline-threads", type=int,
                        help="CPU benchmark baseline threads (default: original torch thread count)")
    args = parser.parse_args()
    if args.baseline_threads is not None and args.baseline_threads < 1:
        parser.error("--baseline-threads must be positive")
    if args.benchmark_reference:
        benchmark_reference(args.baseline_threads)
        return
    print(f"FlashKDA prefill E2E accuracy vs fp32 torch reference (tol<{TOL:g}):")
    print("  columns: [K2 path] case  initial_state->final_state  errors")
    results = []
    for tag, path, H, seq_lens in CASES:
        # Zero initial state: final_state may be either dtype, since upstream
        # allows an fp32 final state to be filled from a zero initial state.
        results += check(tag, path, H, seq_lens, None,
                         (torch.bfloat16, torch.float32))
        # With an initial state the two dtypes must match, so one each.
        results += check(tag, path, H, seq_lens, torch.bfloat16, (torch.bfloat16,))
        results += check(tag, path, H, seq_lens, torch.float32, (torch.float32,))
    # Same physical alignment and caller-owned buffers as the dispatch entry.
    for tag, path, H, seq_lens, pad_tokens in PADDED_CASES:
        results += check(tag, path, H, seq_lens, None,
                         (torch.bfloat16, torch.float32),
                         pad_tokens=pad_tokens)
        results += check(tag, path, H, seq_lens, torch.bfloat16,
                         (torch.bfloat16,), pad_tokens=pad_tokens)
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} checks OK")
    assert n_ok == len(results), f"{len(results) - n_ok} check(s) failed"
    print("E2E ACCURACY OK")


if __name__ == "__main__":
    main()
