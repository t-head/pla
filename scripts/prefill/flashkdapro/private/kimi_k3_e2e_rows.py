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

"""Kimi-K3 E2E sweep -> rows[] in the flashkda-e2e-vs-fla.html report schema.

Measures pla.prefill.flashkdapro.flashkda_fwd (CR #29320733 / commit 3f436ec,
default dispatch) vs FLA-Triton chunk_kda, per unique Kimi-K3 prefill shape,
for both no_state and with_state modes. Emits one JSONL row per (shape, mode)
matching the report's rows[] schema:
  case_id, suite, mode, H, B, S, D, T_total, token_heads, seq_lens, status,
  elapsed_s, ppu_median_ms, ppu_mean_ms, ppu_p10_ms, ppu_p90_ms, ppu_cv,
  fla_median_ms, fla_mean_ms, fla_p10_ms, fla_p90_ms, fla_cv, speedup,
  retested, sample_count

Run inside hp_ppu15_flashqla_kda:
  FLA_SOURCE_PATH=/mnt/ssd/hpwu/flash-linear-attention \
  PYTHONPATH=tests:.:$FLA_SOURCE_PATH CUDA_VISIBLE_DEVICES=2 \
  python3 tests/kimi_k3_e2e_rows.py --tp 8 --output /tmp/rows_tp8.jsonl --limit 150
"""
import argparse, json, math, os, re, statistics, time
import torch
from bench_vs_fla import D, LOWER_BOUND, run_fla_chunk_kda
from pla.prefill.flashkdapro import flashkda_fwd as _pla_fwd

DEV = "cuda"
CASE_RE = re.compile(r"num_heads_qk:(\d+),.*?batch_size:(\d+),dtype:bf16,cu_seqlens_q:\[([\d,]+)\]")


def run_ppu(q, k, v, g, beta, A_log, dt_bias, scale, h0, cu):
    # out/final_state are caller-allocated out-params; allocate inside the timed
    # region so this stays comparable to the FLA baseline, which allocates too.
    T, H, Dh = q.shape[1], q.shape[2], q.shape[3]
    n_seq = 1 if cu is None else cu.numel() - 1
    o = torch.empty(T, H, Dh, dtype=torch.bfloat16, device=q.device)
    # The production ABI accepts fp32 state directly and requires initial/final
    # state dtypes to match.  Converting h0 inside the timed lambda measured a
    # standalone PyTorch cast on every iteration, not FlashKDApro, and produced
    # high-CV false regressions for small H=12 shapes.
    state_dtype = h0.dtype if h0 is not None else torch.bfloat16
    fin = torch.empty(n_seq, H, Dh, Dh, dtype=state_dtype, device=q.device)
    _pla_fwd(q[0], k[0], v[0], g[0], beta[0], A_log, dt_bias, o, scale,
             LOWER_BOUND, h0, fin, cu)
    return o.unsqueeze(0), fin


def parse_list(path):
    seen, out = set(), []
    with open(path) as fh:
        for line in fh:
            m = CASE_RE.search(line)
            if not m:
                continue
            H = int(m.group(1))
            cu = [int(x) for x in m.group(3).split(",")]
            sl = tuple(cu[i + 1] - cu[i] for i in range(len(cu) - 1))
            key = (H, len(sl), sum(sl), sl)
            if key in seen:
                continue
            seen.add(key)
            out.append((H, len(sl), sum(sl), sl))
    return out


def suite_of(N, seq_lens):
    if N == 1:
        return "fixed"
    return "equal" if len(set(seq_lens)) == 1 else "ragged"


def stats(samples):
    xs = sorted(samples)
    n = len(xs)
    med = statistics.median(xs)
    mean = sum(xs) / n
    p10 = xs[max(0, int(0.1 * n) - 0)] if n > 1 else xs[0]
    p90 = xs[min(n - 1, int(0.9 * n))] if n > 1 else xs[0]
    cv = (statistics.pstdev(xs) / mean) if mean > 0 and n > 1 else 0.0
    return med, mean, p10, p90, cv


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record()
        torch.cuda.synchronize()
        samples.append(st.elapsed_time(en))
    return samples


def build(H, T_total, seq_lens, input_mode="deterministic", static=None,
          normalize_qk=False):
    N = len(seq_lens)
    is_varlen = N > 1
    scale = 1.0 / math.sqrt(D)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
                      dtype=torch.long, device=DEV) if is_varlen else None
    shape = (1, T_total, H, D)
    if static is not None:
        q, k, v, g, beta, A_log, dt_bias = static
    elif input_mode == "empty":
        # Legacy reproduction only.  Do not use for acceptance: CP's exact
        # zero-state exit is data dependent, and allocator garbage/NaNs make
        # both timing and numerics undefined.
        q = torch.empty(shape, dtype=torch.bfloat16, device=DEV)
        k = torch.empty(shape, dtype=torch.bfloat16, device=DEV)
        v = torch.empty(shape, dtype=torch.bfloat16, device=DEV)
        g = torch.empty(shape, dtype=torch.bfloat16, device=DEV)
        beta = torch.empty((1, T_total, H), dtype=torch.bfloat16, device=DEV)
        A_log = torch.empty(H, dtype=torch.float32, device=DEV)
        dt_bias = torch.empty(H, D, dtype=torch.float32, device=DEV)
    else:
        # Valid, deterministic inputs initialized once outside the timed region.
        # q/k are nonzero so L2 normalization is defined; g/A/dt produce a
        # representative decaying gate and exercise CP's algebraic early exit.
        qk_value = 1.0 / math.sqrt(D) if normalize_qk else 1.0
        q = torch.full(
            shape, qk_value, dtype=torch.bfloat16, device=DEV
        )
        k = torch.full(
            shape, qk_value, dtype=torch.bfloat16, device=DEV
        )
        v = torch.full(shape, 0.25, dtype=torch.bfloat16, device=DEV)
        g = torch.zeros(shape, dtype=torch.bfloat16, device=DEV)
        beta = torch.zeros((1, T_total, H), dtype=torch.bfloat16, device=DEV)
        A_log = torch.zeros(H, dtype=torch.float32, device=DEV)
        dt_bias = torch.zeros(H, D, dtype=torch.float32, device=DEV)
    return q, k, v, g, beta, A_log, dt_bias, scale, cu, is_varlen, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument(
        "--batch", type=int, action="append",
        help="optional exact batch-size filter; repeat for multiple values",
    )
    ap.add_argument("--batch-min", type=int)
    ap.add_argument("--batch-max", type=int)
    ap.add_argument(
        "--tokens-min", type=int,
        help="optional minimum total-token filter applied before sampling",
    )
    ap.add_argument(
        "--tokens-max", type=int,
        help="optional maximum total-token filter applied before sampling",
    )
    ap.add_argument(
        "--input-mode", choices=("deterministic", "empty"),
        default="deterministic",
    )
    ap.add_argument(
        "--fla-pre-normalized", action="store_true",
        help=(
            "conservative coverage mode: pre-normalize q/k and exclude FLA's "
            "l2norm while PPU still executes its normalization"
        ),
    )
    ap.add_argument(
        "--ppu-only", action="store_true",
        help="measure only FlashKDApro (useful for same-build dispatch A/B)",
    )
    args = ap.parse_args()
    H_of = {1: 96, 2: 48, 4: 24, 8: 12}
    lp = f"/mnt/ssd/hpwu/kimi_k3_lists_gino/Kimi-K3_prefill_tp{args.tp}_kvlimit_200GB_model_len_128K_kda_op.list"
    shapes = parse_list(lp)
    if args.batch:
        allowed_batches = set(args.batch)
        shapes = [shape for shape in shapes if shape[1] in allowed_batches]
    if args.batch_min is not None:
        shapes = [shape for shape in shapes if shape[1] >= args.batch_min]
    if args.batch_max is not None:
        shapes = [shape for shape in shapes if shape[1] <= args.batch_max]
    if args.tokens_min is not None:
        shapes = [shape for shape in shapes if shape[2] >= args.tokens_min]
    if args.tokens_max is not None:
        shapes = [shape for shape in shapes if shape[2] <= args.tokens_max]
    # representative sampling: sort by (N, T_total), take an even stride up to limit
    shapes.sort(key=lambda s: (s[1], s[2]))
    if len(shapes) > args.limit:
        step = len(shapes) / args.limit
        shapes = [shapes[int(i * step)] for i in range(args.limit)]
    print(f"[tp{args.tp}] {len(shapes)} shapes H={H_of[args.tp]}", flush=True)

    # warmup context
    try:
        b = build(
            H_of[args.tp], 2048, (512, 512, 512, 512), args.input_mode,
            normalize_qk=args.fla_pre_normalized,
        )
        for _ in range(3):
            run_ppu(*b[:8], None, b[8])
            if not args.ppu_only:
                run_fla_chunk_kda(
                    *b[:8], None, extra={"cu_seqlens": b[8]},
                    pre_normalized=args.fla_pre_normalized,
                )
        torch.cuda.synchronize()
    except Exception as e:
        print("warmup skip", e, flush=True)
    finally:
        if "b" in locals():
            del b

    t0 = time.time()
    # All deterministic shapes for a tp use the same scalar-filled inputs.
    # Initialize the largest T once, outside the timed region, and take a
    # contiguous prefix for each shape.  Reinitializing several GiB for every
    # distinct T made a full coverage scan hours slower without changing the
    # kernel input or measurement.  A [1,:T,H,D] prefix remains contiguous.
    max_t = max(shape[2] for shape in shapes)
    full = build(
        H_of[args.tp], max_t, (max_t,), args.input_mode,
        normalize_qk=args.fla_pre_normalized,
    )
    full_static = full[:7]
    del full
    with open(args.output, "w") as fh:
        for idx, (H, N, T_total, seq_lens) in enumerate(shapes):
            suite = suite_of(N, seq_lens)
            S = seq_lens[0] if N == 1 else (T_total // N)
            static = (
                full_static[0][:, :T_total],
                full_static[1][:, :T_total],
                full_static[2][:, :T_total],
                full_static[3][:, :T_total],
                full_static[4][:, :T_total],
                full_static[5],
                full_static[6],
            )
            (q, k, v, g, beta, A_log, dt_bias, scale, cu,
             is_varlen, _N) = build(
                 H, T_total, seq_lens, args.input_mode, static=static
             )
            for mode in ("no_state", "with_state"):
                if mode == "with_state":
                    h0 = torch.zeros((N, H, D, D), dtype=torch.float32, device=DEV)
                else:
                    h0 = None
                ck = time.time()
                try:
                    ppu_s = time_fn(lambda: run_ppu(q, k, v, g, beta, A_log, dt_bias, scale, h0, cu),
                                    args.warmup, args.iters)
                    pm, pmean, pp10, pp90, pcv = stats(ppu_s)
                    if args.ppu_only:
                        fm = fmean = fp10 = fp90 = fcv = None
                    else:
                        h0f = h0 if h0 is not None else None
                        fla_s = time_fn(lambda: run_fla_chunk_kda(q, k, v, g, beta, A_log, dt_bias, scale,
                                                                  h0f, extra={"cu_seqlens": cu} if is_varlen else {},
                                                                  pre_normalized=args.fla_pre_normalized),
                                        args.warmup, args.iters)
                        fm, fmean, fp10, fp90, fcv = stats(fla_s)
                    row = {
                        "case_id": f"tp{args.tp}-{suite}-H{H}-N{N}-T{T_total}",
                        "suite": suite, "mode": mode, "H": H, "B": N, "S": S, "D": D,
                        "T_total": T_total, "token_heads": T_total * H,
                        "seq_lens": list(seq_lens) if N <= 8 else [seq_lens[0], "...", seq_lens[-1]],
                        "status": "ok", "elapsed_s": round(time.time() - ck, 3),
                        "ppu_median_ms": pm, "ppu_mean_ms": pmean, "ppu_p10_ms": pp10, "ppu_p90_ms": pp90, "ppu_cv": pcv,
                        "fla_median_ms": fm, "fla_mean_ms": fmean, "fla_p10_ms": fp10, "fla_p90_ms": fp90, "fla_cv": fcv,
                        "speedup": (fm / pm if pm > 0 else 0.0)
                        if fm is not None else None,
                        "retested": False, "sample_count": len(ppu_s),
                        "tp": args.tp, "input_mode": args.input_mode,
                        "fla_l2norm_included": not args.fla_pre_normalized,
                    }
                except Exception as e:
                    row = {"case_id": f"tp{args.tp}-{suite}-H{H}-N{N}-T{T_total}", "suite": suite,
                           "mode": mode, "H": H, "B": N, "S": S, "D": D, "T_total": T_total,
                           "token_heads": T_total * H, "seq_lens": list(seq_lens)[:4],
                           "status": "error:" + str(e)[:60], "speedup": 0.0,
                           "tp": args.tp, "input_mode": args.input_mode,
                           "fla_l2norm_included": not args.fla_pre_normalized}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                del h0
            del q, k, v, g, beta, A_log, dt_bias
            if (idx + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  [{idx+1}/{len(shapes)}] {el/60:.1f}min rate={ (idx+1)/el:.2f}/s", flush=True)
    print(f"[tp{args.tp}] DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
