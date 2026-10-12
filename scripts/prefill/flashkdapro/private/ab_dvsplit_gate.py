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

"""A/B the dvsplit gate widening for varlen H>12 on PPU1.5.

Current gate (kernel2_recurrence_nows.cu):
    auto_varlen_dvsplit = is_varlen && (H == 6 || H == 12)
Proposal: widen to cover H=24 (and find the real boundary empirically).

KEY: no rebuild needed. The env override path is
    use_dvsplit = has_override ? (valid_override && (is_varlen || N==1)) : auto_dvsplit
so PLA_FLASHKDA_K2_DVSPLIT=16/32 FORCES dvsplit on any varlen shape, and =0
forces it off. We measure 4 variants per (shape, state-mode):

    default : env unset  -> whatever the current gate picks
    bv16    : DVSPLIT=16 -> forced dvsplit, BV=16 (grid *8)
    bv32    : DVSPLIT=32 -> forced dvsplit, BV=32 (grid *4)
    fla     : FLA-Triton chunk_kda reference

Two suites:
  --suite replay : replay the exact shapes that lost to FLA in the Kimi-K3 sweep
                   (reads losing_shapes.json: [{tp,H,seq_lens}, ...])
  --suite grid   : systematic H x N x T grid with equal splits, to locate the
                   real gate boundary instead of assuming H<=24 && N<=4

Run inside hp_ppu15_flashqla_kda:
  FLA_SOURCE_PATH=/mnt/ssd/hpwu/flash-linear-attention \
  PYTHONPATH=tests:.:$FLA_SOURCE_PATH CUDA_VISIBLE_DEVICES=2 \
  python3 tests/ab_dvsplit_gate.py --suite grid --H 24 --output /tmp/ab_H24.jsonl
"""
import argparse, json, math, os, statistics, time
import torch
from bench_vs_fla import D, LOWER_BOUND, run_fla_chunk_kda
from pla.prefill.flashkdapro import flashkda_fwd as _pla_fwd

DEV = "cuda"
VARIANTS = [("default", None), ("bv16", "16"), ("bv32", "32")]


def set_dvsplit(val):
    k = "PLA_FLASHKDA_K2_DVSPLIT"
    os.environ.pop(k, None)
    if val is not None:
        os.environ[k] = val


def run_ppu(q, k, v, g, beta, A_log, dt_bias, scale, h0, cu):
    # out/final_state are caller-allocated out-params; allocate inside the timed
    # region so this stays comparable to the FLA baseline, which allocates too.
    T, H, Dh = q.shape[1], q.shape[2], q.shape[3]
    n_seq = 1 if cu is None else cu.numel() - 1
    o = torch.empty(T, H, Dh, dtype=torch.bfloat16, device=q.device)
    fin = torch.empty(n_seq, H, Dh, Dh, dtype=torch.bfloat16, device=q.device)
    _pla_fwd(q[0], k[0], v[0], g[0], beta[0], A_log, dt_bias, o, scale,
             LOWER_BOUND, h0.to(torch.bfloat16) if h0 is not None else None,
             fin, cu)
    return o.unsqueeze(0), fin


def build(H, seq_lens):
    T_total = sum(seq_lens)
    N = len(seq_lens)
    scale = 1.0 / math.sqrt(D)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
                      dtype=torch.long, device=DEV)
    t = lambda *sh: torch.empty(sh, dtype=torch.bfloat16, device=DEV)
    return (t(1, T_total, H, D), t(1, T_total, H, D), t(1, T_total, H, D), t(1, T_total, H, D),
            t(1, T_total, H), torch.empty(H, dtype=torch.float32, device=DEV),
            torch.empty(H, D, dtype=torch.float32, device=DEV), scale, cu, N, T_total)


def med_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(iters):
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        xs.append(a.elapsed_time(b))
    return statistics.median(xs)


def shapes_replay(path, only_H):
    out = []
    for s in json.load(open(path)):
        if only_H and s["H"] != only_H:
            continue
        out.append((s["H"], tuple(s["seq_lens"]), "replay"))
    return out


def shapes_grid(only_H):
    Hs = [only_H] if only_H else [12, 24, 48, 96]
    out = []
    for H in Hs:
        for N in (2, 3, 4, 6, 8, 16):
            for T in (8192, 32768):
                if T // N < 64:
                    continue
                base = T // N
                sl = [base] * (N - 1) + [T - base * (N - 1)]
                out.append((H, tuple(sl), "grid-equal"))
            # one skewed variant per N at T=32768: 90% in one segment
            T = 32768
            if N >= 2:
                big = int(T * 0.9)
                rest = T - big
                per = max(64, rest // (N - 1))
                sl = [big] + [per] * (N - 2) + [T - big - per * (N - 2)] if N > 2 else [big, rest]
                if all(x > 0 for x in sl) and sum(sl) == T:
                    out.append((H, tuple(sl), "grid-skew"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["replay", "grid"], required=True)
    ap.add_argument("--losing", default="/tmp/kimi_rows/losing_shapes.json")
    ap.add_argument("--H", type=int, default=0, help="restrict to one H (for per-device sharding)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    shapes = shapes_replay(args.losing, args.H) if args.suite == "replay" else shapes_grid(args.H)
    # dedupe
    seen, uniq = set(), []
    for H, sl, tag in shapes:
        if (H, sl) in seen:
            continue
        seen.add((H, sl)); uniq.append((H, sl, tag))
    shapes = uniq
    print(f"[{args.suite} H={args.H or 'all'}] {len(shapes)} shapes", flush=True)

    # warm CUDA/JIT
    try:
        b = build(shapes[0][0], (512, 512, 512, 512))
        set_dvsplit(None)
        for _ in range(3):
            run_ppu(*b[:8], None, b[8])
            run_fla_chunk_kda(*b[:8], None, extra={"cu_seqlens": b[8]})
        torch.cuda.synchronize()
    except Exception as e:
        print("warmup skip:", e, flush=True)

    t0 = time.time()
    with open(args.output, "w") as fh:
        for i, (H, sl, tag) in enumerate(shapes):
            for mode in ("no_state", "with_state"):
                bl = build(H, sl)
                (q, k, v, g, beta, A_log, dt_bias, scale, cu, N, T_total) = bl
                h0 = torch.zeros((N, H, D, D), dtype=torch.float32, device=DEV) if mode == "with_state" else None
                rec = {"suite": tag, "H": H, "N": N, "T_total": T_total,
                       "seq_lens": list(sl), "mode": mode,
                       "avg_seg": T_total // N, "max_seg": max(sl),
                       "skew": round(max(sl) / (T_total / N), 3)}
                try:
                    for name, envval in VARIANTS:
                        set_dvsplit(envval)
                        rec[name + "_ms"] = med_ms(
                            lambda: run_ppu(q, k, v, g, beta, A_log, dt_bias, scale, h0, cu),
                            args.warmup, args.iters)
                    set_dvsplit(None)
                    rec["fla_ms"] = med_ms(
                        lambda: run_fla_chunk_kda(q, k, v, g, beta, A_log, dt_bias, scale, h0,
                                                  extra={"cu_seqlens": cu}),
                        args.warmup, args.iters)
                    best_name, best_ms = min(((n, rec[n + "_ms"]) for n, _ in VARIANTS), key=lambda x: x[1])
                    rec["ppu_best"] = best_name
                    rec["ppu_best_ms"] = best_ms
                    rec["sp_default"] = rec["fla_ms"] / rec["default_ms"]
                    rec["sp_bv16"] = rec["fla_ms"] / rec["bv16_ms"]
                    rec["sp_bv32"] = rec["fla_ms"] / rec["bv32_ms"]
                    rec["sp_best"] = rec["fla_ms"] / best_ms
                    rec["gain_over_default"] = rec["default_ms"] / best_ms
                    rec["status"] = "ok"
                except Exception as e:
                    rec["status"] = "error:" + str(e)[:80]
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                del q, k, v, g, beta, A_log, dt_bias, h0, bl
                torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(shapes)}] {el/60:.1f}min", flush=True)
    print(f"[{args.suite} H={args.H or 'all'}] DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
