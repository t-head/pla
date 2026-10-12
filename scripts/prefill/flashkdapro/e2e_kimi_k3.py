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

"""FlashKDApro E2E accuracy + performance on representative Kimi-K3 prefill cases.

Pulls typical shapes straight out of the Kimi-K3 prefill `.list` files for every
tensor-parallel degree (tp1/tp2/tp4/tp8 -> per-device head counts H=96/48/24/12)
and exercises `pla.prefill.flashkdapro.flashkda_fwd` on them.

flashkdapro dispatches the *better-performing* case range to the new fast path.
The register-resident (nows) recurrence is selected by (kernel2_recurrence.cu):

    pro fast path  =  (N==1 && T_total>=256)      # single seq
                   || (N*H >= 40)                 # high concurrency
                   || (varlen && H <= 24)          # low-head varlen (HW A/B)

    Within it, single-seq long H in {6,12} additionally gets the register DK-split
    dedicated-leader K2; the newly-generalized varlen branch gets the BV32 dvsplit
    occupancy boost.  Everything else falls through to the flashkda-equivalent
    recurrence.  The affine-associativity K1 rewrite is applied unconditionally.

The strongest wins are therefore the low-head / low-batch / long-sequence corner
— i.e. tp8 (H=12) single- or few-sequence, long-T cases (and tp4/H=24 partially).
This script tags each case with whether it hits the pro fast path so the
accuracy + perf tables make the dispatch visible.

Two phases:
  * accuracy  — small cases only (per-seq T <= --acc-max-t, default 1024), vs the
                self-contained fp32 torch reference (private/test_fwd_e2e.py).
                The reference is a per-chunk Python loop, so long cases are skipped
                here on purpose; the short Kimi-K3 cases still cover every H.
  * perf      — representative real shapes (incl. long / ragged), timed with CUDA
                events (median of repeats). `--vs-fla` also times FLA-Triton
                (fla.ops.kda.chunk_kda, unwrapped) and prints the speedup.

Run inside the PPU1.5 container, from the repo root, as a plain script (so
`import pla` resolves to the *installed* package, not the source tree):

    CUDA_VISIBLE_DEVICES=4 python3 scripts/prefill/flashkdapro/e2e_kimi_k3.py
    CUDA_VISIBLE_DEVICES=4 python3 scripts/prefill/flashkdapro/e2e_kimi_k3.py --vs-fla
    CUDA_VISIBLE_DEVICES=4 python3 scripts/prefill/flashkdapro/e2e_kimi_k3.py \
        --tp 8,4 --per-tp 4 --acc-max-t 512
"""
import argparse
import glob
import math
import os
import re
import sys

# fp32 reference + err_ratio live in private/, next to this script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "private"))

fla_source_path = os.environ.get("FLA_SOURCE_PATH")
if fla_source_path:
    sys.path.insert(0, fla_source_path)

import torch

from pla.prefill.flashkdapro import flashkda_fwd
from test_fwd_e2e import D, TOL, err_ratio, ref_fwd

# Default location of the Kimi-K3 prefill case lists.
# Changed to a relative path or None; the docs state it must be given via --list-dir
DEFAULT_LIST_DIR = None  # force the user to give the path via --list-dir
# or use a path relative to the project
# DEFAULT_LIST_DIR = "caselist/prefill"
LIST_GLOB = "Kimi-K3_prefill_tp{tp}_*kda_op.list"

_CASE_RE = re.compile(
    r"num_heads_qk:(?P<H>\d+).*?"
    r"gate_lower_bound:(?P<lb>-?\d+(?:\.\d+)?).*?"
    r"batch_size:(?P<B>\d+).*?"
    r"cu_seqlens_q:\[(?P<cu>[0-9,\s]+)\]"
)


def parse_list(path):
    """Yield dicts {H, lower, N, cu (list[int]), T_total, seq_lens} for each case."""
    cases = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("[KDA]"):
                continue
            m = _CASE_RE.search(line)
            if not m:
                continue
            cu = [int(x) for x in m.group("cu").split(",") if x.strip() != ""]
            seq_lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
            cases.append(dict(
                H=int(m.group("H")), lower=float(m.group("lb")),
                N=len(seq_lens), cu=cu, T_total=cu[-1] if cu else 0,
                seq_lens=seq_lens,
            ))
    return cases


def pro_fast(c):
    """True if the case hits the flashkdapro register-resident (nows) fast path.

    Mirrors the outer use_nows gate in kernel2_recurrence.cu:
      (N==1 && T_total>=256)          # single seq
      || (N*H >= 40)                  # high concurrency
      || (is_varlen && H <= 24)       # low-head varlen (HW A/B: always >=baseline)
    """
    N, H, T = c["N"], c["H"], c["T_total"]
    is_varlen = N > 1
    return ((N == 1 and T >= 256)
            or (N * H >= 40)
            or (is_varlen and H <= 24))


def pick_perf_cases(cases, k):
    """Curated representative shapes: a long single-seq, a mid ragged, a large-B,
    plus the largest overall — deduped, at most k."""
    if not cases:
        return []
    single = [c for c in cases if c["N"] == 1]
    ragged = [c for c in cases if c["N"] > 1]
    picks = []
    if single:
        picks.append(max(single, key=lambda c: c["T_total"]))          # long single seq
        picks.append(min(single, key=lambda c: c["T_total"]))          # short single seq
    if ragged:
        mid = sorted(ragged, key=lambda c: (c["N"], c["T_total"]))
        picks.append(mid[len(mid) // 2])                                # mid ragged
        picks.append(max(ragged, key=lambda c: c["N"]))                # largest batch
        # newly-generalized varlen fast-path region: small batch, long segments
        # (2<=N<=6, avg_seg=T/N>=512). Pick the longest-avg-segment such case so
        # the A/B run directly exercises the generalized use_nows/dvsplit gate.
        small_long = [c for c in ragged
                      if 2 <= c["N"] <= 6 and (c["T_total"] // c["N"]) >= 512]
        if small_long:
            picks.append(max(small_long,
                             key=lambda c: (c["T_total"] // c["N"], c["T_total"])))
    picks.append(max(cases, key=lambda c: c["T_total"]))               # biggest overall
    # dedup on (N, T_total) preserving order
    seen, out = set(), []
    for c in picks:
        key = (c["N"], c["T_total"])
        if key not in seen:
            seen.add(key); out.append(c)
    return out[:k]


def pick_acc_cases(cases, max_t, k):
    """Small cases (every seq <= max_t) so the fp32 python reference stays cheap."""
    small = [c for c in cases if c["seq_lens"] and max(c["seq_lens"]) <= max_t]
    small.sort(key=lambda c: (c["N"], c["T_total"]))
    out, seen = [], set()
    for c in small:
        key = (c["N"], c["T_total"])
        if key in seen:
            continue
        seen.add(key); out.append(c)
        if len(out) >= k:
            break
    return out


def pick_fast_acc_case(cases, fast_max_t):
    """One smallest single-seq FAST-path case (H in {6,12}, 4096<=T<=fast_max_t)
    so the pro fast path itself gets an accuracy check (kept short: the fp32
    reference is a per-chunk python loop)."""
    fast = [c for c in cases if pro_fast(c) and c["N"] == 1 and c["T_total"] <= fast_max_t]
    return min(fast, key=lambda c: c["T_total"]) if fast else None


def make_inputs(T, H, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device="cuda", generator=g)
    return (
        mk(T, H, D), mk(T, H, D), mk(T, H, D), mk(T, H, D),   # q, k, v, g
        (mk(T, H)),                                            # beta
        torch.randn(H, dtype=torch.float32, device="cuda", generator=g),      # A_log
        torch.randn(H, D, dtype=torch.float32, device="cuda", generator=g),   # dt_bias
    )


def cu_tensor(seq_lens):
    return torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(),
                        dtype=torch.int64, device="cuda")


def run_acc(tp, c, seed=20260816):
    H, seq_lens = c["H"], c["seq_lens"]
    T = sum(seq_lens)
    lower = c["lower"]
    scale = 1.0 / math.sqrt(D)
    q, k, v, g, beta, A_log, dt_bias = make_inputs(T, H, seed)
    cu = cu_tensor(seq_lens) if c["N"] > 1 else None

    out = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
    final = torch.empty(c["N"], H, D, D, dtype=torch.bfloat16, device=q.device)
    flashkda_fwd(q, k, v, g, beta, A_log, dt_bias, out, scale, lower, None,
                 final, cu)
    torch.cuda.synchronize()

    zero_h0 = torch.zeros(H, D, D, dtype=torch.bfloat16)
    ref_outs, ref_states, off = [], [], 0
    for L in seq_lens:
        sl = slice(off, off + L)
        ro, rs = ref_fwd(q[sl].cpu(), k[sl].cpu(), v[sl].cpu(), g[sl].cpu(),
                         beta[sl].cpu(), A_log.cpu(), dt_bias.cpu(), scale, lower,
                         zero_h0, L, H)
        ref_outs.append(ro); ref_states.append(rs); off += L
    e_out = err_ratio(out.cpu().reshape(-1), torch.cat(ref_outs, 0).reshape(-1))
    e_st = err_ratio(final.cpu().reshape(-1), torch.stack(ref_states, 0).reshape(-1))
    ok = e_out < TOL and e_st < TOL
    fast = "FAST" if pro_fast(c) else "base"
    print(f"  tp{tp} H{H:<2d} N{c['N']:<3d} T{T:<6d} [{fast}]  "
          f"out={e_out:.3e} state={e_st:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def _time_ms(fn, warmup, iters, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(repeats):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort(); meds.append(ts[len(ts) // 2])
    meds.sort()
    return meds[len(meds) // 2]


def run_perf(tp, c, warmup, iters, repeats, fla_fn=None, seed=20260816):
    H, seq_lens = c["H"], c["seq_lens"]
    T = sum(seq_lens)
    lower = c["lower"]
    scale = 1.0 / math.sqrt(D)
    q, k, v, g, beta, A_log, dt_bias = make_inputs(T, H, seed)
    cu = cu_tensor(seq_lens) if c["N"] > 1 else None

    # out/final_state are caller-allocated out-params; allocate inside the timed
    # region so this stays comparable to the FLA baseline, which allocates too.
    def _ppu_call():
        out = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
        final = torch.empty(c["N"], H, D, D, dtype=torch.bfloat16, device=q.device)
        flashkda_fwd(q, k, v, g, beta, A_log, dt_bias, out, scale, lower, None,
                     final, cu)

    ppu = _time_ms(_ppu_call, warmup, iters, repeats)
    fast = "FAST" if pro_fast(c) else "base"
    line = f"  tp{tp} H{H:<2d} N{c['N']:<3d} T{T:<6d} [{fast}]  ppu={ppu:.4f}ms"
    if fla_fn is not None:
        try:
            fla = _time_ms(lambda: fla_fn(q, k, v, g, beta, A_log, dt_bias, scale, lower, cu),
                           warmup, iters, repeats)
            line += f"  fla={fla:.4f}ms  speedup={fla / ppu:.3f}x"
        except Exception as exc:  # noqa: BLE001
            line += f"  fla=ERR({type(exc).__name__})"
    print(line)


def maybe_load_fla():
    """Return an fla_fn(q,k,v,g,beta,A_log,dt_bias,scale,lower,cu)->() or None."""
    try:
        import inspect
        from fla.ops.kda import chunk_kda
        chunk_kda = inspect.unwrap(chunk_kda)
    except Exception as exc:  # noqa: BLE001
        print(f"[fla] unavailable ({type(exc).__name__}: {exc}); skipping --vs-fla")
        return None

    def fla_fn(q, k, v, g, beta, A_log, dt_bias, scale, lower, cu):
        # FLA packed contract: [1, T_total, H, D]; cu_seqlens int32.
        cu32 = cu.to(torch.int32) if cu is not None else None
        chunk_kda(q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
                  g=g.unsqueeze(0), beta=beta.unsqueeze(0), scale=scale,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=lower,
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True, cu_seqlens=cu32)
    return fla_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-dir", default=DEFAULT_LIST_DIR)
    ap.add_argument("--tp", default="1,2,4,8", help="comma list of TP degrees")
    ap.add_argument("--per-tp", type=int, default=4, help="perf cases per tp")
    ap.add_argument("--acc-per-tp", type=int, default=3, help="accuracy cases per tp")
    ap.add_argument("--acc-max-t", type=int, default=1024,
                    help="skip accuracy on any seq longer than this (ref is slow)")
    ap.add_argument("--fast-acc-max-t", type=int, default=4096,
                    help="also accuracy-check one FAST-path case (H in {6,12}, "
                         "N==1) up to this length; 0 disables")
    ap.add_argument("--no-acc", action="store_true")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--vs-fla", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    tps = [int(x) for x in args.tp.split(",") if x.strip()]
    per_tp_cases = {}
    for tp in tps:
        matches = glob.glob(os.path.join(args.list_dir, LIST_GLOB.format(tp=tp)))
        if not matches:
            print(f"[warn] no list file for tp{tp} in {args.list_dir}")
            continue
        per_tp_cases[tp] = parse_list(sorted(matches)[0])
        H = per_tp_cases[tp][0]["H"] if per_tp_cases[tp] else "?"
        print(f"[list] tp{tp}: {len(per_tp_cases[tp])} cases, H={H}")

    if not args.no_acc:
        print("\n== ACCURACY (vs fp32 torch reference; small cases only) ==")
        n_ok = n_tot = 0
        for tp in tps:
            cases = per_tp_cases.get(tp, [])
            acc_cases = pick_acc_cases(cases, args.acc_max_t, args.acc_per_tp)
            if args.fast_acc_max_t > 0:
                fc = pick_fast_acc_case(cases, args.fast_acc_max_t)
                if fc is not None:
                    acc_cases.append(fc)   # validate the pro FAST path too
            for c in acc_cases:
                n_tot += 1
                n_ok += int(run_acc(tp, c))
        print(f"  accuracy: {n_ok}/{n_tot} within TOL={TOL}")

    if not args.no_perf:
        fla_fn = maybe_load_fla() if args.vs_fla else None
        print("\n== PERFORMANCE (median ms; [FAST]=pro fast path) ==")
        for tp in tps:
            for c in pick_perf_cases(per_tp_cases.get(tp, []), args.per_tp):
                run_perf(tp, c, args.warmup, args.iters, args.repeats, fla_fn)


if __name__ == "__main__":
    main()
