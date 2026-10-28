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

"""FlashKDA prefill correctness test (xtest entry) -- full dispatch paths + real caselist shapes.

Path under test: ``pla.prefill.flashkdapro.flashkda_fwd`` (the CuTe kernel chain of Kimi-K3 KDA
prefill). dispatch/caselist retain the FP32 reference in
``scripts/prefill/flashkdapro/private/test_fwd_e2e.py`` and ``err_ratio < TOL``.
master uses an independent FP64 closed form with per-head
``RMS(actual - reference) / RMS(reference) <= 0.05`` or absolute error <= 1e-6.

## Relation to the two manual scripts (reused only, never rewritten)

  * ``scripts/prefill/flashkdapro/e2e_acc_full.py``
    provides the dispatch path matrix ``CASES`` and the single-case body ``check()``: initial_state /
    final_state dtype combinations, the bit-identical contract of ``final_state=None``, chunk tails that are not a
    multiple of 16, and all-zero padded tails. The K2 variants of flashkda_fwd are **selected by shape**
    (dvsplit BV16/BV32, full-DV nows, varlen nows, baseline DVT32); this table is the shortest set of shapes
    that hits every variant once.
  * ``scripts/prefill/flashkdapro/e2e_kimi_k3.py``
    provides caselist parsing and representative shape selection: ``parse_list()`` takes
    ``num_heads_qk`` / ``gate_lower_bound`` / ``cu_seqlens_q`` from the ``.list``, ``pick_acc_cases()``
    keeps only small cases where every sequence is <= max_t (the fp32 reference is a per-chunk python loop, long cases
    never finish inside the accuracy gate), and ``pro_fast()`` replicates the use_nows gating of kernel2_recurrence.cu
    so the report shows whether a shape hits the pro fast path or the baseline.

## Three test stages

  * ``dispatch``: runs the path matrix of e2e_acc_full, making sure every K2 variant and state contract is hit.
  * ``caselist``: runs the real shapes of ``caselist/prefill/Kimi-K3_prefill_tp{1,2,4,8}_*kda_op.list``
    (tp1/2/4/8 map to per-card H=96/48/24/12), taking ``--per-tp`` of them per tp.
  * ``master``: tc02 serial FP32 master state; reduce output over valid tokens x V
    and state over V x K, independently per sequence and head. Skip output for
    empty sequences; use the absolute-error fallback for a zero reference.

## Output

Markers are printed per sub-case following the tests/case_report.py protocol; acompute.PLAResultParser turns each
shape into one ES record and one JUnit testcase row; the manual scripts' original prints are kept as info lines
inside the sub-case, and go straight into the ES detail field on failure.

Usage (inside the container, from the repo root; must be run as a plain script, otherwise the source tree's pla/ shadows the installed package):

    python tests/test_kda_prefill_flashkdapro.py
    python tests/test_kda_prefill_flashkdapro.py --phase caselist --tp 8,4 --per-tp 4
    python tests/test_kda_prefill_flashkdapro.py --acc-max-t 512
"""
import argparse
import glob
import io
import os
import sys
from contextlib import redirect_stdout

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
# Same as other pla test scripts: do not add the repo root to sys.path; the source tree's pla/ has no build artifacts
# and would shadow the installed package entirely.
_PRO_DIR = os.path.join(_REPO_ROOT, "scripts", "prefill", "flashkdapro")
for _p in (_TESTS_DIR, _PRO_DIR, os.path.join(_PRO_DIR, "private")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from case_report import CaseReporter  # noqa: E402

import e2e_acc_full as acc  # noqa: E402  dispatch matrix + check()
import e2e_kimi_k3 as k3  # noqa: E402   caselist parsing + representative shape selection

DEFAULT_LIST_DIR = "caselist/prefill"
DEFAULT_TPS = (1, 2, 4, 8)

# (h0 dtype, list of final_state dtypes to examine) -- semantics in the docstring of e2e_acc_full.check:
# a zero initial state allows final_state to be bf16 or fp32; with an initial state upstream requires both dtypes to match.
H0_MODES = (
    (None, (torch.bfloat16, torch.float32)),
    (torch.bfloat16, (torch.bfloat16,)),
    (torch.float32, (torch.float32,)),
)


def _dt_tag(dtype):
    if dtype is None:
        return "zero"
    return "f32" if dtype == torch.float32 else "bf16"


def _run_captured(fn, *args, **kwargs):
    """Run the reference script's body once and capture its prints as info lines.

    The manual scripts "print as they go"; calling them directly mixes their lines between the case markers, and the
    numbers on failure never reach the detail field. Capturing and re-emitting keeps the original info and allows building detail.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        ret = fn(*args, **kwargs)
    return ret, buf.getvalue().rstrip("\n")


def case_dispatch(rep, tag, path, H, seq_lens, h0_dtype, final_dtypes, pad_tokens):
    flags, text = _run_captured(
        acc.check, tag, path, H, seq_lens, h0_dtype, final_dtypes,
        pad_tokens=pad_tokens)
    rep.info(f"K2 path:{path}, num_heads:{H}, batch_size:{len(seq_lens)}, "
             f"seq_lens:{seq_lens}, pad_tokens:{pad_tokens}, "
             f"initial_state:{_dt_tag(h0_dtype)}")
    for line in text.splitlines():
        rep.info(line)
    n_ok = sum(1 for f in flags if f)
    if n_ok == len(flags):
        return []
    bad = [l.strip() for l in text.splitlines() if "FAIL" in l]
    return [f"{n_ok}/{len(flags)} checks OK"] + bad[:4]


def case_from_caselist(rep, tp, c):
    ok, text = _run_captured(k3.run_acc, tp, c)
    fast = "FAST" if k3.pro_fast(c) else "base"
    rep.info(f"tp:{tp}, num_heads:{c['H']}, batch_size:{c['N']}, "
             f"T_total:{c['T_total']}, gate_lower_bound:{c['lower']}, "
             f"dispatch:{fast}")
    for line in text.splitlines():
        rep.info(line)
    if ok:
        return []
    return [f"tp{tp} H{c['H']} N{c['N']} T{c['T_total']} accuracy fail: {text.strip()}"]


def build_caselist_plan(list_dir, tps, per_tp, acc_max_t):
    """Walk the prefill caselists per tp and pick the representative shapes of each tp."""
    plan, sources = [], []
    for tp in tps:
        pattern = os.path.join(list_dir, k3.LIST_GLOB.format(tp=tp))
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: no matching caselist: {pattern}")
            continue
        path = matches[0]
        sources.append(path)
        cases = k3.parse_list(path)
        picked = k3.pick_acc_cases(cases, acc_max_t, per_tp)
        if not picked:
            print(f"WARNING: {os.path.basename(path)} has no case where every sequence is "
                  f"<= {acc_max_t}, skipping")
            continue
        for c in picked:
            label = f"tp{tp}_H{c['H']}_N{c['N']}_T{c['T_total']}"
            plan.append((label, case_from_caselist, (tp, c)))
    return plan, sources


def _master_metrics(actual, expected):
    actual, expected = actual.cpu().double(), expected.cpu().double()
    assert actual.shape == expected.shape
    delta = actual - expected
    if not delta.numel():
        return 0.0, 0.0, True
    error_rms = float(delta.square().mean().sqrt())
    reference_rms = float(expected.square().mean().sqrt())
    rel = error_rms / reference_rms if reference_rms > 0 else (
        0.0 if error_rms == 0 else float("inf"))
    absolute = float(delta.abs().max())
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()
                  and torch.isfinite(delta).all())
    return rel, absolute, finite and (rel <= 0.05 or absolute <= 1e-6)


def _check_master_metric_contract():
    expected = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    actual = expected + 0.125
    relative, _, passed = _master_metrics(actual, expected)
    previous = float((actual - expected).norm() / expected.norm())
    assert abs(relative - previous) < 1e-14 and passed
    repeated, _, _ = _master_metrics(actual.repeat(7, 1), expected.repeat(7, 1))
    assert abs(relative - repeated) < 1e-14
    zero = torch.zeros(2, 2, dtype=torch.float64)
    assert _master_metrics(zero, zero) == (0.0, 0.0, True)
    assert _master_metrics(zero + 1e-7, zero)[2]
    assert not _master_metrics(zero + 1e-3, zero)[2]
    for value in (float("nan"), float("inf")):
        assert not _master_metrics(torch.full_like(zero, value), zero)[2]
    empty = torch.empty(0, dtype=torch.float64)
    assert _master_metrics(empty, empty) == (0.0, 0.0, True)
    print("relative RMS metric contract passed: equivalence, repetition, zero reference, empty input, nonfinite values")


def case_master_state(rep, mode, scenario, state_dtype, heads=4, single=False, master_tokens=None):
    saved = {k: v for k, v in os.environ.items() if k.startswith("PLA_FLASHKDA_")}
    for key in saved:
        del os.environ[key]
    os.environ["PLA_FLASHKDA_CP"] = "0"
    if mode.startswith("cp"):
        os.environ["PLA_FLASHKDA_CP"] = "1" if mode == "cp_auto" else "2"
        if mode != "cp_auto":
            os.environ["PLA_FLASHKDA_CP_BV"] = mode[2:]
    elif mode == "shared":
        os.environ["PLA_FLASHKDA_K2_NOWS"] = "0"
    elif mode == "full_dv":
        os.environ["PLA_FLASHKDA_K2_NOWS"] = "1"
        os.environ["PLA_FLASHKDA_K2_DVSPLIT"] = "0"
    elif mode == "dk4":
        os.environ.update(PLA_FLASHKDA_K2_NOWS="1", PLA_FLASHKDA_K2_DVSPLIT="16",
                          PLA_FLASHKDA_K2_DKSPLIT="4", PLA_FLASHKDA_K2_DKLEADER="0")
    try:
        return _case_master_state(rep, scenario, state_dtype, heads, single, master_tokens)
    finally:
        for key in list(os.environ):
            if key.startswith("PLA_FLASHKDA_"):
                del os.environ[key]
        os.environ.update(saved)


def _case_master_state(rep, scenario, state_dtype, heads, single, master_tokens=None):
    from pla.prefill.flashkdapro import flashkda_fwd

    lengths = [0, 17, 4099] if scenario != "initial_bits" else [0, 1, 17]
    if single:
        lengths = [17 if scenario == "initial_bits" else 4099]
    if master_tokens is not None:
        assert master_tokens > 17
        lengths[-1] = master_tokens
    rep.info(f"sequence_lengths={lengths}, heads={heads}, state_dtype={state_dtype}, scenario={scenario}")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    total, dim = offsets[-1], 128
    q = torch.zeros(total + 3, heads, dim, dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    updating = scenario in ("small_update", "fresh_update")
    if updating:
        k[..., 0] = 1
        v.fill_(1)
    raw_value = -12.0 if scenario == "weak_decay" else -100.0
    raw = torch.full_like(q, raw_value)
    beta = torch.full((total + 3, heads), -12.0, dtype=q.dtype, device=q.device)
    a_log = torch.zeros(heads, device=q.device)
    dt_bias = torch.zeros(heads, dim, device=q.device)
    h0 = torch.linspace(-0.25, 0.25, len(lengths) * heads * dim * dim,
                        dtype=torch.float32).reshape(len(lengths), heads, dim, dim)
    if updating:
        h0.zero_()
        if scenario == "small_update":
            h0[..., 0] = 0.1253
    h0 = h0.to(state_dtype).cuda()
    h0_before = h0.cpu().clone()
    initial = None if scenario == "fresh_update" else h0
    cu = torch.tensor(offsets, dtype=torch.int32, device=q.device)
    out = torch.full_like(q, 7)
    final = torch.empty_like(h0)
    inputs = (q, k, v, raw, beta, a_log, dt_bias)
    kwargs = dict(initial_state=initial, cu_seqlens=cu, raw_cu_seqlens=tuple(offsets))
    flashkda_fwd(*inputs, out, final_state=final, **kwargs)
    no_state = torch.full_like(out, 9)
    flashkda_fwd(*inputs, no_state, final_state=None, **kwargs)
    failures = []
    if not torch.equal(out, no_state):
        failures.append("final_state=None changed output")
    if torch.count_nonzero(out[total:]).item() != 0:
        failures.append("padding was not zeroed")
    if scenario == "initial_bits" and not torch.equal(final.cpu(), h0_before):
        failures.append("identity recurrence lost initial-state bits")

    g = -5.0 * torch.sigmoid(torch.tensor(raw_value, dtype=torch.float64))
    b = torch.sigmoid(torch.tensor(-12.0, dtype=torch.float64))
    decay = g.exp()
    recurrence = decay * (1 - b) if updating else decay
    if os.environ.get("PLA_FLASHKDA_CP") in ("1", "2"):
        failures += _check_cp_master_stages(rep, inputs, initial, cu, offsets,
                                            decay, recurrence, b, updating)
        if state_dtype == torch.float32 and updating and torch.equal(final, final.bfloat16().float()):
            failures.append("CP FP32 final state contains only BF16 master values")
    for seq, length in enumerate(lengths):
        start = h0_before[seq].double()
        end = start * decay.pow(length)
        steps = torch.arange(1, length + 1, dtype=torch.float64)
        values = recurrence.pow(steps)[:, None, None] * start[..., 0]
        if updating:
            values += (b * (-torch.expm1(steps * recurrence.log())) / (1 - recurrence))[:, None, None]
            end[..., 0] = (recurrence.pow(length) * start[..., 0]
                          + b * (-torch.expm1(length * recurrence.log())) / (1 - recurrence))
        expected_out = values * dim ** -0.5
        chained_out, chained_final = None, None
        if state_dtype == torch.float32 and length > 17:
            carry = None if initial is None else h0[seq:seq + 1].clone()
            parts = []
            split = (length // 32) * 16
            for begin, stop in ((0, split), (split, length)):
                sl = slice(offsets[seq] + begin, offsets[seq] + stop)
                part = torch.empty_like(out[sl])
                next_state = torch.empty_like(h0[seq:seq + 1])
                part_cu = torch.tensor([0, stop - begin], dtype=torch.int32, device=q.device)
                flashkda_fwd(*(t[sl] for t in inputs[:5]), a_log, dt_bias, part,
                             initial_state=carry, final_state=next_state,
                             cu_seqlens=part_cu, raw_cu_seqlens=(0, stop - begin))
                parts.append(part)
                carry = next_state
            chained_out, chained_final = torch.cat(parts), carry[0]
        for head in range(heads):
            pairs = [("state", final[seq, head], end[head])]
            if length:
                pairs.append(("output", out[offsets[seq]:offsets[seq + 1], head], expected_out[:, head]))
            if chained_out is not None:
                pairs.extend((("chained_state", chained_final[head], end[head]),
                              ("chained_output", chained_out[:, head], expected_out[:, head])))
            for target, actual, expected in pairs:
                rel, absolute, passed = _master_metrics(actual, expected)
                rep.info(f"seq={seq} head={head} {target} relative_rms={rel:.8g} max_abs={absolute:.8g}")
                if not passed:
                    failures.append(f"seq={seq} head={head} {target}: relative_rms={rel:g} max_abs={absolute:g}")

    slots = len(lengths) + 4
    indices = torch.tensor([3] if single else [5, 1, 3], dtype=torch.int32, device=q.device)
    pool = torch.full((slots, heads, dim, dim), 11, dtype=state_dtype, device=q.device)
    pool[indices.long()] = h0
    before = pool.clone()
    indexed_out = torch.full_like(out, 13)
    flashkda_fwd(*inputs, indexed_out, state_pool=pool, state_indices=indices,
                 cu_seqlens=cu, raw_cu_seqlens=tuple(offsets))
    if not torch.equal(indexed_out, out) or not torch.equal(pool[indices.long()], final):
        failures.append("indexed/compact state mismatch")
    untouched = [i for i in range(slots) if i not in indices.cpu().tolist()]
    if not torch.equal(pool[untouched], before[untouched]):
        failures.append("unmapped pool slot changed")
    if not torch.equal(h0.cpu(), h0_before):
        failures.append("read-only initial state changed")
    return failures


def _check_cp_master_stages(rep, inputs, initial, cu, offsets, decay, recurrence, beta, updating):
    from pla.prefill.flashkdapro import _C

    total, heads, dim = inputs[0].shape
    plan = dict(_C.flashkda_cp_plan(total, heads, cu, tuple(offsets)))
    expected_fallback = (os.environ["PLA_FLASHKDA_CP"] == "1" and heads == 8
                         and len(offsets) == 4 and max(b - a for a, b in zip(offsets, offsets[1:])) < 32768)
    assert plan["engage"] == (not expected_fallback), plan
    rep.info(f"CP plan={plan}; expected_serial_fallback={expected_fallback}")
    if expected_fallback:
        return []
    assert plan["n_seg"] > len(offsets) - 1, plan
    chains, carries = _C.flashkda_cp_stages(*inputs, dim ** -0.5, -5., initial, cu, tuple(offsets))
    assert chains.dtype == carries.dtype == torch.bfloat16
    assert tuple(chains.shape) == (2, plan["n_seg"], heads, dim, dim)
    assert tuple(carries.shape) == (plan["n_seg"], heads, dim, dim)
    chains, carries = chains.cpu(), carries.cpu()
    failures = []
    initial_cpu = (torch.zeros(len(offsets) - 1, heads, dim, dim, dtype=torch.float64)
                   if initial is None else initial.cpu().bfloat16().double())
    checked_segments = 0
    for seq in range(len(offsets) - 1):
        first, count = plan["seq_seg"][2 * seq:2 * seq + 2]
        for seg in range(first, first + count):
            checked_segments += 1
            begin, length, _, outslot = plan["seg_tbl"][4 * seg:4 * seg + 4]
            steps = begin - offsets[seq]
            expected_carry = initial_cpu[seq] * decay.pow(steps)
            if updating:
                expected_carry[..., 0] = (initial_cpu[seq, ..., 0] * recurrence.pow(steps)
                    + beta * (-torch.expm1(steps * recurrence.log())) / (1 - recurrence))
            pairs = [("carry", carries[seg], expected_carry)]
            if outslot < 0:
                expected_m = torch.eye(dim, dtype=torch.float64) * decay.pow(length)
                expected_b = torch.zeros(dim, dim, dtype=torch.float64)
                if updating:
                    expected_m[0, 0] = recurrence.pow(length)
                    expected_b[:, 0] = beta * (-torch.expm1(length * recurrence.log())) / (1 - recurrence)
                pairs += [("M", chains[1, seg], expected_m.expand(heads, -1, -1)),
                          ("B", chains[0, seg], expected_b.expand(heads, -1, -1))]
            for target, actual, expected in pairs:
                for head in range(heads):
                    rel, absolute, passed = _master_metrics(actual[head], expected[head])
                    rep.info(f"seg={seg} head={head} {target} relative_rms={rel:.8g} max_abs={absolute:.8g}")
                    if not passed:
                        failures.append(f"seg={seg} head={head} {target}: relative_rms={rel:g} max_abs={absolute:g}")
    assert checked_segments == plan["n_seg"]
    return failures


def case_master_short_random(rep, tokens, state_dtype):
    from pla.prefill.flashkdapro import flashkda_fwd

    saved = {k: v for k, v in os.environ.items() if k.startswith("PLA_FLASHKDA_")}
    for key in saved:
        del os.environ[key]
    os.environ["PLA_FLASHKDA_CP"] = "0"
    try:
        heads, dim = 32, 128
        inputs = acc.make_inputs(tokens, heads, 20261023)
        gen = torch.Generator(device="cuda").manual_seed(20261024)
        initial = None if state_dtype is None else torch.randn(
            1, heads, dim, dim, dtype=state_dtype, device="cuda", generator=gen) * 0.1
        initial_before = None if initial is None else initial.cpu().clone()
        q, k, v, raw, beta, a_log, dt_bias = [x.cpu() for x in inputs]
        q, k = [(x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
                 ).bfloat16().double() for x in (q, k)]
        q *= dim ** -0.5
        decay = (-5 * torch.sigmoid(a_log.double().exp()[None, :, None]
                                   * (raw.double() + dt_bias.double()))).exp()
        beta = beta.double().sigmoid()
        state = torch.zeros(heads, dim, dim, dtype=torch.float64) if initial is None else initial_before[0].double()
        reference = torch.empty(tokens, heads, dim, dtype=torch.float64)
        for t in range(tokens):
            state *= decay[t, :, None, :]
            delta = (v[t].double() - (state * k[t, :, None, :]).sum(-1)) * beta[t, :, None]
            state += delta[:, :, None] * k[t, :, None, :]
            reference[t] = (state * q[t, :, None, :]).sum(-1)
        cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
        out = torch.empty_like(inputs[2])
        failures = []
        for final_dtype in ((torch.bfloat16, torch.float32) if initial is None else (state_dtype,)):
            final = torch.empty(1, heads, dim, dim, dtype=final_dtype, device="cuda")
            flashkda_fwd(*inputs, out, initial_state=initial, final_state=final, cu_seqlens=cu)
            no_state = torch.empty_like(out)
            flashkda_fwd(*inputs, no_state, initial_state=initial, cu_seqlens=cu)
            if not torch.equal(out, no_state):
                failures.append("final_state=None changed random output")
            for head in range(heads):
                for target, actual, expected in (("output", out[:, head], reference[:, head]),
                                                 ("state", final[0, head], state[head])):
                    rel, absolute, passed = _master_metrics(actual, expected)
                    rep.info(f"head={head} {target} relative_rms={rel:.8g} max_abs={absolute:.8g}")
                    if not passed:
                        failures.append(f"head={head} {target} relative_rms={rel:g} max_abs={absolute:g}")
        if initial is not None and not torch.equal(initial.cpu(), initial_before):
            failures.append("read-only random initial state changed")
        return failures
    finally:
        for key in list(os.environ):
            if key.startswith("PLA_FLASHKDA_"):
                del os.environ[key]
        os.environ.update(saved)


def case_prepare_diagonal(rep, layout, threads, chunks_per_cta):
    """Check every chunk diagonal before affine composition, including tails."""
    from pla.prefill.flashkdapro import _C

    saved = {k: v for k, v in os.environ.items() if k.startswith("PLA_FLASHKDA_")}
    for key in saved:
        del os.environ[key]
    os.environ.update(PLA_FLASHKDA_CP="0", PLA_FLASHKDA_K1_THREADS=str(threads),
                      PLA_FLASHKDA_K1_CPC=str(chunks_per_cta))
    try:
        lengths = {"fixed": (50,), "ragged": (1, 0, 16, 33),
                   "prefix": (1, 0, 16, 17) * 16}[layout]
        heads = 32 if layout == "fixed" else 4
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        total = offsets[-1] + (0 if layout == "fixed" else 7)
        q, k, _, raw, beta, a_log, dt_bias = acc.make_inputs(total, heads, 20261023)
        cu = None if layout == "fixed" else torch.tensor(offsets, dtype=torch.int32, device=q.device)
        tiles = (total + 15) // 16 + (0 if cu is None else len(lengths))
        normalized = [(x.cpu().float() * torch.rsqrt(
            x.cpu().float().square().sum(-1, keepdim=True) + 1e-6)).bfloat16().double()
            for x in (q, k)]
        failures = []
        for scale in (128 ** -0.5, 0.125):
            dots = (normalized[0] * normalized[1]).sum(-1) * scale
            expected = torch.zeros(heads, tiles, 16, dtype=torch.float64)
            tile = 0
            for start, stop in zip(offsets, offsets[1:]):
                for begin in range(start, stop, 16):
                    count = min(16, stop - begin)
                    expected[:, tile, :count] = dots[begin:begin + count].T
                    tile += 1
            gate_zero = None
            for lower in (0.0, -0.75, -5.0):
                mqk = _C.flashkda_prepare(q, k, raw, beta, a_log, dt_bias,
                                         scale, lower, cu)[5].cpu().view(heads, tiles, 16, 16)
                actual = mqk.diagonal(dim1=-2, dim2=-1).contiguous()
                if gate_zero is None:
                    gate_zero = actual.clone()
                elif not torch.equal(actual, gate_zero):
                    failures.append(f"diagonal depends on gate at scale={scale} lower={lower}")
                if not torch.isfinite(mqk).all() or torch.count_nonzero(mqk.triu(1)):
                    failures.append("non-finite or nonzero upper-triangular Mqk")
                if torch.count_nonzero(actual[expected == 0]):
                    failures.append("zero or padded diagonal changed")
                for head in range(heads):
                    rel, absolute, passed = _master_metrics(actual[head], expected[head])
                    rep.info(f"threads={threads} cpc={chunks_per_cta} layout={layout} "
                             f"scale={scale} lower={lower} head={head} diagonal "
                             f"relative_rms={rel:.8g} max_abs={absolute:.8g}")
                    if not passed:
                        failures.append(f"head={head} diagonal relative_rms={rel:g} max_abs={absolute:g}")
        return failures
    finally:
        for key in list(os.environ):
            if key.startswith("PLA_FLASHKDA_"):
                del os.environ[key]
        os.environ.update(saved)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["all", "dispatch", "caselist", "master",
                                       "master-short", "master-cp", "prepare-diagonal"], default="all")
    ap.add_argument("--list-dir", default=DEFAULT_LIST_DIR,
                    help="prefill caselist directory (relative to the repo root)")
    ap.add_argument("--tp", default=",".join(str(t) for t in DEFAULT_TPS),
                    help="tp degrees to walk (per-card H=96/48/24/12)")
    ap.add_argument("--per-tp", type=int, default=2,
                    help="how many representative shapes to take from each tp caselist (default 2)")
    ap.add_argument("--acc-max-t", type=int, default=1024,
                    help="per-sequence length cap of the caselist stage: the fp32 reference is a per-chunk "
                         "python loop, long cases never finish inside the accuracy gate (default 1024)")
    ap.add_argument("--master-tokens", type=int,
                    help="Override the longest effective sequence in every FP64 master-state case")
    args = ap.parse_args()
    if args.master_tokens is not None and args.master_tokens <= 17:
        ap.error("--master-tokens must be greater than 17")

    list_dir = args.list_dir if os.path.isabs(args.list_dir) \
        else os.path.join(_REPO_ROOT, args.list_dir)
    tps = [int(t) for t in args.tp.split(",") if t.strip()]

    plan, sources = [], []
    if args.phase in ("all", "dispatch"):
        for tag, path, H, seq_lens in acc.CASES:
            for h0_dtype, final_dtypes in H0_MODES:
                plan.append((f"{tag}_{_dt_tag(h0_dtype)}", case_dispatch,
                             (tag, path, H, seq_lens, h0_dtype, final_dtypes, 0)))
        # padded tail: only examined with zero and bf16 initial states, consistent with the manual script
        for tag, path, H, seq_lens, pad_tokens in (
                ("b1-h12-s2205-pad1", "padded varlen", 12, [2205], 1),
                ("b4-h12-ragged-pad7", "padded varlen", 12, [100, 273, 16, 272], 7)):
            for h0_dtype, final_dtypes in H0_MODES[:2]:
                plan.append((f"{tag}_{_dt_tag(h0_dtype)}", case_dispatch,
                             (tag, path, H, seq_lens, h0_dtype, final_dtypes,
                              pad_tokens)))
    if args.phase in ("all", "caselist"):
        cl_plan, sources = build_caselist_plan(list_dir, tps, args.per_tp,
                                               args.acc_max_t)
        if not cl_plan and args.phase == "caselist":
            print(f"ERROR: no usable Kimi-K3 prefill caselist under {list_dir}",
                  file=sys.stderr)
            return 2
        plan += cl_plan

    if args.phase in ("all", "master"):
        _check_master_metric_contract()
        if torch.cuda.get_device_capability() == (8, 9):
            for mode in ("auto", "shared", "full_dv", "dk4"):
                for scenario in ("initial_bits", "weak_decay", "small_update", "fresh_update"):
                    for dtype in (torch.bfloat16, torch.float32):
                        plan.append((f"master_{mode}_{scenario}_{_dt_tag(dtype)}",
                                     case_master_state, (mode, scenario, dtype, 4, False, args.master_tokens)))
            for heads in (6, 8):
                for scenario in ("initial_bits", "weak_decay", "small_update", "fresh_update"):
                    for dtype in (torch.bfloat16, torch.float32):
                        plan.append((f"master_auto_h{heads}_n1_{scenario}_{_dt_tag(dtype)}",
                                     case_master_state, ("auto", scenario, dtype, heads, True, args.master_tokens)))
        elif args.phase == "master":
            print("ERROR: FP32 serial master-state regression requires tc02", file=sys.stderr)
            return 2

    if args.phase in ("all", "master-cp"):
        _check_master_metric_contract()
        if torch.cuda.get_device_capability() == (8, 9):
            for mode in ("cp_auto", "cp16", "cp32", "cp128"):
                for single in (True, False):
                    for scenario in ("weak_decay", "small_update", "fresh_update"):
                        for dtype in (torch.bfloat16, torch.float32):
                            plan.append((f"master_{mode}_single{int(single)}_{scenario}_{_dt_tag(dtype)}",
                                         case_master_state, (mode, scenario, dtype, 8, single,
                                                             args.master_tokens or 8183)))
        elif args.phase == "master-cp":
            print("ERROR: CP FP32 master-state regression requires tc02", file=sys.stderr)
            return 2

    if args.phase in ("all", "master-short"):
        _check_master_metric_contract()
        if torch.cuda.get_device_capability() == (8, 9):
            for mode, tokens in (("auto", 24), ("auto", 194), ("shared", 8192)):
                for scenario in ("initial_bits", "weak_decay", "small_update", "fresh_update"):
                    for dtype in (torch.bfloat16, torch.float32):
                        plan.append((f"master_h32_{mode}_t{tokens}_{scenario}_{_dt_tag(dtype)}",
                                     case_master_state, (mode, scenario, dtype, 32, True, tokens)))
            for tokens in (1, 16, 24, 194, 255, 256):
                for dtype in (None, torch.bfloat16, torch.float32):
                    plan.append((f"master_h32_random_t{tokens}_{_dt_tag(dtype)}",
                                 case_master_short_random, (tokens, dtype)))
        elif args.phase == "master-short":
            print("ERROR: FP32 short master-state regression requires tc02", file=sys.stderr)
            return 2

    if args.phase in ("all", "prepare-diagonal"):
        _check_master_metric_contract()
        if torch.cuda.get_device_capability() == (8, 9):
            for threads in (64, 96, 128, 256):
                for layout in ("fixed", "ragged", "prefix"):
                    plan.append((f"prepare_diagonal_{layout}_nt{threads}",
                                 case_prepare_diagonal, (layout, threads, 1)))
            plan.append(("prepare_diagonal_fixed_nt128_cpc2",
                         case_prepare_diagonal, ("fixed", 128, 2)))
        elif args.phase == "prepare-diagonal":
            print("ERROR: stable prepare diagonal regression requires tc02", file=sys.stderr)
            return 2

    # Case file tags the sub-case names with their source; in dispatch-only mode there is no caselist, so tag the script itself
    case_file = sources[0] if sources else "tests/test_kda_prefill_flashkdapro.py"
    rep = CaseReporter("kda_prefill_flashkdapro", case_file, total=len(plan))
    rep.header()
    if args.phase not in ("master", "master-short", "master-cp", "prepare-diagonal"):
        print(f"oracle: fp32 torch ref_fwd (private/test_fwd_e2e.py), "
              f"judge err_ratio < {acc.TOL:g}")
    print(f"phase: {args.phase}")
    if args.phase in ("all", "prepare-diagonal"):
        print("diagonal oracle: FP64 dot of normalized BF16 Q/K; gate invariance; per-head RMS <= 0.05 OR max_abs <= 1e-6")
    if args.phase in ("all", "master", "master-short", "master-cp"):
        print("master oracle: FP64 closed-form recurrence; per-head RMS(error)/RMS(reference) <= 0.05 OR max_abs <= 1e-6")
    if sources:
        print(f"caselists: {[os.path.basename(s) for s in sources]}")
        print(f"per-tp picks: {args.per_tp}, acc_max_t: {args.acc_max_t}")

    for label, fn, fn_args in plan:
        rep.start(label)
        if rep.aborted:
            rep.fail("skipped: an earlier case already brought the device down")
            continue
        with rep.guard():
            torch.cuda.empty_cache()
            failures = fn(rep, *fn_args)
            if failures:
                rep.fail("; ".join(failures[:5]))
            else:
                rep.ok()

    return rep.summary()


if __name__ == "__main__":
    sys.exit(main())
