# SPDX-License-Identifier: Apache-2.0
"""Cached replay accuracy, lifecycle, and opt-in performance validation.

python -B tests/test_gdn_replay_cached.py --static
python -B tests/test_gdn_replay_cached.py --device cuda:N
python -B tests/test_gdn_replay_cached.py --device cuda:N --suite performance

Only this NEW file is needed. Importing it, including its mathematical and typed
head references, requires no torch, CUDA, Triton, or production PLA import.
Numerical suites never select an implicit device, fall back to CPU, or count a
skip as a pass. Performance is separate and has no invented latency threshold.
FP32/TF32 is gated by the typed reference; unquantized FP64 error is reported.
IEEE retains the mathematical budget. The static tests check the HLM adapter
call signatures without importing an accelerator runtime.

Reusable fixtures: CachedCase, make_cached_case, copy_case, public_bundle,
adapter_kwargs, reference_cached, reference_fold, write_masks. All fixtures are
CPU-seeded; device copies and reference evaluation belong OUTSIDE timed regions.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, fields, is_dataclass, replace
import importlib
import importlib.util
import itertools
import json
import math
import inspect
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pla" / "decode" / "gdn_replayssm"
ADAPTER = ROOT / "pla" / "decode" / "gdn_replayssm_hlm" / "__init__.py"
UPDATE = "fused_recurrent_gated_delta_rule_cached_replay_update"
COMMIT = "commit_gdn_cached_replay_history_layers"
BUDGETS = {
    "float32": {"output": (1e-4, 1e-3), "state": (1.5e-3, 1e-3)},
    "bfloat16": {"output": (1e-3, 1e-2), "state": (3e-3, 1e-2)},
    "float16": {"output": (5e-4, 3e-3), "state": (1.5e-3, 3e-3)},
}
OPTIONS = argparse.Namespace(device=None, range_case=None)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def quantize(x, dtype):
    """RNE storage conversion, independently implemented with stdlib bit casts."""
    x = f32(x)
    if dtype == "float32":
        return x
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<e", x))[0]
    if dtype != "bfloat16":
        raise ValueError(dtype)
    if not math.isfinite(x):
        return x
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def tf32_rtz(x):
    """Truncate FP32 operands toward zero, preserving NaN/Inf (not TF32 RNE)."""
    x = f32(x)
    if not math.isfinite(x):
        return x
    bits = struct.unpack("<I", struct.pack("<f", x))[0] & 0xffffe000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@dataclass(frozen=True)
class CachedCase:
    B: int = 3
    H: int = 2
    HV: int = 4
    K: int = 32
    V: int = 16
    T: int = 4
    L: int = 11
    dtype: str = "bfloat16"
    state_dtype: str = "float32"
    norm: str = "l2_outside_sqrt"
    gate_mode: str = "processed"
    dot_precision: str = "tf32"
    epsilon: float = 1e-6
    scale: float | None = None

    def validate(self):
        if (self.B < 0 or not 1 <= self.T <= 8 or not self.T <= self.L <= 16
                or not 1 <= self.K <= 256 or min(self.H, self.HV, self.V) < 1
                or self.HV % self.H):
            raise ValueError("require B>=0, H|HV, K<=256 and 1<=T<=8, T<=L<=16")
        if self.dtype not in BUDGETS or self.state_dtype not in (self.dtype, "float32"):
            raise ValueError("state must have input dtype or float32")
        if self.norm not in ("none", "l2_inside_sqrt", "l2_outside_sqrt"):
            raise ValueError("invalid norm")
        if self.gate_mode not in ("processed", "raw") or self.dot_precision not in ("ieee", "tf32"):
            raise ValueError("invalid numeric policy")
        return self


class Arithmetic:
    """FP32 operation boundaries, not a claim to duplicate GPU reduction order.

    Products/sums are FP32; fsum fixes a deterministic CPU reduction before the
    final FP32 rounding. GPU tree/FMA differences use the unchanged GDN budgets.
    Only tensor-core-style dot operands use TF32, not the scalar new-U solve.
    """
    def __init__(self, case, typed):
        self.c, self.typed = case, typed
        self.r = f32 if typed else float

    def add(self, x, y):
        return self.r(x + y)

    def sub(self, x, y):
        return self.r(x - y)

    def mul(self, x, y):
        return self.r(x * y)

    def exp(self, x):
        return self.r(math.exp(x))

    def sum(self, xs):
        return self.r(math.fsum(xs))

    def dot(self, x, y, tensor_core=True):
        if self.typed and tensor_core and self.c.dtype == "float32" and self.c.dot_precision == "tf32":
            x, y = list(map(tf32_rtz, x)), list(map(tf32_rtz, y))
        return self.sum(self.mul(a, b) for a, b in zip(x, y))

    def store(self, x):
        return quantize(x, self.c.dtype) if self.typed else x

    def normalized(self, row, scale=1.0):
        if self.c.norm == "none":
            inverse = self.r(scale)
        else:
            square = self.sum(self.mul(x, x) for x in row)
            if self.c.norm == "l2_inside_sqrt":
                denom = self.r(math.sqrt(self.add(square, self.c.epsilon)))
            else:
                denom = self.add(self.r(math.sqrt(square)), self.c.epsilon)
            inverse = self.r(scale / denom)
        return [self.store(self.mul(x, inverse)) for x in row]

    def gating(self, gate, beta, A_log, dt_bias):
        if self.c.gate_mode == "processed":
            return self.r(gate), self.r(beta)
        x = self.add(gate, dt_bias)
        # Source's thresholded softplus, with explicit intermediate rounding.
        sp = self.r(math.log(self.add(1.0, self.exp(x)))) if x <= 20.0 else x
        return self.mul(-self.exp(A_log), sp), self.r(1.0 / self.add(1.0, self.exp(-beta)))


def head_math64(case, state, old_u, old_k, old_G, q, k, v, gate, beta, A_log=0.0, dt_bias=0.0):
    """Independent mathematical oracle: materialize prefix, then recurrent GDN.

    No coefficient rounding, checkpoint splitting, or triangular solver is used.
    All inputs are Python FP64 values from source-quantized tensors.
    """
    a = Arithmetic(case, False)
    p, T, V, K = len(old_G), len(q), len(state), len(state[0])
    last = old_G[-1] if p else 0.0
    s = [[state[i][j] * math.exp(last) + math.fsum(
        old_u[h][i] * old_k[h][j] * math.exp(last - old_G[h]) for h in range(p))
          for j in range(K)] for i in range(V)]
    outputs, updates, keys, totals, snapshots = [], [], [], [], []
    G = last
    scale = case.scale if case.scale is not None else K ** -0.5
    for t in range(T):
        qt, kt = a.normalized(q[t], scale), a.normalized(k[t])
        gt, bt = a.gating(gate[t], beta[t], A_log, dt_bias)
        G += gt
        decayed = [[x * math.exp(gt) for x in row] for row in s]
        u = [bt * (v[t][i] - math.fsum(x * y for x, y in zip(decayed[i], kt))) for i in range(V)]
        s = [[decayed[i][j] + u[i] * kt[j] for j in range(K)] for i in range(V)]
        outputs.append([math.fsum(x * y for x, y in zip(row, qt)) for row in s])
        updates.append(u)
        keys.append(kt)
        totals.append(G)
        snapshots.append(s)
    return dict(out=outputs, u=updates, k=keys, G=totals, states=snapshots)


def head_typed(case, state, old_u, old_k, old_G, q, k, v, gate, beta, A_log=0.0, dt_bias=0.0):
    """Source-operation reference; no PLA or upstream kernel is an oracle.

    State FP32 -> input-dtype hi+lo before projections; history coefficients ->
    U dtype before history dot; forward-solved new U stays FP32 until storage.
    """
    a = Arithmetic(case, True)
    p, T, V, K = len(old_G), len(q), len(state), len(state[0])
    scale = case.scale if case.scale is not None else K ** -0.5
    kn = [a.normalized(row) for row in k]
    qn = [a.normalized(row, scale) for row in q]
    gated = [a.gating(gate[t], beta[t], A_log, dt_bias) for t in range(T)]
    local, total, running = [], [], 0.0
    for gt, _ in gated:
        running = a.add(running, gt)
        local.append(running)
        total.append(a.add(old_G[-1] if p else 0.0, running))
    hi = [[a.store(x) for x in row] for row in state]
    lo = [[a.store(a.sub(x, h)) for x, h in zip(row, high)] for row, high in zip(state, hi)]

    def project(row, i):
        high = a.dot(row, hi[i])
        return a.add(high, a.dot(row, lo[i])) if case.state_dtype == "float32" else high

    def history_coefficients(rows):
        return [[a.store(a.mul(a.dot(row, old_k[h]), a.exp(a.sub(total[t], old_G[h]))))
                 for h in range(p)] for t, row in enumerate(rows)]

    # Coefficients are shared across V; do not recompute K dot products per V.
    k_history = history_coefficients(kn)
    q_history = history_coefficients(qn)
    old_u_columns = [[old_u[h][i] for h in range(p)] for i in range(V)]

    def history(coeff, i):
        return a.dot(coeff, old_u_columns[i])

    unrounded, outputs = [], []
    for t in range(T):
        bt = gated[t][1]
        rhs = [a.mul(bt, a.sub(a.sub(v[t][i], a.mul(a.exp(total[t]), project(kn[t], i))),
                                      history(k_history[t], i))) for i in range(V)]
        lower = [a.mul(a.mul(bt, a.dot(kn[t], kn[j])), a.exp(a.sub(total[t], total[j])))
                 for j in range(t)]
        unrounded.append([a.sub(rhs[i], a.dot(lower, [unrounded[j][i] for j in range(t)], False))
                          for i in range(V)])
        causal = [a.mul(a.dot(qn[t], kn[j]), a.exp(a.sub(total[t], total[j]))) for j in range(t + 1)]
        outputs.append([a.store(a.add(a.add(a.mul(a.exp(total[t]), project(qn[t], i)),
                                                history(q_history[t], i)),
                                          a.dot(causal, [unrounded[j][i] for j in range(t + 1)], False)))
                        for i in range(V)])
    return dict(out=outputs, u=[[a.store(x) for x in row] for row in unrounded],
                unrounded_u=unrounded, k=kn, G=total, local_G=local)


def fold_head(case, state, u, k, G, *, typed=True):
    """CachedUKG Materialize: scaled U is rounded into hi AND lo separately.

    Public Materialize multiplies FP32 keys with these two parts; it does not
    apply the verify dot_precision operand transform.
    """
    if not G:
        return [row[:] for row in state]
    a = Arithmetic(case, typed)
    scaled = [[a.mul(x, a.exp(a.sub(G[-1], G[j]))) for x in row] for j, row in enumerate(u)]
    hi = [[a.store(x) for x in row] for row in scaled]
    lo = [[a.store(a.sub(x, h)) for x, h in zip(row, high)] for row, high in zip(scaled, hi)]
    result = []
    for i, row in enumerate(state):
        result.append([])
        for channel, value in enumerate(row):
            delta = a.sum(a.mul(k[j][channel], hi[j][i]) + a.mul(k[j][channel], lo[j][i])
                          for j in range(len(G))) if typed else math.fsum(
                              k[j][channel] * scaled[j][i] for j in range(len(G)))
            value = a.add(a.mul(value, a.exp(G[-1])), delta)
            result[-1].append(quantize(value, case.state_dtype) if typed else value)
    return result


def torch_module():
    return importlib.import_module("torch")


def _layout_copy(tensor, device, padded=False):
    """One-element unaligned offset; optional padding only between outer rows.

    Required packed head/channel tails remain contiguous. Retain the full backing
    allocation on the view for guard verification; no untracked empty_strided gaps.
    """
    t = torch_module()
    if not padded:
        return tensor.to(device=device).clone()
    width = math.prod(tensor.shape[1:]) if tensor.ndim else 1
    rows = tensor.shape[0] if tensor.ndim else 1
    stride = (width + 3, *tensor.contiguous().stride()[1:])
    backing = t.full((1 + rows * (width + 3) + 1,), 19, dtype=tensor.dtype, device=device)
    view = backing.as_strided(tensor.shape, stride, storage_offset=1)
    view.copy_(tensor)
    view._cached_test_backing = backing
    return view


def make_cached_case(case=CachedCase(), *, counts=None, buffers=None, lengths=None,
                     shared_mapping=False, null_rows=(), null_replays=(), strided=False,
                     explicit_cu=False, seed=107, device="cpu", integer="int64"):
    """Return a dict of source-quantized tensors plus CachedCase configuration.

    counts/buffers/lengths arguments are per work row; stored counts and buffers
    are replay-SLOT indexed. Active writable replay slots are always unique.
    Null history metadata is deliberately not dereferenceable through row=-1.
    """
    t = torch_module()
    c = case.validate()
    rng = random.Random(seed)
    slots = c.B + 3
    lengths = [c.T] * c.B if lengths is None else list(lengths)
    counts = [min(3, c.L)] * c.B if counts is None else list(counts)
    buffers = [row % 2 for row in range(c.B)] if buffers is None else list(buffers)
    if not all(len(x) == c.B for x in (counts, buffers, lengths)):
        raise ValueError("fixture row metadata length mismatch")
    if any(not 0 <= n <= c.T for n in lengths) or any(not 0 <= p <= c.L for p in counts):
        raise ValueError("fixture counts/lengths out of range")
    if any(b not in (0, 1) for b in buffers):
        raise ValueError("fixture buffer out of range")
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if not explicit_cu and any(n != c.T for n in lengths):
        raise ValueError("ragged fixtures require explicit_cu=True")
    dtype, state_dtype, index_dtype = (getattr(t, name) for name in (c.dtype, c.state_dtype, integer))

    def rand(shape, std=0.2, dt=dtype, mean=0.0):
        vals = [rng.gauss(mean, std) for _ in range(math.prod(shape))]
        return t.tensor(vals, dtype=t.float64).reshape(shape).to(dt)

    sources = list(range(1, c.B + 1))
    replays = sources[:] if shared_mapping else list(range(2, c.B + 2))[::-1]
    state = rand((slots, c.HV, c.V, c.K), 0.2, state_dtype)
    u = t.full((slots, 2, c.L, c.HV, c.V), 7.0, dtype=dtype)
    hk = t.full((slots, 2, c.L, c.H, c.K), -3.0, dtype=dtype)
    G = t.full((slots, 2, c.HV, c.L), 11.0, dtype=t.float32)
    slot_counts = t.zeros(slots, dtype=index_dtype)
    slot_buffers = t.zeros(slots, dtype=index_dtype)
    for r, p, b in zip(replays, counts, buffers):
        slot_counts[r], slot_buffers[r] = p, b
        u[r, b, :p] = rand((p, c.HV, c.V), 0.12)
        hk[r, b, :p] = rand((p, c.H, c.K), c.K ** -0.5)
        G[r, b, :, :p] = -rand((c.HV, p), 0.025, t.float32, 0.08).abs().cumsum(-1)
    for row in null_rows:
        sources[row] = -1
    for row in null_replays:
        replays[row] = -1
    n = cu[-1]
    raw = c.gate_mode == "raw"
    data = dict(config=c, q=rand((n, c.H, c.K), c.K ** -0.5),
                k=rand((n, c.H, c.K), c.K ** -0.5), v=rand((n, c.HV, c.V), 0.4),
                gate=rand((n, c.HV), 0.3, dtype, -1.0) if raw else -rand((n, c.HV), 0.03, t.float32, 0.1).abs(),
                beta=rand((n, c.HV), 0.4, dtype) if raw else rand((n, c.HV), 0.04, t.float32, 0.4),
                A_log=rand((c.HV,), 0.1, t.float32, -1.0) if raw else None,
                dt_bias=rand((c.HV,), 0.1, t.float32) if raw else None,
                state=state, u=u, hk=hk, G=G,
                out=t.full((n, c.HV, c.V), 17.0, dtype=dtype),
                sources=t.tensor(sources, dtype=index_dtype), replays=t.tensor(replays, dtype=index_dtype),
                counts=slot_counts, buffers=slot_buffers, starts=None,
                cu=t.tensor(cu, dtype=index_dtype) if explicit_cu else None)
    return copy_case(data, device=device, strided=strided)


def copy_case(data, *, device="cpu", strided=False):
    """Clone without losing optional values; strided=True adds offset/padding guards."""
    t = torch_module()
    return {key: _layout_copy(value.detach().cpu(), device, strided)
            if isinstance(value, t.Tensor) else value for key, value in data.items()}


def public_bundle(api, data):
    c, d = data["config"], data
    return dict(inputs=api.ReplayInputs(d["q"], d["k"], d["v"], d["gate"], d["beta"],
                                        d["A_log"], d["dt_bias"], c.scale),
                state=d["state"], history=api.ReplayHistory(api.HistoryFormat.CACHED_UKG_DOUBLE,
                                                          d["u"], d["hk"], d["G"]),
                mapping=api.ReplayMapping(d["sources"], d["replays"]),
                step=api.ReplayStep(d["counts"], d["starts"], d["buffers"]), out=d["out"], workspace=None,
                numeric_policy=api.NumericPolicy(c.norm, c.epsilon, c.gate_mode, c.dot_precision))


def adapter_kwargs(data, *, items=None, n_items=None, packed=False):
    """TRT-LLM uses a shared state/history slot and fixed [B,T,...] ranks."""
    t = torch_module()
    c, d = data["config"], data
    if d["cu"] is not None or c.norm == "l2_inside_sqrt":
        raise ValueError("adapter fixture requires fixed T and none/outside normalization")
    kw = dict(q=d["q"].reshape(c.B, c.T, c.H, c.K), k=d["k"].reshape(c.B, c.T, c.H, c.K),
              v=d["v"].reshape(c.B, c.T, c.HV, c.V), g=d["gate"].reshape(c.B, c.T, c.HV),
              beta=d["beta"].reshape(c.B, c.T, c.HV), ssm_states=d["state"], state_indices=d["sources"],
              old_u=d["u"], old_k=d["hk"], old_G=d["G"], old_beta=None,
              cache_buf_idx=d["buffers"], prev_num_accepted_tokens=d["counts"], history_size=c.L,
              scale=c.scale, use_qk_l2norm_in_kernel=c.norm != "none", A_log=d["A_log"],
              dt_bias=d["dt_bias"], replay_work_items=items, n_writes=n_items,
              use_all_layer_commit=items is not None, output=d["out"].reshape(c.B, c.T, c.HV, c.V))
    if packed:
        kw["packed_qkv"] = t.cat([d["q"].reshape(c.B * c.T, -1), d["k"].reshape(c.B * c.T, -1),
                                     d["v"].reshape(c.B * c.T, -1)], dim=-1).contiguous()
    return kw


def _rows(data):
    c = data["config"]
    cu = data["cu"].tolist() if data["cu"] is not None else [row * c.T for row in range(c.B + 1)]
    for row, (s, r) in enumerate(zip(data["sources"].tolist(), data["replays"].tolist())):
        begin, end = cu[row:row + 2]
        if s < 0 or r < 0 or begin == end:
            continue
        p, b = int(data["counts"][r]), int(data["buffers"][r])
        yield row, s, r, begin, end, p, b


def reference_cached(data, *, typed=True):
    """Return expected CPU out/U/K/G and structural write masks; never fold state."""
    t = torch_module()
    d, c = copy_case(data), data["config"]
    expected = {key: d[key].double().clone() for key in ("out", "u", "hk", "G")}
    masks = {key: t.zeros_like(d[key], dtype=t.bool) for key in expected}
    head_ref = head_typed if typed else head_math64
    for _, s, r, begin, end, p, b in _rows(d):
        overflow = p + end - begin > c.L
        wb, offset = (1 - b, 0) if overflow else (b, p)
        for hv in range(c.HV):
            h = hv // (c.HV // c.H)
            result = head_ref(c, d["state"][s, hv].tolist(), d["u"][r, b, :p, hv].tolist(),
                              d["hk"][r, b, :p, h].tolist(), d["G"][r, b, hv, :p].tolist(),
                              d["q"][begin:end, h].tolist(), d["k"][begin:end, h].tolist(),
                              d["v"][begin:end, hv].tolist(), d["gate"][begin:end, hv].tolist(),
                              d["beta"][begin:end, hv].tolist(),
                              float(d["A_log"][hv]) if d["A_log"] is not None else 0.0,
                              float(d["dt_bias"][hv]) if d["dt_bias"] is not None else 0.0)
            sl = slice(offset, offset + end - begin)
            expected["out"][begin:end, hv] = t.tensor(result["out"], dtype=t.float64)
            expected["u"][r, wb, sl, hv] = t.tensor(result["u"], dtype=t.float64)
            expected["hk"][r, wb, sl, h] = t.tensor(result["k"], dtype=t.float64)
            gs = result.get("local_G") if overflow and typed else result["G"]
            if overflow and not typed:
                gs = [g - (float(d["G"][r, b, hv, p - 1]) if p else 0.0) for g in gs]
            expected["G"][r, wb, hv, sl] = t.tensor(gs, dtype=t.float64)
        masks["out"][begin:end] = True
        masks["u"][r, wb, offset:offset + end - begin] = True
        masks["hk"][r, wb, offset:offset + end - begin] = True
        masks["G"][r, wb, :, offset:offset + end - begin] = True
    return expected, masks


def write_masks(data):
    """Structural masks, independent of numerical equality or canary values."""
    t = torch_module()
    d, c = copy_case(data), data["config"]
    result = {key: t.zeros_like(d[key], dtype=t.bool) for key in ("out", "u", "hk", "G")}
    for _, _, r, begin, end, p, b in _rows(d):
        wb, off = (1 - b, 0) if p + end - begin > c.L else (b, p)
        result["out"][begin:end] = True
        result["u"][r, wb, off:off + end - begin] = True
        result["hk"][r, wb, off:off + end - begin] = True
        result["G"][r, wb, :, off:off + end - begin] = True
    return result


def reference_fold(data, work, *, typed=True):
    """CPU checkpoint after explicit (source,replay,count,buffer) OLD work items."""
    t = torch_module()
    d, c = copy_case(data), data["config"]
    state = d["state"].double().clone()
    for source, replay, count, buffer in work:
        if source < 0 or count == 0:
            continue
        for hv in range(c.HV):
            h = hv // (c.HV // c.H)
            value = fold_head(c, d["state"][source, hv].tolist(), d["u"][replay, buffer, :count, hv].tolist(),
                              d["hk"][replay, buffer, :count, h].tolist(),
                              d["G"][replay, buffer, hv, :count].tolist(), typed=typed)
            state[source, hv] = t.tensor(value, dtype=t.float64)
    return state


def source_function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def signature(node):
    a = node.args
    return (tuple(x.arg for x in a.posonlyargs + a.args), tuple(x.arg for x in a.kwonlyargs),
            tuple(ast.literal_eval(x) for x in a.defaults),
            tuple(ast.literal_eval(x) if x is not None else inspect.Parameter.empty for x in a.kw_defaults))


class StaticTests(unittest.TestCase):
    def test_ast_and_lazy_imports(self):
        path = Path(__file__).resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        compile(tree, str(path), "exec")
        top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("torch" in ast.dump(n) or "triton" in ast.dump(n) for n in top_imports))
        self.assertNotIn("torch", sys.modules)

    def test_exact_adapter_signatures(self):
        update_names = (
            "q k v g beta ssm_states state_indices old_u old_k old_G old_beta cache_buf_idx "
            "prev_num_accepted_tokens history_size scale use_qk_l2norm_in_kernel A_log dt_bias "
            "launch_with_pdl replay_work_items n_writes block_v num_warps use_l2_state_cache "
            "use_l2_streaming_inputs use_l2_shared_inputs main_num_stages packed_qkv use_all_layer_commit output"
        )
        self.assertEqual(signature(source_function(ADAPTER, UPDATE)), (
            tuple(update_names.split()), (), (None, False, None, None, False, *(None,) * 9, False, None), (),
        ))
        commit_names = (
            "ssm_states ssm_state_descriptors ssm_state_num_layers ssm_state_layer_stride ssm_state_slot_stride "
            "old_u old_k old_G replay_work_items n_writes history_size persistent_waves commit_block_v "
            "commit_num_warps commit_pipeline_stages"
        )
        required = inspect.Parameter.empty
        self.assertEqual(signature(source_function(ADAPTER, COMMIT)), (
            (), tuple(commit_names.split()), (), (required, *(None,) * 4, *(required,) * 6, 2, None, None, None),
        ))

    def test_public_step_and_verify_signature(self):
        node = source_function(PACKAGE / "contracts.py", "ReplayStep")
        fields = [n for n in node.body if isinstance(n, ast.AnnAssign)]
        self.assertEqual([n.target.id for n in fields], ["lengths", "starts", "buffers"])
        self.assertIsNone(fields[0].value)
        self.assertTrue(all(ast.literal_eval(n.value) is None for n in fields[1:]))
        args = source_function(PACKAGE / "__init__.py", "gdn_replay_verify").args
        self.assertEqual([n.arg for n in args.args], ["inputs", "state", "history", "mapping", "step", "out", "workspace"])
        self.assertEqual([n.arg for n in args.kwonlyargs], ["query_start_loc", "max_verify_len", "numeric_policy"])

    def test_budgets_match_existing_reference_without_importing_it(self):
        tree = ast.parse((ROOT / "scripts/decode/gdn_replayssm/reference.py").read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(x, ast.Name) and x.id == "TOLERANCES" for x in n.targets))
        self.assertEqual(BUDGETS, ast.literal_eval(assignment.value))

    def test_storage_rounding_and_tf32_rtz(self):
        self.assertEqual(quantize(1 + 2 ** -8, "bfloat16"), 1.0)
        self.assertEqual(quantize(1 + 3 * 2 ** -8, "bfloat16"), 1 + 2 ** -6)
        self.assertEqual(quantize(1 + 2 ** -11, "float16"), 1.0)
        for sign in (1, -1):
            self.assertEqual(tf32_rtz(sign * (1 + 3 * 2 ** -12)), float(sign))
            self.assertEqual(tf32_rtz(sign * float("inf")), sign * float("inf"))
        self.assertTrue(math.isnan(tf32_rtz(float("nan"))))
        self.assertEqual(math.copysign(1, tf32_rtz(-0.0)), -1)

    def test_norm_and_raw_gate_hand_values(self):
        for norm, expected in (("none", 1e-4), ("l2_inside_sqrt", 0.09950371902099892),
                               ("l2_outside_sqrt", 0.9900990099009901)):
            a = Arithmetic(CachedCase(norm=norm), False)
            self.assertAlmostEqual(a.normalized([1e-4])[0], expected, places=14)
        a = Arithmetic(CachedCase(gate_mode="raw"), False)
        g, b = a.gating(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(g, -math.log(2.0), places=15)
        self.assertEqual(b, 0.5)
        self.assertEqual(a.gating(21.0, 0.0, 0.0, 0.0)[0], -21.0)

    def test_hand_recurrence_and_fold(self):
        c = CachedCase(B=1, H=1, HV=1, K=1, V=1, T=1, L=1, dtype="float32", norm="none", dot_precision="ieee", scale=1.0)
        args = (c, [[2.0]], [], [], [], [[1.0]], [[1.0]], [[3.0]], [0.0], [0.5])
        for ref in (head_math64, head_typed):
            result = ref(*args)
            self.assertEqual(result["out"], [[2.5]])
            self.assertEqual(result["u"], [[0.5]])
        self.assertEqual(fold_head(c, [[2.0]], [[0.5]], [[1.0]], [0.0]), [[2.5]])

    def test_hi_lo_state_and_fold_are_not_single_cast(self):
        c = CachedCase(B=1, H=1, HV=1, K=1, V=1, T=1, L=1, norm="none", scale=1.0)
        x = f32(1 + 2 ** -9)
        result = head_typed(c, [[x]], [], [], [], [[1.0]], [[1.0]], [[0.0]], [0.0], [0.5])
        self.assertEqual(result["unrounded_u"], [[f32(-0.5 * x)]])
        self.assertNotEqual(result["unrounded_u"][0][0], -0.5 * quantize(x, c.dtype))
        folded = fold_head(c, [[0.0]], [[1.0]], [[1.0]], [math.log(x)])
        self.assertEqual(folded, [[1.0]])  # Last cumulative gate is not a per-token U weight.
        folded = fold_head(c, [[0.0]], [[1.0], [0.0]], [[1.0], [1.0]], [0.0, math.log(x)])
        self.assertEqual(folded, [[x]])

    def test_history_coefficient_rounding_and_unrounded_new_U(self):
        c = CachedCase(B=1, H=1, HV=1, K=1, V=1, T=1, L=2, norm="none", scale=1.0)
        result = head_typed(c, [[0.0]], [[1.0]], [[1.0]], [0.0],
                            [[1.0]], [[1.0]], [[0.0]], [math.log(0.999)], [0.5])
        # exp(G-Ghist) rounds to one in BF16 BEFORE coefficient * cached U.
        self.assertEqual(result["unrounded_u"], [[-0.5]])
        self.assertNotEqual(result["unrounded_u"][0][0], f32(-0.5 * f32(0.999)))
        c = replace(c, T=2)
        result = head_typed(c, [[0.0]], [], [], [], [[1.0], [1.0]],
                            [[1.0], [1.0]], [[1.0], [0.0]], [0.0, 0.0], [0.3, 0.3])
        self.assertEqual(result["unrounded_u"][1][0], f32(-f32(0.3) * f32(0.3)))
        self.assertNotEqual(result["unrounded_u"][1][0], f32(-f32(0.3) * result["u"][0][0]))

    def test_typed_and_mathematical_reference_on_small_random_inputs(self):
        rng = random.Random(19)
        for dtype in BUDGETS:
            c = CachedCase(H=1, HV=1, K=3, V=2, T=4, dtype=dtype, dot_precision="ieee", scale=0.5)
            values = lambda rows, width: [[quantize(rng.uniform(-0.3, 0.3), dtype) for _ in range(width)] for _ in range(rows)]
            args = (c, values(2, 3), values(2, 2), values(2, 3), [-0.1, -0.2],
                    values(4, 3), values(4, 3), values(4, 2), [-0.1] * 4, [0.4] * 4)
            math_result, typed_result = head_math64(*args), head_typed(*args)
            atol, rtol = BUDGETS[dtype]["output"]
            for actual, expected in zip(itertools.chain.from_iterable(typed_result["out"]),
                                        itertools.chain.from_iterable(math_result["out"])):
                self.assertLessEqual(abs(actual - expected), atol + rtol * abs(expected))

    def test_capacity_boundary_and_zero_acceptance_equations(self):
        for T, L in ((1, 1), (4, 11), (8, 15), (8, 16)):
            CachedCase(T=T, L=L).validate()
            p = L - T
            self.assertFalse(p + T > L)
            self.assertTrue(p + T + 1 > L)
            for accepted in range(T + 1):
                new_count = accepted if p + 1 + T > L else p + 1 + accepted
                self.assertEqual(new_count, accepted)
        with self.assertRaises(ValueError):
            CachedCase(T=8, L=7).validate()


class DeviceBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OPTIONS.device:
            raise RuntimeError("NOT_RUN: invoke this file with explicit --device cuda:N")
        cls.t = torch_module()
        cls.device = cls.t.device(OPTIONS.device)
        cls.t.cuda.set_device(cls.device)
        cls.api = importlib.import_module("pla.decode.gdn_replayssm")
        cls.adapter = importlib.import_module("pla.decode.gdn_replayssm_hlm")

    def cpu(self, value):
        return value.detach().cpu().clone()

    def bits(self, actual, expected, message="unexpected write"):
        aa, bb = self.cpu(actual).contiguous(), self.cpu(expected).contiguous()
        self.assertEqual((aa.shape, aa.dtype), (bb.shape, bb.dtype), message)
        self.assertTrue(self.t.equal(aa.reshape(-1).view(self.t.uint8), bb.reshape(-1).view(self.t.uint8)), message)

    def close(self, actual, expected, dtype, kind="output"):
        atol, rtol = BUDGETS[dtype][kind]
        self.t.testing.assert_close(self.cpu(actual).double(), self.cpu(expected).double(), atol=atol, rtol=rtol)

    def launch(self, data, *, decode=False, bundle=None):
        kw = public_bundle(self.api, data) if bundle is None else bundle
        if decode:
            return self.api.gdn_replay_decode(**kw)
        return self.api.gdn_replay_verify(**kw, query_start_loc=data["cu"], max_verify_len=data["config"].T)

    def read_only(self, data, before, *, state=True):
        names = ("q", "k", "v", "gate", "beta", "A_log", "dt_bias", "sources", "replays", "counts", "buffers", "starts", "cu")
        for key in (*names, *(("state",) if state else ())):
            if data[key] is not None:
                self.bits(data[key], before[key], key + " must remain read-only")

    def check_writes(self, data, before, expected, masks):
        for key in ("out", "u", "hk", "G"):
            actual = self.cpu(data[key])
            self.bits(actual[~masks[key]], before[key][~masks[key]], key + " outside candidate write footprint")
            if masks[key].any():
                if key == "G":
                    self.t.testing.assert_close(actual[masks[key]].double(), expected[key][masks[key]], atol=1e-6, rtol=1e-6)
                else:
                    self.close(actual[masks[key]], expected[key][masks[key]], data["config"].dtype, "output")

    def checked_public(self, data, *, decode=False, math_check=True, bundle=None):
        before = copy_case(data)
        expected, masks = reference_cached(before)
        for key, mask in write_masks(before).items():
            self.assertTrue(self.t.equal(mask, masks[key]))
        guard_before = {key: self.cpu(value._cached_test_backing)
                        for key, value in data.items() if hasattr(value, "_cached_test_backing")}
        result = self.launch(data, decode=decode, bundle=bundle)
        self.assertEqual(result.data_ptr(), data["out"].data_ptr())
        self.t.cuda.synchronize(self.device)
        self.read_only(data, before)
        self.check_writes(data, before, expected, masks)
        if math_check:
            math_expected, _ = reference_cached(before, typed=False)
            config = data["config"]
            if config.dtype == "float32" and config.dot_precision == "tf32":
                actual = self.cpu(data["out"]).double()[masks["out"]]
                mathematical = math_expected["out"][masks["out"]]
                self.assertTrue(self.t.isfinite(actual).all(), "nonfinite TF32 output")
                self.assertTrue(self.t.isfinite(mathematical).all(), "nonfinite FP64 diagnostic reference")
                difference = (actual - mathematical).abs()
                atol, rtol = BUDGETS[config.dtype]["output"]
                print(json.dumps(dict(type="tf32_mathematical_diagnostic", case=vars(config),
                                      elements=actual.numel(),
                                      max_absolute=float(difference.max()) if actual.numel() else 0.0,
                                      relative_l2=float(difference.norm() / mathematical.norm().clamp_min(1e-20)),
                                      outside_ieee_budget=int((difference > atol + rtol * mathematical.abs()).sum()),
                                      acceptance="typed/source reference; unquantized error is diagnostic")), flush=True)
            else:
                self.close(data["out"], math_expected["out"], config.dtype)
        # Check padding/allocation boundaries independently of all logical masks.
        for key, saved in guard_before.items():
            value = data[key]
            touched = self.t.zeros_like(saved, dtype=self.t.bool)
            if key in masks:
                view = touched.as_strided(value.shape, value.stride(), value.storage_offset())
                view.copy_(masks[key])
            self.bits(self.cpu(value._cached_test_backing)[~touched], saved[~touched], key + " padding guard")
        return before, expected

    def fold_public(self, data, work):
        if not work:
            return
        t, api = self.t, self.api
        cols = list(zip(*work))
        ints = lambda xs: t.tensor(xs, dtype=t.int64, device=self.device)
        mw = api.MaterializeWork(state_indices=ints(cols[0]), replay_indices=ints(cols[1]),
                                 lengths=ints(cols[2]), buffers=ints(cols[3]))
        h = api.ReplayHistory(api.HistoryFormat.CACHED_UKG_DOUBLE, data["u"].unsqueeze(0),
                              data["hk"].unsqueeze(0), data["G"].unsqueeze(0))
        api.gdn_replay_materialize(data["state"].unsqueeze(0), h, mw)

    def check_fold(self, data, before, work):
        dtype = data["config"].dtype
        self.close(data["state"], reference_fold(before, work), dtype, "state")
        self.close(data["state"], reference_fold(before, work, typed=False), dtype, "state")
        untouched = [slot for slot in range(data["state"].shape[0]) if slot not in [w[0] for w in work]]
        self.bits(data["state"][untouched], before["state"][untouched], "fold wrote an unlisted checkpoint")

    def overflow_work(self, before):
        return [(s, r, p, b) for _, s, r, start, end, p, b in _rows(before)
                if p + end - start > before["config"].L]

    def adapter_commit(self, data, items, n_items, **kwargs):
        getattr(self.adapter, COMMIT)(ssm_states=data["state"].unsqueeze(0), old_u=data["u"].unsqueeze(0),
                                     old_k=data["hk"].unsqueeze(0), old_G=data["G"].unsqueeze(0),
                                     replay_work_items=items, n_writes=n_items,
                                     history_size=data["config"].L, **kwargs)


class AccuracyTests(DeviceBase):
    def test_dtype_state_gates_norm_matrix(self):
        for dtype in BUDGETS:
            for state_dtype in dict.fromkeys((dtype, "float32")):
                for raw, norm in itertools.product((False, True), ("none", "l2_inside_sqrt", "l2_outside_sqrt")):
                    c = CachedCase(dtype=dtype, state_dtype=state_dtype, gate_mode="raw" if raw else "processed", norm=norm)
                    with self.subTest(dtype=dtype, state=state_dtype, gate=c.gate_mode, norm=norm):
                        d = make_cached_case(c, counts=[0, c.L - c.T, c.L], device=self.device)
                        self.checked_public(d)

    def test_fused_gate_preserves_source_addition_rounding(self):
        c = CachedCase(B=1, H=1, HV=1, K=1, V=1, T=1, L=1, dtype="float32",
                       gate_mode="raw", norm="none", scale=1.0, dot_precision="ieee")
        d = make_cached_case(c, counts=[0], device=self.device)
        d["gate"].fill_(-20)
        d["A_log"].fill_(20)
        d["dt_bias"].zero_()
        self.checked_public(d, math_check=False)
        self.assertEqual(float(d["G"][int(d["replays"][0]), 0, 0, 0]), 0.0)

    def test_T_capacity_both_banks_and_key_dimension_edges(self):
        for T, L, K, V in ((1, 1, 1, 1), (1, 7, 31, 17), (4, 11, 64, 16),
                           (8, 15, 128, 32), (8, 16, 256, 16)):
            for bank in (0, 1):
                c = CachedCase(T=T, L=L, K=K, V=V)
                with self.subTest(T=T, L=L, K=K, bank=bank):
                    d = make_cached_case(c, counts=[0, L - T, min(L, L - T + 1)],
                                         buffers=[bank] * 3, device=self.device)
                    self.checked_public(d)

    def test_fixed_128_cached_projection_policies(self):
        for T, state_dtype, gate, norm in itertools.product((1, 4, 8), ("bfloat16", "float32"),
                                                           ("raw", "processed"), ("none", "l2_outside_sqrt")):
            c = CachedCase(B=1, H=1, HV=1, K=128, V=128, T=T, L=16, state_dtype=state_dtype,
                           gate_mode=gate, norm=norm, scale=0.125)
            with self.subTest(T=T, state=state_dtype, gate=gate, norm=norm):
                d = make_cached_case(c, counts=[16], integer="int32", device=self.device)
                self.checked_public(d)

    def test_fixed_128_cached_indices_and_padding(self):
        for T, integer, state_dtype in itertools.product((1, 4), ("int32", "int64"), ("bfloat16", "float32")):
            c = CachedCase(B=2, H=1, HV=2, K=128, V=128, T=T, L=16, state_dtype=state_dtype)
            d = make_cached_case(c, counts=[16-T, 17-T], integer=integer, strided=True, device=self.device)
            d["starts"] = _layout_copy(self.t.zeros_like(d["counts"], device="cpu"), self.device, True)
            self.checked_public(d)

    def test_fixed_128_cached_grouped_heads(self):
        for ratio, state_dtype in itertools.product((1, 8), ("bfloat16", "float32")):
            c = CachedCase(B=1, H=2, HV=2*ratio, K=128, V=128, T=4, L=16, state_dtype=state_dtype)
            d = make_cached_case(c, counts=[13], integer="int32", device=self.device)
            self.checked_public(d)

    def test_fixed_128_ragged_uses_actual_lengths(self):
        c = CachedCase(B=3, H=1, HV=2, K=128, V=128, T=8, L=16)
        d = make_cached_case(c, counts=[15, 12, 16], lengths=[1, 4, 0], explicit_cu=True, device=self.device)
        self.checked_public(d)

    def test_small_norms_distinguish_inside_and_outside_sqrt(self):
        for norm in ("none", "l2_inside_sqrt", "l2_outside_sqrt"):
            c = CachedCase(H=1, HV=1, K=1, V=1, T=1, L=1, dtype="float32", norm=norm,
                           scale=1.0, dot_precision="ieee")
            d = make_cached_case(c, counts=[0, 0, 0], device=self.device)
            d["q"].fill_(1e-4)
            d["k"].fill_(1e-4)
            d["state"].fill_(0.25)
            d["v"].fill_(0.5)
            self.checked_public(d)

    def test_ieee_and_tf32_operand_policies(self):
        for policy in ("ieee", "tf32"):
            c = CachedCase(dtype="float32", dot_precision=policy, norm="none", scale=0.75)
            with self.subTest(policy=policy):
                self.checked_public(make_cached_case(c, counts=[0, 7, 9], device=self.device))

    def test_tf32_compatibility_and_ieee_accuracy_are_distinct(self):
        x = f32(1.0009)
        results = {}
        for policy in ("ieee", "tf32"):
            c = CachedCase(B=1, H=1, HV=1, K=128, V=16, T=1, L=16, dtype="float32",
                           dot_precision=policy, norm="none", scale=1.0)
            d = make_cached_case(c, counts=[0], device=self.device)
            d["q"].zero_()
            d["q"][0, 0, :2] = self.t.tensor([x, 1.0], device=self.device)
            for key in ("k", "v", "gate", "beta", "state"):
                d[key].zero_()
            slot = int(d["sources"][0])
            d["state"][slot, 0, :, 0] = x
            d["state"][slot, 0, :, 1] = -1.0
            self.checked_public(d)
            results[policy] = float(d["out"][0, 0, 0])
        self.assertEqual(results["tf32"], 0.0)
        self.assertAlmostEqual(results["ieee"], x * x - 1.0, delta=1e-7)
        atol, rtol = BUDGETS["float32"]["output"]
        self.assertGreater(x * x - 1.0, 2 * atol + rtol * (x * x - 1.0))

    def test_ragged_actual_n_independent_mapping_null_and_empty(self):
        c = CachedCase(B=6, T=8, L=11)
        d = make_cached_case(c, counts=[10, 7, 11, 3, 9, 4], lengths=[1, 4, 0, 8, 8, 0],
                             null_rows=(3, 4), null_replays=(4,), explicit_cu=True, device=self.device)
        self.checked_public(d)
        # Row0 appends exactly at L; using max T instead of actual n flips bank.
        self.assertEqual(int(d["counts"][int(d["replays"][0])]), 10)

    def test_unaligned_strided_inputs_outputs_and_metadata(self):
        for integer in ("int32", "int64"):
            d = make_cached_case(CachedCase(), counts=[0, 7, 11], strided=True, integer=integer, device=self.device)
            d["starts"] = _layout_copy(self.t.zeros_like(d["counts"], device="cpu"), self.device, True)
            self.assertEqual(d["q"].storage_offset(), 1)
            self.assertFalse(d["q"].is_contiguous())
            self.checked_public(d)

    def test_decode_is_T1_verify_and_fixed_T_matches_explicit_cu(self):
        for T in (1, 4, 8):
            c = CachedCase(T=T, L=11)
            fixed = make_cached_case(c, counts=[0, 11 - T, 11], device=self.device)
            explicit = copy_case(fixed, device=self.device)
            explicit["cu"] = self.t.arange(c.B + 1, dtype=self.t.int64, device=self.device) * T
            self.checked_public(fixed, decode=T == 1)
            self.checked_public(explicit)
            for key in ("out", "u", "hk", "G", "state"):
                self.bits(fixed[key], explicit[key], key + " fixed/explicit mismatch")

    def test_workspace_query_zero_and_optional_workspace_untouched(self):
        d = make_cached_case(CachedCase(), counts=[0, 7, 11], device=self.device)
        b = public_bundle(self.api, d)
        self.assertEqual(self.api.gdn_replay_workspace_elements(b["inputs"], b["state"], b["history"], b["mapping"], 4), 0)
        workspace = self.t.full((17,), 29.0, dtype=self.t.float32, device=self.device)
        saved = self.cpu(workspace)
        self.checked_public(d, bundle={**b, "workspace": workspace})
        self.bits(workspace, saved)

    def test_equal_heads_and_mixed_gate_storage_types(self):
        for dtype in BUDGETS:
            c = CachedCase(H=3, HV=3, K=17, V=9, dtype=dtype, norm="none")
            d = make_cached_case(c, counts=[0, 7, 11], device=self.device)
            d["gate"] = d["gate"].to(getattr(self.t, dtype))
            d["beta"] = d["beta"].to(getattr(self.t, dtype))
            self.checked_public(d)

    def test_adapter_T1_T8_null_and_ignored_dt_bias(self):
        for T in (1, 8):
            c = CachedCase(T=T, L=15, state_dtype="bfloat16")
            d = make_cached_case(c, shared_mapping=True, counts=[0, 15, 15 - T], null_rows=(2,), device=self.device)
            before = copy_case(d)
            expected, masks = reference_cached(before)
            ignored = self.t.full((c.HV,), float("nan"), device=self.device)
            saved = self.cpu(ignored)
            getattr(self.adapter, UPDATE)(**{**adapter_kwargs(d), "dt_bias": ignored})
            self.check_writes(d, before, expected, masks)
            self.check_fold(d, before, self.overflow_work(before))
            self.read_only(d, before, state=False)
            self.bits(ignored, saved)

    def test_null_and_zero_batch_no_writes(self):
        for c, opts in ((CachedCase(B=0), {}), (CachedCase(B=2), dict(null_rows=(0, 1))),
                        (CachedCase(B=2), dict(lengths=[0, 0], explicit_cu=True))):
            with self.subTest(B=c.B, opts=opts):
                self.checked_public(make_cached_case(c, device=self.device, **opts))

    def test_adapter_raw_processed_normalization_and_packed_qkv(self):
        for dtype, raw, norm, packed in itertools.product(BUDGETS, (False, True), (False, True), (False, True)):
            c = CachedCase(dtype=dtype, gate_mode="raw" if raw else "processed", norm="l2_outside_sqrt" if norm else "none")
            with self.subTest(dtype=dtype, raw=raw, norm=norm, packed=packed):
                d = make_cached_case(c, shared_mapping=True, counts=[0, 7, 11], integer="int32", device=self.device)
                before = copy_case(d)
                expected, masks = reference_cached(before)
                kwargs = adapter_kwargs(d, packed=packed)
                # packed_qkv must be the source, not just an unused accepted argument.
                if packed:
                    kwargs.update(q=self.t.zeros_like(kwargs["q"]), k=self.t.zeros_like(kwargs["k"]), v=self.t.zeros_like(kwargs["v"]))
                old_beta = self.t.full_like(d["beta"], float("nan"))
                kwargs["old_beta"] = old_beta
                result = getattr(self.adapter, UPDATE)(**kwargs)
                self.assertEqual(result.data_ptr(), d["out"].data_ptr())
                self.check_writes(d, before, expected, masks)
                self.read_only(d, before, state=False)
                self.bits(old_beta, self.t.full_like(old_beta, float("nan")), "old_beta is ignored/read-only")
                self.check_fold(d, before, self.overflow_work(before))


class LifecycleTests(DeviceBase):
    def test_tp1_t4_packed_and_separate_mixed_inline_fold(self):
        for raw in (False, True):
            c = CachedCase(B=2, H=16, HV=48, K=128, V=128, T=4, L=16,
                           state_dtype="float32", norm="l2_outside_sqrt",
                           gate_mode="raw" if raw else "processed")
            with self.subTest(gate_mode=c.gate_mode):
                d = make_cached_case(c, counts=[12, 13], buffers=[0, 1],
                                     shared_mapping=True, integer="int32", device=self.device)
                if raw:
                    # The raw gate uses the source's FP32 addition rounding.
                    d["gate"][:, 0].fill_(-20)
                    d["A_log"][0] = 20
                    d["dt_bias"][0] = 0
                before = copy_case(d)
                expected, masks = reference_cached(before)
                update = getattr(self.adapter, UPDATE)
                update(**adapter_kwargs(d, packed=raw))
                self.t.cuda.synchronize(self.device)
                self.check_writes(d, before, expected, masks)
                self.check_fold(d, before, self.overflow_work(before))
                self.read_only(d, before, state=False)

    def test_fixed_128_inline_mixed_overflow_and_rejection(self):
        for state_dtype in ("bfloat16", "float32"):
            c = CachedCase(B=5, H=2, HV=4, K=128, V=128, T=4, L=16, state_dtype=state_dtype)
            d = make_cached_case(c, counts=[0, 12, 13, 16, 16], buffers=[0, 1, 0, 1, 0],
                                 shared_mapping=True, null_rows=(4,), integer="int32", device=self.device)
            for _ in range(2):
                before = copy_case(d)
                expected, masks = reference_cached(before)
                getattr(self.adapter, UPDATE)(**adapter_kwargs(d))
                self.t.cuda.synchronize(self.device)
                self.check_writes(d, before, expected, masks)
                self.check_fold(d, before, self.overflow_work(before))
                self.read_only(d, before, state=False)
                for row, s, r, start, end, count, bank in _rows(before):
                    if count + end - start > c.L:
                        d["counts"][r] = 0
                        d["buffers"][r] = 1 - bank

    def test_fixed_128_graph_device_metadata(self):
        for T, state_dtype in itertools.product((1, 4), ("bfloat16", "float32")):
            c = CachedCase(B=2, H=2, HV=4, K=128, V=128, T=T, L=16, state_dtype=state_dtype)
            d = make_cached_case(c, counts=[16-T, 16], shared_mapping=True, integer="int32", device=self.device)
            function = getattr(self.adapter, UPDATE)
            function(**adapter_kwargs(d))
            self.t.cuda.synchronize(self.device)
            graph = self.t.cuda.CUDAGraph()
            with self.t.cuda.graph(graph):
                function(**adapter_kwargs(d))
            for counts, banks in (([0, 17-T], [0, 1]), ([16, 16-T], [1, 0])):
                reset = make_cached_case(c, counts=counts, buffers=banks, shared_mapping=True,
                                         integer="int32", device=self.device)
                for key, value in reset.items():
                    if isinstance(value, self.t.Tensor):
                        d[key].copy_(value)
                before = copy_case(d)
                expected, masks = reference_cached(before)
                eager = copy_case(d, device=self.device)
                graph.replay()
                function(**adapter_kwargs(eager))
                self.t.cuda.synchronize(self.device)
                self.check_writes(d, before, expected, masks)
                self.check_fold(d, before, self.overflow_work(before))
                self.read_only(d, before, state=False)
                for key in ("out", "u", "hk", "G", "state", "counts", "buffers"):
                    self.bits(d[key], eager[key], key + " captured metadata mismatch")

    def test_deferred_zero_work_still_computes_all_candidates(self):
        c = CachedCase()
        d = make_cached_case(c, shared_mapping=True, counts=[0, 7, 11], integer="int32", device=self.device)
        before = copy_case(d)
        items = self.t.full((c.B, 4), -123, dtype=self.t.int32, device=self.device)
        count = self.t.tensor(0, dtype=self.t.int32, device=self.device)
        saved_items, saved_count = self.cpu(items), self.cpu(count)
        getattr(self.adapter, UPDATE)(**adapter_kwargs(d, items=items, n_items=count))
        expected, masks = reference_cached(before)
        self.check_writes(d, before, expected, masks)
        self.read_only(d, before)
        self.adapter_commit(d, items, count)
        self.bits(d["state"], before["state"], "zero-count commit cannot fold any old prefix")
        self.bits(items, saved_items)
        self.bits(count, saved_count)

    def test_inline_vs_deferred_fold_only_old_prefix(self):
        for dtype in BUDGETS:
            for state_dtype in dict.fromkeys((dtype, "float32")):
                c = CachedCase(dtype=dtype, state_dtype=state_dtype)
                inline = make_cached_case(c, counts=[7, 8, 11], shared_mapping=True, integer="int32", device=self.device)
                deferred = copy_case(inline, device=self.device)
                before = copy_case(inline)
                work = self.overflow_work(before)
                items = self.t.tensor([[931, s, p, b] for s, _, p, b in work] + [[-91, 0, 0, 0]],
                                      dtype=self.t.int32, device=self.device)
                count = self.t.tensor([len(work)], dtype=self.t.int32, device=self.device)
                saved_items, saved_count = self.cpu(items), self.cpu(count)
                getattr(self.adapter, UPDATE)(**adapter_kwargs(inline))
                getattr(self.adapter, UPDATE)(**adapter_kwargs(deferred, items=items, n_items=count))
                self.bits(deferred["state"], before["state"], "deferred compute cannot fold")
                self.read_only(deferred, before)
                self.adapter_commit(deferred, items, count)
                self.bits(items, saved_items)
                self.bits(count, saved_count)
                expected, masks = reference_cached(before)
                for d in (inline, deferred):
                    self.check_writes(d, before, expected, masks)
                    self.check_fold(d, before, work)
                    self.read_only(d, before, state=False)
                for key in ("out", "u", "hk", "G"):
                    self.bits(inline[key], deferred[key], key + " inline/deferred compute")
                # Both paths currently use the existing Materialize. Strict budget
                # permits only its documented reduction-order difference, not drift.
                self.close(inline["state"], deferred["state"], dtype, "state")
                self.bits(inline["state"][1], before["state"][1], "p+T==L must not fold")

    def test_multiround_acceptance_zero_through_T_after_fold(self):
        for dtype, T in itertools.product(BUDGETS, (1, 4, 8)):
            with self.subTest(dtype=dtype, T=T):
                self._acceptance_cycle(dtype, T)

    def _acceptance_cycle(self, dtype, T):
        c = CachedCase(B=2, H=1, HV=2, T=T, L=2 * T - 1, dtype=dtype)
        actual = make_cached_case(c, counts=[c.L, c.L - T], device=self.device)
        model = copy_case(actual)
        for round_no, accept in enumerate(list(range(c.T + 1)) * 3):
            # Counts include all previously consumed tokens. This is prefix
            # acceptance, NOT a claim about full-model bonus-token semantics.
            old = copy_case(model)
            expected, masks = reference_cached(old)
            work = self.overflow_work(old)
            folded = reference_fold(old, work)
            before_actual = copy_case(actual)
            self.launch(actual)
            self.check_writes(actual, before_actual, expected, masks)
            self.read_only(actual, before_actual)
            after_compute = copy_case(actual)
            self.fold_public(actual, work)
            self.close(actual["state"], folded, dtype, "state")
            self.read_only(actual, after_compute, state=False)
            for key in ("out", "u", "hk", "G"):
                self.bits(actual[key], after_compute[key], "fold must not change candidates: " + key)
            untouched = [slot for slot in range(actual["state"].shape[0]) if slot not in [w[0] for w in work]]
            self.bits(actual["state"][untouched], before_actual["state"][untouched])
            for key, value in expected.items():
                model[key].copy_(value)
            model["state"].copy_(folded)
            # Only AFTER folding old metadata/history, advance caller metadata
            # by accepted candidate prefix (including zero on overflow).
            for row, _, r, begin, end, p, b in _rows(old):
                n = accept if row == 0 else c.T - accept
                overflow = p + end - begin > c.L
                for d in (actual, model):
                    d["buffers"][r] = 1 - b if overflow else b
                    d["counts"][r] = n if overflow else p + n
            fresh = make_cached_case(c, seed=301 + round_no)
            for key in ("q", "k", "v", "gate", "beta"):
                actual[key].copy_(fresh[key])
                model[key].copy_(fresh[key])

    def test_inline_adapter_graph_reads_new_counts_and_old_prefix(self):
        c = CachedCase(H=1, HV=2, dtype="float32")
        d = make_cached_case(c, counts=[0, 7, 11], shared_mapping=True, integer="int32", device=self.device)
        kwargs = adapter_kwargs(d)
        warm = self.t.cuda.Stream(device=self.device)
        warm.wait_stream(self.t.cuda.current_stream(self.device))
        with self.t.cuda.stream(warm):
            getattr(self.adapter, UPDATE)(**kwargs)
        self.t.cuda.current_stream(self.device).wait_stream(warm)
        graph = self.t.cuda.CUDAGraph()
        with self.t.cuda.graph(graph):
            getattr(self.adapter, UPDATE)(**kwargs)
        for counts, buffers in (([0, 7, 11], [1, 0, 1]), ([11, 8, 7], [0, 1, 0]), ([1, 11, 0], [1, 1, 0])):
            fresh = make_cached_case(c, counts=counts, buffers=buffers, shared_mapping=True, integer="int32")
            for key, value in fresh.items():
                if isinstance(value, self.t.Tensor):
                    d[key].copy_(value)
            expected, masks = reference_cached(fresh)
            graph.replay()
            self.t.cuda.synchronize(self.device)
            self.check_writes(d, fresh, expected, masks)
            self.read_only(d, fresh, state=False)
            self.check_fold(d, fresh, self.overflow_work(fresh))

    def test_graph_replay_device_dynamic_counts_and_buffers(self):
        c = CachedCase(dtype="float32", T=4, L=11)
        d = make_cached_case(c, counts=[0, 7, 11], device=self.device)
        # All setup, CPU references, and warmup precede capture.
        warm = self.t.cuda.Stream(device=self.device)
        warm.wait_stream(self.t.cuda.current_stream(self.device))
        with self.t.cuda.stream(warm):
            self.launch(d)
        self.t.cuda.current_stream(self.device).wait_stream(warm)
        graph = self.t.cuda.CUDAGraph()
        with self.t.cuda.graph(graph):
            self.launch(d)
        for counts, buffers in (([0, 7, 11], [0, 1, 0]), ([11, 0, 8], [1, 0, 1]), ([7, 11, 1], [0, 0, 1])):
            fresh = make_cached_case(c, counts=counts, buffers=buffers)
            for key, value in fresh.items():
                if isinstance(value, self.t.Tensor):
                    d[key].copy_(value)
            expected, masks = reference_cached(fresh)
            graph.replay()
            self.t.cuda.synchronize(self.device)
            self.check_writes(d, fresh, expected, masks)
            self.read_only(d, fresh)

    def test_layered_workitems_ignored_column_partial_and_zero_device_counter(self):
        c = CachedCase(H=1, HV=2)
        layers = [make_cached_case(c, shared_mapping=True, counts=[3, 7, 11], seed=101 + layer, device=self.device)
                  for layer in range(2)]
        state = self.t.stack([d["state"] for d in layers])
        us, ks, gs = [self.t.stack([d[key] for d in layers]) for key in ("u", "hk", "G")]
        items = self.t.tensor([[2147483647, 1, 3, 0], [-2147483648, 2, 7, 1], [-9, 3, 11, 0]],
                              dtype=self.t.int32, device=self.device)
        n = self.t.tensor([0], dtype=self.t.int32, device=self.device)
        saved = [self.cpu(x) for x in (state, us, ks, gs, items)]
        graph = self.t.cuda.CUDAGraph()
        kwargs = dict(ssm_states=state, old_u=us, old_k=ks, old_G=gs, replay_work_items=items,
                      n_writes=n, history_size=c.L)
        getattr(self.adapter, COMMIT)(**kwargs)
        with self.t.cuda.graph(graph):
            getattr(self.adapter, COMMIT)(**kwargs)
        for active in (0, 1, 2, 0, 3):
            state.copy_(saved[0])
            n.fill_(active)
            graph.replay()
            self.t.cuda.synchronize(self.device)
            for layer, d in enumerate(layers):
                work = [(int(row[1]), int(row[1]), int(row[2]), int(row[3])) for row in saved[-1][:active]]
                self.close(state[layer], reference_fold(d, work), c.dtype, "state")
                untouched = [slot for slot in range(state.shape[1]) if slot not in [w[0] for w in work]]
                self.bits(state[layer, untouched], saved[0][layer, untouched])
            for actual, before in zip((us, ks, gs, items), saved[1:]):
                self.bits(actual, before)
            self.assertEqual(int(n), active)

    def test_layered_descriptors_and_sliced_slot_views(self):
        c = CachedCase(B=2, H=1, HV=2)
        layers = [make_cached_case(c, shared_mapping=True, counts=[3, 7], seed=811 + i, device=self.device) for i in range(2)]
        owners, views = [], []
        for d in layers:
            owner = self.t.full((d["state"].shape[0] * 2 + 1, c.HV, c.V, c.K), 19.0,
                                dtype=d["state"].dtype, device=self.device)
            view = owner[1::2]
            view.copy_(d["state"])
            owners.append(owner)
            views.append(view)
        descriptors = self.t.tensor([[view.data_ptr(), view.stride(0)] for view in views],
                                    dtype=self.t.int64, device=self.device)
        template = self.t.zeros_like(layers[0]["state"])
        us, ks, gs = [self.t.stack([d[key] for d in layers]) for key in ("u", "hk", "G")]
        items = self.t.tensor([[777, 2, 7, 1]], dtype=self.t.int32, device=self.device)
        n = self.t.tensor(1, dtype=self.t.int32, device=self.device)
        saved = [self.cpu(x) for x in (*owners, template, descriptors, us, ks, gs, items, n)]
        getattr(self.adapter, COMMIT)(ssm_states=template, ssm_state_descriptors=descriptors,
                                     old_u=us, old_k=ks, old_G=gs, replay_work_items=items, n_writes=n, history_size=c.L)
        for layer, d in enumerate(layers):
            self.close(views[layer], reference_fold(d, [(2, 2, 7, 1)]), c.dtype, "state")
            self.bits(owners[layer][::2], saved[layer][::2], "descriptor slot padding")
            for slot in (0, 1, 3, 4):
                self.bits(views[layer][slot], d["state"][slot])
        for actual, before in zip((template, descriptors, us, ks, gs, items, n), saved[2:]):
            self.bits(actual, before)

    def test_layered_explicit_state_stride_arguments(self):
        c = CachedCase(B=2, H=1, HV=2)
        ds = [make_cached_case(c, shared_mapping=True, counts=[3, 7], seed=90 + i, device=self.device) for i in range(2)]
        owner = self.t.full((2, ds[0]["state"].shape[0] * 2, c.HV, c.V, c.K), 19.0,
                            dtype=ds[0]["state"].dtype, device=self.device)
        views = owner[:, 1::2]
        for i, d in enumerate(ds):
            views[i].copy_(d["state"])
        saved = self.cpu(owner)
        us, ks, gs = [self.t.stack([d[key] for d in ds]) for key in ("u", "hk", "G")]
        items = self.t.tensor([[0, 1, 3, 0]], dtype=self.t.int32, device=self.device)
        n = self.t.tensor(1, dtype=self.t.int32, device=self.device)
        getattr(self.adapter, COMMIT)(ssm_states=views[0], ssm_state_num_layers=2,
                                     ssm_state_layer_stride=views.stride(0), ssm_state_slot_stride=views.stride(1),
                                     old_u=us, old_k=ks, old_G=gs, replay_work_items=items, n_writes=n, history_size=c.L)
        for i, d in enumerate(ds):
            self.close(views[i], reference_fold(d, [(1, 1, 3, 0)]), c.dtype, "state")
        self.bits(owner[:, ::2], saved[:, ::2])

    def test_adapter_performance_hints_do_not_change_math(self):
        c = CachedCase(H=1, HV=4, K=32, V=32)
        original = make_cached_case(c, shared_mapping=True, counts=[0, 7, 11], device=self.device)
        baseline = copy_case(original, device=self.device)
        getattr(self.adapter, UPDATE)(**adapter_kwargs(baseline))
        variants = [dict(block_v=16, num_warps=4, main_num_stages=2),
                    dict(use_l2_state_cache=True, use_l2_streaming_inputs=True, use_l2_shared_inputs=True)]
        for hints in variants:
            d = copy_case(original, device=self.device)
            getattr(self.adapter, UPDATE)(**adapter_kwargs(d), **hints)
            for key in ("out", "u", "hk", "G", "state"):
                self.bits(d[key], baseline[key], key + " scheduling hint changed values")


RANGE_CASES = ("negative_count", "large_count", "negative_buffer", "large_buffer", "wide_buffer", "nonzero_start",
               "wide_count", "wide_start", "source_oob", "replay_oob", "cu_decreasing", "cu_too_long", "cu_endpoint",
               "commit_n_negative", "commit_n_overflow", "commit_buffer")


class BoundaryTests(DeviceBase):
    def rejected(self, fn, data, extra=()):
        before = copy_case(data)

        def tensors(value):
            if isinstance(value, self.t.Tensor):
                yield value
            elif is_dataclass(value):
                for field in fields(value):
                    yield from tensors(getattr(value, field.name))
            elif isinstance(value, dict):
                for child in value.values():
                    yield from tensors(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from tensors(child)

        extras = [(x, self.cpu(x)) for x in tensors(extra)]
        with self.assertRaises((ValueError, TypeError, RuntimeError, AssertionError)):
            fn()
        self.t.cuda.synchronize(self.device)
        for key, value in data.items():
            if isinstance(value, self.t.Tensor):
                self.bits(value, before[key], "validation must precede writes: " + key)
        for x, old in extras:
            self.bits(x, old)

    def test_invalid_public_types_shapes_and_workspace(self):
        d = make_cached_case(CachedCase(dtype="float32"), device=self.device)
        b = public_bundle(self.api, d)
        api, t = self.api, self.t
        mutations = [dict(step=replace(b["step"], buffers=None)),
                     dict(step=replace(b["step"], lengths=d["counts"].float())),
                     dict(step=replace(b["step"], buffers=d["buffers"].float())),
                     dict(step=replace(b["step"], starts=t.zeros_like(d["counts"], dtype=t.float32))),
                     dict(step=replace(b["step"], buffers=d["buffers"][:-1])),
                     dict(history=replace(b["history"], values=d["u"].half())),
                     dict(history=replace(b["history"], keys=d["hk"].half())),
                     dict(history=replace(b["history"], gates=d["G"].half())),
                     dict(history=replace(b["history"], values=d["u"][:, :1])),
                     dict(history=replace(b["history"], beta=d["beta"])),
                     dict(history=replace(b["history"], value_residual=d["u"], key_residual=d["hk"])),
                     dict(inputs=replace(b["inputs"], k=d["k"][:, :, :-1])),
                     dict(inputs=replace(b["inputs"], q=d["q"].double())),
                     dict(inputs=replace(b["inputs"], gate=d["gate"][:-1])),
                     dict(inputs=replace(b["inputs"], scale=float("nan"))),
                     dict(inputs=replace(b["inputs"], scale=True)),
                     dict(state=d["state"].half()), dict(state=d["state"].cpu()),
                     dict(out=d["out"].double()), dict(out=d["v"]),
                     dict(workspace=t.zeros(1, dtype=t.float16, device=self.device)),
                     dict(workspace=d["state"].reshape(-1)),
                     dict(numeric_policy=api.NumericPolicy(gate_mode="raw"))]
        for i, mutation in enumerate(mutations):
            with self.subTest(mutation=i):
                self.rejected(lambda: self.launch(d, bundle={**b, **mutation}), d, extra=(mutation,))
        for invalid_T in (0, 9, True, 1.5, d["config"].L + 1):
            with self.subTest(max_verify_len=invalid_T):
                self.rejected(lambda: api.gdn_replay_verify(**b, query_start_loc=None, max_verify_len=invalid_T), d)
        for L in (0, 3, 17):
            bad_h = api.ReplayHistory(api.HistoryFormat.CACHED_UKG_DOUBLE,
                                     t.full((6, 2, L, 4, 16), 7.0, device=self.device),
                                     t.full((6, 2, L, 2, 32), -3.0, device=self.device),
                                     t.full((6, 2, 4, L), 11.0, device=self.device))
            self.rejected(lambda: self.launch(d, bundle={**b, "history": bad_h}), d, extra=(bad_h,))
        for K, HV in ((257, 4), (32, 3)):
            # Isolate K<=256 and H|HV rather than relying on unrelated shape errors.
            q = t.full((12, 2, K), 0.1, device=self.device)
            v = t.full((12, HV, 16), 0.2, device=self.device)
            gate = t.full((12, HV), -0.1, device=self.device)
            state = t.full((6, HV, 16, K), 0.3, device=self.device)
            history = api.ReplayHistory(api.HistoryFormat.CACHED_UKG_DOUBLE,
                                       t.full((6, 2, 11, HV, 16), 7.0, device=self.device),
                                       t.full((6, 2, 11, 2, K), -3.0, device=self.device),
                                       t.full((6, 2, HV, 11), 11.0, device=self.device))
            bad = {**b, "inputs": replace(b["inputs"], q=q, k=q.clone(), v=v, gate=gate, beta=-gate),
                   "history": history, "state": state, "out": v.clone()}
            self.rejected(lambda: self.launch(d, bundle=bad), d, extra=(bad,))
        for cu in (t.tensor([0, 4, 8, 12], dtype=t.float32, device=self.device),
                   t.tensor([0, 4, 8, 12], dtype=t.int64, device="cpu"),
                   t.tensor([0, 4, 8], dtype=t.int64, device=self.device)):
            self.rejected(lambda: api.gdn_replay_verify(**b, query_start_loc=cu, max_verify_len=4), d, extra=(cu,))

    def test_adapter_pairing_pdl_and_bad_packed_shape_reject_before_writes(self):
        d = make_cached_case(CachedCase(), shared_mapping=True, integer="int32", device=self.device)
        kw = adapter_kwargs(d)
        items = self.t.zeros((d["config"].B, 4), dtype=self.t.int32, device=self.device)
        n = self.t.zeros(1, dtype=self.t.int32, device=self.device)
        mutations = [dict(replay_work_items=items), dict(n_writes=n), dict(use_all_layer_commit=True),
                     dict(replay_work_items=items, n_writes=n, use_all_layer_commit=False),
                     dict(packed_qkv=d["q"]),
                     dict(A_log=self.t.zeros(d["config"].HV, device=self.device))]
        for mutation in mutations:
            with self.subTest(fields=list(mutation)):
                self.rejected(lambda: getattr(self.adapter, UPDATE)(**{**kw, **mutation}), d, extra=(items, n, mutation))
        if self.t.cuda.get_device_capability(self.device)[0] >= 9:
            with self.assertRaises(NotImplementedError):
                getattr(self.adapter, UPDATE)(**kw, launch_with_pdl=True)
        else:
            baseline = copy_case(d, device=self.device)
            getattr(self.adapter, UPDATE)(**adapter_kwargs(baseline))
            getattr(self.adapter, UPDATE)(**kw, launch_with_pdl=True)
            for key in ("out", "u", "hk", "G", "state", "counts", "buffers"):
                self.bits(d[key], baseline[key])

    def test_layered_commit_invalid_descriptors_counter_and_shapes(self):
        d = make_cached_case(CachedCase(), shared_mapping=True, integer="int32", device=self.device)
        t = self.t
        items = t.tensor([[71, 1, 3, 0]], dtype=t.int32, device=self.device)
        n = t.tensor(1, dtype=t.int32, device=self.device)
        kw = dict(ssm_states=d["state"].unsqueeze(0), old_u=d["u"].unsqueeze(0),
                  old_k=d["hk"].unsqueeze(0), old_G=d["G"].unsqueeze(0), replay_work_items=items,
                  n_writes=n, history_size=d["config"].L)
        mutations = [dict(replay_work_items=items[:, :3]), dict(replay_work_items=items.float()),
                     dict(n_writes=n.float()), dict(n_writes=t.zeros((1, 1), dtype=t.int32, device=self.device)),
                     dict(n_writes=n.cpu()), dict(old_G=kw["old_G"].half()),
                     dict(ssm_states=d["state"], ssm_state_descriptors=t.zeros((1, 2), dtype=t.float32, device=self.device)),
                     dict(ssm_states=d["state"], ssm_state_num_layers=1),
                     dict(ssm_states=d["state"], ssm_state_num_layers=2, ssm_state_layer_stride=0,
                          ssm_state_slot_stride=d["state"].stride(0))]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                self.rejected(lambda: getattr(self.adapter, COMMIT)(**{**kw, **mutation}), d, extra=(items, n, mutation))

    def test_device_range_errors_in_fresh_processes(self):
        for name in RANGE_CASES:
            with self.subTest(problem=name):
                child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--device", str(self.device),
                                        "--range-case", name], text=True, capture_output=True, check=False, timeout=180)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
                reports = [json.loads(line) for line in child.stdout.splitlines() if line.startswith('{"suite":')]
                self.assertEqual(len(reports), 1, child.stdout + child.stderr)
                self.assertEqual(reports[0]["status"], "PASSED")
                self.assertEqual(reports[0]["passed_test_methods"], 1)

    def _invalid_range_child(self):
        name = OPTIONS.range_case
        d = make_cached_case(CachedCase(), counts=[3, 7, 11], explicit_cu=True, integer="int64", device=self.device)
        r = int(d["replays"][0])
        if name.startswith("commit_"):
            items = self.t.tensor([[83, r, 3, 2 if name == "commit_buffer" else 0]],
                                  dtype=self.t.int32, device=self.device)
            n = self.t.tensor(-1 if name == "commit_n_negative" else 2 if name == "commit_n_overflow" else 1,
                              dtype=self.t.int32, device=self.device)
            with self.assertRaises((ValueError, RuntimeError, AssertionError)):
                self.adapter_commit(d, items, n)
                self.t.cuda.synchronize(self.device)
            return  # No more GPU operations in the poisoned child context.
        if name in ("negative_count", "large_count", "wide_count"):
            d["counts"][r] = {"negative_count": -1, "large_count": 12, "wide_count": 2 ** 32}[name]
        elif name in ("negative_buffer", "large_buffer", "wide_buffer"):
            d["buffers"][r] = {"negative_buffer": -1, "large_buffer": 2, "wide_buffer": 2 ** 32}[name]
        elif name in ("nonzero_start", "wide_start"):
            d["starts"] = self.t.zeros_like(d["counts"])
            d["starts"][r] = 1 if name == "nonzero_start" else 2 ** 32
        elif name == "source_oob":
            d["sources"][0] = d["state"].shape[0]
        elif name == "replay_oob":
            d["replays"][0] = d["u"].shape[0]
        elif name == "cu_decreasing":
            d["cu"].copy_(self.t.tensor([0, 4, 3, 12], device=self.device))
        elif name == "cu_too_long":
            d["cu"].copy_(self.t.tensor([0, 5, 8, 12], device=self.device))
        elif name == "cu_endpoint":
            d["cu"][-1] = 13
        else:
            raise AssertionError(name)
        with self.assertRaises((ValueError, RuntimeError, AssertionError)):
            self.launch(d)
            self.t.cuda.synchronize(self.device)
        # Do not touch CUDA after an expected device assert poisons this process.


class PerformanceTests(DeviceBase):
    def test_zero_workspace_no_implicit_tensor_allocation_and_event_measurement(self):
        c = CachedCase(B=8, H=1, HV=4, K=128, V=128, T=4, L=16)
        d = make_cached_case(c, counts=[0, 12, 13, 16] * 2, device=self.device)
        b = public_bundle(self.api, d)
        self.assertEqual(self.api.gdn_replay_workspace_elements(b["inputs"], b["state"], b["history"], b["mapping"], c.T), 0)
        self.checked_public(d)
        # Unused optional caller workspace must also be untouched.
        workspace = self.t.full((13,), 23.0, device=self.device)
        saved = self.cpu(workspace)
        b["workspace"] = workspace
        self.checked_public(d, bundle=b)
        self.bits(workspace, saved)
        for _ in range(3):
            self.launch(d, bundle=b)
        self.t.cuda.synchronize(self.device)
        baseline = self.t.cuda.memory_allocated(self.device)
        self.t.cuda.reset_peak_memory_stats(self.device)
        for _ in range(10):
            self.launch(d, bundle=b)
        self.t.cuda.synchronize(self.device)
        self.assertEqual(self.t.cuda.max_memory_allocated(self.device), baseline,
                         "public cached Verify must not allocate an implicit torch workspace")
        start, end = self.t.cuda.Event(enable_timing=True), self.t.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            self.launch(d, bundle=b)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / 50
        self.assertTrue(math.isfinite(ms) and ms > 0)
        print(json.dumps(dict(suite="performance_measurement", device=str(self.device), mean_ms=ms,
                              repeats=50, speedup="NOT_MEASURED", validation="accuracy checked separately")))


class CountingResult(unittest.TextTestResult):
    passed_test_methods = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_test_methods += 1


def not_run(reason):
    print(json.dumps(dict(suite="cached", status="NOT_RUN", reason=reason, tests_run=0,
                          passed_test_methods=0, numerical="NOT_RUN", performance="NOT_RUN")))
    return 2


def main(argv=None):
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--device", help="explicit cuda:N; no CPU fallback")
    parser.add_argument("--suite", choices=("correctness", "accuracy", "lifecycle", "boundary", "performance"), default="correctness")
    parser.add_argument("--test", action="append", help="exact method name, repeatable")
    parser.add_argument("--range-case", choices=RANGE_CASES, help=argparse.SUPPRESS)
    OPTIONS = parser.parse_args(argv)
    if OPTIONS.static and (OPTIONS.device or OPTIONS.range_case or OPTIONS.suite != "correctness"):
        parser.error("--static cannot select a device or numerical suite")
    if OPTIONS.range_case and OPTIONS.test:
        parser.error("--range-case cannot be combined with --test")
    if OPTIONS.static:
        classes = [StaticTests]
    else:
        if not OPTIONS.device:
            return not_run("explicit --device cuda:N is required")
        if re.fullmatch(r"cuda:[0-9]+", OPTIONS.device) is None:
            parser.error("--device must explicitly name cuda:N")
        if importlib.util.find_spec("torch") is None:
            return not_run("torch is not installed")
        t = torch_module()
        if not t.cuda.is_available():
            return not_run("CUDA is not available")
        if int(OPTIONS.device.split(":")[1]) >= t.cuda.device_count():
            return not_run("selected device does not exist")
        classes = {"correctness": [AccuracyTests, LifecycleTests, BoundaryTests], "accuracy": [AccuracyTests],
                   "lifecycle": [LifecycleTests], "boundary": [BoundaryTests], "performance": [PerformanceTests]}[OPTIONS.suite]
    if OPTIONS.range_case:
        suite = unittest.TestSuite([BoundaryTests("_invalid_range_child")])
    elif OPTIONS.test:
        tests = []
        for name in OPTIONS.test:
            matches = [cls for cls in classes if name in unittest.defaultTestLoader.getTestCaseNames(cls)]
            if len(matches) != 1:
                parser.error("selected test missing or ambiguous: " + name)
            tests.append(matches[0](name))
        suite = unittest.TestSuite(tests)
    else:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
    result = unittest.TextTestRunner(verbosity=2, resultclass=CountingResult).run(suite)
    failed = len(result.errors) + len(result.failures)
    if OPTIONS.static and "torch" in sys.modules:
        failed += 1
    status = "FAILED" if failed else "NOT_RUN" if result.skipped or not result.testsRun else "PASSED"
    print(json.dumps(dict(suite="static" if OPTIONS.static else OPTIONS.suite, status=status,
                          tests_run=result.testsRun, passed_test_methods=result.passed_test_methods,
                          skipped_not_passed=len(result.skipped), failures=failed,
                          numerical="NOT_RUN" if OPTIONS.static else status,
                          performance=status if OPTIONS.suite == "performance" else "NOT_RUN")))
    return 1 if failed else 2 if status == "NOT_RUN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
