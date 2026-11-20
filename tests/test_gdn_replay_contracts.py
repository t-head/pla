# SPDX-License-Identifier: Apache-2.0
"""Locked common replay contracts; no legacy-signature/source-hash duplication.

Static (does not import torch): python -B tests/test_gdn_replay_contracts.py --static
Numerical: python -B tests/test_gdn_replay_contracts.py --device cuda:N
Debug: append --test test_sglang_p6_t4_a3_materializes_nine_without_cursor_commit

Only the explicit device is selected. Missing CUDA/torch is NOT_RUN (exit 2),
not a passing numerical suite. No mocks, upstream imports, or tolerance tuning.
Reference fixtures/budgets come from scripts/decode/gdn_replayssm/reference.py,
shared with test_gdn_replay.py. All expected recurrences run on CPU in FP64.
Invalid device-resident bounds run in isolated processes because a device assert
poisons its CUDA context; structural validation must reject before any writes.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pla" / "decode" / "gdn_replayssm"
REFERENCE = ROOT / "scripts" / "decode" / "gdn_replayssm" / "reference.py"
FORMATS = {
    "COMPACT_DKG_RING": "compact_dkg_ring",
    "RAW_VKG_BETA_WINDOW": "raw_vkg_beta_window",
    "CACHED_UKG_DOUBLE": "cached_ukG_double",
}
FIELDS = {
    "NumericPolicy": ("qk_norm", "epsilon", "gate_mode", "dot_precision"),
    "ReplayInputs": ("q", "k", "v", "gate", "beta", "A_log", "dt_bias", "scale"),
    "ReplayHistory": ("format", "values", "keys", "gates", "value_residual", "key_residual", "beta"),
    "ReplayMapping": ("state_indices", "replay_indices", "null_block_id"),
    "ReplayStep": ("lengths", "starts", "buffers"),
    "MaterializeWork": (
        "state_indices", "replay_indices", "lengths", "starts", "buffers",
        "metadata_indices", "target_state_indices", "enabled", "items", "n_items",
    ),
}
CALLS = {
    "gdn_replay_decode": (
        ("inputs", "state", "history", "mapping", "step", "out", "workspace"),
        ("numeric_policy",),
    ),
    "gdn_replay_verify": (
        ("inputs", "state", "history", "mapping", "step", "out", "workspace"),
        ("query_start_loc", "max_verify_len", "numeric_policy"),
    ),
    "gdn_replay_materialize": (
        ("state_layers", "history", "work"),
        ("numeric_policy", "null_block_id", "state_descriptors"),
    ),
    "gdn_replay_workspace_elements": (
        ("inputs", "state", "history", "mapping", "max_verify_len"), (),
    ),
}
BUDGETS = {
    "float32": {"output": (1e-4, 1e-3), "state": (1.5e-3, 1e-3)},
    "bfloat16": {"output": (1e-3, 1e-2), "state": (3e-3, 1e-2)},
    "float16": {"output": (5e-4, 3e-3), "state": (1.5e-3, 3e-3)},
}
OPTIONS = argparse.Namespace(device=None, range_case=None)
_REF = None


def reference():
    global _REF
    if _REF is None:
        spec = importlib.util.spec_from_file_location("_replay_contract_reference", REFERENCE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load fixture: {REFERENCE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _REF = module
    return _REF


def public_ast(name, module="__init__", seen=()):
    """Follow actual explicit public re-exports, never import the CUDA package."""
    key = (module, name)
    if key in seen:
        raise AssertionError(f"cyclic public export: {key}")
    path = PACKAGE / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return node
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return public_ast(alias.name, node.module, (*seen, key))
    raise AssertionError(f"missing locked public export {name!r} in {path}")


class StaticContracts(unittest.TestCase):
    def test_public_frozen_dataclasses_and_constants(self):
        enum = public_ast("HistoryFormat")
        values = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in enum.body if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        }
        self.assertEqual(values, FORMATS)
        for name, names in FIELDS.items():
            with self.subTest(dataclass=name):
                node = public_ast(name)
                self.assertTrue(any(
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                    and d.func.id == "dataclass"
                    and any(k.arg == "frozen" and ast.literal_eval(k.value) is True for k in d.keywords)
                    for d in node.decorator_list
                ), f"{name} must be @dataclass(frozen=True)")
                declared = [n for n in node.body if isinstance(n, ast.AnnAssign)]
                self.assertEqual(tuple(n.target.id for n in declared), names)
                if name == "MaterializeWork":
                    for n in declared:
                        self.assertIsNotNone(n.value)
                        self.assertIsNone(ast.literal_eval(n.value), n.target.id)
                if name == "NumericPolicy":
                    self.assertEqual([ast.literal_eval(n.value) for n in declared],
                                     ["l2_inside_sqrt", 1e-6, "raw", "tf32"])
                if name == "ReplayMapping":
                    self.assertEqual(ast.literal_eval(declared[-1].value), -1)
                if name == "ReplayStep":
                    self.assertIsNone(declared[0].value)
                    for n in declared[1:]:
                        self.assertIsNotNone(n.value)
                        self.assertIsNone(ast.literal_eval(n.value))
                if name in ("ReplayInputs", "ReplayHistory"):
                    for n in declared[-3:]:
                        self.assertIsNotNone(n.value)
                        self.assertIsNone(ast.literal_eval(n.value))

    def test_common_signatures_not_legacy_adapters(self):
        for name, (positional, keywords) in CALLS.items():
            with self.subTest(function=name):
                args = public_ast(name).args
                self.assertEqual(tuple(a.arg for a in args.posonlyargs + args.args), positional)
                self.assertEqual(tuple(a.arg for a in args.kwonlyargs), keywords)
                self.assertIsNone(args.vararg)
                self.assertIsNone(args.kwarg)
                defaults = dict(zip(keywords, args.kw_defaults))
                if "numeric_policy" in defaults:
                    policy = defaults["numeric_policy"]
                    self.assertIsInstance(policy, ast.Call)
                    self.assertIsInstance(policy.func, ast.Name)
                    self.assertEqual(policy.func.id, "NumericPolicy")
                    self.assertEqual(policy.args, [])
                    self.assertEqual(policy.keywords, [])
                if name == "gdn_replay_verify":
                    self.assertIsNone(defaults["query_start_loc"])
                    self.assertIsNone(defaults["max_verify_len"])
                if name == "gdn_replay_materialize":
                    self.assertEqual(ast.literal_eval(defaults["null_block_id"]), -1)
                    self.assertIsNone(ast.literal_eval(defaults["state_descriptors"]))

    def test_fixed_reference_budgets_and_ast_without_torch(self):
        before = "torch" in sys.modules
        ref = reference()
        self.assertEqual("torch" in sys.modules, before, "fixture import must not import torch")
        for dtype, budget in BUDGETS.items():
            self.assertEqual(ref.TOLERANCES[dtype], budget)
        for path in (Path(__file__).resolve(), REFERENCE, *sorted(PACKAGE.glob("*.py"))):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            compile(tree, str(path), "exec")

    def test_hand_calculated_norm_and_cumulative_gate_contracts(self):
        q, eps = 1e-4, 1e-6
        norms = (q, q / math.sqrt(q * q + eps), q / (math.sqrt(q * q) + eps))
        for actual, expected in zip(norms, (0.0001, 0.09950371902099892, 0.9900990099009901)):
            self.assertAlmostEqual(actual, expected, places=14)
        g0, g1 = math.log(0.5), math.log(0.25)
        sequential = (2 * math.exp(g0) + 3) * math.exp(g1) + 5
        cumulative = 2 * math.exp(g0 + g1) + 3 * math.exp(g1) + 5
        subrange = 4 * math.exp((g0 + g1) - g0) + 5
        self.assertAlmostEqual(sequential, 6.0)
        self.assertAlmostEqual(cumulative, 6.0)
        self.assertAlmostEqual(subrange, 6.0)
        self.assertAlmostEqual(4 * math.exp(g0 + g1) + 5, 5.5)
        self.assertEqual(6 + 3, 9)  # SG P6 + accepted A3, not P6 + proposed T4.
        self.assertGreater(10 + 8, 16)  # Holmes must select old buffer/P10.


class DeviceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OPTIONS.device:
            raise unittest.SkipTest("NOT_RUN: explicit --device cuda:N required")
        cls.t = importlib.import_module("torch")
        cls.device = cls.t.device(OPTIONS.device)
        cls.t.cuda.set_device(cls.device)
        # Exact locked import. Missing public APIs/extensions are failures, not skips.
        if importlib.util.find_spec("pla") is None:
            sys.path.insert(0, str(ROOT))
        cls.api = importlib.import_module("pla.decode.gdn_replayssm")
        for name in ("HistoryFormat", *FIELDS, *CALLS):
            if not hasattr(cls.api, name):
                raise AssertionError(f"missing locked public API: {name}")
        cls.ref = reference()

    def cpu(self, tensor):
        return tensor.detach().cpu().clone()

    def bits_equal(self, actual, expected, message="unexpected write"):
        a, b = self.cpu(actual), self.cpu(expected)
        self.assertEqual(a.shape, b.shape, message)
        self.assertEqual(a.dtype, b.dtype, message)
        # A size-one view can be contiguous while retaining stride > 1.
        aa = self.t.empty(a.numel(), dtype=a.dtype, device="cpu").copy_(a.reshape(-1))
        bb = self.t.empty(b.numel(), dtype=b.dtype, device="cpu").copy_(b.reshape(-1))
        self.assertTrue(self.t.equal(aa.view(self.t.uint8), bb.view(self.t.uint8)), message)

    def close(self, actual, expected, dtype="float32", kind="state"):
        atol, rtol = self.ref.TOLERANCES[dtype][kind]
        self.t.testing.assert_close(self.cpu(actual).double(), self.cpu(expected).double(),
                                    atol=atol, rtol=rtol)

    def tensor_fields(self, *objects):
        result = []
        for obj in objects:
            if isinstance(obj, self.t.Tensor):
                result.append(obj)
            elif is_dataclass(obj):
                result.extend(getattr(obj, f.name) for f in fields(obj)
                              if isinstance(getattr(obj, f.name), self.t.Tensor))
        return result

    def snapshots(self, *objects):
        return [(x, self.cpu(x)) for x in self.tensor_fields(*objects)]

    def unchanged(self, snapshots):
        for tensor, before in snapshots:
            self.bits_equal(tensor, before)

    def on_device(self, value):
        if isinstance(value, self.t.Tensor):
            return self.t.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                        device=self.device).copy_(value)
        if isinstance(value, dict):
            return {k: self.on_device(v) for k, v in value.items()}
        return value

    def ints(self, values, dtype=None):
        return self.ref.strided_int(values, self.device, dtype=dtype or self.t.int64)

    def case(self, *, residual=False, strided=True, state_dtype="float32", lengths=None, **changes):
        config = dict(B=3, H=2, HV=4, K=32, V=16, T=4, L=16,
                      layers=1, history=3, dtype="float32")
        config.update(changes)
        data = self.ref.make_case(self.ref.Case(**config), device="cpu", seed=113,
                                  residual=residual, strided=strided,
                                  state_dtype=state_dtype, lengths=lengths)
        # CPU-generated, source-quantized inputs are identical on every GPU.
        return self.on_device(data)

    def history(self, data, *, layer=None):
        def pick(name):
            value = data[name]
            return value if layer is None or value is None else value[layer]
        return self.api.ReplayHistory(
            self.api.HistoryFormat.COMPACT_DKG_RING,
            pick("d_cache"), pick("k_cache"), pick("g_cache"),
            value_residual=pick("rawv_cache"), key_residual=pick("rawk_cache"),
        )

    def verify_bundle(self, data, *, layer=0, policy=None, scale=None):
        return dict(
            inputs=self.api.ReplayInputs(data["q"][layer], data["k"][layer], data["v"][layer],
                                         data["a"][layer], data["b"][layer],
                                         A_log=data["A_log"][layer], dt_bias=data["dt_bias"][layer],
                                         scale=scale),
            state=data["checkpoint_state"][layer], history=self.history(data, layer=layer),
            mapping=self.api.ReplayMapping(data["ssm_state_indices"], data["replay_indices"],
                                           null_block_id=data["null_block_id"]),
            step=self.api.ReplayStep(data["write_pos"], data["cache_base"]),
            out=data["out"][layer], workspace=None,
            policy=policy or self.api.NumericPolicy(dot_precision="ieee"),
            query_start_loc=data["query_start_loc"], max_verify_len=data["case"].T,
            dtype=data["case"].dtype,
        )

    def launch_verify(self, bundle, *, decode=False):
        args = [bundle[k] for k in ("inputs", "state", "history", "mapping", "step", "out", "workspace")]
        if decode:
            return self.api.gdn_replay_decode(*args, numeric_policy=bundle["policy"])
        return self.api.gdn_replay_verify(*args, query_start_loc=bundle["query_start_loc"],
                                           max_verify_len=bundle["max_verify_len"],
                                           numeric_policy=bundle["policy"])

    def norm(self, x, policy):
        x = x.double()
        if policy.qk_norm == "none":
            return x
        norm2 = x.square().sum(-1, keepdim=True)
        if policy.qk_norm == "l2_inside_sqrt":
            return x / (norm2 + policy.epsilon).sqrt()
        if policy.qk_norm == "l2_outside_sqrt":
            return x / (norm2.sqrt() + policy.epsilon)
        raise AssertionError(f"unknown policy in oracle: {policy.qk_norm}")

    def recurrent(self, state, key, value, gate, beta):
        # Independent CPU FP64 recurrence, never a public/device implementation.
        decayed = state.double() * gate.double().exp()[:, None, None]
        prediction = self.t.einsum("hvk,hk->hv", decayed, key.double())
        delta = beta.double()[:, None] * (value.double() - prediction)
        return decayed + self.t.einsum("hv,hk->hvk", delta, key.double()), delta

    def materialized(self, initial, history, layer, replay, count, start=0, buffer=0, policy=None):
        """FP64 closed-form compact/cached oracle; raw uses sequential recurrence."""
        t, fmt = self.t, history.format
        state = initial.double().clone()
        if not count:
            return state
        policy = policy or self.api.NumericPolicy(dot_precision="ieee")
        values, keys, gates = (getattr(history, n).detach().cpu() for n in ("values", "keys", "gates"))
        if fmt == self.api.HistoryFormat.CACHED_UKG_DOUBLE:
            u = values[layer, replay, buffer, start:start + count].double().transpose(0, 1)
            k = keys[layer, replay, buffer, start:start + count].double().transpose(0, 1)
            G = gates[layer, replay, buffer].double()
            segment = G[:, start:start + count]
            origin = G[:, start - 1] if start else t.zeros_like(G[:, 0])
            total = segment[:, -1] - origin
            weights = (segment[:, -1:] - segment).exp()
        else:
            capacity = values.shape[-2]
            positions = [(start + i) % capacity if fmt == self.api.HistoryFormat.COMPACT_DKG_RING
                         else start + i for i in range(count)]
            u = values[layer, replay][:, positions].double()
            k = keys[layer, replay][:, positions].double()
            g = gates[layer, replay][:, positions].double()
            if fmt == self.api.HistoryFormat.RAW_VKG_BETA_WINDOW:
                k = self.norm(k, policy)
                k = k.repeat_interleave(state.shape[0] // k.shape[0], dim=0)
                beta = self.cpu(history.beta)[layer, replay][:, positions].double()
                for i in range(count):
                    state, _ = self.recurrent(state, k[:, i], u[:, i], g[:, i], beta[:, i])
                return state
            self.assertEqual(fmt, self.api.HistoryFormat.COMPACT_DKG_RING)
            if history.value_residual is not None:
                u += self.cpu(history.value_residual)[layer, replay][:, positions].double()
                k += self.cpu(history.key_residual)[layer, replay][:, positions].double()
            total = g.sum(-1)
            weights = (total[:, None] - g.cumsum(-1)).exp()
        k = k.repeat_interleave(state.shape[0] // k.shape[0], dim=0)
        return state * total.exp()[:, None, None] + t.einsum("htv,htk,ht->hvk", u, k, weights)

    def verify_oracle(self, bundle):
        policy, inputs, h = bundle["policy"], bundle["inputs"], bundle["history"]
        q, k, v, gate, beta = (self.cpu(getattr(inputs, n)).double()
                              for n in ("q", "k", "v", "gate", "beta"))
        q, k = self.norm(q, policy), self.norm(k, policy)
        q *= q.shape[-1] ** -0.5 if inputs.scale is None else inputs.scale
        if policy.gate_mode == "raw":
            x = gate + self.cpu(inputs.dt_bias).double()
            gate = -self.cpu(inputs.A_log).double().exp() * self.t.nn.functional.softplus(x, threshold=20)
            beta = beta.sigmoid()
        expected = self.cpu(bundle["out"]).double()
        checkpoint = self.cpu(bundle["state"])
        counts = self.cpu(bundle["step"].lengths).tolist()
        starts = self.cpu(bundle["step"].starts).tolist()
        offsets = self.cpu(bundle["query_start_loc"]).tolist()
        states = self.cpu(bundle["mapping"].state_indices).tolist()
        replays = self.cpu(bundle["mapping"].replay_indices).tolist()
        layered = replace(h, **{f.name: self.cpu(getattr(h, f.name)).unsqueeze(0) for f in fields(h)
                               if isinstance(getattr(h, f.name), self.t.Tensor)})
        layered = replace(layered, value_residual=None, key_residual=None)
        updates, snapshots = {}, {}
        for row, (slot, replay) in enumerate(zip(states, replays)):
            begin, end = offsets[row:row + 2]
            if slot <= bundle["mapping"].null_block_id:
                expected[begin:end] = 0
                continue
            state = self.materialized(checkpoint[slot], layered, 0, replay, counts[replay], starts[replay])
            heads = state.shape[0] // k.shape[1]
            for offset, token in enumerate(range(begin, end)):
                state, delta = self.recurrent(state, k[token].repeat_interleave(heads, 0),
                                               v[token], gate[token], beta[token])
                expected[token] = self.t.einsum("hvk,hk->hv", state, q[token].repeat_interleave(heads, 0))
                position = (starts[replay] + counts[replay] + offset) % h.values.shape[-2]
                updates[replay, position] = (delta, k[token], gate[token])
                snapshots[row, offset] = state.clone()
        return expected, updates, snapshots

    def check_verify(self, bundle, *, decode=False, launch=None):
        expected, updates, states = self.verify_oracle(bundle)
        readonly = self.snapshots(bundle["inputs"], bundle["state"], bundle["mapping"],
                                  bundle["step"], bundle["query_start_loc"])
        history = bundle["history"]
        before = {f.name: self.cpu(getattr(history, f.name)) for f in fields(history)
                  if isinstance(getattr(history, f.name), self.t.Tensor)}
        if launch is None:
            self.launch_verify(bundle, decode=decode)
        else:
            launch()
        self.t.cuda.synchronize(self.device)
        self.unchanged(readonly)
        self.close(bundle["out"], expected, bundle["dtype"], "output")
        observed_fields = {name: self.cpu(getattr(history, name)) for name in before}
        for name, old in before.items():
            actual = observed_fields[name]
            mask = self.t.zeros_like(old, dtype=self.t.bool)
            for (replay, pos), (delta, key, gate) in updates.items():
                mask[replay, :, pos] = True
                if name in ("values", "keys"):
                    wanted = delta if name == "values" else key
                    residual_name = "value_residual" if name == "values" else "key_residual"
                    observed = actual[replay, :, pos].double()
                    if residual_name in observed_fields:
                        observed += observed_fields[residual_name][replay, :, pos].double()
                    else:
                        wanted = wanted.float().to(actual.dtype).double()
                    self.close(observed, wanted, bundle["dtype"])
                elif name == "gates":
                    self.close(actual[replay, :, pos], gate, bundle["dtype"])
            self.bits_equal(actual[~mask], old[~mask], f"verify changed noncandidate {name}")
        return states

    def mat_fixture(self, fmt, *, dtype="float32", layers=2, state_dtype="float32", **changes):
        data = self.case(dtype=dtype, layers=layers, state_dtype=state_dtype,
                         residual=fmt == "COMPACT_DKG_RING", **changes)
        history = self.history(data)
        if fmt == "RAW_VKG_BETA_WINDOW":
            # Raw keys deliberately have small non-unit norms; cached beta is processed.
            history = self.api.ReplayHistory(
                getattr(self.api.HistoryFormat, fmt), data["d_cache"], data["k_cache"] * 0.23,
                data["g_cache"], beta=self.t.full_like(data["g_cache"], 0.3))
        elif fmt == "CACHED_UKG_DOUBLE":
            values = data["d_cache"].permute(0, 1, 3, 2, 4)
            keys = data["k_cache"].permute(0, 1, 3, 2, 4)
            history = self.api.ReplayHistory(
                getattr(self.api.HistoryFormat, fmt),
                self.t.stack((values, -0.7 * values + 0.04), dim=2).contiguous(),
                self.t.stack((keys, -keys), dim=2).contiguous(),
                self.t.stack((data["g_cache"].cumsum(-1),
                              (data["g_cache"] * 0.43 - 0.02).cumsum(-1)), dim=2).contiguous())
        return data, history

    def work_rows(self, work, null=-1):
        """Read CPU snapshots only, outside the API/graph under test."""
        get = lambda name: self.cpu(getattr(work, name)).tolist() if getattr(work, name) is not None else None
        items, sources, replays = get("items"), get("state_indices"), get("replay_indices")
        n = len(items) if items is not None else len(sources)
        if work.n_items is not None:
            n = int(self.cpu(work.n_items))
        lengths, starts, buffers = get("lengths"), get("starts"), get("buffers")
        metadata, targets, enabled = get("metadata_indices"), get("target_state_indices"), get("enabled")
        rows = []
        for row in range(n):
            source = items[row][1] if items is not None else sources[row]
            target = source if targets is None else targets[row]
            if source <= null or target <= null:
                continue
            if items is not None:
                replay, count, buffer, start = source, items[row][2], items[row][3], 0
            else:
                m = row if metadata is None else metadata[row]
                if enabled is not None and not enabled[m]:
                    continue
                replay, count = replays[row], lengths[m]
                start, buffer = (0 if starts is None else starts[m]), (0 if buffers is None else buffers[m])
            rows.append((source, replay, target, count, start, buffer))
        return rows

    def check_materialize(self, state, history, work, *, policy=None, null=-1,
                          descriptors=None, actual_layers=None, launch=None):
        policy = policy or self.api.NumericPolicy(dot_precision="ieee")
        layers = list(state.unbind(0)) if actual_layers is None else actual_layers
        before = [self.cpu(layer) for layer in layers]
        readonly = self.snapshots(history, work, descriptors)
        rows = self.work_rows(work, null)
        expected = {}
        for layer, initial in enumerate(before):
            for source, replay, target, count, start, buffer in rows:
                expected[layer, target] = self.materialized(initial[source], history, layer, replay,
                                                            count, start, buffer, policy)
        if launch is None:
            self.api.gdn_replay_materialize(state, history, work, numeric_policy=policy,
                                             null_block_id=null, state_descriptors=descriptors)
        else:
            launch()
        self.t.cuda.synchronize(self.device)
        self.unchanged(readonly)
        dtype = str(history.values.dtype).split(".")[-1]
        for layer, actual in enumerate(layers):
            writable = self.t.zeros(actual.shape[0], dtype=self.t.bool)
            for source, replay, target, count, start, buffer in rows:
                if count:
                    writable[target] = True
                    self.close(actual[target], expected[layer, target], dtype)
                elif source != target:
                    writable[target] = True
                    self.bits_equal(actual[target], before[layer][source], "count=0 must bitwise clone")
            self.bits_equal(self.cpu(actual)[~writable], before[layer][~writable],
                            "materialize changed source/inactive/zero-count state")
        return expected

    def assert_rejected_without_writes(self, call, *objects):
        before = self.snapshots(*objects)
        with self.assertRaises((ValueError, TypeError, RuntimeError)):
            call()  # Structural errors must be synchronous, not a later device assert.
        self.t.cuda.synchronize(self.device)
        self.unchanged(before)

    def padded(self, tensor, outer_dims):
        original = tensor.contiguous()
        strides = list(original.stride())
        for axis in reversed(range(outer_dims)):
            strides[axis] = strides[axis + 1] * original.shape[axis + 1] + 7 * (axis + 1)
        size = 1 + sum((n - 1) * s for n, s in zip(original.shape, strides))
        backing = self.t.full((size,), 7, dtype=original.dtype, device=self.device)
        view = backing.as_strided(original.shape, strides).copy_(original)
        gaps = self.t.ones(size, dtype=self.t.bool, device=self.device)
        gaps.as_strided(original.shape, strides).fill_(False)
        return view, (backing, gaps)

    def check_padding(self, pads):
        for backing, gaps in pads:
            self.bits_equal(backing[gaps], self.t.full_like(backing[gaps], 7), "outer-stride padding overwritten")

    def test_runtime_dataclasses_are_frozen_and_policy_is_validated(self):
        self.assertEqual({member.name: member.value for member in self.api.HistoryFormat}, FORMATS)
        policy = self.api.NumericPolicy()
        self.assertEqual(tuple(getattr(policy, f) for f in FIELDS["NumericPolicy"]),
                         ("l2_inside_sqrt", 1e-6, "raw", "tf32"))
        d = self.case()
        b = self.verify_bundle(d)
        objects = [policy, b["inputs"], b["history"], b["mapping"], b["step"], self.api.MaterializeWork()]
        for obj in objects:
            self.assertEqual(tuple(f.name for f in fields(obj)), FIELDS[type(obj).__name__])
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, fields(obj)[0].name, None)
        for change in (dict(qk_norm="l2"), dict(gate_mode="sigmoid"),
                       dict(epsilon=-1), dict(epsilon=float("nan")), dict(epsilon=float("inf")),
                       dict(dot_precision="unknown")):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                self.api.NumericPolicy(**change)

    def test_verify_fp64_varlen_gqa_cache_footprint_and_readonly_checkpoint(self):
        for dtype in BUDGETS:
            for residual in (False, True):
                with self.subTest(dtype=dtype, residual=residual):
                    d = self.case(dtype=dtype, residual=residual, lengths=[4, 0, 2], layers=2)
                    if residual:
                        # Do not impose legacy high-only vs common residual-read semantics.
                        d["rawv_cache"].zero_()
                        d["rawk_cache"].zero_()
                    for layer in range(2):
                        self.check_verify(self.verify_bundle(d, layer=layer))

    def test_verify_gapped_layer_slot_and_replay_metadata_strides(self):
        d = self.case(layers=2, residual=True)
        d["rawv_cache"].zero_()
        d["rawk_cache"].zero_()
        pads = []
        for name in ("checkpoint_state", "d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache"):
            d[name], pad = self.padded(d[name], 2)
            pads.append(pad)
        for name in ("write_pos", "cache_base"):
            d[name] = self.ints(self.cpu(d[name]).tolist(), self.t.int32)
        for layer in range(2):
            self.check_verify(self.verify_bundle(d, layer=layer))
        self.check_padding(pads)

    def test_sglang_p6_t4_a3_materializes_nine_without_cursor_commit(self):
        for dtype in BUDGETS:
            d = self.case(dtype=dtype, history=6, residual=True)
            d["rawv_cache"].zero_()
            d["rawk_cache"].zero_()
            snapshots = self.check_verify(self.verify_bundle(d))
            # Caller acceptance, NOT materialize, owns the cursor transition.
            for replay in self.cpu(d["replay_indices"]).tolist():
                d["write_pos"][replay] = 6 + 3
                d["is_flush"][replay] = 1
            d["is_flush"][int(self.cpu(d["replay_indices"])[1])] = 0
            work = self.api.MaterializeWork(
                state_indices=d["ssm_state_indices"], replay_indices=d["replay_indices"],
                lengths=d["write_pos"], starts=d["cache_base"],
                metadata_indices=d["replay_indices"], enabled=d["is_flush"])
            self.assertIs(work.lengths, d["write_pos"])
            self.assertIs(work.starts, d["cache_base"])
            self.assertIs(work.metadata_indices, d["replay_indices"])
            self.assertIs(work.enabled, d["is_flush"])
            self.check_materialize(d["checkpoint_state"], self.history(d), work)
            for row in (0, 2):
                slot = int(self.cpu(d["ssm_state_indices"])[row])
                self.close(d["checkpoint_state"][0, slot], snapshots[row, 2], dtype)
            for replay in self.cpu(d["replay_indices"]).tolist():
                self.assertEqual(int(self.cpu(d["write_pos"])[replay]), 9)
                self.assertEqual(int(self.cpu(d["cache_base"])[replay]), 14)

    def test_holmes_p10_t8_l16_uses_old_buffer_not_new_a3(self):
        for dtype in BUDGETS:
            for integer in (self.t.int32, self.t.int64):
                with self.subTest(dtype=dtype, integer=integer):
                    d, h = self.mat_fixture("CACHED_UKG_DOUBLE", dtype=dtype)
                    old, new, pending, proposed, accepted = 1, 0, 10, 8, 3
                    self.assertGreater(pending + proposed, h.values.shape[3])
                    h.values[:, 5, new, :accepted].fill_(9)
                    h.keys[:, 5, new, :accepted].fill_(0.125)
                    h.gates[:, 5, new].fill_(-0.01)
                    items = self.t.tensor([[987654, 5, pending, old], [-77, 2, 8, new], [19, 7, 4, old]],
                                          dtype=integer, device=self.device)
                    work = self.api.MaterializeWork(items=items,
                                                     n_items=self.t.tensor([1] if integer == self.t.int32 else 1,
                                                                          dtype=integer, device=self.device))
                    before = self.cpu(d["checkpoint_state"])
                    expected = self.check_materialize(d["checkpoint_state"], h, work)
                    wrong = self.materialized(before[0, 5], h, 0, 5, accepted, buffer=new)
                    self.assertGreater(float((wrong - expected[0, 5]).abs().max()), 0.1)

    def test_materialize_three_protocols_fp64_indexed_subranges_and_tracking(self):
        for fmt in FORMATS:
            for dtype in BUDGETS:
                with self.subTest(format=fmt, dtype=dtype):
                    d, h = self.mat_fixture(fmt, dtype=dtype)
                    work = self.api.MaterializeWork(
                        state_indices=self.ints([5, 2, 7]), replay_indices=self.ints([4, 1, 6]),
                        lengths=self.ints([2, 3, 4, 1, 9]), starts=self.ints([1, 2, 0, 3, 7]),
                        buffers=self.ints([0, 1, 0, 1, 1] if fmt == "CACHED_UKG_DOUBLE" else [0] * 5),
                        metadata_indices=self.ints([4, 0, 2]), enabled=self.ints([1, 1, 0, 1, 1]),
                        target_state_indices=self.ints([8, 9, 10]))
                    self.check_materialize(d["checkpoint_state"], h, work)

    def test_compact_closed_form_agrees_with_existing_fp64_fixture_oracle(self):
        d = self.case(layers=2, residual=True, H=1, HV=2, K=128, V=129)
        cpu_data = {k: self.cpu(v) if isinstance(v, self.t.Tensor) else v for k, v in d.items()}
        h = self.history(d)
        for layer in range(2):
            for row, (slot, replay) in enumerate(zip(cpu_data["ssm_state_indices"].tolist(),
                                                     cpu_data["replay_indices"].tolist())):
                closed = self.materialized(cpu_data["checkpoint_state"][layer, slot], h, layer, replay, 3, 14)
                sequential = self.ref.history_state(cpu_data, layer, row, residual=True)
                self.t.testing.assert_close(closed, sequential, atol=1e-12, rtol=1e-12)
        work = self.api.MaterializeWork(state_indices=d["ssm_state_indices"], replay_indices=d["replay_indices"],
                                         lengths=d["write_pos"], starts=d["cache_base"],
                                         metadata_indices=d["replay_indices"])
        self.check_materialize(d["checkpoint_state"], h, work)

    def test_compact_bf16_high_low_cancellation_is_not_discarded(self):
        for component in ("values", "keys"):
            with self.subTest(component=component):
                d, h = self.mat_fixture("COMPACT_DKG_RING", dtype="bfloat16", B=1, H=1, HV=1)
                d["checkpoint_state"].zero_()
                for tensor in self.tensor_fields(h):
                    tensor.zero_()
                h.values[:, 4, 0, 0, 0] = 128
                h.values[:, 4, 0, 1, 0] = -128
                if component == "values":
                    h.value_residual[:, 4, 0, 0, 0] = 0.25  # BF16 high(128.25)=128, low=0.25.
                    h.keys[:, 4, 0, :2, 0] = 1
                    expected_first = 0.25
                else:
                    # Two unit keys with identical BF16 high parts. Only the source-
                    # quantized low parts survive cancellation by the +/-128 deltas.
                    keys = self.t.zeros((2, 32), dtype=self.t.float32, device="cpu")
                    keys[:, 0] = self.t.tensor([0.501, 0.5])
                    keys[:, 1] = (1 - keys[:, 0].square()).sqrt()
                    high = keys.to(self.t.bfloat16)
                    low = (keys - high.float()).to(self.t.bfloat16)
                    self.assertTrue(self.t.equal(high[0], high[1]))
                    h.keys[:, 4, 0, :2].copy_(high.to(self.device))
                    h.key_residual[:, 4, 0, :2].copy_(low.to(self.device))
                    expected_first = float(128 * (low[0, 0].float() - low[1, 0].float()))
                    self.assertGreater(abs(expected_first), 0.1)
                work = self.api.MaterializeWork(state_indices=self.ints([5]), replay_indices=self.ints([4]),
                                                 lengths=self.ints([2]))
                self.check_materialize(d["checkpoint_state"], h, work)
                self.close(d["checkpoint_state"][:, 5, 0, 0, 0],
                           self.t.full((2,), expected_first), "bfloat16")

    def test_step_g_and_cumulative_G_and_nonzero_origin_are_distinct(self):
        for fmt, start in (("COMPACT_DKG_RING", 0), ("CACHED_UKG_DOUBLE", 0),
                           ("CACHED_UKG_DOUBLE", 1)):
            with self.subTest(format=fmt, start=start):
                d, h = self.mat_fixture(fmt, B=1, H=1, HV=1, layers=1)
                d["checkpoint_state"].zero_()
                d["checkpoint_state"][0, 5, 0, 0, 0] = 4 if start else 2
                for tensor in self.tensor_fields(h):
                    tensor.zero_()
                if fmt == "COMPACT_DKG_RING":
                    h.values[0, 4, 0, :2, 0] = self.t.tensor([3., 5.], device=self.device)
                    h.keys[0, 4, 0, :2, 0] = 1
                    h.gates[0, 4, 0, :2] = self.t.tensor([math.log(0.5), math.log(0.25)], device=self.device)
                else:
                    h.values[0, 4, 1, :2, 0, 0] = self.t.tensor([3., 5.], device=self.device)
                    h.keys[0, 4, 1, :2, 0, 0] = 1
                    h.gates[0, 4, 1, 0, :2] = self.t.tensor([math.log(0.5), math.log(0.125)], device=self.device)
                work = self.api.MaterializeWork(state_indices=self.ints([5]), replay_indices=self.ints([4]),
                                                 lengths=self.ints([2 - start]), starts=self.ints([start]),
                                                 buffers=self.ints([int(fmt == "CACHED_UKG_DOUBLE")]))
                self.check_materialize(d["checkpoint_state"], h, work)
                self.close(d["checkpoint_state"][0, 5, 0, 0, 0], self.t.tensor(6.0))

    def scalar_bundle(self, *, q=1., k=1., state=2., value=3., gate=0., beta=0., policy=None):
        d = self.case(B=1, H=1, HV=1, T=1, history=0)
        for name in ("q", "k", "v", "checkpoint_state"):
            d[name].zero_()
        d["q"][0, 0, 0, 0], d["k"][0, 0, 0, 0] = q, k
        d["v"][0, 0, 0, 0], d["checkpoint_state"][0, 5, 0, 0, 0] = value, state
        d["a"].fill_(gate)
        d["b"].fill_(beta)
        bundle = self.verify_bundle(d, scale=1., policy=policy or self.api.NumericPolicy(
            qk_norm="none", gate_mode="processed", dot_precision="ieee"))
        bundle["inputs"] = replace(bundle["inputs"], A_log=None, dt_bias=None)
        return bundle

    def test_near_zero_q_and_k_keep_three_epsilon_placements(self):
        expected_q = dict(none=0.0008, l2_inside_sqrt=0.7960297521679914,
                          l2_outside_sqrt=7.920792079207921)
        for mode in expected_q:
            policy = self.api.NumericPolicy(qk_norm=mode, epsilon=1e-6,
                                            gate_mode="processed", dot_precision="ieee")
            with self.subTest(mode=mode, vector="q"):
                b = self.scalar_bundle(q=1e-4, state=8, beta=0, policy=policy)
                self.check_verify(b)
                self.close(b["out"][0, 0, 0], self.t.tensor(expected_q[mode]), kind="output")
            with self.subTest(mode=mode, vector="k"):
                b = self.scalar_bundle(k=1e-4, state=0, value=8, beta=1, policy=policy)
                self.check_verify(b)
                unit_q = 1 if mode == "none" else (1 / math.sqrt(1 + 1e-6)
                                                    if mode == "l2_inside_sqrt" else 1 / (1 + 1e-6))
                self.close(b["out"][0, 0, 0], self.t.tensor(expected_q[mode] * unit_q), kind="output")

    def test_raw_materialize_near_zero_keys_obey_numeric_policy(self):
        expected = dict(none=0.0032, l2_inside_sqrt=3.1841190086719656,
                        l2_outside_sqrt=31.683168316831683)
        for mode, value in expected.items():
            with self.subTest(mode=mode):
                d, h = self.mat_fixture("RAW_VKG_BETA_WINDOW", B=1, H=1, HV=1, layers=1)
                d["checkpoint_state"].zero_()
                for tensor in self.tensor_fields(h):
                    tensor.zero_()
                h.values[0, 4, 0, 2, 0] = 32
                h.keys[0, 4, 0, 2, 0] = 1e-4
                h.beta[0, 4, 0, 2] = 1  # History beta is already processed, even with gate_mode='raw'.
                work = self.api.MaterializeWork(state_indices=self.ints([5]), replay_indices=self.ints([4]),
                                                 lengths=self.ints([1]), starts=self.ints([2]))
                policy = self.api.NumericPolicy(qk_norm=mode, epsilon=1e-6, dot_precision="ieee")
                self.check_materialize(d["checkpoint_state"], h, work, policy=policy)
                self.close(d["checkpoint_state"][0, 5, 0, 0, 0], self.t.tensor(value))

    def test_processed_gates_and_beta_are_not_activated_again(self):
        for gate, beta, value, expected in ((math.log(0.5), 0., 999., 1.),
                                           (0., 1., 3., 3.), (math.log(0.5), 0.25, 3., 1.5)):
            with self.subTest(gate=gate, beta=beta):
                b = self.scalar_bundle(gate=gate, beta=beta, value=value)
                self.check_verify(b)
                self.close(b["out"][0, 0, 0], self.t.tensor(expected), kind="output")

    def test_common_ieee_projection_with_16bit_state_and_fp32_history(self):
        for state_dtype in (self.t.float16, self.t.bfloat16):
            for vector in ("q", "k"):
                b = self.scalar_bundle(state=1., value=0., beta=0. if vector == "q" else 1.)
                b["state"] = b["state"].to(state_dtype)
                tensor = getattr(b["inputs"], vector)
                tensor[0, 0, 0] = 1 + 3 * 2**-12
                with self.subTest(state_dtype=state_dtype, vector=vector):
                    self.check_verify(b)

    def test_raw_gate_pair_and_independent_activation(self):
        d = self.case(B=1, T=1, history=0)
        b = self.verify_bundle(d)
        self.check_verify(b)
        for field in ("A_log", "dt_bias"):
            invalid = dict(b, inputs=replace(b["inputs"], **{field: None}))
            self.assert_rejected_without_writes(lambda: self.launch_verify(invalid),
                                                invalid["inputs"], b["state"], b["history"], b["step"], b["out"])

    def test_decode_t1_equals_verify_t1_without_query_start_loc(self):
        for dtype in BUDGETS:
            for norm in ("none", "l2_inside_sqrt", "l2_outside_sqrt"):
                with self.subTest(dtype=dtype, norm=norm):
                    d1 = self.case(dtype=dtype, T=1)
                    d2 = self.case(dtype=dtype, T=1)
                    policy = self.api.NumericPolicy(qk_norm=norm, dot_precision="ieee")
                    # ReplayStep.starts remains the required replay-slot ring base.
                    # Only verify receives query_start_loc; decode must not synthesize it.
                    decode = self.verify_bundle(d1, policy=policy)
                    verify = self.verify_bundle(d2, policy=policy)
                    self.check_verify(decode, decode=True)
                    before = self.t.cuda.memory_stats(self.device)["allocation.all.allocated"]
                    self.launch_verify(decode, decode=True)
                    self.t.cuda.synchronize(self.device)
                    self.assertEqual(self.t.cuda.memory_stats(self.device)["allocation.all.allocated"], before,
                                     "T1 decode allocated synthetic query_start_loc")
                    self.check_verify(verify)
                    self.close(decode["out"], verify["out"], dtype, "output")
                    for a, b in zip(self.tensor_fields(decode["history"]), self.tensor_fields(verify["history"])):
                        self.close(a, b, dtype)

    def test_materialize_zero_work_and_packed_n_items_limit(self):
        for fmt in FORMATS:
            for integer in (self.t.int32, self.t.int64):
                d, h = self.mat_fixture(fmt)
                work = self.api.MaterializeWork(state_indices=self.ints([], integer),
                                                 replay_indices=self.ints([], integer), lengths=self.ints([], integer))
                self.check_materialize(d["checkpoint_state"], h, work)
                for capacity, n in ((0, 0), (3, 0), (3, 1), (3, None)):
                    with self.subTest(format=fmt, integer=integer, capacity=capacity, n=n):
                        items = self.t.tensor([[123, 5, 2, 0], [-3, 2, 3, 0], [7, 7, 4, 0]][:capacity],
                                              dtype=integer, device=self.device).reshape(capacity, 4)
                        work = self.api.MaterializeWork(items=items,
                                                         n_items=None if n is None else self.t.tensor(
                                                             n, dtype=integer, device=self.device),
                                                         target_state_indices=self.ints([8, 9, 10][:capacity], integer))
                        if n is None:
                            self.assert_rejected_without_writes(
                                lambda: self.api.gdn_replay_materialize(d["checkpoint_state"], h, work),
                                d["checkpoint_state"], h, work)
                        else:
                            self.check_materialize(d["checkpoint_state"], h, work)

    def test_materialize_zero_count_clone_inplace_and_null_zero_vs_minus_one(self):
        for fmt in FORMATS:
            for null in (-1, 0):
                for state_dtype in BUDGETS:
                    with self.subTest(format=fmt, null=null, state_dtype=state_dtype):
                        d, h = self.mat_fixture(fmt, dtype=state_dtype, state_dtype=state_dtype)
                        d["checkpoint_state"][:, 5, 0, 0, 0] = -0.0
                        d["checkpoint_state"][:, 2, 0, 0, 0] = -0.0
                        # Slot zero is a real source/target with null=-1, skipped with null=0.
                        work = self.api.MaterializeWork(
                            state_indices=self.ints([0, 5, 2, -1, 7]), replay_indices=self.ints([4, 1, 6, -1, 2]),
                            lengths=self.ints([2, 0, 0, 3, 2]), target_state_indices=self.ints([8, 9, 2, 10, -1]))
                        self.check_materialize(d["checkpoint_state"], h, work, null=null)
                        # Separate launch: target zero must not race the active source-zero row above.
                        to_zero = self.api.MaterializeWork(
                            state_indices=self.ints([7]), replay_indices=self.ints([2]),
                            lengths=self.ints([2]), target_state_indices=self.ints([0]))
                        self.check_materialize(d["checkpoint_state"], h, to_zero, null=null)

    def test_verify_slot_zero_and_minus_one_null_mapping(self):
        for null in (-1, 0):
            d = self.case(B=3, T=1, history=0)
            d["null_block_id"] = null
            d["ssm_state_indices"].copy_(self.ints([0, 5, -1]))
            d["replay_indices"].copy_(self.ints([0, 4, -1]))
            self.check_verify(self.verify_bundle(d))
        d = self.case(B=1, T=1, history=0)
        d["null_block_id"] = 0
        d["ssm_state_indices"].fill_(5)
        d["replay_indices"].zero_()
        self.check_verify(self.verify_bundle(d))

    def test_history_buffer_layer_slot_strides_and_padding(self):
        for fmt in FORMATS:
            d, h = self.mat_fixture(fmt, layers=3)
            state, pad = self.padded(d["checkpoint_state"], 2)
            pads, changes = [pad], {}
            for f in fields(h):
                value = getattr(h, f.name)
                if isinstance(value, self.t.Tensor):
                    changes[f.name], pad = self.padded(value, 3 if fmt == "CACHED_UKG_DOUBLE" else 2)
                    pads.append(pad)
            h = replace(h, **changes)
            work = self.api.MaterializeWork(state_indices=self.ints([5, 2]), replay_indices=self.ints([4, 1]),
                                             lengths=self.ints([5, 4]), starts=self.ints([9, 3]),
                                             buffers=self.ints([1, 0] if fmt == "CACHED_UKG_DOUBLE" else [0, 0]))
            self.check_materialize(state, h, work)
            self.check_padding(pads)

    def test_descriptor_independent_noncontiguous_layers_and_template_readonly(self):
        for fmt in FORMATS:
            d, h = self.mat_fixture(fmt, layers=3)
            layers, pads = [], []
            for layer in d["checkpoint_state"]:
                view, pad = self.padded(layer, 1)
                layers.append(view)
                pads.append(pad)
            # A separate template must not be mistaken for layer zero's storage.
            template = self.t.full_like(layers[0], -19)
            template_before = self.cpu(template)
            descriptor_cpu = self.t.tensor([[layer.data_ptr(), layer.stride(0)] for layer in layers],
                                           dtype=self.t.int64, device="cpu")
            descriptors = descriptor_cpu.to(self.device)  # caller allocation/copy, before launch
            work = self.api.MaterializeWork(state_indices=self.ints([5, 2]), replay_indices=self.ints([4, 1]),
                                             lengths=self.ints([4, 0]), starts=self.ints([2, 0]),
                                             target_state_indices=self.ints([8, 9]))
            self.check_materialize(template, h, work, descriptors=descriptors, actual_layers=layers)
            graph = self.t.cuda.CUDAGraph()
            stream = self.t.cuda.Stream(device=self.device)
            stream.wait_stream(self.t.cuda.current_stream(self.device))
            with self.t.cuda.graph(graph, stream=stream):
                self.api.gdn_replay_materialize(template, h, work, state_descriptors=descriptors)
            for layer in layers:
                layer[8:10].fill_(-31)
            self.check_materialize(template, h, work, descriptors=descriptors,
                                   actual_layers=layers, launch=graph.replay)
            self.bits_equal(template, template_before, "descriptor shape template was written")
            self.check_padding(pads)
            # 'layers' deliberately remains strongly referenced through synchronization.

    def test_preallocated_workspace_and_none_generic_both_match_fp64(self):
        for dtype in BUDGETS:
            outputs = []
            for preallocated in (False, True):
                d = self.case(B=128, H=1, HV=2, K=128, V=128, dtype=dtype)
                b = self.verify_bundle(d, policy=self.api.NumericPolicy())
                readonly = self.snapshots(b["inputs"], b["state"], b["history"], b["mapping"])
                n = self.api.gdn_replay_workspace_elements(b["inputs"], b["state"], b["history"], b["mapping"], 4)
                self.assertIs(type(n), int)
                self.assertGreaterEqual(n, 0)
                if dtype == "bfloat16":
                    self.assertEqual(n, 128 * 2 * 4 * 4 * 128)
                self.unchanged(readonly)
                if preallocated:
                    b["workspace"] = self.t.empty(n, dtype=self.t.float32, device=self.device)
                self.check_verify(b)
                allocated = self.t.cuda.memory_stats(self.device)["allocation.all.allocated"]
                self.launch_verify(b)
                self.t.cuda.synchronize(self.device)
                self.assertEqual(self.t.cuda.memory_stats(self.device)["allocation.all.allocated"], allocated)
                if preallocated and n:
                    stream = self.t.cuda.Stream(device=self.device)
                    stream.wait_stream(self.t.cuda.current_stream(self.device))
                    graph = self.t.cuda.CUDAGraph()
                    with self.t.cuda.graph(graph, stream=stream):
                        self.launch_verify(b)
                    self.check_verify(b, launch=graph.replay)
                    invalid = dict(b, workspace=b["workspace"][:-1])
                    self.assert_rejected_without_writes(lambda: self.launch_verify(invalid),
                                                        b["state"], b["history"], b["step"], b["out"])
                outputs.append(self.cpu(b["out"]))
            self.close(outputs[0], outputs[1], dtype, "output")

    def test_workspace_dispatch_uses_head_work_not_batch_alone(self):
        counts = []
        for batch, heads, value_heads in ((8, 16, 32), (32, 4, 8), (128, 1, 2)):
            d = self.case(B=batch, H=heads, HV=value_heads, K=128, V=128, dtype="bfloat16")
            b = self.verify_bundle(d)
            count = self.api.gdn_replay_workspace_elements(b["inputs"], b["state"], b["history"], b["mapping"], 4)
            self.assertEqual(count, 256 * 4 * 4 * 128)
            b["workspace"] = self.t.empty(count, dtype=self.t.float32, device=self.device)
            self.check_verify(b)
            counts.append(count)
        self.assertEqual(len(set(counts)), 1)

    def test_workspace_dispatch_threshold_tracks_draft_width(self):
        for steps, below, at in ((6, 63, 64), (4, 191, 192), (2, 95, 96), (1, 95, 96)):
            for batch, expected in ((below, 0), (at, at * 4 * steps * 128)):
                with self.subTest(T=steps, B=batch):
                    d = self.case(B=batch, H=1, HV=1, K=128, V=128, T=steps, dtype="bfloat16")
                    b = self.verify_bundle(d)
                    count = self.api.gdn_replay_workspace_elements(
                        b["inputs"], b["state"], b["history"], b["mapping"], steps)
                    self.assertEqual(count, expected)
                    if count:
                        b["workspace"] = self.t.empty(count, dtype=self.t.float32, device=self.device)
                    self.check_verify(b)

    def test_zero_history_with_only_eight_byte_aligned_key(self):
        d = self.case(B=1, H=1, HV=1, K=128, V=128, T=4,
                      history=0, dtype="bfloat16", strided=False)
        b = self.verify_bundle(d)
        original = b["inputs"].k
        backing = self.t.full((original.numel() + 4,), -13, dtype=original.dtype, device=self.device)
        key = backing[4:].view(original.shape).copy_(original)
        self.assertEqual(key.data_ptr() % 16, 8)
        b["inputs"] = replace(b["inputs"], k=key)
        self.check_verify(b)
        self.assertTrue(self.t.equal(backing[:4], self.t.full_like(backing[:4], -13)))

    def test_zero_history_with_only_eight_byte_aligned_key_rows(self):
        d = self.case(B=2, H=1, HV=1, K=128, V=128, T=4, lengths=[1, 1],
                      history=0, dtype="bfloat16", strided=False)
        b = self.verify_bundle(d)
        original = b["inputs"].k
        backing = self.t.full((2, 132), -13, dtype=original.dtype, device=self.device)
        key = backing[:, :128].unsqueeze(1).copy_(original)
        self.assertEqual(key.stride(), (132, 128, 1))
        self.assertEqual(key.data_ptr() % 16, 0)
        self.assertEqual(key[1].data_ptr() % 16, 8)
        b["inputs"] = replace(b["inputs"], k=key)
        self.check_verify(b)
        self.assertTrue(self.t.equal(backing[:, 128:], self.t.full_like(backing[:, 128:], -13)))

    def test_eight_lane_t8_verify_with_ragged_tokens(self):
        for history, lengths in ((0, [8, 3]), (7, [8, 3]), (8, [8, 3]), (15, [1, 1]),
                                 (8, [1, 2]), (8, [4, 5]), (8, [6, 7])):
            for residual in (False, True):
                with self.subTest(history=history, lengths=lengths, residual=residual):
                    d = self.case(B=2, H=2, HV=8, K=128, V=128, T=8, lengths=lengths,
                                  history=0, residual=residual, dtype="bfloat16", strided=False)
                    d["write_pos"].fill_(history)
                    self.check_verify(self.verify_bundle(d))
        for lengths, null_row in (([0, 8], False), ([8, 0], False), ([0, 0], False), ([8, 3], True)):
            with self.subTest(lengths=lengths, null_row=null_row):
                d = self.case(B=2, H=2, HV=8, K=128, V=128, T=8, lengths=lengths,
                              history=0, dtype="bfloat16", strided=False)
                if null_row:
                    d["ssm_state_indices"][1] = -1
                self.check_verify(self.verify_bundle(d))

    def test_tiny_t4_verify_with_wrapped_history(self):
        for history, length in ((0, 4), (8, 4), (12, 4), (15, 1)):
            for residual in (False, True):
                for dot_precision in ("ieee", "tf32"):
                    with self.subTest(history=history, length=length, residual=residual,
                                      dot_precision=dot_precision):
                        d = self.case(B=1, H=4, HV=12, K=128, V=128, T=4, lengths=[length],
                                      history=0, residual=residual, dtype="bfloat16", strided=False)
                        d["write_pos"].fill_(history)
                        policy = self.api.NumericPolicy(dot_precision=dot_precision)
                        self.check_verify(self.verify_bundle(d, policy=policy))

    def test_t4_verify_parallel_dispatch_boundary(self):
        for heads in (32, 64, 96, 128):
            for history, length in ((0, 4), (12, 4), (15, 1)):
                for dot_precision in ("ieee", "tf32"):
                    with self.subTest(heads=heads, history=history, length=length, dot_precision=dot_precision):
                        d = self.case(B=1, H=heads // 8, HV=heads, K=128, V=128, T=4,
                                      lengths=[length], history=0, residual=True, dtype="bfloat16", strided=False)
                        for name in ("ssm_state_indices", "replay_indices", "write_pos", "cache_base", "query_start_loc"):
                            d[name] = d[name].to(self.t.int32)
                        d["a"], d["b"] = d["a"].bfloat16(), d["b"].bfloat16()
                        d["write_pos"].fill_(history)
                        policy = self.api.NumericPolicy(dot_precision=dot_precision)
                        self.check_verify(self.verify_bundle(d, policy=policy))

    def test_empty_tokens_on_native_fast_paths(self):
        for batch in (1, 192):
            with self.subTest(B=batch):
                d = self.case(B=batch, H=1, HV=1, K=128, V=128, T=4,
                              history=0, dtype="bfloat16", strided=False, lengths=[0] * batch)
                b = self.verify_bundle(d)
                count = self.api.gdn_replay_workspace_elements(
                    b["inputs"], b["state"], b["history"], b["mapping"], 4)
                if count:
                    b["workspace"] = self.t.empty(count, dtype=self.t.float32, device=self.device)
                self.check_verify(b)

    def test_compact_materialize_preserves_tiny_key_residual_and_decay_suffix(self):
        for dtype in ("bfloat16", "float32"):
            for fast in (False, True):
                for problem in ("residual", "decay", "merge_overflow"):
                    with self.subTest(dtype=dtype, fast=fast, problem=problem):
                        d, h = self.mat_fixture("COMPACT_DKG_RING", dtype=dtype,
                            layers=1, B=1, H=1, HV=1, K=128, V=128, strided=False)
                        d["checkpoint_state"].zero_()
                        for tensor in (h.values, h.keys, h.gates, h.value_residual, h.key_residual):
                            tensor.zero_()
                        h.keys[0, 4, 0, :2].fill_(1)
                        if problem == "residual":
                            h.values[0, 4, 0, 0].fill_(2 ** 24)
                            h.values[0, 4, 0, 1].fill_(-(2 ** 24))
                            h.key_residual[0, 4, 0, 0].fill_(2 ** -24)
                        elif problem == "merge_overflow":
                            h.keys[0, 4, 0].zero_()
                            h.keys[0, 4, 0, 0].fill_(float(2 ** 127))
                            h.key_residual[0, 4, 0, 0].fill_(float(2 ** 127))
                            h.values[0, 4, 0, 0].fill_(0.25)
                        else:
                            h.values[0, 4, 0, 0].fill_(1)
                            h.gates[0, 4, 0, 0] = -(2 ** 24)
                            h.gates[0, 4, 0, 1] = -1
                        replays = self.ints([4])
                        if fast:
                            count = self.t.zeros(h.values.shape[1], dtype=self.t.int64, device=self.device)
                            count[4] = 2
                            work = self.api.MaterializeWork(state_indices=self.ints([5]),
                                replay_indices=replays, metadata_indices=replays, lengths=count,
                                starts=self.t.zeros_like(count))
                        else:
                            work = self.api.MaterializeWork(state_indices=self.ints([5]),
                                replay_indices=replays, lengths=self.ints([2]))
                        self.check_materialize(d["checkpoint_state"], h, work)

    def test_compact_fast_fold_at_wide_dispatch_boundary(self):
        for batch in (63, 64, 65):
            for index_dtype in (self.t.int32, self.t.int64):
                for residual in (False, True):
                    for flag_dtype in (None, self.t.int8, self.t.int32, self.t.int64):
                        with self.subTest(batch=batch, index_dtype=index_dtype, residual=residual, flag_dtype=flag_dtype):
                            d, history = self.mat_fixture("COMPACT_DKG_RING", dtype="bfloat16",
                                layers=1, B=batch, H=4, HV=16, K=128, V=128, strided=False)
                            if not residual:
                                history = replace(history, value_residual=None, key_residual=None)
                            replay = d["replay_indices"].to(index_dtype)
                            lengths = self.t.zeros(history.values.shape[1], dtype=index_dtype, device=self.device)
                            starts = self.t.zeros_like(lengths)
                            lengths[replay.long()] = self.t.tensor(
                                [(row * 7) % 17 for row in range(batch)], dtype=index_dtype, device=self.device)
                            starts[replay.long()] = self.t.tensor(
                                [(row * 3 + 15) % 16 for row in range(batch)], dtype=index_dtype, device=self.device)
                            enabled = None
                            if flag_dtype is not None:
                                enabled = self.t.ones_like(lengths, dtype=flag_dtype)
                                enabled[replay[::5].long()] = 0
                            work = self.api.MaterializeWork(state_indices=d["ssm_state_indices"].to(index_dtype),
                                replay_indices=replay, metadata_indices=replay, lengths=lengths, starts=starts,
                                enabled=enabled)
                            self.check_materialize(d["checkpoint_state"], history, work)

    def test_workspace_aliases_rejected_before_any_write(self):
        for alias in ("state", "out", "q", "history"):
            d = self.case(B=128, H=1, HV=2, K=128, V=64)
            b = self.verify_bundle(d)
            n = self.api.gdn_replay_workspace_elements(b["inputs"], b["state"], b["history"], b["mapping"], 4)
            original = (b[alias] if alias in ("state", "out") else
                        b["inputs"].q if alias == "q" else b["history"].values)
            backing = self.t.empty(max(n, original.numel(), 1), dtype=self.t.float32, device=self.device)
            view = backing[:original.numel()].view(original.shape).copy_(original)
            if alias in ("state", "out"):
                b[alias] = view
            elif alias == "q":
                b["inputs"] = replace(b["inputs"], q=view)
            else:
                b["history"] = replace(b["history"], values=view)
            b["workspace"] = backing[:max(n, 1)]
            with self.subTest(alias=alias):
                self.assert_rejected_without_writes(lambda: self.launch_verify(b), b["inputs"], b["state"],
                                                    b["out"], b["history"], b["mapping"], b["step"])

    def test_verify_generic_and_indexed_materialize_are_graph_safe_without_allocations(self):
        d = self.case(B=4, history=3)
        b = self.verify_bundle(d)
        self.check_verify(b)  # warm extension/dispatch; workspace remains None
        stream = self.t.cuda.Stream(device=self.device)
        stream.wait_stream(self.t.cuda.current_stream(self.device))
        with self.t.cuda.stream(stream):
            self.launch_verify(b)
        self.t.cuda.current_stream(self.device).wait_stream(stream)
        self.t.cuda.synchronize(self.device)
        before = self.t.cuda.memory_stats(self.device)["allocation.all.allocated"]
        self.launch_verify(b)
        self.t.cuda.synchronize(self.device)
        self.assertEqual(self.t.cuda.memory_stats(self.device)["allocation.all.allocated"], before,
                         "workspace=None generic hot path allocated a torch tensor")
        graph = self.t.cuda.CUDAGraph()
        with self.t.cuda.graph(graph, stream=stream):
            self.launch_verify(b)
        for _ in range(2):
            b["out"].fill_(17)
            self.check_verify(b, launch=graph.replay)
        work = self.api.MaterializeWork(state_indices=d["ssm_state_indices"], replay_indices=d["replay_indices"],
                                         lengths=d["write_pos"], starts=d["cache_base"],
                                         metadata_indices=d["replay_indices"],
                                         enabled=self.ints([1] * d["write_pos"].numel()),
                                         target_state_indices=self.ints([0, 1, 3, 4]))
        history = self.history(d)
        state = d["checkpoint_state"]
        self.check_materialize(state, history, work)
        launch = lambda: self.api.gdn_replay_materialize(state, history, work)
        before = self.t.cuda.memory_stats(self.device)["allocation.all.allocated"]
        launch()
        self.t.cuda.synchronize(self.device)
        self.assertEqual(self.t.cuda.memory_stats(self.device)["allocation.all.allocated"], before,
                         "indexed metadata was gathered/copied in the hot path")
        materialize_graph = self.t.cuda.CUDAGraph()
        with self.t.cuda.graph(materialize_graph, stream=stream):
            launch()
        for target in (0, 1, 3, 4):
            state[:, target].fill_(-31)
        self.check_materialize(state, history, work, launch=materialize_graph.replay)

    def test_packed_n_items_remains_device_dynamic_during_graph_replay(self):
        d, h = self.mat_fixture("CACHED_UKG_DOUBLE")
        state = d["checkpoint_state"]
        work = self.api.MaterializeWork(
            items=self.t.tensor([[9, 5, 3, 1], [-7, 2, 4, 0], [2, 7, 2, 1]],
                                dtype=self.t.int32, device=self.device),
            n_items=self.t.tensor(1, dtype=self.t.int32, device=self.device),
            target_state_indices=self.ints([8, 9, 10]))
        self.check_materialize(state, h, work)
        graph = self.t.cuda.CUDAGraph()
        stream = self.t.cuda.Stream(device=self.device)
        stream.wait_stream(self.t.cuda.current_stream(self.device))
        with self.t.cuda.graph(graph, stream=stream):
            self.api.gdn_replay_materialize(state, h, work)
        for n in (0, 2, 1):
            work.n_items.fill_(n)  # caller change outside capture/launch, not an API cursor commit
            state[:, 8:11].fill_(-23 - n)  # fresh canaries also detect a stale larger row count
            self.check_materialize(state, h, work, launch=graph.replay)

    def test_verify_rejects_raw_history(self):
        d = self.case()
        b = self.verify_bundle(d)
        for fmt in ("RAW_VKG_BETA_WINDOW",):
            _, h = self.mat_fixture(fmt, layers=1)
            h = replace(h, **{f.name: getattr(h, f.name)[0] for f in fields(h)
                              if isinstance(getattr(h, f.name), self.t.Tensor)})
            invalid = dict(b, history=h)
            for decode in (False, True):
                if decode:
                    one = self.case(T=1)
                    invalid = dict(self.verify_bundle(one), history=h)
                with self.subTest(format=fmt, decode=decode):
                    self.assert_rejected_without_writes(lambda: self.launch_verify(invalid, decode=decode),
                                                        invalid["state"], h, invalid["step"], invalid["out"])

    def test_verify_rejects_structural_lengths_format_and_workspace_errors(self):
        d = self.case()
        base = self.verify_bundle(d)
        n = self.api.gdn_replay_workspace_elements(base["inputs"], base["state"], base["history"], base["mapping"], 4)
        invalids = [
            dict(base, inputs=replace(base["inputs"], q=base["inputs"].q.unsqueeze(0))),
            dict(base, step=replace(base["step"], lengths=d["write_pos"].float())),
            dict(base, step=replace(base["step"], lengths=d["write_pos"].unsqueeze(0))),
            dict(base, step=replace(base["step"], starts=d["cache_base"][:-1])),
            dict(base, history=replace(base["history"], value_residual=base["history"].values)),
            dict(base, history=replace(base["history"], beta=base["history"].gates)),
            dict(base, history=replace(base["history"], format="not_a_history_format")),
            dict(base, query_start_loc=d["query_start_loc"][:-1]),
            dict(base, max_verify_len=0),
            dict(base, workspace=self.t.empty(n, dtype=self.t.float64, device=self.device)),
            dict(base, workspace=self.t.empty((1, n), dtype=self.t.float32, device=self.device)),
        ]
        if n:
            invalids.append(dict(base, workspace=self.t.empty(n - 1, dtype=self.t.float32, device=self.device)))
        for index, b in enumerate(invalids):
            with self.subTest(invalid=index):
                self.assert_rejected_without_writes(lambda: self.launch_verify(b), b["inputs"], b["state"],
                                                    b["history"], b["mapping"], b["step"], b["out"])

    def test_materialize_rejects_format_rank_residual_and_metadata_errors_early(self):
        for fmt in FORMATS:
            d, h = self.mat_fixture(fmt)
            state = d["checkpoint_state"]
            work = self.api.MaterializeWork(state_indices=self.ints([5, 2]), replay_indices=self.ints([4, 1]),
                                             lengths=self.ints([3, 2]))
            invalid_histories = [replace(h, format="not_a_history_format"),
                                 replace(h, values=h.values.unsqueeze(0)),
                                 replace(h, gates=h.gates.unsqueeze(0))]
            if fmt == "COMPACT_DKG_RING":
                invalid_histories += [replace(h, key_residual=None), replace(h, beta=h.gates)]
            elif fmt == "RAW_VKG_BETA_WINDOW":
                invalid_histories += [replace(h, beta=None), replace(h, value_residual=h.values,
                                                                      key_residual=h.keys)]
            else:
                invalid_histories += [replace(h, beta=h.gates), replace(h, value_residual=h.values,
                                                                        key_residual=h.keys),
                                      replace(h, values=h.values[:, :, :1]),
                                      replace(h, gates=h.gates.transpose(-1, -2).contiguous())]
            for index, invalid in enumerate(invalid_histories):
                with self.subTest(format=fmt, history_error=index):
                    self.assert_rejected_without_writes(
                        lambda: self.api.gdn_replay_materialize(state, invalid, work), state, invalid, work)
            invalid_work = [replace(work, lengths=None), replace(work, lengths=work.lengths.float()),
                            replace(work, lengths=work.lengths.unsqueeze(0)),
                            replace(work, starts=self.ints([0])), replace(work, buffers=self.ints([0])),
                            replace(work, enabled=self.ints([1])), replace(work, metadata_indices=self.ints([0])),
                            replace(work, replay_indices=self.ints([1])),
                            replace(work, target_state_indices=self.ints([8]))]
            for index, invalid in enumerate(invalid_work):
                with self.subTest(format=fmt, work_error=index):
                    self.assert_rejected_without_writes(
                        lambda: self.api.gdn_replay_materialize(state, h, invalid), state, h, invalid)

    def test_packed_work_cannot_mix_indexed_fields_and_requires_gpu_integer_count(self):
        d, h = self.mat_fixture("CACHED_UKG_DOUBLE")
        state = d["checkpoint_state"]
        items = self.t.tensor([[17, 5, 2, 1]], dtype=self.t.int64, device=self.device)
        packed = self.api.MaterializeWork(items=items, n_items=self.t.tensor(1, dtype=self.t.int64, device=self.device))
        invalids = [replace(packed, items=items[:, :3]), replace(packed, items=items.float()),
                    replace(packed, n_items=self.t.tensor(1, dtype=self.t.int64, device="cpu")),
                    replace(packed, n_items=self.t.tensor(1., device=self.device)),
                    replace(packed, n_items=self.t.ones((1, 1), dtype=self.t.int32, device=self.device))]
        for name in ("state_indices", "replay_indices", "lengths", "starts", "buffers", "metadata_indices", "enabled"):
            invalids.append(replace(packed, **{name: self.ints([1])}))
        for index, invalid in enumerate(invalids):
            with self.subTest(invalid=index):
                self.assert_rejected_without_writes(
                    lambda: self.api.gdn_replay_materialize(state, h, invalid), state, h, invalid)

    def test_descriptor_layer_count_rank_dtype_errors_rejected_early(self):
        d, h = self.mat_fixture("COMPACT_DKG_RING", layers=2)
        layers = list(d["checkpoint_state"].unbind(0))
        good = self.t.tensor([[layer.data_ptr(), layer.stride(0)] for layer in layers],
                             dtype=self.t.int64, device="cpu").to(self.device)
        work = self.api.MaterializeWork(state_indices=self.ints([5]), replay_indices=self.ints([4]), lengths=self.ints([2]))
        for descriptors in (good[:1], good.float(), good.unsqueeze(0), good[:, :1], good.cpu()):
            self.assert_rejected_without_writes(
                lambda: self.api.gdn_replay_materialize(layers[0], h, work, state_descriptors=descriptors),
                *layers, h, work, descriptors)

    def test_materialize_rejects_device_count_and_window_bounds(self):
        cases = [f"{fmt}:{problem}" for fmt in FORMATS for problem in ("negative", "overflow")]
        cases += ["RAW_VKG_BETA_WINDOW:window", "CACHED_UKG_DOUBLE:window",
                  "CACHED_UKG_DOUBLE:buffer", "CACHED_UKG_DOUBLE:n_negative", "CACHED_UKG_DOUBLE:n_overflow",
                  "VERIFY:length_small", "VERIFY:start_small", "VERIFY:length_prepared", "VERIFY:start_prepared"]
        for case in cases:
            with self.subTest(case=case):
                child = subprocess.run(
                    [sys.executable, "-B", str(Path(__file__).resolve()), "--device", str(self.device),
                     "--range-case", case], capture_output=True, text=True, timeout=180, check=False)
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
                reports = [json.loads(line) for line in child.stdout.splitlines() if line.startswith('{"suite":')]
                self.assertEqual(len(reports), 1, child.stdout)
                self.assertEqual(reports[0]["status"], "PASSED")
                self.assertEqual(reports[0]["passed_test_methods"], 1)

    def _isolated_invalid_range(self):
        fmt, problem = OPTIONS.range_case.split(":")
        if fmt == "VERIFY":
            batch = 192 if problem.endswith("prepared") else 1
            d = self.case(B=batch, H=1, HV=1, K=128, V=128, T=4,
                          dtype="bfloat16", strided=False)
            b = self.verify_bundle(d)
            lengths = b["step"].lengths.to(self.t.int64)
            starts = b["step"].starts.to(self.t.int64)
            (lengths if problem.startswith("length") else starts).fill_(2 ** 32)
            b["step"] = self.api.ReplayStep(lengths, starts)
            count = self.api.gdn_replay_workspace_elements(
                b["inputs"], b["state"], b["history"], b["mapping"], 4)
            if count:
                b["workspace"] = self.t.empty(count, dtype=self.t.float32, device=self.device)
            with self.assertRaises((ValueError, RuntimeError)):
                self.launch_verify(b)
                self.t.cuda.synchronize(self.device)
            return
        d, h = self.mat_fixture(fmt, layers=1)
        capacity = 16
        count = -1 if problem == "negative" else capacity + 1 if problem == "overflow" else 2
        work = self.api.MaterializeWork(state_indices=self.ints([5]), replay_indices=self.ints([4]),
                                         lengths=self.ints([count]), starts=self.ints([15 if problem == "window" else 0]),
                                         buffers=self.ints([2 if problem == "buffer" else 0]))
        if problem.startswith("n_"):
            work = self.api.MaterializeWork(
                items=self.t.tensor([[0, 5, 2, 0]], dtype=self.t.int32, device=self.device),
                n_items=self.t.tensor(-1 if problem == "n_negative" else 2, dtype=self.t.int32, device=self.device))
        # No GPU access after an expected asynchronous assertion: this process exits.
        with self.assertRaises((ValueError, RuntimeError)):
            self.api.gdn_replay_materialize(d["checkpoint_state"], h, work)
            self.t.cuda.synchronize(self.device)


class CountingResult(unittest.TextTestResult):
    passed_test_methods = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_test_methods += 1


def not_run(reason):
    print(json.dumps(dict(suite="device", status="NOT_RUN", reason=reason,
                          tests_run=0, passed_test_methods=0, numerical="NOT_RUN")))
    return 2


def main(argv=None):
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--static", action="store_true", help="AST/contract constants only; never import torch")
    parser.add_argument("--device", help="explicit cuda:N; no implicit GPU or CPU fallback")
    parser.add_argument("--test", action="append", help="select exact test method name (repeatable)")
    parser.add_argument("--range-case", help=argparse.SUPPRESS)
    OPTIONS = parser.parse_args(argv)
    if OPTIONS.static and (OPTIONS.device or OPTIONS.range_case):
        parser.error("--static cannot select a device or a numerical range case")
    if OPTIONS.range_case and OPTIONS.test:
        parser.error("--range-case and --test cannot be combined")
    if OPTIONS.static:
        classes = [StaticContracts]
    else:
        if OPTIONS.device is None:
            return not_run("explicit --device cuda:N is required")
        if re.fullmatch(r"cuda:[0-9]+", OPTIONS.device) is None:
            parser.error("--device must explicitly name cuda:N")
        if importlib.util.find_spec("torch") is None:
            return not_run("torch is not installed")
        t = importlib.import_module("torch")
        if not t.cuda.is_available():
            return not_run("CUDA is not available")
        if int(OPTIONS.device.split(":")[1]) >= t.cuda.device_count():
            return not_run(f"selected device {OPTIONS.device} does not exist")
        classes = [DeviceContracts]
    if OPTIONS.range_case:
        parts = OPTIONS.range_case.split(":")
        if len(parts) != 2 or parts[0] not in (*FORMATS, "VERIFY") or parts[1] not in {
            "negative", "overflow", "window", "buffer", "n_negative", "n_overflow",
            "length_small", "start_small", "length_prepared", "start_prepared",
        }:
            parser.error("invalid internal range case")
        suite = unittest.TestSuite([DeviceContracts("_isolated_invalid_range")])
    elif OPTIONS.test:
        tests = []
        for name in OPTIONS.test:
            matches = [cls for cls in classes if name in unittest.defaultTestLoader.getTestCaseNames(cls)]
            if len(matches) != 1:
                parser.error(f"selected test missing or ambiguous: {name}")
            tests.append(matches[0](name))
        suite = unittest.TestSuite(tests)
    else:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
    result = unittest.TextTestRunner(verbosity=2, resultclass=CountingResult).run(suite)
    failures = len(result.errors) + len(result.failures)
    status = "FAILED" if failures else "NOT_RUN" if result.skipped or not result.testsRun else "PASSED"
    if OPTIONS.static and "torch" in sys.modules:
        status = "FAILED"
        failures += 1
        print("static mode unexpectedly imported torch", file=sys.stderr)
    print(json.dumps(dict(suite="static" if OPTIONS.static else "device", status=status,
                          tests_run=result.testsRun, passed_test_methods=result.passed_test_methods,
                          skipped_not_passed=len(result.skipped), failures=failures,
                          numerical="NOT_RUN" if OPTIONS.static else status, tolerances=BUDGETS)))
    return 1 if failures else 2 if status == "NOT_RUN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
