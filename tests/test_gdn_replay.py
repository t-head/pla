# SPDX-License-Identifier: Apache-2.0
"""Compact Replay API tests: adjacent upstream cases + NEW independent FP64 oracle.

SGLang @40a64e1 has no direct numerical UT for these three public functions.
This is NOT a complete extraction. Fold64 slots/acceptance/strides/null/tracking/
256 rounds and the two GQA128 configurations are adapted from adjacent tests.
Raw-fold's 2*FP32-eps tolerance is intentionally NOT claimed for compact math.

Offline: python -B tests/test_gdn_replay.py --suite static [--sglang-root PATH]
CPU:     python -B tests/test_gdn_replay.py --suite oracle --device cpu
Device:  python -B tests/test_gdn_replay.py --suite device --device cuda:0
         [--sglang-root PATH]  # original-file differential; otherwise NOT_RUN
No implicit GPU selection. Missing dependencies and skips are not counted as passes.
"""
import argparse
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "decode" / "gdn_replayssm"
sys.path.insert(0, str(SCRIPTS))
from reference import (  # noqa: E402
    ADJACENT_TESTS, Case, MUTATED, PUBLIC_FUNCTIONS, SOURCE_COMMIT, SOURCE_SHA256,
    TOLERANCES, UPSTREAM_CASES, clone_tree, compact_kwargs, cursor_kwargs,
    decode_kwargs, history_state, make_case, normalized, oracle_compact,
    oracle_cursor, oracle_decode, recurrent_step, restore_mutated,
    snapshot_mutated, strided_int, torch_module,
)
from upstream import check_source, load_upstream  # noqa: E402

OPTIONS = argparse.Namespace(device=None, sglang_root=None)


class StaticTests(unittest.TestCase):
    def test_ast_no_torch_needed(self):
        files = (Path(__file__).resolve(), SCRIPTS / "reference.py", SCRIPTS / "upstream.py")
        for path in files:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
            self.assertIn("SPDX-License-Identifier: Apache-2.0", source)

    def test_provenance_and_fixed_budgets(self):
        self.assertEqual(SOURCE_COMMIT, "40a64e1b5ba00c73b7f8a123a903a2123b7d467c")
        self.assertEqual(len(SOURCE_SHA256), 64)
        self.assertEqual(len(ADJACENT_TESTS), 2)
        self.assertEqual(TOLERANCES["float32"]["output"], (1e-4, 1e-3))
        self.assertEqual(TOLERANCES["bfloat16"]["state"], (3e-3, 1e-2))
        self.assertEqual(TOLERANCES["float16"]["output"], (5e-4, 3e-3))
        for config in UPSTREAM_CASES.values():
            Case(**config).validate()

    def test_cursor_equations_without_torch(self):
        # Includes zero acceptance on a current flush: wp becomes zero, but base
        # does NOT move. Replay index, not physical state index, owns cursors.
        data = dict(case=Case(), replay_indices=[3, 1, -1, 2],
                    accept_lens=[0, 1, 4, 4], write_pos=[99, 8, 4, 7],
                    cache_base=[99, 14, 14, 14], is_flush=[0, 1, 0, 1], null_block_id=-1)
        oracle_cursor(data)
        self.assertEqual(data["write_pos"], [99, 1, 8, 0])
        self.assertEqual(data["cache_base"], [99, 6, 14, 14])
        self.assertEqual(data["is_flush"], [0, 0, 0, 0])  # equality at 8+2T=L
        data["accept_lens"] = [1, 0, 0, 1]
        oracle_cursor(data)
        self.assertEqual(data["write_pos"][2], 9)
        self.assertEqual(data["is_flush"][2], 1)  # strict >, not >=

    def test_optional_source_fingerprint(self):
        if OPTIONS.sglang_root is None:
            self.skipTest("source fingerprint NOT_RUN: no --sglang-root")
        source, metadata = check_source(OPTIONS.sglang_root)
        self.assertTrue(metadata["sha_verified"])
        functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        self.assertTrue(set(PUBLIC_FUNCTIONS) <= functions.keys())
        circular = functions["commit_gdn_replayssm_circular"].args
        self.assertFalse(circular.args)
        self.assertIn("d_residual_cache", [x.arg for x in circular.kwonlyargs])


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = torch_module()  # missing torch is an error, not a passing skip

    def test_hand_computed_fp64_step(self):
        t = self.torch
        state, delta = recurrent_step(t.tensor([[[2.0]]], dtype=t.float64),
                                      t.tensor([[1.0]], dtype=t.float64),
                                      t.tensor([[3.0]], dtype=t.float64),
                                      t.tensor([0.0], dtype=t.float64),
                                      t.tensor([0.5], dtype=t.float64))
        self.assertEqual(state.dtype, t.float64)
        self.assertEqual(float(delta), 0.5)
        self.assertEqual(float(state), 2.5)
        zero = normalized(t.zeros(1, 64, dtype=t.float64))
        self.assertTrue(t.equal(zero, t.zeros_like(zero)))

    def test_sequential_compact_equals_closed_form(self):
        t = self.torch
        data = make_case(Case(layers=2))
        c = data["case"]
        for layer in range(c.layers):
            for row in range(c.B):
                slot, replay = int(data["ssm_state_indices"][row]), int(data["replay_indices"][row])
                p = (t.arange(c.history) + int(data["cache_base"][replay])) % c.L
                g = data["g_cache"][layer, replay][:, p].double()
                d = (data["d_cache"][layer, replay][:, p].double()
                     + data["rawv_cache"][layer, replay][:, p].double())
                k = (data["k_cache"][layer, replay][:, p].double()
                     + data["rawk_cache"][layer, replay][:, p].double())
                k = k.repeat_interleave(c.HV // c.H, dim=0)
                weights = (g.sum(-1, keepdim=True) - g.cumsum(-1)).exp()
                expected = data["checkpoint_state"][layer, slot].double() * g.sum(-1).exp()[:, None, None]
                expected += t.einsum("htv,htk,ht->hvk", d, k, weights)
                t.testing.assert_close(history_state(data, layer, row, residual=True), expected,
                                       atol=1e-12, rtol=1e-12)

    def test_cache_encoding_and_lifetime(self):
        t = self.torch
        for dtype in TOLERANCES:
            data = make_case(Case(dtype=dtype, layers=1, history=0))
            before = snapshot_mutated(data)
            snapshots = oracle_decode(data)
            self.assertTrue(t.equal(data["checkpoint_state"], before["checkpoint_state"]))
            for key in ("write_pos", "cache_base", "is_flush"):
                self.assertTrue(t.equal(data[key], before[key]))
            # Newly encoded high+low can materialize an accepted prefix, but
            # commit must only run after cursors include the accepted count.
            oracle_cursor(data, fold_every=True)
            oracle_compact(data)
            for row in range(data["case"].B):
                slot = int(data["ssm_state_indices"][row])
                n = int(data["accept_lens"][row])
                atol, rtol = TOLERANCES[dtype]["state"]
                t.testing.assert_close(data["checkpoint_state"][0, slot].double(),
                                       snapshots[0, row][n - 1], atol=atol, rtol=rtol)
            replay = data["replay_indices"].long()
            self.assertTrue(t.equal(data["write_pos"][replay], t.zeros(3, dtype=t.int32)))


class DeviceTests(unittest.TestCase):
    differential_exclusions = set()

    @classmethod
    def setUpClass(cls):
        if OPTIONS.device is None:
            raise unittest.SkipTest("device suite NOT_RUN: explicit --device cuda:N required")
        cls.torch = torch_module()
        device = cls.torch.device(OPTIONS.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("device suite requires explicit cuda:N")
        cls.torch.cuda.set_device(device)
        cls.device = str(device)
        cls.api = importlib.import_module("pla.decode.gdn_replayssm_sglang")
        for name in PUBLIC_FUNCTIONS:
            if not callable(getattr(cls.api, name, None)):
                raise AssertionError(f"missing compact Replay API: {name}")
        cls.upstream = None
        if OPTIONS.sglang_root:
            props = cls.torch.cuda.get_device_properties(device)
            cls.upstream = load_upstream(
                OPTIONS.sglang_root,
                tf32_supported=not cls.torch.version.hip and props.major >= 8,
                gfx95_supported="gfx95" in getattr(props, "gcnArchName", ""))

    def assert_close(self, actual, expected, dtype, kind="state"):
        atol, rtol = TOLERANCES[dtype][kind]
        self.torch.testing.assert_close(actual.double(), expected.double(), atol=atol, rtol=rtol)

    def write_masks(self, data, operation, mode="verify", track=None, steps=None):
        """Structural write footprint; never infer untouched cells by float equality."""
        masks = {name: self.torch.zeros_like(data[name], dtype=self.torch.bool)
                 for name in MUTATED if data[name] is not None}
        c = data["case"]
        for row in range(c.B):
            slot, replay = int(data["ssm_state_indices"][row]), int(data["replay_indices"][row])
            if operation == "cursor":
                if replay > data["null_block_id"]:
                    for name in ("write_pos", "cache_base", "is_flush"):
                        masks[name][replay] = True
                continue
            if operation == "decode":
                bos, eos = (int(x) for x in data["query_start_loc"][row:row + 2])
                if slot <= data["null_block_id"]:
                    if mode != "flush":
                        masks["out"][:, bos:eos] = True
                    continue
                flush = bool(data["is_flush"][replay])
                if (mode == "verify" and flush) or (mode == "flush" and not flush):
                    continue
                wp, base = int(data["write_pos"][replay]), int(data["cache_base"][replay])
                if flush and wp > 0:
                    masks["checkpoint_state"][:, slot] = True
                masks["out"][:, bos:eos] = True
                paired = data["rawv_cache"] is not None and data["rawk_cache"] is not None
                for offset in range(min(eos - bos, c.L - wp)):
                    p = (base + wp + offset) & (c.L - 1)
                    for name in ("d_cache", "k_cache"):
                        masks[name][:, replay, :, p] = True
                    masks["g_cache"][:, replay, :, p] = True
                    if paired:
                        for name in ("rawv_cache", "rawk_cache"):
                            masks[name][:, replay, :, p] = True
                        if data["beta_cache"] is not None:
                            masks["beta_cache"][:, replay, :, p] = True
                continue
            if slot <= data["null_block_id"]:
                continue
            if bool(data["is_flush"][replay]):
                masks["checkpoint_state"][:, slot] = True
                for name in ("write_pos", "cache_base", "is_flush"):
                    masks[name][replay] = True
            if track is not None and steps is not None and int(track[row]) > data["null_block_id"] and int(steps[row]) >= 0:
                masks["checkpoint_state"][:, int(track[row])] = True
        return masks

    def compare(self, actual, expected, before):
        t = self.torch
        raw = expected["beta_cache"] is not None
        for name in MUTATED:
            if actual[name] is None:
                self.assertIsNone(expected[name])
                continue
            a, e = actual[name], expected[name]
            if not a.is_floating_point() or (raw and name in ("rawv_cache", "rawk_cache")):
                self.assertTrue(t.equal(a, e), name)
            elif not (not raw and name in ("rawv_cache", "rawk_cache")):
                self.assert_close(a, e, actual["case"].dtype, "output" if name == "out" else "state")
            # Every unmodified location is a bitwise canary, including rejected
            # history entries, inactive/null slots, output rows and cursor holes.
            mask = ~self.writable[name]
            self.assertTrue(t.equal(a[mask], before[name][mask]), f"untouched {name}")
        if not raw:
            for hi, lo in (("d_cache", "rawv_cache"), ("k_cache", "rawk_cache")):
                if actual[lo] is not None:
                    self.assert_close(actual[hi].double() + actual[lo].double(),
                                      expected[hi].double() + expected[lo].double(), actual["case"].dtype)

    def run_decode(self, data, mode="verify", normalize=True, scale=None):
        self.writable = self.write_masks(data, "decode", mode)
        before = snapshot_mutated(data)
        expected = clone_tree(data)
        oracle_decode(expected, mode=mode, normalize=normalize, scale=scale)
        direct = clone_tree(data) if self.upstream and data["q"].numel() else None
        if self.upstream and not data["q"].numel():
            self.differential_exclusions.add("zero-token verify: original Triton makes unmasked Q/K loads and faults; CUDA is checked against the oracle")
        for layer in range(data["case"].layers):
            kwargs = decode_kwargs(data, layer, mode=mode, normalize=normalize, scale=scale)
            result = self.api.gdn_replayssm_spec_decode(**kwargs)
            self.assertIs(result, kwargs["out"])
            if direct is not None:
                self.upstream.gdn_replayssm_spec_decode(**decode_kwargs(
                    direct, layer, mode=mode, normalize=normalize, scale=scale))
        self.compare(data, expected, before)
        if direct is not None:
            self.compare(data, direct, before)
        return data

    def run_cursor(self, data, fold_every=False, cache_buf_len=None):
        self.writable = self.write_masks(data, "cursor")
        before = snapshot_mutated(data)
        expected, direct = clone_tree(data), clone_tree(data) if self.upstream else None
        oracle_cursor(expected, fold_every=fold_every, cache_buf_len=cache_buf_len)
        self.api.commit_gdn_replayssm_spec(**cursor_kwargs(data, fold_every=fold_every, cache_buf_len=cache_buf_len))
        self.compare(data, expected, before)
        if direct is not None:
            self.upstream.commit_gdn_replayssm_spec(**cursor_kwargs(direct, fold_every=fold_every, cache_buf_len=cache_buf_len))
            self.compare(data, direct, before)

    def run_compact(self, data, track=None, steps=None, residual=True):
        self.writable = self.write_masks(data, "compact", track=track, steps=steps)
        before = snapshot_mutated(data)
        expected, direct = clone_tree(data), clone_tree(data) if self.upstream and data["case"].B else None
        if self.upstream and not data["case"].B:
            self.differential_exclusions.add("zero-row compact commit: original uses a zero-dimensional grid; CUDA is checked against the oracle")
        oracle_compact(expected, track, steps, residual=residual)
        self.api.commit_gdn_replayssm_circular(**compact_kwargs(data, track, steps, residual=residual))
        self.compare(data, expected, before)
        if direct is not None:
            self.upstream.commit_gdn_replayssm_circular(**compact_kwargs(direct, track, steps, residual=residual))
            self.compare(data, direct, before)

    def test_verify_shapes_dtypes_and_strides(self):
        for config in UPSTREAM_CASES.values():
            for dtype in TOLERANCES:
                for state_dtype in ("float32", dtype):
                    for strided in (False, True):
                        with self.subTest(config=config, dtype=dtype, state_dtype=state_dtype, strided=strided):
                            data = make_case(Case(**config, dtype=dtype, layers=1), device=self.device,
                                             state_dtype=state_dtype, strided=strided)
                            self.run_decode(data)

    def test_large_batch_projection_dispatch(self):
        for width in (1, 2, 3, 4, 6, 8):
            data = make_case(Case(B=128, H=1, HV=2, K=128, V=128,
                                  T=width, layers=1), device=self.device,
                             lengths=[1 + row % width for row in range(128)])
            data["is_flush"][data["replay_indices"][::7].long()] = 1
            self.run_decode(data)
        data = make_case(Case(B=128, H=1, HV=2, K=128, V=128,
                              layers=1), device=self.device, raw=True)
        self.run_decode(data)
        self.run_cursor(data, fold_every=True)

    def test_compat_tf32_specializations_with_varlen_and_flush(self):
        for batch, heads, width in ((8, 2, 2), (32, 4, 4), (32, 6, 4)):
            with self.subTest(batch=batch, value_heads=heads, width=width):
                data = make_case(Case(B=batch, H=2, HV=heads, K=128, V=128, T=width, layers=1),
                                 device=self.device, strided=False,
                                 lengths=[row % (width + 1) for row in range(batch)])
                data["is_flush"][data["replay_indices"][1::7].long()] = 1
                data["ssm_state_indices"][3] = -1
                self.run_decode(data)

    def test_asymmetric_state_layout_and_spec_lengths(self):
        # New cases, not claimed as upstream UT extraction. V != K catches an
        # accidental [K,V] checkpoint transpose hidden by both square UT shapes.
        for T in (1, 2, 4):
            data = make_case(Case(K=128, V=64, T=T, history=5), device=self.device)
            self.run_decode(data)
            self.run_cursor(data, fold_every=True)
            self.run_compact(data)

    def test_layer_slot_strides_and_value_tail(self):
        t = self.torch
        data = make_case(Case(K=128, V=129, H=2, HV=4), device=self.device)
        buffers = []
        for name in ("checkpoint_state", "d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache"):
            original = data[name]
            strides = list(original.stride())
            strides[1] += 1
            strides[0] = original.shape[1] * strides[1] + 3
            size = 1 + sum((n - 1) * s for n, s in zip(original.shape, strides))
            backing = t.full((size,), 7, device=self.device, dtype=original.dtype)
            data[name] = backing.as_strided(original.shape, strides).copy_(original)
            gaps = t.ones(size, device=self.device, dtype=t.bool)
            gaps.as_strided(original.shape, strides).fill_(False)
            buffers.append((backing, gaps))
        for name in ("write_pos", "cache_base", "is_flush"):
            data[name] = strided_int(data[name].tolist(), self.device, dtype=data[name].dtype)
        upstream = self.upstream
        self.upstream = None
        try:
            self.run_decode(data)
            self.run_cursor(data, fold_every=True)
            self.run_compact(data, strided_int([1, 0, 3], self.device), strided_int([1, -1, 2], self.device))
        finally:
            self.upstream = upstream
        if upstream is not None:
            self.differential_exclusions.add("gapped layer/slot strides with V=129: PPU Triton disagrees with the FP64 oracle; CUDA retains oracle and padding-canary checks")
        for backing, gaps in buffers:
            self.assertTrue(t.all(backing[gaps] == 7).item())

    def test_generic_dimension_and_window_bounds(self):
        for dim, width, capacity in ((256, 8, 32), (64, 16, 32), (32, 32, 64)):
            data = make_case(Case(B=1, H=1, HV=2, K=dim, V=128,
                                  T=width, L=capacity, history=0, layers=1), device=self.device)
            self.run_decode(data)
            self.run_cursor(data, fold_every=True)
            self.run_compact(data)
        data = make_case(Case(B=1, H=1, HV=1, K=256, V=128,
                              T=32, L=64, history=0, layers=1), device=self.device)
        before = snapshot_mutated(data)
        with self.assertRaises((RuntimeError, ValueError)):
            self.api.gdn_replayssm_spec_decode(**decode_kwargs(data))
        for name in before:
            self.assertTrue(self.torch.equal(data[name], before[name]), name)

    def test_verify_no_residual_and_no_normalization(self):
        for residual in (False, True):
            data = make_case(device=self.device, residual=residual)
            # Keep unnormalized recurrence well conditioned; do not change tolerances.
            data["q"].mul_(0.1)
            data["k"].mul_(0.1)
            self.run_decode(data, normalize=False, scale=0.25)

    def test_exact_both_and_flush_raw_lifetime(self):
        for dtype in TOLERANCES:
            for mode in ("both", "flush"):
                data = make_case(Case(dtype=dtype), device=self.device, raw=True)
                data["is_flush"][data["replay_indices"][[0, 2]].long()] = 1
                self.run_decode(data, mode)
                self.run_cursor(data)
                # Positive acceptance moves base to the draft records; no decode
                # cursor reset is allowed before this advance.
                self.run_decode(data, "both")

    def test_exact_requires_raw_buffers(self):
        for mode in ("both", "flush"):
            for missing in ("rawv_cache", "rawk_cache", "beta_cache"):
                data = make_case(device=self.device, raw=True)
                data[missing] = None
                with self.assertRaises(ValueError):
                    self.api.gdn_replayssm_spec_decode(**decode_kwargs(data, mode=mode))

    def test_cursor_zero_flush_null_nonidentity_and_strict_threshold(self):
        for fold_every in (False, True):
            for buffer_len in (16, 32):
                data = make_case(device=self.device)
                data["accept_lens"].copy_(self.torch.tensor([0, 1, 4], device=self.device))
                data["write_pos"].fill_(8)
                data["is_flush"][int(data["replay_indices"][0])] = 1
                self.run_cursor(data, fold_every, buffer_len)
                data["replay_indices"][1] = -1
                self.run_cursor(data, fold_every, buffer_len)
        data = make_case(device=self.device)
        data["write_pos"].fill_(4)
        data["accept_lens"].fill_(4)
        self.run_cursor(data)
        data["accept_lens"].fill_(1)
        self.run_cursor(data)

    def test_contiguous_metadata_zero_accept_flush_and_wrap(self):
        data = make_case(device=self.device, strided=False)
        t = self.torch
        zero, inactive, folding = data["replay_indices"].tolist()
        data["accept_lens"].copy_(t.tensor([0, 1, 4], device=self.device))
        data["write_pos"].fill_(8)
        data["is_flush"][zero] = 1
        data["replay_indices"][1] = -1
        self.run_cursor(data)
        self.assertEqual(int(data["write_pos"][zero]), 0)
        self.assertEqual(int(data["write_pos"][inactive]), 8)
        self.assertEqual(int(data["write_pos"][folding]), 12)
        data["ssm_state_indices"][1] = -1
        self.run_compact(data)
        self.assertEqual(int(data["cache_base"][folding]), 10)
        self.assertEqual(int(data["write_pos"][folding]), 0)
        self.assertEqual(int(data["is_flush"][folding]), 0)

    def test_extended_cursor_base_feeds_verify_and_fold(self):
        for B, HV, K, strided in ((1, 1, 32, True), (1, 1, 128, False), (48, 4, 128, False)):
            with self.subTest(batch=B, value_heads=HV, key_dim=K, strided=strided):
                data = make_case(Case(B=B, H=1, HV=HV, K=K, V=128, T=4, L=16,
                                      history=9, layers=1), device=self.device, strided=strided)
                if strided:
                    for name in ("write_pos", "cache_base", "is_flush"):
                        data[name] = strided_int(data[name].tolist(), self.device)
                data["cache_base"].fill_(14)
                data["is_flush"].fill_(1)
                data["accept_lens"].fill_(2)
                active = data["replay_indices"].long()
                self.run_cursor(data, cache_buf_len=32)
                self.assertTrue(bool((data["cache_base"][active] == 23).all()))
                self.run_decode(data)
                self.run_cursor(data, fold_every=True, cache_buf_len=32)
                self.run_compact(data)
                self.assertTrue(bool((data["cache_base"][active] == 11).all()))

    def test_compact_keeps_value_residual_and_cross_term(self):
        for K, HV, L in ((32, 1, 16), (128, 1, 16), (128, 48, 16), (128, 48, 32)):
            with self.subTest(key_dim=K, value_heads=HV, capacity=L):
                data = make_case(Case(B=1, H=1, HV=HV, K=K, V=128, T=4, L=L, history=2,
                                      layers=1), device=self.device, strided=False)
                for name in ("checkpoint_state", "d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache"):
                    data[name].zero_()
                r = int(data["replay_indices"][0])
                slot = int(data["ssm_state_indices"][0])
                base = data["case"].L - 2
                data["d_cache"][0, r, :, base, 17] = 2 ** 24
                data["d_cache"][0, r, :, base + 1, 17] = -(2 ** 24)
                data["rawv_cache"][0, r, :, base, 17] = 1
                data["k_cache"][0, r, 0, base:, :2] = 1
                data["rawk_cache"][0, r, 0, base:, 1] = 0.25
                data["is_flush"][r] = 1
                expected = self.torch.zeros_like(data["checkpoint_state"])
                expected[0, slot, :, 17, 0] = 1
                expected[0, slot, :, 17, 1] = 1.25
                self.api.commit_gdn_replayssm_circular(**compact_kwargs(data))
                self.torch.testing.assert_close(data["checkpoint_state"], expected, atol=0, rtol=0)

    def test_all_layer_compact_tracking_null_reset_and_wrap(self):
        for dtype in TOLERANCES:
            for residual in (False, True):
                for state_dtype in ("float32", dtype):
                    data = make_case(Case(dtype=dtype, layers=3), device=self.device,
                                     residual=residual, state_dtype=state_dtype)
                    self.run_decode(data)
                    self.run_cursor(data, fold_every=True)
                    data["ssm_state_indices"][1] = -1
                    # Row 2 is track-only: snapshot without changing checkpoint/cursors.
                    data["is_flush"][int(data["replay_indices"][2])] = 0
                    track = strided_int([1, 0, 3], self.device)
                    steps = strided_int([1, -1, 2], self.device)
                    self.run_compact(data, track, steps, residual)

    def test_empty_varlen_and_null(self):
        for mode in ("verify", "both", "flush"):
            data = make_case(device=self.device, raw=mode != "verify", lengths=[4, 0, 2])
            data["ssm_state_indices"][2] = -1
            data["is_flush"][int(data["replay_indices"][1])] = int(mode != "verify")
            self.run_decode(data, mode)
        for B in (0, 3):
            data = make_case(Case(B=B), device=self.device, lengths=[0] * B)
            data["accept_lens"].zero_()
            self.run_decode(data)
            self.run_cursor(data)
            self.run_compact(data)

    def test_verify_history_ignores_low_parts(self):
        data = make_case(device=self.device)
        changed = clone_tree(data)
        # Only history is modified; both calls overwrite the same draft cells.
        # The low parts must NOT affect verify output in this source revision.
        changed["rawv_cache"].add_(0.25)
        changed["rawk_cache"].sub_(0.125)
        self.run_decode(data)
        self.run_decode(changed)
        self.assertTrue(self.torch.equal(data["out"], changed["out"]))

    def test_null_zero_and_empty_fold(self):
        data = make_case(device=self.device)
        data["null_block_id"] = 0
        data["ssm_state_indices"][1] = 0
        data["replay_indices"][1] = 0
        self.run_decode(data)
        self.run_cursor(data, fold_every=True)
        self.run_compact(data)
        data = make_case(Case(history=0), device=self.device)
        data["accept_lens"].zero_()
        data["is_flush"].fill_(1)
        self.run_compact(data)

    def run_chain(self, c, rounds):
        data = make_case(c, device=self.device)
        truth = data["checkpoint_state"].double().clone()
        for step in range(rounds):
            fresh = make_case(c, device=self.device, seed=1000 + step)
            for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias"):
                data[name].copy_(fresh[name])
            baseline = clone_tree(data)
            baseline["checkpoint_state"] = truth.clone()
            baseline["write_pos"].zero_()
            baseline["is_flush"].zero_()
            snapshots = oracle_decode(baseline)
            self.run_decode(data)
            self.assert_close(data["out"], baseline["out"], c.dtype, "output")
            data["accept_lens"].copy_(self.torch.tensor(
                [0 if (step + row) % 11 == 0 else 1 + (step + row) % c.T for row in range(c.B)], device=self.device))
            for row in range(c.B):
                slot, n = int(data["ssm_state_indices"][row]), int(data["accept_lens"][row])
                if n:
                    truth[0, slot].copy_(snapshots[0, row][n - 1])
            # Alternate forced compact folds and natural capacity folds.
            self.run_cursor(data, fold_every=step % 7 == 0)
            self.run_compact(data)
            for row in range(c.B):
                slot = int(data["ssm_state_indices"][row])
                live = history_state(data, 0, row, residual=True)
                self.assert_close(live, truth[0, slot], c.dtype)

    def test_256_round_independent_recurrence(self):
        # Same case/round count as adjacent raw-fold UT, but this is a NEW compact
        # regression and uses the fixed dtype budgets, never the raw-fold eps.
        for dtype in TOLERANCES:
            self.run_chain(Case(dtype=dtype, layers=1, history=0), 256)

    def test_gqa128_40_rounds(self):
        # Adjacent linear test's shape/40-step coverage, adapted to T=4 compact
        # windows. Its L=1 semantics cannot represent compact's L>=2*T contract.
        for name in ("gqa128", "gqa128-wide"):
            for dtype in TOLERANCES:
                for L in (16, 32):
                    with self.subTest(name=name, dtype=dtype, L=L):
                        self.run_chain(Case(**UPSTREAM_CASES[name], L=L, dtype=dtype,
                                            layers=1, history=0), 40)

    def test_graph_replay_restores_mutated_state(self):
        t = self.torch
        for exact, large in ((False, False), (True, False), (False, True)):
            case = Case(B=128, H=1, HV=2, K=128, V=128, layers=1) if large else Case(layers=2)
            data = make_case(case, device=self.device, raw=exact)
            if exact:
                data["is_flush"].fill_(1)
            before = snapshot_mutated(data)
            expected = clone_tree(data)
            self.writable = self.write_masks(data, "decode", "both" if exact else "verify")
            for name, mask in self.write_masks(data, "cursor").items():
                self.writable[name] |= mask
            oracle_decode(expected, mode="both" if exact else "verify")
            oracle_cursor(expected, fold_every=True)
            if not exact:
                for name, mask in self.write_masks(expected, "compact").items():
                    self.writable[name] |= mask
                oracle_compact(expected)

            def launch():
                for layer in range(data["case"].layers):
                    self.api.gdn_replayssm_spec_decode(**decode_kwargs(data, layer, mode="both" if exact else "verify"))
                self.api.commit_gdn_replayssm_spec(**cursor_kwargs(data, fold_every=True))
                if not exact:
                    self.api.commit_gdn_replayssm_circular(**compact_kwargs(data))

            stream = t.cuda.Stream(device=self.device)
            stream.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(stream):
                launch()
            t.cuda.current_stream().wait_stream(stream)
            restore_mutated(data, before)
            t.cuda.synchronize()
            graph = t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                launch()
            for _ in range(3):
                restore_mutated(data, before)
                graph.replay()
                t.cuda.synchronize()
                self.compare(data, expected, before)

    def test_rejects_overlapping_strided_q_and_output(self):
        t = self.torch
        data = make_case(Case(B=1, H=1, HV=1, K=32, V=64, T=2,
                              history=0, layers=1, dtype="float32"), device=self.device)
        backing = t.zeros(256, device=self.device)
        kwargs = decode_kwargs(data)
        kwargs["q"] = backing.as_strided((2, 1, 32), (128, 32, 1))
        kwargs["out"] = backing.as_strided((2, 1, 64), (128, 64, 1), storage_offset=1)
        with self.assertRaises(RuntimeError):
            self.api.gdn_replayssm_spec_decode(**kwargs)

    def test_mixed_state_dtype_tf32_cancellation(self):
        t = self.torch
        data = make_case(Case(B=1, H=1, HV=1, K=32, V=32, T=1,
                              history=0, layers=1, dtype="float32"),
                         state_dtype="float16", device=self.device)
        data["checkpoint_state"].zero_()
        data["q"].zero_(); data["k"].zero_()
        data["q"][..., 0] = 1; data["q"][..., 1] = -1
        data["k"][..., 0] = 1; data["k"][..., 1] = 1 + 3 * 2**-12
        data["v"].fill_(1024); data["b"].zero_()
        expected = clone_tree(data)
        kw = decode_kwargs(data, normalize=False, scale=1)
        kw["dot_precision"] = "tf32"
        self.api.gdn_replayssm_spec_decode(**kw)
        t.testing.assert_close(data["out"], t.zeros_like(data["out"]), atol=0, rtol=0)
        if self.upstream is not None:
            kw = decode_kwargs(expected, normalize=False, scale=1)
            kw["dot_precision"] = "tf32"
            self.upstream.gdn_replayssm_spec_decode(**kw)
            t.testing.assert_close(data["out"], expected["out"], atol=0, rtol=0)

    def test_compact_fp16_decay_split_underflow(self):
        t = self.torch
        data = make_case(Case(B=1, H=1, HV=1, K=32, V=32, T=2,
                              history=2, layers=1, dtype="float16"),
                         residual=False, device=self.device)
        for name in ("checkpoint_state", "d_cache", "k_cache", "g_cache"):
            data[name].zero_()
        r = int(data["replay_indices"][0])
        base = int(data["cache_base"][r])
        data["d_cache"][0, r, 0, base, 0] = 1
        data["k_cache"][0, r, 0, base, 0] = 65504
        data["g_cache"][0, r, 0, (base + 1) % 16] = -18
        data["is_flush"][r] = 1
        track, steps = strided_int([3], self.device), strided_int([1], self.device)
        expected = clone_tree(data)
        self.api.commit_gdn_replayssm_circular(**compact_kwargs(data, track, steps, residual=False))
        t.testing.assert_close(data["checkpoint_state"], t.zeros_like(data["checkpoint_state"]), atol=0, rtol=0)
        if self.upstream is not None:
            self.upstream.commit_gdn_replayssm_circular(**compact_kwargs(expected, track, steps, residual=False))
            t.testing.assert_close(data["checkpoint_state"], expected["checkpoint_state"], atol=0, rtol=0)

    def test_upstream_differential_status(self):
        if self.upstream is None:
            self.skipTest("upstream differential NOT_RUN: no --sglang-root; oracle checks remain enabled")
        self.assertTrue(self.upstream.PROVENANCE["sha_verified"])


def main(argv=None):
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("static", "oracle", "device", "all"), default="static")
    parser.add_argument("--device", help="cpu for oracle; explicit cuda:N for device tests")
    parser.add_argument("--sglang-root", type=Path)
    parser.add_argument("--test", action="append", help="run selected test method names")
    OPTIONS = parser.parse_args(argv)
    if OPTIONS.suite in ("device", "all") and (not OPTIONS.device or not OPTIONS.device.startswith("cuda:")):
        parser.error("device/all suite requires explicit --device cuda:N")
    if OPTIONS.suite == "oracle" and OPTIONS.device not in (None, "cpu"):
        parser.error("CPU oracle suite accepts only --device cpu")
    if OPTIONS.suite != "static" and importlib.util.find_spec("torch") is None:
        print(json.dumps(dict(suite=OPTIONS.suite, status="NOT_RUN", reason="torch is not installed",
                              tests_run=0, passed_test_methods=0, numerical="NOT_RUN",
                              upstream_differential="NOT_RUN", source_commit=SOURCE_COMMIT)))
        return 2
    classes = {"static": [StaticTests], "oracle": [OracleTests], "device": [DeviceTests],
               "all": [StaticTests, OracleTests, DeviceTests]}[OPTIONS.suite]
    if OPTIONS.test:
        tests = [cls(name) for cls in classes for name in OPTIONS.test if hasattr(cls, name)]
        if len(tests) != len(OPTIONS.test):
            parser.error("selected test is missing or ambiguous in this suite")
        suite = unittest.TestSuite(tests)
    else:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)

    class CountingResult(unittest.TextTestResult):
        passed_test_methods = 0

        def addSuccess(self, test):
            super().addSuccess(test)
            self.passed_test_methods += 1

    result = unittest.TextTestRunner(verbosity=2, resultclass=CountingResult).run(suite)
    failed = len(result.failures) + len(result.errors)
    report = dict(suite=OPTIONS.suite, status="FAILED" if failed else "PARTIAL_NOT_RUN" if result.skipped else "PASSED",
                  tests_run=result.testsRun, passed_test_methods=result.passed_test_methods,
                  skipped_not_passed=len(result.skipped), failures=failed,
                  upstream_differential=("PARTIAL" if DeviceTests.differential_exclusions else "SEE_TEST_RESULTS") if OPTIONS.sglang_root and OPTIONS.suite in ("device", "all") else "NOT_RUN",
                  differential_exclusions=sorted(DeviceTests.differential_exclusions),
                  numerical="NOT_RUN" if OPTIONS.suite == "static" else "SEE_TEST_RESULTS",
                  source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256,
                  tolerances=TOLERANCES)
    print(json.dumps(report, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
