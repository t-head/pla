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

"""Pure-CPU tests for FlashQLA's architecture-aware CP planner model.

The production module depends on PyTorch only for tensor materialization.  A
small fallback stub keeps these planner tests runnable on developer hosts that
do not have a CPU PyTorch wheel; no CUDA runtime or PPU is used.
"""

import importlib.util
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _install_torch_stub_if_needed():
    try:
        import torch  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _FakeDevice:
        type = "cpu"
        index = None

    class _FakeTensor(list):
        def __init__(self, values, dtype=None, device=None):
            super().__init__(values)
            self.dtype = dtype
            self.device = _FakeDevice()
            self.is_cuda = False

        def numel(self):
            return len(self)

        def tolist(self):
            return list(self)

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _FakeTensor
    torch_stub.int8 = "int8"
    torch_stub.int32 = "int32"
    torch_stub.tensor = _FakeTensor
    torch_stub.cuda = types.SimpleNamespace(current_device=lambda: 0)
    sys.modules["torch"] = torch_stub


def _load_cp_preprocess():
    _install_torch_stub_if_needed()
    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root / "pla" / "prefill" / "flashqla" / "private" /
        "cp_preprocess.py"
    )
    # Keep the package hierarchy so the real _sanity/_bypass load without importing the GPU extension.
    package = "_flashqla_cp_cpu_test"
    for suffix in ("", ".prefill", ".prefill.flashqla", ".prefill.flashqla.private"):
        name = package + suffix
        parent = types.ModuleType(name)
        parent.__path__ = [str(repo_root / "pla" / suffix.lstrip(".").replace(".", "/"))]
        sys.modules[name] = parent
    module_name = package + ".prefill.flashqla.private.cp_preprocess"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during decoration.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cp = _load_cp_preprocess()


class Tc01RealGatePlannerTest(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(
            os.environ, {"PLA_SANITY_CHECK": "0", "PLA_BYPASS_EXEC": "0"}
        )
        environment.start()
        self.addCleanup(environment.stop)
        cp._cp_plan_cache.clear()

    def tearDown(self):
        cp._cp_plan_cache.clear()

    @staticmethod
    def _cu(*lengths):
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        return cp.torch.tensor(cumulative, dtype=cp.torch.int32)

    @staticmethod
    def _raw_cu(*lengths):
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        return tuple(cumulative)

    @staticmethod
    def _tc01_device_model(_raw_cu_seqlens):
        return cp._planner_execution_model_from_properties(8, 0, 64)

    @staticmethod
    def _tc02_device_model(_raw_cu_seqlens):
        return cp._planner_execution_model_from_properties(8, 9, 39)

    def test_architecture_models_keep_physical_tc01_shards(self):
        tc01 = cp._planner_execution_model_from_properties(8, 0, 64)
        tc02 = cp._planner_execution_model_from_properties(8, 9, 39)

        self.assertEqual(tc01.architecture, "tc01")
        self.assertEqual(tc01.physical_cu_count, 64)
        self.assertEqual(tc01.fused_cu_count, 64)
        self.assertEqual(tc01.fused_head_shards, 2)
        self.assertEqual(tc02.architecture, "tc02")
        self.assertEqual(tc02.fused_cu_count, 39)
        self.assertEqual(tc02.fused_head_shards, 1)

    def test_tc01_real_gate_guard_boundaries(self):
        cases = (
            ((9999,), 32, 8, False),
            ((10000,), 32, 8, True),
            ((8000, 23999), 32, 8, False),
            ((8000, 24000), 32, 8, True),
            ((4000, 4000, 23999), 32, 8, False),
            ((4000, 4000, 24000), 32, 8, True),
            ((2000, 2000, 2000, 24000), 32, 8, False),
            ((32768,), 32, 16, False),
            ((10000,), 64, 8, False),
        )
        for lengths, chunk_size, heads, expected in cases:
            with self.subTest(lengths=lengths, heads=heads):
                allowed = cp._tc01_real_gate_cp_allowed(
                    self._raw_cu(*lengths), chunk_size, heads
                )
                self.assertEqual(allowed, expected)

    def test_tc01_overlap_guard_boundaries(self):
        accepted = (
            ((14308, 2076), 32, 16, 2),
            ((14309, 2075), 32, 16, 2),
            ((27825, 4943), 32, 16, 2),
            ((27826, 4942), 32, 16, 2),
            ((16384, 16384), 32, 48, 16),
            ((20000, 12768), 32, 64, 8),
            ((24817, 7951), 32, 64, 16),
            ((15900, 9000, 7868), 32, 16, 4),
            ((17039, 8000, 7729), 32, 32, 16),
            ((12000, 11000, 9768), 32, 48, 16),
            ((12000, 11000, 9768), 32, 64, 8),
            ((8192, 8192, 8192, 8192), 32, 16, 2),
            ((14293, 7000, 6000, 5475), 32, 16, 2),
            ((6554, 6554, 6554, 6553, 6553), 32, 32, 8),
        )
        rejected = (
            ((14309, 2075), 32, 16, 4),
            ((14309, 2075), 64, 16, 2),
            ((24816, 7952), 32, 64, 16),
            ((15901, 9000, 7867), 32, 16, 4),
            ((17038, 8000, 7730), 32, 32, 16),
            ((16384, 16384), 32, 48, 8),
            ((8192, 8192, 8192, 8192), 32, 32, 8),
            ((6554, 6554, 6554, 6553, 6553), 32, 32, 2),
            ((16384, 16383), 32, 48, 16),
            ((16384, 16384), 64, 48, 16),
            ((16384, 16384), 32, 48, None),
        )
        self.assertEqual(cp.TC01_OVERLAP_WARMUP_THRESHOLD, -4.0)
        for lengths, chunk_size, heads, key_heads in accepted:
            with self.subTest(lengths=lengths, accepted=True):
                self.assertTrue(cp._tc01_overlap_warmup_allowed(
                    self._raw_cu(*lengths), chunk_size, heads, key_heads
                ))
        for lengths, chunk_size, heads, key_heads in rejected:
            with self.subTest(lengths=lengths, accepted=False):
                self.assertFalse(cp._tc01_overlap_warmup_allowed(
                    self._raw_cu(*lengths), chunk_size, heads, key_heads
                ))

    def test_tc01_overlap_flag_promotes_only_the_new_planner_region(self):
        cu = self._cu(16384, 16384)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            default_plan = cp.calc_cp_seqs(
                cu, 32, 48, num_k_heads=16
            )
            overlap_plan = cp.calc_cp_seqs(
                cu, 32, 48, num_k_heads=16,
                tc01_enable_overlap_candidate=True,
            )
            unsupported_plan = cp.calc_cp_seqs(
                cu, 32, 48, num_k_heads=8,
                tc01_enable_overlap_candidate=True,
            )

        self.assertFalse(default_plan[0])
        self.assertTrue(overlap_plan[0])
        self.assertFalse(unsupported_plan[0])
        self.assertEqual(len(cp._cp_plan_cache), 3)

    def test_tc01_p1_h16_hg2_uses_exact_five_segment_full_wave_plan(self):
        raw_cu = self._raw_cu(14309, 2075)
        cu = self._cu(14309, 2075)
        b1_cu = self._cu(8346)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            serial_plan = cp.calc_cp_seqs(cu, 32, 16, num_k_heads=2)
            overlap_plan = cp.calc_cp_seqs(
                cu, 32, 16, num_k_heads=2,
                tc01_enable_overlap_candidate=True,
            )
            b1_plan = cp.calc_cp_seqs(
                b1_cu, 32, 16, num_k_heads=2,
                tc01_enable_overlap_candidate=True,
            )
            raw_overlap_plan = cp.calc_cp_seqs(
                cu, 32, 16, num_k_heads=2,
                tc01_enable_overlap_candidate=True,
                tc01_packed_raw_gate=True,
                tc01_force_candidate=True,
            )
            threshold = cp.tc01_overlap_warmup_threshold(
                cu, raw_cu, 32, 16, 2,
                use_raw_gating=False,
                indexed_state=True,
                has_initial_state=True,
            )
            raw_threshold = cp.tc01_overlap_warmup_threshold(
                cu, raw_cu, 32, 16, 2,
                use_raw_gating=True,
                indexed_state=True,
                has_initial_state=True,
            )

        self.assertFalse(serial_plan[0])
        self.assertTrue(overlap_plan[0])
        self.assertTrue(b1_plan[0])
        self.assertEqual(
            b1_plan[1].tolist(), [0, 1696, 3392, 5088, 6784, 8346]
        )
        self.assertEqual(threshold, -4.0)
        self.assertEqual(raw_threshold, -0.75)
        self.assertEqual(len(raw_overlap_plan[1]) - 1, 7)
        self.assertEqual(
            overlap_plan[1].tolist(),
            [0, 3584, 7168, 10752, 14309, 16384],
        )
        self.assertEqual(overlap_plan[2].tolist(), [0, 4, 5])
        self.assertEqual(overlap_plan[3].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(overlap_plan[4].tolist(), [0, 0, 0, 1, 1])

    def test_tc01_serial_lpt_boundaries(self):
        for lengths, heads, key_heads, chunk_size, expected in (
            ((15104, 1280), 16, 2, 32, True),
            ((1280, 15104), 16, 2, 32, True),
            ((15502, 17039, 227), 16, 2, 32, True),
            ((0, 32768), 16, 2, 32, True),
            ((8192, 8192), 16, 2, 32, True),
            ((8192, 8191), 16, 2, 32, False),
            ((16384, 16385), 16, 2, 32, False),
            ((16384,), 16, 2, 32, False),
            ((8192, 8192, 8192, 8192), 16, 2, 32, False),
            ((15104, 1280), 16, 4, 32, False),
            ((15104, 1280), 8, 2, 32, False),
            ((15104, 1280), 16, 2, 64, False),
        ):
            with self.subTest(lengths=lengths, heads=heads, key_heads=key_heads):
                self.assertEqual(cp._tc01_serial_lpt_allowed(
                    self._raw_cu(*lengths), chunk_size, heads, key_heads), expected)

    def test_tc01_serial_lpt_is_raw_gate_scoped_and_forceable(self):
        cu = self._cu(15104, 1280)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            raw_plan = cp.calc_cp_seqs(
                cu, 32, 16, num_k_heads=2, tc01_packed_raw_gate=True,
                tc01_enable_overlap_candidate=True)
            self.assertEqual(raw_plan, (False, None, None, None, None))
            self.assertIs(cp.calc_cp_seqs(
                cu, 32, 16, num_k_heads=2, tc01_packed_raw_gate=True,
                tc01_enable_overlap_candidate=True), raw_plan)
            forced = cp.calc_cp_seqs(
                cu, 32, 16, num_k_heads=2, tc01_packed_raw_gate=True,
                tc01_enable_overlap_candidate=True, tc01_force_candidate=True)
            activated = cp.calc_cp_seqs(
                cu, 32, 16, num_k_heads=2,
                tc01_enable_overlap_candidate=True)
            self.assertTrue(forced[0])
            self.assertTrue(activated[0])
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc02_device_model
        ), mock.patch.object(cp, "_tc01_serial_lpt_allowed") as guard:
            cp.calc_cp_seqs(cu, 32, 16, num_k_heads=2, tc01_packed_raw_gate=True)
            guard.assert_not_called()

    def test_tc01_p2_measured_segment_geometry_boundaries(self):
        accepted = (
            ((8192,), 32, 16, 2, False, (52,)),
            ((24000,), 32, 16, 2, False, (108,)),
            ((411, 2716), 32, 16, 2, True, (13, 22)),
            ((6397, 1795), 32, 32, 4, True, (67, 57)),
            ((7173, 1019), 32, 32, 4, True, (75, 32)),
            ((1093, 7208), 32, 32, 4, True, (35, 76)),
            ((13118, 14417, 5233), 32, 32, 4, True, (103, 91, 82)),
            ((12521, 3863), 32, 32, 4, True, (98, 121)),
            ((13913, 2471), 32, 32, 4, True, (109, 78)),
            ((14309, 2075), 32, 32, 4, True, (112, 65)),
            ((1888, 12451, 2045), 32, 32, 4, True, (59, 98, 64)),
            ((4770, 7863, 8574), 32, 32, 4, True, (75, 82, 90)),
            ((4173, 9174, 10003), 32, 32, 4, True, (66, 96, 105)),
            ((1988, 13107, 1289), 32, 32, 4, True, (63, 103, 41)),
            ((27029, 5739), 32, 64, 8, True, (169, 180)),
        )
        rejected = (
            ((3499,), 32, 16, 2),
            ((3000, 863), 32, 16, 2),
            ((7950, 242), 32, 32, 4),
            ((27029, 5739), 64, 64, 8),
            ((27029, 5739), 32, 64, 16),
        )
        for (lengths, chunk_size, heads, key_heads,
             use_packed_raw_spec, expected) in accepted:
            with self.subTest(lengths=lengths, accepted=True):
                self.assertEqual(cp._tc01_measured_segment_chunks(
                    self._raw_cu(*lengths), chunk_size, heads, key_heads,
                    use_packed_raw_spec=use_packed_raw_spec,
                ), expected)
        for lengths, chunk_size, heads, key_heads in rejected:
            with self.subTest(lengths=lengths, accepted=False):
                self.assertIsNone(cp._tc01_measured_segment_chunks(
                    self._raw_cu(*lengths), chunk_size, heads, key_heads,
                    use_packed_raw_spec=True,
                ))

    def test_tc01_p2_measured_plans_use_validated_boundaries(self):
        cases = (
            ((8192,), 16, 2,
             [0, 1664, 3328, 4992, 6656, 8192], -0.75, False),
            ((24000,), 16, 2,
             [0, 3456, 6912, 10368, 13824, 17280, 20736, 24000],
             -3.25, False),
            ((411, 2716), 16, 2,
             [0, 411, 1115, 1819, 2523, 3127], -0.75, True),
            ((6397, 1795), 32, 4,
             [0, 2144, 4288, 6397, 8192], -0.5, True),
            ((7173, 1019), 32, 4,
             [0, 2400, 4800, 7173, 8192], -0.25, True),
            ((1093, 7208), 32, 4,
             [0, 1093, 3525, 5957, 8301], -0.3125, True),
            ((13118, 14417, 5233), 32, 4,
             [0, 3296, 6592, 9888, 13118, 16030, 18942, 21854, 24766,
              27535, 30159, 32768], -0.25, True),
            ((12521, 3863), 32, 4,
             [0, 3136, 6272, 9408, 12521, 16384], -0.25, True),
            ((13913, 2471), 32, 4,
             [0, 3488, 6976, 10464, 13913, 16384], -0.25, True),
            ((14309, 2075), 32, 4,
             [0, 3584, 7168, 10752, 14309, 16384], -0.25, True),
            ((1888, 12451, 2045), 32, 4,
             [0, 1888, 5024, 8160, 11296, 14339, 16384], -0.25, True),
            ((4770, 7863, 8574), 32, 4,
             [0, 2400, 4770, 7394, 10018, 12633, 15513, 18393, 21207],
             -0.34375, True),
            ((4173, 9174, 10003), 32, 4,
             [0, 2112, 4173, 7245, 10317, 13347, 16707, 20067, 23350],
             -0.25, True),
            ((1988, 13107, 1289), 32, 4,
             [0, 1988, 5284, 8580, 11876, 15095, 16384], -0.25, True),
            ((27029, 5739), 64, 8,
             [0, 5408, 10816, 16224, 21632, 27029, 32768],
             -0.0625, True),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            for (lengths, heads, key_heads, expected, expected_threshold,
                 use_raw_gating) in cases:
                with self.subTest(lengths=lengths):
                    raw_cu = self._raw_cu(*lengths)
                    cu = self._cu(*lengths)
                    plan = cp.calc_cp_seqs(
                        cu, 32, heads, num_k_heads=key_heads,
                        tc01_enable_overlap_candidate=True,
                        tc01_packed_raw_gate=use_raw_gating,
                    )
                    threshold = cp.tc01_overlap_warmup_threshold(
                        cu, raw_cu, 32, heads, key_heads,
                        use_raw_gating=use_raw_gating,
                        indexed_state=True,
                        has_initial_state=True,
                    )
                    self.assertTrue(plan[0])
                    self.assertEqual(plan[1].tolist(), expected)
                    self.assertEqual(threshold, expected_threshold)

    def test_tc01_low_margin_threshold_rejects_lower_neighbors(self):
        cases = (
            ((8191,), 16, 2),
            ((22937, 9831), 16, 2),
            ((9830, 9831, 13107), 16, 2),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            for lengths, heads, key_heads in cases:
                with self.subTest(lengths=lengths):
                    raw_cu = self._raw_cu(*lengths)
                    cu = self._cu(*lengths)
                    self.assertIsNone(cp.tc01_overlap_warmup_threshold(
                        cu, raw_cu, 32, heads, key_heads,
                        use_raw_gating=False,
                        indexed_state=True,
                        has_initial_state=True,
                    ))

    def test_tc01_packed_raw_gate_thresholds_are_abi_scoped(self):
        cases = (
            ((3500,), 16, 2, 5, -0.75, None),
            ((411, 2716), 16, 2, 5, -0.75, None),
            ((3000, 864), 16, 2, 7, -0.75, None),
            ((1900, 1500, 1257), 16, 2, 9, -0.75, None),
            ((2048, 2048, 2048, 2048), 16, 2, 12, -0.75, None),
            ((6397, 1795), 32, 4, 4, -0.5, None),
            ((6957, 1235), 32, 4, 4, -0.3125, None),
            ((7173, 1019), 32, 4, 4, -0.25, None),
            ((1093, 7208), 32, 4, 4, -0.3125, None),
            ((13118, 14417, 5233), 32, 4, 11, -0.25, None),
            ((12521, 3863), 32, 4, 5, -0.25, None),
            ((13913, 2471), 32, 4, 5, -0.25, None),
            ((14309, 2075), 32, 4, 5, -0.25, None),
            ((1888, 12451, 2045), 32, 4, 6, -0.25, None),
            ((4770, 7863, 8574), 32, 4, 8, -0.34375, None),
            ((4173, 9174, 10003), 32, 4, 8, -0.25, None),
            ((1988, 13107, 1289), 32, 4, 6, -0.25, -4.0),
            ((27029, 5739), 64, 8, 6, -0.0625, None),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            for (lengths, heads, key_heads, segments, expected_threshold,
                 precomputed_threshold) in cases:
                with self.subTest(lengths=lengths):
                    raw_cu = self._raw_cu(*lengths)
                    cu = self._cu(*lengths)
                    self.assertEqual(cp._tc01_packed_raw_gate_spec(
                        raw_cu, heads, key_heads
                    ), (segments, expected_threshold))
                    self.assertEqual(cp.tc01_overlap_warmup_threshold(
                        cu, raw_cu, 32, heads, key_heads,
                        use_raw_gating=True,
                        indexed_state=True,
                        has_initial_state=True,
                    ), expected_threshold)
                    self.assertEqual(cp.tc01_overlap_warmup_threshold(
                        cu, raw_cu, 32, heads, key_heads,
                        use_raw_gating=False,
                        indexed_state=True,
                        has_initial_state=True,
                    ), precomputed_threshold)

    def test_tc01_packed_raw_gate_boundaries_reject_unmeasured_neighbors(self):
        cases = (
            ((3499,), 16, 2),
            ((3000, 863), 16, 2),
            ((1900, 1500, 1256), 16, 2),
            ((2048, 2048, 2048, 2047), 16, 2),
            ((823, 2716), 16, 2),
            ((7950, 242), 32, 4),
            ((4096, 4095), 32, 4),
            ((10000, 10976), 32, 4),
            ((27029, 5739), 64, 16),
        )
        for lengths, heads, key_heads in cases:
            with self.subTest(lengths=lengths):
                self.assertIsNone(cp._tc01_packed_raw_gate_spec(
                    self._raw_cu(*lengths), heads, key_heads
                ))

    def test_overlap_threshold_requires_tc01_race_free_raw_gating(self):
        raw_cu = self._raw_cu(16384, 16384)
        cu = self._cu(16384, 16384)
        common = (cu, raw_cu, 32, 48, 16)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            self.assertEqual(cp.tc01_overlap_warmup_threshold(
                *common, use_raw_gating=True, indexed_state=True,
                has_initial_state=True,
            ), -4.0)
            self.assertEqual(cp.tc01_overlap_warmup_threshold(
                *common, use_raw_gating=True, indexed_state=False,
                has_initial_state=False,
            ), -4.0)
            self.assertEqual(cp.tc01_overlap_warmup_threshold(
                *common, use_raw_gating=True, indexed_state=False,
                has_initial_state=True,
            ), -4.0)
            self.assertIsNone(cp.tc01_overlap_warmup_threshold(
                *common, use_raw_gating=False, indexed_state=True,
                has_initial_state=True,
            ))
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc02_device_model
        ):
            self.assertIsNone(cp.tc01_overlap_warmup_threshold(
                *common, use_raw_gating=True, indexed_state=True,
                has_initial_state=True,
            ))

    def test_tc01_dispatch_keeps_only_measured_suffixes(self):
        cases = (
            ((9999,), 8, False),
            ((10000,), 8, True),
            ((654, 1438), 8, False),
            ((8000, 23999), 8, False),
            ((8000, 24000), 8, True),
            ((4000, 4000, 23999), 8, False),
            ((4000, 4000, 24000), 8, True),
            ((2000, 2000, 2000, 24000), 8, False),
            ((32768,), 16, False),
            ((32768,), 32, False),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            for lengths, heads, expected_cp in cases:
                with self.subTest(lengths=lengths, heads=heads):
                    plan = cp.calc_cp_seqs(self._cu(*lengths), 32, heads)
                    self.assertEqual(plan[0], expected_cp)
                    self.assertEqual(plan[1] is not None, expected_cp)

    def test_tc01_admitted_shape_uses_one_legacy_plan(self):
        cu = self._cu(10000)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ), mock.patch.object(
            cp, "_calc_tc01_plan", wraps=cp._calc_tc01_plan
        ) as tc01_plan:
            plan = cp.calc_cp_seqs(cu, 32, 8, num_k_heads=2)

        self.assertTrue(plan[0])
        tc01_plan.assert_called_once()
        self.assertEqual(tc01_plan.call_args.args[3:5], (64, 2))

    def test_retired_tfu_environment_does_not_change_plan_or_cache(self):
        cu = self._cu(10000)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            with mock.patch.dict(os.environ, {cp.TC01_TFU_MODEL_ENV: "0"}):
                env0_plan = cp.calc_cp_seqs(cu, 32, 8, num_k_heads=2)
            with mock.patch.dict(os.environ, {cp.TC01_TFU_MODEL_ENV: "1"}):
                env1_plan = cp.calc_cp_seqs(cu, 32, 8, num_k_heads=2)

        self.assertTrue(env0_plan[0])
        self.assertIs(env1_plan, env0_plan)
        self.assertEqual(len(cp._cp_plan_cache), 1)

    def test_hg_does_not_change_tc01_real_gate_boundary(self):
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            hg2_plan = cp.calc_cp_seqs(
                self._cu(4000, 4000, 24000), 32, 8, num_k_heads=2
            )
            hg4_plan = cp.calc_cp_seqs(
                self._cu(4000, 4000, 24000), 32, 8, num_k_heads=4
            )

        self.assertTrue(hg2_plan[0])
        self.assertTrue(hg4_plan[0])
        self.assertEqual(hg2_plan[1].tolist(), hg4_plan[1].tolist())

    def test_tc02_h8_hg2_long_single_uses_balanced_segments(self):
        cases = (
            (18000, 2, 9),
            (18432, 2, 18),
            (27029, 2, 18),
            (32768, 2, 18),
            (33000, 2, 33),
            (27029, 4, 14),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc02_device_model
        ):
            for length, key_heads, expected_segments in cases:
                with self.subTest(length=length, key_heads=key_heads):
                    plan = cp.calc_cp_seqs(
                        self._cu(length), 32, 8, num_k_heads=key_heads)
                    self.assertTrue(plan[0])
                    self.assertEqual(plan[1].numel() - 1, expected_segments)

    def test_tc02_semantics_and_cache_ignore_tc01_env(self):
        cu = self._cu(448)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc02_device_model
        ):
            with mock.patch.dict(os.environ, {cp.TC01_TFU_MODEL_ENV: "0"}):
                env0_plan = cp.calc_cp_seqs(
                    cu, 32, 4, num_k_heads=2
                )
            with mock.patch.dict(os.environ, {cp.TC01_TFU_MODEL_ENV: "1"}):
                env1_plan = cp.calc_cp_seqs(
                    cu, 32, 4, num_k_heads=2
                )

        self.assertTrue(env0_plan[0])
        self.assertIs(env1_plan, env0_plan)
        self.assertEqual(len(cp._cp_plan_cache), 1)
        cached_model = cp._cp_plan_cache[0][0][-2]
        self.assertEqual(cached_model.architecture, "tc02")
        self.assertEqual(cached_model.fused_cu_count, 39)
        self.assertEqual(cached_model.fused_head_shards, 1)


class Tc02RealGatePlannerTest(unittest.TestCase):
    def setUp(self):
        cp._cp_plan_cache.clear()

    def tearDown(self):
        cp._cp_plan_cache.clear()

    @staticmethod
    def _raw_cu(*lengths):
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        return tuple(cumulative)

    @staticmethod
    def _cu(*lengths):
        return cp.torch.tensor(
            Tc02RealGatePlannerTest._raw_cu(*lengths),
            dtype=cp.torch.int32,
        )

    @staticmethod
    def _tc02_device_model(_raw_cu_seqlens):
        return cp._planner_execution_model_from_properties(8, 9, 39)

    def test_tc02_real_gate_aligned_suffix_boundaries(self):
        for heads, batches in cp._TC02_REAL_GATE_CP_MIN_TOKENS.items():
            for batch, threshold in batches.items():
                below = [threshold - 1] * batch
                above = [threshold] * batch
                with self.subTest(heads=heads, batch=batch, side="below"):
                    self.assertFalse(cp._tc02_real_gate_cp_allowed(
                        self._raw_cu(*below), 32, heads, 64
                    ))
                with self.subTest(heads=heads, batch=batch, side="above"):
                    self.assertTrue(cp._tc02_real_gate_cp_allowed(
                        self._raw_cu(*above), 32, heads, 64
                    ))

    def test_tc02_real_gate_long_tail_rectangles(self):
        cases = (
            ((3680, 368, 368, 368), 8, 14, True),
            ((3679, 368, 368, 368), 8, 14, False),
            ((3680, 368, 368, 368), 8, 13, False),
            ((5280, 512, 512), 16, 8, True),
            ((5279, 512, 512), 16, 8, False),
            ((5280, 3000, 3000), 16, 8, False),
            ((6400, 2000), 24, 8, True),
            ((6400, 3000), 24, 8, False),
            ((6560, 1000, 1000), 24, 8, True),
            ((6560, 2000, 2000), 24, 8, False),
        )
        for lengths, heads, segments, expected in cases:
            with self.subTest(lengths=lengths, heads=heads, segments=segments):
                self.assertEqual(cp._tc02_real_gate_cp_allowed(
                    self._raw_cu(*lengths), 32, heads, segments
                ), expected)

    def test_tc02_guard_rejects_uncalibrated_topologies(self):
        self.assertFalse(cp._tc02_real_gate_cp_allowed(
            self._raw_cu(32768), 32, 32, 64
        ))
        self.assertFalse(cp._tc02_real_gate_cp_allowed(
            self._raw_cu(32768), 64, 8, 64
        ))

    def test_tc02_dispatch_applies_real_gate_guard(self):
        cases = (
            ((447,), 8, 2, False),
            ((448,), 8, 2, True),
            ((1503,), 16, 8, False),
            ((1504,), 16, 8, True),
            ((2079,), 24, 8, False),
            ((2080,), 24, 8, True),
            ((32768,), 32, 16, False),
            ((32768,), 128, 16, False),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc02_device_model
        ):
            for lengths, heads, key_heads, expected_cp in cases:
                with self.subTest(lengths=lengths, heads=heads):
                    plan = cp.calc_cp_seqs(
                        self._cu(*lengths), 32, heads,
                        num_k_heads=key_heads,
                    )
                    self.assertEqual(plan[0], expected_cp)


class FusedFwdContractTest(unittest.TestCase):
    """Check shared-body inputs and launch forwarding without a PPU compiler."""

    VARIANTS = (
        ("fused_fwd_cp_cute_kernel", "FF_LAUNCH_LEGACY_S"),
        ("fused_fwd_cp_cute_indexed_kernel", "FF_LAUNCH_INDEXED_S"),
        ("fused_fwd_cp_cute_indexed_vd_inplace_kernel",
         "FF_LAUNCH_INDEXED_VD_INPLACE_S"),
        ("fused_fwd_cp_cute_runtime_kernel", "FF_LAUNCH_RUNTIME_S"),
        ("fused_fwd_cp_cute_runtime_head_major_kernel",
         "FF_LAUNCH_RUNTIME_HEAD_MAJOR_S"),
    )

    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        kernels = root / "src/prefill/flashqla/kernels"
        self.source = (kernels / "fused_fwd_cute.cu").read_text()
        self.body = (kernels / "fused_fwd_cute_body.inc").read_text()

    def _parameters(self, kernel):
        match = re.search(rf"\b{kernel}\((.*?)\)\s*\{{", self.source, re.S)
        self.assertIsNotNone(match, kernel)
        return [re.search(r"(\w+)\s*$", part).group(1)
                for part in match.group(1).split(",")]

    def test_shared_body_inputs_exist_in_every_variant(self):
        self.assertEqual(self.source.count(
            '#include "fused_fwd_cute_body.inc"'), len(self.VARIANTS))
        for kernel, _ in self.VARIANTS:
            with self.subTest(kernel=kernel):
                parameters = self._parameters(kernel)
                for name in ("num_warmup_chunks", "fallback_mask"):
                    self.assertIn(name, self.body)
                    self.assertIn(name, parameters)
                if kernel != "fused_fwd_cp_cute_kernel":
                    self.assertIn("raw_state_input", self.body)
                    self.assertIn("raw_state_input", parameters)

    def test_launch_arguments_match_every_variant_signature(self):
        for kernel, macro in self.VARIANTS:
            with self.subTest(kernel=kernel):
                launch = self.source.split(f"#define {macro}(", 1)[1].split(
                    "} while (0)", 1)[0]
                self.assertIn(kernel + "<", launch)
                arguments = launch.split(">>>(", 1)[1].split(");", 1)[0]
                arguments = re.sub(r"reinterpret_cast<[^>]+>\((\w+)\)",
                                   r"\1", arguments)
                arguments = [part.replace("\\", "").strip()
                             for part in arguments.split(",")]
                self.assertEqual(arguments, self._parameters(kernel))


if __name__ == "__main__":
    unittest.main()
