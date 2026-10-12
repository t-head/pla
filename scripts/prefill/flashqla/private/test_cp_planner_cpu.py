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
    # Keep the package hierarchy so the real _sanity/_bypass load, but do not import the GPU extension.
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


class Tc01TfuPlannerModelTest(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(os.environ, {"PLA_SANITY_CHECK": "0", "PLA_BYPASS_EXEC": "0"})
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
    def _tc01_device_model(_raw_cu_seqlens):
        return cp._planner_execution_model_from_properties(8, 0, 64)

    @staticmethod
    def _tc02_device_model(_raw_cu_seqlens):
        return cp._planner_execution_model_from_properties(8, 9, 39)

    def test_default_on_exact_zero_rollback_and_architecture_models(self):
        with mock.patch.dict(os.environ, {cp.TC01_TFU_MODEL_ENV: "0"}):
            tc01_rollback = cp._planner_execution_model_from_properties(
                8, 0, 64)
        self.assertEqual(tc01_rollback.architecture, "tc01")
        self.assertEqual(tc01_rollback.physical_cu_count, 64)
        self.assertEqual(tc01_rollback.fused_cu_count, 64)
        self.assertEqual(tc01_rollback.fused_head_shards, 2)
        self.assertFalse(tc01_rollback.tc01_tfu_union_enabled)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cp.TC01_TFU_MODEL_ENV, None)
            tc01_default = cp._planner_execution_model_from_properties(
                8, 0, 64)
        with mock.patch.dict(os.environ, {cp.TC01_TFU_MODEL_ENV: "1"}):
            tc01_env1 = cp._planner_execution_model_from_properties(8, 0, 64)
            tc02_tfu_env = cp._planner_execution_model_from_properties(
                8, 9, 39)
        self.assertEqual(tc01_default, tc01_env1)
        self.assertEqual(tc01_default.architecture, "tc01")
        self.assertEqual(tc01_default.physical_cu_count, 64)
        self.assertEqual(tc01_default.fused_cu_count, 20)
        self.assertEqual(tc01_default.fused_head_shards, 1)
        self.assertTrue(tc01_default.tc01_tfu_union_enabled)
        self.assertIsNone(tc01_default.preprocessing_cu_count)
        self.assertEqual(
            tc01_default.preprocessing_model, "architecture_e2e_thresholds")
        self.assertEqual(tc02_tfu_env.fused_cu_count, 39)
        self.assertEqual(tc02_tfu_env.fused_head_shards, 1)
        self.assertFalse(tc02_tfu_env.tc01_tfu_union_enabled)

        for enabled_value in (
            "", "false", "true", "TRUE", " 1", "01", "2", "-1",
        ):
            with self.subTest(value=enabled_value):
                with mock.patch.dict(
                    os.environ,
                    {cp.TC01_TFU_MODEL_ENV: enabled_value},
                ):
                    model = cp._planner_execution_model_from_properties(
                        8, 0, 64)
                self.assertEqual(model.fused_cu_count, 20)
                self.assertEqual(model.fused_head_shards, 1)
                self.assertTrue(model.tc01_tfu_union_enabled)

    def test_tc01_union_uses_tfu_candidate_only_after_legacy_serial(self):
        cu = self._cu(16384)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ), mock.patch.object(
            cp, "_calc_tc01_plan", wraps=cp._calc_tc01_plan
        ) as tc01_plan, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cp.TC01_TFU_MODEL_ENV, None)
            plan = cp.calc_cp_seqs(cu, 32, 32)

        self.assertTrue(plan[0])
        self.assertEqual(plan[1].numel() - 1, 8)
        self.assertEqual(tc01_plan.call_count, 2)
        legacy_args = tc01_plan.call_args_list[0].args
        candidate_args = tc01_plan.call_args_list[1].args
        self.assertEqual(legacy_args[3:5], (64, 2))
        self.assertEqual(candidate_args[3:5], (20, 1))

    def test_tc01_union_preserves_legacy_cp_layout_without_tfu_replan(self):
        cu = self._cu(1056)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ), mock.patch.object(
            cp, "_calc_tc01_plan", wraps=cp._calc_tc01_plan
        ) as tc01_plan, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cp.TC01_TFU_MODEL_ENV, None)
            plan = cp.calc_cp_seqs(cu, 32, 12)

        self.assertTrue(plan[0])
        self.assertEqual(plan[1].numel() - 1, 5)
        tc01_plan.assert_called_once()
        self.assertEqual(tc01_plan.call_args.args[3:5], (64, 2))

    def test_tc01_default_env1_share_cache_and_zero_has_distinct_key(self):
        cu = self._cu(16384)
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(cp.TC01_TFU_MODEL_ENV, None)
                default_plan = cp.calc_cp_seqs(cu, 32, 32)
            with mock.patch.dict(
                os.environ, {cp.TC01_TFU_MODEL_ENV: "1"}
            ):
                env1_plan = cp.calc_cp_seqs(cu, 32, 32)
            with mock.patch.dict(
                os.environ, {cp.TC01_TFU_MODEL_ENV: "0"}
            ):
                rollback_plan = cp.calc_cp_seqs(cu, 32, 32)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(cp.TC01_TFU_MODEL_ENV, None)
                default_plan_again = cp.calc_cp_seqs(cu, 32, 32)

        self.assertTrue(default_plan[0])
        self.assertIs(env1_plan, default_plan)
        self.assertIs(default_plan_again, default_plan)
        self.assertFalse(rollback_plan[0])
        self.assertIsNot(rollback_plan, default_plan)
        self.assertEqual(len(cp._cp_plan_cache), 2)
        cached_models = {entry[0][-2] for entry in cp._cp_plan_cache}
        self.assertEqual(
            {model.fused_cu_count for model in cached_models}, {20, 64})
        self.assertEqual(
            {model.tc01_tfu_union_enabled for model in cached_models},
            {False, True},
        )

    def test_tc01_exact_zero_boundaries_remain_legacy_model(self):
        cases = (
            ((1823,), 16, False, 1),
            ((1824,), 16, True, 4),
            ((1024, 1024), 6, True, 8),
            ((1024, 1024), 16, False, 2),
            ((16384,), 32, False, 1),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ), mock.patch.dict(
            os.environ, {cp.TC01_TFU_MODEL_ENV: "0"}
        ):
            for lengths, heads, expected_cp, expected_segments in cases:
                with self.subTest(lengths=lengths, heads=heads):
                    plan = cp.calc_cp_seqs(self._cu(*lengths), 32, heads)
                    actual_segments = (
                        plan[1].numel() - 1 if plan[0] else len(lengths))
                    self.assertEqual(plan[0], expected_cp)
                    self.assertEqual(actual_segments, expected_segments)

    def test_tc01_tfu_union_boundaries_pin_selection_and_segments(self):
        cases = (
            # Existing CP decisions and deeper legacy layouts stay exact.
            (1024, 12, True, 4),
            (1056, 12, True, 5),
            (1824, 16, True, 4),
            (2048, 16, True, 4),
            # The TFU model may add CP only where legacy remains serial.
            (16384, 32, True, 8),
            (16416, 32, False, 1),
        )
        with mock.patch.object(
            cp, "_device_execution_model", side_effect=self._tc01_device_model
        ), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cp.TC01_TFU_MODEL_ENV, None)
            for length, heads, expected_cp, expected_segments in cases:
                with self.subTest(length=length, heads=heads):
                    plan = cp.calc_cp_seqs(self._cu(length), 32, heads)
                    actual_segments = (
                        plan[1].numel() - 1 if plan[0] else 1)
                    self.assertEqual(plan[0], expected_cp)
                    self.assertEqual(actual_segments, expected_segments)

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
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(cp.TC01_TFU_MODEL_ENV, None)
                default_plan = cp.calc_cp_seqs(
                    cu, 32, 4, num_k_heads=2)
            with mock.patch.dict(
                os.environ, {cp.TC01_TFU_MODEL_ENV: "0"}
            ):
                env0_plan = cp.calc_cp_seqs(
                    cu, 32, 4, num_k_heads=2)
            with mock.patch.dict(
                os.environ, {cp.TC01_TFU_MODEL_ENV: "1"}
            ):
                env1_plan = cp.calc_cp_seqs(
                    cu, 32, 4, num_k_heads=2)

        self.assertTrue(env0_plan[0])
        self.assertIs(default_plan, env0_plan)
        self.assertIs(env1_plan, env0_plan)
        self.assertEqual(len(cp._cp_plan_cache), 1)
        cached_model = cp._cp_plan_cache[0][0][-2]
        self.assertEqual(cached_model.architecture, "tc02")
        self.assertEqual(cached_model.fused_cu_count, 39)
        self.assertEqual(cached_model.fused_head_shards, 1)


if __name__ == "__main__":
    unittest.main()
