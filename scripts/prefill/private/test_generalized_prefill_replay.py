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

"""CPU-only parser and ABI-routing tests for generalized prefill replay."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "scripts" / "prefill" / "generalized_prefill_replay.py"
LIST_DIR = REPO_ROOT / "caselist" / "prefill"


def _load_runner():
    """Import the runner without requiring torch or a GPU on the host."""
    fake_torch = ModuleType("torch")
    fake_nn = ModuleType("torch.nn")
    fake_functional = ModuleType("torch.nn.functional")
    fake_torch.nn = fake_nn
    fake_nn.functional = fake_functional
    modules = {
        "torch": fake_torch,
        "torch.nn": fake_nn,
        "torch.nn.functional": fake_functional,
    }
    spec = importlib.util.spec_from_file_location(
        "pla_test_generalized_prefill_replay", RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load replay runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


replay = _load_runner()


class TestInventoryAndParser(unittest.TestCase):
    def test_current_prefill_inventory(self):
        inventory = replay.discover_case_lists(LIST_DIR, "all")
        records = replay.load_records(LIST_DIR, "all")
        self.assertEqual(len(inventory), 26)
        self.assertEqual(Counter(family for family, _ in inventory), {
            "gdn": 16,
            "kda": 10,
        })
        self.assertEqual(len(records), 34_042)
        self.assertEqual(Counter(record["family"] for record in records), {
            "gdn": 20_890,
            "kda": 13_152,
        })
        self.assertEqual([record["index"] for record in records], list(range(34_042)))
        self.assertEqual(records[20_890]["family"], "kda")

    def test_record_parser_keeps_cumulative_offsets_together(self):
        line = (
            "[GDN] gdn_attn:batch_size:3,num_heads_v:32,num_heads_qk:16,"
            "cu_seqlens_q:[0,24,184,358],gate_lower_bound:-5.0."
        )
        record = replay.parse_record(Path("synthetic.list"), 7, line, "gdn")
        self.assertEqual(record["cu"], (0, 24, 184, 358))
        self.assertEqual(record["seq_lens"], (24, 160, 174))
        self.assertEqual((record["N"], record["T"]), (3, 358))
        self.assertEqual(record["lower"], -5.0)

    def test_cli_defaults_and_explicit_optimized_abis(self):
        required = [
            "--list-dir", str(LIST_DIR),
            "--mode", "perf",
            "--output", "/tmp/replay.jsonl",
        ]
        defaults = replay.parse_args(required)
        self.assertEqual(defaults.gdn_path, "legacy")
        self.assertFalse(defaults.kda_trusted_host_offsets)
        self.assertFalse(defaults.kda_indexed_state)
        self.assertFalse(defaults.record_first_call)

        optimized = replay.parse_args(required + [
            "--gdn-path", "packed",
            "--kda-trusted-host-offsets",
            "--kda-indexed-state",
            "--record-first-call",
        ])
        self.assertEqual(optimized.gdn_path, "packed")
        self.assertTrue(optimized.kda_trusted_host_offsets)
        self.assertTrue(optimized.kda_indexed_state)
        self.assertTrue(optimized.record_first_call)

        for flag, value, message in (
            ("--warmup", "-1", "--warmup must be non-negative"),
            ("--repeat", "0", "--repeat must be positive"),
        ):
            with self.subTest(flag=flag):
                with patch("sys.stderr") as stderr:
                    with self.assertRaisesRegex(SystemExit, "2"):
                        replay.parse_args(required + [flag, value])
                self.assertIn(message, "".join(
                    call.args[0] for call in stderr.write.call_args_list
                ))


class TestGdnAbiRouting(unittest.TestCase):
    def setUp(self):
        self.values = {
            name: object()
            for name in (
                "packed_qkv", "raw_gate_a", "raw_beta_b", "A_log",
                "dt_bias", "cu_seqlens", "state_pool", "state_indices",
            )
        }
        self.record = {"H": 32, "Hg": 16, "cu": (0, 3, 8)}
        self.q, self.k, self.v = object(), object(), object()
        self.output = object()
        self.split_op = MagicMock(return_value=(self.q, self.k, self.v))
        self.legacy_op = MagicMock(
            return_value=(object(), object(), self.output, None, None),
        )
        self.packed_op = MagicMock(
            return_value=(object(), object(), self.output, None, None),
        )
        self.ops = self.split_op, self.legacy_op, self.packed_op

    def _common(self, auto_cp):
        return {
            "scale": 1.0 / (128**0.5),
            "initial_state": self.values["state_pool"],
            "initial_state_indices": self.values["state_indices"],
            "output_final_state": False,
            "auto_cp": auto_cp,
            "raw_cu_seqlens": self.record["cu"],
        }

    def test_split_route_uses_fused_split_and_raw_gating_api(self):
        result = replay.run_gdn_optimized_path(
            "split", self.values, self.record, auto_cp=False, ops=self.ops,
        )
        self.assertIs(result[2], self.output)
        self.split_op.assert_called_once_with(
            self.values["packed_qkv"], 32, 16,
        )
        self.packed_op.assert_not_called()
        self.legacy_op.assert_called_once_with(
            self.q,
            self.k,
            self.v,
            None,
            None,
            cu_seqlens=self.values["cu_seqlens"],
            normalize_qk=True,
            raw_gate_a=self.values["raw_gate_a"],
            raw_beta_b=self.values["raw_beta_b"],
            A_log=self.values["A_log"],
            dt_bias=self.values["dt_bias"],
            **self._common(False),
        )

    def test_packed_route_uses_six_positional_arguments(self):
        result = replay.run_gdn_optimized_path(
            "packed", self.values, self.record, auto_cp=True, ops=self.ops,
        )
        self.assertIs(result[2], self.output)
        self.split_op.assert_not_called()
        self.legacy_op.assert_not_called()
        self.packed_op.assert_called_once_with(
            self.values["packed_qkv"],
            self.values["raw_gate_a"],
            self.values["raw_beta_b"],
            self.values["A_log"],
            self.values["dt_bias"],
            self.values["cu_seqlens"],
            num_qk_heads=16,
            num_v_heads=32,
            **self._common(True),
        )

    def test_old_wheel_can_load_split_but_not_packed(self):
        fake_flashqla = ModuleType("pla.prefill.flashqla")
        fake_flashqla.chunk_gated_delta_rule_fwd = self.legacy_op
        fake_prefill = ModuleType("pla.prefill")
        fake_prefill.flashqla = fake_flashqla
        fake_pla = ModuleType("pla")
        fake_pla.prefill = fake_prefill
        modules = {
            "pla": fake_pla,
            "pla.prefill": fake_prefill,
            "pla.prefill.flashqla": fake_flashqla,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(replay, "_load_gdn_split_helper", return_value=self.split_op),
        ):
            split_ops = replay.load_gdn_optimized_ops("split")
            self.assertIsNone(split_ops[2])
            with self.assertRaisesRegex(RuntimeError, "gdn_prefill_packed_qkv_fwd"):
                replay.load_gdn_optimized_ops("packed")


class TestKdaAbiRouting(unittest.TestCase):
    def setUp(self):
        self.inputs = tuple(object() for _ in range(8))
        self.record = {"N": 1, "H": 16, "cu": (0, 2205), "lower": -5.0}
        self.out = object()
        self.final = object()
        replay.torch.bfloat16 = object()
        replay.torch.empty_like = MagicMock(return_value=self.out)
        replay.torch.empty = MagicMock(return_value=self.final)

    def test_trusted_offsets_use_new_out_param_keyword(self):
        op = MagicMock()
        out, final = replay.run_kda_candidate(
            op, True, self.inputs, self.record,
            exact_b1_cu=False, trusted_host_offsets=True,
        )
        self.assertIs(out, self.out)
        self.assertIs(final, self.final)
        q, k, v, g, beta, A_log, dt_bias, cu = self.inputs
        op.assert_called_once_with(
            q, k, v, g, beta, A_log, dt_bias,
            self.out, 1.0 / (128**0.5), -5.0, None, self.final, cu,
            raw_cu_seqlens=self.record["cu"],
        )

    def test_legacy_kda_call_keeps_old_positional_abi(self):
        expected = object()
        op = MagicMock(return_value=expected)
        actual = replay.run_kda_candidate(
            op, False, self.inputs, self.record,
            exact_b1_cu=False, trusted_host_offsets=False,
        )
        self.assertIs(actual, expected)
        q, k, v, g, beta, A_log, dt_bias, _ = self.inputs
        op.assert_called_once_with(
            q, k, v, g, beta, A_log, dt_bias,
            1.0 / (128**0.5), -5.0, None, None,
        )

    def test_indexed_state_uses_pool_without_compact_final(self):
        op = MagicMock()
        pool, indices = object(), object()
        out, returned_pool = replay.run_kda_candidate(
            op, True, self.inputs, self.record,
            exact_b1_cu=True, trusted_host_offsets=True,
            state_pool=pool, state_indices=indices,
        )
        self.assertIs(out, self.out)
        self.assertIs(returned_pool, pool)
        q, k, v, g, beta, A_log, dt_bias, cu = self.inputs
        op.assert_called_once_with(
            q, k, v, g, beta, A_log, dt_bias,
            self.out, 1.0 / (128**0.5), -5.0, None, None, cu,
            state_pool=pool,
            state_indices=indices,
            raw_cu_seqlens=self.record["cu"],
        )

    def test_kda_input_builder_uses_int32_cu(self):
        replay.torch.bfloat16 = object()
        replay.torch.float32 = object()
        replay.torch.int32 = object()
        replay.torch.full = MagicMock(side_effect=lambda *args, **kwargs: object())
        replay.torch.full_like = MagicMock(side_effect=lambda *args, **kwargs: object())
        replay.torch.zeros = MagicMock(side_effect=lambda *args, **kwargs: object())
        replay.torch.tensor = MagicMock(return_value=self.inputs[-1])
        record = dict(self.record, T=2205)
        replay.make_kda_inputs(record, False, seed=0)
        replay.torch.tensor.assert_called_once_with(
            record["cu"], dtype=replay.torch.int32, device="cuda",
        )


class TestJsonAbiMetadata(unittest.TestCase):
    def test_selected_metadata_records_actual_routes(self):
        args = SimpleNamespace(
            mode="accuracy", reference="serial", gdn_path="packed",
            gdn_auto_cp=True, kda_exact_b1_cu=True,
            kda_trusted_host_offsets=True,
            kda_indexed_state=True,
        )
        gdn = replay.selected_abi_metadata(args, {"family": "gdn"}, None)
        self.assertEqual(gdn["gdn_path"], "packed")
        self.assertEqual(gdn["gdn_reference_path"], "split")
        self.assertEqual(gdn["gdn_state_abi"], "indexed_pool")
        self.assertEqual(gdn["gdn_gate_distribution"], "accuracy_stress")
        self.assertIsNone(gdn["gdn_gate_decay_divisor"])
        self.assertIsNone(gdn["gdn_swa_ratio"])

        perf_args = SimpleNamespace(
            mode="perf", reference="serial", gdn_path="packed",
            gdn_auto_cp=True, kda_exact_b1_cu=True,
            kda_trusted_host_offsets=True,
            kda_indexed_state=True,
        )
        perf_gdn = replay.selected_abi_metadata(
            perf_args, {"family": "gdn"}, None,
        )
        self.assertEqual(
            perf_gdn["gdn_gate_distribution"],
            "flashqla_slow_decay_swa75",
        )
        self.assertEqual(perf_gdn["gdn_gate_decay_divisor"], 16)
        self.assertEqual(perf_gdn["gdn_swa_ratio"], 0.75)

        with patch.dict(
            replay.os.environ,
            {"PLA_FLASHKDA_CP_PLAN_CACHE_SIZE": "4"},
        ):
            kda = replay.selected_abi_metadata(
                args, {"family": "kda"}, out_params=True,
            )
        self.assertEqual(kda["kda_cu_dtype"], "int32")
        self.assertTrue(kda["kda_trusted_host_offsets"])
        self.assertEqual(kda["kda_state_abi"], "indexed_fp32_pool")
        self.assertEqual(kda["kda_beta_abi"], "raw_strided")
        self.assertEqual(kda["kda_cp_plan_cache_size_env"], "4")

    @staticmethod
    def _record(family):
        return {
            "index": 0,
            "family": family,
            "source": f"synthetic_{family}_op.list",
            "line": 1,
            "shape_id": "0123456789abcdef",
            "H": 16,
            "Hg": 8,
            "N": 1,
            "T": 32,
            "cu": (0, 32),
            "lower": -5.0,
        }

    def test_main_routes_packed_perf_and_writes_json_metadata(self):
        record = self._record("gdn")
        ops = (object(), object(), object())
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "gdn.jsonl"
            with (
                patch.object(replay, "discover_case_lists", return_value=[
                    ("gdn", Path(record["source"])),
                ]),
                patch.object(replay, "load_records", return_value=[record]),
                patch.object(replay, "load_gdn_optimized_ops", return_value=ops),
                patch.object(replay, "extension_identity", return_value={}),
                patch.object(
                    replay, "perf_gdn", return_value=(1.25, 1.75),
                ) as perf,
            ):
                status = replay.main([
                    "--list-dir", str(LIST_DIR),
                    "--family", "gdn",
                    "--mode", "perf",
                    "--output", str(output),
                    "--gdn-path", "packed",
                    "--record-first-call",
                ])
            self.assertEqual(status, 0)
            perf.assert_called_once()
            self.assertEqual(perf.call_args.args[5], "packed")
            self.assertIs(perf.call_args.args[6], ops)
            self.assertTrue(perf.call_args.args[7])
            row = json.loads(output.read_text())
        self.assertEqual(row["gdn_path"], "packed")
        self.assertEqual(row["gdn_gating_abi"], "raw")
        self.assertEqual(row["inventory_file_count"], 1)
        self.assertEqual(row["inventory_record_count"], 1)
        self.assertEqual(row["ms"], 1.25)
        self.assertEqual(row["first_call_ms"], 1.75)
        self.assertTrue(row["record_first_call"])

    def test_main_routes_kda_trusted_offsets_and_writes_json_metadata(self):
        record = self._record("kda")
        kda_op = object()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "kda.jsonl"
            with (
                patch.object(replay, "discover_case_lists", return_value=[
                    ("kda", Path(record["source"])),
                ]),
                patch.object(replay, "load_records", return_value=[record]),
                patch.object(
                    replay, "load_kda",
                    return_value=(kda_op, True, True, True),
                ),
                patch.object(replay, "extension_identity", return_value={}),
                patch.object(
                    replay, "perf_kda", return_value=(2.5, None),
                ) as perf,
            ):
                status = replay.main([
                    "--list-dir", str(LIST_DIR),
                    "--family", "kda",
                    "--mode", "perf",
                    "--output", str(output),
                    "--kda-trusted-host-offsets",
                    "--kda-indexed-state",
                ])
            self.assertEqual(status, 0)
            perf.assert_called_once()
            self.assertIs(perf.call_args.args[5], kda_op)
            self.assertTrue(perf.call_args.args[7])
            self.assertTrue(perf.call_args.args[8])
            self.assertFalse(perf.call_args.args[9])
            row = json.loads(output.read_text())
        self.assertEqual(row["kda_api"], "flashkdapro_out_params")
        self.assertEqual(row["kda_cu_dtype"], "int32")
        self.assertTrue(row["kda_trusted_host_offsets"])
        self.assertEqual(row["kda_state_abi"], "indexed_fp32_pool")
        self.assertEqual(row["ms"], 2.5)
        self.assertNotIn("first_call_ms", row)
        self.assertFalse(row["record_first_call"])


if __name__ == "__main__":
    unittest.main()
