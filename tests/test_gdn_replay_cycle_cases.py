# SPDX-License-Identifier: Apache-2.0
"""Offline cycle-matrix tests; OPF_ROOT explicitly selects canonical model JSON."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CASE_MODULE = ROOT / "scripts" / "decode" / "gdn_replayssm" / "cycle_cases.py"
STATIC_MODELS = {
    "Qwen3-Next-80B-A3B": {
        "H": 16, "HV": 32, "K": 128, "V": 128,
        "conv_width": 4, "linear_layers": 36,
    },
    "Qwen3.8-Flash-Next": {
        "H": 16, "HV": 48, "K": 128, "V": 128,
        "conv_width": 4, "linear_layers": 36,
    },
}
MODEL_NAMES = tuple(STATIC_MODELS)
MAX_NAME = "Qwen3.8-Max"
MAX_MODEL = {"H": 16, "HV": 128, "K": 128, "V": 128, "conv_width": 4, "linear_layers": 69}
_SPEC = importlib.util.spec_from_file_location("_gdn_cycle_cases_offline", CASE_MODULE)
cycle_cases = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cycle_cases
_SPEC.loader.exec_module(cycle_cases)
CycleCase = cycle_cases.CycleCase
load_models = cycle_cases.load_models
make_case = cycle_cases.make_case
matrix = cycle_cases.matrix


def _documents():
    """Synthetic parser fixtures, never evidence of canonical source validation."""
    documents = {}
    for name, info in STATIC_MODELS.items():
        config = {
            "linear_num_key_heads": info["H"],
            "linear_num_value_heads": info["HV"],
            "linear_key_head_dim": info["K"],
            "linear_value_head_dim": info["V"],
            "linear_conv_kernel_dim": info["conv_width"],
            "num_hidden_layers": 48,
            "full_attention_interval": 4,
        }
        if name == MODEL_NAMES[0]:
            config.update(model_type="qwen3_next", torch_dtype="bfloat16")
            documents[name] = config
        else:
            config.update(
                model_type="qwen4_exp_text", dtype="bfloat16",
                mamba_ssm_dtype="float32",
                layer_types=["linear_attention"] * 3 + ["full_attention"],
            )
            config["layer_types"] *= 12
            documents[name] = {"model_type": "qwen4_exp", "text_config": config}
    return documents


def _fixture_models(documents, *, indent=None):
    raw = {f"{name}.json": json.dumps(doc, indent=indent).encode("utf-8")
           for name, doc in documents.items()}
    with patch.object(Path, "read_bytes", autospec=True,
                      side_effect=lambda path: raw[path.name]):
        return load_models(ROOT / "__synthetic_opf_fixture_only__"), raw


def _simulate(case, cycles=3):
    """Integer-only cursor/flush simulation, NOT the numerical GDN recurrence."""
    wp = 0
    replay_base_position = 0
    base_state = 7
    accepted = 0
    pending = []
    result = []
    for _ in range(cycles):
        starts = []
        before = base_state
        for _ in range(case.L + 1):
            starts.append(wp)
            for _ in range(case.A):
                accepted += 1
                pending.append(accepted)
            wp += case.A
            flush = wp == case.L if case.mode == "decode" else wp + 2 * case.T > case.L
            if flush:
                committed = wp
                replay_base_position += committed
                # A toy state fold makes an accidental zero reset observable.
                for token in pending:
                    base_state = 2 * base_state + token
                pending.clear()
                wp = 0
                result.append({
                    "starts": starts, "committed": committed, "wp": wp,
                    "replay_base": replay_base_position,
                    "state_before": before, "state_after": base_state,
                })
                break
        else:
            raise AssertionError("simulation did not reach a flush")
    return result


class CycleCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = (load_models(os.environ["OPF_ROOT"])
                      if "OPF_ROOT" in os.environ else deepcopy(STATIC_MODELS))

    def case(self, **changes):
        return replace(make_case(self.models, MODEL_NAMES[0], 1, 1), **changes)

    def test_frozen_dataclass_and_locked_signatures(self):
        self.assertEqual(tuple(field.name for field in fields(CycleCase)), (
            "model", "tp", "B", "T", "A", "mode", "H", "HV", "K", "V", "L", "layers",
        ))
        case = CycleCase(MODEL_NAMES[0], 1, 1, 4, 4, "mtp", 16, 32)
        self.assertEqual((case.K, case.V, case.L, case.layers), (128, 128, 16, 1))
        self.assertIs(case.validate(), case)
        with self.assertRaises(FrozenInstanceError):
            case.A = 3
        self.assertEqual(tuple(inspect.signature(make_case).parameters), (
            "models", "model", "tp", "B", "T", "A", "mode", "L", "layers",
        ))
        for name, parameter in inspect.signature(matrix).parameters.items():
            if name != "models":
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_default_matrix_counts_unique_keys_and_single_layer(self):
        cases = matrix(self.models)
        self.assertEqual(len(cases), 1050)
        self.assertEqual(Counter(case.mode for case in cases), {"decode": 50, "mtp": 1000})
        self.assertEqual(len({case.key for case in cases}), 1050)
        for model in self.models:
            self.assertEqual(Counter(c.mode for c in cases if c.model == model),
                             {"decode": 25, "mtp": 500})
        for case in cases:
            with self.subTest(key=case.key):
                self.assertEqual(case.layers, 1)
                self.assertEqual(case.L, 16)
                self.assertRegex(case.key, r"\A[A-Za-z0-9_.-]+\Z")
                self.assertEqual(case.key, replace(case).key)
                self.assertEqual(case.H, (16 + case.tp - 1) // case.tp)
                global_hv = STATIC_MODELS[case.model]["HV"]
                self.assertEqual(case.HV, (global_hv + case.tp - 1) // case.tp)
                self.assertEqual(case.HV // case.H, global_hv // 16)
                self.assertGreater(case.HV, case.H)
                self.assertIs(case.validate(), case)

    def test_matrix_order_and_custom_iterable_axes(self):
        models = dict(reversed(tuple(self.models.items())))
        modes, tps, batches, widths = ("mtp", "decode"), (16, 1), (8, 1), (6, 2)
        cases = matrix(models, modes=iter(modes), tps=iter(tps),
                       batches=iter(batches), widths=iter(widths))
        expected = []
        for model in models:
            for tp in tps:
                for B in batches:
                    for mode in modes:
                        pairs = [(1, 1)] if mode == "decode" else [
                            (T, A) for T in widths for A in range(1, T + 1)
                        ]
                        expected.extend((model, tp, B, mode, T, A) for T, A in pairs)
        self.assertEqual([(c.model, c.tp, c.B, c.mode, c.T, c.A) for c in cases], expected)
        default = matrix(self.models)
        self.assertEqual((default[0].mode, default[0].T, default[0].A), ("decode", 1, 1))
        self.assertEqual([(c.T, c.A) for c in default[1:21]],
                         [(T, A) for T in (2, 4, 6, 8) for A in range(1, T + 1)])
        self.assertEqual((default[21].tp, default[21].B), (1, 8))
        self.assertEqual(default[525].model, MODEL_NAMES[1])

    def test_mode_subsets_and_acceptance_is_not_incremented(self):
        self.assertEqual(len(matrix(self.models, modes=("decode",))), 50)
        self.assertEqual(len(matrix(self.models, modes=("mtp",))), 1000)
        for T in (2, 4, 6, 8):
            for A in range(1, T + 1):
                case = make_case(self.models, MODEL_NAMES[0], 1, 1, T=T, A=A)
                self.assertEqual((case.T, case.A), (T, A))
        self.assertEqual(make_case(self.models, MODEL_NAMES[0], 1, 1, A=3).A, 3)
        with self.assertRaisesRegex(ValueError, "0 < A <= T"):
            make_case(self.models, MODEL_NAMES[0], 1, 1, A=5)
        with self.assertRaisesRegex(ValueError, "T=A=1"):
            make_case(self.models, MODEL_NAMES[0], 1, 1, mode="decode")

    def test_single_case_synthetic_oversize_and_explicit_layers(self):
        case = make_case(self.models, MODEL_NAMES[1], 1, 4096)
        self.assertEqual((case.B, case.H, case.HV, case.layers), (4096, 16, 48, 1))
        self.assertNotIn(4096, {c.B for c in matrix(self.models)})
        expanded = make_case(self.models, MODEL_NAMES[1], 16, 1, L=32, layers=36)
        self.assertEqual((expanded.H, expanded.HV, expanded.layers, expanded.rounds), (1, 3, 36, 7))
        self.assertTrue(all(c.layers == 36 for c in matrix(self.models, layers=36)))

    def test_validation_rejects_invalid_values(self):
        invalid = (
            {"model": "Qwen3.5-Flash-Next"}, {"tp": 3}, {"mode": "MTP"},
            {"H": 0}, {"HV": 0}, {"H": 32, "HV": 16}, {"HV": 33},
            {"K": 64}, {"V": 64}, {"L": 12}, {"L": 4},
            {"A": 5}, {"T": 9}, {"layers": 0}, {"layers": -1},
            {"mode": "decode"}, {"mode": "decode", "T": 1, "A": 2},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.case(**change).validate()
        for name in ("tp", "B", "T", "A", "H", "HV", "K", "V", "L", "layers"):
            for value in (0, -1, True, 1.0, "1", None):
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    self.case(**{name: value}).validate()
        for attr in ("rounds", "key"):
            with self.assertRaises(ValueError):
                getattr(self.case(A=0), attr)
        self.assertEqual(self.case(mode="decode", T=1, A=1, L=1).rounds, 1)
        self.assertEqual(self.case(T=1, A=1, L=2).rounds, 1)

    def test_matrix_rejects_bad_or_duplicate_axes(self):
        for options in (
            {"modes": ("other",)}, {"modes": ("mtp", "mtp")},
            {"tps": (1, 1)}, {"tps": (3,)}, {"tps": (0,)},
            {"batches": (1, 1)}, {"batches": (0,)},
            {"widths": (2, 2)}, {"widths": (0,)}, {"widths": (9,)},
            {"widths": (2.0,)}, {"L": 12}, {"L": 0}, {"layers": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                matrix(self.models, **options)
        self.assertEqual(matrix(self.models, modes=()), [])
        self.assertEqual(matrix({}, modes=()), [])

    def test_changed_model_metadata_cannot_claim_the_same_matrix(self):
        for field in ("H", "HV", "K", "V", "conv_width", "linear_layers"):
            models = deepcopy(self.models)
            models[MODEL_NAMES[0]][field] += 1
            for build in (lambda: matrix(models),
                          lambda: make_case(models, MODEL_NAMES[0], 1, 1)):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    build()
        with self.assertRaisesRegex(ValueError, "unknown model"):
            matrix({"Qwen3.5-Flash-Next": STATIC_MODELS[MODEL_NAMES[1]]})
        with self.assertRaisesRegex(ValueError, "missing"):
            make_case({}, MODEL_NAMES[0], 1, 1)

    def test_key_covers_identity_and_explicit_shapes(self):
        original = self.case()
        variants = (
            {"model": MODEL_NAMES[1]}, {"tp": 2}, {"B": 8}, {"T": 6},
            {"A": 3}, {"mode": "decode", "T": 1, "A": 1},
            {"L": 32}, {"layers": 36}, {"H": 8}, {"HV": 64},
        )
        keys = [original.key] + [replace(original, **change).key for change in variants]
        self.assertEqual(len(keys), len(set(keys)))

    def test_rounds_against_independent_flush_simulation_for_entire_matrix(self):
        for case in matrix(self.models):
            with self.subTest(case=case.key):
                cycles = _simulate(case)
                for index, cycle in enumerate(cycles, 1):
                    self.assertEqual(len(cycle["starts"]), case.rounds)
                    self.assertEqual(cycle["starts"], list(range(0, case.rounds * case.A, case.A)))
                    self.assertEqual(cycle["committed"], case.rounds * case.A)
                    self.assertEqual(cycle["wp"], 0)
                    self.assertEqual(cycle["replay_base"], index * case.rounds * case.A)
                    if case.mode == "mtp":
                        self.assertLessEqual(cycle["starts"][-1] + 2 * case.T, case.L)
                        self.assertGreater(cycle["committed"] + 2 * case.T, case.L)

    def test_hand_checked_mtp_boundaries(self):
        for T, A, rounds, starts in ((4, 1, 9, list(range(9))), (4, 4, 3, [0, 4, 8])):
            case = self.case(T=T, A=A)
            self.assertEqual(case.rounds, rounds)
            self.assertEqual(_simulate(case)[0]["starts"], starts)
        for A in range(1, 9):
            case = self.case(T=8, A=A)
            self.assertEqual(case.rounds, 1)
            self.assertEqual(_simulate(case)[0]["starts"], [0])

    def test_decode_flushes_once_after_exactly_sixteen_steps(self):
        case = make_case(self.models, MODEL_NAMES[0], 1, 1, T=1, A=1, mode="decode")
        cycles = _simulate(case)
        self.assertEqual(case.rounds, 16)
        self.assertEqual(len(cycles), 3)
        for cycle in cycles:
            self.assertEqual(cycle["starts"], list(range(16)))
            self.assertEqual(cycle["committed"], 16)
        self.assertEqual([cycle["replay_base"] for cycle in cycles], [16, 32, 48])

    def test_replay_base_and_toy_state_advance_across_cycles_not_reset(self):
        for case in (self.case(A=1), self.case(A=4), self.case(T=8, A=8),
                     self.case(mode="decode", T=1, A=1)):
            cycles = _simulate(case)
            for index, cycle in enumerate(cycles):
                expected = 7
                for token in range(1, cycle["replay_base"] + 1):
                    expected = 2 * expected + token
                self.assertEqual(cycle["state_after"], expected)
                if index:
                    self.assertEqual(cycle["state_before"], cycles[index - 1]["state_after"])
                    self.assertGreater(cycle["replay_base"], cycles[index - 1]["replay_base"])


class JsonSourceTests(unittest.TestCase):
    def test_fixture_shapes_scopes_and_exact_raw_byte_sha(self):
        models, raw = _fixture_models(_documents())
        self.assertEqual(tuple(models), MODEL_NAMES)
        for name, info in models.items():
            self.assertEqual({key: info[key] for key in STATIC_MODELS[name]}, STATIC_MODELS[name])
            self.assertEqual(set(info), set(STATIC_MODELS[name]) | {"source_path", "sha256"})
            self.assertEqual(info["sha256"], hashlib.sha256(raw[f"{name}.json"]).hexdigest())
            self.assertEqual(Path(info["source_path"]).name, f"{name}.json")
            self.assertTrue(Path(info["source_path"]).is_absolute())
        reformatted, _ = _fixture_models(_documents(), indent=2)
        self.assertNotEqual(models[MODEL_NAMES[0]]["sha256"], reformatted[MODEL_NAMES[0]]["sha256"])

    def test_source_parameter_changes_are_loud(self):
        changes = {
            "linear_num_key_heads": 8, "linear_num_value_heads": 64,
            "linear_key_head_dim": 64, "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 3, "num_hidden_layers": 44,
            "full_attention_interval": 3, "model_type": "renamed_model",
        }
        for model in MODEL_NAMES:
            for field, value in changes.items():
                documents = _documents()
                config = documents[model].get("text_config", documents[model])
                config[field] = value
                with self.subTest(model=model, field=field), self.assertRaisesRegex(ValueError, field):
                    _fixture_models(documents)
            for field in ("linear_num_key_heads", "linear_num_value_heads", "linear_conv_kernel_dim"):
                documents = _documents()
                documents[model].get("text_config", documents[model]).pop(field)
                with self.subTest(model=model, missing=field), self.assertRaisesRegex(ValueError, field):
                    _fixture_models(documents)

    def test_bf16_input_not_ssm_dtype_and_conflicting_aliases(self):
        for model in MODEL_NAMES:
            for dtype in ("float16", "float32", None):
                documents = _documents()
                config = documents[model].get("text_config", documents[model])
                key = "torch_dtype" if model == MODEL_NAMES[0] else "dtype"
                if dtype is None:
                    config.pop(key)
                else:
                    config[key] = dtype
                with self.subTest(model=model, dtype=dtype), self.assertRaisesRegex(ValueError, "dtype"):
                    _fixture_models(documents)
        documents = _documents()
        documents[MODEL_NAMES[0]]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "dtype"):
            _fixture_models(documents)

    def test_layer_schedule_and_nested_model_identity(self):
        documents = _documents()
        schedule = documents[MODEL_NAMES[1]]["text_config"]["layer_types"]
        schedule[0], schedule[3] = schedule[3], schedule[0]
        with self.assertRaisesRegex(ValueError, "layer_types"):
            _fixture_models(documents)
        documents = _documents()
        documents[MODEL_NAMES[1]]["model_type"] = "qwen3_next"
        with self.assertRaisesRegex(ValueError, "model_type"):
            _fixture_models(documents)
        documents = _documents()
        nested = documents[MODEL_NAMES[1]].pop("text_config")
        documents[MODEL_NAMES[1]].update({k: v for k, v in nested.items() if k != "model_type"})
        with self.assertRaisesRegex(ValueError, "text_config"):
            _fixture_models(documents)

    def test_opt_in_max_matrix_and_source_contract(self):
        config = {
            "model_type": "qwen3_5_moe_text", "dtype": "bfloat16",
            "linear_num_key_heads": 16, "linear_num_value_heads": 128,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "num_hidden_layers": 92,
            "full_attention_interval": 4,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
        }
        document = {"model_type": "qwen3_5_moe", "text_config": config}
        raw = json.dumps(document).encode("utf-8")
        with patch.object(Path, "read_bytes", return_value=raw):
            models = load_models(ROOT / "__synthetic_opf_fixture_only__", names=(MAX_NAME,))
            self.assertEqual(tuple(load_models(ROOT / "__synthetic_opf_fixture_only__",
                                               names=(MAX_NAME,))), (MAX_NAME,))
        self.assertEqual({key: models[MAX_NAME][key] for key in MAX_MODEL}, MAX_MODEL)
        cases = matrix(models)
        self.assertEqual(Counter(case.mode for case in cases), {"decode": 25, "mtp": 500})
        self.assertEqual(sorted({(c.tp, c.H, c.HV) for c in cases}),
                         [(1, 16, 128), (2, 8, 64), (4, 4, 32), (8, 2, 16), (16, 1, 8)])
        for field, value in (("num_hidden_layers", 48), ("linear_num_value_heads", 48),
                             ("model_type", "qwen4_exp_text")):
            changed = deepcopy(document)
            changed["text_config"][field] = value
            with patch.object(Path, "read_bytes", return_value=json.dumps(changed).encode()):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    load_models(ROOT / "__synthetic_opf_fixture_only__", names=(MAX_NAME,))
        with self.assertRaisesRegex(ValueError, "duplicate model"):
            load_models(ROOT / "__synthetic_opf_fixture_only__", names=(MAX_NAME, MAX_NAME))
        with self.assertRaisesRegex(ValueError, "unknown model"):
            load_models(ROOT / "__synthetic_opf_fixture_only__", names=("Qwen3.5-Max",))

    def test_bad_or_missing_json_never_falls_back(self):
        for raw in (b"{broken", b"[]", b"null"):
            with patch.object(Path, "read_bytes", return_value=raw):
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    load_models(ROOT / "__synthetic_opf_fixture_only__")
        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError("missing JSON")):
            with self.assertRaisesRegex(FileNotFoundError, "missing JSON"):
                load_models(ROOT / "__synthetic_opf_fixture_only__")

    def test_actual_json_sha_and_heads_offline_only(self):
        root = os.environ.get("OPF_ROOT")
        if not root:
            self.skipTest("set OPF_ROOT to validate supplied model configurations")
        root = Path(root).expanduser()
        paths = [root / "opf" / "models" / "config" / f"{name}.json" for name in MODEL_NAMES]
        models = load_models(root)
        for name, path in zip(MODEL_NAMES, paths):
            info = models[name]
            raw = path.read_bytes()
            document = json.loads(raw)
            config = document if name == MODEL_NAMES[0] else document["text_config"]
            self.assertEqual(info["source_path"], str(path.resolve()))
            self.assertEqual(info["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual({key: info[key] for key in STATIC_MODELS[name]}, STATIC_MODELS[name])
            self.assertEqual((config["linear_num_key_heads"], config["linear_num_value_heads"]),
                             (16, STATIC_MODELS[name]["HV"]))
        self.assertEqual(Counter(c.mode for c in matrix(models)), {"decode": 50, "mtp": 1000})

    def test_all_public_paths_without_torch_or_opf_imports(self):
        script = """
import importlib.util, json, sys

def no_accelerator_imports(event, args):
    if event == 'import' and args[0].split('.')[0] in ('torch', 'opf', 'triton', 'pla'):
        raise AssertionError('forbidden offline import: ' + args[0])

sys.addaudithook(no_accelerator_imports)
spec = importlib.util.spec_from_file_location('_isolated_cycle_cases', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
models = json.loads(sys.argv[2])
assert len(module.matrix(models)) == 1050
case = module.make_case(models, next(iter(models)), 1, 4096)
assert case.validate() is case and case.rounds == 3 and case.key
if sys.argv[3]:
    assert len(module.matrix(module.load_models(sys.argv[3]))) == 1050
assert not any(name.split('.')[0] in ('torch', 'opf', 'triton', 'pla') for name in sys.modules)
"""
        root = os.environ.get("OPF_ROOT", "")
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, str(CASE_MODULE), json.dumps(STATIC_MODELS), root],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    source = f"explicit JSON root: {os.environ['OPF_ROOT']}" if "OPF_ROOT" in os.environ else "fixed static model dictionaries"
    print(f"OFFLINE ONLY; matrix tests use {source}; no GPU execution or performance evidence.", file=sys.stderr)
    unittest.main(verbosity=2)
