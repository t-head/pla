# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only namespace boundaries, not native/GPU numerical validation.

Run: python -B tests/test_gdn_replay_namespaces.py
Only torch and the single shared native extension are faked; package imports
execute the actual Python sources, without loading torch or probing a device.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, is_dataclass
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
COMMON = "pla.decode.gdn_replayssm"
SGLANG = "pla.decode.gdn_replayssm_sglang"
HLM = "pla.decode.gdn_replayssm_hlm"
RETIRED = "pla.decode.gdn_replay"
CONTRACTS = (
    "HistoryFormat", "NumericPolicy", "ReplayInputs", "ReplayHistory",
    "ReplayMapping", "ReplayStep", "MaterializeWork",
)
COMMON_CALLS = (
    "gdn_replay_verify", "gdn_replay_decode", "gdn_replay_materialize",
    "gdn_replay_workspace_elements",
)
SG_CALLS = (
    "gdn_replayssm_spec_decode", "commit_gdn_replayssm_spec",
    "commit_gdn_replayssm_circular",
)
HLM_CALLS = (
    "fused_recurrent_gated_delta_rule_cached_replay_update",
    "commit_gdn_cached_replay_history_layers",
)


@contextmanager
def isolated_packages():
    torch = ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    for name in ("float16", "bfloat16", "float32"):
        setattr(torch, name, object())
    torch.version = SimpleNamespace(hip=None)
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    native = ModuleType(COMMON + "._C")
    for name in ("verify", "cached_verify", "materialize", "commit", "cursors"):
        setattr(native, name, Mock(name=name))
    # Restore both the module cache and search path even when an import fails.
    # Fresh real parents prevent stale child attributes from hiding dependencies.
    with patch.dict(sys.modules), patch.object(sys, "path", [str(ROOT), *sys.path]):
        for name in tuple(sys.modules):
            if name == "pla" or name.startswith("pla."):
                del sys.modules[name]
        sys.modules["torch"] = torch
        sys.modules[native.__name__] = native
        yield native


class NamespaceTests(unittest.TestCase):
    def assert_common_boundary(self, common):
        self.assertEqual(set(common.__all__), set(CONTRACTS + COMMON_CALLS))
        for name in COMMON_CALLS:
            self.assertTrue(callable(getattr(common, name)))
        for name in (*SG_CALLS, *HLM_CALLS, "sglang", "trtllm"):
            self.assertFalse(hasattr(common, name), name)

    def test_common_import_does_not_load_or_export_adapters(self):
        with isolated_packages():
            common = importlib.import_module(COMMON)
            self.assert_common_boundary(common)
            for name in (SGLANG, HLM, RETIRED, COMMON + ".sglang", COMMON + ".trtllm"):
                self.assertNotIn(name, sys.modules)

    def test_adapter_first_imports_use_only_the_shared_extension(self):
        for order in ((SGLANG, HLM), (HLM, SGLANG)):
            with self.subTest(order=order), isolated_packages() as native:
                for name in order:
                    importlib.import_module(name)
                common = importlib.import_module(COMMON)
                sg = sys.modules[SGLANG]
                hlm = sys.modules[HLM]
                self.assert_common_boundary(common)
                for adapter, names in ((sg, SG_CALLS), (hlm, HLM_CALLS)):
                    self.assertEqual(set(adapter.__all__), set(names))
                    for name in names:
                        self.assertTrue(callable(getattr(adapter, name)))
                self.assertIs(common._verify, native.verify)
                self.assertIs(common._cached_verify, native.cached_verify)
                self.assertIs(common._materialize, native.materialize)
                self.assertIs(sg._verify, native.verify)
                self.assertIs(sg._commit, native.commit)
                self.assertIs(sg._cursors, native.cursors)
                self.assertIs(hlm._cached_verify, native.cached_verify)
                self.assertEqual(
                    {name for name in sys.modules if name.startswith("pla.") and name.endswith("._C")},
                    {COMMON + "._C"},
                )

    def test_public_contracts_are_identical_and_frozen(self):
        with isolated_packages():
            hlm = importlib.import_module(HLM)
            common = importlib.import_module(COMMON)
            contracts = importlib.import_module(COMMON + ".contracts")
            for name in CONTRACTS:
                with self.subTest(contract=name):
                    cls = getattr(common, name)
                    self.assertIs(cls, getattr(contracts, name))
                    self.assertIs(cls, getattr(hlm, name))
                    if name != "HistoryFormat":
                        self.assertTrue(is_dataclass(cls))
                        self.assertTrue(cls.__dataclass_params__.frozen)
            step = hlm.ReplayStep(lengths=object())
            self.assertIsInstance(step, common.ReplayStep)
            self.assertIsNone(step.starts)
            self.assertIsNone(step.buffers)
            with self.assertRaises(FrozenInstanceError):
                step.starts = object()
            self.assertEqual(common.NumericPolicy(), hlm.NumericPolicy())

    def test_retired_root_and_nested_adapters_are_not_importable(self):
        self.assertFalse((ROOT / "pla" / "decode" / "gdn_replay").exists())
        with isolated_packages():
            importlib.import_module(COMMON)
            for name in (RETIRED, COMMON + ".sglang", COMMON + ".trtllm",
                         SGLANG + "._C", HLM + "._C"):
                with self.subTest(module=name):
                    with self.assertRaises(ModuleNotFoundError) as caught:
                        importlib.import_module(name)
                    self.assertEqual(caught.exception.name, name)

    def test_each_build_selector_uses_one_shared_extension(self):
        path = ROOT / "setup.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        selection = [node for node in tree.body if (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALL_NAMESPACES" for t in node.targets)
        ) or isinstance(node, ast.FunctionDef) and node.name == "_requested_namespaces"]
        native = next(node for node in tree.body if isinstance(node, ast.If) and any(
            isinstance(child, ast.Constant) and child.value == COMMON + "._C" for child in ast.walk(node)))
        decode_dir = ROOT / "src" / "decode"
        shared_source_dirs = {decode_dir / package.rsplit(".", 1)[-1] / "common"
                              for package in (COMMON, SGLANG, HLM)}
        hlm = decode_dir / HLM.rsplit(".", 1)[-1]
        specialized = {
            "ppu10": hlm / "tc01" / "gdn_replayssm_cached_ppu10.cu",
            "ppu15": hlm / "tc02" / "gdn_replayssm_cached_ppu15.cu",
        }
        for names in (COMMON, SGLANG, HLM, ",".join((SGLANG, HLM, COMMON)), ""):
            with self.subTest(selector=names), patch.dict(os.environ, {"PLA_BUILD_NAMESPACE": names}), patch.object(sys, "argv", ["setup.py"]):
                namespace = dict(os=os, sys=sys)
                exec(compile(ast.Module(selection, type_ignores=[]), str(path), "exec"), namespace)
                selected = namespace["_requested_namespaces"]()
                build = dict(NAMESPACES=selected, os=os, here=str(ROOT), HGCC_THREADS="1", ext_modules=[], Path=Path,
                             Extension=lambda **kwargs: SimpleNamespace(**kwargs))
                exec(compile(ast.Module([native], type_ignores=[]), str(path), "exec"), build)
                extension, = build["ext_modules"]
                self.assertEqual(extension.name, COMMON + "._C")
                sources = set(map(Path, extension.sources))
                self.assertEqual({source.parent for source in sources - set(specialized.values())},
                                 shared_source_dirs)
                for target, source in specialized.items():
                    self.assertEqual(getattr(extension, f"{target}_sources"), [str(source)])
                    self.assertIn(source, sources)
                    self.assertTrue({str(header) for header in source.parent.glob("*.cuh")}
                                    <= set(extension.depends))
                self.assertEqual({flag for flag in extension.extra_compile_args["hgcc"] if flag.startswith("-arch=")},
                                 {"-arch=ppu_10", "-arch=ppu_15"})
                for source in extension.sources + extension.depends:
                    self.assertTrue(Path(source).is_file(), source)

    def test_active_sources_have_no_retired_imports_or_script_paths(self):
        paths = list((ROOT / "scripts" / "decode" / "gdn_replayssm").rglob("*.py"))
        paths.extend(ROOT / "tests" / name for name in (
            "test_gdn_replay.py", "test_gdn_replay_contracts.py",
            "test_gdn_replay_cached.py",
        ))
        for package in (COMMON, SGLANG, HLM):
            paths.extend((ROOT / package.replace(".", "/")).glob("*.py"))
        for path in paths:
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                        names = [node.value]
                        self.assertNotIn("scripts/decode/gdn_replay/", node.value)
                    else:
                        continue
                    for name in names:
                        self.assertFalse(name == RETIRED or name.startswith(RETIRED + "."), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
