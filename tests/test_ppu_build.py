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

"""CPU-only tests for the PPU-native setuptools backend."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class FakeBuildExt:
    def __init__(self):
        self.initialize_options()

    def initialize_options(self):
        self.build_temp = None


class FakeExtension:
    def __init__(self, name, sources, include_dirs=None, extra_compile_args=None,
                 extra_link_args=None):
        self.name = name
        self.sources = sources
        self.include_dirs = include_dirs or []
        self.extra_compile_args = extra_compile_args or {}
        self.extra_link_args = extra_link_args or []
        self.define_macros = []
        self.depends = []
        self.extra_objects = []


def load_backend(torch_root, source_text=None):
    fake_torch = types.ModuleType("torch")
    fake_torch.__path__ = [str(torch_root)]
    fake_torch._C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)
    fake_setuptools = types.ModuleType("setuptools")
    fake_command = types.ModuleType("setuptools.command")
    fake_build_ext = types.ModuleType("setuptools.command.build_ext")
    fake_build_ext.build_ext = FakeBuildExt
    modules = {
        "torch": fake_torch,
        "setuptools": fake_setuptools,
        "setuptools.command": fake_command,
        "setuptools.command.build_ext": fake_build_ext,
    }
    spec = importlib.util.spec_from_file_location("ppu_build_test_module", ROOT / "ppu_build.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        if source_text is None:
            spec.loader.exec_module(module)
        else:
            exec(compile(source_text, spec.origin, "exec"), module.__dict__)
    return module


class PPUBuildExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.torch_root = self.root / "torch"
        (self.torch_root / "include/torch/csrc/api/include").mkdir(parents=True)
        (self.torch_root / "lib").mkdir()
        self.backend = load_backend(self.torch_root)
        self.command = self.backend.PPUBuildExtension()
        self.command._hgcc = "/native/ppu/bin/hgcc"
        self.command._framework_includes = []
        self.command._link_flags = ["-ltorch", "-ltorch_cuda", "-lc10_cuda", "-lhggc_wrapper"]

    def tearDown(self):
        self.temporary.cleanup()

    def test_compile_rules_use_hgcc_and_collision_free_objects(self):
        first = self.root / "one/foo.cu"
        second = self.root / "two/foo.cc"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_text("// device\n")
        second.write_text("// host\n")
        include = self.root / "include"
        include.mkdir()
        extension = FakeExtension(
            "package._C",
            sources=[str(first), str(second)],
            include_dirs=[str(include)],
            extra_compile_args={"hgcc": ["-O2"], "cxx": ["-O1"]},
        )
        output = (self.root / "objects").resolve()
        expected = [self.command._object_path(output, source) for source in (first, second)]

        def fake_ninja(_):
            for path in expected:
                path.touch()

        self.command._run_ninja = fake_ninja
        objects = self.command.compile_sources(extension.sources, output, extension)
        ninja = (output / "build.ninja").read_text()

        self.assertEqual(objects, [str(path) for path in expected])
        self.assertNotEqual(expected[0].name, expected[1].name)
        self.assertIn("rule hgcc_compile", ninja)
        self.assertIn("rule cxx_compile", ninja)
        self.assertRegex(ninja, r"build .*foo_[0-9a-f]{12}\.o: hgcc_compile .*foo\.cu")
        self.assertRegex(ninja, r"build .*foo_[0-9a-f]{12}\.o: cxx_compile .*foo\.cc")
        self.assertIn("-DTORCH_EXTENSION_NAME=_C", ninja)
        self.assertIn("-D_GLIBCXX_USE_CXX11_ABI=1", ninja)
        hgcc_command = ninja.split("rule hgcc_compile\n", 1)[1].splitlines()[0]
        cxx_command = ninja.split("rule cxx_compile\n", 1)[1].splitlines()[0]
        self.assertIn("-Xcompiler=-fPIC", hgcc_command)
        self.assertIn(
            "c++ -O1 -std=c++17 -fPIC -DTORCH_EXTENSION_NAME", cxx_command
        )
        self.assertNotIn("-include", cxx_command)
        self.assertNotIn("nvcc", ninja)
        self.assertNotIn("CUDA_HOME", ninja)

    def test_header_dependency_is_emitted_for_kkt_consumers(self):
        header = self.root / "kkt_solve_cute_impl.cuh"
        header.write_text("#pragma once\n")
        sources = [self.root / f"kkt_solve_cute_tc{arch}.cu" for arch in ("01", "02")]
        for source in sources:
            source.write_text('#include "kkt_solve_cute_impl.cuh"\n')
        extension = FakeExtension("pla.prefill.flashqla._C", list(map(str, sources)))
        extension.depends = [str(header)]
        output = self.root / "objects"
        expected = [self.command._object_path(output, source) for source in sources]

        def fake_ninja(_):
            for path in expected:
                path.touch()

        self.command._run_ninja = fake_ninja
        self.command.compile_sources(extension.sources, output, extension,
                                     depends=extension.depends)
        edges = [line for line in (output / "build.ninja").read_text().splitlines()
                 if line.startswith("build ")]
        self.assertEqual(len(edges), 2)
        for edge in edges:
            self.assertTrue(edge.endswith(" | " + str(header.resolve())))

    def test_configure_toolchain_uses_only_native_ppu_sdk_paths(self):
        sdk = self.root / "ppu-sdk"
        (sdk / "bin").mkdir(parents=True)
        (sdk / "bin/hgcc").touch()
        (sdk / "bin/hgcc").chmod(0o755)
        (sdk / "include").mkdir()
        (sdk / "targets/x86_64-linux/include").mkdir(parents=True)
        (sdk / "lib").mkdir()
        with patch.dict("os.environ", {"PPU_SDK_DIR": str(sdk)}, clear=False):
            self.command._configure_toolchain()

        sdk = sdk.resolve()
        self.assertEqual(self.command._hgcc, str(sdk / "bin/hgcc"))
        self.assertIn(f"-L{sdk / 'lib'}", self.command._link_flags)
        self.assertNotIn("-lcuda", self.command._link_flags)
        self.assertNotIn("-lcudart", self.command._link_flags)

    def make_toolchain(self, name, compiler="hgcc"):
        sdk = (self.root / name).resolve()
        (sdk / "bin").mkdir(parents=True)
        driver = sdk / "bin" / compiler
        driver.touch()
        driver.chmod(0o755)
        (sdk / "include").mkdir()
        (sdk / "lib64/stubs").mkdir(parents=True)
        return sdk, driver

    def test_explicit_sdk_precedes_environment_and_path(self):
        sdk, driver = self.make_toolchain("explicit")
        other, _ = self.make_toolchain("environment")
        path_sdk, _ = self.make_toolchain("on-path")
        env = {"PPU_SDK_DIR": str(sdk), "PPU_SDK": str(other),
               "PATH": str(path_sdk / "bin")}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(self.command._find_toolchain(), (sdk, driver))

    def test_sdk_environment_is_used_without_explicit_override(self):
        sdk, driver = self.make_toolchain("environment")
        with patch.dict("os.environ", {"PPU_SDK": str(sdk), "PATH": ""}, clear=True):
            self.assertEqual(self.command._find_toolchain(), (sdk, driver))

    def test_path_fallback_keeps_headers_and_libraries_together(self):
        sdk, driver = self.make_toolchain("relocated")
        env = {"PPU_SDK_DIR": str(self.root / "missing"),
               "PPU_SDK": str(self.root / "old-sdk"), "PATH": str(sdk / "bin")}
        with patch.dict("os.environ", env, clear=True):
            self.command._configure_toolchain()
        self.assertEqual(self.command._hgcc, str(driver))
        self.assertEqual(self.command._ppu_sdk, sdk)
        self.assertIn(str(sdk / "include"), self.command._framework_includes)
        self.assertIn(f"-L{sdk / 'lib64'}", self.command._link_flags)
        self.assertIn(f"-L{sdk / 'lib64/stubs'}", self.command._link_flags)

    def test_non_executable_driver_falls_back_to_path(self):
        bad_sdk, bad_driver = self.make_toolchain("not-executable")
        bad_driver.chmod(0o644)
        sdk, driver = self.make_toolchain("working")
        env = {"PPU_SDK_DIR": str(bad_sdk), "PATH": str(sdk / "bin")}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(self.command._find_toolchain(), (sdk, driver))

    def test_driver_symlink_preserves_sdk_entry_directory(self):
        sdk = (self.root / "entry").resolve()
        (sdk / "bin").mkdir(parents=True)
        _, internal = self.make_toolchain("internal", "driver")
        driver = sdk / "bin/hgcc"
        driver.symlink_to(internal)
        with patch.dict("os.environ", {"PATH": str(sdk / "bin")}, clear=True):
            self.assertEqual(self.command._find_toolchain(), (sdk, driver))

    def test_default_root_is_checked_after_path(self):
        with patch.dict("os.environ", {"PATH": ""}, clear=True), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(self.backend.os, "access", return_value=True):
            self.assertEqual(
                self.command._find_toolchain(),
                (Path("/usr/local/PPU_SDK"), Path("/usr/local/PPU_SDK/bin/hgcc")),
            )

    def test_missing_compiler_reports_searched_paths(self):
        with patch.dict("os.environ", {"PATH": ""}, clear=True), \
                patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "/usr/local/PPU_SDK/bin/hgcc.*PATH"):
                self.command._find_toolchain()

    def test_compiler_rename_does_not_require_sdk_layout_rewrite(self):
        source = (ROOT / "ppu_build.py").read_text().replace("hgcc", "devicecc")
        backend = load_backend(self.torch_root, source)
        original, _ = self.make_toolchain("original")
        sdk, driver = self.make_toolchain("selected", "devicecc")
        env = {"PPU_SDK": str(original), "PATH": str(sdk / "bin")}
        with patch.dict("os.environ", env, clear=True):
            command = backend.PPUBuildExtension()
            command._configure_toolchain()
        self.assertEqual(command._devicecc, str(driver))
        self.assertEqual(command._ppu_sdk, sdk)
        self.assertIn(str(sdk / "include"), command._framework_includes)
        self.assertNotIn(str(original / "include"), command._framework_includes)
        self.assertIn(f"-L{sdk / 'lib64/stubs'}", command._link_flags)

    def test_link_invokes_hgcc_without_cuda_driver_libraries(self):
        extension = FakeExtension(
            "package._C",
            sources=[],
            extra_link_args=[
                "-Wl,-Bsymbolic",
                "-Wl,--version-script=/source/exports.map",
            ],
        )
        output = self.root / "package/_C.so"
        with patch.object(self.backend.subprocess, "check_call") as check_call:
            command = self.command._link(extension, ["one.o", "two.o"], output)

        check_call.assert_called_once_with(command)
        self.assertEqual(command[0], "/native/ppu/bin/hgcc")
        self.assertIn("-Xlinker=-Bsymbolic", command)
        self.assertIn("-Xlinker=--version-script=/source/exports.map", command)
        self.assertNotIn("-Wl,-Bsymbolic", command)
        self.assertIn("-lhggc_wrapper", command)
        self.assertNotIn("-lcuda", command)
        self.assertNotIn("-lcudart", command)


class ReplayNamespaceTests(unittest.TestCase):
    def load_setup(self, namespaces):
        setuptools = types.ModuleType("setuptools")
        setuptools.Extension = FakeExtension
        setuptools.find_packages = lambda **kwargs: []
        captured = {}
        setuptools.setup = lambda **kwargs: captured.update(kwargs)
        backend = types.ModuleType("ppu_build")
        backend.PPUBuildExtension = FakeBuildExt
        spec = importlib.util.spec_from_file_location("replay_setup_test", ROOT / "setup.py")
        setup = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"setuptools": setuptools, "ppu_build": backend}), \
                patch.dict("os.environ", {"PLA_BUILD_NAMESPACE": namespaces}), \
                patch.object(sys, "argv", ["setup.py"]):
            spec.loader.exec_module(setup)
        return setup, captured["ext_modules"]

    def test_native_namespace_builds_without_sglang_binding(self):
        setup, extensions = self.load_setup("pla.decode.kda_replayssm_vllm")
        self.assertEqual([ext.name for ext in extensions], ["pla.decode.kda_replayssm_vllm._C"])
        sources = {Path(source).relative_to(ROOT).as_posix() for source in extensions[0].sources}
        self.assertEqual(sources, {
            "src/decode/kda_replayssm/kda_mtp_verify.cu",
            "src/decode/kda_replayssm/kda_mtp_fold.cu",
            "src/decode/kda_replayssm_vllm/kda_replayssm_vllm.cpp",
            "src/decode/kda_replayssm_vllm/commit_plan.cu",
        })
        self.assertTrue(all(Path(source).is_file() for source in extensions[0].sources))
        spec = importlib.util.spec_from_file_location("replay_build_test", ROOT / "scripts/build.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        self.assertEqual(build.ALL_NAMESPACES, setup.ALL_NAMESPACES)
        self.assertFalse(build.selects_k_last_decode("pla.decode.kda_replayssm_vllm"))
        self.assertTrue(build.selects_k_last_decode("pla.decode"))

    def test_sglang_build_selectors_share_one_extension(self):
        legacy = "pla.decode.kda_mtp_sglang"
        backend = "pla.decode.kda_replayssm_sglang"
        for selector in (legacy, backend, f"{legacy},{backend}"):
            with self.subTest(selector=selector):
                _, extensions = self.load_setup(selector)
                self.assertEqual([ext.name for ext in extensions], [f"{backend}._C"])
                sources = {Path(source).relative_to(ROOT).as_posix()
                           for source in extensions[0].sources}
                self.assertEqual(sources, {
                    "src/decode/kda_replayssm/kda_mtp_verify.cu",
                    "src/decode/kda_replayssm/kda_mtp_fold.cu",
                    "src/decode/kda_replayssm_sglang/kda_replayssm_sglang.cpp",
                    "src/decode/kda_replayssm_sglang/kda_mtp_verify_host.cpp",
                    "src/decode/kda_replayssm_sglang/kda_mtp_fold_host.cpp",
                })
                self.assertTrue(all(Path(source).is_file() for source in extensions[0].sources))

    def test_legacy_sglang_imports_use_the_same_backend(self):
        legacy = "pla.decode.kda_mtp_sglang"
        backend = "pla.decode.kda_replayssm_sglang"
        extension = types.ModuleType(f"{backend}._C")
        torch = types.ModuleType("torch")
        torch.Tensor = object
        with patch.dict(sys.modules, {"torch": torch, extension.__name__: extension}), \
                patch.object(sys, "path", [str(ROOT), *sys.path]):
            self.assertIs(importlib.import_module(f"{legacy}._C"), extension)
            old = importlib.import_module(legacy)
            new = importlib.import_module(backend)
            self.assertIs(old._C, new._C)
            self.assertEqual(old.__all__, new.__all__)
            for name in old.__all__:
                self.assertIs(getattr(old, name), getattr(new, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
