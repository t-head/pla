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


def load_backend(torch_root):
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
        spec.loader.exec_module(module)
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

    def test_configure_toolchain_uses_only_native_ppu_sdk_paths(self):
        sdk = self.root / "ppu-sdk"
        (sdk / "bin").mkdir(parents=True)
        (sdk / "bin/hgcc").touch()
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
