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

"""Setuptools backend for building PLA extensions with the PPU-native toolchain."""

import hashlib
import os
import shlex
import subprocess
import sysconfig
from pathlib import Path

import torch
from setuptools.command.build_ext import build_ext


class PPUBuildExtension(build_ext):
    """Compile and link extension modules directly with the PPU toolchain."""

    def initialize_options(self):
        super().initialize_options()
        self._ppu_sdk = None
        self._hgcc = None
        self._framework_includes = None
        self._host_compat_flags = None
        self._link_flags = None

    def build_extensions(self):
        self._configure_toolchain()
        for extension in self.extensions:
            self.build_extension(extension)

    def _configure_toolchain(self):
        sdk = Path(
            os.environ.get("PPU_SDK_DIR")
            or os.environ.get("PPU_SDK")
            or "/usr/local/PPU_SDK"
        ).resolve()
        hgcc = sdk / "bin" / "hgcc"
        if not hgcc.is_file():
            raise RuntimeError(
                f"Cannot find the PPU compiler at {hgcc}. "
                "Set PPU_SDK_DIR to the native PPU SDK root."
            )

        torch_root = Path(torch.__path__[0]).resolve()
        compat_wrapper = torch_root / ".ppu_compat" / "compatible_wrapper.h"
        include_dirs = [
            torch_root / "include",
            torch_root / "include" / "torch" / "csrc" / "api" / "include",
            Path(sysconfig.get_path("include")),
            sdk / "include",
            sdk / "targets" / "x86_64-linux" / "include",
        ]
        self._framework_includes = list(dict.fromkeys(
            str(path.resolve()) for path in include_dirs if path.is_dir()
        ))
        # PPU PyTorch headers still expose the upstream CUDA-named C++ ABI. The
        # wheel-provided wrapper maps that host ABI to native PPU SDK types. It
        # must not be injected into .cu files, which use native hggc headers.
        self._host_compat_flags = (
            ["-include", str(compat_wrapper)] if compat_wrapper.is_file() else []
        )

        library_dirs = [torch_root / "lib", sdk / "lib", sdk / "lib64"]
        self._link_flags = [f"-L{path}" for path in library_dirs if path.is_dir()]
        self._link_flags += [
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch_python",
            "-lc10",
            "-lc10_cuda",
            "-lhggc_wrapper",
            "-lhg_wrapper",
        ]
        self._ppu_sdk = sdk
        self._hgcc = str(hgcc)

    @staticmethod
    def _quote(items):
        return " ".join(shlex.quote(str(item)) for item in items)

    @staticmethod
    def _ninja_path(path):
        return str(path).replace("$", "$$").replace(" ", "$ ").replace(":", "$:")

    @staticmethod
    def _object_path(output_dir, source):
        source = Path(source).resolve()
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        return Path(output_dir) / f"{source.stem}_{digest}.o"

    def _base_flags(self, extension):
        module = extension.name.rsplit(".", 1)[-1]
        abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        common = [
            f"-DTORCH_EXTENSION_NAME={module}",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        ]
        for name, value in extension.define_macros or []:
            common.append(f"-D{name}" if value is None else f"-D{name}={value}")
        return common

    def compile_sources(self, sources, output_dir, extension, extra_postargs=None, depends=None):
        """Compile one source group and return its collision-free object paths."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        sources = [Path(source).resolve() for source in sources]
        objects = [self._object_path(output_dir, source) for source in sources]
        includes = list(extension.include_dirs or []) + self._framework_includes
        include_flags = [f"-I{Path(path).resolve()}" for path in dict.fromkeys(includes)]
        postargs = extra_postargs or extension.extra_compile_args or {}
        hgcc_flags = list(postargs.get("hgcc", []))
        hgcc_flags += [
            "-Xcompiler=-fPIC",
            "-U__HGGC_NO_HALF_OPERATORS__",
            "-U__HGGC_NO_HALF_CONVERSIONS__",
            "-U__HGGC_NO_HALF2_OPERATORS__",
            "-U__HGGC_NO_BFLOAT16_CONVERSIONS__",
        ]
        cxx_flags = list(postargs.get("cxx", []))
        if not any(flag.startswith("-std=") for flag in cxx_flags):
            cxx_flags.append("-std=c++17")
        base_flags = self._base_flags(extension)

        ninja = output_dir / "build.ninja"
        with ninja.open("w") as stream:
            stream.write("ninja_required_version = 1.3\n\n")
            stream.write("rule hgcc_compile\n")
            stream.write(
                "  command = " + self._quote(
                    [self._hgcc] + hgcc_flags + base_flags + include_flags
                    + ["-c", "$in", "-o", "$out"]
                ).replace("'$in'", "$in").replace("'$out'", "$out") + "\n"
            )
            stream.write("  description = HGCC $in\n\n")
            stream.write("rule cxx_compile\n")
            stream.write(
                "  command = " + self._quote(
                    [os.environ.get("CXX", "c++")] + cxx_flags + ["-fPIC"]
                    + self._host_compat_flags + base_flags + include_flags
                    + ["-c", "$in", "-o", "$out"]
                ).replace("'$in'", "$in").replace("'$out'", "$out") + "\n"
            )
            stream.write("  description = CXX $in\n\n")
            implicit = ""
            if depends:
                implicit = " | " + " ".join(self._ninja_path(Path(path).resolve()) for path in depends)
            for source, obj in zip(sources, objects):
                rule = "hgcc_compile" if source.suffix == ".cu" else "cxx_compile"
                stream.write(
                    f"build {self._ninja_path(obj)}: {rule} "
                    f"{self._ninja_path(source)}{implicit}\n"
                )
            stream.write("\ndefault " + " ".join(self._ninja_path(obj) for obj in objects) + "\n")

        self._run_ninja(ninja)
        missing = [str(path) for path in objects if not path.is_file()]
        if missing:
            raise RuntimeError("PPU compilation did not produce objects: " + ", ".join(missing))
        return [str(path) for path in objects]

    def _run_ninja(self, ninja_file):
        jobs = os.environ.get("MAX_JOBS", str(max(1, os.cpu_count() or 1)))
        subprocess.check_call(["ninja", "-f", str(ninja_file), f"-j{jobs}"])

    @staticmethod
    def _hgcc_link_args(arguments):
        """Translate GCC driver linker forwarding into hgcc's native syntax."""
        translated = []
        for argument in arguments:
            if argument.startswith("-Wl,"):
                translated.extend(
                    f"-Xlinker={option}"
                    for option in argument[len("-Wl,"):].split(",")
                    if option
                )
            else:
                translated.append(argument)
        return translated

    def _link(self, extension, objects, output):
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [self._hgcc, "-shared", "-o", str(output)]
        command += list(objects)
        command += self._hgcc_link_args(extension.extra_link_args or [])
        command += self._link_flags
        subprocess.check_call(command)
        return command

    def build_extension(self, extension):
        output_dir = Path(self.build_temp).resolve() / extension.name
        manifest = None
        if hasattr(extension, "fla_builder"):
            _, manifest = extension.fla_builder.compile_objects(self, extension)

        objects = self.compile_sources(
            extension.sources,
            output_dir,
            extension,
            depends=extension.depends,
        )
        objects += list(extension.extra_objects or [])
        link_command = self._link(extension, objects, self.get_ext_fullpath(extension.name))

        if manifest is not None:
            manifest["link_command"] = link_command
            extension.fla_builder.finish_manifest(self, extension, output_dir, manifest)
