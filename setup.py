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

"""PLA build configuration — one wheel, one extension per API namespace.

Namespaces and their compiled artifacts:

    pla.decode                  -> pla/decode/_C.*.so                    (src/decode/k-last/**)
    pla.decode.gdn.v_last       -> pla/decode/gdn/v_last/_C.*.so         (src/decode/v-last/**)
    pla.decode.kda_mtp_sglang   -> pla/decode/kda_mtp_sglang/_C.*.so     (src/decode/kda_mtp_sglang/**)
    pla.prefill.flashqla        -> pla/prefill/flashqla/_C.*.so          (src/prefill/flashqla/**)
    pla.prefill.flashkdapro     -> pla/prefill/flashkdapro/_C.*.so       (src/prefill/flashkdapro/**)

Selecting what to build (default: everything):

    pip install .                                   # all namespaces
    PLA_BUILD_NAMESPACE=pla.decode pip install .    # k-last decode only
    PLA_BUILD_NAMESPACE=pla.prefill pip install .   # prefill only
    python setup.py build_ext --inplace --namespace=pla.decode.gdn.v_last

``PLA_BUILD_NAMESPACE`` accepts a comma-separated list. An exact namespace name
selects just that one, so ``pla.decode`` does not drag in the nested
``pla.decode.gdn.v_last``; a token that is not itself a namespace prefix-matches
(``pla.prefill`` selects both ``pla.prefill.flashqla`` and
``pla.prefill.flashkdapro``). It is the env-var form because ``pip install .``
gives no clean way to forward custom options.

Build speed:
  * ninja + all cores (override with MAX_JOBS)
  * per-file hgcc ``--threads`` (override with HGCC_THREADS)
  * the v-last kernels are instantiated one translation unit per H value; as a
    single TU each took ~25 min and dominated the whole build
  * restricting the namespace skips the other namespaces' translation units --
    src/decode/k-last/generated alone can be 200+ files in a full build (see
    scripts/decode/generate_kernel_instantiations.py)

The prefill extensions (flashqla and flashkdapro) need ACTLIZE, the PPU
adaptation of CUTLASS 3.6.0 (git submodule at 3rdparty/actlize, branch v1.0.0).
get_actlize_includes() below will initialise it automatically if it is missing,
or you can point PPU_ACTLIZE_DIR at an existing checkout:

    git submodule update --init --recursive
"""
import glob
import importlib.util
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup

from ppu_build import PPUBuildExtension

here = os.path.abspath(os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Namespace selection
# ---------------------------------------------------------------------------
ALL_NAMESPACES = ("pla.decode", "pla.decode.gdn.v_last",
                  "pla.decode.kda_mtp_sglang",
                  "pla.prefill.flashqla", "pla.prefill.flashkdapro")


def _requested_namespaces():
    """Resolve the namespaces to build from --namespace / PLA_BUILD_NAMESPACE."""
    raw = os.environ.get("PLA_BUILD_NAMESPACE", "")
    # Consume --namespace=... / --namespace ... so setuptools does not choke.
    argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--namespace="):
            raw = arg.split("=", 1)[1]
        elif arg == "--namespace" and i + 1 < len(sys.argv):
            raw = sys.argv[i + 1]
            i += 1
        else:
            argv.append(arg)
        i += 1
    sys.argv[:] = argv

    if not raw.strip():
        return list(ALL_NAMESPACES)

    selected = []
    for token in (t.strip() for t in raw.split(",")):
        if not token:
            continue
        # Exact match wins, so `pla.decode` selects only the k-last extension and
        # not the nested `pla.decode.gdn.v_last`. A token that is not itself a
        # namespace (e.g. `pla.prefill`, `pla.decode.gdn`) prefix-matches instead.
        if token in ALL_NAMESPACES:
            matched = [token]
        else:
            matched = [ns for ns in ALL_NAMESPACES if ns.startswith(token + ".")]
        if not matched:
            raise SystemExit(
                f"Unknown namespace '{token}'. Known namespaces: {', '.join(ALL_NAMESPACES)}"
            )
        for ns in matched:
            if ns not in selected:
                selected.append(ns)
    return selected


NAMESPACES = _requested_namespaces()

# Ninja parallelism is shared by all PPU extension builds.
os.environ.setdefault("MAX_JOBS", str(multiprocessing.cpu_count()))
HGCC_THREADS = os.environ.get("HGCC_THREADS", "16")

ext_modules = []

# ---------------------------------------------------------------------------
# pla.decode
# ---------------------------------------------------------------------------
if "pla.decode" in NAMESPACES:
    generated_sources = sorted(glob.glob(os.path.join(here, 'src', 'decode', 'k-last', 'generated', '*.cu')))
    if not generated_sources:
        raise RuntimeError(
            "src/decode/k-last/generated is empty. Run "
            "`python scripts/decode/generate_kernel_instantiations.py --default-only` first.")

    decode_sources = [
        'src/decode/k-last/fused_sigmoid_gating.cpp',
        'src/decode/k-last/fused_recurrent_kda.cpp',
        # host-only: no device code, so g++ compiles it and it may keep using torch
        'src/decode/k-last/fused_sigmoid_gating_cuda_k_last.cpp',
        'src/decode/k-last/fused_sigmoid_gating_cuda_k_last_packed.cpp',
        'src/decode/k-last/fused_kda_decode_kernel_ref.cu',
        # host halves of the two TUs above: they hold the tensor unpacking, which
        # cannot live in a device TU compiled against the native headers
        'src/decode/k-last/fused_kda_decode_ref_host.cpp',
        'src/decode/k-last/fused_kda_decode_mega.cu',
        'src/decode/k-last/fused_kda_decode_mega_host.cpp',
    ] + generated_sources

    ext_modules.append(
        Extension(
            name='pla.decode._C',
            sources=decode_sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'hgcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-lineinfo',
                    # '-mllvm', '-ppu-simt-branch=false',
                    '-mllvm', '-ppu-patch-fence-ppu=false',
                    '-mllvm', '-wno-loop-miss-transform',
                    '-mllvm', '-ppu-cg-to-kp1=true',
                    '-mllvm', '-ppu-fix-uninit=true',
                    '-mllvm', '-ppu-force-warpage=true',
                    '-mllvm', '-sort-copy-before-coalesce=true',
                    '-mllvm', '-ppu-eliminate-early-v2s=true',
                    '-mllvm', '-ppu-pref-fma-reuse=true',
                    '--threads', HGCC_THREADS,
                    # Real device targets, not the vm_* virtual IR ones: the
                    # PPU runtime does not JIT, so a vm_* (or missing) -arch
                    # makes every launch fail with 801 operation-not-supported
                    # while hggcDeviceSynchronize() still returns success.
                    # Emit native images for both supported PPU generations.
                    '-arch=ppu_10,ppu_15'
                ]
            },
            # MUST be absolute: ninja does not run from the project root, so
            # relative -I paths would not resolve. src/ is needed because the
            # kernels live in src/decode/ but #include "common.cuh" from src/.
            include_dirs=[os.path.join(here, "src"),
                          os.path.join(here, "src", "decode", "k-last")],
            extra_link_args=[],
        )
    )

# ---------------------------------------------------------------------------
# pla.decode.gdn.v_last
# ---------------------------------------------------------------------------
if "pla.decode.gdn.v_last" in NAMESPACES:
    v_last_dir = os.path.join(here, "src", "decode", "v-last")
    # One translation unit per H value: the bigbatch/smallbatch template cascade
    # (dtypes x H x HV x hv_per_cta) took ~25 min as a single TU, and it is the
    # critical path of the whole build. Splitting it lets ninja overlap them.
    v_last_sources = [os.path.join(v_last_dir, "fused_sigmoid_gating_v_last.cpp")] + \
        sorted(glob.glob(os.path.join(v_last_dir, "fused_sigmoid_gating_*_h*.cu")))

    ext_modules.append(
        Extension(
            name='pla.decode.gdn.v_last._C',
            sources=v_last_sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'hgcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-lineinfo',
                    '-mllvm', '-ppu-patch-fence-ppu=false',
                    '-mllvm', '-wno-loop-miss-transform',
                    '-mllvm', '-ppu-cg-to-kp1=true',
                    '-mllvm', '-ppu-fix-uninit=true',
                    '-mllvm', '-ppu-force-warpage=true',
                    '-mllvm', '-sort-copy-before-coalesce=true',
                    '-mllvm', '-ppu-eliminate-early-v2s=true',
                    '-mllvm', '-ppu-pref-fma-reuse=true',
                    '--threads', HGCC_THREADS,
                    # Real device targets, not the vm_* virtual IR ones: the
                    # PPU runtime does not JIT, so a vm_* (or missing) -arch
                    # makes every launch fail with 801 operation-not-supported
                    # while hggcDeviceSynchronize() still returns success.
                    # Emit native images for both supported PPU generations.
                    '-arch=ppu_10,ppu_15'
                ]
            },
            # Absolute, for the same reason as pla.decode above.
            include_dirs=[os.path.join(here, "src"), v_last_dir],
            extra_link_args=[],
        )
    )

# ---------------------------------------------------------------------------
# pla.decode.kda_mtp_sglang
# ---------------------------------------------------------------------------
if "pla.decode.kda_mtp_sglang" in NAMESPACES:
    kda_mtp_dir = os.path.join(here, "src", "decode", "kda_mtp_sglang")
    kda_mtp_sources = [
        os.path.join(kda_mtp_dir, "kda_mtp_sglang.cpp"),
        os.path.join(kda_mtp_dir, "kda_mtp_verify.cu"),
        os.path.join(kda_mtp_dir, "kda_mtp_verify_host.cpp"),
        os.path.join(kda_mtp_dir, "kda_mtp_fold.cu"),
        os.path.join(kda_mtp_dir, "kda_mtp_fold_host.cpp"),
    ]

    ext_modules.append(
        Extension(
            name='pla.decode.kda_mtp_sglang._C',
            sources=kda_mtp_sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'hgcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-lineinfo',
                    '-mllvm', '-ppu-patch-fence-ppu=false',
                    '-mllvm', '-wno-loop-miss-transform',
                    '-mllvm', '-ppu-cg-to-kp1=true',
                    '-mllvm', '-ppu-fix-uninit=true',
                    '-mllvm', '-ppu-force-warpage=true',
                    '-mllvm', '-sort-copy-before-coalesce=true',
                    '-mllvm', '-ppu-eliminate-early-v2s=true',
                    '-mllvm', '-ppu-pref-fma-reuse=true',
                    '--threads', HGCC_THREADS,
                    # Real device targets, not the vm_* virtual IR ones: the
                    # PPU runtime does not JIT, so a vm_* (or missing) -arch
                    # makes every launch fail with 801 operation-not-supported
                    # while hggcDeviceSynchronize() still returns success.
                    # Emit native images for both supported PPU generations.
                    '-arch=ppu_10,ppu_15'
                ]
            },
            # Absolute path: ninja does not run from the project root. Only
            # src/ is needed; the device sources use native PPU SDK headers.
            include_dirs=[os.path.join(here, "src")],
            extra_link_args=[],
        )
    )


# ---------------------------------------------------------------------------
# pla.prefill.flashqla
# ---------------------------------------------------------------------------


ACTLIZE_SUBMODULE = Path(here) / "3rdparty" / "actlize"
# actlize keeps the upstream include roots (include/cute, include/cutlass) and
# the cutlass:: / cute:: namespaces, but the PPU headers that cutlass3 kept under
# include/ppu/ are flattened into cute/, and the acompute10000 / acompute10500
# families are spelled ppu0010 / ppu0015. Probe exactly the headers the prefill
# kernels need, so a cutlass3 checkout (or an actlize tree too old to have them)
# is rejected up front instead of failing later inside hgcc.
ACTLIZE_REQUIRED_HEADERS = (
    "include/cute/tensor.hpp",
    "include/cute/ppu_tensor_mix.hpp",
    "include/cute/arch/mma_ppu0015.hpp",
    "include/cute/atom/mma_traits_ppu0015.hpp",
    "include/cute/arch/copy_ppu0015_aiu.hpp",
    "include/cute/atom/copy_traits_ppu0015_aiu.hpp",
    "include/cute/arch/copy_ppu0010_aiu.hpp",
    "include/cute/atom/copy_traits_ppu0010_aiu.hpp",
)

ACTLIZE_HELP = f"""
{'=' * 78}
ACTLIZE headers not found -- cannot build the pla.prefill extensions.
{'=' * 78}

The FlashQLA and FlashKDA prefill kernels need ACTLIZE, the PPU adaptation of
CUTLASS 3.6.0, which this repo carries as a git submodule at 3rdparty/actlize.

Fix it with ONE of the following, then re-run the build:

  1) Initialise the submodule (preferred):

         git submodule update --init --recursive

     The build already tried this for you; git's own error is printed above and
     is the authoritative reason it failed. The usual causes are:

       - No network access to the remote. The submodule clones from
             https://github.com/t-head/actlize.git
         branch v1.0.0.

       - This tree is not a usable git checkout: built from a source tarball
         (no .git), or git refused it as "dubious ownership" because the repo
         belongs to another user (common inside containers). For the latter:
             git config --global --add safe.directory <this directory>

  2) Point at an existing checkout instead:

         export PPU_ACTLIZE_DIR=/path/to/actlize

     The directory must carry the CuTe headers plus the PPU ppu0010 / ppu0015
     MMA and AIU headers the prefill kernels use.

  3) Skip prefill entirely and build only the decode namespaces:

         python scripts/build.py --namespace pla.decode
         python scripts/build.py --namespace pla.decode.gdn.v_last

NOTE: the pre-actlize CUTLASS3 PPU fork (branch v3.6.0_ppu_dev) is NOT
compatible any more -- it keeps the PPU headers under include/ppu/ and spells
the traits Acompute10000 / Acompute10500.
{'=' * 78}
"""


def _actlize_includes_at(root):
    """Return include dirs for a PLA prefill-compatible ACTLIZE checkout."""
    root = Path(root)
    if any(not (root / header).exists() for header in ACTLIZE_REQUIRED_HEADERS):
        return None
    inc = [str(root / "include")]
    util = root / "tools" / "util" / "include"
    if util.exists():
        inc.append(str(util))
    return inc


def _dir_has_content(path):
    return Path(path).is_dir() and any(Path(path).iterdir())


def get_actlize_includes():
    """Locate the ACTLIZE headers, initialising the submodule if needed.

    Resolution order:
      1. 3rdparty/actlize, if the submodule is already populated
      2. PPU_ACTLIZE_DIR
      3. `git submodule update --init --recursive`, then 3rdparty/actlize again

    If all three fail the build is aborted with printed instructions rather than
    left to die later on a confusing `cute/tensor.hpp: No such file` from hgcc.
    """
    # 1. Already-populated submodule wins: it is the pinned, known-good revision.
    if _dir_has_content(ACTLIZE_SUBMODULE):
        inc = _actlize_includes_at(ACTLIZE_SUBMODULE)
        if inc:
            print(f"[pla.prefill] ACTLIZE: {ACTLIZE_SUBMODULE} (submodule)")
            return inc
        sys.stderr.write(
            f"[pla.prefill] WARNING: {ACTLIZE_SUBMODULE} exists but is missing "
            "required PPU prefill headers; treating it as unusable.\n")

    # 2. Explicit override.
    env = os.environ.get("PPU_ACTLIZE_DIR", "").strip()
    if env:
        inc = _actlize_includes_at(env)
        if inc:
            print(f"[pla.prefill] ACTLIZE: {env} (PPU_ACTLIZE_DIR)")
            return inc
        sys.stderr.write(
            f"[pla.prefill] WARNING: PPU_ACTLIZE_DIR={env} is missing required "
            "PPU prefill headers; ignoring it.\n")

    # 3. Try to populate the submodule ourselves. Output is not captured so the
    #    user sees git's own progress and error text.
    #    -c safe.directory bypasses git's dubious-ownership refusal for this one
    #    invocation (no persistent config change). We are already executing this
    #    repo's setup.py, so the trust decision has effectively been made.
    print("[pla.prefill] ACTLIZE not found; running "
          "`git submodule update --init --recursive` ...", flush=True)
    try:
        rc = subprocess.call(
            ["git", "-c", f"safe.directory={here}",
             "submodule", "update", "--init", "--recursive"], cwd=here)
    except OSError as exc:
        rc = -1
        sys.stderr.write(f"[pla.prefill] could not run git: {exc}\n")
    if rc == 0:
        inc = _actlize_includes_at(ACTLIZE_SUBMODULE)
        if inc:
            print(f"[pla.prefill] ACTLIZE: {ACTLIZE_SUBMODULE} "
                  "(submodule, just initialised)")
            return inc
        sys.stderr.write(
            "[pla.prefill] git submodule update succeeded but "
            f"{ACTLIZE_SUBMODULE} is still missing required PPU prefill headers.\n")
    else:
        sys.stderr.write(
            f"[pla.prefill] git submodule update failed (exit {rc}).\n")

    # Print to both streams: pip captures stdout and only replays it on failure,
    # while stderr shows up immediately. Belt and braces so the steps cannot be
    # missed, and repeat them in the SystemExit message setuptools prints.
    sys.stdout.write(ACTLIZE_HELP)
    sys.stdout.flush()
    sys.stderr.write(ACTLIZE_HELP)
    sys.stderr.flush()
    raise SystemExit(ACTLIZE_HELP)


# Both prefill extensions target the same PPU1.5 CuTe toolchain (same PPU LLVM
# passes, same CuTe/CUTLASS defines, same target arch), so the flags live in
# one place and each extension appends only what is specific to it.
PREFILL_CXX_FLAGS = ["-O3", "-std=c++20", "-DNDEBUG",
                     "-Wno-deprecated-declarations"]

PREFILL_HGCC_FLAGS = [
    "-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES",
    # source-line mapping for ACU per-line stall analysis (no perf impact)
    "-lineinfo",
    # hgcc does not take -W* itself; the deprecation warnings come from the host
    # half of the translation unit, so hand the flag to the host compiler.
    "-Xcompiler", "-Wno-deprecated-declarations",
    "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
    # PPU LLVM passes (FlashMLA canonical list)
    "-mllvm", "-ppu-max-vreg-count=256",
    "-mllvm", "-ppu-sink-matrix-addr=true",
    "-mllvm", "-ppu-max-alloca-byte-size=1024",
    "-mllvm", "-ppu-sink-async-addr=true",
    "-mllvm", "-ppu-sink-load-addr=true",
    "-mllvm", "-ppu-sink-store-addr=true",
    "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
    "-mllvm", "-ppu-force-warpage=true",
    "-mllvm", "-ppu-force-vregrr=true",
    # PPU defines
    "-DUSE_PPU", "-DUSE_AIU=1",
    # CUTLASS / CuTe
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DCUTLASS_VERSIONS_GENERATED", "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1", "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    # Real device targets (see the decode extensions for why vm_* is wrong)
    "-arch=ppu_10,ppu_15",
    "--threads", HGCC_THREADS,
]

# Resolve the shared prefill include dirs once, so a build that selects both
# prefill namespaces does not run (and log) the ACTLIZE lookup twice.
if any(ns.startswith("pla.prefill.") for ns in NAMESPACES):
    actlize_includes = get_actlize_includes()


if "pla.prefill.flashqla" in NAMESPACES:
    prefill_dir = os.path.join(here, "src", "prefill", "flashqla")

    prefill_sources = [os.path.join(prefill_dir, "host", "torch_extension_cute.cc")] + \
        sorted(glob.glob(os.path.join(prefill_dir, "kernels", "*.cu")))

    ext_modules.append(
        Extension(
            name="pla.prefill.flashqla._C",
            sources=prefill_sources,
            include_dirs=[
                os.path.join(prefill_dir, "kernels"),  # flashqla_cute_traits.cuh
                os.path.join(here, "src"),
            ] + actlize_includes,
            extra_link_args=[],
            extra_compile_args={
                "cxx": PREFILL_CXX_FLAGS,
                "hgcc": PREFILL_HGCC_FLAGS + [
                    # Production GEMM schedule: 4->3->1->2->6->5 (~13% faster on
                    # ZW-M890P). Kept overridable for A/B, defaults to production.
                    "-DFF_GEMM_REORDER=" + os.getenv("FF_GEMM_REORDER", "1"),
                ],
            },
        )
    )


# ---------------------------------------------------------------------------
# pla.prefill.flashkdapro
# ---------------------------------------------------------------------------
if "pla.prefill.flashkdapro" in NAMESPACES:
    flashkda_dir = os.path.join(here, "src", "prefill", "flashkdapro")

    # Keep this manifest explicit.  FlashKDApro contains architecture-specific
    # source trees and private profiling/probe files; a recursive glob can
    # silently omit a moved production TU or accidentally compile an experiment.
    flashkda_source_manifest = (
        os.path.join("host", "torch_extension_cute.cc"),
        os.path.join("kernels", "kernel1_prepare.cu"),
        os.path.join("kernels", "kernel2_recurrence.cu"),
        os.path.join("kernels", "kernel2_recurrence_nows.cu"),
        os.path.join("kernels", "kernel3_cp.cu"),
        os.path.join("tc01", "kernel_bt64_prefill_tc01.cu"),
    )
    flashkda_sources = [
        os.path.join(flashkda_dir, source)
        for source in flashkda_source_manifest
    ]
    missing_flashkda_sources = [
        source for source in flashkda_sources if not os.path.isfile(source)
    ]
    if missing_flashkda_sources:
        raise RuntimeError(
            "FlashKDApro source manifest contains missing files: "
            + ", ".join(missing_flashkda_sources))

    ext_modules.append(
        Extension(
            name="pla.prefill.flashkdapro._C",
            sources=flashkda_sources,
            include_dirs=[
                # flashkda_cute_traits.cuh / _f16acc.cuh / _ppu_util.cuh
                os.path.join(flashkda_dir, "kernels"),
                os.path.join(here, "src"),
            ] + actlize_includes,
            extra_link_args=[],
            extra_compile_args={
                "cxx": PREFILL_CXX_FLAGS,
                "hgcc": PREFILL_HGCC_FLAGS
            },
        )
    )

    # The monolithic FLA is compiled in head/stage groups; keep the source flag order and the tc01 arch config.
    fla_dir = os.path.join(flashkda_dir, "tc01", "fla")
    spec = importlib.util.spec_from_file_location("fla_monolithic_build", os.path.join(fla_dir, "monolithic_build.py"))
    fla_monolithic_build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fla_monolithic_build)
    selected_heads = fla_monolithic_build.build_heads(os.environ.get("PLA_FLA_BUILD_HEADS"))
    probe_only = os.environ.get("PLA_FLA_BUILD_ONLY", "0") == "1"
    if "bdist_wheel" in sys.argv and (probe_only or selected_heads != fla_monolithic_build.HEADS):
        raise RuntimeError("a partial-head / FLA-only link probe cannot publish a wheel")
    extension = Extension(
        name="pla.prefill.flashkdapro._tc01_fla._C",
        sources=[os.path.join(fla_dir, name) for name in ("runtime_binding.cc", "runtime.cc", "dispatch.cc")],
        include_dirs=[fla_dir, os.path.join(here, "src")] + actlize_includes,
        extra_compile_args={"cxx": list(fla_monolithic_build.HOST_FLAGS), "hgcc": []},
        extra_link_args=["-Wl,-Bsymbolic", "-Wl,--version-script=" + os.path.join(fla_dir, "exports.map")],
        depends=[os.path.join(fla_dir, "exports.map")],
    )
    extension.fla_source, extension.fla_heads = fla_dir, selected_heads
    extension.fla_builder = fla_monolithic_build
    ext_modules.append(extension)
    if probe_only:
        ext_modules = [extension]


print(f"[pla] building namespaces: {', '.join(NAMESPACES)} "
      f"(MAX_JOBS={os.environ['MAX_JOBS']}, HGCC_THREADS={HGCC_THREADS})")

setup(
    name='pla',
    version='1.0.0',
    author='Gino Wu',
    description='PPU Linear Attention kernels (decode + prefill)',
    long_description='',
    license='BSD-3-Clause',
    license_files=('LICENSE', 'NOTICE', 'LICENSES/*.txt'),
    packages=find_packages(include=["pla", "pla.*"]),
    ext_modules=ext_modules,
    cmdclass={'build_ext': PPUBuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
)
