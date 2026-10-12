#!/usr/bin/env python3
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

"""Unified build script for pla kernels.

Usage:
    python scripts/build.py --mode default   # fastest: only default combo
    python scripts/build.py --mode tuned     # only combos from lut/tuned_result_*.csv
    python scripts/build.py --mode full      # all 102 combos
    python scripts/build.py --mode full --tune  # full build + auto-tune + tuned rebuild

    # build a single API namespace (default: all)
    python scripts/build.py --namespace pla.decode
    python scripts/build.py --namespace pla.prefill

--mode / --tune only affect pla.decode (they select which k_last kernel template
combinations get instantiated); pla.prefill has no template sweep.
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DECODE_SCRIPT_DIR = os.path.join(SCRIPT_DIR, "decode")

BUILD_ENV = os.environ.copy()

# Keep in sync with ALL_NAMESPACES in setup.py.
ALL_NAMESPACES = ("pla.decode", "pla.decode.gdn.v_last",
                  "pla.decode.kda_mtp_sglang",
                  "pla.prefill.flashqla", "pla.prefill.flashkdapro")


def selects_k_last_decode(namespace):
    """True if the namespace filter includes pla.decode (the k-last extension).

    Only that extension consumes the generated kernel instantiations, so a
    v-last- or prefill-only build must not spend time regenerating them. Uses
    the same exact-match-then-prefix rule as setup.py.
    """
    if not namespace:
        return True
    for token in (t.strip() for t in namespace.split(",")):
        if not token:
            continue
        if token in ALL_NAMESPACES:
            if token == "pla.decode":
                return True
        elif any(ns == "pla.decode" for ns in ALL_NAMESPACES if ns.startswith(token + ".")):
            return True
    return False


def run(cmd, cwd=PROJECT_DIR):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=BUILD_ENV)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Build pla kernels")
    parser.add_argument("--mode", choices=["default", "tuned", "full"], default="default",
                        help="Kernel instantiation mode for pla.decode (default: default)")
    parser.add_argument("--tune", action="store_true",
                        help="Run auto-tune after full build, then rebuild in tuned mode")
    parser.add_argument("--tuned-csv", type=str, default=os.path.join(PROJECT_DIR, "lut/tuned_result_*.csv"),
                        help="Glob pattern for tuned CSV files (default: lut/tuned_result_*.csv)")
    parser.add_argument("--namespace", type=str, default=None,
                        help="Only build this API namespace, e.g. pla.decode, "
                             "pla.decode.gdn.v_last or pla.prefill "
                             "(comma-separated; default: all)")
    args = parser.parse_args()

    if args.namespace:
        BUILD_ENV["PLA_BUILD_NAMESPACE"] = args.namespace
        print(f"Namespace filter: {args.namespace}")

    build_decode = selects_k_last_decode(args.namespace)

    gen_script = os.path.join(DECODE_SCRIPT_DIR, "generate_kernel_instantiations.py")

    if not build_decode:
        # No k-last kernels requested: nothing to instantiate, just compile.
        print("=== Build (no k-last decode kernels; skipping instantiation) ===")
        run("pip install .")
        print("\nBuild complete.")
        return

    if args.tune:
        print("=== Phase 1: Full build (all combos) ===")
        run(f"python3 {gen_script}")
        run("pip install .")

        print("\n=== Phase 2: Auto-tune ===")
        tune_script = os.path.join(DECODE_SCRIPT_DIR, "auto_tune.py")
        run(f"python3 {tune_script} --output {args.tuned_csv}")

        print("\n=== Phase 3: Tuned rebuild ===")
        run(f"python3 {gen_script} --tuned-csv {args.tuned_csv}")
        run("pip install .")

    elif args.mode == "default":
        print("=== Default build (fastest) ===")
        run(f"python3 {gen_script} --default-only")
        run("pip install .")

    elif args.mode == "tuned":
        import glob
        if not glob.glob(args.tuned_csv):
            print(f"Error: no files match {args.tuned_csv}. "
                  "Run with --tune first or --mode full --tune.")
            sys.exit(1)
        print("=== Tuned build ===")
        run(f"python3 {gen_script} --tuned-csv '{args.tuned_csv}'")
        run("pip install .")

    elif args.mode == "full":
        print("=== Full build (all combos) ===")
        run(f"python3 {gen_script}")
        run("pip install .")

    print("\nBuild complete.")


if __name__ == "__main__":
    main()
