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

"""Generated-artifact consistency and shared-memory layout checks that need no PyTorch/PPU.

Run: python tests/test_generated_k_last.py
Checks against the checked-in template combinations, compatible with full/default/tuned builds, without changing the tuning table.
"""

import ast
import csv
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "src/decode/k-last/generated"
spec = importlib.util.spec_from_file_location(
    "kernel_codegen", ROOT / "scripts/decode/generate_kernel_instantiations.py"
)
codegen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codegen)


class GeneratedKLastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pattern = re.compile(r"k_last_bv(\d+)_wn(\d+)_sn(\d+)_hv(\d+)\.cu")
        cls.combos = sorted(
            tuple(map(int, match.groups()))
            for path in GEN_DIR.glob("k_last_bv*.cu")
            if (match := pattern.fullmatch(path.name))
        )
        if not cls.combos:
            raise AssertionError("no k-last generated artifacts found")

    def assert_generated(self, name, expected):
        path = GEN_DIR / name
        self.assertTrue(path.is_file(), f"missing generated artifact: {name}")
        self.assertTrue(
            path.read_text() == expected,
            f"generated artifact is stale: {name}; regenerate it with the current generator and commit it together",
        )

    def test_k_last_wrappers(self):
        for bv, wn, sn, hv in self.combos:
            name = f"k_last_bv{bv}_wn{wn}_sn{sn}_hv{hv}.cu"
            with self.subTest(file=name):
                self.assert_generated(name, codegen.generate_k_last_cu(bv, wn, sn, hv))

    def test_packed_wrappers(self):
        expected_names = set()
        for bv, wn, sn, hv in self.combos:
            name = f"k_last_packed_bv{bv}_wn{wn}_sn{sn}_hv{hv}.cu"
            expected_names.add(name)
            with self.subTest(file=name):
                self.assert_generated(name, codegen.generate_k_last_packed_cu(bv, wn, sn, hv))
        self.assertEqual(expected_names, {p.name for p in GEN_DIR.glob("k_last_packed_bv*.cu")})

    def test_dispatch_headers(self):
        for suffix, generator in (
            ("k_last", codegen.generate_dispatch_header_k_last),
            ("k_last_packed", codegen.generate_dispatch_header_k_last_packed),
        ):
            name = f"fused_sigmoid_gating_dispatch_{suffix}.h"
            with self.subTest(file=name):
                self.assert_generated(name, generator(self.combos))


class TunedTableTest(unittest.TestCase):
    @staticmethod
    def row(kernel="k_last_mtp", mode="chain", bv=32):
        return dict(device_name="ZW-M890P", kernel=kernel, B=16,
                    H=4, HV=12, K=128, V=128, dtype="bf16", state_dtype="fp32",
                    BV=bv, WARP_NUM=4, STAGE_NUM=4, HV_PER_CTA=1,
                    latency="10.0", unit="us", mode=mode)

    def test_upsert_preserves_mtp_modes_and_packed_state_dtype(self):
        spec = importlib.util.spec_from_file_location(
            "decode_tuner", ROOT / "scripts/decode/auto_tune.py")
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
        chain = self.row()
        tree = self.row(mode="tree", bv=64)
        packed = self.row(kernel="k_last_packed", mode="")
        bf16 = dict(packed, state_dtype="bf16")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tuned.csv"
            fields = list(chain)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows((chain, tree, packed, bf16))
            tuner.merge_tuned_rows(
                path, [f for f in fields if f != "mode"],
                [dict(tree, BV=128)], "ZW-M890P")
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["mode"], "chain")
            self.assertEqual(rows[0]["BV"], "32")
            self.assertEqual(rows[1]["mode"], "tree")
            self.assertEqual(rows[1]["BV"], "128")
            self.assertEqual([r["state_dtype"] for r in rows[2:]], ["fp32", "bf16"])

    def test_legacy_partial_upsert_preserves_all_modes_and_buckets(self):
        spec = importlib.util.spec_from_file_location(
            "decode_tuner", ROOT / "scripts/decode/auto_tune.py")
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
        for mode in ("chain", "tree"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "tuned.csv"
                legacy = [dict(self.row(mode="", bv=bv), B=batch)
                          for batch, bv in ((1, 32), (64, 64), (512, 128))]
                explicit = dict(self.row(mode=mode, bv=16), B=64)
                untouched = [self.row(kernel="k_last_packed", mode=""),
                             dict(self.row(kernel="k_last_packed", mode=""), state_dtype="bf16"),
                             dict(legacy[0], device_name="ZW-M890L")]
                fields = list(explicit)
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows([explicit, *legacy, *untouched])
                update = dict(self.row(mode=mode, bv=128), B=1)
                tuner.merge_tuned_rows(path, fields, [update], "ZW-M890P")
                first = path.read_bytes()
                tuner.merge_tuned_rows(path, fields, [update], "ZW-M890P")
                self.assertEqual(first, path.read_bytes())
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 9)
                by_key = {(r["mode"], int(r["B"])): int(r["BV"]) for r in rows
                          if r["kernel"] == "k_last_mtp" and r["device_name"] == "ZW-M890P"}
                other = "chain" if mode == "tree" else "tree"
                self.assertEqual(by_key, {(mode, 1): 128, (mode, 64): 16, (mode, 512): 128,
                                          (other, 1): 32, (other, 64): 64, (other, 512): 128})
                self.assertEqual(rows[-3:], [{k: str(v) for k, v in r.items()} for r in untouched])
                _, records = codegen.load_tuned_csv(str(path))
                for selected_mode in ("chain", "tree"):
                    entries = codegen._bucket_table_entries([r for r in records
                        if r["kernel"] == "k_last_mtp" and r["device_name"] == "ZW-M890P"
                        and r["mode"] == selected_mode])
                    self.assertEqual(len(entries), 33)
                    self.assertEqual(entries[4], "{%d, 4, 4, 1}" % by_key[(selected_mode, 64)])
                    self.assertEqual(entries[-1], "{128, 4, 4, 1}")

    def test_upsert_migrates_csv_without_mode_column(self):
        spec = importlib.util.spec_from_file_location(
            "decode_tuner", ROOT / "scripts/decode/auto_tune.py")
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
        fields = [name for name in self.row() if name != "mode"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows([dict(self.row(bv=32), B=1), dict(self.row(bv=64), B=64)])
            tuner.merge_tuned_rows(path, fields,
                [dict(self.row(mode="tree", bv=128), B=64)], "ZW-M890P")
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({(r["mode"], int(r["B"])): int(r["BV"]) for r in rows},
                             {("chain", 1): 32, ("chain", 64): 64,
                              ("tree", 1): 32, ("tree", 64): 128})

    def test_flashnext_890p_lut_coverage(self):
        path = ROOT / "lut/tuned_result_ZW-M890P.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        variants = (("k_last", "fp32", ""),
                    ("k_last_packed", "fp32", ""),
                    ("k_last_packed", "bf16", ""),
                    ("k_last_mtp", "fp32", "chain"),
                    ("k_last_mtp", "fp32", "tree"))
        valid = set(codegen.get_valid_combinations())
        for heads in (6, 12, 24, 48):
            for kernel, dtype, mode in variants:
                matching = [r for r in rows if int(r["B"]) <= 512 and
                    (int(r["H"]), int(r["HV"]), int(r["K"]), int(r["V"])) ==
                    (heads // 3, heads, 128, 128) and
                    (r["kernel"], r["state_dtype"], r.get("mode", "")) ==
                    (kernel, dtype, mode)]
                self.assertEqual(sorted(int(r["B"]) for r in matching),
                                 [1, *range(16, 513, 16)])
                for r in matching:
                    self.assertIn(tuple(int(r[n]) for n in
                        ("BV", "WARP_NUM", "STAGE_NUM", "HV_PER_CTA")), valid)

    def assert_tuned_header(self, text, generated_dir=GEN_DIR):
        default = codegen.generate_default_tuned_dispatch()
        if text == default:
            return
        prefix = "// PLA_TUNED_CSV: "
        marker = ""
        if text.startswith(prefix):
            marker = text.splitlines(keepends=True)[0]
            paths = json.loads(marker[len(prefix):])
            self.assertIsInstance(paths, list)
            self.assertTrue(paths)
            records = []
            for path in paths:
                self.assertIsInstance(path, str)
                _, source = codegen.load_tuned_csv(str(generated_dir / path))
                records.extend(source)
        else:
            _, records = codegen.load_tuned_csv(str(ROOT / "lut/tuned_result_*.csv"))
        self.assertEqual(text, marker + codegen.generate_tuned_dispatch(records))

    def test_current_build_tuned_header(self):
        self.assert_tuned_header((GEN_DIR / "tuned_dispatch.h").read_text())

    def test_all_generation_modes_and_stale_header_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom = root / "tuned_result_ZW-M890P.csv"
            row = self.row()
            with custom.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            for name, arguments in (
                ("default", ["--default-only"]), ("full", []),
                ("tuned", ["--tuned-csv", str(ROOT / "lut/tuned_result_*.csv")]),
                ("custom", ["--tuned-csv", str(custom)]),
            ):
                with self.subTest(mode=name):
                    out = root / name
                    with mock.patch.object(codegen, "GENERATED_DIR", str(out)), \
                            mock.patch.object(sys, "argv", ["generate", *arguments]):
                        codegen.main()
                    text = (out / "tuned_dispatch.h").read_text()
                    self.assert_tuned_header(text, out)
                    with self.assertRaises(AssertionError):
                        self.assert_tuned_header(text.replace("DEFAULT_PARAMS", "STALE_PARAMS"), out)
            custom.write_text(custom.read_text().replace(",32,4,4,1,", ",64,4,4,1,"))
            with self.assertRaises(AssertionError):
                self.assert_tuned_header((root / "custom/tuned_dispatch.h").read_text(), root / "custom")

    def test_decode_build_tracks_tuned_header(self):
        tree = ast.parse((ROOT / "setup.py").read_text())
        extensions = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "Extension"]
        decode = next(n for n in extensions if any(
            k.arg == "name" and isinstance(k.value, ast.Constant)
            and k.value.value == "pla.decode._C" for k in n.keywords))
        depends = next(k.value for k in decode.keywords if k.arg == "depends")
        paths = eval(compile(ast.Expression(depends), "<depends>", "eval"),
                     {"os": os, "here": str(ROOT)})
        self.assertIn(str(GEN_DIR / "tuned_dispatch.h"), paths)

    def test_prefill_build_tracks_kernel_headers(self):
        tree = ast.parse((ROOT / "setup.py").read_text())
        extension = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "Extension" and any(
                k.arg == "name" and isinstance(k.value, ast.Constant)
                and k.value.value == "pla.prefill.flashqla._C" for k in n.keywords))
        depends = next(k.value for k in extension.keywords if k.arg == "depends")

        def assigned(name):
            return next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

        if isinstance(depends, ast.Name):
            depends = assigned(depends.id)
        prefill = ROOT / "src/prefill/flashqla"
        scope = {"os": os, "glob": glob, "re": re, "here": str(ROOT),
                 "prefill_dir": str(prefill)}
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "local_headers")
        exec(compile(ast.Module([helper], []), "<setup.py>", "exec"), scope)
        for name in ("prefill_sources", "prefill_include_dirs"):
            scope[name] = eval(compile(ast.Expression(assigned(name)), name, "eval"),
                               scope)
        paths = eval(compile(ast.Expression(depends), "<depends>", "eval"), scope)
        tracked = {os.path.relpath(path, ROOT) for path in paths}
        # Host-side headers outside kernels/ relink _C as well.
        for header in ("src/prefill/flashqla/kernels/kkt_solve_cute_impl.cuh",
                       "src/prefill/flashqla/kernels/fused_fwd_cute_body.inc",
                       "src/prefill/flashqla/host/sanity_checks.h",
                       "src/prefill/sanity_check.h", "src/prefill/instrument.h",
                       "src/sanity_check.h", "src/op_forge_case.h",
                       "src/bypass_exec.h", "src/ppu_torch_stream.h"):
            self.assertIn(header, tracked)
        # Every repository file a source or tracked header includes is tracked.
        for path in scope["prefill_sources"] + list(paths):
            path = Path(path)
            for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"',
                                   path.read_text(), re.M):
                for base in [path.parent] + [Path(d) for d in scope["prefill_include_dirs"]]:
                    if (base / name).is_file():
                        self.assertIn(os.path.relpath((base / name).resolve(), ROOT),
                                      tracked, f"{path.name} includes {name}")
                        break
        self.assertTrue(all(path.startswith("src" + os.sep) for path in tracked))

    def test_generator_retains_chain_tree_tables(self):
        header = codegen.generate_tuned_dispatch([
            self.row(mode="chain", bv=32), self.row(mode="tree", bv=128)])
        self.assertIn("table_k_last_mtp_chain_4_12_128_128", header)
        self.assertIn("table_k_last_mtp_tree_4_12_128_128", header)
        self.assertIn("if (!tree)", header)
        self.assertIn("if (tree)", header)
        self.assertIn("{32, 4, 4, 1}", header)
        self.assertIn("{128, 4, 4, 1}", header)


class SharedLayoutTest(unittest.TestCase):
    def compile_and_run(self, source_text):
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("requires a C++17 host compiler")
        with tempfile.TemporaryDirectory(prefix="pla-layout-") as directory:
            source = Path(directory) / "layout.cpp"
            binary = Path(directory) / "layout"
            source.write_text(source_text)
            subprocess.run([compiler, "-std=c++17", "-I", str(ROOT), str(source),
                            "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_k_last_shared_layout_alignment_and_bounds(self):
        self.compile_and_run(r'''
#include "src/decode/k-last/k_last_shared_layout.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>
int main() {
    using Span = std::pair<std::size_t, std::size_t>;
    for (int sn : {4, 5, 6, 8}) for (int wn : {4, 8})
    for (int bv : {16, 32, 64, 128}) for (int hv : {1, 2, 4})
    for (int t : {0, 1, 2, 3, 4, 7, 64})
    for (bool kda : {false, true}) for (bool mtp : {false, true}) {
        const pla::KLastSharedLayout s(sn, 128, wn, bv, hv, kda, mtp, t);
        const std::size_t bg_width = kda ? hv*128 : hv;
        std::vector<Span> spans{{s.h, 4*sn*128*wn}, {s.q, 4*128}, {s.k, 4*128},
            {s.bv, 2*bv*hv}, {s.reduction_q, 4*wn}, {s.reduction_k, 4*wn},
            {s.beta, 4*hv}, {s.bg, 4*bg_width}};
        if (mtp) {
            spans.insert(spans.end(), {{s.v_mtp, 2*t*bv*hv}, {s.q_mtp, 4*t*128},
                {s.k_mtp, 4*t*128}, {s.q_raw, 2*t*128}, {s.k_raw, 2*t*128},
                {s.beta_mtp, 4*t*hv}, {s.bg_mtp, 4*t*bg_width}});
        }
        std::size_t previous_end = 0;
        for (const auto& span : spans) {
            assert(span.first % 16 == 0);
            assert(span.first >= previous_end);
            assert(span.first + span.second <= s.total_bytes);
            previous_end = span.first + span.second;
        }
        assert(s.total_bytes % 16 == 0);
        // Individual rows used by float4/cp.async also retain alignment.
        if (mtp) for (int row=0; row<t; ++row) {
            assert((s.q_mtp + row*128*4) % 16 == 0);
            assert((s.k_mtp + row*128*4) % 16 == 0);
            assert((s.q_raw + row*128*2) % 16 == 0);
            assert((s.k_raw + row*128*2) % 16 == 0);
        }
    }
    // The diagnosed BV16/WN4/SN4/HV1/T3 GDN case must be vector aligned.
    const pla::KLastSharedLayout regression(4, 128, 4, 16, 1, false, true, 3);
    assert(regression.q_mtp % 16 == 0 && regression.k_mtp % 16 == 0);
}
''')

    def test_alignment_all_supported_configs(self):
        self.compile_and_run(r'''
#include "src/decode/k-last/k_last_shared_layout.h"
#include <cassert>
#include <cstddef>
#include <initializer_list>

int main() {
    constexpr int K = 128, BV = 128;
    for (int warps : {4, 8}) for (int stages : {3, 4, 5, 6})
    for (bool parallel : {false, true}) {
        const pla::KLastSharedLayout s(stages, K, warps, BV, 1, true, false, 1, 2);
        const auto allocation = s.total_bytes + (parallel ? sizeof(float) * BV : 0);
        const auto old_bg = 4ull * (stages * K * warps + 2 * K + 2 * warps + 1)
                            + 2ull * BV;
        // This is the diagnosed scalar-beta -> float4-decay alignment defect.
        assert(old_bg % 16 == 4);
        assert(s.bg == old_bg + 12 && s.bg % 16 == 0);
        assert(s.beta + sizeof(float) <= s.bg);
        assert(s.reduction_k == s.reduction_q + sizeof(float) * warps);
        for (int lane = 0; lane < 32; ++lane) {
            for (auto base : {s.q, s.k, s.bg}) {
                const auto address = base + sizeof(float) * lane * 4;
                assert(address % 16 == 0);
                assert(address + 16 <= base + sizeof(float) * K);
                assert(address + 16 <= allocation);
            }
        }
        assert(s.bg + sizeof(float) * K == s.total_bytes);
        assert(s.total_bytes % 16 == 0);
        if (parallel)
            assert(s.total_bytes + sizeof(float) * BV == allocation);
    }
}
''')


if __name__ == "__main__":
    unittest.main()
