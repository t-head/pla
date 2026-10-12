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

import importlib.util
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

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
