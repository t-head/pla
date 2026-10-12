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

"""CPU contract of the BT64 default path: real C++ gating, build and intermediate lifetimes."""
import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'src/prefill/flashkdapro'


def block(source, marker):
    start = source.index('{', source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError('unclosed block: ' + marker)


class Bt64Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = (SRC / 'host/torch_extension_cute.cc').read_text()
        cls.kernel = (SRC / 'tc01/kernel_bt64_prefill_tc01.cu').read_text()
        cls.p1 = (SRC / 'tc01/bt64_glm_n1_recurrence.cuh').read_text()

    def test_manifest(self):
        tree = ast.parse((ROOT / 'setup.py').read_text())
        value, = [node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Assign) and any(
                      isinstance(t, ast.Name) and t.id == 'flashkda_source_manifest'
                      for t in node.targets)]
        paths = [Path(*(ast.literal_eval(a) for a in item.args))
                 for item in value.elts]
        self.assertEqual(paths.count(Path('tc01/kernel_bt64_prefill_tc01.cu')), 1)
        self.assertTrue(all((SRC / p).is_file() for p in paths))
        self.assertIn('launch_flashkda_bt64_prefill_tc01(',
                      (SRC / 'common/launch_api.h').read_text())

    def test_actual_cpp_gate(self):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        if compiler is None:
            self.skipTest('no C++ compiler available')
        body = block(self.host, 'bool use_tc01_bt64_candidate(')
        prefix = r'''
#include <cstdlib>
#include <cassert>
#include <initializer_list>
namespace flashkda {
enum class KrWorkspaceLayout { kNormalRaw, kContiguousRawKrT, kAffine };
namespace host_runtime {
bool tc01 = true;
bool current_device_is_tc01() { return tc01; }
}}
struct CpPlan { bool engage; };
int cp_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}
bool gate(int T_total, int H, int N, bool is_varlen,
          const CpPlan& cp, flashkda::KrWorkspaceLayout kr_layout) {
'''
        suffix = r'''
}
int main() {
    using L = flashkda::KrWorkspaceLayout;
    const char* bt = "PLA_FLASHKDA_TC01_BT64";
    const char* cp = "PLA_FLASHKDA_CP";
    for (const char* env : {static_cast<const char*>(nullptr), "0", "1", "", "true", "10"}) {
        if (env) setenv(bt, env, 1); else unsetenv(bt);
        const bool on = !env || (env[0] == '1' && env[1] == '\0');
        const bool probe = env && env[0] == '1' && env[1] == '\0';
        for (const char* mode : {"0", "1", "2"}) {
            setenv(cp, mode, 1);
            const bool forced = mode[0] == '2';
            for (bool arch : {false, true}) {
                flashkda::host_runtime::tc01 = arch;
                for (bool varlen : {false, true}) {
                    for (bool engaged : {false, true}) {
                        for (L layout : {L::kNormalRaw, L::kContiguousRawKrT, L::kAffine}) {
                            const bool common = arch && varlen && !forced && layout != L::kAffine;
                            for (int t : {24, 64, 255, 256, 257, 4095, 4096, 8192, 32768}) {
                                assert(gate(t, 8, 1, varlen, {engaged}, layout)
                                       == (common && on && t >= 256));
                                for (int n : {2, 3, 4, 5})
                                    assert(gate(t, 8, n, varlen, {engaged}, layout)
                                           == (common && probe && n <= 4 && t >= 64));
                                for (int h : {4, 12, 24, 48, 96}) {
                                    assert(gate(t, h, 1, varlen, {engaged}, layout)
                                           == (common && on && h == 4 && t >= 256));
                                    assert(gate(t, h, 4, varlen, {engaged}, layout)
                                           == (common && probe && !engaged && (h == 12 || h == 24) && t >= 8192));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
'''
        with tempfile.TemporaryDirectory(prefix='.bt64-gate-', dir=ROOT) as directory:
            source, binary = Path(directory) / 'gate.cc', Path(directory) / 'gate'
            source.write_text(prefix + body + suffix)
            subprocess.run([compiler, '-std=c++17', str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_p1_launch_scope(self):
        launch = block(self.kernel, 'void launch_bt64_impl(')
        condition = 'H == 8 && N == 1 && T_total >= 256'
        branch = block(launch, 'if (' + condition + ')')
        self.assertIn('flashkda_bt64_recurrence_n1_kernel', branch)
        self.assertNotIn('flashkda_bt64_recurrence_n1_kernel', launch.replace(branch, ''))
        self.assertIn('flashkda_bt64_recurrence_kernel', launch.replace(branch, ''))
        self.assertEqual(self.kernel.count('#include "bt64_glm_n1_recurrence.cuh"'), 1)
        self.assertIn('!flashkda::host_runtime::current_device_is_tc01()',
                      block(self.kernel, 'void launch_flashkda_bt64_prefill_tc01('))
        # The CP experiment is not shipped with the default path; if the workspace keeps that entry, keep checking the isolation contract.
        if 'void launch_bt64_segment_recurrence(' in self.kernel:
            self.assertNotIn('recurrence_n1_kernel', block(
                self.kernel, 'void launch_bt64_segment_recurrence('))
            self.assertIn('!flashkda::host_runtime::current_device_is_tc01()', block(
                self.kernel, 'void launch_flashkda_bt64_cp_segment_tc01('))

    def test_state_layout_bijection_and_vec16(self):
        offset_body = re.sub(r'\s+', '', block(self.p1, 'int recurrence_state_offset_tc01('))
        self.assertEqual(offset_body, 'returnn*DK+(k^((n&7)<<3));')
        offset = lambda n, k: n * 128 + (k ^ ((n & 7) << 3))
        self.assertEqual(sorted(offset(n, k) for n in range(16) for k in range(128)),
                         list(range(2048)))
        for n in range(16):
            for k in range(0, 128, 8):
                start = offset(n, k)
                self.assertEqual(start % 8, 0)
                self.assertEqual([offset(n, k + i) for i in range(8)],
                                 list(range(start, start + 8)))
        pairs = [(lane >> 2) + 8 * (i >> 1) for lane in range(32) for i in range(4)]
        self.assertEqual(len(pairs), 128)
        for kb in range(8):
            owned = [(n, k + half) for lane in range(32) for i in range(4)
                     for n in [(lane >> 2) + 8 * (i >> 1)]
                     for k in [kb * 16 + 2 * (lane & 3) + 8 * (i & 1)]
                     for half in range(2)]
            self.assertEqual(sorted(owned), [(n, k) for n in range(16)
                                           for k in range(kb * 16, (kb + 1) * 16)])

    def test_state_layout_io_contract(self):
        body = block(self.p1, 'flashkda_bt64_recurrence_n1_kernel(')
        for marker in ('sm.state[recurrence_state_offset_tc01(lc, dk)] = value;',
                       'const BF16 value = sm.state[recurrence_state_offset_tc01(lc, dk)];',
                       'recurrence_checkpoint_state_t_tc01(',
                       'init_state_indices ? init_state_indices[seq] : seq',
                       'has_state_out && final_state != nullptr',
                       'final_state_indices[seq] : seq',
                       'if (final_state_is_fp32)',
                       'init_state_is_fp32'):
            self.assertIn(marker, body)
        checkpoint = block(self.p1, 'void recurrence_checkpoint_state_t_tc01(')
        self.assertIn('(int64_t(gbt * H + head) * DV + dv_base + n) * DK + k', checkpoint)
        self.assertIn('*reinterpret_cast<const uint4*>(state + recurrence_state_offset_tc01(n, k))', checkpoint)
        update = block(self.p1, 'void recurrence_store_state_t_acc_tc01(')
        self.assertIn('BF16(old * gates[v] + rC(v, 0, 0))', update)
        self.assertLess(update.index('__syncwarp();'),
                        update.index('fk::bf16_cfrag_to_bfrag_layout_u32'))
        self.assertIn('state + recurrence_state_offset_tc01(n, k)) = b_regs[i]', update)
        self.assertIn('sizeof(RecurrenceSmem) == 23040', self.kernel)

    def test_allocation_and_fallback(self):
        allocation = self.host.index('bool bt64_ready = use_tc01_bt64_candidate(')
        oom = self.host.index('catch (const c10::OutOfMemoryError&)', allocation)
        prepare = self.host.index('launch_prepare_range(0, total_tiles, 0, H, nullptr, stream);', oom)
        launch = self.host.index('launch_flashkda_bt64_prefill_tc01(', prepare)
        self.assertLess(allocation, oom)
        self.assertLess(oom, prepare)
        self.assertLess(prepare, launch)
        self.assertIn('bt64_ready = false;', block(self.host[allocation:], 'catch (const c10::OutOfMemoryError&)'))
        self.assertIn('return;', self.host[launch:self.host.index('if (!cp.engage)', launch)])

    def test_pipeline_order(self):
        launch = block(self.kernel, 'void launch_bt64_impl(')
        # The workspace may keep a separately experimental segmented wrapper; the release path itself does not depend on it.
        if 'launch_bt64_parallel_front<' in launch:
            launch = (block(self.kernel, 'void launch_bt64_parallel_front(')
                      + launch + block(self.kernel, 'void launch_bt64_parallel_back('))
        markers = ['"split pair launch"', '"split merge launch"',
                   '"prepare recompute launch"', '"recompute launch"',
                   '"recurrence launch"', '"prepare_qg launch"', '"output launch"']
        positions = [launch.index(m) for m in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('BT64_WS_OFF_W_OR_A_U == fk::BT64_WS_OFF_W_U', self.kernel)
        self.assertIn('BT64_W_UNITS == BT * DK', self.kernel)

    def test_required_pointer_guards(self):
        launch = block(self.kernel, 'void launch_flashkda_bt64_prefill_tc01(')
        for guard in ('!flashkda::host_runtime::current_device_is_tc01()',
                      '!o || !v || !ws16 || !bt_ws || !x || !h_checkpoint || !cu_seqlens',
                      'has_state_in && !init_state', 'has_state_out && !final_state'):
            self.assertIn(guard, launch)
        self.assertEqual(launch.count('bt64::launch_bt64_impl<'), 2)


if __name__ == '__main__':
    unittest.main()
