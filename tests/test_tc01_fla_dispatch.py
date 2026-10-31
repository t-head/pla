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

"""Standalone CPU regression: real Python entry/rules and build layout, with torch and the AOT extension mocked.

Run: python3 tests/test_tc01_fla_dispatch.py
Does not load the installed pla/torch, does not run GPU kernels, and does not replace full wheel/PPU validation.
"""
import ast
from contextlib import nullcontext
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'pla/prefill/flashkdapro'
SOURCE = ROOT / 'src/prefill/flashkdapro/tc01/fla'
ALIAS = '_tc01_dispatch_test_package.prefill.flashkdapro'
HEADS = (0, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
CP_HEADS = (4, 6, 8, 12)
STAGES = ('l2norm', 'metadata', 'gate', 'inter', 'recompute', 'recurrence',
          'cp_recurrence_output', 'token_output')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HostTensor:
    """Provides Host attributes only; any unexpected D2H fails the test."""

    def __init__(self, shape, dtype='torch.bfloat16', device='cuda:0'):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.startswith('cuda:')

    def size(self, axis):
        return self.shape[axis]

    def contiguous(self):
        return self

    def item(self):
        raise AssertionError('dispatch must not do D2H')

    cpu = tolist = item

    def __getitem__(self, index):
        raise AssertionError('dispatch must not read device metadata')


def synthetic_rules(api_threshold=128):
    """Synthetic rules for branch regression only; not a performance calibration result."""
    return dict(metrics={metric: dict(b1=dict(tables=[
        dict(heads=8, min_t=1, threshold=threshold)]))
        for metric, threshold in (('graph_ms', 128), ('api_ms', api_threshold))})


class DispatchTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        torch = types.ModuleType('torch')
        torch.Tensor = HostTensor
        torch.int32, torch.int64 = 'torch.int32', 'torch.int64'
        torch.cuda = types.SimpleNamespace(
            get_device_capability=Mock(return_value=(8, 0)),
            device=Mock(side_effect=lambda device: nullcontext()),
            is_current_stream_capturing=Mock(return_value=False))
        torch.iinfo = lambda dtype: types.SimpleNamespace(min=-(1 << 31), max=(1 << 31) - 1)
        sys.modules['torch'] = self.torch = torch
        # Keep the real relative-import hierarchy; load _sanity without importing the GPU extension.
        for name, path in ((ALIAS.split('.')[0], ROOT / 'pla'),
                           (ALIAS.rsplit('.', 1)[0], ROOT / 'pla/prefill')):
            parent = types.ModuleType(name)
            parent.__path__ = [str(path)]
            sys.modules[name] = parent
        legacy = types.ModuleType(ALIAS + '._C')
        legacy.flashkda_fwd = Mock(return_value=None)
        sys.modules[legacy.__name__] = legacy
        self.legacy = legacy.flashkda_fwd
        extension = types.ModuleType(ALIAS + '._tc01_fla._C')
        extension.runtime_abi = 'tc01_fla_monolithic_runtime_v2'
        extension.kernel_abi = 'tc01_fla_packed_v3_cpstate_v1'
        extension.cp_abi = 'tc01_fla_cp_state_passing_v2'
        extension.cp_heads = CP_HEADS
        extension.supported_heads = HEADS[1:]
        extension.generic_head_max = 65535
        extension.manifest_schema = 'tc01_fla_monolithic_build_v2'
        extension.forward = Mock(return_value=None)
        sys.modules[extension.__name__] = self.extension = extension
        self.package = load_module(ALIAS, PACKAGE / '__init__.py')
        self.ops = sys.modules[ALIAS + '.ops']

    def inputs(self, tokens=64, heads=8):
        return [HostTensor((tokens, heads, 128)) for _ in range(4)] + [
            HostTensor((tokens, heads)), HostTensor((heads,), 'torch.float32'),
            HostTensor((heads, 128), 'torch.float32'), HostTensor((tokens, heads, 128))]

    def rules(self, value):
        module = importlib.import_module(ALIAS + '._dispatch_rules')
        mocked = patch.object(module, 'load_rules', return_value=value)
        mocked.start()
        self.addCleanup(mocked.stop)
        return module

    def assert_fla_unloaded(self):
        self.assertNotIn(ALIAS + '._tc01_fla', sys.modules)
        self.assertNotIn(ALIAS + '._tc01_fla.monolithic', sys.modules)
        self.extension.forward.assert_not_called()

    def test_legacy_sanity_precedes_int64_conversion(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'legacy'
        os.environ['PLA_SANITY_CHECK'] = '1'
        cu = HostTensor((2,), 'torch.int64')
        with patch.object(self.ops._sanity, 'offsets', side_effect=AssertionError('illegal offsets')):
            with self.assertRaisesRegex(AssertionError, 'illegal offsets'):
                self.package.flashkda_fwd(*self.inputs(), cu_seqlens=cu)
        self.legacy.assert_not_called()

    def test_default_and_empty_use_rules_for_both_backends(self):
        self.rules(synthetic_rules())
        for value in (None, ''):
            for tokens, backend in ((64, 'legacy'), (256, 'fla')):
                with self.subTest(value=value, tokens=tokens):
                    if value is None:
                        os.environ.pop('PLA_FLASHKDA_TC01_IMPL', None)
                    else:
                        os.environ['PLA_FLASHKDA_TC01_IMPL'] = value
                    self.legacy.reset_mock()
                    self.extension.forward.reset_mock()
                    result = self.package.flashkda_fwd(*self.inputs(tokens), raw_cu_seqlens=(0, tokens))
                    self.assertIsNone(result)
                    self.assertEqual(self.legacy.call_count, int(backend == 'legacy'))
                    self.assertEqual(self.extension.forward.call_count, int(backend == 'fla'))

    def test_auto_legacy_does_not_load_fla(self):
        self.rules(synthetic_rules())
        self.package.flashkda_fwd(*self.inputs(), raw_cu_seqlens=(0, 64))
        self.legacy.assert_called_once()
        self.assert_fla_unloaded()

    def test_metrics_require_conservative_intersection(self):
        self.rules(synthetic_rules(api_threshold=32))
        self.package.flashkda_fwd(*self.inputs(), raw_cu_seqlens=(0, 64))
        self.extension.forward.assert_called_once()
        self.legacy.assert_not_called()

    def test_explicit_overrides_bypass_rules(self):
        with patch.object(self.ops, 'dispatch_tc01_impl', side_effect=AssertionError('rules must not be called')):
            for backend in ('fla', 'legacy'):
                with self.subTest(backend=backend):
                    os.environ['PLA_FLASHKDA_TC01_IMPL'] = backend
                    self.legacy.reset_mock()
                    self.extension.forward.reset_mock()
                    self.package.flashkda_fwd(*self.inputs())
                    self.assertEqual(self.legacy.call_count, int(backend == 'legacy'))
                    self.assertEqual(self.extension.forward.call_count, int(backend == 'fla'))

    def test_invalid_impl_fails_before_extension(self):
        for value in ('auto', 'staged', 'FLA', ' '):
            with self.subTest(value=value):
                os.environ['PLA_FLASHKDA_TC01_IMPL'] = value
                with self.assertRaisesRegex(ValueError, 'PLA_FLASHKDA_TC01_IMPL'):
                    self.package.flashkda_fwd(*self.inputs())
        self.legacy.assert_not_called()
        self.assert_fla_unloaded()

    def test_tc02_ignores_tc01_configuration_and_rules(self):
        self.torch.cuda.get_device_capability.return_value = (8, 9)
        os.environ['PLA_FLASHKDA_TC01_FLA_RUNTIME'] = 'invalid'
        with patch.object(self.ops, 'dispatch_tc01_impl', side_effect=AssertionError('tc02 must not be diverted')):
            for value in ('', 'fla', 'legacy', 'invalid'):
                os.environ['PLA_FLASHKDA_TC01_IMPL'] = value
                self.package.flashkda_fwd(*self.inputs())
        self.assertEqual(self.legacy.call_count, 4)
        self.assert_fla_unloaded()
        self.assertNotIn(ALIAS + '._dispatch_rules', sys.modules)

    def test_retired_environment_has_no_runtime_effect(self):
        self.rules(synthetic_rules())
        for value in ('staged', 'monolithic', '', 'invalid'):
            for tokens, backend in ((64, 'legacy'), (256, 'fla')):
                with self.subTest(value=value, tokens=tokens):
                    os.environ['PLA_FLASHKDA_TC01_FLA_RUNTIME'] = value
                    os.environ['PLA_FLASHKDA_FLA_BUILD'] = value
                    self.legacy.reset_mock()
                    self.extension.forward.reset_mock()
                    self.package.flashkda_fwd(*self.inputs(tokens), raw_cu_seqlens=(0, tokens))
                    self.assertEqual(self.legacy.call_count, int(backend == 'legacy'))
                    self.assertEqual(self.extension.forward.call_count, int(backend == 'fla'))

    def test_uncalibrated_inputs_use_fla_without_d2h(self):
        self.rules(synthetic_rules())
        cases = [
            (self.inputs(heads=5), dict(raw_cu_seqlens=(0, 64))),
            (self.inputs(), dict(cu_seqlens=HostTensor((2,), 'torch.int32'))),
            (self.inputs(), dict(cu_seqlens=HostTensor((2,), 'torch.int64'))),
            (self.inputs(), dict(raw_cu_seqlens=(0, 32))),
            (self.inputs(), dict(raw_cu_seqlens=(0, 0, 64))),
            (self.inputs(), dict(raw_cu_seqlens=(0, 64), final_state=HostTensor((1, 8, 128, 128)))),
            (self.inputs(), dict(raw_cu_seqlens=(0, 64), state_pool=HostTensor((4, 8, 128, 128), 'torch.float32'),
                                state_indices=HostTensor((1,), 'torch.int32'))),
            (self.inputs(), dict(raw_cu_seqlens=(0, 64), cu_seqlens=HostTensor((3,), 'torch.int32'))),
            (self.inputs(), dict(raw_cu_seqlens=(False, 64))),
        ]
        cases.append(([HostTensor((64, 8))] + self.inputs()[1:], dict(raw_cu_seqlens=(0, 64))))
        for args, kwargs in cases:
            with self.subTest(kwargs=kwargs, shape=args[0].shape):
                self.extension.forward.reset_mock()
                self.package.flashkda_fwd(*args, **kwargs)
                self.extension.forward.assert_called_once()
        self.legacy.assert_not_called()

    def test_missing_rules_use_fla(self):
        rules = importlib.import_module(ALIAS + '._dispatch_rules')
        with patch.object(rules, 'FROZEN_MODULE', Mock(is_file=Mock(return_value=False))):
            self.package.flashkda_fwd(*self.inputs(), raw_cu_seqlens=(0, 64))
        self.extension.forward.assert_called_once()
        self.legacy.assert_not_called()

    def test_fla_forwards_every_argument_unchanged(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'fla'
        args = self.inputs()
        optional = [0.25, -3.0, HostTensor((2, 8, 128, 128)), HostTensor((2, 8, 128, 128)),
                    HostTensor((3,), 'torch.int64'), None, HostTensor((2,), 'torch.int32'), [0, 32, 64]]
        self.assertIsNone(self.package.flashkda_fwd(*(args + optional)))
        actual = self.extension.forward.call_args.args
        self.assertEqual(len(actual), 16)
        for received, original in zip(actual, args + optional):
            self.assertIs(received, original)
        self.legacy.assert_not_called()

    def test_fla_pool_and_default_scale_are_not_rewritten(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'fla'
        pool = HostTensor((4, 8, 128, 128))
        indices = HostTensor((1,), 'torch.int32')
        self.package.flashkda_fwd(*self.inputs(), state_pool=pool, state_indices=indices)
        actual = self.extension.forward.call_args.args
        self.assertIsNone(actual[8])
        self.assertEqual(actual[10:13], (None, None, None))
        self.assertIs(actual[13], pool)
        self.assertIs(actual[14], indices)

    def test_indexed_bf16_auto_legacy_preserves_pool_mapping(self):
        self.rules(synthetic_rules())
        pool = HostTensor((4, 8, 128, 128))
        indices = HostTensor((1,), 'torch.int32')
        cu = HostTensor((2,), 'torch.int32')
        raw = (0, 64)
        args = self.inputs()
        self.package.flashkda_fwd(*args, state_pool=pool, state_indices=indices,
                                 cu_seqlens=cu, raw_cu_seqlens=raw)
        self.legacy.assert_called_once_with(*args, 128 ** -0.5, -5.0, pool, pool, cu, indices, raw)
        self.assert_fla_unloaded()

    def test_legacy_keeps_int64_conversion_and_range_check(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'legacy'
        cu = Mock(dtype='torch.int64', device='cuda:0')
        cu32 = HostTensor((2,), 'torch.int32')
        cu.to.return_value.contiguous.return_value = cu32
        cu.__getitem__ = Mock(return_value=Mock(item=Mock(return_value=64)))
        args = self.inputs()
        self.package.flashkda_fwd(*args, cu_seqlens=cu)
        self.assertIs(self.legacy.call_args.args[12], cu32)
        cu.to.assert_called_once_with('torch.int32')
        cu.__getitem__.return_value.item.return_value = 1 << 31
        with self.assertRaisesRegex(ValueError, 'int32 range'):
            self.package.flashkda_fwd(*args, cu_seqlens=cu)
        self.assertEqual(self.legacy.call_count, 1)

    def test_legacy_keeps_device_and_state_checks(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'legacy'
        for kwargs in (dict(state_pool=HostTensor((1, 8, 128, 128))),
                       dict(state_indices=HostTensor((1,), 'torch.int32')),
                       dict(final_state=HostTensor((1, 8, 128, 128), device='cuda:1'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.package.flashkda_fwd(*self.inputs(), **kwargs)
        self.legacy.assert_not_called()

    def test_unsupported_device_rejected(self):
        self.torch.cuda.get_device_capability.return_value = (9, 0)
        with self.assertRaisesRegex(RuntimeError, 'capability'):
            self.package.flashkda_fwd(*self.inputs())
        args = self.inputs()
        args[0] = HostTensor((64, 8, 128), device='cpu')
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.package.flashkda_fwd(*args)
        self.assert_fla_unloaded()

    def test_cold_capture_rejected_and_warm_entry_cached(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'fla'
        self.torch.cuda.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'Graph capture'):
            self.package.flashkda_fwd(*self.inputs())
        self.extension.forward.assert_not_called()
        self.torch.cuda.is_current_stream_capturing.return_value = False
        self.package.flashkda_fwd(*self.inputs())
        self.torch.cuda.is_current_stream_capturing.return_value = True
        self.package.flashkda_fwd(*self.inputs())
        self.assertEqual(self.extension.forward.call_count, 2)
        self.assertEqual(self.torch.cuda.is_current_stream_capturing.call_count, 2)

    def test_abi_and_incomplete_build_are_rejected(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'fla'
        for name, value in (('runtime_abi', 'bad'), ('kernel_abi', 'bad'),
                            ('cp_abi', 'bad'), ('cp_heads', (4, 8)),
                            ('supported_heads', (4, 8)), ('generic_head_max', 8),
                            ('manifest_schema', 'bad')):
            with self.subTest(name=name), patch.object(self.extension, name, value):
                with self.assertRaisesRegex(RuntimeError, 'ABI/head/CP'):
                    self.package.flashkda_fwd(*self.inputs())
        self.extension.forward.assert_not_called()
        self.legacy.assert_not_called()

    def test_fla_errors_do_not_fall_back(self):
        os.environ['PLA_FLASHKDA_TC01_IMPL'] = 'fla'
        self.extension.forward.side_effect = RuntimeError('test runtime error')
        with self.assertRaisesRegex(RuntimeError, 'test runtime error'):
            self.package.flashkda_fwd(*self.inputs())
        self.legacy.assert_not_called()


class BuildTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('tc01_build_test', SOURCE / 'monolithic_build.py')
        self.build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.build)
        self.flags = json.loads((SOURCE / 'build_flags.json').read_text())

    def setup_extensions(self, environment=None, command='build_ext'):
        """Only executes the real setup.py FlashKDA config block, without loading torch/SDK or running a build."""
        tree = ast.parse((ROOT / 'setup.py').read_text())
        blocks = [node for node in tree.body if isinstance(node, ast.If)
                  and any(isinstance(item, ast.Constant) and item.value == 'pla.prefill.flashkdapro'
                          for item in ast.walk(node.test))]
        self.assertEqual(len(blocks), 1)
        namespace = dict(here=str(ROOT), os=os, sys=sys, importlib=importlib,
                         NAMESPACES=['pla.prefill.flashkdapro'], ext_modules=[],
                         Extension=lambda **kwargs: types.SimpleNamespace(**kwargs),
                         actlize_includes=[], PREFILL_CXX_FLAGS=[], PREFILL_HGCC_FLAGS=[])
        module = ast.Module(body=blocks, type_ignores=[])
        with patch.dict(os.environ, environment or {}, clear=True), patch.object(sys, 'argv', ['setup.py', command]):
            exec(compile(module, str(ROOT / 'setup.py'), 'exec'), namespace)
        return namespace['ext_modules']

    def test_build_has_only_legacy_and_monolithic_extensions(self):
        for value in (None, 'staged', 'monolithic', 'invalid', ''):
            with self.subTest(value=value):
                env = {} if value is None else {'PLA_FLASHKDA_FLA_BUILD': value}
                extensions = self.setup_extensions(env, command='bdist_wheel')
                self.assertEqual([ext.name for ext in extensions],
                                 ['pla.prefill.flashkdapro._C', 'pla.prefill.flashkdapro._tc01_fla._C'])
                self.assertEqual(extensions[1].fla_heads, HEADS)
                self.assertEqual([Path(p).name for p in extensions[1].sources],
                                 ['runtime_binding.cc', 'runtime.cc', 'dispatch.cc'])
                self.assertEqual(extensions[1].extra_compile_args,
                                 dict(cxx=self.build.HOST_FLAGS, hgcc=[]))
                self.assertEqual(extensions[1].extra_link_args,
                                 ['-Wl,-Bsymbolic', '-Wl,--version-script=' + str(SOURCE / 'exports.map')])
                for ext in extensions:
                    self.assertTrue(all(Path(source).is_file() for source in ext.sources))

    def test_partial_probes_work_but_cannot_publish_wheels(self):
        for env in ({'PLA_FLA_BUILD_HEADS': '0,4,8'}, {'PLA_FLA_BUILD_ONLY': '1'},
                    {'PLA_FLA_BUILD_HEADS': '0', 'PLA_FLA_BUILD_ONLY': '1'}):
            with self.subTest(env=env):
                extensions = self.setup_extensions(env)
                self.assertEqual(len(extensions), 1 if env.get('PLA_FLA_BUILD_ONLY') == '1' else 2)
                self.assertEqual(extensions[-1].fla_heads, self.build.build_heads(env.get('PLA_FLA_BUILD_HEADS')))
                with self.assertRaisesRegex(RuntimeError, 'cannot publish a wheel'):
                    self.setup_extensions(env, command='bdist_wheel')
        for value in ('', '4', '0,0', '0,5', 'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build.build_heads(value)

    def test_object_groups_sources_heads_and_flag_order(self):
        self.assertEqual(self.build.HEADS, HEADS)
        self.assertEqual(self.build.STAGES, STAGES)
        groups = self.build.object_groups(self.flags)
        expected = [(stage, head) for stage in STAGES
                    for head in ((0,) if stage in ('l2norm', 'metadata') else
                                 (CP_HEADS if stage == 'cp_recurrence_output' else HEADS))]
        expected += [('adapter', head) for head in HEADS]
        self.assertEqual([(g['stage'], g['heads']) for g in groups], expected)
        self.assertEqual(len(groups), 72)
        self.assertEqual(sum(len(g['sources']) for g in groups), 87)
        kernels = set()
        objects = set()
        for group in groups:
            stage, head = group['stage'], group['heads']
            define = [f'-DFLA_HEADS={head}']
            if stage == 'adapter':
                self.assertEqual(group['flags'], dict(cxx=self.build.HOST_FLAGS + define, hgcc=[]))
            else:
                for key, old in (('cxx', 'cflags'), ('hgcc', 'cuda_cflags')):
                    self.assertEqual(group['flags'][key], self.flags[stage][old] + define)
            for name in group['sources']:
                self.assertTrue((SOURCE / name).is_file(), name)
                path = (stage, head, Path(name).stem)
                self.assertNotIn(path, objects)
                objects.add(path)
                if name.endswith('.cu'):
                    kernels.add(name)
        self.assertEqual(kernels, {p.name for p in SOURCE.glob('*.cu')})
        with self.assertRaisesRegex(ValueError, 'list is incomplete'):
            self.build.object_groups({})

    def test_cp_auto_short_buckets_and_explicit_overrides(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('requires a C++17 host compiler')
        runtime = (SOURCE / 'runtime.cc').read_text()
        selectors = runtime.split('struct CpSelection {', 1)[1].split(
            'std::shared_mutex init_mutex;', 1)[0]
        source = r'''
#include "cp_plan.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>
using namespace fla_runtime;
struct Arguments { std::optional<std::vector<int64_t>> raw; };
struct Shape { int n; int h; int64_t t = 0; };
struct Launchers {
  void (*cp_recurrence_output)();
  Launchers(bool available) : cp_recurrence_output(available ? +[]{} : nullptr) {}
};
void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}
struct CpSelection {
''' + selectors + r'''
int main() {
  unsetenv("PLA_FLASHKDA_TC01_FLA_CP");
  unsetenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO");
  for (int h : {4, 6, 8}) {
    assert(select_fla_cp_auto_split(h, 0) == 0);
    assert(select_fla_cp_auto_split(h, -1) == 0);
    assert(select_fla_cp_auto_split(h, 102401) == 0);
    for (int t : {1, 16, 19, 32, 63, 64, 65, 96, 127, 128}) {
      Arguments a{std::vector<int64_t>{0, t}};
      assert(select_cp(a, {1, h}, {true}).split == 1);
      assert(select_cp_algorithm(1, h, t) == CpAlgorithm::State);
      assert(!select_cp(Arguments{}, {1, h}, {true}).active());
      assert(!select_cp(a, {2, h}, {true}).active());
      assert(!select_cp(a, {1, h}, {false}).active());
      setenv("PLA_FLASHKDA_TC01_FLA_CP", "off", 1);
      assert(!select_cp(a, {1, h}, {true}).active());
      setenv("PLA_FLASHKDA_TC01_FLA_CP", "1", 1);
      assert(select_cp_algorithm(1, h, t) == CpAlgorithm::Legacy);
      setenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO", "state", 1);
      assert(select_cp_algorithm(1, h, t) == CpAlgorithm::State);
      setenv("PLA_FLASHKDA_TC01_FLA_CP", "auto", 1);
      setenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO", "legacy", 1);
      assert(select_cp_algorithm(1, h, t) == CpAlgorithm::Legacy);
      if (h == 4) assert(!select_cp(a, {1, h}, {true}).active());
      unsetenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO");
    }
  }
  for (int t : {2049, 8192, 12287, 65537, 102400})
    assert(select_fla_cp_auto_split(4, t) == 0);
  assert(select_fla_cp_auto_split(4, 2048) == 1);
  for (int t : {12288, 16384, 65536}) {
    assert(select_fla_cp_auto_split(4, t) == 5);
    assert(select_cp_algorithm(5, 4, t) == CpAlgorithm::State);
  }
  for (int h : {6, 8}) {
    assert(select_fla_cp_auto_split(h, 102400) == 1);
    assert(select_cp_algorithm(1, h, 65536) == CpAlgorithm::State);
    assert(select_cp_algorithm(1, h, 65537) == CpAlgorithm::Legacy);
  }
  assert(select_fla_cp_auto_split(12, 8191) == 0);
  assert(select_fla_cp_auto_split(12, 8192) == 1);
  for (int h : {1, 5, 16, 24, 96})
    assert(select_fla_cp_auto_split(h, 19) == 0);
  for (int h : {4, 6, 8, 12}) {
    for (int t : {4095, 4096, 4097, 8191, 8192, 8193}) {
      for (const char* automatic : {static_cast<const char*>(nullptr), "", "auto"}) {
        if (automatic) setenv("PLA_FLASHKDA_TC01_FLA_CP", automatic, 1);
        else unsetenv("PLA_FLASHKDA_TC01_FLA_CP");
        for (const char* algorithm : {static_cast<const char*>(nullptr), "", "state", "legacy", "invalid"}) {
          if (algorithm) setenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO", algorithm, 1);
          else unsetenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO");
          const bool expected = h == 8 && (t == 4096 || t == 8192) &&
              (!algorithm || !algorithm[0] || std::strcmp(algorithm, "state") == 0);
          const auto cp = select_cp(Arguments{}, {1, h, t}, {true});
          assert(cp.active() == expected);
          if (expected) assert(cp.split == 1 && cp.count == 1 && cp.descriptor_row == -1);
          assert(!select_cp(Arguments{}, {2, h, t}, {true}).active());
          assert(!select_cp(Arguments{}, {1, h, t}, {false}).active());
        }
      }
      for (const char* opt_out : {"off", "0"}) {
        setenv("PLA_FLASHKDA_TC01_FLA_CP", opt_out, 1);
        unsetenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO");
        assert(!select_cp(Arguments{}, {1, h, t}, {true}).active());
      }
      for (const char* forced : {"1", "2", "3", "4", "5", "8"}) {
        setenv("PLA_FLASHKDA_TC01_FLA_CP", forced, 1);
        bool rejected = false;
        try { select_cp(Arguments{}, {1, h, t}, {true}); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
      }
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / 'cp_policy')
            subprocess.run([compiler, '-std=c++17', '-x', 'c++', '-I', str(SOURCE),
                            '-', '-o', binary], input=source, text=True, check=True)
            subprocess.run([binary], check=True)

    def test_n1_metadata_matches_device_geometry_and_tail(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('requires a C++17 host compiler')
        metadata = (SOURCE / 'metadata.cu').read_text()
        geometry = '__global__ void cp_device_geometry(' + metadata.split(
            '__global__ void cp_device_geometry(', 1)[1].split(
                '// Device-offset output tail:', 1)[0]
        fused = 'template<class Offset>\n__global__ void metadata_n1_tail(' + metadata.split(
            'template<class Offset>\n__global__ void metadata_n1_tail(', 1)[1].split(
                '}  // namespace', 1)[0]
        # Run the actual kernel bodies as scalar host functions, executing CTA0 last.
        # This covers metadata/tail semantics, not device scheduling or performance.
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define __global__
struct Dim { int x; } blockIdx, threadIdx, blockDim, gridDim;
struct uint4 { uint32_t x, y, z, w; };
uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) { return {x,y,z,w}; }
void __trap() { throw std::invalid_argument("invalid device offsets"); }
''' + geometry + fused + r'''
template<class Offset> void check(int tokens, int split, int grid) {
  const int capacity=(tokens+63)/64;
  constexpr int width=9, row_vectors=64;
  constexpr uint32_t poison=0x7fc00000u;
  std::vector<int32_t> indices(2*capacity+2, -7), bounds(width, -7);
  std::vector<int64_t> cp_offsets(width, -7);
  int32_t cu[2]={-7,-7}, slot=-7;
  int64_t offsets[2]={-7,-7};
  std::vector<uint4> out(int64_t(tokens)*row_vectors+2);
  blockDim.x=256; gridDim.x=grid;
  for(int length : {tokens, tokens/2+1, 12289, 12287, 65, 0, 19, tokens-1, tokens}) {
    if(length>tokens) continue;
    Offset input[2]={0, static_cast<Offset>(length)};
    std::fill(out.begin(),out.end(),make_uint4(poison,poison,poison,poison));
    std::fill(indices.begin(),indices.end(),-7);
    std::fill(bounds.begin(),bounds.end(),-7);
    std::fill(cp_offsets.begin(),cp_offsets.end(),-7);
    cu[0]=cu[1]=slot=-7; offsets[0]=offsets[1]=-7;
    for(blockIdx.x=grid-1;blockIdx.x>=0;--blockIdx.x)
      for(threadIdx.x=255;threadIdx.x>=0;--threadIdx.x)
        metadata_n1_tail(input,cu,offsets,indices.data()+1,&slot,tokens,
            capacity,bounds.data(),cp_offsets.data(),split,width,12288,65536,
            out.data()+1,row_vectors);
    const int chunks=(length+63)/64;
    assert(cu[0]==0 && cu[1]==length && slot==0);
    assert(offsets[0]==0 && offsets[1]==chunks);
    assert(indices.front()==-7 && indices.back()==-7);
    for(int i=0;i<capacity;++i) {
      assert(indices[2*i+1]==(i<chunks ? 0 : -1));
      assert(indices[2*i+2]==(i<chunks ? i : -1));
    }
    if(split>1) {
      std::vector<int32_t> expected_bounds(width,-7);
      std::vector<int64_t> expected_offsets(width,-7);
      threadIdx.x=0;
      cp_device_geometry(cu,expected_bounds.data(),expected_offsets.data(),
                         split,width,12288,65536);
      assert(bounds==expected_bounds && cp_offsets==expected_offsets);
    } else {
      assert(std::all_of(bounds.begin(),bounds.end(),[](int v){return v==-7;}));
      assert(std::all_of(cp_offsets.begin(),cp_offsets.end(),[](int64_t v){return v==-7;}));
    }
    for(int64_t i=0;i<int64_t(out.size());++i) {
      const uint32_t expected=i>int64_t(length)*row_vectors &&
          i<=int64_t(tokens)*row_vectors ? 0 : poison;
      assert(out[i].x==expected && out[i].y==expected &&
             out[i].z==expected && out[i].w==expected);
    }
  }
  blockIdx.x=threadIdx.x=0;
  for(auto input : {std::vector<Offset>{1,19}, std::vector<Offset>{0,-1},
                    std::vector<Offset>{0,static_cast<Offset>(tokens+1)}}) {
    bool rejected=false;
    try { metadata_n1_tail(input.data(),cu,offsets,indices.data()+1,&slot,
        tokens,capacity,bounds.data(),cp_offsets.data(),split,width,12288,
        65536,out.data()+1,row_vectors); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
}
int main() {
  for(int tokens : {16384,32768})
    for(int split : {0,1,5})
      for(int grid : {3,512}) {
        check<int32_t>(tokens,split,grid);
        check<int64_t>(tokens,split,grid);
      }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / 'metadata_n1')
            subprocess.run([compiler, '-std=c++17', '-O2', '-x', 'c++', '-',
                            '-o', binary], input=source, text=True, check=True)
            subprocess.run([binary], check=True)

    def test_runtime_abi_agrees_with_python_and_build(self):
        contract = {}
        exec((PACKAGE / '_tc01_fla/_contract.py').read_text(), contract)
        header = (SOURCE / 'runtime.h').read_text()
        self.assertIn('"' + self.build.RUNTIME_ABI + '"', header)
        self.assertIn('"' + self.build.KERNEL_ABI + '"', header)
        self.assertIn('"' + self.build.CP_ABI + '"', header)
        self.assertEqual(contract['ABI'], self.build.KERNEL_ABI)
        self.assertEqual(contract['CP_ABI'], self.build.CP_ABI)
        self.assertEqual(contract['CP_HEADS'], CP_HEADS)
        self.assertEqual(contract['HEADS'], HEADS[1:])

    def test_staged_runtime_and_bindings_are_absent(self):
        self.assertFalse((PACKAGE / '_tc01_fla/runtime.py').exists())
        self.assertFalse((SOURCE / 'binding_common.h').exists())
        self.assertEqual({p.name for p in SOURCE.glob('*_binding.cc')}, {'runtime_binding.cc'})
        retired = ('PLA_FLASHKDA_TC01_FLA_RUNTIME', 'PLA_FLASHKDA_FLA_BUILD',
                   'select_fla_runtime', 'FLA_MONOLITHIC', 'FLA_STAGE_ABI', 'binding_common.h')
        files = list(PACKAGE.rglob('*.py')) + [ROOT / 'setup.py']
        files += [p for p in SOURCE.iterdir() if p.suffix in ('.py', '.h', '.cc', '.cu')]
        for path in files:
            text = path.read_text()
            for name in retired:
                self.assertNotIn(name, text, str(path))

    def test_partial_head_adapters_link_optional_prepared_stages(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('requires a C++17 host compiler')
        runtime = (SOURCE / 'runtime.cc').read_text()
        self.assertNotRegex(runtime, r'launch_fla_(?:token_l2|inter_solve_prepared)__fla_h\d+')
        self.assertIn('cuda_check(table.token_l2(', runtime)
        self.assertIn('cuda_check(table.inter_solve_prepared(', runtime)
        condition = runtime.split('const bool prepared_n1 = ', 1)[1].split(';', 1)[0]
        main = r'''
#include "launchers.h"
#include <cassert>
#include <initializer_list>
struct Shape { int n, h; int64_t t; };
bool prepared(const Shape& s, const fla_runtime::Launchers& table) {
  return CONDITION;
}
int main() {
  for (int h : {4, 6, 8, 12, 16}) {
    const auto& table = fla_runtime::launchers_for(h);
    const bool available = (table.heads == 4 || table.heads == 8);
    assert(bool(table.token_l2) == available);
    assert(bool(table.inter_solve_prepared) == available);
    assert(table.token && table.inter);
    for (int t : {4095,4096,4097,8192,16384,32768,32769}) {
      const bool expected = available &&
          (t == 4096 || t == 8192 || t == 16384 || t == 32768);
      assert(prepared({1,h,t}, table) == expected);
      assert(!prepared({2,h,t}, table));
      auto missing = table;
      missing.token_l2 = nullptr;
      assert(!prepared({1,h,t}, missing));
      missing = table;
      missing.inter_solve_prepared = nullptr;
      assert(!prepared({1,h,t}, missing));
    }
  }
}
'''.replace('CONDITION', condition)
        # Compile the real adapters and dispatcher. Stubs supply only symbols
        # for the selected head, so omitted H4/H8 dependencies fail at link time.
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'hggc_runtime.h').write_text(
                '#pragma once\nusing hggcError_t = int; using hggcStream_t = void*;\n')
            common = [compiler, '-std=c++17', '-I', str(temp), '-I', str(SOURCE)]
            objects = {}
            for head in (0, 4, 6, 8):
                preprocessed = subprocess.run(
                    common + ['-E', '-P', f'-DFLA_HEADS={head}', str(SOURCE / 'head_adapter.cc')],
                    capture_output=True, text=True, check=True).stdout
                declarations = re.findall(r'hggcError_t\s+(\w+)\(([^;{}]*)\);', preprocessed)
                declarations = [(name, args) for name, args in declarations
                                if name.endswith(f'__fla_h{head}')]
                self.assertGreater(len(declarations), 8)
                optional = [name for name, _ in declarations
                            if name.startswith(('launch_fla_token_l2__',
                                                'launch_fla_inter_solve_prepared__'))]
                self.assertEqual(len(optional), 2 if head in (4, 8) else 0)
                definitions = '\n'.join(
                    f'hggcError_t {name}({args}) {{ return 0; }}' for name, args in declarations)
                output = temp / f'adapter_{head}.o'
                subprocess.run(common + ['-x', 'c++', '-c', '-', '-o', str(output)],
                               input=preprocessed + '\n' + definitions, text=True, check=True)
                objects[head] = output
            for heads in ((0,), (0, 4), (0, 8), (0, 4, 8), (0, 6)):
                with self.subTest(heads=heads):
                    static = ' '.join(f'X({head})' for head in heads if head)
                    cp = ' '.join(f'X({head})' for head in heads if head in CP_HEADS)
                    (temp / 'runtime_heads.h').write_text(
                        f'#define FLA_EACH_STATIC_HEAD(X) {static}\n'
                        f'#define FLA_EACH_CP_HEAD(X) {cp}\n')
                    dispatcher = temp / 'dispatch.o'
                    subprocess.run(common + ['-c', str(SOURCE / 'dispatch.cc'), '-o', str(dispatcher)],
                                   check=True)
                    probe = temp / 'probe.o'
                    subprocess.run(common + ['-x', 'c++', '-c', '-', '-o', str(probe)],
                                   input=main, text=True, check=True)
                    binary = temp / ('partial_heads_' + '_'.join(map(str, heads)))
                    subprocess.run(common + [str(probe), str(dispatcher)] +
                                   [str(objects[head]) for head in heads] + ['-o', str(binary)],
                                   check=True)
                    subprocess.run([str(binary)], check=True)

    def test_device_target_matrix_covers_long_cp_boundaries(self):
        path = ROOT / 'scripts/prefill/flashkdapro/private/test_tc01_fla_monolithic_cp.py'
        text = path.read_text()
        tree = ast.parse(text)
        selected = [node for node in tree.body
                    if (isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'DEVICE_TARGET_TOKENS'
                        for target in node.targets)) or
                    (isinstance(node, ast.FunctionDef) and node.name in
                     ('device_target_lengths', 'device_trusted_reference'))]
        self.assertEqual(len(selected), 3)
        namespace = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        self.assertEqual(namespace['DEVICE_TARGET_TOKENS'], (4096, 8192, 16384, 32768))
        self.assertIn('for tokens in DEVICE_TARGET_TOKENS:', text)
        self.assertIn('lengths = device_target_lengths(tokens)', text)
        for tokens in namespace['DEVICE_TARGET_TOKENS']:
            lengths = namespace['device_target_lengths'](tokens)
            self.assertEqual(lengths[0], tokens)
            self.assertEqual(lengths[-1], tokens)
            self.assertIn(0, lengths)
            self.assertIn(19, lengths)
            self.assertTrue(all(0 <= length <= tokens for length in lengths))
            if tokens >= 16384:
                self.assertTrue({12287, 12288, 12289}.issubset(lengths))
                reference = namespace['device_trusted_reference']
                self.assertFalse(reference(4, tokens, 12287))
                self.assertTrue(reference(4, tokens, 12288))
                self.assertTrue(reference(4, tokens, 12289))
                self.assertFalse(reference(8, tokens, 12288))

    def test_merged_prepare_uses_shared_probe_abi_and_single_device_guard(self):
        host = (ROOT / 'src/prefill/flashkdapro/host/torch_extension_cute.cc').read_text()
        header = (ROOT / 'src/prefill/flashkdapro/common/launch_api.h').read_text()
        kernel = (ROOT / 'src/prefill/flashkdapro/kernels/kernel1_prepare.cu').read_text()
        self.assertIn('bool flashkda_kr_is_preswizzled(hggcStream_t stream);', header)
        self.assertIn('bool flashkda_kr_is_preswizzled(hggcStream_t stream) {', kernel)
        prepare = host.split('std::vector<torch::Tensor> flashkda_prepare(', 1)[1].split(
            '// flashkda_fwd: full forward', 1)[0]
        self.assertEqual(prepare.count('CUDAGuard device_guard(q.device());'), 1)
        self.assertLess(prepare.index('flashkda_kr_is_preswizzled(stream)'),
                        prepare.index('torch::zeros('))
        code = re.sub(r'//[^\n]*', '', host)
        self.assertNotRegex(code, r'(?<!:)\bKDA_SEG_STRIDE\b')
        # bypass_cp_plan has been removed; the plan dict is built only once by the normal flashkda_cp_plan.
        for key in ('fma_front', 'bt64_align', 'seg_bt'):
            self.assertEqual(host.count(f'd["{key}"] = cp.{key};'), 1)

    def test_fla_sanity_precedes_initialization_and_kernel_launch(self):
        runtime = (SOURCE / 'runtime.cc').read_text().split('void forward(Arguments a) {', 1)[1]
        self.assertLess(runtime.index('pla::sanity::Context sanity'), runtime.index('initialize('))
        self.assertLess(runtime.index('sp::offsets('), runtime.index('trusted_plan('))
        self.assertLess(runtime.index('sp::slots('), runtime.index('at::empty('))
        self.assertIn('a.raw && sanity.values_enabled()', runtime)
        self.assertIn('sp::same_view(a.initial, a.final)', runtime)

    def test_cp_template_declarations_survive_non_tc01_device_pass(self):
        compiler = shutil.which('clang++') or shutil.which('c++')
        if compiler is None:
            self.skipTest('no C++ preprocessor; the other static checks still run')
        command = [compiler, '-E', '-P', '-x', 'c++', '-D__HGGC_ARCH__=150', '-']
        body = subprocess.run(command, input=(SOURCE / 'cp_pretriangular_body.cuh').read_text(),
                              text=True, capture_output=True, check=True).stdout
        self.assertIn('enum class CpPass { Zero, Transition, Correction, LegacyZero, LegacyRepair };', body)
        source = (SOURCE / 'cp_recurrence_output.cu').read_text()
        prefix = 'namespace {' + source.split('namespace {', 1)[1].split(
            'extern "C" hggcError_t configure_fla_cp_pretriangular_output()', 1)[0]
        prefix = subprocess.run(command, input=prefix, text=True, capture_output=True,
                                check=True).stdout
        self.assertIn('void fla_cp_state_prefix(', prefix)
        self.assertIn('__trap();', prefix)
        recurrence = (SOURCE / 'recurrence.cu').read_text()
        constants = 'namespace fla_recurrence {' + recurrence.split(
            'namespace fla_recurrence {', 1)[1].split(
                '#if !defined(__HGGC_ARCH__)', 1)[0] + '}'
        for bv16, expected in ((0, 32), (1, 16)):
            with self.subTest(bv16=bv16):
                output = subprocess.run(
                    command[:-1] + [f'-DFLA_RECURRENCE_BV16={bv16}', '-'],
                    input=constants, text=True, capture_output=True, check=True).stdout
                self.assertIn(f'constexpr int kBv = {expected};', output)

    def test_all_head_symbols_expand_without_runtime_mode_macro(self):
        compiler = shutil.which('clang++') or shutil.which('c++')
        if compiler is None:
            self.skipTest('no C++ preprocessor; the other static checks still run')
        header = (SOURCE / 'symbol_namespace.h').read_text()
        names = re.findall(r'^#define (\w+) FLA_JOIN_HEAD\(\1, FLA_HEADS\)$', header, re.M)
        self.assertGreater(len(names), 30)
        for name in ('fla_inter_solve_cute_legacy', 'fla_inter_solve_cute_wide',
                     'fla_inter_solve_bf16_d128_bt64_bc16_h8_wide_kernel'):
            self.assertIn(name, names)
        source = '#include "heads.h"\n' + '\n'.join(names) + '\n'
        for head in HEADS:
            with self.subTest(head=head):
                result = subprocess.run([compiler, '-E', '-P', '-x', 'c++', '-I', str(SOURCE),
                                         f'-DFLA_HEADS={head}', '-'], input=source, text=True,
                                        capture_output=True, check=True)
                self.assertEqual(result.stdout.splitlines()[-len(names):],
                                 [f'{name}__fla_h{head}' for name in names])


if __name__ == '__main__':
    unittest.main(verbosity=2)
