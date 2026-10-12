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
