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

"""Grouped object build for the monolithic extension; the pure layout/manifest functions can be audited on CPU."""
import hashlib
import json
import os
from pathlib import Path
import re

HEADS = (0, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
CP_HEADS = (4, 6, 8, 12)
RUNTIME_ABI = 'tc01_fla_monolithic_runtime_v2'
KERNEL_ABI = 'tc01_fla_packed_v3_cpstate_v1'
CP_ABI = 'tc01_fla_cp_state_passing_v2'
HOST_FLAGS = ['-O3', '-std=c++20', '-DNDEBUG', '-Wno-deprecated-declarations']
STAGES = ('l2norm', 'metadata', 'gate', 'inter', 'recompute', 'recurrence',
          'cp_recurrence_output', 'token_output')


def build_heads(value=None):
    if value is None:
        return HEADS
    try:
        result = tuple(sorted(int(x) for x in value.split(',')))
    except ValueError as error:
        raise ValueError('PLA_FLA_BUILD_HEADS must be comma-separated integers') from error
    if not result or 0 not in result or len(set(result)) != len(result) or not set(result) <= set(HEADS):
        raise ValueError('the head subset must contain H0, with no duplicates and nothing outside the static list')
    return result


def object_groups(flags, heads=HEADS):
    if set(flags) != set(STAGES):
        raise ValueError('stage compile flag list is incomplete')
    groups = []
    for stage in STAGES:
        sources = ['token.cu', 'output.cu'] if stage == 'token_output' else [stage + '.cu']
        if stage == 'cp_recurrence_output':
            sources.append('cp_recurrence_bv16.cu')
        if stage in ('l2norm', 'metadata'):
            stage_heads = (0,)
        elif stage == 'cp_recurrence_output':
            stage_heads = tuple(h for h in heads if h in CP_HEADS)
        else:
            stage_heads = heads
        for h in stage_heads:
            # cuda_cflags is the stable manifest schema consumed by the CI
            # compatibility rewriter; the native backend routes it to hgcc.
            groups.append(dict(stage=stage, heads=h, sources=sources,
                               flags={key: list(flags[stage][old]) + [f'-DFLA_HEADS={h}']
                                      for key, old in (('cxx', 'cflags'), ('hgcc', 'cuda_cflags'))}))
    for h in heads:
        groups.append(dict(stage='adapter', heads=h, sources=['head_adapter.cc'],
                           flags={'cxx': HOST_FLAGS + [f'-DFLA_HEADS={h}'], 'hgcc': []}))
    return groups


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_changed(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def compile_objects(command, ext):
    """Compile each flag group through the shared PPU-native build backend."""
    root = Path(ext.fla_source).resolve()
    output = Path(command.build_temp).resolve()
    heads = ext.fla_heads
    write_changed(output / 'runtime_heads.h',
                  '#pragma once\n#define FLA_EACH_STATIC_HEAD(X) ' +
                  ' '.join(f'X({h})' for h in heads if h) +
                  '\n#define FLA_EACH_CP_HEAD(X) ' +
                  ' '.join(f'X({h})' for h in heads if h in CP_HEADS) + '\n')
    ext.include_dirs = list(ext.include_dirs) + [str(output)]
    dependencies = ([str(p) for pattern in ('*.h', '*.cuh')
                     for p in sorted(root.glob(pattern))]
                    + [str(root / 'recurrence.cu'), str(root / 'cp_recurrence_output.cu'),
                       str(root / 'build_flags.json')])
    records, objects = [], []
    flags = json.loads((root / 'build_flags.json').read_text())
    for group in object_groups(flags, heads):
        directory = output / group['stage'] / ('h' + str(group['heads']))
        directory.mkdir(parents=True, exist_ok=True)
        args = {key: group['flags'][key] for key in ('cxx', 'hgcc')}
        sources = [str(root / name) for name in group['sources']]
        built = command.compile_sources(
            sources, directory, ext, extra_postargs=args, depends=dependencies)
        if len(built) != len(sources) or any(not Path(p).is_file() for p in built):
            raise RuntimeError('FLA object group missing: ' + str(group))
        objects.extend(built)
        ninjas = {str(p.relative_to(output)): dict(sha256=sha(p), text=p.read_text())
                  for p in directory.rglob('build.ninja')}
        if not ninjas:
            raise RuntimeError('FLA object group has no actual Ninja evidence')
        records.append(dict(group, actual_postargs=args, ninjas=ninjas,
                            sources_sha256={name: sha(root / name) for name in group['sources']},
                            objects={str(Path(p).relative_to(output)): sha(p) for p in built}))
    if len(set(objects)) != len(objects):
        raise RuntimeError('different FLA object groups overwrote the same path')
    ext.extra_objects = list(ext.extra_objects or []) + objects
    ext.depends = list(ext.depends or []) + dependencies + objects + [str(output / 'runtime_heads.h')]
    manifest = dict(schema='tc01_fla_monolithic_build_v2', runtime_abi=RUNTIME_ABI,
                    kernel_abi=KERNEL_ABI, cp_abi=CP_ABI,
                    heads=list(heads), cp_heads=[h for h in heads if h in CP_HEADS],
                    full_build=heads == HEADS,
                    groups=records, flags_sha256=sha(root / 'build_flags.json'),
                    source_sha256={p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file()},
                    link_args=ext.extra_link_args)
    write_changed(output / 'object_manifest.json', json.dumps(manifest, indent=2) + '\n')
    return output, manifest


def finish_manifest(command, ext, output, manifest):
    binary = Path(command.get_ext_fullpath(ext.name)).resolve()
    manifest['binary'] = dict(path=str(binary), sha256=sha(binary))
    manifest['host_sources'] = {str(p): sha(p) for p in ext.sources}
    ninja = output / 'build.ninja'
    text = ninja.read_text()
    host_objects = re.findall(r'^build (.+): cxx_compile .+$', text, re.M)
    if len(host_objects) != len(ext.sources) or len(set(host_objects)) != len(host_objects):
        raise RuntimeError('monolithic runtime host object list is wrong')
    manifest['host_objects'] = {name: sha(name) for name in host_objects}
    manifest['host_ninja'] = dict(sha256=sha(ninja), text=text)
    generated_heads = Path(command.build_temp).resolve() / 'runtime_heads.h'
    manifest['generated_heads_sha256'] = sha(generated_heads)
    if not manifest.get('link_command'):
        raise RuntimeError('build manifest is missing the actual link command')
    payload = json.dumps(manifest, indent=2) + '\n'
    write_changed(output / 'build_manifest.json', payload)
    # The sidecar inside build_lib is installed with the wheel; nothing is written into the source package directory.
    write_changed(binary.parent / '_build_manifest.json', payload)
