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

"""Monolithic AOT entry; the hot path creates no scratch and holds no global compute lock."""
import importlib
import threading

import torch

from ._contract import ABI, CP_ABI, CP_HEADS, HEADS

_ENTRY = None
_LOAD_LOCK = threading.Lock()


def forward(q, k, v, g, beta, a_log, dt_bias, out, scale, lower_bound,
            initial_state, final_state, cu_seqlens, state_pool, state_indices, raw_cu_seqlens):
    global _ENTRY
    entry = _ENTRY
    if entry is None:
        with _LOAD_LOCK:
            if _ENTRY is None:
                with torch.cuda.device(q.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError('the tc01 FLA monolithic extension must be loaded and warmed up outside Graph capture')
                    module = importlib.import_module(f'{__package__}._C')
                    if (getattr(module, 'runtime_abi', None) != 'tc01_fla_monolithic_runtime_v2' or
                            getattr(module, 'kernel_abi', None) != ABI or
                            getattr(module, 'cp_abi', None) != CP_ABI or
                            tuple(getattr(module, 'cp_heads', ())) != CP_HEADS or
                            tuple(getattr(module, 'supported_heads', ())) != HEADS or
                            getattr(module, 'generic_head_max', None) != 65535 or
                            getattr(module, 'manifest_schema', None) != 'tc01_fla_monolithic_build_v2'):
                        raise RuntimeError('tc01 FLA monolithic extension ABI/head/CP mismatch, or an incomplete probe build')
                    _ENTRY = module.forward
            entry = _ENTRY
    return entry(q, k, v, g, beta, a_log, dt_bias, out, scale, lower_bound,
                 initial_state, final_state, cu_seqlens, state_pool, state_indices, raw_cu_seqlens)
