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

"""Backend selection does not load the tc01 extension; tc02 ignores the tc01 environment variables and rule engine entirely.

The diversion decision (monolithic FLA vs legacy) consumes only host integers and tensor shapes; hot-path D2H is forbidden.
During Graph capture the decision depends only on host/shape features visible at capture time; replay does not re-select.
"""
import os


def select_backend(capability):
    capability = tuple(capability)
    if capability == (8, 9):
        return 'legacy'
    if capability != (8, 0):
        raise RuntimeError(f'FlashKDA does not support capability {capability}')
    value = os.environ.get('PLA_FLASHKDA_TC01_IMPL', '')
    if value in ('', 'fla'):
        return 'fla'
    if value == 'legacy':
        return 'legacy'
    raise ValueError(f'PLA_FLASHKDA_TC01_IMPL={value!r}; only fla or legacy is allowed')


def explicit_tc01_impl():
    """Explicit PLA_FLASHKDA_TC01_IMPL value (already validated by select_backend); when non-empty it bypasses the rule engine."""
    return os.environ.get('PLA_FLASHKDA_TC01_IMPL', '')


def dispatch_tc01_impl(q, cu_seqlens, raw_cu_seqlens, initial_state, final_state,
                       state_pool, state_indices):
    """Frozen-rule diversion: returns 'fla' or 'legacy'; reads host attributes only, no .item()/.cpu()/.tolist().

    Called only on tc01 when PLA_FLASHKDA_TC01_IMPL is unset or empty.
    The calibrated space is stateless and indexed_bf16 (including the equivalent state_pool form);
    lengths come only from trusted host offsets; device-only metadata, padding, empty sequences and
    uncalibrated interfaces are always 'fla', never a silent fallback to legacy.
    """
    from ._dispatch_rules import RULE_HEADS, classify_mode, decide, load_rules
    if q.ndim != 3:
        return 'fla'
    heads = q.shape[1]
    if heads not in RULE_HEADS:
        return 'fla'
    mode = classify_mode(initial_state, final_state, state_pool, state_indices)
    if mode is None:
        return 'fla'
    if (not isinstance(raw_cu_seqlens, (tuple, list)) or len(raw_cu_seqlens) < 2
            or not all(type(value) is int for value in raw_cu_seqlens)):
        return 'fla'
    raw = tuple(raw_cu_seqlens)
    if raw[0] != 0 or raw[-1] != q.shape[0] or any(b < a for a, b in zip(raw, raw[1:])):
        return 'fla'
    batch = len(raw) - 1
    if cu_seqlens is not None and (cu_seqlens.ndim < 1 or cu_seqlens.shape[0] != len(raw)):
        return 'fla'
    if mode == 'indexed_bf16':
        if state_pool is not None:
            if state_indices.shape[0] != batch:
                return 'fla'
        else:
            if (initial_state.shape[0] != batch or final_state.shape[0] != batch
                    or state_indices.shape[0] != batch):
                return 'fla'
    rule = load_rules()
    if rule is None:
        return 'fla'
    return decide(rule, heads, [raw[i + 1] - raw[i] for i in range(batch)], mode,
                  metadata='trusted_int32', padding=0)
