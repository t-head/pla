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

"""tc01 new/old diversion rule engine: consumes only host integers and tensor shapes; hot-path D2H is forbidden.

This module does not import torch; tensors are consumed only through host attributes such as .shape/.dtype.
Decisions follow the frozen rule table (the RULES dict generated into ``_dispatch_rules_frozen.py`` from
the frozen JSON ``dispatch_rules_frozen_v1.json``): unknown/uncalibrated inputs are always ``fla``
(the new monolithic FLA), never a silent fallback to legacy. The production decision is a conservative intersection: legacy
is taken only when both the graph_ms and api_ms metrics predict legacy.
"""
import threading
from pathlib import Path

RULE_HEADS = (64, 32, 16, 8, 4, 96, 48, 24, 12, 6)
RULE_MODES = ('stateless', 'indexed_bf16')
FEATURE_CONTRACT = 'shape_features_v1'
FROZEN_MODULE = Path(__file__).with_name('_dispatch_rules_frozen.py')

_LOAD_LOCK = threading.Lock()
_RULES = None
_RULES_LOADED = False


def shape_features(heads, lengths):
    """Field-by-field identical to the fitted feature contract shape_features_v1; zero lengths contribute no chunks."""
    if type(heads) is not int or heads <= 0 or not lengths or any(
            type(length) is not int or length < 0 for length in lengths):
        raise ValueError('illegal head/sequence length')
    chunks = [(length + 63) // 64 for length in lengths]
    total, longest = sum(chunks), max(chunks)
    active = sum(length > 0 for length in lengths)
    batch = len(lengths)
    batch_bucket = next((edge for edge in (1, 2, 4, 8, 16, 32, 64) if batch <= edge), 0)
    skew_bucket = next((edge for edge in (1, 2, 4) if active * longest <= edge * total), 8)
    return dict(heads=heads, batch=batch, active=active, tokens=sum(lengths), max_length=max(lengths),
                csum=total, cmax=longest, c16=sum((length + 15) // 16 for length in lengths),
                beff=total / longest if longest else 0., rho=active * longest / total if total else 0.,
                fill=sum(lengths) / (64 * total) if total else 0., head_chunks=heads * total,
                batch_bucket=batch_bucket, skew_bucket=skew_bucket)


def classify_mode(initial_state, final_state, state_pool, state_indices):
    """Returns the calibrated mode name or None (uncalibrated interface); reads host attributes only, never triggers D2H.

    Calibrated space: stateless and indexed_bf16 (including the equivalent state_pool/state_indices forms).
    indexed_fp32, initial-only/final-only, unindexed initial/final, a bare pool
    and other uncalibrated interfaces are always None.
    """
    if initial_state is None and final_state is None and state_pool is None and state_indices is None:
        return 'stateless'
    if state_indices is None:
        return None
    if state_pool is not None:
        if initial_state is not None or final_state is not None:
            return None
        state = state_pool
    else:
        if initial_state is None or final_state is None:
            return None
        if str(initial_state.dtype) != str(final_state.dtype):
            return None
        state = initial_state
    if str(state.dtype) != 'torch.bfloat16' or str(state_indices.dtype) != 'torch.int32':
        return None
    return 'indexed_bf16'


def load_rules():
    """Lazily load the frozen rule table; a missing file means uncalibrated (all fla), corrupted content fails immediately."""
    global _RULES, _RULES_LOADED
    if _RULES_LOADED:
        return _RULES
    with _LOAD_LOCK:
        if _RULES_LOADED:
            return _RULES
        _RULES_LOADED = True
        if not FROZEN_MODULE.is_file():
            _RULES = None
            return _RULES
        from . import _dispatch_rules_frozen as frozen
        value = frozen.RULES
        if value.get('schema') != 'tc01_dispatch_rules_frozen_v1':
            raise ValueError('frozen rule schema mismatch')
        if value.get('feature_contract') != FEATURE_CONTRACT:
            raise ValueError('frozen rule feature contract mismatch')
        if value.get('decision') != 'conservative_intersection_v1':
            raise ValueError('frozen rule decision contract mismatch')
        if sorted(value.get('metrics', {})) != ['api_ms', 'graph_ms']:
            raise ValueError('frozen rules must contain both graph_ms and api_ms metric tables')
        _RULES = value
        return _RULES


def _predict_tables(metric_tables, heads, lengths, mode, metadata, padding):
    """Single-metric table prediction; same structure as the offline simulator, never modifies production routing."""
    if (heads not in RULE_HEADS or mode not in RULE_MODES or metadata != 'trusted_int32' or padding):
        return 'new', 'uncalibrated interface contract'
    feature = shape_features(heads, lengths)
    if feature['active'] != feature['batch']:
        return 'new', 'empty sequence is uncalibrated'
    b1 = metric_tables.get('b1')
    if b1 is not None:
        for table in b1['tables']:
            if table['heads'] != heads:
                continue
            if (len(lengths) == 1 and table['threshold'] is not None
                    and table['min_t'] <= lengths[0] <= table['threshold']):
                return 'legacy', 'fitted single-sequence safe prefix'
    packed = metric_tables.get('packed')
    if packed is not None:
        for table in packed['tables']:
            if table['heads'] != heads or len(lengths) == 1:
                continue
            if table['batch_bucket'] != feature['batch_bucket']:
                continue
            if packed['kind'] == 'R2' and table['skew_bucket'] != feature['skew_bucket']:
                continue
            if any(not bounds[0] <= feature[name] <= bounds[1] for name, bounds in table['domain'].items()):
                continue
            if all(table[name] is None or feature[key] <= table[name] for name, key in
                   (('csum_limit', 'csum'), ('cmax_limit', 'cmax'), ('length_limit', 'max_length'))):
                return 'legacy', 'hit a fitted packed region'
    return 'new', 'outside the fitted legacy-path regions'


def decide(rule, heads, lengths, mode, metadata='trusted_int32', padding=0):
    """Conservative intersection: legacy only when both metrics predict legacy; without a rule table always fla."""
    if rule is None:
        return 'fla'
    verdicts = {metric: _predict_tables(rule['metrics'][metric], heads, lengths, mode,
                                        metadata, padding)
                for metric in ('graph_ms', 'api_ms')}
    backend = 'legacy' if all(item[0] == 'legacy' for item in verdicts.values()) else 'fla'
    return backend


def explain(rule, heads, lengths, mode, metadata='trusted_int32', padding=0):
    """Decision details (for debugging/offline tests); shares the same tables as decide."""
    if rule is None:
        return dict(backend='fla', reason='no frozen rule table', rule_version=None, metrics={})
    verdicts = {metric: dict(backend=item[0], reason=item[1])
                for metric, item in
                ((metric, _predict_tables(rule['metrics'][metric], heads, lengths, mode,
                                          metadata, padding))
                 for metric in ('graph_ms', 'api_ms'))}
    return dict(backend='legacy' if all(item['backend'] == 'legacy' for item in verdicts.values()) else 'fla',
                reason='two-metric conservative intersection', rule_version=rule['version'], metrics=verdicts)
