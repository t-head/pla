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

"""tc01 AUTO CP + FMA-front dispatch and correctness regression.

The production AUTO gate is intentionally additive: only varlen H8/N1 shapes
in the measured interval may change from serial to CP.  Exact environment value
"0" restores the prior tc01 AUTO-off behavior; mode=2 remains the generic force
path.  Device correctness compares AUTO against the established serial path for
both output and final state under slow decay so segment carries are non-vacuous.
"""

import os

import torch

from pla.prefill.flashkdapro import _C
from pla.prefill.flashkdapro import flashkda_fwd

D = 128
TOL = 2e-2
ROLLBACK = "PLA_FLASHKDA_CP_TC01_TFU_MODEL"


def _set_env(cp=1, rollback=None, bv=None, mult=None, fma=None,
             threads=None, cpc=None):
    values = {
        "PLA_FLASHKDA_CP": cp,
        ROLLBACK: rollback,
        "PLA_FLASHKDA_CP_BV": bv,
        "PLA_FLASHKDA_CP_MULT": mult,
        "PLA_FLASHKDA_CP_SEG_TILES": None,
        "PLA_FLASHKDA_K1_FMA_SPLIT": fma,
        "PLA_FLASHKDA_K1_THREADS": threads,
        "PLA_FLASHKDA_K1_CPC": cpc,
    }
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    _C.flashkda_cp_plan_cache_clear()


def _plan(T, H=8, raw=(0,)):
    offsets = raw if len(raw) > 1 else (0, T)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    return dict(_C.flashkda_cp_plan(T, H, cu, offsets))


def _assert_plan_boundaries():
    # The combined AUTO path is exact and monotone at its two calibrated seams.
    expected = {
        4095: (False, 0, 32),
        4096: (True, 2, 16),
        16383: (True, 2, 16),
        16384: (True, 16, 128),
        32768: (True, 16, 128),
    }
    _set_env(cp=1)
    for T, (engage, segments, bv) in expected.items():
        plan = _plan(T)
        assert bool(plan["engage"]) is engage, (T, plan)
        assert int(plan["n_seg"]) == segments, (T, plan)
        assert int(plan["bv"]) == bv, (T, plan)
        assert bool(plan["fma_front"]) is engage, (T, plan)

    padded_short = _plan(16384, raw=(0, 8192))
    assert padded_short["engage"] and padded_short["n_seg"] == 2
    assert padded_short["bv"] == 16 and padded_short["fma_front"], padded_short
    padded_below = _plan(16384, raw=(0, 4095))
    assert not padded_below["engage"] and not padded_below["fma_front"]

    # Exact zero is the only rollback spelling.
    _set_env(cp=1, rollback=0)
    assert not _plan(32768)["engage"]
    _set_env(cp=1, rollback="false")
    assert _plan(32768)["engage"]

    for overrides in ({"fma": 0}, {"threads": 256}, {"cpc": 2}):
        _set_env(cp=1, **overrides)
        incompatible = _plan(32768)
        assert not incompatible["engage"]
        assert not incompatible["fma_front"]

    # The additive gate never broadens adjacent head/batch/fixed shapes.
    _set_env(cp=1)
    assert not _plan(32768, H=12)["engage"]
    assert not _plan(8192, H=8, raw=(0, 4096, 8192))["engage"]
    assert not _C.flashkda_cp_plan(32768, 8, None, None)["engage"]

    # Force mode retains caller-selected BV/mult on the same architecture.
    _set_env(cp=2, bv=32, mult=4)
    forced = _plan(8192)
    assert forced["engage"] and forced["bv"] == 32
    assert forced["n_seg"] == 4 and not forced["fma_front"], forced


def _inputs(T, seed, valid_tokens=None):
    torch.manual_seed(seed)
    if valid_tokens is None:
        valid_tokens = T
    q = torch.randn(T, 8, D, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.1
    # Slow-decay gate keeps inter-segment carries live.
    g = (torch.randn(T, 8, D, device="cuda", dtype=torch.float32) * 0.3 - 4.5).bfloat16()
    beta = torch.randn(T, 8, device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(8, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(8, D, device="cuda", dtype=torch.float32)
    state = torch.randn(1, 8, D, D, device="cuda", dtype=torch.bfloat16) * 0.02
    cu = torch.tensor([0, valid_tokens], device="cuda", dtype=torch.int32)
    return q, k, v, g, beta, a_log, dt_bias, state, cu


def _run(x):
    q, k, v, g, beta, a_log, dt_bias, state, cu = x
    out = torch.empty_like(v)
    final = torch.empty_like(state)
    flashkda_fwd(
        q, k, v, g, beta, a_log, dt_bias, out,
        initial_state=state, final_state=final, cu_seqlens=cu,
        raw_cu_seqlens=(0, int(cu[-1].item())),
    )
    return out, final


def _relative(actual, expected):
    return ((actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)).item()


def _assert_correctness():
    cases = (
        (4096, 4096, 2026090901),
        (16384, 16384, 2026090902),
        (16384, 8192, 2026090903),
    )
    for T, valid_tokens, seed in cases:
        x = _inputs(T, seed, valid_tokens)
        _set_env(cp=0)
        serial_out, serial_state = _run(x)
        torch.cuda.synchronize()

        _set_env(cp=1)
        plan = _plan(T, raw=(0, valid_tokens))
        assert plan["engage"] and plan["fma_front"], plan
        auto_out, auto_state = _run(x)
        torch.cuda.synchronize()
        stats_after_miss = dict(_C.flashkda_cp_plan_cache_stats())
        assert stats_after_miss["size"] == 1
        assert stats_after_miss["misses"] == 1
        assert stats_after_miss["h2d_tables"] == 2

        repeat_out, repeat_state = _run(x)
        torch.cuda.synchronize()
        stats_after_hit = dict(_C.flashkda_cp_plan_cache_stats())
        assert stats_after_hit["hits"] == 1
        assert torch.equal(repeat_out, auto_out)
        assert torch.equal(repeat_state, auto_state)

        out_error = _relative(auto_out, serial_out)
        state_error = _relative(auto_state, serial_state)
        assert out_error < TOL, (T, valid_tokens, out_error)
        assert state_error < TOL, (T, valid_tokens, state_error)
        if valid_tokens < T:
            assert torch.count_nonzero(auto_out[valid_tokens:]).item() == 0
        print(
            f"T={T} valid={valid_tokens} nseg={plan['n_seg']} "
            f"bv={plan['bv']} fma={int(plan['fma_front'])} "
            f"out_err={out_error:.6e} state_err={state_error:.6e} OK"
        )
        del x, serial_out, serial_state, auto_out, auto_state
        del repeat_out, repeat_state
        torch.cuda.empty_cache()


def main():
    # This file validates the legacy CH16/BT64 CP planner and numerical path;
    # the monolithic FLA CP path has its own canonical test.  Never let the
    # public dispatcher route these numerical checks into FLA.
    os.environ["PLA_FLASHKDA_TC01_IMPL"] = "legacy"
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    capability = (int(props.major), int(props.minor))
    if capability != (8, 0):
        print(f"SKIP: tc01 test requires compute capability 8.0, got {capability}")
        return
    _assert_plan_boundaries()
    _assert_correctness()
    _set_env(cp=1)
    print("ALL TC01 KDA AUTO CP + FMA-FRONT CASES OK")


if __name__ == "__main__":
    main()
