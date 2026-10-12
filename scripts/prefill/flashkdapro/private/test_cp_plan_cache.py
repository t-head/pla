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

"""OPT-05: value-keyed KDA CP plan-cache and stream-lifetime regression."""

import os

import torch

from pla.prefill.flashkdapro import flashkda_fwd
from pla.prefill.flashkdapro import _C


T = 1025  # one physical padding token; raw offsets end at 1024
H = 2
D = 128
RAW = (0, 513, 1024)


def _set_env(**values) -> None:
    names = (
        "PLA_FLASHKDA_CP",
        "PLA_FLASHKDA_CP_BV",
        "PLA_FLASHKDA_CP_SEG_TILES",
        "PLA_FLASHKDA_CP_MULT",
        "PLA_FLASHKDA_CP_BALANCE",
        "PLA_FLASHKDA_CP_MAX_H",
        "PLA_FLASHKDA_CP_PLAN_CACHE_SIZE",
    )
    for suffix in names:
        value = values.get(suffix.removeprefix("PLA_FLASHKDA_"))
        if value is None:
            os.environ.pop(suffix, None)
        else:
            os.environ[suffix] = str(value)


def _make_inputs():
    torch.manual_seed(20260826)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(H, D, device="cuda", dtype=torch.float32)
    h0 = torch.randn(2, H, D, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v, g, beta, a_log, dt_bias, h0


def _run(inputs, raw, *, stream=None):
    q, k, v, g, beta, a_log, dt_bias, h0 = inputs
    cu = torch.tensor(raw, device="cuda", dtype=torch.int32)
    out = torch.full_like(v, 17)
    final = torch.empty_like(h0)

    def call():
        flashkda_fwd(
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            out,
            initial_state=h0,
            final_state=final,
            cu_seqlens=cu,
            raw_cu_seqlens=tuple(raw),
        )

    if stream is None:
        call()
    else:
        stream.wait_stream(torch.cuda.current_stream())
        cu.record_stream(stream)
        out.record_stream(stream)
        final.record_stream(stream)
        with torch.cuda.stream(stream):
            call()
    return out, final, cu


def _assert_stats(**expected) -> None:
    stats = dict(_C.flashkda_cp_plan_cache_stats())
    for name, value in expected.items():
        assert stats[name] == value, (name, stats, expected)


def test_raw_offsets_and_hit(inputs) -> None:
    _set_env(CP=2, CP_BV=32, CP_SEG_TILES=4, CP_MULT=4,
             CP_PLAN_CACHE_SIZE=4)
    cu = torch.tensor(RAW, device="cuda", dtype=torch.int32)
    fallback = dict(_C.flashkda_cp_plan(T, H, cu))
    trusted = dict(_C.flashkda_cp_plan(T, H, cu, RAW))
    for field in ("engage", "n_seg", "bv", "max_seg_tiles",
                  "seg_tbl", "seq_seg"):
        assert fallback[field] == trusted[field], field

    _C.flashkda_cp_plan_cache_clear()
    out0, final0, _ = _run(inputs, RAW)
    _assert_stats(size=1, hits=0, misses=1, d2h_reads=0, h2d_tables=2)

    # A newly allocated device cu_seqlens with the same trusted Host values is
    # deliberately a hit: tensor identity is not part of the key.
    out1, final1, _ = _run(inputs, RAW)
    torch.cuda.synchronize()
    _assert_stats(size=1, hits=1, misses=1, d2h_reads=0, h2d_tables=2)
    assert torch.equal(out0, out1)
    assert torch.equal(final0, final1)
    assert torch.count_nonzero(out1[RAW[-1]:]).item() == 0


def test_fallback_and_negative_cache(inputs) -> None:
    _set_env(CP=2, CP_BV=32, CP_SEG_TILES=4, CP_MULT=4,
             CP_PLAN_CACHE_SIZE=4)
    _C.flashkda_cp_plan_cache_clear()
    q, k, v, g, beta, a_log, dt_bias, h0 = inputs
    cu = torch.tensor(RAW, device="cuda", dtype=torch.int32)

    def fallback_call():
        out = torch.empty_like(v)
        final = torch.empty_like(h0)
        flashkda_fwd(
            q, k, v, g, beta, a_log, dt_bias, out,
            initial_state=h0, final_state=final, cu_seqlens=cu)
        return out, final

    out0, final0 = fallback_call()
    out1, final1 = fallback_call()
    torch.cuda.synchronize()
    # Without trusted Host values a value-keyed lookup still needs one D2H per
    # call. The hit saves planning/tables, but cannot remove that synchronization.
    _assert_stats(size=1, hits=1, misses=1, d2h_reads=2, h2d_tables=2)
    assert torch.equal(out0, out1)
    assert torch.equal(final0, final1)

    # Candidate shapes rejected by the detailed planner are cached as well.
    _set_env(CP=2, CP_BV=32, CP_SEG_TILES=4, CP_MULT=1,
             CP_PLAN_CACHE_SIZE=4)
    _C.flashkda_cp_plan_cache_clear()
    _run(inputs, RAW)
    _run(inputs, RAW)
    torch.cuda.synchronize()
    _assert_stats(size=1, hits=1, misses=1, d2h_reads=0, h2d_tables=0)


def test_env_key_and_lru(inputs) -> None:
    _set_env(CP=2, CP_BV=32, CP_SEG_TILES=4, CP_MULT=4,
             CP_PLAN_CACHE_SIZE=4)
    _C.flashkda_cp_plan_cache_clear()
    _run(inputs, RAW)
    os.environ["PLA_FLASHKDA_CP_BV"] = "16"
    _run(inputs, RAW)
    os.environ["PLA_FLASHKDA_CP_BV"] = "32"
    _run(inputs, RAW)
    torch.cuda.synchronize()
    _assert_stats(size=2, hits=1, misses=2, h2d_tables=4)

    # Use negative plans so the LRU policy itself can be exercised cheaply.
    _set_env(CP=2, CP_BV=32, CP_SEG_TILES=4, CP_MULT=1,
             CP_PLAN_CACHE_SIZE=4)
    _C.flashkda_cp_plan_cache_clear()
    raws = [(0, split, 1024) for split in (500, 501, 502, 503, 504)]
    for raw in raws:
        _run(inputs, raw)
    torch.cuda.synchronize()
    _assert_stats(size=4, hits=0, misses=5, evictions=1, h2d_tables=0)
    _run(inputs, raws[0])
    torch.cuda.synchronize()
    _assert_stats(size=4, hits=0, misses=6, evictions=2)


def test_cross_stream_eviction(inputs) -> None:
    _set_env(CP=2, CP_BV=32, CP_SEG_TILES=4, CP_MULT=4,
             CP_PLAN_CACHE_SIZE=4)
    _C.flashkda_cp_plan_cache_clear()
    torch.cuda.synchronize()
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    out0, final0, _ = _run(inputs, RAW, stream=stream0)
    out1, final1, _ = _run(inputs, RAW, stream=stream1)
    _assert_stats(size=1, hits=1, misses=1, h2d_tables=2)

    # Drop the last cache reference while both streams may still consume the
    # descriptors, then pressure the allocator. recordStream must defer reuse.
    _C.flashkda_cp_plan_cache_clear()
    scratch = [torch.empty(4096, device="cuda", dtype=torch.int32)
               for _ in range(32)]
    del scratch
    stream0.synchronize()
    stream1.synchronize()
    assert torch.equal(out0, out1)
    assert torch.equal(final0, final1)


def test_disabled_cache(inputs) -> None:
    _set_env(CP=2, CP_BV=32, CP_SEG_TILES=4, CP_MULT=4,
             CP_PLAN_CACHE_SIZE=0)
    _C.flashkda_cp_plan_cache_clear()
    _run(inputs, RAW)
    _run(inputs, RAW)
    torch.cuda.synchronize()
    _assert_stats(capacity=0, size=0, hits=0, misses=2,
                  d2h_reads=0, h2d_tables=4)


def main() -> None:
    inputs = _make_inputs()
    test_raw_offsets_and_hit(inputs)
    test_fallback_and_negative_cache(inputs)
    test_env_key_and_lru(inputs)
    test_cross_stream_eviction(inputs)
    test_disabled_cache(inputs)
    _set_env()
    _C.flashkda_cp_plan_cache_clear()
    print("ALL KDA CP PLAN CACHE CASES OK")


if __name__ == "__main__":
    main()
