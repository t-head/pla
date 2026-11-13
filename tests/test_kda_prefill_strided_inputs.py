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

"""FlashKDA prefill strided-input contract.

Every input and state of ``flashkda_fwd`` may be an arbitrary strided view. The
result must be bit-identical to the dense call on the same values, for both tc01
backends (monolithic FLA and legacy) and the tc02 legacy backend, in eager mode
and under Graph capture.

Run: python -m pytest tests/test_kda_prefill_strided_inputs.py -q
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/PPU")

D = 128


def _backends():
    if not torch.cuda.is_available():
        return ["legacy"]
    return ["fla", "legacy"] if torch.cuda.get_device_capability() == (8, 0) else ["legacy"]


@pytest.fixture(params=_backends())
def backend(request, monkeypatch):
    if torch.cuda.get_device_capability() == (8, 0):
        monkeypatch.setenv("PLA_FLASHKDA_TC01_IMPL", request.param)
    return request.param


def _place(x, layout):
    """Copy ``x`` into a non-dense view of the requested layout."""
    if layout == "dense":
        return x.clone()
    if layout == "permute01":
        y = torch.empty((x.shape[1], x.shape[0]) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
        y = y.transpose(0, 1)
    elif layout == "pad_last":
        y = torch.empty(tuple(x.shape[:-1]) + (x.shape[-1] + 24,), dtype=x.dtype, device=x.device)
        y = y[..., :x.shape[-1]]
    elif layout == "pad_dim1":
        y = torch.empty((x.shape[0], x.shape[1] + 1) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
        y = y[:, :x.shape[1]]
    elif layout == "transpose_last":
        y = torch.empty(tuple(x.shape[:-2]) + (x.shape[-1], x.shape[-2]), dtype=x.dtype, device=x.device)
        y = y.transpose(-1, -2)
    elif layout == "step2":
        y = torch.empty((2 * x.shape[0],) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)[::2]
    elif layout == "misaligned":
        # Contiguous, but the base pointer is off the 16-byte vector alignment.
        flat = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
        y = flat[1:].view(x.shape)
    else:
        raise ValueError(layout)
    y.copy_(x)
    assert torch.equal(y, x)
    return y


def _inputs(lengths, heads, seed):
    torch.manual_seed(seed)
    total = sum(lengths)
    dev = "cuda"
    q = (torch.randn(total, heads, D, device=dev) * 0.5).to(torch.bfloat16)
    k = (torch.randn(total, heads, D, device=dev) * 0.5).to(torch.bfloat16)
    v = (torch.randn(total, heads, D, device=dev) * 0.5).to(torch.bfloat16)
    g = torch.randn(total, heads, D, device=dev).to(torch.bfloat16)
    beta = torch.randn(total, heads, device=dev).to(torch.bfloat16)
    a_log = torch.rand(heads, device=dev) * 0.5
    dt_bias = torch.randn(heads, D, device=dev) * 0.1
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, g=g, beta=beta, A_log=a_log, dt_bias=dt_bias), cu, tuple(offsets)


INPUT_LAYOUTS = dict(q="permute01", k="pad_last", v="pad_dim1", g="misaligned",
                     beta="transpose_last", A_log="step2", dt_bias="transpose_last")


def _strided(inputs, layouts=INPUT_LAYOUTS):
    return {name: _place(x, layouts.get(name, "dense")) for name, x in inputs.items()}


def _call(inputs, out, **kwargs):
    from pla.prefill.flashkdapro import flashkda_fwd
    flashkda_fwd(inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
                 inputs["A_log"], inputs["dt_bias"], out, **kwargs)


def _snapshot(tensors):
    return {name: x.clone() for name, x in tensors.items()}


def _assert_unchanged(before, after):
    for name, x in before.items():
        assert torch.equal(x, after[name]), f"{name} was modified"


CASES = [
    # (lengths, heads, raw offsets): varlen with an empty and a ragged-tail sequence,
    # plus the N1 prepared-token shape and its device-offset CP bucket.
    ((37, 64, 0, 20), 4, True),
    ((4096,), 8, True),
    ((4096,), 8, False),
]


@pytest.mark.parametrize("lengths,heads,raw", CASES)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_strided_inputs_and_states_match_dense(backend, lengths, heads, raw, state_dtype):
    inputs, cu, offsets = _inputs(lengths, heads, seed=11)
    n = len(lengths)
    initial = (torch.randn(n, heads, D, D, device="cuda") * 0.05).to(state_dtype)
    meta = dict(cu_seqlens=cu, raw_cu_seqlens=offsets if raw else None)

    out_ref = torch.empty_like(inputs["q"])
    final_ref = torch.empty_like(initial)
    _call(inputs, out_ref, initial_state=initial, final_state=final_ref, **meta)
    out_again = torch.empty_like(out_ref)
    final_again = torch.empty_like(final_ref)
    _call(inputs, out_again, initial_state=initial, final_state=final_again, **meta)
    assert torch.equal(out_again, out_ref) and torch.equal(final_again, final_ref), \
        "dense reference is not deterministic"

    strided = _strided(inputs)
    before = _snapshot(strided)
    initial_s = _place(initial, "transpose_last")
    initial_before = initial_s.clone()
    final_s = _place(torch.full_like(initial, -3.0), "pad_last")
    out = torch.empty_like(out_ref)
    _call(strided, out, initial_state=initial_s, final_state=final_s, **meta)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(final_s, final_ref)
    _assert_unchanged(before, strided)
    assert torch.equal(initial_s, initial_before)

    # Each input alone, to make sure no single tensor relies on another being staged.
    for name, layout in INPUT_LAYOUTS.items():
        single = dict(inputs)
        single[name] = _place(inputs[name], layout)
        out = torch.empty_like(out_ref)
        final = torch.empty_like(final_ref)
        _call(single, out, initial_state=initial, final_state=final, **meta)
        assert torch.equal(out, out_ref), name
        assert torch.equal(final, final_ref), name


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_strided_state_pool_matches_dense(backend, state_dtype):
    lengths, heads = (37, 64, 0, 20), 4
    inputs, cu, offsets = _inputs(lengths, heads, seed=12)
    slots = 7
    pool0 = (torch.randn(slots, heads, D, D, device="cuda") * 0.05).to(state_dtype)
    indices = torch.tensor([5, 0, 3, 2], dtype=torch.int32, device="cuda")
    meta = dict(cu_seqlens=cu, raw_cu_seqlens=offsets, state_indices=indices)

    pool_ref = pool0.clone()
    out_ref = torch.empty_like(inputs["q"])
    _call(inputs, out_ref, state_pool=pool_ref, **meta)

    for layout in ("pad_dim1", "transpose_last", "step2"):
        pool = _place(pool0, layout)
        out = torch.empty_like(out_ref)
        _call(_strided(inputs), out, state_pool=pool, **meta)
        torch.cuda.synchronize()
        assert torch.equal(out, out_ref), layout
        assert torch.equal(pool, pool_ref), layout

    # Separate indexed initial/final buffers, one dense and one strided.
    final_ref = torch.full_like(pool0, -3.0)
    _call(inputs, out_ref, initial_state=pool0, final_state=final_ref, **meta)
    final = _place(torch.full_like(pool0, -3.0), "pad_last")
    out = torch.empty_like(out_ref)
    _call(inputs, out, initial_state=pool0, final_state=final, **meta)
    assert torch.equal(out, out_ref)
    assert torch.equal(final, final_ref)
    initial = _place(pool0, "pad_dim1")
    final = torch.full_like(pool0, -3.0)
    _call(inputs, out, initial_state=initial, final_state=final, **meta)
    assert torch.equal(out, out_ref)
    assert torch.equal(final, final_ref)


def test_strided_inputs_under_graph_capture(backend):
    lengths, heads = (37, 64, 0, 20), 4
    inputs, cu, offsets = _inputs(lengths, heads, seed=13)
    slots = 5
    pool0 = (torch.randn(slots, heads, D, D, device="cuda") * 0.05).to(torch.bfloat16)
    indices = torch.tensor([4, 1, 0, 2], dtype=torch.int32, device="cuda")
    meta = dict(cu_seqlens=cu, raw_cu_seqlens=offsets, state_indices=indices)

    pool_ref = pool0.clone()
    out_ref = torch.empty_like(inputs["q"])
    _call(inputs, out_ref, state_pool=pool_ref, **meta)

    strided = _strided(inputs)
    pool = _place(pool0, "pad_dim1")
    out = torch.empty_like(out_ref)
    # Warm up on the capture stream: trusted raw plans are cached per stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(strided, out, state_pool=pool, **meta)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _call(strided, out, state_pool=pool, **meta)
    for _ in range(2):
        pool.copy_(pool0)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, out_ref)
        assert torch.equal(pool, pool_ref)


def test_final_state_sharing_out_storage_is_rejected(backend):
    lengths, heads = (32,), 4
    inputs, cu, offsets = _inputs(lengths, heads, seed=14)
    buffer = torch.empty(2 * 32 * heads * D + 2 * heads * D * D + 64, dtype=torch.bfloat16, device="cuda")
    out = buffer[:32 * heads * D].view(32, heads, D)
    final = buffer[-(heads * D * (D + 8)):].view(1, heads, D, D + 8)[..., :D]
    with pytest.raises(RuntimeError, match="final_state must not share storage with out"):
        _call(inputs, out, final_state=final, cu_seqlens=cu, raw_cu_seqlens=offsets)
