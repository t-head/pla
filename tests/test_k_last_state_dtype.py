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

"""k-last native state dtype regression: FP32/BF16 storage, FP32 recurrence.

Run: python -m pytest tests/test_k_last_state_dtype.py -q
Requires a built pla.decode; skipped without a device, and no other PLA kernel is used as the numerical reference.
"""

import pytest

from test_sanity_check import inputs, launch, native
import torch
import torch.nn.functional as F

from pla.decode import (
    fused_sigmoid_gating_delta_rule_forward_k_last as forward,
    fused_sigmoid_gating_delta_rule_forward_k_last_packed as packed_forward,
)
from pla.decode.private import (
    fused_sigmoid_gating_delta_rule_forward_k_last_tunable,
    fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/PPU")


def make_case(state_dtype, activation_dtype, is_kda, mode, *,
              H=4, HV=8, B=3, T=None, seed=2026, padded=True):
    torch.manual_seed(seed)
    device = "cuda"
    D = 128
    T = 1 if mode == "decode" else (4 if T is None else T)
    lengths = [T] * B
    if mode == "ragged":
        assert B == 3
        lengths = [T, T // 2, 0]
    nt = sum(lengths)

    def rand(*shape, dtype=activation_dtype):
        return torch.randn(*shape, device=device, dtype=dtype)

    # Page stride with gaps; slot 0 is valid, slot -1 is SGLang padding.
    slots = max(7, B + 2)
    backing = rand(slots, HV * D * D + 256, dtype=state_dtype) * 0.1
    state = backing[:, : HV * D * D].view(slots, HV, D, D)
    state_ids = [3, 0, -1] if padded and B == 3 else list(range(B))
    cache_ids = [2, 0, 1] if padded and B == 3 else list(reversed(range(B)))
    cache = None if mode == "decode" else torch.full(
        (max(4, B + 1), T + 3, HV, D, D), -7.0, dtype=state_dtype, device=device)
    parents = None
    if mode == "tree":
        parents = torch.tensor([[max(0, t - 2) for t in range(T)]] * B,
                               dtype=torch.int32, device=device)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                      device=device, dtype=torch.int32)
    args = dict(
        A_log=rand(HV, dtype=torch.float32),
        a=rand(1, nt, HV, D) if is_kda else rand(1, nt, HV),
        dt_bias=rand(HV, D, dtype=torch.float32 if T > 1 else activation_dtype)
        if is_kda else rand(HV),
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=rand(1, nt, H, D), k=rand(1, nt, H, D), v=rand(1, nt, HV, D),
        b=rand(1, nt, HV),
        initial_state_source=state,
        initial_state_indices=torch.tensor(state_ids, device=device, dtype=torch.int32),
        scale=D ** -0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        is_kda=is_kda,
        disable_state_update=mode not in ("decode", "writeback"),
        intermediate_states_buffer=None if mode == "decode" else cache,
        intermediate_state_indices=None if mode == "decode" else torch.tensor(cache_ids, device=device, dtype=torch.int32),
        cache_steps_or_num_accept=None if mode == "decode" else T,
        retrieve_parent_token=parents,
    )
    return args, backing, lengths, state_ids, cache_ids


def reference(args, lengths, state_ids, cache_ids):
    """Recur in the [V,K] layout; the rounded BF16 snapshot is read only when a branch reloads."""
    q, k, v = (args[n][0].float() for n in ("q", "k", "v"))
    if args["use_qk_l2norm_in_kernel"]:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q *= args["scale"]
    HV = v.shape[1]
    q, k = (x.repeat_interleave(HV // x.shape[1], dim=1) for x in (q, k))
    state = args["initial_state_source"].clone()
    cache = args["intermediate_states_buffer"]
    cache = None if cache is None else cache.clone()
    gate_x = args["a"][0].float() + args["dt_bias"].float()
    exp_A = args["A_log"].float().exp()
    if args["is_kda"]:
        exp_A = exp_A[:, None]
    if args.get("lower_bound") is not None:
        decay = (args["lower_bound"] * torch.sigmoid(exp_A * gate_x)).exp()
    else:
        decay = (-exp_A * F.softplus(gate_x)).exp()
    if not args["is_kda"]:
        decay = decay[..., None]
    beta = args["b"][0].float().sigmoid()
    out = torch.zeros_like(v)
    parents = args["retrieve_parent_token"]
    bos = 0
    for n, length in enumerate(lengths):
        slot = state_ids[n]
        if slot < 0 or (not args.get("is_sglang", True) and slot == 0) or length == 0:
            bos += length
            continue
        h = state[slot].float()
        for t in range(length):
            if parents is not None and t > 0:
                parent = int(parents[n, t])
                if parent >= 0 and parent != t - 1:
                    h = cache[cache_ids[n], parent].float()
            i = bos + t
            h = h * decay[i, :, None, :]
            delta = (v[i] - (h * k[i, :, None, :]).sum(-1)) * beta[i, :, None]
            h = h + delta[..., None] * k[i, :, None, :]
            out[i] = (h * q[i, :, None, :]).sum(-1)
            # A separate cache is the SGLang contract; vLLM only writes snapshots into the main pool's 2-D slot table.
            if cache is not None and args.get("is_sglang", True):
                cache[cache_ids[n], t] = h
        if not args["disable_state_update"]:
            state[slot] = h
        bos += length
    return out[None].to(args["v"].dtype), state, cache


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_kda", [False, True])
@pytest.mark.parametrize("mode", ["decode", "verify", "writeback", "tree", "ragged", "graph", "tunable"])
@pytest.mark.parametrize("H,HV", [(4, 8), (6, 6), (6, 18)])
def test_k_last_state_dtype(state_dtype, activation_dtype, is_kda, mode, H, HV):
    args, backing, lengths, state_ids, cache_ids = make_case(
        state_dtype, activation_dtype, is_kda, mode, H=H, HV=HV)
    expected_out, expected_state, expected_cache = reference(args, lengths, state_ids, cache_ids)
    backing_before = backing.clone()
    cache = args["intermediate_states_buffer"]
    state = args["initial_state_source"]
    state_ptr, cache_ptr = state.data_ptr(), cache.data_ptr() if cache is not None else None
    if mode == "tunable":
        out = fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
            **args, **tunable_config())
    elif mode == "graph":
        forward(**args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward(**args)
        for _ in range(3):
            cache.fill_(-7.0)
            graph.replay()
    else:
        out = forward(**args)
    torch.cuda.synchronize()
    assert state.data_ptr() == state_ptr
    assert state.dtype == state_dtype
    # BF16 last-bit rounding allows about 1 ULP, which still catches layout/dispatch/async pipeline errors.
    torch.testing.assert_close(out, expected_out, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(state, expected_state, atol=2e-3, rtol=8e-3)
    torch.testing.assert_close(backing[:, -256:], backing_before[:, -256:], atol=0, rtol=0)
    if args["disable_state_update"]:
        torch.testing.assert_close(backing, backing_before, atol=0, rtol=0)
    else:
        torch.testing.assert_close(backing[[1, 2, 4, 5, 6]], backing_before[[1, 2, 4, 5, 6]], atol=0, rtol=0)
    if cache is not None:
        assert cache.dtype == state_dtype and cache.data_ptr() == cache_ptr
        torch.testing.assert_close(cache, expected_cache, atol=2e-3, rtol=8e-3)
        untouched = expected_cache == -7.0
        assert torch.equal(cache[untouched], expected_cache[untouched])


@pytest.mark.parametrize("state_dtype,cache_dtype,message", [
    (torch.float16, torch.float16, "state dtype must be"),
    (torch.float32, torch.bfloat16, "must have the same dtype"),
    (torch.bfloat16, torch.float32, "must have the same dtype"),
])
@pytest.mark.parametrize("tunable", [False, True])
def test_k_last_rejects_state_dtype(state_dtype, cache_dtype, message, tunable):
    args, _, _, _, _ = make_case(state_dtype, torch.bfloat16, False, "verify")
    args["intermediate_states_buffer"] = args["intermediate_states_buffer"].to(cache_dtype)
    with pytest.raises(RuntimeError, match=message):
        if tunable:
            fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
                **args, bv=64, wn=8, sn=4, hv_per_cta=1)
        else:
            forward(**args)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("H,HV", [(4, 8), (6, 6), (12, 24)])
def test_k_last_kda_lower_bound(state_dtype, H, HV):
    args, _, lengths, state_ids, cache_ids = make_case(
        state_dtype, torch.bfloat16, True, "tree", H=H, HV=HV)
    args["lower_bound"] = -5.0
    expected_out, _, expected_cache = reference(args, lengths, state_ids, cache_ids)
    out = forward(**args)
    torch.testing.assert_close(out, expected_out, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(args["intermediate_states_buffer"], expected_cache, atol=2e-3, rtol=8e-3)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("H,HV", [(4, 8), (6, 6), (12, 24)])
def test_k_last_vllm_slot_table(state_dtype, H, HV):
    args, backing, lengths, _, _ = make_case(
        state_dtype, torch.bfloat16, False, "verify", H=H, HV=HV)
    # Keep only one request, so different requests do not conflict when writing state slots.
    for key in ("a", "b", "q", "k", "v"):
        args[key] = args[key][:, :4].contiguous()
    args["cu_seqlens"] = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    args["initial_state_indices"] = torch.tensor([3], device="cuda", dtype=torch.int32)
    expected_out, _, expected_cache = reference(args, [4], [3], [2])
    args["initial_state_indices"] = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.int32)
    args["cache_steps_or_num_accept"] = torch.tensor([3], device="cuda", dtype=torch.int32)
    args["intermediate_states_buffer"] = None
    args["intermediate_state_indices"] = None
    args["is_sglang"] = False
    before = backing.clone()
    out = forward(**args)
    torch.testing.assert_close(out, expected_out, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(args["initial_state_source"][1:5], expected_cache[2, :4], atol=2e-3, rtol=8e-3)
    torch.testing.assert_close(backing[:, -256:], before[:, -256:], atol=0, rtol=0)
    torch.testing.assert_close(backing[[0, 5, 6]], before[[0, 5, 6]], atol=0, rtol=0)


def tunable_config():
    # Keep the existing pipeline-safe config of tc01 MTP.
    tc01 = torch.cuda.get_device_capability() == (8, 0)
    return dict(bv=16 if tc01 else 64, wn=4 if tc01 else 8, sn=4, hv_per_cta=1)


def packed_args(args):
    nt = args["q"].shape[1]
    return dict(
        mixed_qkv=torch.cat([args[k].reshape(nt, -1) for k in ("q", "k", "v")], -1),
        a=args["a"].reshape(nt, -1), b=args["b"].reshape(nt, -1),
        A_log=args["A_log"], dt_bias=args["dt_bias"],
        softplus_beta=args["softplus_beta"], softplus_threshold=args["softplus_threshold"],
        scale=args["scale"], initial_state_source=args["initial_state_source"],
        initial_state_indices=args["initial_state_indices"], out=torch.empty_like(args["v"]),
        use_qk_l2norm_in_kernel=args["use_qk_l2norm_in_kernel"], cu_seqlens=args["cu_seqlens"],
        is_kda=args["is_kda"], is_sglang=args.get("is_sglang", True),
    )


def check_head_case(kernel, H, HV, B=2, seed=2026, state_dtype=torch.float32,
                    activation_dtype=torch.bfloat16, is_kda=True, norm=True, T=4,
                    mode=None, lower_bound=None, tunable=False, graph=False, gate_layout=None,
                    slot_mode=None):
    mode = mode or ("verify" if kernel == "k_last_mtp" else "decode")
    args, backing, lengths, state_ids, cache_ids = make_case(
        state_dtype, activation_dtype, is_kda, mode, H=H, HV=HV, B=B,
        T=T, seed=seed, padded=slot_mode is not None)
    if slot_mode is not None:
        args["is_sglang"] = slot_mode == "sglang"
    args["use_qk_l2norm_in_kernel"] = norm
    if lower_bound is not None:
        args["lower_bound"] = lower_bound
    expected_out, expected_state, expected_cache = reference(args, lengths, state_ids, cache_ids)
    before = backing.clone()
    if gate_layout is not None:
        # Compute the independent reference in the canonical layout first, then transform to the caller layout; covers addressing of every token.
        nt, D = args["q"].shape[1], args["q"].shape[-1]
        if gate_layout == "heads3":
            args["a"] = args["a"].view(nt, HV, D)
        elif gate_layout == "strided_heads3":
            storage = args["a"].new_full((nt, HV + 1, D), 17)
            storage[:, :HV].copy_(args["a"].view(nt, HV, D))
            args["a"] = storage[:, :HV]
        elif gate_layout == "flat2":
            args["a"] = args["a"].view(nt, HV * D)
        elif gate_layout == "flat3":
            args["a"] = args["a"].view(1, nt, HV * D)
        elif gate_layout == "batch4":
            for key in ("q", "k", "v", "a"):
                args[key] = args[key].view(B, -1, *args[key].shape[-2:])
        else:
            raise ValueError(gate_layout)
    kwargs = packed_args(args) if kernel == "k_last_packed" else args
    if kernel == "k_last_packed":
        call = fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable if tunable else packed_forward
    else:
        call = fused_sigmoid_gating_delta_rule_forward_k_last_tunable if tunable else forward
    if tunable:
        kwargs = dict(kwargs, **tunable_config())
    cache = args["intermediate_states_buffer"]
    state_ptr = args["initial_state_source"].data_ptr()
    if graph:
        call(**kwargs)
        torch.cuda.synchronize()
        backing.copy_(before)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = call(**kwargs)
        for _ in range(3):
            backing.copy_(before)
            if cache is not None:
                cache.fill_(-7.0)
            g.replay()
    else:
        out = call(**kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.reshape_as(expected_out), expected_out, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(args["initial_state_source"], expected_state, atol=2e-3, rtol=8e-3)
    assert args["initial_state_source"].data_ptr() == state_ptr
    assert torch.equal(backing[:, -256:], before[:, -256:])
    written = {s for s, length in zip(state_ids, lengths)
               if length and s >= 0 and (args.get("is_sglang", True) or s > 0)}
    untouched = [s for s in range(backing.shape[0]) if s not in written]
    assert torch.equal(backing[untouched], before[untouched])
    if args["disable_state_update"]:
        assert torch.equal(backing, before)
    if cache is not None:
        torch.testing.assert_close(cache, expected_cache, atol=2e-3, rtol=8e-3)
        assert torch.equal(cache[expected_cache == -7.0], expected_cache[expected_cache == -7.0])


@pytest.mark.parametrize("kernel", ["k_last", "k_last_mtp", "k_last_packed"])
@pytest.mark.parametrize("H", [6, 12, 24, 48, 96])
@pytest.mark.parametrize("B", [1, 2, 43])
@pytest.mark.parametrize("seed", [2026, 2027, 2028])
def test_nonpow2_core(kernel, H, B, seed):
    check_head_case(kernel, H, H, B=B, seed=seed)


@pytest.mark.parametrize("kernel", ["k_last", "k_last_mtp", "k_last_packed"])
@pytest.mark.parametrize("H,HV", [(3, 3), (5, 5), (7, 7), (3, 6), (6, 18), (12, 24),
                                  (4, 12)] + [(h, h) for h in (1, 2, 4, 8, 16, 32, 64)])
@pytest.mark.parametrize("is_kda", [False, True])
def test_head_ratios(kernel, H, HV, is_kda):
    check_head_case(kernel, H, HV, is_kda=is_kda)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_kda", [False, True])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("tunable,graph", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("kernel", ["k_last", "k_last_mtp", "k_last_packed"])
def test_nonpow2_entry_modes(kernel, state_dtype, activation_dtype, is_kda, norm, tunable, graph):
    check_head_case(kernel, 6, 18, state_dtype=state_dtype,
                    activation_dtype=activation_dtype, is_kda=is_kda, norm=norm,
                    tunable=tunable, graph=graph)


@pytest.mark.parametrize("kernel", ["k_last", "k_last_mtp", "k_last_packed"])
@pytest.mark.parametrize("slot_mode", ["sglang", "vllm"])
@pytest.mark.parametrize("graph", [False, True])
def test_nonpow2_null_slots(kernel, slot_mode, graph):
    check_head_case(kernel, 6, 18, B=3, state_dtype=torch.bfloat16,
                    slot_mode=slot_mode, graph=graph)


@pytest.mark.parametrize("H,HV", [(4, 8), (6, 18)])
@pytest.mark.parametrize("mode", ["decode", "tree"])
@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("layout", ["heads3", "strided_heads3", "flat2", "flat3", "batch4"])
def test_kda_gate_layouts(H, HV, mode, tunable, layout):
    check_head_case("k_last" if mode == "decode" else "k_last_mtp", H, HV,
                    mode=mode, tunable=tunable, gate_layout=layout)


@pytest.mark.parametrize("T", range(2, 9))
@pytest.mark.parametrize("mode", ["verify", "tree", "ragged", "writeback"])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_nonpow2_mtp_lengths(T, mode, lower_bound):
    check_head_case("k_last_mtp", 6, 6, B=3, T=T, mode=mode, lower_bound=lower_bound)


@pytest.mark.parametrize("kernel", ["k_last", "k_last_packed"])
@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("bad", ["zero_h", "head_ratio", "qkv_width", "a", "b", "A_log",
                                  "dt_bias", "state", "hv_cta_zero", "hv_cta_tail", "bv_zero"])
def test_rejects_bad_heads(kernel, tunable, bad):
    if bad.startswith("hv_cta") or bad == "bv_zero":
        if not tunable:
            pytest.skip("CTA parameters are only exposed by the tunable API")
    if bad == "qkv_width" and kernel != "k_last_packed":
        pytest.skip("only the packed API accepts a QKV width")
    args, backing, _, _, _ = make_case(torch.float32, torch.bfloat16, True, "decode", H=6, HV=18)
    kwargs = packed_args(args) if kernel == "k_last_packed" else args
    if bad in ("zero_h", "head_ratio"):
        h = 0 if bad == "zero_h" else 5
        if kernel == "k_last_packed":
            kwargs["mixed_qkv"] = kwargs["mixed_qkv"].new_zeros(3, (2 * h + 18) * 128)
        else:
            kwargs["q"] = kwargs["q"][:, :, :h].contiguous()
            kwargs["k"] = kwargs["k"][:, :, :h].contiguous()
    elif bad == "qkv_width":
        kwargs["mixed_qkv"] = kwargs["mixed_qkv"][:, :-1].contiguous()
    elif bad in ("a", "b", "A_log", "dt_bias"):
        kwargs[bad] = kwargs[bad][..., :-1].contiguous()
    elif bad == "state":
        kwargs["initial_state_source"] = kwargs["initial_state_source"][:, :-1].contiguous()
    if tunable:
        kwargs.update(tunable_config())
        if bad == "hv_cta_zero":
            kwargs["hv_per_cta"] = 0
        elif bad == "hv_cta_tail":
            kwargs["hv_per_cta"] = 2
        elif bad == "bv_zero":
            kwargs["bv"] = 0
    call = packed_forward if kernel == "k_last_packed" else forward
    if tunable:
        call = (fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable if kernel == "k_last_packed"
                else fused_sigmoid_gating_delta_rule_forward_k_last_tunable)
    before = backing.clone()
    # Diagnostic mode can reject a malformed token/head layout before the
    # ordinary dispatch check; both paths must fail before touching state.
    with pytest.raises(RuntimeError, match="heads|head|HV|QKV|tile|PLA_SANITY_CHECK"):
        call(**kwargs)
    torch.cuda.synchronize()
    assert torch.equal(backing, before)
    assert torch.ones(8, device="cuda").sum().item() == 8


# State pool padding, cache and vector alignment contract.
@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("accepted", [0, -7, -(2**31)])
def test_nonpositive_accepted_uses_first_column(native, tunable, accepted):
    args = inputs()
    saved = args["initial_state_source"].clone()
    expected = launch(native, args, tunable).clone()
    expected_state = args["initial_state_source"].clone()
    args["initial_state_source"].copy_(saved)
    args["cache_steps_or_num_accept"].fill_(accepted)
    actual = launch(native, args, tunable)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(args["initial_state_source"], expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("tunable", [False, True])
def test_empty_request_never_consumes_slot_or_accepted(native, tunable):
    args = inputs((3, 0))
    args["initial_state_indices"][1].fill_(2**31 - 1)
    args["cache_steps_or_num_accept"][1] = 2**31 - 1
    args["initial_state_source"][4:].fill_(float("nan"))
    out = launch(native, args, tunable)
    assert torch.isfinite(out).all()
    assert torch.isnan(args["initial_state_source"][4:]).all()


@pytest.mark.parametrize("tunable", [False, True])
def test_unused_slot_columns_are_not_validated(native, tunable):
    args = inputs((1, 1), columns=3)
    args["initial_state_indices"][:, 1:].fill_(2**31 - 1)
    out = launch(native, args, tunable)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("is_sglang", [False, True])
def test_negative_state_masks_do_not_consume_token_values(native, tunable, is_sglang):
    args = inputs((1, 1), columns=1)
    args["initial_state_indices"][1, 0] = -17
    args["is_sglang"] = is_sglang
    if is_sglang:
        args["initial_state_indices"] = args["initial_state_indices"].flatten()
        args["initial_state_indices"][0] = 0  # SGLang's slot zero is real.
        args["cache_steps_or_num_accept"] = None
    for name in ("q", "k", "v"):
        args[name][:, 1].fill_(float("nan"))
    out = launch(native, args, tunable)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[:, 1]) == 0


@pytest.mark.parametrize("tunable", [False, True])
def test_negative_cache_and_parent_masks(native, tunable):
    args = inputs((2, 2), columns=2)
    args["is_sglang"] = True
    args["initial_state_indices"] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    args["cache_steps_or_num_accept"] = 2
    args["intermediate_states_buffer"] = torch.full(
        (2, 3, 4, 128, 128), float("nan"), device="cuda")
    args["intermediate_state_indices"] = torch.tensor([0, -17], device="cuda", dtype=torch.int32)
    args["retrieve_parent_token"] = torch.tensor(
        [[2**31 - 1, -17], [2**31 - 1, 2**31 - 1]], device="cuda", dtype=torch.int64)
    out = launch(native, args, tunable)
    assert torch.isfinite(out).all()
    assert torch.isnan(args["intermediate_states_buffer"][1]).all()


@pytest.mark.parametrize("tunable", [False, True])
def test_packed_empty_request_ignores_invalid_slot(native, tunable):
    source = inputs((1, 0), columns=1)
    packed = torch.cat([source[n][0].flatten(1) for n in ("q", "k", "v")], dim=1)
    args = {name: source[name] for name in (
        "a", "b", "A_log", "dt_bias", "softplus_beta", "softplus_threshold",
        "scale", "initial_state_source", "use_qk_l2norm_in_kernel", "cu_seqlens", "is_kda")}
    args.update(mixed_qkv=packed, out=torch.empty_like(source["v"]),
                initial_state_indices=torch.tensor([0, 2**31 - 1], device="cuda", dtype=torch.int32),
                is_sglang=True)
    if tunable:
        out = native.fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable(
            **args, bv=16, wn=4, sn=4, hv_per_cta=1)
    else:
        out = native.fused_sigmoid_gating_delta_rule_forward_k_last_packed(**args)
    assert torch.isfinite(out).all()


def _sglang_cache_inputs(state_dtype):
    args = inputs((3, 3), columns=3)
    args["is_sglang"] = True
    args["initial_state_indices"] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    args["initial_state_source"] = args["initial_state_source"].to(state_dtype).normal_(0, 0.01)
    args["cache_steps_or_num_accept"] = 3
    args["intermediate_states_buffer"] = torch.zeros(
        2, 4, 4, 128, 128, dtype=state_dtype, device="cuda")
    args["intermediate_state_indices"] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    # Both rows reload token zero at the third token (a nonsequential parent).
    # The cache comparison
    # therefore covers writes and subsequent parent-state vector reads.
    args["retrieve_parent_token"] = torch.tensor([[0, 0, 0], [0, 0, 0]], device="cuda", dtype=torch.int32)
    return args


def _offset_copy(tensor, offset_elements):
    storage = torch.empty(tensor.numel() + offset_elements,
                          dtype=tensor.dtype, device=tensor.device)
    shifted = storage[offset_elements:].view_as(tensor)
    shifted.copy_(tensor)
    return shifted


@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("cache_ids", [[2**31 - 1, 2**31 - 1], [0, 0]])
def test_vllm_1d_ignores_cache_payload(native, tunable, cache_ids):
    args = _sglang_cache_inputs(torch.bfloat16)
    args.update(is_sglang=False, retrieve_parent_token=None)
    args["initial_state_indices"] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    args["intermediate_state_indices"] = torch.tensor(cache_ids, device="cuda", dtype=torch.int32)
    cache = args["intermediate_states_buffer"].fill_(-7)
    state_before = args["initial_state_source"].clone()
    expected = launch(native, dict(args, intermediate_states_buffer=None,
                                   intermediate_state_indices=None), tunable).clone()
    actual = launch(native, args, tunable)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(args["initial_state_source"], state_before, rtol=0, atol=0)
    assert torch.all(cache == -7)


@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("is_sglang", [False, True])
@pytest.mark.parametrize("missing", ["indices", "buffer"])
def test_incomplete_cache_pair_leaves_state_unchanged(native, tunable, is_sglang, missing):
    args = _sglang_cache_inputs(torch.bfloat16)
    args.update(is_sglang=is_sglang, retrieve_parent_token=None)
    args["initial_state_indices"] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    cache = args["intermediate_states_buffer"].fill_(-7)
    state_before = args["initial_state_source"].clone()
    expected = launch(native, dict(args, intermediate_states_buffer=None,
                                   intermediate_state_indices=None), tunable).clone()
    if missing == "indices":
        args["intermediate_state_indices"] = None  # Kernel keeps cache_idx=-1.
    else:
        args["intermediate_states_buffer"] = None
        args["intermediate_state_indices"].fill_(2**31 - 1)
    actual = launch(native, args, tunable)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(args["initial_state_source"], state_before, rtol=0, atol=0)
    assert torch.all(cache == -7)


@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("defect", ["dtype", "length"])
def test_unused_vllm_cache_still_requires_valid_index_metadata(native, tunable, defect):
    args = _sglang_cache_inputs(torch.bfloat16)
    args.update(is_sglang=False, retrieve_parent_token=None)
    args["initial_state_indices"] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    if defect == "dtype":
        args["intermediate_state_indices"] = args["intermediate_state_indices"].long()
    else:
        args["intermediate_state_indices"] = args["intermediate_state_indices"][:1]
    with pytest.raises(RuntimeError, match="intermediate_state_indices must be contiguous int32"):
        launch(native, args, tunable)


@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_misaligned_cache_rejected_before_launch(native, tunable, state_dtype):
    args = _sglang_cache_inputs(state_dtype)
    cache = _offset_copy(args["intermediate_states_buffer"], 1)
    args["intermediate_states_buffer"] = cache
    state_before = args["initial_state_source"].clone()
    assert cache.data_ptr() % (4 * cache.element_size()) != 0
    with pytest.raises(RuntimeError, match="intermediate cache base requires.*vector alignment"):
        launch(native, args, tunable)
    torch.testing.assert_close(args["initial_state_source"], state_before, rtol=0, atol=0)
    assert torch.count_nonzero(cache) == 0


@pytest.mark.parametrize("tunable", [False, True])
@pytest.mark.parametrize("layout", ["source_base", "source_pitch", "cache_base"])
def test_regular_bf16_eight_byte_state_alignment(native, tunable, layout):
    args = _sglang_cache_inputs(torch.bfloat16)
    initial_state = args["initial_state_source"].clone()
    expected_out = launch(native, args, tunable).clone()
    expected_state = args["initial_state_source"].clone()
    expected_cache = args["intermediate_states_buffer"].clone()
    args["initial_state_source"].copy_(initial_state)
    args["intermediate_states_buffer"].zero_()
    if layout == "source_base":
        args["initial_state_source"] = _offset_copy(initial_state, 4)
        assert args["initial_state_source"].data_ptr() % 16 == 8
    elif layout == "source_pitch":
        state = args["initial_state_source"]
        pitch = state[0].numel() + 4  # BF16 slot pitch is 8 mod 16 bytes.
        storage = torch.empty(state.size(0) * pitch, dtype=state.dtype, device=state.device)
        strided = storage.as_strided(state.shape, (pitch, *state.stride()[1:]))
        strided.copy_(initial_state)
        args["initial_state_source"] = strided
        assert (strided.stride(0) * strided.element_size()) % 16 == 8
    else:
        args["intermediate_states_buffer"] = _offset_copy(args["intermediate_states_buffer"], 4)
        assert args["intermediate_states_buffer"].data_ptr() % 16 == 8
    actual_out = launch(native, args, tunable)
    # Same production kernel and inputs: only the storage address/pitch changes.
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(args["initial_state_source"], expected_state, rtol=0, atol=0)
    torch.testing.assert_close(args["intermediate_states_buffer"], expected_cache, rtol=0, atol=0)
