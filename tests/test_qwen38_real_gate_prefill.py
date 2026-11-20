#!/usr/bin/env python3
"""Qwen3.8-Max layer-0 real-gating coverage for the PLA GDN prefill ABI.

The committed fixture is the audited 2092-token, 8-head prefix of the full
32768-token, 128-head corpus used for the tc01 dispatch sweep. The prefix is
bitwise identical to the original OperatorsForge checkpoint audit.
"""

import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from pla.prefill.flashqla import chunk_gated_delta_rule_fwd, gdn_prefill_packed_qkv_fwd
from pla.prefill.flashqla.ops import SUPPORTED_HEAD_CONFIGS
from pla.prefill.flashqla.private.cp_preprocess import CHUNK_SIZE, calc_cp_seqs
from prefill_utils import ref_gdr, tensor_metrics


D = 128
H = 8
HG = 2
LENGTHS = (654, 1438)
RAW_CU = (0, LENGTHS[0], sum(LENGTHS))
FIXTURE = Path(__file__).with_name("qwen38_max_layer0_gate_2092x8.pt")
FIXTURE_SHA256 = "4c7581a21824d930b49d6588c0ab45d695ae855c37067fda0d08e2dd6b05b35c"
FULL_CORPUS_SHA256 = "f664127ae6852fc11976f278056a9e2a7bd023bd7cb4b398812ee19913a6a3f4"
REFERENCE_PREFIX_SHA256 = "c697c9aad3c695df9bf003f78363019e6d9681625a3a4ebf78dfd3214b0d84b3"


def _metrics(actual, expected):
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    cosine = F.cosine_similarity(
        actual.unsqueeze(0), expected.unsqueeze(0)
    ).item()
    relative = (
        (actual - expected).abs().max()
        / (expected.abs().max() + 1e-8)
    ).item()
    return cosine, relative


def _run(q, k, v, cu, state_pool, state_indices, *, gate, raw):
    common = {
        "scale": D ** -0.5,
        "initial_state": state_pool,
        "initial_state_indices": state_indices,
        "cu_seqlens": cu,
        "output_final_state": False,
        "auto_cp": raw,
        "normalize_qk": True,
        "raw_cu_seqlens": RAW_CU,
    }
    if raw:
        return chunk_gated_delta_rule_fwd(
            q,
            k,
            v,
            None,
            None,
            raw_gate_a=gate["raw_gate_a"],
            raw_beta_b=gate["raw_beta_b"],
            A_log=gate["A_log"],
            dt_bias=gate["dt_bias"],
            **common,
        )

    activated_g = -gate["A_log"].exp()[None, None, :] * F.softplus(
        gate["raw_gate_a"].float()[None, :, :]
        + gate["dt_bias"][None, None, :]
    )
    activated_beta = gate["raw_beta_b"].float().sigmoid()[None, :, :]
    return chunk_gated_delta_rule_fwd(
        q, k, v, activated_g, activated_beta, **common
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
def test_qwen38_real_gate_raw_indexed_state_prefill():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    gate_cpu = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    assert set(gate_cpu) == {"raw_gate_a", "raw_beta_b", "A_log", "dt_bias"}
    assert gate_cpu["raw_gate_a"].shape == (sum(LENGTHS), H)
    assert gate_cpu["raw_beta_b"].shape == (sum(LENGTHS), H)
    assert gate_cpu["A_log"].shape == (H,)
    assert gate_cpu["dt_bias"].shape == (H,)
    assert gate_cpu["raw_gate_a"].dtype == torch.bfloat16
    assert gate_cpu["raw_beta_b"].dtype == torch.bfloat16
    assert gate_cpu["A_log"].dtype == torch.float32
    assert gate_cpu["dt_bias"].dtype == torch.float32

    gate = {name: tensor.contiguous().cuda() for name, tensor in gate_cpu.items()}
    torch.manual_seed(20261013)
    total = sum(LENGTHS)
    q = (torch.randn(1, total, HG, D, device="cuda") * 0.1).to(torch.bfloat16)
    k = (torch.randn(1, total, HG, D, device="cuda") * 0.1).to(torch.bfloat16)
    v = (torch.randn(1, total, H, D, device="cuda") * 0.1).to(torch.bfloat16)
    cu = torch.tensor(RAW_CU, dtype=torch.int32, device="cuda")
    state_indices = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    base_pool = torch.randn(6, H, D, D, device="cuda", dtype=torch.float32) * 0.01
    raw_pool = base_pool.clone()
    activated_pool = base_pool.clone()

    use_cp, _, _, _, _ = calc_cp_seqs(
        cu, CHUNK_SIZE, H, RAW_CU, num_k_heads=HG
    )
    assert not use_cp

    raw_result = _run(
        q, k, v, cu, raw_pool, state_indices, gate=gate, raw=True
    )
    activated_result = _run(
        q, k, v, cu, activated_pool, state_indices, gate=gate, raw=False
    )
    assert raw_result[4] is None
    assert activated_result[4] is None

    output_cos, output_rel = _metrics(raw_result[2], activated_result[2])
    state_cos, state_rel = _metrics(
        raw_pool[state_indices], activated_pool[state_indices]
    )
    assert output_cos > 0.999 and output_rel < 0.05
    assert state_cos > 0.999 and state_rel < 0.05

    untouched = torch.tensor([0, 2, 4, 5], dtype=torch.long, device="cuda")
    assert torch.equal(raw_pool[untouched], base_pool[untouched])
    assert torch.equal(activated_pool[untouched], base_pool[untouched])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("output_final_state", [False, True])
@pytest.mark.parametrize("fresh_prefix", [False, True])
@pytest.mark.parametrize("auto_cp", [False, True])
@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("storage_offset", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("capture", [False, True])
def test_gva_history(
    dtype, state_dtype, indexed, output_final_state, fresh_prefix, auto_cp, heads,
    monkeypatch, storage_offset, capture,
):
    from pla.prefill.flashqla import ops

    total, qk_heads = 8192, heads // 3
    q = torch.ones(1, total, qk_heads, D, device="cuda", dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.zeros(1, total, heads, D, device="cuda", dtype=dtype)
    gate = torch.full((1, total, heads), -1e-7, device="cuda")
    gate[0, 64] = torch.tensor(
        [-6.0, -12.0, -0.25, -0.8], device="cuda").repeat(3)[:heads]
    beta = torch.zeros_like(gate)
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    indices = torch.tensor([3], dtype=torch.int32, device="cuda")
    base_pool = torch.ones(5, heads, D, D, device="cuda", dtype=state_dtype)
    base_pool *= torch.linspace(0.5, 1.5, D, device="cuda")[None, None, :, None]
    base_pool *= torch.linspace(0.75, 1.25, D, device="cuda")[None, None, None, :]
    if fresh_prefix:
        q.zero_()
        q[..., 0] = 1
        k[0, 0, :, 0] = 1
        v[0, 0] = 16 * torch.linspace(0.5, 1.5, D, device="cuda")
        beta[0, 0] = 1
        base_pool[3].zero_()
        decay = gate[0, 1:].double().cumsum(0).exp().float()
        expected_state = torch.zeros(heads, D, D, device="cuda")
        expected_state[:, :, 0] = (
            decay[-1, :, None] * v[0, 0].float() / (1 + 1e-6) ** 0.5)
        decay = torch.cat((torch.ones(1, heads, device="cuda"), decay), dim=0)
        expected_output = (
            decay[None, :, :, None] * v[:, :1].float()
            / (D * (1 + 1e-6)) ** 0.5)
    else:
        decay = gate[0].double().cumsum(0).exp().float()
        expected_state = base_pool[3].float() * decay[-1, :, None, None]
        expected_output = (
            decay[None, :, :, None] * base_pool[3].float().mean(-1)[None, None])
    state_before = (base_pool.clone() if indexed else
                    base_pool[3:4].transpose(-1, -2).contiguous())
    storage = torch.full((storage_offset + state_before.numel() + 4,), -123.0,
                         device="cuda", dtype=state_dtype)
    state = storage[storage_offset:storage_offset + state_before.numel()].view_as(state_before)
    state.copy_(state_before)
    assert state.is_contiguous() and state.storage_offset() == storage_offset

    def run():
        return chunk_gated_delta_rule_fwd(
            q, k, v, gate, beta, initial_state=state,
            initial_state_indices=indices if indexed else None,
            cu_seqlens=cu, raw_cu_seqlens=(0, total), normalize_qk=True,
            output_final_state=output_final_state, auto_cp=auto_cp,
        )

    with patch.object(ops, "calc_cp_seqs", wraps=ops.calc_cp_seqs) as planner:
        result = run()
        if auto_cp:
            assert planner.call_count == 1
            props = torch.cuda.get_device_properties(q.device)
            if (props.major, props.minor, props.multi_processor_count) == (8, 9, 39):
                plan = calc_cp_seqs(cu, 32, heads, (0, total), num_k_heads=qk_heads)
                assert plan[0] and plan[1].numel() == (13 if heads == 6 else 7)
        else:
            planner.assert_not_called()
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            state.copy_(state_before)
            run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            state.copy_(state_before)
            result = run()
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    assert bool((storage[:storage_offset] == -123).all())
    assert bool((storage[storage_offset + state.numel():] == -123).all())
    torch.testing.assert_close(result[2].float(), expected_output, rtol=0.05, atol=0.0)
    if indexed:
        actual_state = state[3].float()
        untouched = torch.tensor([0, 1, 2, 4], device="cuda")
        assert torch.equal(state[untouched], base_pool[untouched])
    elif output_final_state:
        actual_state = result[4][0].float().transpose(-1, -2)
    if indexed or output_final_state:
        relative = ((actual_state - expected_state).flatten(1).norm(dim=1)
                    / expected_state.flatten(1).norm(dim=1))
        assert float(relative.max()) < 0.05
    if output_final_state:
        assert result[4].dtype == state_dtype
        if indexed:
            assert torch.equal(result[4], state[indices])
    else:
        assert result[4] is None


def _fallback_mask(pattern, segments, heads):
    def both_transitions(mask):
        steps = mask[1:] - mask[:-1]
        return bool((steps == 1).any(0).all()) and bool((steps == -1).any(0).all())

    seg = torch.arange(segments)[:, None]
    head = torch.arange(heads)[None, :]
    if pattern == "alternate":
        mask = (seg + head) % 2
    elif pattern == "runs":
        mask = (seg // 2 + head) % 2
    elif pattern == "random":
        generator = torch.Generator().manual_seed(0)
        mask = torch.randint(0, 2, (segments, heads), generator=generator)
        while not both_transitions(mask):
            mask = torch.randint(0, 2, (segments, heads), generator=generator)
    else:
        mask = torch.full((segments, heads), int(pattern == "ones"))
    # Every head of a mixed pattern sees both a 0->1 and a 1->0 transition.
    assert pattern in ("zeros", "ones") or both_transitions(mask)
    return mask.to(device="cuda", dtype=torch.int8)


def _check_correct_h0(
    heads, segments, dtype, offset, raw_mode, pattern, capture, *, folded=False,
):
    from pla.prefill.flashqla import ops

    shape = (segments, heads, D, D)
    size = segments * heads * D * D
    storage = torch.full((size + offset + 8,), -123.0, device="cuda", dtype=dtype)
    ht = storage[offset:offset + size].view(shape)
    values = torch.arange(size, device="cuda").remainder(17).view(shape).float() / 128
    ht.copy_(values)
    assert ht.is_contiguous() and ht.storage_offset() == offset
    # Power-of-two diagonals keep the recurrence exact; neighbouring segments
    # differ so a transition read from the wrong mt buffer is caught.
    seg = torch.arange(segments, device="cuda")[:, None]
    head = torch.arange(heads, device="cuda")[None, :]
    scale = 2.0 ** -(1 + (seg + 2 * head) % 3).float()
    mt = (torch.eye(D, device="cuda") * scale[..., None, None]).to(dtype)
    if folded:
        mt_storage = torch.empty(size + offset, device="cuda", dtype=dtype)
        shifted_mt = mt_storage[offset:].view(shape)
        shifted_mt.copy_(mt)
        mt = shifted_mt
        assert mt.is_contiguous() and mt.storage_offset() == offset
    mask = _fallback_mask(pattern, segments, heads)
    mapping = torch.tensor([0, segments], dtype=torch.int32, device="cuda")
    raw = torch.ones((3, heads, D, D), device="cuda") if raw_mode != "none" else None
    indices = torch.tensor([2], dtype=torch.int32, device="cuda") if raw_mode == "indexed" else None
    kwargs = dict(
        raw_h0=raw,
        raw_h0_indices=indices,
        raw_h0_k_last=raw_mode == "indexed",
        folded=folded,
    )
    expected = []
    state = torch.zeros((heads, D, D), device="cuda") if raw is None else raw[0].clone()
    decay = (mask.float() * scale)[..., None, None]
    for i in range(segments):
        expected.append(state.clone())
        state = ht[i].float() + decay[i] * state.to(dtype).float()
    expected = torch.stack(expected)

    def run():
        return ops.flashqla_cute.correct_h0(ht, mt, mask, mapping, 1, **kwargs)

    actual = run()
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(ht, values.to(dtype))
    assert bool((storage[:offset] == -123).all())
    assert bool((storage[offset + size:] == -123).all())
    if raw is not None:
        assert bool((raw == 1).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("heads,segments", [(6, 12), (12, 6)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", range(9))
@pytest.mark.parametrize("raw_mode", ["none", "legacy", "indexed"])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_correct_h0_unaligned_ht(heads, segments, dtype, offset, raw_mode, fallback, capture):
    _check_correct_h0(heads, segments, dtype, offset, raw_mode,
                      "ones" if fallback else "zeros", capture)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("heads,segments", [(6, 12), (12, 6)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("raw_mode", ["none", "legacy", "indexed"])
@pytest.mark.parametrize("pattern", ["alternate", "runs", "random"])
@pytest.mark.parametrize("capture", [False, True])
def test_correct_h0_mixed_fallback(heads, segments, dtype, offset, raw_mode, pattern, capture):
    # Offset 0 takes the 890P pipeline, whose mt prefetch and wait depend on
    # the current and next masks; offset 1 takes the scalar path.
    _check_correct_h0(heads, segments, dtype, offset, raw_mode, pattern, capture)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("offset", [0, 1])
def test_folded_fp32_correction_mt_offset(offset):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires an M890P device")
    _check_correct_h0(
        12, 6, torch.float32, offset, "indexed", "alternate", False, folded=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("threshold", [-4.0, None])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_cp_finite_initial_state_carry(
    dtype, state_dtype, indexed, threshold, output_final_state,
):
    """Decay-only inputs expose a lost carry even when its absolute value is tiny."""
    lengths = (8192, 259)
    offsets = (0, lengths[0], sum(lengths))
    heads, qk_heads = 16, 2
    q = torch.ones(1, sum(lengths), qk_heads, D, device="cuda", dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.zeros(1, sum(lengths), heads, D, device="cuda", dtype=dtype)
    gate = torch.full((1, sum(lengths), heads), -1e-7, device="cuda")
    # Exercise both sides of the warmup threshold in different heads. The
    # -12 heads also catch truncation in the established prepare/correct path.
    impulses = torch.tensor([-6.0, -12.0, -0.25, -6.0], device="cuda").repeat(4)
    gate[0, 64] = impulses
    gate[0, lengths[0] + 64] = impulses
    beta = torch.zeros_like(gate)
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    indices = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    base_pool = torch.ones(5, heads, D, D, device="cuda", dtype=state_dtype)
    # Non-symmetric state detects accidental DK/DV transposes as well as loss.
    base_pool *= torch.linspace(0.5, 1.5, D, device="cuda")[None, None, :, None]
    base_pool *= torch.linspace(0.75, 1.25, D, device="cuda")[None, None, None, :]
    initial = base_pool[indices].float()
    expected_output = torch.empty(1, sum(lengths), heads, D, device="cuda")
    expected_state = torch.empty_like(initial)
    for batch, (start, end) in enumerate(zip(offsets, offsets[1:])):
        decay = gate[0, start:end].double().cumsum(0).exp().float()
        expected_state[batch] = initial[batch] * decay[-1, :, None, None]
        expected_output[0, start:end] = (
            decay[:, :, None] * initial[batch].mean(-1)[None, :, :])

    # A forced feasible plan makes this a CP regression on either architecture;
    # do not silently accept a serial fallback as evidence of correct CP carry.
    from pla.prefill.flashqla import ops

    for auto_cp in (False, True):
        state = (base_pool.clone() if indexed else
                 initial.to(state_dtype).transpose(-1, -2).contiguous())
        with patch.object(ops, "calc_cp_seqs", wraps=ops.calc_cp_seqs) as planner:
            result = chunk_gated_delta_rule_fwd(
                q, k, v, gate, beta,
                initial_state=state,
                initial_state_indices=indices if indexed else None,
                cu_seqlens=cu, raw_cu_seqlens=offsets,
                normalize_qk=True, output_final_state=output_final_state,
                auto_cp=auto_cp, overlap_warmup_threshold=threshold,
                force_tc01_cp_candidate=auto_cp,
            )
            if auto_cp:
                assert planner.call_count == 1
                plan = calc_cp_seqs(
                    cu, CHUNK_SIZE, heads, offsets, num_k_heads=qk_heads,
                    tc01_force_candidate=True)
                assert plan[0] and plan[1].numel() > cu.numel()
            else:
                planner.assert_not_called()
        torch.testing.assert_close(
            result[2].float(), expected_output, rtol=0.05, atol=0.0)
        if indexed:
            actual_state = state[indices].float()
            untouched = torch.tensor([0, 2, 4], device="cuda")
            assert torch.equal(state[untouched], base_pool[untouched])
        elif output_final_state:
            actual_state = result[4].float().transpose(-1, -2)
        if indexed or output_final_state:
            relative = ((actual_state - expected_state).flatten(2).norm(dim=2)
                        / expected_state.flatten(2).norm(dim=2))
            assert float(relative.max()) < 0.05, "CP lost initial-state carry"
        if output_final_state:
            assert result[4].dtype == state_dtype
            if indexed:
                assert torch.equal(result[4], state[indices])
        else:
            assert result[4] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_auto_cp_initial_state_carry(dtype):
    """Keep the original single-sequence serving repro on production dispatch."""
    from pla.prefill.flashqla import ops

    total, heads, qk_heads = 8192, 16, 2
    q = torch.ones(1, total, qk_heads, D, device="cuda", dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.zeros(1, total, heads, D, device="cuda", dtype=dtype)
    gate = torch.full((1, total, heads), -1e-7, device="cuda")
    gate[:, 64] = -6.0
    beta = torch.zeros_like(gate)
    cu = torch.tensor([0, total], device="cuda", dtype=torch.int32)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    expected = gate[0].double().sum(0).exp().float()[:, None, None]
    for auto_cp in (False, True):
        pool = torch.zeros(2, heads, D, D, device="cuda")
        pool[1] = 1.0
        with patch.object(ops.flashqla_cute, "fused_fwd",
                          wraps=ops.flashqla_cute.fused_fwd) as fused:
            result = chunk_gated_delta_rule_fwd(
                q, k, v, gate, beta,
                initial_state=pool, initial_state_indices=indices,
                cu_seqlens=cu, normalize_qk=True,
                output_final_state=False, auto_cp=auto_cp)
            expected_cp = auto_cp and torch.cuda.get_device_capability() != (8, 0)
            assert (fused.call_args.args[7] is not None) == expected_cp
        relative = (pool[1] - expected).norm() / expected.expand_as(pool[1]).norm()
        assert float(relative) < 0.05, "auto CP lost initial-state carry"
        torch.testing.assert_close(
            result[2][0, -1].float(), expected[:, 0].expand(heads, D),
            rtol=0.05, atol=0.0)
        assert result[4] is None
        assert not torch.count_nonzero(pool[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_auto_cp_fresh_prefix(dtype):
    """Token zero writes history that None and explicit zero must both retain."""
    from pla.prefill.flashqla import ops

    total, heads, qk_heads = 8192, 16, 2
    q = torch.zeros(1, total, qk_heads, D, device="cuda", dtype=dtype)
    q[..., 0] = 1
    k = torch.zeros_like(q)
    k[:, 0, :, 0] = 1
    v = torch.zeros(1, total, heads, D, device="cuda", dtype=dtype)
    v[:, 0] = 1
    gate = torch.full((1, total, heads), -1e-7, device="cuda")
    gate[:, 64] = -0.8
    beta = torch.zeros_like(gate)
    beta[:, 0] = 1
    cu = torch.tensor([0, total], device="cuda", dtype=torch.int32)
    expected_state = gate[0, 1:].double().sum(0).exp().float() / (1 + 1e-6) ** 0.5
    expected_output = expected_state / (D * (1 + 1e-6)) ** 0.5
    for auto_cp in (False, True):
        for zero_state in (False, True):
            initial = (torch.zeros(1, heads, D, D, device="cuda")
                       if zero_state else None)
            with patch.object(ops.flashqla_cute, "fused_fwd",
                              wraps=ops.flashqla_cute.fused_fwd) as fused:
                result = chunk_gated_delta_rule_fwd(
                    q, k, v, gate, beta, initial_state=initial,
                    cu_seqlens=cu, normalize_qk=True,
                    output_final_state=True, auto_cp=auto_cp)
                if not zero_state:
                    expected_cp = auto_cp and torch.cuda.get_device_capability() != (8, 0)
                    assert (fused.call_args.args[7] is not None) == expected_cp
            torch.testing.assert_close(
                result[4][0, :, 0].float(), expected_state[:, None].expand(heads, D),
                rtol=0.05, atol=0.0)
            assert not torch.count_nonzero(result[4][0, :, 1:])
            torch.testing.assert_close(
                result[2][0, -1].float(), expected_output[:, None].expand(heads, D),
                rtol=0.05, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("heads,qk_heads,lengths", [
    (16, 2, (8192,)), (32, 4, (6397, 1795)),
    *((8, hg, lengths) for hg in (1, 2, 4) for lengths in (
        (10000,), (32768,), (8000, 24000), (4000, 4000, 24000))),
])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("form", ["none", "legacy", "indexed"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_final_state", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_tc01_automatic_overlap_stays_serial(
    heads, qk_heads, lengths, packed, form, state_dtype, output_final_state, capture,
):
    import itertools
    from pla.prefill.flashqla import gdn_prefill_packed_qkv_fwd, ops

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("tc01 automatic CP admission coverage")
    torch.manual_seed(20261112)
    total, batch = sum(lengths), len(lengths)
    offsets = tuple(itertools.accumulate((0, *lengths)))
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    qkv = torch.randn(total, (2 * qk_heads + heads) * D,
                      device="cuda", dtype=torch.bfloat16) * 0.1
    q, k, v = (part.reshape(1, total, h, D).contiguous()
               for part, h in zip(qkv.split([qk_heads * D, qk_heads * D, heads * D], -1),
                                  (qk_heads, qk_heads, heads)))
    raw_a = torch.randn(total, heads, device="cuda", dtype=torch.bfloat16)
    raw_b = torch.randn_like(raw_a)
    a_log = torch.full((heads,), -5.0, device="cuda")
    bias = torch.zeros_like(a_log)
    g = -a_log.exp()[None, None] * F.softplus(raw_a.float()[None])
    beta = raw_b.float().sigmoid()[None]
    state = indices = None
    if form == "indexed":
        state = torch.randn(batch + 2, heads, D, D, device="cuda", dtype=state_dtype) * 0.01
        indices = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)
    elif form == "legacy":
        state = torch.randn(batch, heads, D, D, device="cuda", dtype=state_dtype) * 0.01
    base = None if state is None else state.clone()

    def run(auto_cp):
        if state is not None:
            state.copy_(base)
        common = dict(initial_state=state, initial_state_indices=indices,
                      output_final_state=output_final_state, final_state_dtype=state_dtype,
                      auto_cp=auto_cp, raw_cu_seqlens=offsets)
        if packed:
            return gdn_prefill_packed_qkv_fwd(
                qkv, raw_a, raw_b, a_log, bias, cu,
                num_qk_heads=qk_heads, num_v_heads=heads, **common)
        return chunk_gated_delta_rule_fwd(q, k, v, g, beta,
                                         cu_seqlens=cu, normalize_qk=True, **common)

    ref = run(False)
    expected_output = ref[2].cpu()
    expected_final = None if ref[4] is None else ref[4].cpu()
    expected_pool = None if state is None else state.cpu()
    del ref
    with patch.object(ops.flashqla_cute, "fused_fwd", wraps=ops.flashqla_cute.fused_fwd) as fused, \
            patch.object(ops.flashqla_cute, "prepare_h") as prepare, \
            patch.object(ops.flashqla_cute, "correct_h0") as correct:
        result = run(True)
        assert fused.call_args.args[7] is None
        prepare.assert_not_called()
        correct.assert_not_called()
    if capture:
        del result
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = run(True)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(result[2].cpu(), expected_output)
    if expected_final is None:
        assert result[4] is None
    else:
        assert torch.equal(result[4].cpu(), expected_final)
    if state is not None:
        assert torch.equal(state.cpu(), expected_pool)
        if form == "indexed":
            assert torch.equal(state[0], base[0])
            assert torch.equal(state[-1], base[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("threshold", [-0.75, None])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_cp_fresh_prefix_history(dtype, state_dtype, threshold, output_final_state):
    """Preserve internal carry in replay and staged paths across ragged batches."""
    from pla.prefill.flashqla import ops

    offsets = (0, 8192, 8451)
    total, heads, qk_heads = offsets[-1], 16, 2
    q = torch.zeros(1, total, qk_heads, D, device="cuda", dtype=dtype)
    q[..., 0] = 1
    k = torch.zeros_like(q)
    v = torch.zeros(1, total, heads, D, device="cuda", dtype=dtype)
    gate = torch.full((1, total, heads), -1e-7, device="cuda")
    beta = torch.zeros_like(gate)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    impulses = torch.tensor([-0.8, -6.0, -10.25, -0.25], device="cuda").repeat(4)
    expected_output = torch.zeros_like(v, dtype=torch.float32)
    expected_state = torch.zeros(2, heads, D, D, device="cuda")
    for batch, (start, end) in enumerate(zip(offsets, offsets[1:])):
        k[0, start, :, 0] = 1
        v[0, start] = torch.linspace(0.5, 1.5, D, device="cuda") * (batch + 1)
        beta[0, start] = 1
        gate[0, start + 64] = impulses
        decay = gate[0, start:end].double().cumsum(0)
        decay = (decay - decay[0]).exp().float()
        prefix = v[0, start].float() / (1 + 1e-6) ** 0.5
        expected_state[batch, :, 0] = prefix * decay[-1, :, None]
        expected_output[0, start:end] = (
            decay[:, :, None] * prefix[None] / (D * (1 + 1e-6)) ** 0.5)

    for form in ("none", "legacy", "indexed"):
        initial = indices = None
        if form == "legacy":
            initial = torch.zeros_like(expected_state, dtype=state_dtype)
        elif form == "indexed":
            initial = torch.zeros(5, heads, D, D, device="cuda", dtype=state_dtype)
            indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
        with patch.object(ops.flashqla_cute, "fused_fwd",
                          wraps=ops.flashqla_cute.fused_fwd) as fused, \
             patch.object(ops.flashqla_cute, "get_warmup_chunks",
                          wraps=ops.flashqla_cute.get_warmup_chunks) as warmup:
            result = chunk_gated_delta_rule_fwd(
                q, k, v, gate, beta, initial_state=initial,
                initial_state_indices=indices, cu_seqlens=cu,
                raw_cu_seqlens=offsets, normalize_qk=True,
                final_state_dtype=state_dtype, output_final_state=output_final_state,
                auto_cp=True, force_tc01_cp_candidate=True,
                overlap_warmup_threshold=threshold)
            assert fused.call_args.args[7] is not None
            warmup.assert_not_called()
            if threshold is not None and form != "legacy":
                assert torch.all(fused.call_args.args[-1] == 1)
            else:
                assert fused.call_args.args[13].numel() > 0
        torch.testing.assert_close(
            result[2].float(), expected_output, rtol=0.05, atol=0.0)
        if form == "indexed":
            actual = initial[indices].float().transpose(-1, -2)
            assert not torch.count_nonzero(initial[[0, 2, 4]])
        elif output_final_state:
            actual = result[4].float()
        if form == "indexed" or output_final_state:
            torch.testing.assert_close(actual, expected_state, rtol=0.05, atol=0.0)
        if output_final_state:
            assert result[4].dtype == state_dtype
        else:
            assert result[4] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
def test_overlap_without_initial_state_preserves_full_warmup():
    """Even a fresh sequence needs every predecessor state transition."""
    from pla.prefill.flashqla import ops

    total, heads, qk_heads = 8192, 16, 2
    q = torch.ones(1, total, qk_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros(1, total, heads, D, device="cuda", dtype=torch.bfloat16)
    gate = torch.full((1, total, heads), -0.1, device="cuda")
    cu = torch.tensor([0, total], device="cuda", dtype=torch.int32)
    with patch.object(ops.flashqla_cute, "get_warmup_chunks",
                      wraps=ops.flashqla_cute.get_warmup_chunks) as warmup, \
         patch.object(ops.flashqla_cute, "fused_fwd",
                      wraps=ops.flashqla_cute.fused_fwd) as fused:
        result = chunk_gated_delta_rule_fwd(
            q, k, v, gate, torch.zeros_like(gate), cu_seqlens=cu,
            output_final_state=False, auto_cp=True,
            overlap_warmup_threshold=-4.0, force_tc01_cp_candidate=True)
        warmup.assert_not_called()
        assert fused.call_args.args[7] is not None
        assert fused.call_args.args[-3] is None
        assert torch.all(fused.call_args.args[-1] == 1)
        cp_cu = fused.call_args.args[6]
        full_chunks = torch.div(
            cp_cu[1:] - cp_cu[:-1] + CHUNK_SIZE - 1,
            CHUNK_SIZE, rounding_mode="floor")
        assert torch.equal(
            fused.call_args.args[-2].view(-1, heads),
            full_chunks[:, None].expand(-1, heads))
    assert result[4] is None
    assert not torch.count_nonzero(result[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("form", ["none", "legacy", "indexed"])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_tc01_lpt_sequence_mapping(dtype, state_dtype, form, output_final_state):
    import itertools
    from pla.prefill.flashqla import gdn_prefill_packed_qkv_fwd, ops

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("tc01 scheduling coverage")
    heads, qk_heads = 16, 2
    schedules = list(itertools.permutations((7553, 24903, 312)))
    schedules += [(15104, 1280), (1280, 15104), (8192, 8192), (6000, 6000, 6000)]
    torch.manual_seed(20261022)
    for lengths in schedules:
        offsets = (0, *itertools.accumulate(lengths))
        valid = offsets[-1]
        total = valid + 97
        packed = torch.randn(total, (2 * qk_heads + heads) * D,
                             device="cuda", dtype=dtype) * 0.1
        raw_a = torch.randn(total, heads, device="cuda", dtype=dtype)
        raw_b = torch.randn_like(raw_a)
        a_log = torch.full((heads,), -7.0, device="cuda")
        dt_bias = torch.linspace(-0.5, 0.5, heads, device="cuda")
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        slots = [4, 1, 3][:len(lengths)]
        indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
        pool = (torch.randn(6, heads, D, D, device="cuda") * 0.1).to(state_dtype)
        pool_before = pool.clone()
        initial = (None if form == "none" else pool if form == "indexed"
                   else pool[indices].transpose(-1, -2).contiguous())
        expected_outputs, expected_states = [], []
        for batch, (start, end) in enumerate(zip(offsets, offsets[1:])):
            h0 = (None if form == "none" else
                  pool_before[slots[batch]:slots[batch] + 1].transpose(-1, -2).contiguous())
            reference = gdn_prefill_packed_qkv_fwd(
                packed[start:end], raw_a[start:end], raw_b[start:end], a_log, dt_bias,
                torch.tensor([0, end - start], dtype=torch.int32, device="cuda"),
                num_qk_heads=qk_heads, num_v_heads=heads, initial_state=h0,
                auto_cp=False, output_final_state=True, final_state_dtype=state_dtype,
                raw_cu_seqlens=(0, end - start))
            expected_outputs.append(reference[2])
            expected_states.append(reference[4])
        with patch.object(ops.flashqla_cute, "fused_fwd",
                          wraps=ops.flashqla_cute.fused_fwd) as fused:
            actual = gdn_prefill_packed_qkv_fwd(
                packed, raw_a, raw_b, a_log, dt_bias, cu,
                num_qk_heads=qk_heads, num_v_heads=heads, initial_state=initial,
                initial_state_indices=indices if form == "indexed" else None,
                auto_cp=True, output_final_state=output_final_state,
                final_state_dtype=state_dtype, raw_cu_seqlens=offsets)
            assert fused.call_args.args[7] is None
            assert fused.call_args.args[9] == valid
        output_cos, output_rel = _metrics(
            actual[2][:, :valid], torch.cat(expected_outputs, dim=1))
        assert output_cos > 0.999 and output_rel < 0.05
        assert not torch.count_nonzero(actual[2][:, valid:])
        if form == "indexed":
            state = pool[indices].transpose(-1, -2)
            untouched = [slot for slot in range(6) if slot not in slots]
            assert torch.equal(pool[untouched], pool_before[untouched])
        elif output_final_state:
            state = actual[4]
        if form == "indexed" or output_final_state:
            state_cos, state_rel = _metrics(state, torch.cat(expected_states))
            assert state_cos > 0.999 and state_rel < 0.05
        if output_final_state:
            assert actual[4].dtype == state_dtype
        else:
            assert actual[4] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tc01_lpt_empty_sequence_keeps_indexed_slots(dtype):
    from pla.prefill.flashqla import gdn_prefill_packed_qkv_fwd

    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("tc01 scheduling coverage")
    total, heads, qk_heads = 16384, 16, 2
    packed = torch.zeros(total, (2 * qk_heads + heads) * D, device="cuda", dtype=dtype)
    raw = torch.zeros(total, heads, device="cuda", dtype=dtype)
    for offsets in ((0, 0, total), (0, total, total),
                    (0, 0, total, total), (0, total, total, total)):
        pool = torch.arange(6, device="cuda", dtype=torch.float32)[:, None, None, None]
        pool = pool.expand(6, heads, D, D).contiguous()
        before = pool.clone()
        slots = [4, 1, 3][:len(offsets) - 1]
        result = gdn_prefill_packed_qkv_fwd(
            packed, raw, raw, torch.full((heads,), -30.0, device="cuda"),
            torch.zeros(heads, device="cuda"),
            torch.tensor(offsets, dtype=torch.int32, device="cuda"),
            num_qk_heads=qk_heads, num_v_heads=heads, initial_state=pool,
            initial_state_indices=torch.tensor(slots, dtype=torch.int32, device="cuda"),
            auto_cp=True, output_final_state=False, raw_cu_seqlens=offsets)
        assert torch.equal(pool, before)
        assert not torch.count_nonzero(result[2])
        assert result[4] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prepare_h_partial_tail_chunk_reads_inside_segment(dtype):
    """A segment off the chunk grid keeps its warmup inside its own tokens.

    CP hands prepare_h the full segment chunk count, so a length that is not a
    multiple of CHUNK_SIZE leaves a partial chunk at the segment tail. Counting
    the read origin back from the segment end instead of forward from the chunk
    grid moved that origin below the segment start: negative for the first
    segment, and into the preceding tokens for every later one.
    """
    from pla.prefill.flashqla import ops

    length, heads, qk_heads = 897, 16, 2
    assert length % CHUNK_SIZE, "the regression needs a partial tail chunk"
    pad, total = CHUNK_SIZE, 2 * length
    offsets = (0, length, total)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    # Two segments carrying the same tokens. Every prepare_h input repeats, so
    # the two states can only differ through a read outside the segment.
    torch.manual_seed(20261029)

    def _repeated(*shape):
        return (torch.randn(1, length, *shape, device="cuda") * 0.1).repeat(
            (1, 2) + (1,) * len(shape))

    q = _repeated(qk_heads, D).to(dtype)
    k = _repeated(qk_heads, D).to(dtype)
    v = _repeated(heads, D).to(dtype)
    gate = -_repeated(heads).abs() - 1e-3
    beta = torch.rand(1, length, heads, device="cuda").repeat(1, 2, 1)

    # Build A and the chunk-local gate cumsum exactly as production does; both
    # live on the chunk grid anchored at each segment start.
    g_cum, A, _, _, _ = chunk_gated_delta_rule_fwd(
        q, k, v, gate, beta, scale=D ** -0.5, cu_seqlens=cu,
        raw_cu_seqlens=offsets, output_final_state=False, auto_cp=False)

    def _padded(tensor):
        storage = torch.zeros(
            (1, pad + total) + tensor.shape[2:],
            device="cuda", dtype=tensor.dtype)
        storage[:, pad:] = tensor
        return storage

    storages = [_padded(tensor) for tensor in (k, v, A, g_cum, beta)]
    warmup = torch.full(((len(offsets) - 1) * heads,),
                        (length + CHUNK_SIZE - 1) // CHUNK_SIZE,
                        device="cuda", dtype=torch.int32)

    def _prepare():
        views = [storage[:, pad:] for storage in storages]
        return ops.flashqla_cute.prepare_h(*views, cu, warmup, heads, qk_heads)

    ht, mt = (tensor.clone() for tensor in _prepare())
    for storage in storages:
        storage[:, :pad] = float("nan")
    poisoned_ht, poisoned_mt = _prepare()

    # The padding sits before the logical tensors, so it must not be reachable.
    assert torch.equal(ht, poisoned_ht) and torch.equal(mt, poisoned_mt)
    assert torch.isfinite(ht.float()).all() and torch.isfinite(mt.float()).all()
    first, second = ht.view(len(offsets) - 1, heads, D, D)
    assert torch.equal(first, second)


# Folded long-prefill regression: full history, packed tails and resident state.
@pytest.fixture(scope="module")
def folded_precision():
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU M890P")
    props = torch.cuda.get_device_properties("cuda")
    if (props.major, props.minor, props.multi_processor_count) != (8, 9, 39):
        pytest.skip("requires the 39-CU folded prefill schedule")
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = tf32


def assert_folded_close(actual, expected):
    cosine, relative, finite = tensor_metrics(actual, expected)
    assert finite and cosine > 0.999 and relative < 0.05, (cosine, relative, finite)


_REFERENCE_GEMM_BATCH_LIMIT = 65535


def folded_prefill_reference(q, k, v, g, beta, offsets, initial_state):
    # The FP32 reference batches one GEMM per (chunk, value head). Start
    # each slice on a sequence-local chunk boundary and carry its final state.
    chunks_per_slice = min(32768 // CHUNK_SIZE, _REFERENCE_GEMM_BATCH_LIMIT // v.size(2))
    slice_size = chunks_per_slice * CHUNK_SIZE
    outputs, final_states = [], []
    for seq, (start, stop) in enumerate(zip(offsets, offsets[1:])):
        state = initial_state[seq : seq + 1]
        for begin in range(start, stop, slice_size):
            sl = slice(begin, min(begin + slice_size, stop))
            qf, kf = q[:, sl].float(), k[:, sl].float()
            qf = qf * (qf.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            kf = kf * (kf.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            _, output, _, _, state = ref_gdr.chunk_gated_delta_rule_fwd(
                qf, kf, v[:, sl].float(), g[:, sl], beta[:, sl],
                initial_state=state,
                chunk_size=CHUNK_SIZE,
            )
            outputs.append(output)
        final_states.append(state)
    return torch.cat(outputs, dim=1), torch.cat(final_states, dim=0)


@pytest.mark.parametrize(
    "lengths,physical,packed,auto_cp,expect_folded,heads,qk_heads",
    [
        (*case, 24, 8)
        for case in [
            ((32768,), 32768, False, True, True),
            ((32768,), 32768, False, False, True),
            ((8193, 8191), 16416, False, True, True),
            ((32749,), 32768, True, True, False),
            ((65536,), 65536, False, True, True),
            ((65537,), 65600, False, True, True),
            ((98304,), 98304, False, True, True),
            ((131072,), 131072, False, True, True),
            ((32768, 32768, 32768, 32768), 131072, False, True, True),
            ((511,), 512, True, True, False),
            ((512,), 512, False, True, True),
            ((512, 512), 1024, False, True, True),
        ]
    ]
    + [
        ((4097,), 4160, False, True, True, h, hg)
        for h, hg in sorted(SUPPORTED_HEAD_CONFIGS)
    ]
    + [
        ((4096, 4096), 8192, False, True, h != hg, h, hg)
        for h, hg in sorted(SUPPORTED_HEAD_CONFIGS)
    ]
    + [
        ((32749,), 32768, True, True, False, 12, 4),
        ((32749,), 32768, False, True, True, 12, 4),
        ((4095, 8192), 12288, True, True, False, 32, 8),
        ((128, 16256), 16384, True, True, False, 64, 8),
        ((4096, 8192), 12288, False, True, True, 64, 16),
        ((4096, 8193), 12352, True, True, False, 64, 16),
        ((4096, 32768), 36864, True, True, False, 128, 16),
        ((1, 32767), 32768, False, True, True, 24, 8),
        ((1023,), 1024, True, True, False, 4, 2),
        ((1024,), 1024, False, True, True, 4, 2),
        ((8192,), 8192, True, True, False, 6, 2),
        ((8192,), 8192, True, True, False, 12, 4),
        ((8191,), 8191, True, True, False, 24, 8),
        ((605,), 605, True, True, True, 24, 8),
        ((605,), 640, True, True, False, 24, 8),
        ((605,), 605, True, False, False, 24, 8),
        ((121, 799), 920, True, True, True, 16, 4),
        ((799, 121), 920, True, True, False, 16, 4),
        ((1987,), 1987, True, True, False, 8, 2),
        ((8192,), 8192, True, True, True, 24, 8),
        ((32768,), 32768, True, True, True, 24, 8),
        ((32768,), 32768, True, True, True, 12, 4),
        ((4096, 8192), 12288, True, True, True, 64, 8),
        ((1,) * 7 + (32761,), 32768, True, True, True, 8, 2),
        ((32768,) + (1,) * 7, 32775, True, True, True, 8, 2),
    ],
)
@torch.inference_mode()
def test_folded_prefill(
    folded_precision,
    lengths,
    physical,
    packed,
    auto_cp,
    expect_folded,
    heads,
    qk_heads,
    monkeypatch,
):
    from pla.prefill.flashqla import _C

    prepare = _C.folded_coefficients
    calls = []

    def track(*args, **kwargs):
        calls.append(args[-1])
        return prepare(*args, **kwargs)

    monkeypatch.setattr(_C, "folded_coefficients", track)
    torch.manual_seed(20261118)
    q = torch.randn(1, physical, qk_heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, physical, heads, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    gates = torch.randn(physical, 2 * heads, device="cuda", dtype=torch.bfloat16)
    raw_a, raw_b = gates[:, :heads], gates[:, heads:]
    alog = torch.linspace(-2, -1, heads, device="cuda")
    bias = torch.zeros(heads, device="cuda")
    g = (-alog.exp() * F.softplus(raw_a.float() + bias))[None].contiguous()
    beta = raw_b.float().sigmoid()[None].contiguous()
    offsets = (0, *torch.tensor(lengths).cumsum(0).tolist())
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    slots = 2 * len(lengths) + 2
    indices = torch.arange(slots - 1, 1, -2, device="cuda", dtype=torch.int32)
    initial = torch.randn(slots, heads, 128, 128, device="cuda") * 0.01
    pool = initial.clone()
    valid = sum(lengths)
    kkt = ref_gdr.torch_kkt_fwd

    def bounded_kkt(k, g, beta, cu_seqlens, chunk_size):
        # Check the documented GEMM limit even on runtimes that split
        # oversized reference calls internally.
        assert cu_seqlens is None and k.size(0) == 1
        assert math.ceil(k.size(1) / chunk_size) * g.size(-1) <= _REFERENCE_GEMM_BATCH_LIMIT
        return kkt(k, g, beta, cu_seqlens, chunk_size)

    monkeypatch.setattr(ref_gdr, "torch_kkt_fwd", bounded_kkt)
    expected, expected_state = folded_prefill_reference(
        q, k, v, g, beta, offsets, initial[indices].transpose(-1, -2).contiguous(),
    )
    common = dict(
        initial_state=pool,
        initial_state_indices=indices,
        raw_cu_seqlens=offsets,
        auto_cp=auto_cp,
        output_final_state=True,
    )
    if packed:
        qk_width, v_width = qk_heads * 128, heads * 128
        width = 2 * qk_width + v_width
        storage = torch.full(
            (physical, width + 64), 0.125, device="cuda", dtype=q.dtype
        )
        qkv = storage[:, 17 : 17 + width]
        qkv[:, :qk_width].copy_(q.reshape(physical, qk_width))
        qkv[:, qk_width : 2 * qk_width].copy_(k.reshape(physical, qk_width))
        qkv[:, 2 * qk_width :].copy_(v.reshape(physical, v_width))
        inputs = (storage, gates, alog, bias)
        saved = [x.clone() for x in inputs]
        result = gdn_prefill_packed_qkv_fwd(
            qkv,
            raw_a,
            raw_b,
            alog,
            bias,
            cu,
            num_qk_heads=qk_heads,
            num_v_heads=heads,
            **common,
        )
    else:
        inputs = (q, k, v, g, beta)
        saved = [x.clone() for x in inputs]
        result = chunk_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu,
            normalize_qk=True,
            **common,
        )
    assert bool(calls) == expect_folded
    if calls:
        assert calls == [(heads, qk_heads) != (24, 8) or max(lengths) > 65536]
    assert_folded_close(result[2][:, :valid], expected)
    assert_folded_close(pool[indices], expected_state.transpose(-1, -2))
    assert torch.equal(result[4], pool[indices])
    assert torch.count_nonzero(result[2][:, valid:]) == 0
    unused = [i for i in range(slots) if i not in indices.tolist()]
    assert torch.equal(pool[unused], initial[unused])
    for value, before in zip(inputs, saved):
        assert torch.equal(value, before)


@pytest.mark.parametrize("metadata_kind", ["list", "cpu_tensor", "device_tensor"])
@torch.inference_mode()
def test_calibrated_prefill_host_offsets(folded_precision, metadata_kind, monkeypatch):
    from pla.prefill.flashqla import _C

    def unexpected_folded(*args, **kwargs):
        pytest.fail("8K 12/4 prefill must use the calibrated CP path")

    monkeypatch.setattr(_C, "folded_coefficients", unexpected_folded)
    tokens, heads, qk_heads = 8192, 12, 4
    q = torch.zeros(1, tokens, qk_heads, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(1, tokens, heads, 128, device="cuda", dtype=q.dtype)
    g = torch.zeros(1, tokens, heads, device="cuda")
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    pool = torch.zeros(1, heads, 128, 128, device="cuda")
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    host_offsets = [0, tokens]
    raw_cu = {
        "list": host_offsets,
        "cpu_tensor": torch.tensor(host_offsets, dtype=torch.int64, device="cpu"),
        "device_tensor": cu,
    }[metadata_kind]

    args = dict(cu_seqlens=cu, raw_cu_seqlens=raw_cu,
                initial_state=pool, initial_state_indices=indices)
    if metadata_kind == "device_tensor":
        with pytest.raises(ValueError, match="raw_cu_seqlens must be a 1-D CPU"):
            chunk_gated_delta_rule_fwd(q, q, v, g, g, **args)
    else:
        chunk_gated_delta_rule_fwd(q, q, v, g, g, **args)


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("heads,qk_heads", [(24, 8), (12, 4)])
@torch.inference_mode()
def test_folded_cp_preserves_early_history(folded_precision, fresh, heads, qk_heads):
    total = 32768
    q = torch.ones(1, total, qk_heads, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros(1, total, heads, 128, device="cuda", dtype=q.dtype)
    g = torch.full((1, total, heads), -1e-7, device="cuda")
    g[:, 64] = -12
    beta = torch.zeros_like(g)
    indices = torch.tensor([3], device="cuda", dtype=torch.int32)
    pool = torch.ones(5, heads, 128, 128, device="cuda")
    value = torch.linspace(8, 24, 128, device="cuda")[None].expand(heads, -1)
    if fresh:
        pool[3].zero_()
        k[:, 0, :, 0] = 1
        v[:, 0] = value
        beta[:, 0] = 1
    before = pool.clone()
    decay = g[0].double().cumsum(0).exp().float()
    expected_state = before[3] * decay[-1, :, None, None]
    if fresh:
        expected_state[:, :, 0] = value * decay[-1, :, None]
    result = chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        initial_state_indices=indices,
        cu_seqlens=torch.tensor([0, total], device="cuda", dtype=torch.int32),
        raw_cu_seqlens=(0, total),
        normalize_qk=True,
        output_final_state=False,
    )
    assert result[4] is None
    assert_folded_close(pool[3], expected_state)
    assert torch.equal(pool[[0, 1, 2, 4]], before[[0, 1, 2, 4]])


@pytest.mark.parametrize("heads,qk_heads,log_decay", [(12, 4, -7.5e-7), (24, 8, -4.5e-7)])
@torch.inference_mode()
def test_folded_cp_keeps_small_segment_decays(folded_precision, heads, qk_heads, log_decay):
    tokens, dimension = 32768, 128
    q = torch.zeros((1, tokens, qk_heads, dimension), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros((1, tokens, heads, dimension), device="cuda", dtype=q.dtype)
    g = torch.full((1, tokens, heads), log_decay, device="cuda", dtype=torch.float32)
    beta = torch.zeros_like(g)
    pool = torch.ones((3, heads, dimension, dimension), device="cuda", dtype=torch.float32)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    # With beta=0, the state decays independently of the Q/K and V inputs.
    # Each 39-CU CP segment decays by less than one BF16 rounding step near 1.
    factor = g[0, :, 0].double().sum().exp().float()
    expected = pool[1] * factor
    chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, cu_seqlens=cu, raw_cu_seqlens=(0, tokens),
        initial_state=pool, initial_state_indices=indices, auto_cp=True,
        output_final_state=False,
    )
    torch.testing.assert_close(pool[1], expected, rtol=5e-4, atol=2e-5)
    assert torch.all(pool[[0, 2]] == 1)


@pytest.mark.parametrize("heads,qk_heads", [(24, 8), (12, 4)])
@torch.inference_mode()
def test_folded_packed_weak_gate_decay(folded_precision, heads, qk_heads):
    tokens = 32768
    qkv = torch.zeros(tokens, (2 * qk_heads + heads) * D, device="cuda", dtype=torch.bfloat16)
    raw_a = torch.full((tokens, heads), -18, device="cuda", dtype=qkv.dtype)
    raw_b = torch.zeros_like(raw_a)
    alog = torch.full((heads,), 8.0, device="cuda")
    bias = torch.zeros_like(alog)
    pool = torch.ones((3, heads, D, D), device="cuda")
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    g_cum, _, _, _, _ = gdn_prefill_packed_qkv_fwd(
        qkv, raw_a, raw_b, alog, bias, cu,
        num_qk_heads=qk_heads, num_v_heads=heads,
        initial_state=pool, initial_state_indices=indices,
        raw_cu_seqlens=(0, tokens), output_final_state=False,
    )

    # The FP32 addition in log(1 + exp(-18)) rounds to one, but the
    # preserved per-token decay reduces the state to about 0.226 at 32K.
    gate = -math.exp(8) * math.log1p(math.exp(-18))
    torch.testing.assert_close(
        g_cum[0, -1], torch.full_like(g_cum[0, -1], CHUNK_SIZE * gate),
    )
    torch.testing.assert_close(
        pool[1], torch.full_like(pool[1], math.exp(tokens * gate)), rtol=5e-4, atol=2e-5,
    )
    assert torch.all(pool[[0, 2]] == 1)


@torch.inference_mode()
def test_folded_packed_fp64_guard_uses_fp32_beta(folded_precision, monkeypatch):
    from pla.prefill.flashqla import ops

    tokens, heads, qk_heads = 8192, 24, 8
    width = qk_heads * D
    qkv = torch.zeros((tokens, 2 * width + heads * D), device="cuda", dtype=torch.bfloat16)
    qkv[:, 0] = qkv[:, width] = 1
    qkv[:, D + 1] = qkv[:, width + D + 1] = 1
    qkv[:, 2 * width] = 0.125
    raw_a = torch.full((tokens, heads), -18, device="cuda", dtype=qkv.dtype)
    raw_b = torch.zeros_like(raw_a)
    raw_b[:, 0] = 32
    raw_b[:, 3] = 7
    alog = torch.full((heads,), 8.0, device="cuda")
    bias = torch.zeros_like(alog)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    slots = torch.tensor([1], device="cuda", dtype=torch.int32)
    pool = torch.ones((3, heads, D, D), device="cuda")

    def forward():
        return gdn_prefill_packed_qkv_fwd(
            qkv, raw_a, raw_b, alog, bias, cu,
            num_qk_heads=qk_heads, num_v_heads=heads,
            initial_state=pool, initial_state_indices=slots,
            raw_cu_seqlens=(0, tokens),
        )

    with patch.object(ops.flashqla_cute, "folded_coefficients",
                      wraps=ops.flashqla_cute.folded_coefficients) as folded, \
         patch.object(ops.flashqla_cute, "_precision_repair",
                      wraps=ops.flashqla_cute._precision_repair) as repair:
        _, _, output, _, state = forward()
        assert folded.call_count == repair.call_count == 1
        repair_beta = repair.call_args.args[3]
        assert repair_beta.dtype == torch.float32
        assert repair_beta[0, 0, 0] == 1
        assert repair_beta[0, 0, 3] < 1
        assert repair_beta[0, 0, 3].bfloat16() == 1
        assert repair.call_args.kwargs["flags"][0, 0]
    folded_pool = pool.clone()
    pool.fill_(1)
    monkeypatch.setenv("PLA_FLASHQLA_TC02_FOLDED", "0")
    _, _, original_output, _, original_state = forward()
    assert torch.equal(output[:, :, :3], original_output[:, :, :3])
    assert torch.equal(state[:, :3], original_state[:, :3])
    assert torch.all(folded_pool[[0, 2]] == 1)
    assert torch.all(pool[[0, 2]] == 1)


@pytest.mark.parametrize(
    "length,auto_cp", [(98304, True), (131072, True), (262144, True), (131072, False)]
)
@pytest.mark.parametrize("heads,qk_heads", [(24, 8), (12, 4)])
@torch.inference_mode()
def test_folded_long_weak_decay(folded_precision, length, auto_cp, heads, qk_heads):
    torch.manual_seed(20261118)
    q = torch.randn(1, length, qk_heads, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    k = (
        (torch.randn(1, 1, qk_heads, 128, device="cuda", dtype=q.dtype) * 0.1)
        .expand_as(q)
        .contiguous()
    )
    v = torch.randn(1, length, heads, 128, device="cuda", dtype=q.dtype) * 0.1
    g = torch.zeros(1, length, heads, device="cuda")
    beta = torch.full_like(g, 0.99)
    pool = torch.randn(3, heads, 128, 128, device="cuda") * 0.02
    before = pool.clone()
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    expected, state = folded_prefill_reference(
        q, k, v, g, beta, (0, length), pool[indices].transpose(-1, -2).contiguous(),
    )
    result = chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool,
        initial_state_indices=indices,
        cu_seqlens=torch.tensor([0, length], device="cuda", dtype=torch.int32),
        raw_cu_seqlens=(0, length),
        normalize_qk=True,
        auto_cp=auto_cp,
        output_final_state=False,
    )
    assert_folded_close(result[2], expected)
    assert_folded_close(pool[indices], state.transpose(-1, -2))
    assert result[4] is None
    assert torch.equal(pool[[0, 2]], before[[0, 2]])
