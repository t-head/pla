#!/usr/bin/env python3
"""Qwen3.8-Max layer-0 real-gating coverage for the PLA GDN prefill ABI.

The committed fixture is the audited 2092-token, 8-head prefix of the full
32768-token, 128-head corpus used for the tc01 dispatch sweep. The prefix is
bitwise identical to the original OperatorsForge checkpoint audit.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from pla.prefill.flashqla import chunk_gated_delta_rule_fwd
from pla.prefill.flashqla.private.cp_preprocess import CHUNK_SIZE, calc_cp_seqs


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
            assert (fused.call_args.args[7] is not None) == auto_cp
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
                # The None path must really use production CP, not pass by
                # silently disabling the optimization for this input.
                if not zero_state:
                    assert (fused.call_args.args[7] is not None) == auto_cp
            torch.testing.assert_close(
                result[4][0, :, 0].float(), expected_state[:, None].expand(heads, D),
                rtol=0.05, atol=0.0)
            assert not torch.count_nonzero(result[4][0, :, 1:])
            torch.testing.assert_close(
                result[2][0, -1].float(), expected_output[:, None].expand(heads, D),
                rtol=0.05, atol=0.0)


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
