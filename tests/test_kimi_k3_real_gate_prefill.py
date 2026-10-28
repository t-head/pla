#!/usr/bin/env python3
"""Kimi-K3 layer-0 real-gating coverage for the PLA KDA prefill ABI.

The fixture contains the TP8 rank-0 head slice produced from the official
Kimi-K3 checkpoint at revision f831ab66814297da540d832a5235f8e904f29d06.
Its gate tensors were computed from real token embeddings through layer-0
RMSNorm and the f_a, f_b, and beta projections.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from pla.prefill.flashkdapro import flashkda_fwd


D = 128
H = 12
T = 14
LOWER_BOUND = -5.0
SCALE = D ** -0.5
TOKEN_IDS = (
    47,
    8156,
    16215,
    19624,
    472,
    3019,
    24980,
    72,
    1040,
    18,
    365,
    1772,
    4661,
    13,
)
FIXTURE = Path(__file__).with_name("kimi_k3_layer0_tp8_rank0_gate_14x12.pt")
FIXTURE_SHA256 = "87dc7c462606cd1711d5c29e63b6bba11cead733a8e6c6bca56bd03db1981ab9"
REFERENCE = (
    Path(__file__).parents[1]
    / "scripts"
    / "prefill"
    / "flashkdapro"
    / "private"
    / "test_fwd_e2e.py"
)


def _load_reference():
    spec = importlib.util.spec_from_file_location("flashkda_test_reference", REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a PPU device")
def test_kimi_k3_real_gate_prefill_matches_reference():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    gate_cpu = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    assert set(gate_cpu) == {"token_ids", "g", "beta", "A_log", "dt_bias"}
    assert tuple(gate_cpu["token_ids"].tolist()) == TOKEN_IDS
    assert gate_cpu["token_ids"].dtype == torch.int32
    assert gate_cpu["g"].shape == (T, H, D)
    assert gate_cpu["beta"].shape == (T, H)
    assert gate_cpu["A_log"].shape == (H,)
    assert gate_cpu["dt_bias"].shape == (H, D)
    assert gate_cpu["g"].dtype == torch.bfloat16
    assert gate_cpu["beta"].dtype == torch.bfloat16
    assert gate_cpu["A_log"].dtype == torch.float32
    assert gate_cpu["dt_bias"].dtype == torch.float32

    generator = torch.Generator().manual_seed(20261020)
    q_cpu = (torch.randn(T, H, D, generator=generator) * 0.1).to(torch.bfloat16)
    k_cpu = (torch.randn(T, H, D, generator=generator) * 0.1).to(torch.bfloat16)
    v_cpu = (torch.randn(T, H, D, generator=generator) * 0.1).to(torch.bfloat16)
    initial_cpu = (
        torch.randn(1, H, D, D, generator=generator) * 0.01
    ).to(torch.bfloat16)

    q = q_cpu.cuda()
    k = k_cpu.cuda()
    v = v_cpu.cuda()
    gate = {
        name: tensor.cuda()
        for name, tensor in gate_cpu.items()
        if name != "token_ids"
    }
    initial = initial_cpu.cuda()
    output = torch.empty(T, H, D, dtype=torch.bfloat16, device="cuda")
    final = torch.empty(1, H, D, D, dtype=torch.bfloat16, device="cuda")

    flashkda_fwd(
        q,
        k,
        v,
        gate["g"],
        gate["beta"],
        gate["A_log"],
        gate["dt_bias"],
        output,
        SCALE,
        LOWER_BOUND,
        initial,
        final,
    )
    torch.cuda.synchronize()

    reference = _load_reference()
    expected_output, expected_final = reference.ref_fwd(
        q_cpu,
        k_cpu,
        v_cpu,
        gate_cpu["g"],
        gate_cpu["beta"],
        gate_cpu["A_log"],
        gate_cpu["dt_bias"],
        SCALE,
        LOWER_BOUND,
        initial_cpu[0],
        T,
        H,
    )
    output_error = reference.err_ratio(output.cpu(), expected_output)
    state_error = reference.err_ratio(final[0].cpu(), expected_final)
    assert output_error < reference.TOL
    assert state_error < reference.TOL
