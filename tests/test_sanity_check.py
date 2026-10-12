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

"""Centralized coverage of decode/prefill native sanity, pre-conversion metadata and the capture contract."""
import contextlib
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "pla_prefill_sanity_test_helper", _ROOT / "pla/prefill/_sanity.py")
sanity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sanity)


@pytest.fixture(autouse=True)
def enable_sanity(monkeypatch):
    monkeypatch.setenv("PLA_SANITY_CHECK", "1")


@pytest.fixture
def decode_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/PPU and compiled PLA extensions are required")


@pytest.fixture
def native(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/PPU is required")
    monkeypatch.setenv("PLA_SANITY_CHECK", "1")
    return importlib.import_module("pla.decode._C")


def inputs(lengths=(3, 2), columns=3):
    device = "cuda"
    dtype = torch.bfloat16
    tokens = sum(lengths)
    heads = 4
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    rows = len(lengths)
    return dict(
        A_log=torch.zeros(heads, device=device),
        a=torch.zeros(tokens, heads, dtype=dtype, device=device),
        dt_bias=torch.zeros(heads, dtype=dtype, device=device),
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=128**-0.5,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
        intermediate_states_buffer=None,
        intermediate_state_indices=None,
        retrieve_parent_token=None,
        q=torch.randn(1, tokens, heads, 128, dtype=dtype, device=device),
        k=torch.randn(1, tokens, heads, 128, dtype=dtype, device=device),
        v=torch.randn(1, tokens, heads, 128, dtype=dtype, device=device),
        b=torch.zeros(tokens, heads, dtype=dtype, device=device),
        initial_state_source=torch.zeros(
            rows * columns + 1, heads, 128, 128, device=device),
        initial_state_indices=torch.arange(
            1, rows * columns + 1, device=device, dtype=torch.int32
        ).reshape(rows, columns),
        cu_seqlens=torch.tensor(cu, device=device, dtype=torch.int32),
        cache_steps_or_num_accept=torch.ones(rows, device=device, dtype=torch.int32),
        disable_state_update=True,
        is_sglang=False,
    )


@pytest.mark.parametrize("defect", [
    "gate_tokens", "channel_stride", "slot_columns", "cu_backwards",
    "cu_end", "accepted_column", "state_slot", "index_dtype", "cpu_input",
])
@pytest.mark.parametrize("tunable", [False, True])
def test_native_rejects_invalid_verify_input(native, defect, tunable):
    args = inputs()
    if defect == "gate_tokens":
        args["a"] = args["a"][:-1]
    elif defect == "channel_stride":
        args["k"] = torch.empty(
            1, 5, 4, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    elif defect == "slot_columns":
        args["initial_state_indices"] = args["initial_state_indices"][:, :2]
    elif defect == "cu_backwards":
        args["cu_seqlens"] = torch.tensor([0, 4, 3], device="cuda", dtype=torch.int32)
    elif defect == "cu_end":
        args["cu_seqlens"][-1] = 6
    elif defect == "accepted_column":
        args["cache_steps_or_num_accept"][0] = 4
    elif defect == "state_slot":
        args["initial_state_indices"][0, 0] = args["initial_state_source"].size(0)
    elif defect == "index_dtype":
        args["initial_state_indices"] = args["initial_state_indices"].to(torch.float32)
    elif defect == "cpu_input":
        args["q"] = args["q"].cpu()
    if tunable:
        args.update(bv=16, wn=4, sn=4, hv_per_cta=1)
        op = native.fused_sigmoid_gating_delta_rule_forward_k_last_tunable
    else:
        op = native.fused_sigmoid_gating_delta_rule_forward_k_last
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        op(**args)


def test_valid_ragged_four_column_verify(native):
    args = inputs((4, 3), columns=4)
    out = native.fused_sigmoid_gating_delta_rule_forward_k_last(**args)
    assert out.shape == args["v"].shape
    assert torch.isfinite(out).all()


def test_empty_request_preserves_state(native):
    args = inputs((3, 0, 2))
    args["initial_state_source"][4:7].fill_(7.0)
    before = args["initial_state_source"][4:7].clone()
    native.fused_sigmoid_gating_delta_rule_forward_k_last(**args)
    torch.testing.assert_close(args["initial_state_source"][4:7], before, rtol=0, atol=0)


def test_capture_with_sanity_matches_eager(native):
    args = inputs()
    op = native.fused_sigmoid_gating_delta_rule_forward_k_last
    saved = args["initial_state_source"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op(**args)
    torch.cuda.current_stream().wait_stream(stream)
    args["initial_state_source"].copy_(saved)
    expected = op(**args).clone()
    expected_state = args["initial_state_source"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(**args)
    args["initial_state_source"].copy_(saved)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(args["initial_state_source"], expected_state, rtol=1e-2, atol=1e-3)


def launch(native, args, tunable):
    if tunable:
        return native.fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
            **args, bv=16, wn=4, sn=4, hv_per_cta=1)
    return native.fused_sigmoid_gating_delta_rule_forward_k_last(**args)


# Other decode operator families.
def zeros(*shape, dtype=torch.bfloat16):
    return torch.zeros(shape, device="cuda", dtype=dtype)


def indices(values, dtype=torch.int64):
    return torch.tensor(values, device="cuda", dtype=dtype)


def mega_case():
    # H=12 is shared by the PPU and reference dispatch tables (ref excludes 6).
    heads, batch, slots, dim = 12, 2, 4, 12 * 128
    args = dict(
        x=zeros(batch, 3 * dim + 7)[:, :3 * dim],
        weight=zeros(3, 4, dim, dtype=torch.float32), bias=None,
        conv_state=zeros(slots, 3, 3 * dim).transpose(1, 2),
        raw_g=zeros(1, batch, heads, 128),
        raw_beta=zeros(1, batch, heads + 1)[:, :, :heads],
        a_log=zeros(heads, dtype=torch.float32),
        dt_bias=zeros(dim, dtype=torch.float32),
        state_indices=indices([1, 2], torch.int32),
        state=zeros(slots, heads, 128, 128, dtype=torch.float32),
        out=zeros(1, batch, heads, 128), lower_bound=-5.0,
        output_gate=None, norm_weight=None, norm_eps=1e-5,
    )
    op = importlib.import_module("pla.decode._C").fused_kda_decode_mega_forward
    return op, args


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("impl", ["ppu", "ref"])
@pytest.mark.parametrize("defect", ["slot", "duplicate", "alignment", "scalar"])
def test_mega_rejects_unsafe_inputs(monkeypatch, impl, defect):
    monkeypatch.setenv("KDA_DECODE_IMPL", impl)
    op, args = mega_case()
    if defect == "slot":
        args["state_indices"][0] = args["state"].size(0)
    elif defect == "duplicate":
        args["state_indices"].fill_(1)
    elif defect == "alignment":
        state = args["state"]
        args["state"] = zeros(state.numel() + 1, dtype=state.dtype)[1:].view_as(state)
    else:
        args["lower_bound"] = float("nan")
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        op(**args)


def recurrent_case():
    args = dict(
        q=zeros(1, 2, 4, 128), k=zeros(1, 2, 4, 128), v=zeros(1, 2, 4, 128),
        raw_g=zeros(1, 2, 4, 128), raw_beta=zeros(1, 2, 4),
        A_log=zeros(4, dtype=torch.float32),
        dt_bias=zeros(4, 128, dtype=torch.float32), lower_bound=-5.0,
        initial_state=zeros(3, 4, 128, 128, dtype=torch.float32),
        cu_seqlens=indices([0, 2]), ssm_state_indices=indices([[1, 2]]),
        num_accepted_tokens=indices([1]),
    )
    op = importlib.import_module("pla.decode._C").fused_recurrent_kda
    return op, args


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("defect", ["overflow", "raw_shape", "output_shape"])
def test_recurrent_checks_before_conversion(defect):
    op, args = recurrent_case()
    if defect == "overflow":
        args["ssm_state_indices"][0, 0] = 2**32 + 1
    elif defect == "raw_shape":
        args["raw_g"] = args["raw_g"].reshape(2, 4, 128)
    else:
        args["out"] = zeros(1, 1, 4, 128)
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        op(**args)


def verify_case():
    args = [
        zeros(1, 4, 1, 128), zeros(1, 4, 1, 128), zeros(1, 4, 1, 128),
        zeros(1, 4, 1, 128), zeros(1, 4, 1),
        zeros(128, 4, dtype=torch.float32), zeros(128, 4, dtype=torch.float32),
        zeros(128, 4, dtype=torch.float32),
        zeros(3, 3, 128).transpose(1, 2),
        zeros(3, 3, 128).transpose(1, 2),
        zeros(3, 3, 128).transpose(1, 2),
        zeros(1, dtype=torch.float32), zeros(128, dtype=torch.float32),
        zeros(3, 1, 128, 128, dtype=torch.float32), zeros(1, 4, 1, 128),
        zeros(2, 2, 128, 3), zeros(2, 2, 128, 3), zeros(2, 2, 128, 3),
        zeros(3, 1, 2, 128), zeros(3, 1, 2, 128),
        zeros(3, 1, 2, 128, dtype=torch.float32), zeros(3, 1, 2, dtype=torch.float32),
        indices([1, 2]), indices([0, 1]), indices([0, 2, 4]), 128**-0.5, -5.0,
    ]
    op = importlib.import_module("pla.decode.kda_mtp_sglang._C").verify_dspark
    return op, args


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("defect", ["cu", "scratch", "slot", "overflow", "duplicate"])
def test_verify_metadata_failure(defect):
    op, args = verify_case()
    if defect == "cu":
        args[24] = indices([0, 1, 4])
    elif defect == "scratch":
        args[23][0] = 2
    elif defect == "slot":
        args[22][0] = 3
    elif defect == "overflow":
        args[22][0] = 2**32 + 1
    else:
        args[23].fill_(0)
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        op(*args)


def fold_case():
    args = [
        zeros(1, 3, 1, 128, 128, dtype=torch.float32),
        zeros(1, 3, 1, 2, 128), zeros(1, 3, 1, 2, 128),
        zeros(1, 3, 1, 2, 128, dtype=torch.float32),
        zeros(1, 3, 1, 2, dtype=torch.float32),
        indices([0, 1]), indices([2, 1]), None, None, 1, -1,
    ]
    op = importlib.import_module("pla.decode.kda_mtp_sglang._C").fold_all_layers
    return op, args


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("defect", ["accepted", "slot", "track_step", "track_collision"])
def test_fold_metadata_failure(defect):
    op, args = fold_case()
    if defect == "accepted":
        args[6][0] = 3
    elif defect == "slot":
        args[5][0] = 3
    elif defect == "track_step":
        args[7], args[8] = indices([2, -1]), indices([2, -1])
    else:
        args[7], args[8] = indices([1, -1]), indices([0, -1])
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        op(*args)


def scatter_case():
    # Neighbouring source windows intentionally overlap; that is the supported
    # rollback view, while separate destination slots must not overlap.
    source = torch.arange(1 * 2 * 128 * 4, device="cuda", dtype=torch.float32)
    source = source.reshape(1, 2, 128, 4).unfold(-1, 3, 1).transpose(2, 3)
    args = [zeros(1, 3, 128, 3, dtype=torch.float32), source,
            indices([0, 1]), indices([1, 0])]
    op = importlib.import_module("pla.decode.kda_mtp_sglang._C").conv_window_scatter
    return op, args


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("defect", ["step", "slot", "overflow", "duplicate"])
def test_scatter_metadata_failure(defect):
    op, args = scatter_case()
    if defect == "step":
        args[3][0] = 2
    elif defect == "slot":
        args[2][0] = 3
    elif defect == "overflow":
        args[2][0] = 2**32
    else:
        args[2].fill_(0)
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        op(*args)


@pytest.mark.usefixtures("decode_device")
def test_scatter_preserves_valid_overlapping_source():
    op, args = scatter_case()
    op(*args)
    torch.testing.assert_close(args[0][0, 0], args[1][0, 0, 1], rtol=0, atol=0)
    torch.testing.assert_close(args[0][0, 1], args[1][0, 1, 0], rtol=0, atol=0)


@pytest.mark.usefixtures("decode_device")
def test_fold_ignores_unaccepted_ring_tail():
    op, args = fold_case()
    args[6].fill_(1)
    for ring in args[1:5]:
        ring[:, :, :, 1:].fill_(float("nan"))
    args[3][:, :, :, :1].fill_(float("-inf"))
    op(*args)
    assert torch.isfinite(args[0]).all()


@pytest.mark.usefixtures("decode_device")
def test_verify_ignores_padding_values():
    op, args = verify_case()
    args[22][1] = -1
    args[23][1] = -1
    for value in args[:5]:
        value[:, 2:].fill_(float("nan"))
    args[3][:, :2].fill_(float("-inf"))
    args[4][:, :2].fill_(float("inf"))
    op(*args)
    assert torch.isfinite(args[14]).all()


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("impl", ["ppu", "ref"])
def test_mega_preserves_shared_null_zero_convention(monkeypatch, impl):
    # The mega ABI has no is_sglang flag. Duplicate zero slots remain accepted
    # for vLLM graph padding; this does not diagnose SGLang real-slot-zero races.
    monkeypatch.setenv("KDA_DECODE_IMPL", impl)
    op, args = mega_case()
    args["state_indices"].zero_()
    op(**args)


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("factory", [mega_case, recurrent_case])
def test_raw_gate_logits_allow_saturating_infinities(factory):
    op, args = factory()
    args["raw_g"].fill_(float("-inf"))
    args["raw_beta"].fill_(float("inf"))
    op(**args)


def v_last_case():
    args = [
        zeros(1, dtype=torch.float32), zeros(2, 1), zeros(1), 1.0, 20.0,
        zeros(1, 2, 1, 128), zeros(1, 2, 1, 128), zeros(1, 2, 1, 128),
        zeros(2, 1), zeros(3, 1, 128, 128, dtype=torch.float32),
        zeros(1, 1, 2, 1, 128), indices([0, 1], torch.int32),
        indices([0, 1, 2], torch.int32),
        128**-0.5, 1, 2, 1, 1, 128, 128, True, True, True,
    ]
    op = importlib.import_module("pla.decode.gdn.v_last._C").fused_sigmoid_gating_delta_rule_forward
    return op, args


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("defect", ["length", "slot", "dtype"])
def test_v_last_native_failure(defect):
    op, args = v_last_case()
    if defect == "length":
        args[12] = indices([0, 2, 2], torch.int32)
    elif defect == "slot":
        args[11][0] = 3
    else:
        args[1] = args[1].float()
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        op(*args)


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("factory", [
    mega_case, recurrent_case, verify_case, fold_case, scatter_case, v_last_case,
])
def test_valid_capture_has_no_d2h(factory):
    op, args = factory()

    def invoke():
        return op(**args) if isinstance(args, dict) else op(*args)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    graph.replay()
    torch.cuda.synchronize()


@pytest.mark.usefixtures("decode_device")
def test_ref_capture_has_no_d2h(monkeypatch):
    monkeypatch.setenv("KDA_DECODE_IMPL", "ref")
    test_valid_capture_has_no_d2h(mega_case)


@pytest.mark.usefixtures("decode_device")
@pytest.mark.parametrize("factory", [mega_case, v_last_case])
def test_launch_uses_tensor_device(factory):
    if torch.cuda.device_count() < 2:
        pytest.skip("two devices are required")
    with torch.cuda.device(1):
        op, args = factory()
    with torch.cuda.device(0):
        op(**args) if isinstance(args, dict) else op(*args)
    torch.cuda.synchronize(1)


# Prefill metadata and native entries.
@pytest.mark.parametrize("values,capacity", [
    ([1, 3], 3),
    ([0, 4, 3], 8),
    ([0, 9], 8),
    ([0, 2**32 + 4, 8], 8),
    ([0, -1, 8], 8),
    ([0], 8),
])
def test_offsets_reject_invalid_values_before_narrowing(values, capacity):
    with pytest.raises(AssertionError, match="PLA_SANITY_CHECK"):
        sanity.offsets(torch.tensor(values, dtype=torch.int64), capacity)


def test_offsets_reject_wrong_dtype_before_conversion():
    with pytest.raises(AssertionError, match="int32 or int64"):
        sanity.offsets(torch.tensor([0.0, 8.0]), 8)


def test_offsets_compare_supplied_host_metadata():
    with pytest.raises(AssertionError, match="must equal device"):
        sanity.offsets(torch.tensor([0, 3, 8], dtype=torch.int32), 8, (0, 4, 8))


def test_offsets_allow_padding_empty_sequences_and_strided_metadata():
    cu = torch.tensor([0, -99, 3, -99, 3, -99, 7, -99], dtype=torch.int64)[::2]
    assert sanity.offsets(cu, 8, (0, 3, 3, 7)) == (0, 3, 3, 7)


def test_capture_skips_value_read_but_keeps_static_validation(monkeypatch):
    monkeypatch.setattr(sanity, "capturing", lambda _: True)
    def forbidden_cpu(*args, **kwargs):
        raise RuntimeError("unexpected D2H")
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    cu = torch.tensor([0, 8], dtype=torch.int64)
    assert sanity.offsets(cu, 8, (0, 8)) == (0, 8)
    with pytest.raises(AssertionError, match="token capacity"):
        sanity.offsets(cu, 8, (0, 9))
    with pytest.raises(AssertionError, match="during CUDA graph capture"):
        sanity.require_host_read(cu, "test planning")


def test_failed_capture_probe_fails_before_value_reads(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    def failed_probe():
        raise RuntimeError("probe failed")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", failed_probe)
    with pytest.raises(AssertionError, match="cannot determine"):
        sanity.capturing(SimpleNamespace(is_cuda=True, device="cuda:0"))


def test_disabled_path_does_not_touch_inputs_or_cuda(monkeypatch):
    monkeypatch.delenv("PLA_SANITY_CHECK")
    assert sanity.offsets(object(), -1, (9, 1)) == (9, 1)
    sanity.require_host_read(object(), "unused")


@pytest.fixture
def prefill_native():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/PPU")
    from pla.prefill.flashqla import _C as qla
    from pla.prefill.flashkdapro import _C as kda
    return qla, kda


def _qla_inputs():
    T, H, Hg = 64, 4, 2
    q = torch.zeros((1, T, Hg, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros((1, T, H, 128), device="cuda", dtype=torch.bfloat16)
    a = torch.zeros((1, T, H, 32), device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros((1, T, H), device="cuda")
    cu = torch.tensor([0, 32, T], device="cuda", dtype=torch.int32)
    chunks = torch.tensor([[0, 0], [1, 0]], device="cuda", dtype=torch.int32)
    return q, v, a, gate, cu, chunks


@pytest.mark.parametrize("case", ["cu", "duplicate_chunk", "missing_chunk", "gate_stride"])
def test_qla_chunk_contract(prefill_native, case):
    qla, _ = prefill_native
    _, _, _, gate, cu, chunks = _qla_inputs()
    if case == "cu":
        cu = torch.tensor([0, 65, 64], device="cuda", dtype=torch.int32)
    elif case == "duplicate_chunk":
        chunks = torch.tensor([[0, 0], [0, 0]], device="cuda", dtype=torch.int32)
    elif case == "missing_chunk":
        chunks = chunks[:1]
    else:
        gate = torch.empty((1, 64, 8), device="cuda")[:, :, :4]
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        qla.chunk_local_cumsum(gate, cu, chunks)


def test_qla_kkt_checks_beta_shape(prefill_native):
    qla, _ = prefill_native
    q, _, _, gate, cu, chunks = _qla_inputs()
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        qla.kkt_solve(q, gate[:, :1], cu, 4, chunks)


def test_qla_qk_normalization_checks_key_shape(prefill_native):
    qla, _ = prefill_native
    q, _, _, gate, cu, chunks = _qla_inputs()
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        qla.chunk_local_cumsum_qk_norm(gate, cu, chunks, q, q[:, :1])


def test_qla_raw_gate_checks_valid_token_endpoint(prefill_native):
    qla, _ = prefill_native
    q, _, _, _, cu, chunks = _qla_inputs()
    a = torch.zeros((64, 4), device="cuda", dtype=torch.bfloat16)
    params = torch.zeros(4, device="cuda")
    with pytest.raises(RuntimeError, match="valid_num_tokens must match"):
        qla.chunk_local_cumsum_qk_norm_gated(a, a, params, params, cu, chunks, q, q, 63)


def test_qla_warmup_rejects_partial_unmasked_segment(prefill_native):
    qla, _ = prefill_native
    _, _, _, gate, _, _ = _qla_inputs()
    cu = torch.tensor([0, 31, 64], device="cuda", dtype=torch.int32)
    mask = torch.tensor([0, 1], device="cuda", dtype=torch.int8)
    with pytest.raises(RuntimeError, match="full 32-token chunks"):
        qla.get_warmup_chunks(gate, mask, cu)


def test_qla_prepare_rejects_warmup_beyond_segment(prefill_native):
    qla, _ = prefill_native
    q, v, a, gate, cu, _ = _qla_inputs()
    warmup = torch.full((2, 4), 2, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="num_warmup_chunks out of range"):
        qla.prepare_h(q, v, a, gate, gate, cu, warmup, 4, 2)


def test_qla_correct_h0_rejects_empty_raw_mapping(prefill_native):
    qla, _ = prefill_native
    state = torch.zeros((2, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros((2, 4), device="cuda", dtype=torch.int8)
    mapping = torch.tensor([0, 0, 2], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="at least one CP segment"):
        qla.correct_h0(state, state, mask, mapping, 2)


@pytest.mark.parametrize("case", ["cp_map", "duplicate_state", "scale", "host_endpoint"])
def test_qla_fused_metadata(prefill_native, case):
    qla, _ = prefill_native
    q, v, a, gate, cu, _ = _qla_inputs()
    pool = torch.zeros((2, 4, 128, 128), device="cuda")
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    mapping = None
    scale = 0.125
    valid = 64
    if case == "cp_map":
        mapping = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    elif case == "duplicate_state":
        indices.zero_()
    elif case == "scale":
        scale = 1e300
    else:
        valid = 63
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        qla.fused_fwd(q, q, v, a, gate, gate, cu, mapping, cu, valid,
                      4, 2, scale, None, pool, indices)


@pytest.mark.parametrize("case", ["k_shape", "q_stride", "cu_overflow", "state_slot", "host_cu", "scale"])
def test_kda_native_contract(prefill_native, case):
    _, kda = prefill_native
    q = torch.zeros((32, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = q.clone()
    v = q.clone()
    g = q.clone()
    beta = torch.zeros((32, 4), device="cuda", dtype=torch.bfloat16)
    A_log = torch.zeros(4, device="cuda")
    dt_bias = torch.zeros((4, 128), device="cuda")
    out = torch.empty_like(q)
    cu = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
    pool = torch.zeros((2, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    raw, scale = (0, 16, 32), 0.125
    if case == "k_shape":
        k = k[:1]
    elif case == "q_stride":
        q = torch.zeros((32, 8, 128), device="cuda", dtype=torch.bfloat16)[:, :4]
    elif case == "cu_overflow":
        cu = torch.tensor([0, 2**32 + 16, 32], device="cuda", dtype=torch.int64)
    elif case == "state_slot":
        indices[1] = 2
    elif case == "host_cu":
        raw = (0, 15, 32)
    else:
        scale = 1e300
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        kda.flashkda_fwd(q, k, v, g, beta, A_log, dt_bias, out, scale, -5.0,
                         pool, pool, cu, indices, raw)


@pytest.mark.parametrize("case", ["cu_overflow", "host_cu", "state_slot", "duplicate_state", "partial_alias"])
def test_kda_tc01_fla_preserves_value_checks_before_launch(case):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/PPU")
    from pla.prefill.flashkdapro._tc01_fla import _C as fla

    q = torch.zeros((32, 4, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    beta = torch.zeros((32, 4), device="cuda", dtype=torch.bfloat16)
    A_log = torch.zeros(4, device="cuda")
    dt_bias = torch.zeros((4, 128), device="cuda")
    cu = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
    state = torch.zeros((3, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    initial = final = state[:2]
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    raw = None
    if case == "cu_overflow":
        cu = torch.tensor([0, 2**32 + 16, 32], device="cuda", dtype=torch.int64)
    elif case == "host_cu":
        raw = (0, 15, 32)
    elif case == "state_slot":
        indices[1] = 2
    elif case == "duplicate_state":
        indices.zero_()
    else:
        final = state[1:]
        indices = None
    # Every illegal input must be rejected before arch initialization or kernel launch.
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        fla.forward(q, q, q, q, beta, A_log, dt_bias, out, 0.125, -5.0,
                    initial, final, cu, None, indices, raw)


@pytest.mark.parametrize("entry", ["flashkda_prepare", "flashkda_cp_stages"])
def test_kda_private_stages_check_before_launch(prefill_native, entry):
    _, kda = prefill_native
    q = torch.zeros((32, 4, 128), device="cuda", dtype=torch.bfloat16)
    bad_gate = q[:1]
    beta = torch.zeros((32, 4), device="cuda", dtype=torch.bfloat16)
    A_log = torch.zeros(4, device="cuda")
    dt_bias = torch.zeros((4, 128), device="cuda")
    with pytest.raises(RuntimeError, match="PLA_SANITY_CHECK"):
        if entry == "flashkda_prepare":
            kda.flashkda_prepare(q, q, bad_gate, beta, A_log, dt_bias, 0.125, -5.0)
        else:
            kda.flashkda_cp_stages(q, q, q, bad_gate, beta, A_log, dt_bias, 0.125, -5.0)


def test_kda_host_cp_plan_checks_cpu_metadata_copy(prefill_native):
    _, kda = prefill_native
    cu = torch.tensor([0, 16, 32], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="must equal CPU cu_seqlens"):
        kda.flashkda_cp_plan(32, 4, cu, (0, 15, 32))


def test_qla_cumsum_capture_has_no_host_value_reads(prefill_native):
    qla, _ = prefill_native
    _, _, _, gate, cu, chunks = _qla_inputs()
    gate.fill_(-0.1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = qla.chunk_local_cumsum(gate, cu, chunks)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = qla.chunk_local_cumsum(gate, cu, chunks)
    graph.replay()
    torch.testing.assert_close(actual, expected)


def test_kda_capture_accepts_strided_beta_and_padded_tokens(prefill_native, monkeypatch):
    _, kda = prefill_native
    monkeypatch.setenv("PLA_FLASHKDA_CP", "0")
    q = torch.zeros((32, 4, 128), device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros((32, 8), device="cuda", dtype=torch.bfloat16)[:, :4]
    A_log = torch.zeros(4, device="cuda")
    dt_bias = torch.zeros((4, 128), device="cuda")
    cu = torch.tensor([0, 15, 29], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    def run():
        kda.flashkda_fwd(q, q, q, q, beta, A_log, dt_bias, out, 0.125, -5.0,
                         None, None, cu, None, (0, 15, 29))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    graph.replay()
    torch.testing.assert_close(out, expected)
    assert torch.count_nonzero(out[29:]).item() == 0


@pytest.mark.parametrize("entry", ["prepare", "fwd", "cp_stages"])
@pytest.mark.parametrize("sanity_enabled", [False, True])
def test_kda_uses_tensor_device_and_restores_callers_device(
    prefill_native, monkeypatch, entry, sanity_enabled,
):
    _, kda = prefill_native
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA/PPU devices")
    monkeypatch.setenv("PLA_SANITY_CHECK", "1" if sanity_enabled else "0")
    monkeypatch.setenv("PLA_FLASHKDA_CP", "2")
    monkeypatch.setenv("PLA_FLASHKDA_CP_MULT", "2")
    monkeypatch.setenv("PLA_FLASHKDA_CP_SEG_TILES", "1")
    target, caller = 1, 0
    with torch.cuda.device(target):
        q = torch.randn((64, 4, 128), device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn_like(q) * 0.1
        g = torch.zeros_like(q)
        beta = torch.zeros((64, 4), device="cuda", dtype=torch.bfloat16)
        A_log = torch.zeros(4, device="cuda")
        dt_bias = torch.zeros((4, 128), device="cuda")
        cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)

    def run():
        if entry == "prepare":
            return kda.flashkda_prepare(q, q, g, beta, A_log, dt_bias, 0.125, -5.0, cu)
        if entry == "cp_stages":
            return kda.flashkda_cp_stages(
                q, q, v, g, beta, A_log, dt_bias, 0.125, -5.0,
                None, cu, (0, 64))
        out = torch.empty_like(q)
        kda.flashkda_fwd(q, q, v, g, beta, A_log, dt_bias, out, 0.125, -5.0,
                         None, None, cu, None, (0, 64))
        return (out,)

    with torch.cuda.device(target):
        expected = run()
    torch.cuda.synchronize(target)
    with torch.cuda.device(caller):
        actual = run()
        assert torch.cuda.current_device() == caller
    torch.cuda.synchronize(target)
    for result, reference in zip(actual, expected):
        assert result.device.index == target
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("sanity_enabled", [False, True])
def test_kda_prepare_cold_capture_requires_eager_layout_probe(prefill_native, sanity_enabled):
    # The process-wide native cache may already be warm from other tests. A
    # fresh interpreter makes the first prepare call occur inside capture.
    script = textwrap.dedent("""
        import torch
        from pla.prefill.flashkdapro import _C as kda

        q = torch.zeros((32, 4, 128), device="cuda", dtype=torch.bfloat16)
        beta = torch.zeros((32, 4), device="cuda", dtype=torch.bfloat16)
        A_log = torch.zeros(4, device="cuda")
        dt_bias = torch.zeros((4, 128), device="cuda")
        cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                kda.flashkda_prepare(q, q, q, beta, A_log, dt_bias, 0.125, -5.0, cu)
        except RuntimeError as error:
            assert "KR layout probe is cold during CUDA graph capture" in str(error), str(error)
            assert "run flashkda_prepare once in eager mode" in str(error), str(error)
        else:
            raise AssertionError("cold prepare capture must request eager warmup")
        # Failing before allocation/probe/D2H leaves the CUDA context usable;
        # eager initialization can subsequently observe and cache the real ABI.
        outputs = kda.flashkda_prepare(q, q, q, beta, A_log, dt_bias, 0.125, -5.0, cu)
        torch.cuda.synchronize()
        assert len(outputs) == 6
        print("cold capture rejected before probe; eager warmup succeeded")
    """)
    environment = os.environ.copy()
    environment["PLA_SANITY_CHECK"] = "1" if sanity_enabled else "0"
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=_ROOT, env=environment,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
