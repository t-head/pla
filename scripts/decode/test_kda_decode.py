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

"""KDA (Kimi Delta Attention) unit tests for the PLA PPU CUDA op.

Validates ``pla.decode.kda.fused_recurrent_kda`` against the vendored pure-PyTorch KDA
reference. This op is only used for **speculative decoding** (plain decode goes
through the packed kernels), so the cases below cover the two spec layouts:

  1. **vLLM spec decode** -- 2D ``ssm_state_indices [N, S]`` plus
     ``num_accepted_tokens``: the recurrence starts from the last accepted
     token's slot and every timestep commits its state into its own slot.
  2. **sglang spec decode** -- 1D ``ssm_state_indices [N]``: the recurrence runs
     over the draft tokens and commits the final state back once, at the end.

Each layout also covers its framework's padding convention, which is the one
place where the two genuinely disagree and which ``is_sglang`` selects:

  * vLLM (``is_sglang=False``): ``BlockPool`` reserves block 0, so slot ``<= 0``
    is CUDA-graph padding.
  * sglang (``is_sglang=True``): ``MambaPool`` has an extra scratch row, so slot
    0 is valid and ``PAD_SLOT_ID`` is -1.

Provenance / adaptation notes
------------------------------
* The symbol under test, ``fused_recurrent_kda``, is imported from *this*
  workspace's ``pla`` package (the PPU CUDA implementation), and its signature
  mirrors vLLM's Triton
  ``vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py``.
* Golden references (``naive_recurrent_kda`` / ``naive_kda_gate``) are vendored
  in ``scripts/decode/_kda_reference.py``; the per-framework slot bookkeeping is
  replicated in ``_reference`` below.
* This workspace's kernel only implements the **fused-gate mode** (raw gate plus
  ``A_log`` / ``dt_bias``, decay and beta sigmoid computed inside the kernel) and
  is instantiated for **head_dim K == V == 128** with fp16/bf16 activations only.

Run (on the PPU server, inside the container):
    SAIL_PLA_CUDA=1 CUDA_VISIBLE_DEVICES=1 python scripts/decode/test_kda_decode.py

Run it as a plain script, not via ``python -m`` with ``PYTHONPATH=.``: that
puts the repo root on sys.path, where the source ``pla/`` package shadows the
installed one and has no compiled ``_C``.
"""

import os
import sys

import torch
import torch.nn.functional as F

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    import pytest
except ImportError:  # allow running via __main__ without pytest
    pytest = None

from pla.decode.kda import fused_recurrent_kda

from _kda_reference import assert_close, naive_kda_gate, naive_recurrent_kda

device = "cuda"
D = 128  # PPU kernel is instantiated for head_dim K == V == 128 only


def _params(param_list, ids_fmt):
    """Parametrize decorator that degrades to a no-op without pytest."""
    if pytest is None:
        def deco(fn):
            fn._param_list = param_list
            return fn
        return deco
    return pytest.mark.parametrize(
        ids_fmt[0],
        [pytest.param(*t, id=ids_fmt[1].format(*t)) for t in param_list],
    )


def _make_gate_params(H, dt_bias_dtype=torch.float32):
    """A_log and dt_bias. dt_bias is fp32: that is the only dtype the KDA gate
    is instantiated for, so a non-default value is only useful for negative tests."""
    A_log = torch.log(torch.empty(H, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device).to(dt_bias_dtype)
    return A_log, dt_bias


def _make_inputs(total, H, dtype):
    q = torch.rand(1, total, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total, H, D, dtype=dtype, device=device)
    raw_g = torch.randn(1, total, H, D, dtype=dtype, device=device)
    raw_beta = torch.randn(1, total, H, dtype=dtype, device=device)
    return q, k, v, raw_g, raw_beta


def _ref_step(q, k, v, raw_g, raw_beta, A_log, dt_bias, state_kfirst, scale):
    """One reference timestep. state_kfirst is [1, H, K, V]; returns (o, state)."""
    g = naive_kda_gate(raw_g.float(), A_log, dt_bias)
    return naive_recurrent_kda(
        q=F.normalize(q.float(), p=2, dim=-1),
        k=F.normalize(k.float(), p=2, dim=-1),
        v=v.clone(),
        g=g,
        beta=raw_beta.float().sigmoid(),
        scale=scale,
        initial_state=state_kfirst,
        output_final_state=True,
    )


def _reference(q, k, v, raw_g, raw_beta, A_log, dt_bias, pool,
               cu, indices, num_accepted, scale, is_sglang):
    """Replicate the framework's slot bookkeeping over the naive KDA recurrence.

    ``pool`` is updated in place (k-last ``[slots, H, V, K]``); returns the
    reference output. A 2D ``indices`` commits one state per timestep (vLLM spec
    decode), a 1D one commits a single state at the end (sglang). Padded requests
    contribute zeros and no writes.
    """
    per_token_slots = indices.dim() == 2
    # sglang: PAD_SLOT_ID = -1, slot 0 is a valid scratch row.
    # vLLM:   NULL_BLOCK_ID = 0, so anything <= 0 is padding.
    def invalid(slot):
        return slot < 0 if is_sglang else slot <= 0

    out = torch.zeros_like(v)
    for i in range(cu.numel() - 1):
        bos, eos = int(cu[i].item()), int(cu[i + 1].item())
        if eos == bos:
            continue

        if per_token_slots:
            col = int(num_accepted[i].item()) - 1 if num_accepted is not None else 0
            start_slot = int(indices[i, max(col, 0)].item())
        else:
            start_slot = int(indices[i].item())
        if invalid(start_slot):
            continue

        # pool is k-last [slots,H,V,K]; the naive reference is k-first [.,H,K,V]
        state = pool[start_slot].transpose(-1, -2).contiguous().unsqueeze(0)
        for t in range(eos - bos):
            idx = bos + t
            o_t, state = _ref_step(
                q[:, idx:idx + 1], k[:, idx:idx + 1], v[:, idx:idx + 1],
                raw_g[:, idx:idx + 1], raw_beta[:, idx:idx + 1],
                A_log, dt_bias, state, scale,
            )
            out[:, idx:idx + 1] = o_t.to(out.dtype)
            if per_token_slots:
                dst = int(indices[i, t].item())
                if not invalid(dst):
                    pool[dst] = state.squeeze(0).transpose(-1, -2)
        if not per_token_slots:
            pool[start_slot] = state.squeeze(0).transpose(-1, -2)

    return out


def _run_case(N, S, H, dtype, is_sglang, pad_last, seed,
             seq_lens=None, page_pad=0):
    """Build a spec-decode batch, run kernel vs reference, and compare.

    ``page_pad > 0`` simulates vLLM's page-packed state pool: each slot
    has ``page_pad`` extra elements (representing conv_state) so that
    ``stride(0) > H * V * K``.  This exercises the per-timestep state
    write stride fix (must use ``h0_slot_stride``, not the hardcoded
    ``HV * V * K``).
    """
    torch.manual_seed(seed)
    scale = D ** -0.5
    seq_lens = [S] * N if seq_lens is None else list(seq_lens)
    assert len(seq_lens) == N and max(seq_lens) <= S
    total = sum(seq_lens)

    if is_sglang:
        # sglang: slot 0 is a legitimate scratch row, so the table starts at 0
        # and the pad id is -1.
        max_slots = total + 1
        perm = torch.randperm(max_slots, device=device)
        # Pin request 0 to slot 0 so every sglang case actually exercises the
        # "slot 0 is valid" rule -- that is the whole point of is_sglang=True,
        # and vLLM would zero this request out instead. Swap rather than
        # overwrite: overwriting can leave slot 0 assigned to a second request as
        # well, and duplicated slots make the comparison meaningless because the
        # reference commits them in sequence while the kernel runs them at once.
        zero_pos = int((perm == 0).nonzero()[0, 0].item())
        perm[zero_pos] = perm[0].clone()
        perm[0] = 0
        indices = perm[:N].int()
        pad_value = -1
    else:
        # vLLM: slot 0 is reserved as null_block, so real slots start at 1.
        table_slots = N * S
        max_slots = table_slots * 2 + 1
        indices = (torch.randperm(max_slots - 1, device=device)[:table_slots] + 1).int().view(N, S)
        pad_value = 0

    if pad_last and N > 1:
        indices[-1] = pad_value

    if page_pad > 0:
        # Simulate vLLM's page-packed layout: each slot carries conv_state
        # data before the ssm_state region, so stride(0) > H*D*D.
        state_elems = H * D * D
        page_elems = state_elems + page_pad
        pool_raw = torch.randn(
            max_slots * page_elems, dtype=torch.float32, device=device)
        pool = torch.as_strided(
            pool_raw, (max_slots, H, D, D), (page_elems, D * D, D, 1))
        assert pool.stride(0) > state_elems, "page_pad must make stride(0) > H*V*K"
    else:
        pool = torch.randn(
            max_slots, H, D, D, dtype=torch.float32, device=device)
    num_accepted = (
        None if is_sglang
        else torch.randint(1, S + 1, (N,), dtype=torch.int32, device=device)
    )

    q, k, v, raw_g, raw_beta = _make_inputs(total, H, dtype)
    A_log, dt_bias = _make_gate_params(H)
    cu = torch.tensor([0, *seq_lens], dtype=torch.int32, device=device).cumsum(0)

    ref_pool, tri_pool = pool.clone(), pool.clone()
    ref_out = _reference(q, k, v, raw_g, raw_beta, A_log, dt_bias,
                         ref_pool, cu, indices, num_accepted, scale, is_sglang)

    out_buf = torch.empty_like(v)
    tri_out, final_state = fused_recurrent_kda(
        q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
        A_log=A_log, dt_bias=dt_bias, lower_bound=None,
        initial_state=tri_pool, cu_seqlens=cu, ssm_state_indices=indices,
        num_accepted_tokens=num_accepted, out=out_buf, is_sglang=is_sglang,
    )

    assert tri_out is out_buf
    assert final_state is tri_pool
    tag = "sglang" if is_sglang else "vllm"
    assert_close(f"{tag}_o", ref_out, tri_out, 0.02)
    assert_close(f"{tag}_state", ref_pool, tri_pool, 0.02)

    if pad_last and N > 1:
        # A padded request must emit exact zeros.
        pad_out = tri_out[:, sum(seq_lens[:-1]):]
        assert torch.count_nonzero(pad_out).item() == 0, \
            f"{tag}: padded request emitted non-zeros"
    if is_sglang:
        # Request 0 sits on slot 0: under sglang semantics it must have been
        # computed and committed, not treated as padding.
        assert torch.count_nonzero(tri_out[:, :seq_lens[0]]).item() > 0, \
            "sglang: slot 0 was wrongly treated as padding"
        assert not torch.equal(tri_pool[0], pool[0]), \
            "sglang: slot 0 should have been committed"
    else:
        # vLLM's reserved null row must never be written.
        assert_close("vllm_null_row", pool[0], tri_pool[0], 0.0)


# ===========================================================================
# 1. vLLM spec decode: 2D [N, S] slot table + num_accepted_tokens
# ===========================================================================
_VLLM_CASES = [
    # N, S, H, dtype
    (2, 4, 4, torch.bfloat16),
    (2, 2, 8, torch.float16),
    (1, 4, 4, torch.bfloat16),
    (4, 3, 8, torch.float16),
    (3, 4, 16, torch.bfloat16),
]


@_params(_VLLM_CASES, ("N,S,H,dtype", "N{}-S{}-H{}-{}"))
def test_fused_recurrent_kda_vllm_spec(N, S, H, dtype):
    """vLLM: start at the accepted slot, commit one state per draft token."""
    _run_case(N, S, H, dtype, is_sglang=False, pad_last=False, seed=123)


@_params(_VLLM_CASES[:3], ("N,S,H,dtype", "N{}-S{}-H{}-{}"))
def test_fused_recurrent_kda_vllm_spec_padding(N, S, H, dtype):
    """vLLM: a padded request (slot 0) yields zeros and touches no state."""
    _run_case(N, S, H, dtype, is_sglang=False, pad_last=True, seed=7)


_VLLM_RAGGED_CASES = [
    # The first case is not divisible by N; the second has an integer average
    # that is smaller than the longest request. The third mirrors trailing
    # zero-length CUDA Graph padding where total tokens happens to equal N.
    ((2, 3), 4, torch.bfloat16),
    ((2, 4), 4, torch.bfloat16),
    ((2, 0), 4, torch.bfloat16),
]


@_params(_VLLM_RAGGED_CASES, ("seq_lens,H,dtype", "lens{}-H{}-{}"))
def test_fused_recurrent_kda_vllm_spec_ragged(seq_lens, H, dtype):
    """vLLM: size MTP shared memory for the longest ragged request."""
    _run_case(
        len(seq_lens), max(seq_lens), H, dtype,
        is_sglang=False, pad_last=seq_lens[-1] == 0,
        seed=456, seq_lens=seq_lens,
    )


# ===========================================================================
# 1b. vLLM spec decode with page-packed state pool (non-contiguous stride)
# ===========================================================================
@_params(_VLLM_CASES[:3], ("N,S,H,dtype", "N{}-S{}-H{}-{}"))
def test_fused_recurrent_kda_vllm_spec_page_packed(N, S, H, dtype):
    """vLLM: page-packed state pool where stride(0) > H*V*K.

    Each slot has extra conv_state elements sharing the page, so the
    per-timestep state write must use ``h0_slot_stride`` (not the
    hardcoded ``HV * V * K``) to land in the correct memory.
    """
    _run_case(N, S, H, dtype, is_sglang=False, pad_last=False,
             seed=789, page_pad=36864)


# ===========================================================================
# 2. sglang spec decode: 1D slot table, single commit at the end
# ===========================================================================
_SGLANG_CASES = [
    # N, S, H, dtype
    (2, 4, 4, torch.bfloat16),
    (2, 2, 8, torch.float16),
    (1, 4, 4, torch.bfloat16),
    (4, 3, 8, torch.float16),
    (3, 4, 16, torch.bfloat16),
]


@_params(_SGLANG_CASES, ("N,S,H,dtype", "N{}-S{}-H{}-{}"))
def test_fused_recurrent_kda_sglang_spec(N, S, H, dtype):
    """sglang: 1D slots (0 is valid), final state committed once at the end."""
    _run_case(N, S, H, dtype, is_sglang=True, pad_last=False, seed=321)


@_params(_SGLANG_CASES[:3], ("N,S,H,dtype", "N{}-S{}-H{}-{}"))
def test_fused_recurrent_kda_sglang_spec_padding(N, S, H, dtype):
    """sglang: PAD_SLOT_ID (-1) yields zeros and touches no state."""
    _run_case(N, S, H, dtype, is_sglang=True, pad_last=True, seed=11)


# ===========================================================================
# 3. is_sglang actually switches the slot-0 convention
# ===========================================================================
def test_is_sglang_switches_slot0_semantics():
    """One 1D table containing slot 0, run under both conventions.

    Slot 0 is a valid scratch row for sglang but ``NULL_BLOCK_ID`` for vLLM, so
    the very same input must be computed in one case and zeroed in the other.
    This is what pins down ``is_sglang``; requests on other slots must agree.
    """
    torch.manual_seed(99)
    N, S, H, dtype = 2, 4, 4, torch.bfloat16
    total = N * S

    pool = torch.randn(total + 2, H, D, D, dtype=torch.float32, device=device)
    # Request 0 -> slot 0 (the contested one), request 1 -> a plainly valid slot.
    indices = torch.tensor([0, 3], dtype=torch.int32, device=device)
    q, k, v, raw_g, raw_beta = _make_inputs(total, H, dtype)
    A_log, dt_bias = _make_gate_params(H)
    cu = torch.arange(0, total + 1, step=S, dtype=torch.int32, device=device)

    def run(is_sglang):
        p = pool.clone()
        o, _ = fused_recurrent_kda(
            q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
            A_log=A_log, dt_bias=dt_bias, lower_bound=None,
            initial_state=p, cu_seqlens=cu, ssm_state_indices=indices,
            is_sglang=is_sglang,
        )
        return o, p

    out_sg, pool_sg = run(True)
    out_vl, pool_vl = run(False)

    # Request 0 (slot 0): computed under sglang, treated as padding under vLLM.
    assert torch.count_nonzero(out_sg[:, :S]).item() > 0, \
        "sglang: slot 0 must be computed"
    assert torch.count_nonzero(out_vl[:, :S]).item() == 0, \
        "vLLM: slot 0 is NULL_BLOCK_ID and must be zeroed"
    assert not torch.equal(pool_sg[0], pool[0]), \
        "sglang: slot 0 must be committed"
    assert torch.equal(pool_vl[0], pool[0]), \
        "vLLM: slot 0 must never be written"

    # Request 1 sits on a valid slot under both conventions -> identical results.
    assert_close("slot0_switch_other_req", out_sg[:, S:], out_vl[:, S:], 0.0)
    assert_close("slot0_switch_other_state", pool_sg[3], pool_vl[3], 0.0)
    print("is_sglang correctly switches the slot-0 convention")


# ===========================================================================
# 4. A 2D slot table is vLLM-only; sglang must be rejected loudly.
# ===========================================================================
def test_sglang_rejects_2d_indices():
    """sglang has no per-token slot layout, so a 2D table must raise."""
    torch.manual_seed(0)
    N, S, H = 2, 2, 4
    total = N * S
    pool = torch.randn(total + 1, H, D, D, dtype=torch.float32, device=device)
    indices = (torch.arange(total, device=device, dtype=torch.int32) + 1).view(N, S)
    q, k, v, raw_g, raw_beta = _make_inputs(total, H, torch.bfloat16)
    A_log, dt_bias = _make_gate_params(H)
    cu = torch.arange(0, total + 1, step=S, dtype=torch.int32, device=device)

    try:
        fused_recurrent_kda(
            q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
            A_log=A_log, dt_bias=dt_bias, lower_bound=None,
            initial_state=pool, cu_seqlens=cu, ssm_state_indices=indices,
            is_sglang=True,
        )
    except ValueError:
        print("2D+is_sglang correctly rejected")
        return
    raise AssertionError("expected a ValueError for 2D indices with is_sglang=True")


# ===========================================================================
# 5. dt_bias is consumed at full fp32 precision on the MTP-verify path
# ===========================================================================
def test_dt_bias_fp32_precision_is_used():
    """The KDA MTP kernel must read dt_bias at fp32, not round it to q's dtype.

    S = 4 makes this a multi-token (MTP-verify) call, which is the path that reads
    dt_bias as fp32. Both runs below pass an fp32 tensor, so both take the same
    instantiation; only the *values* differ, one of them pre-rounded to bf16. A
    kernel that truncated dt_bias internally would collapse the two into identical
    output. The exact run must also track the fp32 reference at least as closely,
    since rounding dt_bias perturbs every per-K decay through
    softplus(raw_g + dt_bias) and the recurrence compounds that over the sequence.
    """
    torch.manual_seed(4242)
    N, S, H, dtype = 2, 4, 4, torch.bfloat16
    total, scale = N * S, D ** -0.5

    pool = torch.randn(total + 2, H, D, D, dtype=torch.float32, device=device)
    indices = torch.tensor([0, 3], dtype=torch.int32, device=device)
    q, k, v, raw_g, raw_beta = _make_inputs(total, H, dtype)
    A_log, dt_bias = _make_gate_params(H)
    cu = torch.arange(0, total + 1, step=S, dtype=torch.int32, device=device)

    ref_pool = pool.clone()
    ref_out = _reference(q, k, v, raw_g, raw_beta, A_log, dt_bias,
                        ref_pool, cu, indices, None, scale, is_sglang=True)

    def run(dtb):
        p = pool.clone()
        o, _ = fused_recurrent_kda(
            q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
            A_log=A_log, dt_bias=dtb, lower_bound=None,
            initial_state=p, cu_seqlens=cu, ssm_state_indices=indices,
            is_sglang=True,
        )
        return o

    out_exact = run(dt_bias)
    out_rounded = run(dt_bias.to(dtype).float())  # bf16 values, still an fp32 tensor

    assert not torch.equal(out_exact, out_rounded), (
        "rounding dt_bias to bf16 changed nothing, so the kernel is truncating "
        "dt_bias internally instead of reading it as fp32")

    err_exact = (ref_out.float() - out_exact.float()).norm().item()
    err_rounded = (ref_out.float() - out_rounded.float()).norm().item()
    print(f"dt_bias exact err={err_exact:.6f} vs bf16-rounded err={err_rounded:.6f}")
    assert err_exact <= err_rounded, (
        f"exact fp32 dt_bias ({err_exact:.6f}) should not be worse than "
        f"bf16-rounded values ({err_rounded:.6f})")


@_params([(torch.bfloat16,), (torch.float16,)], ("dt_bias_dtype", "dtb{}"))
def test_rejects_non_fp32_dt_bias(dt_bias_dtype):
    """Only fp32 is instantiated for KDA, so 16-bit dt_bias must raise.

    bfloat16 is covered too even though it matches q's dtype here: the KDA gate
    has no scalar_t instantiation, and silently widening would hide the mismatch.
    """
    torch.manual_seed(0)
    N, S, H = 2, 2, 4
    total = N * S
    pool = torch.randn(total + 1, H, D, D, dtype=torch.float32, device=device)
    indices = torch.arange(total, device=device, dtype=torch.int32)[:N]
    q, k, v, raw_g, raw_beta = _make_inputs(total, H, torch.bfloat16)
    A_log, dt_bias = _make_gate_params(H, dt_bias_dtype)
    cu = torch.arange(0, total + 1, step=S, dtype=torch.int32, device=device)

    try:
        fused_recurrent_kda(
            q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
            A_log=A_log, dt_bias=dt_bias, lower_bound=None,
            initial_state=pool, cu_seqlens=cu, ssm_state_indices=indices,
            is_sglang=True,
        )
    except ValueError as e:
        print(f"{dt_bias_dtype} dt_bias correctly rejected: {e}")
        return
    raise AssertionError(f"expected a ValueError for a {dt_bias_dtype} dt_bias")


# ===========================================================================
# 6. Pure decode narrows dt_bias to q's dtype inside the wrapper
# ===========================================================================
_DECODE_CASES = [
    # N, S, H, dtype -- S == 1 is one token per request, the non-MTP path.
    (4, 1, 4, torch.bfloat16),
    (1, 1, 8, torch.float16),
    (8, 1, 16, torch.bfloat16),
]


@_params(_DECODE_CASES, ("N,S,H,dtype", "N{}-S{}-H{}-{}"))
def test_fused_recurrent_kda_pure_decode(N, S, H, dtype):
    """One token per request: the wrapper must narrow the fp32 dt_bias itself.

    Callers always hand fused_recurrent_kda an fp32 dt_bias, but this shape takes
    the non-MTP kernel, which reads dt_bias at q's dtype. Were the wrapper to
    forward fp32 unchanged, the kernel entry point's dtype check would reject it.
    """
    _run_case(N, S, H, dtype, is_sglang=True, pad_last=False, seed=555)


def _main():
    if not torch.cuda.is_available():
        print("CUDA device not available; skipping (run on the PPU server).")
        return
    suites = [
        (test_fused_recurrent_kda_vllm_spec, _VLLM_CASES),
        (test_fused_recurrent_kda_vllm_spec_padding, _VLLM_CASES[:3]),
        (test_fused_recurrent_kda_vllm_spec_ragged, _VLLM_RAGGED_CASES),
        (test_fused_recurrent_kda_vllm_spec_page_packed, _VLLM_CASES[:3]),
        (test_fused_recurrent_kda_sglang_spec, _SGLANG_CASES),
        (test_fused_recurrent_kda_sglang_spec_padding, _SGLANG_CASES[:3]),
        (test_is_sglang_switches_slot0_semantics, [()]),
        (test_sglang_rejects_2d_indices, [()]),
        (test_dt_bias_fp32_precision_is_used, [()]),
        (test_rejects_non_fp32_dt_bias, [(torch.bfloat16,), (torch.float16,)]),
        (test_fused_recurrent_kda_pure_decode, _DECODE_CASES),
    ]
    passed = failed = skipped = 0
    for fn, cases in suites:
        for case in cases:
            try:
                fn(*case)
                passed += 1
                print(f"[PASS] {fn.__name__}{case}")
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "Skipped":
                    skipped += 1
                    print(f"[SKIP] {fn.__name__}{case}: {e}")
                else:
                    failed += 1
                    print(f"[FAIL] {fn.__name__}{case}: {e}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
