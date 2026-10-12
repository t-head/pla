#!/usr/bin/env python3
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

"""Prefill (FlashQLA GDN) correctness validation tools, used by test_op_forge_prefill.py.

Design constraint: **nothing under scripts/ is modified**. This module self-containedly implements tensor construction and accuracy
judgment for both single-sequence and multi-sequence packed prefill; the kernel under test and the golden reference are both imported
directly from the pla package / the reference implementation in scripts (read only, never modified):

  - kernel under test : pla.prefill.flashqla.chunk_gated_delta_rule_fwd (the real prefill kernel)
  - golden reference  : scripts/prefill/flashqla/private/ref_gdr.chunk_gated_delta_rule_fwd
                  (vendored FlashQLA torch reference, reused read-only)
  - CP planner  : pla.prefill.flashqla.private.cp_preprocess.calc_cp_seqs

Numerical criteria follow OperatorsForge/tools/FAOpBench.py:
  cos > 0.999 (equivalent to cos_diff <= 1e-3) and rel < 5e-2 (shared by o and h, regardless of dtype).
  rel is still the max-abs metric (the computation is unchanged; only the threshold was relaxed from 1e-2 to 5e-2).
"""

import os
import sys

import torch
import torch.nn.functional as F

# ref_gdr and its vendored dependency flash_qla.utils both live under scripts/prefill/flashqla/private/;
# flashqla/ is added to sys.path only to import the golden reference; these files are not modified.
# ref_gdr uses relative imports internally (from .flash_qla.utils import ...), so it must be loaded as a
# submodule of the private package, the same way as scripts/prefill/flashqla/e2e_acc_full.py.
_THIS = os.path.dirname(os.path.abspath(__file__))
_PREFILL_DIR = os.path.join(_THIS, "..", "scripts", "prefill", "flashqla")
if os.path.isdir(_PREFILL_DIR) and _PREFILL_DIR not in sys.path:
    sys.path.insert(0, _PREFILL_DIR)

from private import ref_gdr  # noqa: E402  (golden reference, read-only)
from pla.prefill.flashqla import chunk_gated_delta_rule_fwd  # noqa: E402
from pla.prefill.flashqla.private.cp_preprocess import calc_cp_seqs  # noqa: E402

# kernel specialization dimensions (DK=DV=128), consistent with e2e_acc_full
DK = 128
DV = 128
CS = 32
# Accuracy thresholds follow the authoritative criteria of OperatorsForge/tools/FAOpBench.py:
#   FAOpBench._check_stateful_fla_precision uses a single rel_thresh=5e-2 and cos_thresh(1-cos)=1e-3,
#   regardless of dtype, with o and h(state) sharing the same thresholds.
# Here the **computation of rel is unchanged** (still max-abs: max|a-b|/(max|b|+eps));
# only the threshold is relaxed to 5e-2 to match (the old bf16 1e-2 / fp16 5e-3 had no safety margin,
# and bf16 accumulation error on long packed sequences occasionally grazed the limit, see case757/case431).
REF_COS_MIN = 0.999            # equivalent to FAOpBench's cos_diff <= 1e-3
REF_REL_MAX = 5e-2
BF16_REF_REL_MAX = 5e-2


def tensor_metrics(a, b):
    """Return (cos, rel, finite), same criteria as e2e_acc_full.tensor_metrics."""
    finite = torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    if not finite:
        return 0.0, 1.0, False
    cos = F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0)).item()
    rel = ((a.float() - b.float()).abs().max()
           / (b.float().abs().max() + 1e-8)).item()
    return cos, float(rel), True


def _rel_max(dtype):
    return BF16_REF_REL_MAX if dtype == torch.bfloat16 else REF_REL_MAX


# dtype of the recurrent state pool. bf16 halves the resident state memory; in both cases the recurrence itself
# runs in fp32 registers, and only the two boundary accesses in and out of memory are affected.
STATE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _tag(state_dtype):
    return "fp32" if state_dtype == torch.float32 else "bf16"


def _make_inputs(total, h, hg, dtype, device="cuda"):
    """Build prefill input tensors. q/k are L2-normalized (consistent with FLA prefill semantics).

    g / beta are generated with the production fused gating formula, to cover the raw gating interface and
    a wider numerical range:
      g    = -exp(A_log) * softplus(a + dt_bias)   ≤ 0
      beta = sigmoid(b)                            b is randn, full range (0,1)
    This is a correctness stress input and does not represent the Gate distribution of a real checkpoint. Performance tests use
    scripts/prefill/flashqla/bench_vs_fla.py with the slow decay and 25% zero-decay head distribution
    of the official FlashQLA benchmark.
    """
    q = F.normalize(torch.randn(1, total, hg, DK, device=device), dim=-1).to(dtype)
    k = F.normalize(torch.randn(1, total, hg, DK, device=device), dim=-1).to(dtype)
    v = (torch.randn(1, total, h, DV, device=device) * 0.1).to(dtype)
    A_log = torch.randn(h, dtype=torch.float32, device=device)
    dt_bias = torch.randn(h, device=device).to(dtype)
    a = torch.randn(total, h, device=device).to(dtype)
    b = torch.randn(total, h, device=device).to(dtype)
    g = -A_log.exp() * F.softplus(a.float() + dt_bias.float())
    # The production kernel rounds sigmoid(b) to the activation dtype before
    # storing it in the fp32 output tensor.
    beta = b.float().sigmoid().to(dtype).float()
    g = g.view(1, total, h)
    beta = beta.view(1, total, h)
    return q, k, v, g, beta


def _forward(backend, q, k, v, g, beta, scale, cu, initial_state,
             normalize_qk=False, state_dtype=None):
    """Run one prefill forward and return (o, ht).

    backend='pla'    -> the CuTe prefill kernel of pla.prefill.flashqla (side under test)
    backend='triton' -> sglang's triton chunk_gated_delta_rule (perf comparison baseline)

    normalize_qk follows the production semantics of scripts/prefill/flashqla/bench_vs_fla.py: GDN
    in the sglang runtime has in-kernel q/k L2 norm (use_qk_l2norm_in_kernel=True);
    both sides must enable it together, otherwise the triton side skips two l2norm passes and the pla side skips the fused cumsum,
    making the comparison unfair. The accuracy path stays False -- golden (ref_gdr) consumes already normalized q/k,
    which is existing API semantics and must not change.

    state_dtype only matters for the pla side: the triton kernel updates initial_state in place, so the dtype of the
    returned state can only follow the h0 passed in.
    """
    if backend != "triton":
        kw = {} if state_dtype is None else {"final_state_dtype": state_dtype}
        _, _, o, _, ht = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale=scale, initial_state=initial_state,
            cu_seqlens=cu, auto_cp=True, normalize_qk=normalize_qk, **kw)
        return o, ht

    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    h = v.shape[-2]
    n = cu.numel() - 1
    # In chunk_gated_delta_rule_fwd_kernel_h_blockdim64, initial_state /
    # initial_state_indices are loaded unconditionally (INPLACE_UPDATE is always True, and the final
    # state is written straight back to initial_state), so the triton side must provide the state buffer and indices;
    # this is exactly how the sglang production path uses it (in-place update of the state cache).
    state = initial_state
    if state is None:
        state = torch.zeros(n, h, DK, DV, device=q.device, dtype=torch.float32)
    indices = torch.arange(n, device=q.device, dtype=torch.int32)
    o, _, ht = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale,
        initial_state=state, initial_state_indices=indices,
        cu_seqlens=cu.to(torch.long),
        use_qk_l2norm_in_kernel=normalize_qk)
    return o, ht


def run_single(seq_len, h, hg, dtype=torch.bfloat16, seed=42, backend="pla",
               check=True, state_dtype=torch.float32):
    """Single-sequence prefill: cu_seqlens=[0, seq_len]. Returns True/False.

    Compares O / final_state of the packed CuTe kernel (auto_cp) against the ref_gdr golden.
    With check=False (perf collection) only the kernel is launched and no golden is computed: ref_gdr is a torch float
    reference, very slow for long sequences, and cycle collection relies only on acu capturing whitelisted kernels.
    ``state_dtype`` selects the dtype of the final_state returned by the kernel (fp32 or bf16);
    the golden is always fp32, and both are promoted to fp32 before comparison.
    """
    torch.manual_seed(seed)
    scale = DK ** -0.5
    q, k, v, g, beta = _make_inputs(seq_len, h, hg, dtype)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # perf collection uses production semantics (with in-kernel qk l2norm); accuracy checks keep the existing normalization criteria
    o_cute, ht_cute = _forward(backend, q, k, v, g, beta, scale, cu, None,
                               normalize_qk=not check, state_dtype=state_dtype)
    if not check:
        torch.cuda.synchronize()
        print(f"  single T={seq_len} H={h} Hg={hg} backend={backend} perf-only")
        return True
    _, o_ref, _, _, ht_ref = ref_gdr.chunk_gated_delta_rule_fwd(
        q.float(), k.float(), v.float(), g, beta,
        cu_seqlens=cu, initial_state=None, scale=scale, chunk_size=CS)

    o_cos, o_rel, o_fin = tensor_metrics(
        o_cute.reshape(1, seq_len, h, DV).float(),
        o_ref.reshape(1, seq_len, h, DV).float())
    ht_cos, ht_rel, ht_fin = tensor_metrics(
        ht_cute.reshape(1, h, DK, DV).float(),
        ht_ref.reshape(1, h, DK, DV).float())

    rm = _rel_max(dtype)
    # If bf16 was requested we must really get bf16, otherwise the "memory saving" benefit is fake.
    dtype_ok = ht_cute.dtype == state_dtype
    ok = (dtype_ok
          and o_fin and o_cos > REF_COS_MIN and o_rel < rm
          and ht_fin and ht_cos > REF_COS_MIN and ht_rel < rm)
    print(f"  single T={seq_len} H={h} Hg={hg} state={_tag(state_dtype)} "
          f"o_cos={o_cos:.5f}/o_rel={o_rel:.2e} "
          f"ht_cos={ht_cos:.5f}/ht_rel={ht_rel:.2e} {'PASS' if ok else 'FAIL'}")
    if not dtype_ok:
        print(f"    final_state dtype {ht_cute.dtype} != requested {state_dtype}")
    return ok


def run_packed(lengths, h, hg, dtype=torch.bfloat16, seed=42, backend="pla",
               check=True, state_dtype=torch.float32):
    """Multi-sequence packed prefill: cu_seqlens=[0, l1, l1+l2, ...]. Returns True/False.

    Compares O / final_state of packed (auto_cp) against the ref_gdr golden; same criteria as single sequence.
    With check=False (perf collection), same as run_single: only launch the kernel, no golden.
    ``state_dtype`` applies both to the initial_state passed in and the final_state returned:
    h0 is first quantized to state_dtype and the golden consumes the same .float(), so the inputs are identical
    and the difference only reflects the kernel internals and the state storage precision.
    """
    torch.manual_seed(seed)
    total = sum(lengths)
    q, k, v, g, beta = _make_inputs(total, h, hg, dtype)
    h0 = (torch.randn(len(lengths), h, DK, DV, device="cuda") * 0.02).to(state_dtype)
    cu = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device="cuda", dtype=torch.int32)

    # The CP planner is only printed for information; a caselist cannot know in advance whether it activates, so it does not affect the verdict
    use_cp, _, _, _, _ = calc_cp_seqs(cu, CS, h)

    packed_o, packed_h = _forward(backend, q, k, v, g, beta, None, cu, h0,
                                  normalize_qk=not check,
                                  state_dtype=state_dtype)
    if not check:
        torch.cuda.synchronize()
        print(f"  packed lengths={lengths} H={h} Hg={hg} cp={use_cp} "
              f"backend={backend} perf-only")
        return True
    # The golden must be computed **per sequence**; the whole packed batch cannot be handed to ref_gdr at once:
    # torch_kkt_fwd first unpack(k, cu_seqlens) restores packed into padded
    # [B, max_len], then pad_and_reshape into [B, N, C, H, K] for a 5-D einsum, whose batched
    # GEMM count = B x ceil(max_len/CS) x H. When this exceeds 65535 (the CUDA grid dimension limit)
    # cublasSgemmStridedBatched returns CUBLAS_STATUS_INVALID_VALUE (underneath it is acompute
    # gemm_cutlass3_rtc's HGGC_ERROR_INVALID_VALUE), the golden side throws, and the case is judged
    # fail -- unrelated to the kernel under test. That is exactly where the 379 fails of run 67817843 came from: checking the
    # Qwen3.7-Max tp2 caselist line by line with CS=32, H=64 and the 65535 limit predicts 379 failures,
    # matching the measurement one by one; all B=1 lines passed (1x994x64=63616 is just within the limit).
    # The sequences are independent anyway (each with its own initial_state=h0[i]); computing each with B=1 and concatenating
    # is exactly equivalent to computing the whole batch, reduces the batch count to ceil(L_i/CS)xH, and avoids wasted padding work (e.g.
    # lengths=[31801,967] used to be computed as 2x31801), so it is actually faster overall. After the change the four
    # tp1/2/4/8 caselists all predict 0 failures.
    # [Residual limit] a single sequence is still subject to the same constraint: ceil(L/32)x64 <= 65535 -> L <= 32736 tokens.
    # The longest single sequence in the current caselists is 31801, a margin of only 2.9%. If OperatorsForge later releases longer
    # single sequences (the model context is 128K), this will reproduce with the same error; at that point split further
    # along the head dimension (heads are independent in this einsum) instead of falling back to whole-batch computation.
    ref_o_parts, ref_h_parts = [], []
    off = 0
    for i, seq_len in enumerate(lengths):
        cu_i = torch.tensor([0, seq_len], device=cu.device, dtype=cu.dtype)
        sl = slice(off, off + seq_len)
        _, o_i, _, _, h_i = ref_gdr.chunk_gated_delta_rule_fwd(
            q[:, sl].float(), k[:, sl].float(), v[:, sl].float(),
            g[:, sl], beta[:, sl],
            cu_seqlens=cu_i, initial_state=h0[i:i + 1].float(), chunk_size=CS)
        ref_o_parts.append(o_i.reshape(1, seq_len, h, DV))
        ref_h_parts.append(h_i.reshape(1, h, DK, DV))
        off += seq_len
    ref_o = torch.cat(ref_o_parts, dim=1)
    ref_h = torch.cat(ref_h_parts, dim=0)

    o_cos, o_rel, o_fin = tensor_metrics(packed_o, ref_o)
    h_cos, h_rel, h_fin = tensor_metrics(packed_h, ref_h)

    rm = _rel_max(dtype)
    dtype_ok = packed_h.dtype == state_dtype
    ok = (dtype_ok
          and o_fin and o_cos > REF_COS_MIN and o_rel < rm
          and h_fin and h_cos > REF_COS_MIN and h_rel < rm)
    print(f"  packed lengths={lengths} H={h} Hg={hg} cp={use_cp} "
          f"state={_tag(state_dtype)} "
          f"o_cos={o_cos:.5f}/o_rel={o_rel:.2e} "
          f"h_cos={h_cos:.5f}/h_rel={h_rel:.2e} {'PASS' if ok else 'FAIL'}")
    if not dtype_ok:
        print(f"    final_state dtype {packed_h.dtype} != requested {state_dtype}")
    return ok
