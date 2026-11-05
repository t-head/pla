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

"""End-to-end forward test: flashkda_fwd (Kernel1+Kernel2) vs torch reference.

Numerical semantics aligned to upstream tests/torch_ref.py:
  * L2 normalize with kernel-match warp-shuffle FMA tree + `+1e-6` in rsqrt
  * Gate: `lower*LOG2E * sigmoid(exp2(A_log*LOG2E) * (g + dt_bias))`, exp2 with FTZ,
    sigmoid emulating PTX tanh.approx.f32 semantics
  * k_decayed / q_decayed / k_inv / k_restored computed in bf16 (bf16 exp-mul-bf16),
    scale as bf16 tensor
  * L = mm(k_decayed, k_inv.T) fp32-acc -> fp16 storage
  * Mqk = q_decayed @ k_inv.T (bf16)
  * INV = Neumann series in fp16 with fp16 storage (three matmul_fp16acc rounds)
  * v_chunk / U / _out done in bf16
  * delta_s = mm(k_restored.T, U) fp32-acc
  * state update via fp32_fma (fp64-emulated FMA), then bf16 round-transpose

Since the PPU container's "cuda" device is actually PPU (no NV runtime, no cuBLAS
fp16-accum, no CUDA-inline compile), sigmoid_tanh and matmul_fp16acc are emulated
in pure PyTorch. This is slightly *more* precise than upstream's true fp16-acc
cublas path, so the reference bounds the kernel error from a tighter side — hence
the tolerance was tightened from 3e-2 to 1e-2 (upstream check_optimized_fwd
uses rtol=1e-2 against its cuBLAS-exact ref).
"""
from contextlib import contextmanager
import os

import torch

CH, D = 16, 128

LOG2E = 1.4426950408889634


def err_ratio(x, y):
    x = x.float(); y = y.float()
    return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)).item()


# ---------------------------------------------------------------------------
# Numeric helpers, matching torch_ref.py semantics
# ---------------------------------------------------------------------------

def _fp32_ex2_ftz(x):
    """torch.special.exp2 with flush-to-zero for denormals; input promoted to fp32."""
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    r = torch.special.exp2(x)
    tiny = torch.finfo(torch.float32).tiny
    return torch.where(r.abs() < tiny, torch.zeros_like(r), r)


def _fp32_fma(c, a, b):
    """fp32 fused multiply-add emulated with fp64 intermediate to match kernel FMA."""
    assert c.dtype == torch.float32 and a.dtype == torch.float32 and b.dtype == torch.float32
    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)


def _sigmoid_tanh_f32(x):
    """Emulates PTX `tanh.approx.f32`-based sigmoid: sigmoid(x) = tanh(x/2)/2 + 0.5.

    Computed in fp32 to match the CUDA kernel used by upstream torch_ref.
    torch.tanh matches tanh.approx.f32 to within a few ULP for finite inputs.
    """
    xf = x.to(torch.float32)
    return 0.5 * torch.tanh(xf * 0.5) + 0.5


def _l2_normalize_kernel_match(x):
    """Match the PPU kernel's warp-shuffle tree reduction with FMA.

    Groups the last dim (D=128) into 16 lanes × 8 elements, accumulates squares
    per-lane with fp32 FMA (fp64-emulated), then reduces across the 16 lanes with
    the same XOR-butterfly the kernel uses, rsqrt with `+1e-6` epsilon.
    """
    assert x.shape[-1] == 128, f"L2 kernel-match requires D=128, got {x.shape[-1]}"
    xf = x.float()
    groups = xf.reshape(*xf.shape[:-1], 16, 8)
    partials = torch.zeros(*xf.shape[:-1], 16, dtype=torch.float32, device=x.device)
    for i in range(8):
        partials = _fp32_fma(partials, groups[..., i], groups[..., i])
    idx16 = torch.arange(16, device=x.device)
    for offset in [8, 4, 2, 1]:
        partials = partials + partials[..., idx16 ^ offset]
    inv_norm = torch.rsqrt(partials[..., 0:1] + 1e-6)
    return (xf * inv_norm).to(x.dtype)


def _matmul_fp16acc(a, b):
    """Emulated fp16-accumulate matmul.

    Upstream torch_ref uses cuBLAS `CUBLAS_COMPUTE_16F` which accumulates in fp16
    along K. Since cuBLAS is not available on the PPU container we approximate by
    casting inputs to fp16 and using `torch.mm` (fp32-acc under the hood on most
    backends), then rounding the result to fp16. This is slightly more precise
    than the true fp16-acc; the fp16 storage precision of intermediates is what
    matters most for INV convergence, which is preserved.
    """
    a16 = a.to(torch.float16)
    b16 = b.to(torch.float16)
    return torch.mm(a16, b16).to(torch.float16)


# ---------------------------------------------------------------------------
# Reference forward (single sequence, N=1) — aligned to upstream torch_ref.py
# ---------------------------------------------------------------------------

def ref_fwd(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, h0, T, H):
    """Reference forward with dtype/rounding semantics matching torch_ref.py.

    Inputs (single sequence, N=1):
      q,k,v,g : [T,H,D] bf16
      beta    : [T,H]   bf16
      A_log   : [H]     fp32
      dt_bias : [H,D]   fp32
      h0      : [H,D,D] bf16 or fp32 (initial state)

    Returns:
      out   : [T,H,D] fp32
      final : [H,D,D] fp32
    """
    device = q.device
    assert q.shape[-1] == D, f"Expected last dim {D}, got {q.shape[-1]}"

    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    beta = beta.to(torch.bfloat16)
    A_log = A_log.to(torch.float32)
    dt_bias = dt_bias.to(torch.float32)

    # L2-normalize q,k using kernel-match reduction.
    q = _l2_normalize_kernel_match(q)
    k = _l2_normalize_kernel_match(k)

    # Gate: g <- (lower_bound * LOG2E) * sigmoid(exp2(A_log * LOG2E) * (g + dt_bias))
    g_f = g.to(torch.float32) + dt_bias.unsqueeze(0)                          # [T,H,D]
    a_log_exp = _fp32_ex2_ftz(A_log * LOG2E).unsqueeze(0).unsqueeze(-1)       # [1,H,1]
    gate_scale = lower_bound * LOG2E
    g_gate = gate_scale * _sigmoid_tanh_f32(a_log_exp * g_f)                  # [T,H,D] fp32

    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16, device=device)

    out = torch.zeros(T, H, D, dtype=torch.float32, device=device)
    work_state = h0.to(torch.bfloat16).clone()                                # [H,D,D] bf16
    n_chunks = (T + CH - 1) // CH

    for h in range(H):
        state_slice = work_state[h]                                           # [D,D] bf16
        for ci in range(n_chunks):
            t0 = ci * CH
            actual = min(CH, T - t0)

            g_chunk = torch.zeros(CH, D, dtype=torch.float32, device=device)
            q_chunk = torch.zeros(CH, D, dtype=torch.bfloat16, device=device)
            k_chunk = torch.zeros(CH, D, dtype=torch.bfloat16, device=device)
            v_chunk = torch.zeros(CH, D, dtype=torch.bfloat16, device=device)
            beta_chunk = torch.zeros(CH, dtype=torch.bfloat16, device=device)

            g_chunk[:actual] = g_gate[t0:t0 + actual, h]
            q_chunk[:actual] = q[t0:t0 + actual, h]
            k_chunk[:actual] = k[t0:t0 + actual, h]
            v_chunk[:actual] = v[t0:t0 + actual, h]
            beta_chunk[:actual] = beta[t0:t0 + actual, h]

            g_cumsum = g_chunk.cumsum(dim=0)                                  # fp32 [CH,D]
            g_total = g_cumsum[-1:]                                           # fp32 [1,D]

            k_decayed = k_chunk * _fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
            q_decayed = q_chunk * _fp32_ex2_ftz(g_cumsum).to(torch.bfloat16) * scale_bf16
            k_inv = k_chunk * _fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
            k_restored = k_inv * _fp32_ex2_ftz(g_total).to(torch.bfloat16)

            # L = k_decayed @ k_inv.T in fp32 acc, stored fp16.
            L = torch.mm(k_decayed.float(), k_inv.float().t()).to(torch.float16)
            # Mqk = q_decayed @ k_inv.T (bf16 default).
            Mqk = torch.matmul(q_decayed, k_inv.t())

            beta_activated = _sigmoid_tanh_f32(beta_chunk.to(torch.float32))
            beta_bf16 = beta_activated.to(torch.bfloat16).unsqueeze(-1)       # [CH,1]
            beta_fp16 = beta_activated.to(torch.float16).unsqueeze(-1)        # [CH,1]

            L = torch.tril(L, diagonal=-1) * beta_fp16                        # fp16
            Mqk = torch.tril(Mqk)                                             # bf16

            # Neumann series in fp16 with fp16 accumulate storage.
            INV = torch.eye(CH, dtype=torch.float16, device=device) - L
            L2 = _matmul_fp16acc(L, L)
            INV = INV + _matmul_fp16acc(INV, L2)
            L4 = _matmul_fp16acc(L2, L2)
            INV = INV + _matmul_fp16acc(INV, L4)
            L8 = _matmul_fp16acc(L4, L4)
            INV = INV + _matmul_fp16acc(INV, L8)
            INV = INV.to(torch.bfloat16)

            v_minus = v_chunk - torch.matmul(k_decayed, state_slice.t())
            vbeta = v_minus * beta_bf16
            U = torch.matmul(INV, vbeta)                                      # bf16
            _out = torch.matmul(q_decayed, state_slice.t()) + torch.matmul(Mqk, U)

            delta_s = torch.mm(k_restored.float().t(), U.float())             # fp32 [D,D]

            g_total_exp = _fp32_ex2_ftz(g_total).squeeze(0).unsqueeze(-1)     # fp32 [D,1]
            new_state = _fp32_fma(delta_s, state_slice.to(torch.float32).t(), g_total_exp)
            state_slice = new_state.to(torch.bfloat16).t()
            work_state[h] = state_slice

            out[t0:t0 + actual, h] = _out[:actual].float()

    return out, work_state.float()


@contextmanager
def reference_threads(threads=None):
    """Small CPU matrices need bounded intra-op parallelism, not a thread pool per head."""
    threads = int(os.environ.get("PLA_FLASHKDA_REF_THREADS", "1") if threads is None else threads)
    if threads < 1:
        raise ValueError("PLA_FLASHKDA_REF_THREADS must be positive")
    previous = torch.get_num_threads()
    if previous != threads:
        torch.set_num_threads(threads)
    try:
        yield
    finally:
        if previous != threads:
            torch.set_num_threads(previous)


@torch.no_grad()
def ref_fwd_batched(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, h0, T, H):
    """CPU oracle batching independent heads; chunk carry and all rounding boundaries stay serial.

    The scalar ref_fwd above remains the independent regression oracle. No
    production kernel, state precision, or accuracy threshold is changed.
    Scratch is bounded by one chunk across H heads, not the sequence length.
    """
    if q.device.type != "cpu":
        raise ValueError("ref_fwd_batched requires CPU inputs")
    if tuple(q.shape) != (T, H, D) or T < 0 or H < 1:
        raise ValueError("q must match [T,H,128] with T>=0 and H>0")
    with reference_threads():
        return _ref_fwd_batched(q, k, v, g, beta, A_log, dt_bias,
                                scale, lower_bound, h0, T, H)


def _ref_fwd_batched(q, k, v, g, beta, A_log, dt_bias, scale, lower_bound, h0, T, H):
    q, k, v, g, beta = [x.to(torch.bfloat16) for x in (q, k, v, g, beta)]
    A_log, dt_bias = A_log.float(), dt_bias.float()
    q, k = _l2_normalize_kernel_match(q), _l2_normalize_kernel_match(k)
    g_f = g.float() + dt_bias.unsqueeze(0)
    a_log_exp = _fp32_ex2_ftz(A_log * LOG2E).unsqueeze(0).unsqueeze(-1)
    g_gate = (lower_bound * LOG2E) * _sigmoid_tanh_f32(a_log_exp * g_f)
    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16)
    out = torch.zeros(T, H, D, dtype=torch.float32)
    state = h0.to(torch.bfloat16).clone()
    identity = torch.eye(CH, dtype=torch.float16).unsqueeze(0)
    g_chunk = torch.zeros(H, CH, D, dtype=torch.float32)
    q_chunk, k_chunk, v_chunk = [torch.zeros(H, CH, D, dtype=torch.bfloat16) for _ in range(3)]
    beta_chunk = torch.zeros(H, CH, dtype=torch.bfloat16)

    def mm(a, b):
        # Keep the original operand dtype: promoting CPU BF16 products to
        # FP32 can change tie rounding even when every storage cast is kept.
        return torch.bmm(a, b)

    def mm16(a, b):
        return mm(a.to(torch.float16), b.to(torch.float16)).to(torch.float16)

    for t0 in range(0, T, CH):
        actual = min(CH, T - t0)
        for scratch, source in ((g_chunk, g_gate), (q_chunk, q),
                                (k_chunk, k), (v_chunk, v), (beta_chunk, beta)):
            if actual < CH:
                scratch[:, actual:].zero_()
            scratch[:, :actual].copy_(source[t0:t0 + actual].transpose(0, 1))
        g_cumsum = g_chunk.cumsum(dim=1)
        g_total = g_cumsum[:, -1:]
        decay = _fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
        g_total_exp = _fp32_ex2_ftz(g_total)
        k_decayed = k_chunk * decay
        q_decayed = q_chunk * decay * scale_bf16
        k_inv = k_chunk * _fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
        k_restored = k_inv * g_total_exp.to(torch.bfloat16)
        L = mm(k_decayed.float(), k_inv.float().transpose(1, 2)).to(torch.float16)
        Mqk = mm(q_decayed, k_inv.transpose(1, 2))
        beta_activated = _sigmoid_tanh_f32(beta_chunk.float())
        beta_bf16 = beta_activated.to(torch.bfloat16).unsqueeze(-1)
        beta_fp16 = beta_activated.to(torch.float16).unsqueeze(-1)
        L = torch.tril(L, diagonal=-1) * beta_fp16
        Mqk = torch.tril(Mqk)
        INV = identity - L
        L2 = mm16(L, L)
        INV = INV + mm16(INV, L2)
        L4 = mm16(L2, L2)
        INV = INV + mm16(INV, L4)
        L8 = mm16(L4, L4)
        INV = (INV + mm16(INV, L8)).to(torch.bfloat16)
        state_t = state.transpose(1, 2)
        v_minus = v_chunk - mm(k_decayed, state_t)
        U = mm(INV, v_minus * beta_bf16)
        output = mm(q_decayed, state_t) + mm(Mqk, U)
        delta_s = mm(k_restored.float().transpose(1, 2), U.float())
        g_total_exp = g_total_exp.transpose(1, 2)
        state = _fp32_fma(delta_s, state.float().transpose(1, 2),
                          g_total_exp).to(torch.bfloat16).transpose(1, 2)
        out[t0:t0 + actual] = output[:, :actual].transpose(0, 1).float()
    return out, state.float()


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

# err_ratio tolerance. Upstream `test_fwd.py` uses torch.equal (exact) on its
# reference CUTLASS path; test_fwd_vs_fla uses 5e-3; check_optimized_fwd uses
# rtol=1e-2. On PPU we cannot hit bit-exact due to hardware MMA rounding order
# differences, but the reference now matches the kernel's dtype scheme, so
# 1e-2 is a reasonable tight target.
TOL = 1e-2


def main():
    from pla.prefill.flashkdapro import flashkda_fwd
    torch.manual_seed(0)
    T, H = 64, 2
    scale = 1.0 / (D ** 0.5); LOWER = -5.0
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    h0 = torch.randn(1, H, D, D, dtype=torch.bfloat16, device="cuda") * 0.1

    out = torch.empty(T, H, D, dtype=torch.bfloat16, device="cuda")
    fin = torch.empty(1, H, D, D, dtype=torch.bfloat16, device="cuda")
    flashkda_fwd(q, k, v, g, beta, A_log, dt_bias, out,
                 scale, LOWER, h0, fin)
    torch.cuda.synchronize()
    out = out.cpu(); fin = fin[0].cpu()

    rout, rfin = ref_fwd(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(),
                         A_log.cpu(), dt_bias.cpu(), scale, LOWER, h0[0].cpu(), T, H)
    ro = err_ratio(out.reshape(-1), rout.reshape(-1))
    rf = err_ratio(fin.reshape(-1), rfin.reshape(-1))
    print(f"  out   err_ratio={ro:.4e}")
    print(f"  state err_ratio={rf:.4e}")
    assert ro < TOL and rf < TOL, f"E2E forward mismatch vs reference (tol={TOL})"
    print(f"FWD OK: flashkda_fwd matches torch reference (err_ratio<{TOL})")


if __name__ == "__main__":
    main()
