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

from typing import Optional, Dict, Callable, Tuple
import logging
import pandas as pd
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import os
import argparse
import functools


# Add for nvtx profiling
SGLANG_PROFILE = os.getenv("SGLANG_PROFILE", False)
if SGLANG_PROFILE:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push
    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Import NVTX Error! {e}")
        SGLANG_PROFILE = False

if SGLANG_PROFILE == False:
    def th_nvtx_range_push(label):
        pass
    def th_nvtx_range_pop():
        pass

def randn_with_dtype(*shape, dtype, device):
    return torch.randn(*shape, dtype=dtype, device=device)

def torch_fused_recurrent_gated_delta_rule(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    g: torch.Tensor, 
    beta: torch.Tensor, 
    initial_state: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Torch Referrence for recurrent_gated_delta_rule
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    assert HV % H == 0

    S = torch.zeros(B, HV, K, V) if initial_state is None else initial_state.clone()
    o = torch.empty_like(v)

    for t in range(T):
        for hv in range(HV):
            h = hv // (HV // H)
            S_ = S[:, hv, :, :].to(torch.float32)                        # shape = (B, K, V)
            k_ = k[:, t, h, :].to(torch.float32)                         # shape = (B, K)
            v_ = v[:, t, hv, :].to(torch.float32)                        # shape = (B, V)
            q_ = q[:, t, h, :].to(torch.float32)                         # shape = (b, K)
            beta_ = beta[:, t, hv].view(-1, 1)                           # shape = (B, 1)
            g_ = g[:, t, hv].view(-1, 1, 1)                              # shape = (B, 1, 1)

            S_ = torch.exp(g_) * S_      
            v_old = torch.einsum('bkv,bk->bv', S_, k_)                   # shape = (B, V)
            v_diff = beta_ * (v_old - v_)        
            S_diff = torch.einsum('bk,bv->bkv', k_, v_diff)              # shape = (K, V)
            S_new = S_ - S_diff
            o[:, t, hv, :] = torch.einsum('bkv,bk->bv', S_new, q_)       # shape = (B, V)
            S[:, hv, :, :] = S_new
    return o, S


def referrence_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,    
):

    # Compute g = -exp(A_log) * softplus(a + dt_bias)
    beta_x = (a + dt_bias[None, :]) * softplus_beta
    softplus_x = torch.where(beta_x <= softplus_threshold, 1.0 / softplus_beta, F.softplus(beta_x))
    g = A_log.exp() * softplus_x
    beta = b.sigmoid()
    
    # add T Dimension
    g = g.unsqueeze(1)
    beta = beta.unsqueeze(1)

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
    
    if scale is not None:
        q = scale * q
    
    # breakpoint()
    if cu_seqlens is not None:
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        assert (lens == lens[0]).all(), "Torch Reference is NOT supported varlen yet!"
        T, B = q.shape[:2]
        q, k, v, g, beta = map(lambda x: x.view(B, T, *x.shape[2:]), [q, k, v, g, beta])

    # gather State
    initial_state = initial_state_source[initial_state_indices].contiguous()
    
    o, new_S = torch_fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state)
    initial_state_source[initial_state_indices] = initial_state
    return o    




def add_descriptor_to_name(prefix_name: str) -> Callable:
    '''
    This function's purpose is to generate a specific name based on the constant parameters of the Triton kernel.

    '''
    def descriptor_wrapper(proxy) -> str:
        constants = proxy.constants
        descriptors = list()
        descriptors.append(f"B{constants['B']}")
        descriptors.append(f"H{constants['H']}")
        descriptors.append(f"HV{constants['HV']}")
        descriptors.append(f"K{constants['K']}")
        descriptors.append(f"V{constants['V']}")
        descriptors.append(f"BK{constants['BK']}")
        descriptors.append(f"BV{constants['BV']}")
        descriptors.append(f"INIT{int(constants['USE_INITIAL_STATE'])}")
        descriptors.append(f"NORM{int(constants['USE_QK_L2NORM_IN_KERNEL'])}")
        descriptors.append(f"VARLEN{int(constants['IS_VARLEN'])}")
        return f"{prefix_name}_{'_'.join(descriptors)}"

    return descriptor_wrapper


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel_fast(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    cta_group_size: tl.constexpr,
    num_warps: tl.constexpr,
    NUM_STAGES: tl.constexpr, 
    loop_unroll_factor: tl.constexpr
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k = 0
    i_v, i_group, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    group_size = (HV // H)
    i_hv = i_h * group_size + i_group * cta_group_size

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    
    tl.device_assert(T == 1)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k

    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    # Load inputs
    b_q = tl.load(p_q, ).to(tl.float32)
    b_k = tl.load(p_k, ).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
    b_q = b_q * scale

    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)

    for _ in tl.range(0, cta_group_size, loop_unroll_factor=loop_unroll_factor, num_stages=NUM_STAGES):
        if USE_INITIAL_STATE:        
            p_h0 = (h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :])
            valid_mask = (idx >= 0)
            b_h = tl.load(p_h0, mask=valid_mask, other=0).to(tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)

        b_v = tl.load(p_v,).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = b_b.sigmoid()

        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))

        # Store final state back to h0_source with bounds checking
        if USE_INITIAL_STATE:
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=valid_mask)
        p_o += V
        p_v += V
        p_b += 1
        p_a += 1
        p_A_log += 1
        p_dt_bias += 1
        i_hv += 1

def smallest_divisor(n: int) -> int:
    """
    Find the smallest integer (greater than 1) that divides the positive integer n.
    If none exists (i.e. n <= 1), return 1.
    
    Args:
        n: a positive integer
        
    Returns:
        the smallest integer (greater than 1) dividing n, or 1 if none exists
    """
    if n <= 1:
        return 1
    
    # search for the smallest factor starting from 2
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i
    
    # if n is prime, n itself is the smallest factor greater than 1
    return 1

@functools.lru_cache
def get_default_config(N, H, HV):
    assert HV % H == 0 and HV >= H
    num_v_groups = HV // H
    num_ctas = N * H
    total_groups = N * num_v_groups

    if total_groups >= 512:
        config = {'BV': 32, 'num_warps': 2, 'loop_unroll_factor': 1, 'NUM_STAGES': 2, 'cta_group_size': 1}
    elif total_groups >= 128:
        config = {'BV': 32, 'num_warps': 2, 'loop_unroll_factor': 1, 'NUM_STAGES': 1, 'cta_group_size': smallest_divisor(num_v_groups)}
    else:
        config = {'BV': 16, 'num_warps': 2, 'loop_unroll_factor': 2, 'NUM_STAGES': 2, 'cta_group_size': 1}
    
    return config


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_opt: bool = True,
    is_kda: bool = False
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if SGLANG_PROFILE:
        th_nvtx_range_push(
            f"[GDN]:[D][{N}]"
            f"gdn_attn:seqlen_q:[{T}],"
            f"head_dim_qk:{K},"
            f"head_dim_v:{V},"
            f"num_heads_qk:{H},"
            f"num_heads_v:{HV},"
            f"hidden_act:silu,conv_kdim:4,batch_size:{N},dtype:bf16."
            f"use_qk_l2norm_in_kernel:{use_qk_l2norm_in_kernel},"
            f"use_opt:{use_opt},"
            f"initial_state_indices type:{initial_state_indices.dtype},"
            f"initial_state_source type:{initial_state_source.dtype},"
            f"q data type:{q.dtype},"
            f"k data type:{k.dtype},"
            f"v data type:{v.dtype},"
            f"b data type:{b.dtype},"
            f"A_log data type:{A_log.dtype},"
            f"a data type:{a.dtype},"
            f"dt_bias data type:{dt_bias.dtype},"
            f"cu_seqlens data type:{cu_seqlens.dtype},"
        )

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    o = q.new_empty(NK, *v.shape)

    if use_opt == False:
        from pla.decode.gdn.v_last import fused_sigmoid_gating_delta_rule

        fused_sigmoid_gating_delta_rule(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            o=o,
            h0_source=initial_state_source,
            h0_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            T=int(T),
            B=int(N),
            H=int(H),
            HV=int(HV),
            K=int(K),
            V=int(V),
            use_initial_state=(initial_state_source is not None),
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_varlen=(cu_seqlens is not None)
        )
        if SGLANG_PROFILE:
            th_nvtx_range_pop()
        o = o.squeeze(0)
        return o

    if 'SAIL_SGL_PLA_OPT' in os.environ:
        use_opt = (int(os.environ.get("SAIL_SGL_PLA_OPT", 0)) == 1)

    config = get_default_config(N, H, HV)
    BK = triton.next_power_of_2(K)
    if BK != K or V % config['BV'] != 0 or is_kda:
        use_opt = False

    if use_opt:
        kernel_func = fused_sigmoid_gating_delta_rule_update_kernel_fast
        NK = triton.cdiv(K, BK)
        assert NK == 1, "NK > 1 is not supported yet"
        num_v_groups = HV // H
        grid = lambda meta: (triton.cdiv(meta["V"], meta["BV"]), 
                            num_v_groups // meta["cta_group_size"], 
                            N * meta["H"])
    else:
        kernel_func = fused_sigmoid_gating_delta_rule_update_kernel
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        assert NK == 1, "NK > 1 is not supported yet"
        config = {"BV": BV, "num_stages": 3, "num_warps": 2}
        grid = (NK, NV, N * HV)

    kernel_func[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        **config
    )
    if SGLANG_PROFILE:
        th_nvtx_range_pop()

    o = o.squeeze(0)
    return o


def benchmark_func_time(func, warmup_iters=3, num_iterations=8):
    
    for i in range(warmup_iters):
        with torch.cuda.nvtx.range(f"Warmup Iter {i}"):
            o = func()
    
    cudagraph = torch.cuda.CUDAGraph()
    num_replays = 8
    with torch.no_grad(), torch.cuda.graph(cudagraph):
        for i in range(num_replays):
            o = func()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    kernel_times = list()
    with torch.cuda.nvtx.range("main-test"):
        for i in range(num_iterations):
            # Fix 1: create new Event objects every iteration to avoid reuse problems
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)

            # Fix 2: start_event.record() sits right next to the kernel launch, undisturbed by NVTX
            with torch.cuda.nvtx.range(f"Perf-Iter {i}"):
                start_event.record()       # <- right before the kernel
                cudagraph.replay()
                end_event.record()         # <- right after the kernel

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)

            kernel_times.append(elapsed_time / num_replays)

    # Fix 4: warmup is already done separately, so there is no need to discard the first one
    kernel_times = pd.Series(data=kernel_times)
    average = kernel_times.mean()
    print(f"kernel_time = {average*1000:.2f} us")


def tensor_diff_summary(a, b, atol=1e-8, rtol=1e-5, max_show=5):
    if a.shape != b.shape:
        return f"shape mismatch: {a.shape} vs {b.shape}"
    
    if a.dtype != b.dtype:
        print(f"Warning: dtypes differ - a.dtype={a.dtype}, b.dtype={b.dtype}")
    
    if a.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        diff_mask = ~torch.isclose(a, b, atol=atol, rtol=rtol)
    else:
        diff_mask = a != b
    
    n_diff = torch.sum(diff_mask).item()
    
    if n_diff == 0:
        if a.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            return f"the two Tensors are equal within tolerance (atol={atol}, rtol={rtol})"
        else:
            return "the two Tensors are identical"
    
    total_elements = a.numel()
    diff_percentage = (n_diff / total_elements) * 100
    
    summary = f"found {n_diff} differences ({total_elements} elements total, {diff_percentage:.2f}%)"
    
    if a.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        summary += f"\ntolerance: atol={atol}, rtol={rtol}"
    
    if n_diff <= max_show:
        indices = torch.nonzero(diff_mask)
        values_a = a[diff_mask]
        values_b = b[diff_mask]
        
        details = []
        for i in range(n_diff):
            if len(a.shape) == 1:
                idx_str = f"[{indices[i].item()}]"
            else:
                idx_str = "[" + ", ".join(map(str, indices[i].tolist())) + "]"
            
            abs_diff = abs(values_a[i].item() - values_b[i].item())
            if abs(values_b[i].item()) > 1e-12:  
                rel_diff = abs_diff / abs(values_b[i].item())
            else:
                rel_diff = float('inf')
            
            details.append(
                f"{idx_str}: {values_a[i].item():.6g} vs {values_b[i].item():.6g} "
                f"(abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2e})"
            )
        
        summary += "\ndetailed differences:\n" + "\n".join(details)
    elif max_show > 0:
        indices = torch.nonzero(diff_mask)[:max_show]
        values_a = a[diff_mask][:max_show]
        values_b = b[diff_mask][:max_show]
        
        details = []
        for i in range(len(values_a)):
            if len(a.shape) == 1:
                idx_str = f"[{indices[i].item()}]"
            else:
                idx_str = "[" + ", ".join(map(str, indices[i].tolist())) + "]"
            
            abs_diff = abs(values_a[i].item() - values_b[i].item())
            if abs(values_b[i].item()) > 1e-12:
                rel_diff = abs_diff / abs(values_b[i].item())
            else:
                rel_diff = float('inf')
            
            details.append(
                f"{idx_str}: {values_a[i].item():.6g} vs {values_b[i].item():.6g} "
                f"(abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2e})"
            )
        
        summary += f"\nfirst {max_show} example differences:\n" + "\n".join(details)
        summary += f"\n... {n_diff - max_show} more differences not shown"
    
    return summary


def batch_test(batch_size: int, tp_size: int, model_config: Dict, is_benchmark=False, is_varlen=True, dtype=torch.bfloat16, alogtype=torch.float32, is_shuffle=True):

    dk = model_config["linear_key_head_dim"]
    hk = model_config["linear_num_key_heads"]
    dv = model_config["linear_value_head_dim"]
    hv = model_config["linear_num_value_heads"]
    assert (hk % hv == 0 or hv % hk == 0)
    assert (hk % tp_size == 0 or hv % tp_size == 0)
    assert hk <= hv, "Not support group v"
    hk = hk // tp_size
    hv = hv // tp_size

    print(f"Benchmark parameter: batch_size={batch_size}, tp_size={tp_size}")
    print(f"in single device: hk={hk}, hv={hv}, dk={dk}, dv={dv}, dtype={dtype}")

    if is_varlen:
        B, T = 1, batch_size
    else:
        B, T = batch_size, 1
    
    q = randn_with_dtype(B, T, hk, dk, dtype=dtype, device="cuda")
    k = randn_with_dtype(B, T, hk, dk, dtype=dtype, device="cuda")
    v = randn_with_dtype(B, T, hv, dv, dtype=dtype, device=q.device)
    # o = torch.empty_like(v)

    a = randn_with_dtype(batch_size, hv, dtype=dtype, device=q.device)
    b = randn_with_dtype(batch_size, hv, dtype=dtype, device=q.device)

    A_log = randn_with_dtype(hv, dtype=alogtype, device=q.device)
    dt_bias = randn_with_dtype(hv, dtype=dtype, device=q.device)
    softplus_beta = 1.0
    softplus_threshold = 20.0
    scale = 0.08838834764831845

    mamba_pool_size = int(batch_size*1.25)
    initial_state_source = torch.randn(mamba_pool_size, hv, dk, dv, dtype=torch.float32, device=q.device)
    if is_shuffle:
        initial_state_indices = torch.randperm(mamba_pool_size)[:batch_size].to(torch.int32).to(device=q.device)
    else:
        initial_state_indices = torch.arange(batch_size, device=q.device, dtype=torch.int32)
    cu_seqlens = torch.arange(batch_size+1, device=q.device, dtype=torch.int32) if is_varlen else None


    def check_acc():
        initial_state_source_T = initial_state_source.clone()
        o_T = fused_sigmoid_gating_delta_rule_update(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, softplus_beta=softplus_beta, softplus_threshold=softplus_threshold, 
                        scale=scale, initial_state_source=initial_state_source_T, initial_state_indices=initial_state_indices, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True, 
                        use_opt=True)
        
        initial_state_source_R = initial_state_source.clone()
        o_R = fused_sigmoid_gating_delta_rule_update(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, softplus_beta=softplus_beta, softplus_threshold=softplus_threshold, 
                        scale=scale, initial_state_source=initial_state_source_R, initial_state_indices=initial_state_indices, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True, 
                        use_opt=False)

        atol=1e-3
        rtol=1e-2
        print(tensor_diff_summary(initial_state_source_T, initial_state_source_R, atol=atol, rtol=rtol, max_show=16))
        print(tensor_diff_summary(o_T, o_R, atol=1e-3, rtol=1e-2, max_show=16))

    if not is_benchmark:
        check_acc()

    def run_func():
        return fused_sigmoid_gating_delta_rule_update(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, softplus_beta=softplus_beta, softplus_threshold=softplus_threshold, 
                        scale=scale, initial_state_source=initial_state_source, initial_state_indices=initial_state_indices, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True, 
                        use_opt=True)
    if is_benchmark:
        benchmark_func_time(run_func)
    
def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,

        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,

        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }

    key = dtype_str.lower()
    if key not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. "
            f"Supported: {sorted(dtype_map.keys())}"
        )
    return dtype_map[key]

def main():
    # add the argument parser
    parser = argparse.ArgumentParser(description="fused_sigmoid_gating_delta_rule_update Benchmark")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark mode"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=129,
        help="Batch size (default: 129)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="TP size (default: 4)"
    )
    parser.add_argument(
        "--qwen35",
        action="store_true",
        help="Test Qwen35 model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="bf16/fp16/fp32",
    )
    parser.add_argument(
        "--alogtype",
        type=str,
        default="fp32",
        help="bf16/fp16/fp32",
    )
    args = parser.parse_args()

    print(f"*--- fused_sigmoid_gating_delta_rule_update Benchmark ---*")
    print(f"OpenAI Triton Version: {triton.__version__}")

    # Qwen3-Next
    model_config_qwen3_next = {
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
    }

    model_config_qwen35 = {
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_value_head_dim": 128,
    }

    dtype = parse_dtype(args.dtype)
    alogtype = parse_dtype(args.alogtype)

    torch.manual_seed(42)
    torch.cuda.empty_cache()
    BBA = torch.randn(300*1000*1000, dtype=torch.float32).to(device=torch.device("cuda"))

    if args.qwen35:
        batch_test(args.batch, args.tp, model_config_qwen35, args.benchmark, dtype=dtype, alogtype=alogtype)
    else:
        batch_test(args.batch, args.tp, model_config_qwen3_next, args.benchmark, dtype=dtype, alogtype=alogtype)

if __name__ == "__main__":
    main()
