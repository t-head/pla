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

import sys
import os
import time
import argparse
import torch

# Add scripts directory to path to find fused_sigmoid_gating_recurrent
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last

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

def _make_tensors(
    N, T, H, HV, K, V, is_kda, device="cuda", seed=2025,
    dtype=torch.bfloat16, alogtype=torch.float32, cu_seqlens_values=None,
):
    """Create packed input tensors for GDN."""
    torch.manual_seed(seed)
    if cu_seqlens_values is None:
        total_tokens = N * T
        cu_seqlens = torch.arange(0, total_tokens + 1, T, dtype=torch.int32, device=device)
    else:
        if len(cu_seqlens_values) != N + 1:
            raise ValueError("cu_seqlens size must equal batch size + 1")
        total_tokens = cu_seqlens_values[-1]
        cu_seqlens = torch.tensor(cu_seqlens_values, dtype=torch.int32, device=device)
    A_log = torch.randn(HV, dtype=alogtype, device=device)
    dt_bias = torch.randn(HV, dtype=dtype, device=device)
    a = torch.randn(1, total_tokens, HV, dtype=dtype, device=device)
    b = torch.randn(1, total_tokens, HV, dtype=dtype, device=device)
    q = torch.randn(1, total_tokens, H, K, dtype=dtype, device=device)
    k = torch.randn(1, total_tokens, H, K, dtype=dtype, device=device)
    v = torch.randn(1, total_tokens, HV, V, dtype=dtype, device=device)
    indices = torch.arange(N, dtype=torch.int32, device=device)
    initial_state = torch.randn(N, HV, K, V, dtype=torch.float, device=device)

    if is_kda:
        a = torch.randn(1, total_tokens, HV, K, dtype=dtype, device=device)
        # Only the KDA MTP-verify kernel reads dt_bias as fp32; pure decode reads it
        # at the activation dtype. These callers pass no cache buffers, so the path
        # is decided by T_max alone, which collapses to one token per request.
        dt_bias_dtype = dtype if total_tokens == N else torch.float32
        dt_bias = torch.randn(HV, K, dtype=dt_bias_dtype, device=device)
    return A_log, dt_bias, a, b, q, k, v, initial_state, indices, cu_seqlens

def test_correctness(B=64, H=16, HV=64, K=128, V=128, dtype=torch.float16, alogtype=torch.float32):
    torch.manual_seed(42)
    device = "cuda"
    T = 1
    is_kda = False

    print(f"Testing Decode Step (T=1) Correctness: B={B}, H={H}, HV={HV}, K={K}, V={V}, dtype={dtype}...")

    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = _make_tensors(
        B, T, H, HV, K, V, is_kda, dtype=dtype, alogtype=alogtype
    )

    # s_triton = initial_state_source.clone()
    s_triton = state.clone()
    from fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as triton_fused_gating
    )

    out_triton = triton_fused_gating(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
        s_triton, indices, scale=1.0, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, is_kda=is_kda,
        disable_state_update=False
    )

    s_cuda = state.clone()
    out_cuda = fused_sigmoid_gating_delta_rule_forward_k_last(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
        s_cuda, indices, 1.0, True,
        cu_seqlens, is_kda,
        False, None, None, None, None
    )

    print(tensor_diff_summary(s_cuda, s_triton, atol=1e-3, rtol=1e-2, max_show=16))
    print(tensor_diff_summary(out_cuda, out_triton, atol=1e-3, rtol=1e-2, max_show=16))
    torch.testing.assert_close(s_cuda, s_triton, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(out_cuda, out_triton, atol=1e-3, rtol=1e-2)

    print("Decode Correctness Pass")

    return



def benchmark():
    device, dtype = "cuda", torch.bfloat16
    B, H, K, HV, V = 2, 8, 128, 4, 128

    print(f"\\nBenchmarking with K={K}, V={V}, H={H}, HV={HV}...")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['T'],
            x_vals=[1, 16, 64, 128, 256],
            line_arg='provider',
            line_vals=['triton', 'cuda'],
            line_names=['Triton Kernel', 'CUDA Custom Op'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='Time (ms)',
            plot_name='gdn-perf',
            args={'B': B, 'H': H, 'K': K, 'HV': HV, 'V': V}
        )
    )
    def bench(B, T, H, K, HV, V, provider):
        A_log = torch.randn(HV, dtype=torch.float32, device=device)
        is_kda = True
        a = torch.randn(B * T, HV, dtype=dtype, device=device)
        dt_bias = torch.randn(HV, dtype=dtype, device=device)
        if is_kda == True:
            a = torch.randn(B * T, HV, K, dtype=dtype, device=device)
            # fp32 dt_bias only once this becomes an MTP-verify call (T > 1).
            dt_bias = torch.randn(HV, K, device=device,
                                  dtype=torch.float32 if T > 1 else dtype)
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k = torch.randn(B, T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
        b = torch.randn(B, T, HV, dtype=dtype, device=device)
        init_src = torch.zeros(B, HV, V, K, dtype=torch.float32, device=device)
        init_idx = torch.arange(B, dtype=torch.int32, device=device)

        quantiles =[0.5, 0.2, 0.8]
        if provider == 'triton':
            return triton.testing.do_bench(
                lambda: triton_fused_gating(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b, init_src, init_idx, 1.0, True, None, True
                ), quantiles=quantiles
            )
        else:
            return triton.testing.do_bench(
                lambda: fused_sigmoid_gating_delta_rule_forward_k_last(
                    A_log, a, dt_bias, 1.0, 20.0, q, k, v, b, init_src, init_idx, 1.0, True, None, True
                ), quantiles=quantiles
            )

    bench.run(print_data=True, show_plots=False)

def benchmark_decodeonly(B=2, H=4, HV=8, K=128, V=128, dtype=torch.float16):
    T = 1
    device = "cuda"

    print(f"Benchmarking Configurations: Batch={B}, SeqLen={T}, H={H}, HV={HV}, K={K}, V={V}, dtype={dtype}")

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    b = torch.randn(B, T, HV, dtype=dtype, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)

    is_kda = False
    a = torch.randn(B * T, HV, dtype=dtype, device=device)
    dt_bias = torch.randn(HV, dtype=dtype, device=device)
    if is_kda == True:
        a = torch.randn(B * T, HV, K, dtype=dtype, device=device)
        # T == 1 and no cache buffers: pure decode, so dt_bias stays at q's dtype.
        dt_bias = torch.randn(HV, K, dtype=dtype, device=device)
    softplus_beta = 1.0
    softplus_threshold = 20.0
    use_qk_l2norm = True

    print("Benchmarking Custom CUDA...")
    num_warmups = 25
    num_iters = 100

    for _ in range(num_warmups):
        _ = fused_sigmoid_gating_delta_rule_forward_k_last(
            A_log, a, dt_bias, softplus_beta, softplus_threshold,
            q, k, v, b, None, None, None, use_qk_l2norm, None, is_kda,
            False, None, None, None, None
        )
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        _ = fused_sigmoid_gating_delta_rule_forward_k_last(
            A_log, a, dt_bias, softplus_beta, softplus_threshold,
            q, k, v, b, None, None, None, use_qk_l2norm, None, is_kda,
            False, None, None, None, None
        )
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / num_iters * 1000

    print(f" - Custom CUDA Kernel: {cuda_time:.3f} ms / iter")

    if has_triton:
        print("Benchmarking Triton...")
        num_warmups = 25
        num_iters = 100

        for _ in range(num_warmups):
            _ = triton_fused_gating(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b,
                initial_state_source=None,
                initial_state_indices=None,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                is_kda=is_kda
            )
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iters):
            _ = triton_fused_gating(
                A_log, a, dt_bias, softplus_beta, softplus_threshold,
                q, k, v, b,
                initial_state_source=None,
                initial_state_indices=None,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                is_kda=is_kda
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / num_iters * 1000

        print(f" - Triton Kernel: {cuda_time:.3f} ms / iter")

def parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
    }
    key = dtype_str.lower()
    if key not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {sorted(dtype_map.keys())}")
    return dtype_map[key]


def main():
    parser = argparse.ArgumentParser(description="k_last kernel correctness & benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Enable benchmark mode")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--tp", type=int, default=1, help="TP size (default: 1)")
    parser.add_argument("--qwen35", action="store_true", help="Test Qwen35 model (default for k_last)")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")
    parser.add_argument("--alogtype", type=str, default="fp32", help="bf16/fp16/fp32")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    alogtype = parse_dtype(args.alogtype)

    if args.qwen35:
        H, HV, K, V = 16, 64, 128, 128
    else:
        H, HV, K, V = 16, 32, 128, 128

    H = H // args.tp
    HV = HV // args.tp

    print(f"Config: batch={args.batch}, tp={args.tp}, H={H}, HV={HV}, K={K}, V={V}, dtype={dtype}, alogtype={alogtype}")

    test_correctness(B=args.batch, H=H, HV=HV, K=K, V=V, dtype=dtype, alogtype=alogtype)

    if args.benchmark:
        benchmark_decodeonly(B=args.batch, H=H, HV=HV, K=K, V=V, dtype=dtype)


if __name__ == "__main__":
    main()
    #
