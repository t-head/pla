/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../ppu_torch_stream.h"
#include <hggc_runtime.h>
#include <cmath>

#include "common.cuh"
#include "k_last_shared_layout.h"
#include "sanity_check.h"
#include "bypass_exec.h"
#include "op_forge_case.h"
#include "sanity_checks.h"
#include <limits>
#include <unordered_set>
#include "generated/fused_sigmoid_gating_dispatch_k_last_packed.h"
#include "generated/tuned_dispatch.h"

#define CHECK_DTYPE(x, dataType) TORCH_CHECK(x.dtype() == dataType, #x "'s data type should be " #dataType)


static void launch_k_last_packed_impl(
    k_last_packed_launcher_t launcher, int shared_mem_size, dim3 grid, hggcStream_t stream,
    torch::Tensor& A_log, torch::Tensor& a, torch::Tensor& dt_bias,
    float softplus_beta, float softplus_threshold,
    torch::Tensor& mixed_qkv, torch::Tensor& b,
    torch::Tensor& out, void* h0_ptr,
    const int32_t* cu_seq_ptr, const int32_t* h0_idx_ptr,
    float scale_val, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm_in_kernel, bool is_sglang, bool is_kda
) {
    launcher(
        A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(),
        softplus_beta, softplus_threshold,
        mixed_qkv.data_ptr(), b.data_ptr(),
        out.data_ptr(), h0_ptr,
        cu_seq_ptr, h0_idx_ptr,
        scale_val, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm_in_kernel, is_sglang,
        is_kda,
        shared_mem_size, grid, stream
    );
}

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last_packed(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    float softplus_beta, float softplus_threshold,
    std::optional<float> scale,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    torch::Tensor& out,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    std::optional<bool> is_sglang
) {
    if (pla::bypass::enabled()) {
        std::ostringstream flags;
        flags << "is_kda=" << is_kda << " is_sglang=" << is_sglang.value_or(true)
              << " use_qk_l2norm=" << use_qk_l2norm_in_kernel
              << " softplus_beta=" << softplus_beta << " softplus_threshold=" << softplus_threshold;
        if (scale) flags << " scale=" << *scale;
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("k_last_packed", {
            {"mixed_qkv", &mixed_qkv}, {"a", &a}, {"b", &b}, {"A_log", &A_log},
            {"dt_bias", &dt_bias}, {"out", &out},
            {"state", initial_state_source ? &*initial_state_source : nullptr},
            {"slots", initial_state_indices ? &*initial_state_indices : nullptr},
            {"cu", cu_seqlens ? &*cu_seqlens : nullptr}}, flags.str());
        if (mixed_qkv.dim() == 2 && initial_state_source && initial_state_source->dim() == 4) {
            const int HV = initial_state_source->size(1), V = initial_state_source->size(2);
            const int K = initial_state_source->size(3);
            const int64_t qk_width = mixed_qkv.size(1) - static_cast<int64_t>(HV) * V;
            if (K > 0 && qk_width > 0 && qk_width % (2 * K) == 0) {
                const int B = cu_seqlens && cu_seqlens->dim() == 1 ? cu_seqlens->numel() - 1 : mixed_qkv.size(0);
                if (pla::instrument::show_params()) pla::instrument::dump_case(B, 1, K, V, qk_width / (2 * K), HV,
                    get_dtype_string(mixed_qkv), is_kda, cu_seqlens ? &*cu_seqlens : nullptr);
            }
        }
        return out;
    }
    sanity_packed_inputs(mixed_qkv, a, b, A_log, dt_bias, softplus_beta, softplus_threshold,
        scale, initial_state_source, initial_state_indices, out, cu_seqlens, is_kda,
        is_sglang.value_or(true));
    // sglang keeps slot 0 as a valid scratch row, vLLM reserves it as null_block.
    const bool is_sglang_val = is_sglang.value_or(true);
    int B = cu_seqlens.has_value() ? (cu_seqlens.value().size(0) - 1) : mixed_qkv.size(0);
    int T_max = 1;

    TORCH_CHECK(initial_state_source.has_value(), "initial_state_source is required for k_last_packed");
    TORCH_CHECK(initial_state_source->dim() == 4,
                "initial_state_source must have shape [slots, HV, V, K]");
    int HV = initial_state_source->size(-3);
    int V = initial_state_source->size(-2);
    int K = initial_state_source->size(-1);
    TORCH_CHECK(K == 128 && V == 128,
                "fused_sigmoid_gating_delta_rule_forward_k_last_packed requires "
                "head_dim K == V == 128, got K=", K, ", V=", V);

    int H = check_packed_heads(mixed_qkv, a, b, A_log, dt_bias, out, HV, K, V, is_kda);

    if (g_ppu_lib_perf_instrument || g_ppu_lib_show_params) {
        const torch::Tensor* dump_cu_seqlens =
            cu_seqlens.has_value() ? &cu_seqlens.value() : nullptr;
        std::string nvtx_tag = generate_nvtx_tag(
            B, T_max, K, V, H, HV, get_dtype_string(mixed_qkv), is_kda,
            dump_cu_seqlens);
        push_nvtx_and_print(nvtx_tag, true);
    }

    float scale_val = scale.has_value() ? static_cast<float>(scale.value()) : (1.0f / sqrtf((float)K));
    int stride_a = a.stride(a.dim() - 2);
    int stride_qkv = mixed_qkv.stride(0);
    int stride_b = b.stride(b.dim() - 2);

    at::ScalarType state_dtype = initial_state_source.value().scalar_type();
    TORCH_CHECK(state_dtype == torch::kFloat32 || state_dtype == torch::kBFloat16,
                "initial_state_source dtype not supported: ", state_dtype);
    TunedParams tp = get_tuned_k_last_packed(
        B, H, HV, K, V, state_dtype == torch::kBFloat16);

    int hv_per_h = HV / H;
    TORCH_CHECK(tp.bv > 0 && tp.wn > 0 && tp.sn > 0 && V % tp.bv == 0,
                "invalid k-last tile parameters");
    TORCH_CHECK(tp.hv_per_cta > 0 && tp.hv_per_cta <= hv_per_h && hv_per_h % tp.hv_per_cta == 0,
                "HV_PER_CTA must divide HV/H");
    int NV = V / tp.bv;
    dim3 grid(NV, hv_per_h / tp.hv_per_cta, B * H);

    const int32_t* cu_seq_ptr = nullptr;
    const int32_t* h0_idx_ptr = nullptr;

    if (cu_seqlens.has_value()) { CHECK_DTYPE(cu_seqlens.value(), torch::kInt32); cu_seq_ptr = cu_seqlens.value().data_ptr<int32_t>(); }
    if (initial_state_indices.has_value()) { CHECK_DTYPE(initial_state_indices.value(), torch::kInt32); h0_idx_ptr = initial_state_indices.value().data_ptr<int32_t>(); }

    void* initial_state_source_ptr = nullptr;
    // Slot-stride of the state buffer actually passed to the kernel. vLLM
    // packs conv_state and ssm_state into one page, so the slot stride of its
    // ssm_state view is larger than HV*V*K; SGLang's state is contiguous.
    int64_t h0_slot_stride = 0;
    if (initial_state_source.has_value()) {
        // Pass pointer directly — kernel handles both fp32 and bf16 natively.
        initial_state_source_ptr = initial_state_source.value().data_ptr();
        h0_slot_stride = initial_state_source.value().stride(0);
    }

    // H-staging area: bf16 uses TASKS_PER_STAGE=2, so bytes = 2 * 2 * sn * K * wn = 4 * sn * K * wn = same as fp32.
    // Always allocate sizeof(float) * sn * K * wn for the H-staging, which covers both dtypes.
    const pla::KLastSharedLayout shared_layout(tp.sn, K, tp.wn, tp.bv, tp.hv_per_cta,
        is_kda, false, T_max);
    TORCH_CHECK(shared_layout.total_bytes <= std::numeric_limits<int>::max(), "shared allocation exceeds int range");
    const int shared_mem_size = static_cast<int>(shared_layout.total_bytes);

    k_last_packed_launcher_t launcher = get_k_last_packed_launcher_dispatch(
        tp.bv, tp.wn, tp.sn, tp.hv_per_cta,
        mixed_qkv.scalar_type(), A_log.scalar_type(), state_dtype);
    TORCH_CHECK(launcher != nullptr, "No kernel instantiation for BV=", tp.bv, " WN=", tp.wn, " SN=", tp.sn, " HV_PER_CTA=", tp.hv_per_cta, " state_dtype=", state_dtype);

    launch_k_last_packed_impl(launcher, shared_mem_size, grid, pla::current_stream(),
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        mixed_qkv, b, out, initial_state_source_ptr,
        cu_seq_ptr, h0_idx_ptr, scale_val, T_max,
        stride_a, stride_qkv, stride_b, h0_slot_stride, B, H, HV, K, V, use_qk_l2norm_in_kernel, is_sglang_val, is_kda
    );

    return out;
}

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    float softplus_beta, float softplus_threshold,
    std::optional<float> scale,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    torch::Tensor& out,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    int bv, int wn, int sn, int hv_per_cta,
    std::optional<bool> is_sglang
) {
    if (pla::bypass::enabled()) {
        std::ostringstream flags;
        flags << "is_kda=" << is_kda << " is_sglang=" << is_sglang.value_or(true)
              << " use_qk_l2norm=" << use_qk_l2norm_in_kernel
              << " softplus_beta=" << softplus_beta << " softplus_threshold=" << softplus_threshold;
        if (scale) flags << " scale=" << *scale;
        flags << " bv=" << bv << " wn=" << wn << " sn=" << sn << " hv_per_cta=" << hv_per_cta;
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("k_last_packed_tunable", {
            {"mixed_qkv", &mixed_qkv}, {"a", &a}, {"b", &b}, {"A_log", &A_log},
            {"dt_bias", &dt_bias}, {"out", &out},
            {"state", initial_state_source ? &*initial_state_source : nullptr},
            {"slots", initial_state_indices ? &*initial_state_indices : nullptr},
            {"cu", cu_seqlens ? &*cu_seqlens : nullptr}}, flags.str());
        if (mixed_qkv.dim() == 2 && initial_state_source && initial_state_source->dim() == 4) {
            const int HV = initial_state_source->size(1), V = initial_state_source->size(2);
            const int K = initial_state_source->size(3);
            const int64_t qk_width = mixed_qkv.size(1) - static_cast<int64_t>(HV) * V;
            if (K > 0 && qk_width > 0 && qk_width % (2 * K) == 0) {
                const int B = cu_seqlens && cu_seqlens->dim() == 1 ? cu_seqlens->numel() - 1 : mixed_qkv.size(0);
                if (pla::instrument::show_params()) pla::instrument::dump_case(B, 1, K, V, qk_width / (2 * K), HV,
                    get_dtype_string(mixed_qkv), is_kda, cu_seqlens ? &*cu_seqlens : nullptr);
            }
        }
        return out;
    }
    sanity_packed_inputs(mixed_qkv, a, b, A_log, dt_bias, softplus_beta, softplus_threshold,
        scale, initial_state_source, initial_state_indices, out, cu_seqlens, is_kda,
        is_sglang.value_or(true));
    const bool is_sglang_val = is_sglang.value_or(true);
    int B = cu_seqlens.has_value() ? (cu_seqlens.value().size(0) - 1) : mixed_qkv.size(0);
    int T_max = 1;
    TORCH_CHECK(initial_state_source.has_value(), "initial_state_source is required for k_last_packed");
    TORCH_CHECK(initial_state_source->dim() == 4,
                "initial_state_source must have shape [slots, HV, V, K]");
    int HV = initial_state_source->size(-3);
    int V = initial_state_source->size(-2);
    int K = initial_state_source->size(-1);
    TORCH_CHECK(K == 128 && V == 128,
                "fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable requires "
                "head_dim K == V == 128, got K=", K, ", V=", V);
    int H = check_packed_heads(mixed_qkv, a, b, A_log, dt_bias, out, HV, K, V, is_kda);

    float scale_val = scale.has_value() ? static_cast<float>(scale.value()) : (1.0f / sqrtf((float)K));
    int stride_a = a.stride(a.dim() - 2);
    int stride_qkv = mixed_qkv.stride(0);
    int stride_b = b.stride(b.dim() - 2);

    int hv_per_h = HV / H;
    TORCH_CHECK(bv > 0 && wn > 0 && sn > 0 && V % bv == 0,
                "invalid k-last tile parameters");
    TORCH_CHECK(hv_per_cta > 0 && hv_per_cta <= hv_per_h && hv_per_h % hv_per_cta == 0,
                "HV_PER_CTA must divide HV/H");
    int NV = V / bv;
    dim3 grid(NV, hv_per_h / hv_per_cta, B * H);

    const int32_t* cu_seq_ptr = nullptr;
    const int32_t* h0_idx_ptr = nullptr;
    if (cu_seqlens.has_value()) { CHECK_DTYPE(cu_seqlens.value(), torch::kInt32); cu_seq_ptr = cu_seqlens.value().data_ptr<int32_t>(); }
    if (initial_state_indices.has_value()) { CHECK_DTYPE(initial_state_indices.value(), torch::kInt32); h0_idx_ptr = initial_state_indices.value().data_ptr<int32_t>(); }

    void* initial_state_source_ptr = nullptr;
    at::ScalarType state_dtype = torch::kFloat32;
    // Slot-stride of the state buffer actually passed to the kernel. vLLM
    // packs conv_state and ssm_state into one page, so the slot stride of its
    // ssm_state view is larger than HV*V*K; SGLang's state is contiguous.
    int64_t h0_slot_stride = 0;
    if (initial_state_source.has_value()) {
        state_dtype = initial_state_source.value().scalar_type();
        if (state_dtype == torch::kFloat32 || state_dtype == torch::kBFloat16) {
            // Pass pointer directly — kernel handles both fp32 and bf16 natively
            initial_state_source_ptr = initial_state_source.value().data_ptr();
            h0_slot_stride = initial_state_source.value().stride(0);
        } else {
            AT_ERROR("initial_state_source dtype not supported: ", state_dtype);
        }
    }

    const pla::KLastSharedLayout shared_layout(sn, K, wn, bv, hv_per_cta,
        is_kda, false, T_max);
    TORCH_CHECK(shared_layout.total_bytes <= std::numeric_limits<int>::max(), "shared allocation exceeds int range");
    const int shared_mem_size = static_cast<int>(shared_layout.total_bytes);

    k_last_packed_launcher_t launcher = get_k_last_packed_launcher_dispatch(
        bv, wn, sn, hv_per_cta, mixed_qkv.scalar_type(), A_log.scalar_type(), state_dtype);
    TORCH_CHECK(launcher != nullptr, "No kernel instantiation for BV=", bv, " WN=", wn, " SN=", sn, " HV_PER_CTA=", hv_per_cta, " state_dtype=", state_dtype);

    launch_k_last_packed_impl(launcher, shared_mem_size, grid, pla::current_stream(),
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        mixed_qkv, b, out, initial_state_source_ptr,
        cu_seq_ptr, h0_idx_ptr, scale_val, T_max,
        stride_a, stride_qkv, stride_b, h0_slot_stride, B, H, HV, K, V, use_qk_l2norm_in_kernel, is_sglang_val, is_kda
    );

    return out;
}
