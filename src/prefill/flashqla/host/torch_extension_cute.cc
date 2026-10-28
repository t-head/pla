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

/*
 * FlashQLA PPU1.5 CuTe — PyTorch extension binding for pla.prefill.flashqla.
 *
 * Compiled into pla/prefill/flashqla/_C. The Python facades split these ops
 * across two namespaces:
 *
 * All six are exposed under pla.prefill.flashqla.private; the public
 * pla.prefill.flashqla namespace only carries the end-to-end pipeline
 * chunk_gated_delta_rule_fwd (pure Python, in pla/prefill/flashqla/ops.py).
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "../../../ppu_torch_stream.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <mutex>

#include "op_forge_case.h"
#include "../../sanity_check.h"
#include "../../../bypass_exec.h"
#include "../../../op_forge_case.h"
#include "sanity_checks.h"

// ---- forward decls from kernels/*.cu ----
void launch_get_warmup_chunks_cute(
    const float* g, const signed char* ht_mask, const int* cu_seqlens,
    int* num_warmup_chunks, signed char* fallback_mask,
    int batch_size, int num_tokens, int H, float threshold,
    hggcStream_t stream);

void launch_kkt_solve_cute(
    const void* K, const float* beta, void* A, const int* cu_seqlens,
    const int* chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens, int dtype_tag,
    int device_index, int num_cu, int compute_capability,
    hggcStream_t stream);

void launch_correct_h0_no_raw_cute(
    void* cp_h0, const void* ht_buffer, const void* mt_buffer,
    const signed char* fallback_mask, const int* seq_map_r2c,
    int cp_batch_size, int raw_batch_size, int H, bool cp_h0_is_bf16,
    int dtype_tag, int device_index, int num_cu, hggcStream_t stream);

void launch_correct_h0_raw_cute(
    void* cp_h0, const void* raw_h0, const void* ht_buffer,
    const void* mt_buffer, const signed char* fallback_mask,
    const int* seq_map_r2c, const int* raw_h0_indices,
    int cp_batch_size, int raw_batch_size, int H, int raw_h0_k_last,
    bool raw_h0_is_bf16, bool cp_h0_is_bf16,
    int dtype_tag, int device_index, int num_cu, hggcStream_t stream);

void launch_prepare_h_cp_cute(
    const void* k, const void* v, const void* a,
    const float* g, const float* beta, void* ht, void* mt,
    const int* cu_seqlens, const int* num_warmup_chunks,
    int batch_size, int num_tokens, int H, int Hg, int dtype_tag,
    int64_t v_token_stride,
    int device_index, int num_cu, int compute_capability,
    hggcStream_t stream);

void launch_fused_fwd_cp_cute(
    void* o, const void* q, const void* k, const void* v,
    const void* a, const float* beta, const float* g,
    const void* h0, void* ht_out, void* raw_state,
    const void* raw_state_input, bool state_bf16,
    const int* raw_state_indices,
    const int* cu_seqlens, const int* cp_seq_map, const int* raw_cu_seqlens,
    const int* num_warmup_chunks, const signed char* fallback_mask,
    int batch_size, int num_tokens, int raw_batch_size,
    int H, int Hg, float scale, int dtype_tag,
    int64_t v_token_stride,
    int device_index, int num_cu, int compute_capability,
    hggcStream_t stream);

void launch_chunk_local_cumsum_cute(
    const float* g, const int* cu_seqlens, const int* chunk_indices,
    float* g_cum, int num_chunks, int H, hggcStream_t stream);

void launch_chunk_local_cumsum_qk_norm_cute(
    const float* g, const int* cu_seqlens, const int* chunk_indices,
    float* g_cum, const void* q, const void* k, void* q_norm, void* k_norm,
    int num_chunks, int T, int H, int Hg, int dtype_tag,
    hggcStream_t stream);

void launch_chunk_local_cumsum_qk_norm_gated_cute(
    const void* a, const void* b, const float* A_log, const float* dt_bias,
    const int* cu_seqlens, const int* chunk_indices,
    float* g_cum, float* beta,
    const void* q, const void* k, void* q_norm, void* k_norm,
    int num_chunks, int T, int H, int Hg,
    int64_t stride_a, int64_t stride_b,
    int64_t stride_q, int64_t stride_k, int dtype_tag,
    hggcStream_t stream);

namespace {

constexpr int kMaxCachedDevices = 64;

struct LogicalDeviceInfo {
    int index;
    int num_cu;
    int compute_capability;
};

LogicalDeviceInfo logical_device_info(torch::Tensor const& tensor) {
    const int device = tensor.get_device();
    auto query_device_info = [device]() {
        int num_cu = 0;
        int major = 0;
        int minor = 0;
        const hggcError_t cu_status = hggcDeviceGetAttribute(
            &num_cu, hggcDevAttrMultiProcessorCount, device);
        TORCH_CHECK(
            cu_status == hggcSuccess,
            "failed to query multiprocessor count for PPU device ", device,
            ": ", hggcGetErrorString(cu_status));
        TORCH_CHECK(
            num_cu > 0,
            "PPU device ", device,
            " reported invalid multiprocessor count ", num_cu);

        const hggcError_t major_status = hggcDeviceGetAttribute(
            &major, hggcDevAttrComputeCapabilityMajor, device);
        TORCH_CHECK(
            major_status == hggcSuccess,
            "failed to query compute capability major for PPU device ",
            device, ": ", hggcGetErrorString(major_status));
        const hggcError_t minor_status = hggcDeviceGetAttribute(
            &minor, hggcDevAttrComputeCapabilityMinor, device);
        TORCH_CHECK(
            minor_status == hggcSuccess,
            "failed to query compute capability minor for PPU device ",
            device, ": ", hggcGetErrorString(minor_status));

        const int compute_capability = major * 10 + minor;
        TORCH_CHECK(
            compute_capability == 80 || compute_capability == 89,
            "unsupported PPU compute capability ", major, ".", minor,
            " on PPU device ", device,
            "; expected 8.0 (tc01) or 8.9 (tc02)");
        return LogicalDeviceInfo{device, num_cu, compute_capability};
    };
    if (device < 0 || device >= kMaxCachedDevices) {
        return query_device_info();
    }
    // Device properties are stable for the lifetime of a logical device. The
    // once_flag both removes the launch-path runtime query and avoids a race
    // when multiple host threads first use the same device concurrently.
    static std::once_flag configured[kMaxCachedDevices];
    static LogicalDeviceInfo cached_info[kMaxCachedDevices] = {};
    std::call_once(
        configured[device],
        [&]() { cached_info[device] = query_device_info(); });
    return cached_info[device];
}

int g_ppu_lib_show_params = 0;

void init_show_params() {
    g_ppu_lib_show_params = pla::instrument::show_params();
}

int dtype_tag(torch::Tensor const& x) {
    if (x.dtype() == torch::kFloat16) return 0;
    if (x.dtype() == torch::kBFloat16) return 1;
    TORCH_CHECK(false, "tensor must be fp16 or bf16");
}

void* data_ptr_b16(torch::Tensor& x) {
    return x.data_ptr();
}

const void* data_ptr_b16(torch::Tensor const& x) {
    return x.data_ptr();
}

void check_same_b16_dtype(torch::Tensor const& ref, torch::Tensor const& x, const char* name) {
    TORCH_CHECK(x.dtype() == ref.dtype(), name, " must have same dtype as reference");
}

void check_same_cuda_device(torch::Tensor const& ref, torch::Tensor const& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(x.device() == ref.device(), name, " must be on ", ref.device(),
                ", got ", x.device());
}

bool is_supported_head_config(int64_t H, int64_t Hg) {
    return (H == 128 && Hg == 16) ||
           (H == 64 && (Hg == 16 || Hg == 8)) ||
           (H == 48 && Hg == 16) ||
           (H == 32 && (Hg == 16 || Hg == 8 || Hg == 4)) ||
           (H == 24 && Hg == 8) ||
           (H == 16 && (Hg == 8 || Hg == 4 || Hg == 2)) ||
           (H == 12 && Hg == 4) ||
           (H == 8 && (Hg == 4 || Hg == 2 || Hg == 1)) ||
           (H == 6 && Hg == 2) ||
           (H == 4 && Hg == 2);
}

namespace sp = pla::sanity::prefill;

// chunk_local_cumsum: forward chunk-local cumsum of the gate. Returns g_cum
// [1, T, H] fp32. Replaces the host-side Python per-chunk loop.
torch::Tensor chunk_local_cumsum(
    torch::Tensor g,           // [1, T, H] fp32 (raw log-gate)
    torch::Tensor cu_seqlens,  // [batch+1] int32
    torch::Tensor chunk_indices // [num_chunks,2] int32
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.chunk_local_cumsum",
            {{"g", &g}, {"cu_seqlens", &cu_seqlens}, {"chunk_indices", &chunk_indices}});
        return pla::bypass::output(g.sizes(), g.options());
    }
    const pla::sanity::Context sanity("flashqla.chunk_local_cumsum", g);
    if (sanity) {
        sanity.check(g.dim() == 3, "g must be [1,T,H]");
        sp::int_extent(sanity, g.size(1), "T");
        sp::int_extent(sanity, g.size(2), "H");
        sanity_gate(sanity, g, "g", g.size(1), g.size(2));
        const auto cu = sp::offsets(sanity, cu_seqlens, g.size(1));
        sp::chunks(sanity, chunk_indices, cu);
    }
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(g.dim() == 3, "g must be [1,T,H]");
    TORCH_CHECK(g.stride(2) == 1, "g must be contiguous in H");
    check_same_cuda_device(g, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(g, chunk_indices, "chunk_indices");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.dtype() == torch::kInt32,
                "chunk_indices must be int32");
    TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2,
                "chunk_indices must be [num_chunks,2]");
    TORCH_CHECK(chunk_indices.is_contiguous(), "chunk_indices must be contiguous");
    at::cuda::CUDAGuard device_guard(g.device());

    int H = g.size(2);
    int num_chunks = chunk_indices.size(0);

    auto g_cum = torch::empty_like(g);

    const hggcStream_t stream = pla::current_stream();
    launch_chunk_local_cumsum_cute(
        g.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
        chunk_indices.data_ptr<int>(), g_cum.data_ptr<float>(),
        num_chunks, H, stream);

    return g_cum;
}

std::vector<torch::Tensor> chunk_local_cumsum_qk_norm(
    torch::Tensor g,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices,
    torch::Tensor q,
    torch::Tensor k
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.chunk_local_cumsum_qk_norm",
            {{"g", &g}, {"q", &q}, {"k", &k}, {"cu_seqlens", &cu_seqlens},
             {"chunk_indices", &chunk_indices}});
        return {pla::bypass::output(g.sizes(), g.options()),
                pla::bypass::output(q.sizes(), q.options()),
                pla::bypass::output(k.sizes(), k.options())};
    }
    const pla::sanity::Context sanity("flashqla.chunk_local_cumsum_qk_norm", g);
    if (sanity) {
        sanity.check(g.dim() == 3 && q.dim() == 4, "g/q rank mismatch");
        sp::int_extent(sanity, g.size(1), "T");
        sp::int_extent(sanity, g.size(2), "H");
        sp::int_extent(sanity, q.size(2), "Hg");
        sanity_gate(sanity, g, "g", g.size(1), g.size(2));
        sanity_feature(sanity, q, "q", g.size(1), q.size(2), 128);
        sanity_feature(sanity, k, "k", g.size(1), q.size(2), 128);
        const auto cu = sp::offsets(sanity, cu_seqlens, g.size(1));
        sp::chunks(sanity, chunk_indices, cu);
    }
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(g.dim() == 3 && g.stride(2) == 1,
                "g must be [1,T,H] and contiguous in H");
    check_same_cuda_device(g, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(g, chunk_indices, "chunk_indices");
    check_same_cuda_device(g, q, "q");
    check_same_cuda_device(g, k, "k");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32,
                "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.dtype() == torch::kInt32,
                "chunk_indices must be int32");
    TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                chunk_indices.is_contiguous(),
                "chunk_indices must be contiguous [num_chunks,2]");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.sizes() == k.sizes(),
                "q and k must have the same [1,T,Hg,D] shape");
    TORCH_CHECK(q.size(0) == 1 && q.size(1) == g.size(1) && q.size(3) == 128,
                "q and k must be [1,T,Hg,128]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),
                "q and k must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q and k must be fp16 or bf16");
    check_same_b16_dtype(q, k, "k");
    at::cuda::CUDAGuard device_guard(g.device());

    auto g_cum = torch::empty_like(g);
    auto q_norm = torch::empty_like(q);
    auto k_norm = torch::empty_like(k);
    const hggcStream_t stream = pla::current_stream();
    launch_chunk_local_cumsum_qk_norm_cute(
        g.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
        chunk_indices.data_ptr<int>(), g_cum.data_ptr<float>(),
        data_ptr_b16(q), data_ptr_b16(k), data_ptr_b16(q_norm),
        data_ptr_b16(k_norm), chunk_indices.size(0), g.size(1), g.size(2),
        q.size(2), dtype_tag(q), stream);
    return {g_cum, q_norm, k_norm};
}

std::vector<torch::Tensor> chunk_local_cumsum_qk_norm_gated(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices,
    torch::Tensor q,
    torch::Tensor k,
    int64_t valid_num_tokens
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.chunk_local_cumsum_qk_norm_gated",
            {{"a", &a}, {"b", &b}, {"q", &q}, {"k", &k}, {"A_log", &A_log},
             {"dt_bias", &dt_bias}, {"cu_seqlens", &cu_seqlens}, {"chunk_indices", &chunk_indices}});
        auto options = a.options().dtype(torch::kFloat32);
        return {pla::bypass::output({1, a.size(0), a.size(1)}, options),
                pla::bypass::output({1, a.size(0), a.size(1)}, options),
                pla::bypass::output(q.sizes(), q.options()),
                pla::bypass::output(k.sizes(), k.options())};
    }
    const pla::sanity::Context sanity("flashqla.chunk_local_cumsum_qk_norm_gated", a);
    if (sanity) {
        sanity.check(a.dim() == 2 && q.dim() == 4, "a/q rank mismatch");
        sp::int_extent(sanity, a.size(0), "T");
        sp::int_extent(sanity, a.size(1), "H");
        sp::int_extent(sanity, q.size(2), "Hg");
        for (const auto& named : std::initializer_list<std::pair<const char*, torch::Tensor>>{
                 {"a", a}, {"b", b}, {"A_log", A_log}, {"dt_bias", dt_bias}})
            sanity.tensor(named.second, named.first);
        sanity_feature(sanity, q, "q", a.size(0), q.size(2), 128, true);
        sanity_feature(sanity, k, "k", a.size(0), q.size(2), 128, true);
        const auto cu = sp::offsets(sanity, cu_seqlens, a.size(0));
        sp::chunks(sanity, chunk_indices, cu);
        if (sanity.values_enabled())
            sanity.check(cu.back() == (valid_num_tokens < 0 ? a.size(0) : valid_num_tokens),
                         "valid_num_tokens must match cu_seqlens endpoint");
    }
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.sizes() == b.sizes(),
                "a and b must have the same [T,H] shape");
    TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1 &&
                    a.stride(0) >= a.size(1) && b.stride(0) >= b.size(1),
                "a and b must be head-contiguous with a positive token stride");
    TORCH_CHECK(a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16,
                "a and b must be fp16 or bf16");
    check_same_b16_dtype(a, b, "b");
    check_same_b16_dtype(a, q, "q");
    check_same_b16_dtype(a, k, "k");
    check_same_cuda_device(a, b, "b");
    check_same_cuda_device(a, A_log, "A_log");
    check_same_cuda_device(a, dt_bias, "dt_bias");
    check_same_cuda_device(a, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(a, chunk_indices, "chunk_indices");
    check_same_cuda_device(a, q, "q");
    check_same_cuda_device(a, k, "k");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32 &&
                dt_bias.dtype() == torch::kFloat32,
                "A_log and dt_bias must be fp32");
    TORCH_CHECK(A_log.dim() == 1 && dt_bias.dim() == 1 &&
                A_log.numel() == a.size(1) && dt_bias.numel() == a.size(1) &&
                A_log.is_contiguous() && dt_bias.is_contiguous(),
                "A_log and dt_bias must be contiguous [H]");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32,
                "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.dtype() == torch::kInt32 &&
                chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                chunk_indices.is_contiguous(),
                "chunk_indices must be contiguous int32 [num_chunks,2]");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.sizes() == k.sizes(),
                "q and k must have the same [1,T,Hg,D] shape");
    TORCH_CHECK(q.size(0) == 1 && q.size(1) == a.size(0) && q.size(3) == 128,
                "q and k must be [1,T,Hg,128]");
    const int64_t qk_tokens = valid_num_tokens < 0
        ? a.size(0) : valid_num_tokens;
    TORCH_CHECK(qk_tokens > 0 && qk_tokens <= a.size(0),
                "valid_num_tokens must be in (0,T]");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 &&
                    q.stride(2) == 128 && k.stride(2) == 128 &&
                    q.stride(1) >= q.size(2) * q.size(3) &&
                    k.stride(1) >= k.size(2) * k.size(3),
                "q and k must be head-contiguous with a positive token stride");
    at::cuda::CUDAGuard device_guard(a.device());

    auto fp32_options = a.options().dtype(torch::kFloat32);
    auto g_cum = torch::empty({1, a.size(0), a.size(1)}, fp32_options);
    auto beta = torch::empty_like(g_cum);
    // Raw Q/K may be views into one packed post-convolution row. The consumers
    // (KKT and recurrence) deliberately retain their original contiguous
    // normalized layout, so never preserve the packed input strides here.
    auto q_norm = torch::empty(q.sizes(), q.options());
    auto k_norm = torch::empty(k.sizes(), k.options());
    const hggcStream_t stream = pla::current_stream();
    launch_chunk_local_cumsum_qk_norm_gated_cute(
        data_ptr_b16(a), data_ptr_b16(b), A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
        chunk_indices.data_ptr<int>(), g_cum.data_ptr<float>(),
        beta.data_ptr<float>(), data_ptr_b16(q), data_ptr_b16(k),
        data_ptr_b16(q_norm), data_ptr_b16(k_norm),
        chunk_indices.size(0), qk_tokens, a.size(1), q.size(2),
        a.stride(0), b.stride(0), q.stride(1), k.stride(1),
        dtype_tag(q), stream);
    return {g_cum, beta, q_norm, k_norm};
}

// get_warmup_chunks: returns (num_warmup_chunks[batch,H] int32, fallback_mask[batch,H] int8)
std::vector<torch::Tensor> get_warmup_chunks(
    torch::Tensor g,           // [1, T, H] fp32 chunk-local cumsum gate
    torch::Tensor ht_mask,     // [batch] int8
    torch::Tensor cu_seqlens,  // [batch+1] int32
    double threshold
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.get_warmup_chunks",
            {{"g", &g}, {"ht_mask", &ht_mask}, {"cu_seqlens", &cu_seqlens}});
        const auto batch = cu_seqlens.numel() - 1;
        return {pla::bypass::output({batch, g.size(2)}, g.options().dtype(torch::kInt32)),
                pla::bypass::output({batch, g.size(2)}, g.options().dtype(torch::kInt8))};
    }
    const pla::sanity::Context sanity("flashqla.get_warmup_chunks", g);
    if (sanity) {
        sanity.check(g.dim() == 3, "g must be [1,T,H]");
        sp::int_extent(sanity, g.size(1), "T");
        sp::int_extent(sanity, g.size(2), "H");
        sanity_gate(sanity, g, "g", g.size(1), g.size(2));
        const auto cu = sp::offsets(sanity, cu_seqlens, g.size(1));
        const auto mask = sp::binary_mask(sanity, ht_mask, cu_seqlens.numel() - 1, "ht_mask");
        if (sanity.values_enabled())
            for (size_t i = 0; i < mask.size(); ++i)
                sanity.check(mask[i] || (cu[i + 1] - cu[i]) % 32 == 0,
                             "warmup segments without an external state must contain full 32-token chunks");
    }
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(g.dim() == 3, "g must be [1,T,H]");
    check_same_cuda_device(g, ht_mask, "ht_mask");
    check_same_cuda_device(g, cu_seqlens, "cu_seqlens");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(ht_mask.dtype() == torch::kInt8, "ht_mask must be int8");
    TORCH_CHECK(std::isfinite(threshold) && threshold < 0.0 &&
                    std::isfinite(static_cast<float>(threshold)),
                "threshold must be finite, negative, and representable as fp32");
    at::cuda::CUDAGuard device_guard(g.device());

    int T = g.size(1);
    int H = g.size(2);
    int batch_size = cu_seqlens.size(0) - 1;

    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(g.device());
    auto opt_i8  = torch::TensorOptions().dtype(torch::kInt8).device(g.device());
    auto num_warmup = torch::empty({batch_size, H}, opt_i32);
    auto fallback   = torch::empty({batch_size, H}, opt_i8);

    const hggcStream_t stream = pla::current_stream();
    launch_get_warmup_chunks_cute(
        g.data_ptr<float>(),
        reinterpret_cast<const signed char*>(ht_mask.data_ptr<int8_t>()),
        cu_seqlens.data_ptr<int>(),
        num_warmup.data_ptr<int>(),
        reinterpret_cast<signed char*>(fallback.data_ptr<int8_t>()),
        batch_size, T, H, static_cast<float>(threshold), stream);

    return {num_warmup, fallback};
}

// kkt_solve: A = (I + StrictLower(-beta * K K^T))^{-1}, per (chunk, head).
// Returns A [1, T, H, CS] fp16.
torch::Tensor kkt_solve(
    torch::Tensor K,           // [1, T, Hg, DK] fp16
    torch::Tensor beta,        // [1, T, H] fp32
    torch::Tensor cu_seqlens,  // [batch+1] int32
    int64_t H,
    c10::optional<torch::Tensor> chunk_indices_opt
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.kkt_solve",
            {{"K", &K}, {"beta", &beta}, {"cu_seqlens", &cu_seqlens},
             {"chunk_indices", chunk_indices_opt.has_value() ? &chunk_indices_opt.value() : nullptr}});
        return pla::bypass::output({1, K.size(1), H, 32}, K.options());
    }
    const pla::sanity::Context sanity("flashqla.kkt_solve", K);
    if (sanity) {
        sanity.check(K.dim() == 4, "K must be [1,T,Hg,128]");
        sp::int_extent(sanity, K.size(1), "T");
        sp::int_extent(sanity, H, "H");
        sanity.check(K.size(2) > 0 && H % K.size(2) == 0,
                     "H must be divisible by positive Hg");
        sanity_feature(sanity, K, "K", K.size(1), K.size(2), 128);
        sanity_gate(sanity, beta, "beta", K.size(1), H);
        const auto cu = sp::offsets(sanity, cu_seqlens, K.size(1));
        if (chunk_indices_opt.has_value())
            sp::chunks(sanity, chunk_indices_opt.value(), cu);
        sanity.check((chunk_indices_opt.has_value() ? chunk_indices_opt->size(0)
                                                    : (K.size(1) + 31) / 32) * H <=
                         std::numeric_limits<int>::max(),
                     "chunk/head task count exceeds kernel int32 ABI");
    }
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(K.dtype() == torch::kFloat16 || K.dtype() == torch::kBFloat16, "K must be fp16 or bf16");
    TORCH_CHECK(K.dim() == 4, "K must be [1,T,Hg,DK]");
    check_same_cuda_device(K, beta, "beta");
    check_same_cuda_device(K, cu_seqlens, "cu_seqlens");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be fp32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    at::cuda::CUDAGuard device_guard(K.device());

    constexpr int CS = 32;
    int T  = K.size(1);
    int Hg = K.size(2);
    int DK = K.size(3);
    TORCH_CHECK(DK == 128, "kkt_solve expects DK=128");
    int batch_size = cu_seqlens.size(0) - 1;
    int num_chunks = (T + CS - 1) / CS;
    const int* chunk_indices_ptr = nullptr;
    if (chunk_indices_opt.has_value()) {
        auto chunk_indices = chunk_indices_opt.value();
        check_same_cuda_device(K, chunk_indices, "chunk_indices");
        TORCH_CHECK(chunk_indices.dtype() == torch::kInt32,
                    "chunk_indices must be int32");
        TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2,
                    "chunk_indices must be [num_chunks,2]");
        num_chunks = chunk_indices.size(0);
        chunk_indices_ptr = chunk_indices.data_ptr<int>();
    } else {
        TORCH_CHECK(batch_size == 1,
                    "packed multi-sequence kkt_solve requires chunk_indices");
    }

    auto opt_e = torch::TensorOptions().dtype(K.dtype()).device(K.device());
    auto A = torch::empty({1, T, (int)H, CS}, opt_e);

    const auto device = logical_device_info(K);
    const hggcStream_t stream = pla::current_stream();
    launch_kkt_solve_cute(
        data_ptr_b16(K), beta.data_ptr<float>(), data_ptr_b16(A),
        cu_seqlens.data_ptr<int>(),
        chunk_indices_ptr,
        (int)H, Hg, num_chunks, T, dtype_tag(K),
        device.index, device.num_cu, device.compute_capability, stream);

    return A;
}

// correct_h0: affine-semigroup recurrence over CP segments. Returns cp_h0
// [cp_batch, H, DK, DV], fp32 or bf16 per out_bf16. raw_h0 optional (empty
// tensor -> no_raw variant). Indexed raw_h0 uses [slots,H,DV,DK]; legacy raw_h0
// uses [states,H,DK,DV].
//
// cp_h0 is fed straight back to fused_fwd as its h0, and fused_fwd carries a
// single compile-time state dtype, so the caller keeps cp_h0 in the pool's own
// dtype rather than a wider temporary.
torch::Tensor correct_h0(
    torch::Tensor ht_buffer,     // [cp_batch, H, DK, DV] fp16
    torch::Tensor mt_buffer,     // [cp_batch, H, DK, DK] fp16
    torch::Tensor fallback_mask, // [cp_batch * H] int8
    torch::Tensor seq_map_r2c,   // [raw_batch+1] int32
    int64_t raw_batch_size,
    c10::optional<torch::Tensor> raw_h0_opt,
    c10::optional<torch::Tensor> raw_h0_indices_opt,
    bool raw_h0_k_last,
    bool out_bf16
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.correct_h0",
            {{"ht_buffer", &ht_buffer}, {"mt_buffer", &mt_buffer},
             {"fallback_mask", &fallback_mask}, {"seq_map_r2c", &seq_map_r2c},
             {"raw_h0", raw_h0_opt.has_value() ? &raw_h0_opt.value() : nullptr},
             {"raw_h0_indices", raw_h0_indices_opt.has_value() ? &raw_h0_indices_opt.value() : nullptr}});
        return pla::bypass::output(ht_buffer.sizes(), ht_buffer.options().dtype(
            out_bf16 ? torch::kBFloat16 : torch::kFloat32));
    }
    const pla::sanity::Context sanity("flashqla.correct_h0", ht_buffer);
    if (sanity) {
        sanity.tensor(ht_buffer, "ht_buffer");
        sanity.tensor(mt_buffer, "mt_buffer");
        sanity.check(ht_buffer.dim() == 4 && ht_buffer.is_contiguous() &&
                         ht_buffer.size(2) == 128 && ht_buffer.size(3) == 128,
                     "ht_buffer must be contiguous [cp_batch,H,128,128]");
        sp::int_extent(sanity, ht_buffer.size(0), "cp_batch");
        sp::int_extent(sanity, ht_buffer.size(1), "H");
        sp::int_extent(sanity, raw_batch_size, "raw_batch_size");
        sanity.check(mt_buffer.sizes() == ht_buffer.sizes() && mt_buffer.is_contiguous(),
                     "mt_buffer must match contiguous ht_buffer shape");
        sp::binary_mask(sanity, fallback_mask, ht_buffer.size(0) * ht_buffer.size(1), "fallback_mask");
        const auto map = sp::offsets(sanity, seq_map_r2c, ht_buffer.size(0), "seq_map_r2c");
        sanity.check(seq_map_r2c.numel() == raw_batch_size + 1,
                     "seq_map_r2c must have raw_batch_size+1 entries");
        if (sanity.values_enabled()) {
            sanity.check(map.back() == ht_buffer.size(0),
                         "seq_map_r2c must cover all CP segments");
            for (size_t i = 1; i < map.size(); ++i)
                sanity.check(map[i] > map[i - 1],
                             "each raw sequence must own at least one CP segment");
        }
        if (raw_h0_opt.has_value() && raw_h0_opt->numel() > 0) {
            sanity.tensor(raw_h0_opt.value(), "raw_h0");
            sanity.check(raw_h0_opt->dim() == 4, "raw_h0 must have rank four");
            if (raw_h0_indices_opt.has_value())
                sp::slots(sanity, raw_h0_indices_opt.value(), raw_batch_size,
                          raw_h0_opt->size(0), "raw_h0_indices", false);
        }
    }
    TORCH_CHECK(ht_buffer.is_cuda(), "ht_buffer must be CUDA");
    TORCH_CHECK(ht_buffer.dtype() == torch::kFloat16 || ht_buffer.dtype() == torch::kBFloat16, "ht_buffer must be fp16 or bf16");
    check_same_cuda_device(ht_buffer, mt_buffer, "mt_buffer");
    check_same_cuda_device(ht_buffer, fallback_mask, "fallback_mask");
    check_same_cuda_device(ht_buffer, seq_map_r2c, "seq_map_r2c");
    check_same_b16_dtype(ht_buffer, mt_buffer, "mt_buffer");
    TORCH_CHECK(seq_map_r2c.dtype() == torch::kInt32, "seq_map_r2c must be int32");
    TORCH_CHECK(fallback_mask.dtype() == torch::kInt8, "fallback_mask must be int8");

    int cp_batch = ht_buffer.size(0);
    int H  = ht_buffer.size(1);
    int DK = ht_buffer.size(2);
    int DV = ht_buffer.size(3);
    TORCH_CHECK(DK == 128 && DV == 128, "correct_h0 expects DK=DV=128");
    at::cuda::CUDAGuard device_guard(ht_buffer.device());

    auto opt_f = torch::TensorOptions()
        .dtype(out_bf16 ? torch::kBFloat16 : torch::kFloat32)
        .device(ht_buffer.device());
    // The kernel writes the first state and every following CP-segment state.
    auto cp_h0 = torch::empty({cp_batch, H, DK, DV}, opt_f);

    const auto device = logical_device_info(ht_buffer);
    const hggcStream_t stream = pla::current_stream();
    const signed char* fb = reinterpret_cast<signed char*>(fallback_mask.data_ptr<int8_t>());

    const bool has_raw_h0 = raw_h0_opt.has_value() && raw_h0_opt->numel() > 0;
    TORCH_CHECK(!raw_h0_indices_opt.has_value() || has_raw_h0,
                "raw_h0_indices requires a non-empty raw_h0");
    TORCH_CHECK(!raw_h0_k_last || has_raw_h0,
                "raw_h0_k_last requires a non-empty raw_h0");
    if (has_raw_h0) {
        auto raw_h0 = raw_h0_opt.value();
        check_same_cuda_device(ht_buffer, raw_h0, "raw_h0");
        TORCH_CHECK(raw_h0.dtype() == torch::kFloat32 ||
                    raw_h0.dtype() == torch::kBFloat16,
                    "raw_h0 must be fp32 or bf16");
        TORCH_CHECK(raw_h0.is_contiguous(), "raw_h0 must be contiguous");
        if (raw_h0_k_last) {
            TORCH_CHECK(raw_h0.dim() == 4 && raw_h0.size(0) > 0 &&
                        raw_h0.size(1) == H && raw_h0.size(2) == DV &&
                        raw_h0.size(3) == DK,
                        "indexed raw_h0 must be [num_slots,H,DV,DK]");
        } else {
            TORCH_CHECK(raw_h0.dim() == 4 && raw_h0.size(0) > 0 &&
                        raw_h0.size(1) == H && raw_h0.size(2) == DK &&
                        raw_h0.size(3) == DV,
                        "raw_h0 must be [num_states,H,DK,DV]");
        }
        const int* raw_h0_indices_ptr = nullptr;
        if (raw_h0_indices_opt.has_value()) {
            auto raw_h0_indices = raw_h0_indices_opt.value();
            check_same_cuda_device(ht_buffer, raw_h0_indices, "raw_h0_indices");
            TORCH_CHECK(raw_h0_indices.dtype() == torch::kInt32,
                        "raw_h0_indices must be int32");
            TORCH_CHECK(raw_h0_indices.is_contiguous(),
                        "raw_h0_indices must be contiguous");
            TORCH_CHECK(raw_h0_indices.numel() == raw_batch_size,
                        "raw_h0_indices must have raw_batch_size entries");
            raw_h0_indices_ptr = raw_h0_indices.data_ptr<int>();
        } else {
            TORCH_CHECK(raw_h0.size(0) >= raw_batch_size,
                        "raw_h0 must have at least raw_batch_size states");
        }
        launch_correct_h0_raw_cute(
            cp_h0.data_ptr(), raw_h0.data_ptr(),
            data_ptr_b16(ht_buffer), data_ptr_b16(mt_buffer),
            fb, seq_map_r2c.data_ptr<int>(), raw_h0_indices_ptr,
            cp_batch, (int)raw_batch_size, H, raw_h0_k_last ? 1 : 0,
            raw_h0.dtype() == torch::kBFloat16, out_bf16,
            dtype_tag(ht_buffer), device.index, device.num_cu, stream);
    } else {
        launch_correct_h0_no_raw_cute(
            cp_h0.data_ptr(),
            data_ptr_b16(ht_buffer), data_ptr_b16(mt_buffer),
            fb, seq_map_r2c.data_ptr<int>(),
            cp_batch, (int)raw_batch_size, H, out_bf16, dtype_tag(ht_buffer),
            device.index, device.num_cu, stream);
    }
    return cp_h0;
}

// prepare_h: warmup-chunk state recurrence per (batch, head). Returns
// (ht [batch*H, DK, DV] fp16, mt [batch*H, DK, DK] fp16). PPU1.5 uses one
// combined CTA for the additive and homogeneous recurrences while sharing
// K/A/g inputs. PPU1.0/1.1 retains the validated two-pass implementation.
// Inactive transition tiles are not consumed by correct_h0.
std::vector<torch::Tensor> prepare_h(
    torch::Tensor k,                 // [1, T, Hg, DK] fp16
    torch::Tensor v,                 // [1, T, H, DV] fp16
    torch::Tensor a,                 // [1, T, H, CS] fp16
    torch::Tensor g,                 // [1, T, H] fp32
    torch::Tensor beta,              // [1, T, H] fp32
    torch::Tensor cu_seqlens,        // [batch+1] int32
    torch::Tensor num_warmup_chunks, // [batch*H] int32
    int64_t H,
    int64_t Hg
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.prepare_h",
            {{"k", &k}, {"v", &v}, {"a", &a}, {"g", &g}, {"beta", &beta},
             {"cu_seqlens", &cu_seqlens}, {"num_warmup_chunks", &num_warmup_chunks}});
        const auto batch_heads = (cu_seqlens.numel() - 1) * H;
        return {pla::bypass::output({batch_heads, 128, 128}, k.options()),
                pla::bypass::output({batch_heads, 128, 128}, k.options())};
    }
    const pla::sanity::Context sanity("flashqla.prepare_h", k);
    if (sanity) {
        sanity.check(k.dim() == 4, "k must have rank four");
        sp::int_extent(sanity, k.size(1), "T");
        sp::int_extent(sanity, H, "H");
        sanity.check(Hg > 0 && H % Hg == 0, "H must be divisible by positive Hg");
        sanity_feature(sanity, k, "k", k.size(1), Hg, 128);
        sanity_feature(sanity, v, "v", k.size(1), H, 128, true);
        sanity_feature(sanity, a, "a", k.size(1), H, 32);
        sanity_gate(sanity, g, "g", k.size(1), H);
        sanity_gate(sanity, beta, "beta", k.size(1), H);
        const auto cu = sp::offsets(sanity, cu_seqlens, k.size(1));
        sanity.tensor(num_warmup_chunks, "num_warmup_chunks");
        sanity.check(num_warmup_chunks.scalar_type() == torch::kInt32 &&
                         num_warmup_chunks.is_contiguous() &&
                         num_warmup_chunks.numel() == (cu_seqlens.numel() - 1) * H,
                     "num_warmup_chunks must contain contiguous int32 [batch*H] entries");
        sanity.check((cu_seqlens.numel() - 1) * H <= std::numeric_limits<int>::max(),
                     "batch*H exceeds kernel int32 ABI");
        if (sanity.values_enabled()) {
            const auto warmup = sanity.integers(num_warmup_chunks);
            for (size_t i = 0; i < warmup.size(); ++i) {
                const int64_t seq = i / H;
                sanity.check(warmup[i] >= 0 && warmup[i] <= (cu[seq + 1] - cu[seq] + 31) / 32,
                             "num_warmup_chunks out of range at entry ", i);
            }
        }
    }
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(k.dtype() == torch::kFloat16 || k.dtype() == torch::kBFloat16, "k must be fp16 or bf16");
    check_same_cuda_device(k, v, "v");
    check_same_cuda_device(k, a, "a");
    check_same_cuda_device(k, g, "g");
    check_same_cuda_device(k, beta, "beta");
    check_same_cuda_device(k, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(k, num_warmup_chunks, "num_warmup_chunks");
    check_same_b16_dtype(k, v, "v");
    check_same_b16_dtype(k, a, "a");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be fp32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(num_warmup_chunks.dtype() == torch::kInt32,
                "num_warmup_chunks must be int32");
    TORCH_CHECK(k.dim() == 4, "k must be [1,T,Hg,DK]");
    at::cuda::CUDAGuard device_guard(k.device());

    constexpr int DK = 128, DV = 128;
    int T = k.size(1);
    int batch_size = cu_seqlens.size(0) - 1;
    TORCH_CHECK(k.size(3) == DK, "prepare_h expects DK=128");
    TORCH_CHECK(v.dim() == 4 && v.size(0) == 1 && v.size(1) == T &&
                    v.size(2) == H && v.size(3) == DV &&
                    v.stride(3) == 1 && v.stride(2) == DV &&
                    v.stride(1) >= H * DV,
                "v must be [1,T,H,128], head-contiguous, with a positive "
                "token stride");

    auto opt_e = torch::TensorOptions().dtype(k.dtype()).device(k.device());
    auto ht = torch::empty({batch_size * (int)H, DK, DV}, opt_e);
    auto mt = torch::empty({batch_size * (int)H, DK, DK}, opt_e);

    const auto device = logical_device_info(k);
    const hggcStream_t stream = pla::current_stream();
    launch_prepare_h_cp_cute(
        data_ptr_b16(k), data_ptr_b16(v), data_ptr_b16(a),
        g.data_ptr<float>(), beta.data_ptr<float>(), data_ptr_b16(ht), data_ptr_b16(mt),
        cu_seqlens.data_ptr<int>(), num_warmup_chunks.data_ptr<int>(),
        batch_size, T, (int)H, (int)Hg, dtype_tag(k),
        v.stride(1),
        device.index, device.num_cu, device.compute_capability, stream);

    return {ht, mt};
}

// fused_fwd: main forward pass. Returns (o [1,T,H,DV] fp16, ht_out
// [raw_batch,H,DK,DV] fp32 or bf16 per final_state_bf16). When raw_state is
// supplied, final state is written there in place, in raw_state's own dtype,
// and the returned ht_out is empty. Otherwise ht_out may be omitted when the
// caller does not consume the final state. h0 optional (empty -> zero).
//
// The state dtype is a single compile-time kernel parameter, so h0, raw_state
// and ht_out all share it: every state tensor handed in must agree with
// final_state_bf16.
std::vector<torch::Tensor> fused_fwd(
    torch::Tensor q,               // [1, T, Hg, DK] fp16
    torch::Tensor k,               // [1, T, Hg, DK] fp16
    torch::Tensor v,               // [1, T, H, DV] fp16
    torch::Tensor a,               // [1, T, H, CS] fp16
    torch::Tensor beta,            // [1, T, H] fp32
    torch::Tensor g,               // [1, T, H] fp32
    torch::Tensor cu_seqlens,      // [cp_batch+1] int32
    c10::optional<torch::Tensor> cp_seq_map_opt,  // [cp_batch] int32; absent for identity
    torch::Tensor raw_cu_seqlens,  // [raw_batch+1] int32
    int64_t valid_num_tokens,
    int64_t H,
    int64_t Hg,
    double scale,
    c10::optional<torch::Tensor> h0_opt,
    c10::optional<torch::Tensor> raw_state_opt,
    c10::optional<torch::Tensor> raw_state_indices_opt,
    bool output_final_state,
    bool final_state_bf16,
    c10::optional<torch::Tensor> raw_state_input_opt,
    c10::optional<torch::Tensor> num_warmup_opt,
    c10::optional<torch::Tensor> fallback_mask_opt
) {
    if (pla::bypass::enabled()) {
        const c10::cuda::OptionalCUDAGuard guard(q.is_cuda()
            ? c10::optional<c10::Device>(q.device()) : c10::nullopt);
        const auto raw_batch = raw_cu_seqlens.numel() - 1;
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.fused_fwd",
            {{"q", &q}, {"k", &k}, {"v", &v}, {"a", &a}, {"beta", &beta}, {"g", &g},
             {"cu_seqlens", &cu_seqlens}, {"raw_cu_seqlens", &raw_cu_seqlens},
             {"cp_seq_map", cp_seq_map_opt.has_value() ? &cp_seq_map_opt.value() : nullptr},
             {"h0", h0_opt.has_value() ? &h0_opt.value() : nullptr},
             {"raw_state", raw_state_opt.has_value() ? &raw_state_opt.value() : nullptr},
             {"raw_state_indices", raw_state_indices_opt.has_value() ? &raw_state_indices_opt.value() : nullptr},
             {"raw_state_input", raw_state_input_opt.has_value() ? &raw_state_input_opt.value() : nullptr},
             {"num_warmup", num_warmup_opt.has_value() ? &num_warmup_opt.value() : nullptr},
             {"fallback_mask", fallback_mask_opt.has_value() ? &fallback_mask_opt.value() : nullptr}},
            "valid_num_tokens=" + std::to_string(valid_num_tokens) +
            " scale=" + std::to_string(scale) +
            " output_final_state=" + std::to_string(output_final_state));
        if (pla::instrument::show_params()) pla::instrument::dump_case(raw_batch, q.size(1), q.size(3), v.size(3), Hg, H,
            get_dtype_string(q), false, &raw_cu_seqlens, -5.0f, true);
        auto state_options = q.options().dtype(
            final_state_bf16 ? torch::kBFloat16 : torch::kFloat32);
        auto ht = (!raw_state_opt.has_value() && output_final_state)
            ? pla::bypass::output({raw_batch, H, q.size(3), v.size(3)}, state_options)
            : pla::bypass::output({0}, state_options);
        return {pla::bypass::output({1, q.size(1), H, v.size(3)}, q.options()), ht};
    }
    const pla::sanity::Context sanity("flashqla.fused_fwd", q);
    if (sanity) {
        sanity.check(q.dim() == 4, "q must have rank four");
        sp::int_extent(sanity, q.size(1), "T");
        sp::int_extent(sanity, H, "H");
        sanity.check(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)),
                     "scale must be finite and representable as fp32");
        for (const auto& named : std::initializer_list<std::pair<const char*, torch::Tensor>>{
                 {"q", q}, {"k", k}, {"v", v}, {"a", a}, {"beta", beta}, {"g", g}})
            sanity.tensor(named.second, named.first);
        const auto cu = sp::offsets(sanity, cu_seqlens, q.size(1));
        const auto raw = sp::offsets(sanity, raw_cu_seqlens, q.size(1), "raw_cu_seqlens");
        std::vector<int64_t> map;
        if (cp_seq_map_opt.has_value()) {
            sanity.tensor(cp_seq_map_opt.value(), "cp_seq_map");
            sanity.check(cp_seq_map_opt->scalar_type() == torch::kInt32 &&
                             cp_seq_map_opt->dim() == 1 && cp_seq_map_opt->is_contiguous() &&
                             cp_seq_map_opt->numel() == cu_seqlens.numel() - 1,
                         "cp_seq_map must be contiguous int32 [cp_batch]");
            if (sanity.values_enabled()) map = sanity.integers(cp_seq_map_opt.value());
        }
        if (sanity.values_enabled()) {
            sanity.check(cu.back() == valid_num_tokens && raw.back() == valid_num_tokens,
                         "CP/raw cu endpoints must equal valid_num_tokens");
            if (!cp_seq_map_opt.has_value()) {
                sanity.check(cu == raw, "omitted cp_seq_map requires identical CP/raw offsets");
            } else {
                for (size_t i = 0; i < map.size(); ++i) {
                    const int64_t r = map[i];
                    sanity.check(r >= 0 && r < int64_t(raw.size()) - 1,
                                 "cp_seq_map raw sequence out of range at entry ", i);
                    sanity.check(cu[i] >= raw[r] && cu[i + 1] <= raw[r + 1],
                                 "CP segment must lie within its mapped raw sequence");
                    sanity.check((cu[i] - raw[r]) % 32 == 0 &&
                                     (cu[i + 1] == raw[r + 1] || (cu[i + 1] - raw[r]) % 32 == 0),
                                 "CP segment boundaries must preserve raw 32-token chunks");
                }
            }
        }
        if (h0_opt.has_value() && h0_opt->numel() > 0) sanity.tensor(h0_opt.value(), "h0");
        if (raw_state_opt.has_value()) {
            sanity.tensor(raw_state_opt.value(), "raw_state");
            sanity.check(raw_state_opt->dim() == 4, "raw_state must have rank four");
            sanity.check(raw_state_indices_opt.has_value(), "raw_state_indices required with raw_state");
            sp::slots(sanity, raw_state_indices_opt.value(), raw_cu_seqlens.numel() - 1,
                      raw_state_opt->size(0), "raw_state_indices", true, raw);
            // These buffers are consumed while other CP CTAs write the pool.
            for (const auto& input : {q, k, v, a, beta, g})
                sanity.check(!sp::overlaps(raw_state_opt.value(), input),
                             "raw_state must not overlap token inputs");
            if (h0_opt.has_value() && h0_opt->numel() > 0)
                sanity.check(!sp::overlaps(raw_state_opt.value(), h0_opt.value()) ||
                                 (!cp_seq_map_opt.has_value() &&
                                  sp::same_view(raw_state_opt.value(), h0_opt.value())),
                             "raw_state/h0 overlap is unsafe for CP or partial state views");
            if (raw_state_input_opt.has_value()) {
                sanity.tensor(raw_state_input_opt.value(), "raw_state_input");
                sanity.check(!sp::overlaps(raw_state_opt.value(),
                                           raw_state_input_opt.value()),
                             "raw_state_input must be a disjoint overlap snapshot");
            }
        }
    }
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16, "q must be fp16 or bf16");
    TORCH_CHECK(is_supported_head_config(H, Hg),
                "FlashQLA does not support head configuration (H=", H,
                ", Hg=", Hg, ")");
    TORCH_CHECK(q.dim() == 4, "q must be [1,T,Hg,128]");
    TORCH_CHECK(k.dim() == 4, "k must be [1,T,Hg,128]");
    TORCH_CHECK(v.dim() == 4, "v must be [1,T,H,128]");
    TORCH_CHECK(a.dim() == 4, "a must be [1,T,H,32]");
    TORCH_CHECK(beta.dim() == 3, "beta must be [1,T,H]");
    TORCH_CHECK(g.dim() == 3, "g must be [1,T,H]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.stride(3) == 1 && v.stride(2) == 128 &&
                    v.stride(1) >= H * 128,
                "v must be head-contiguous with a positive token stride");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");
    check_same_cuda_device(q, k, "k");
    check_same_cuda_device(q, v, "v");
    check_same_cuda_device(q, a, "a");
    check_same_cuda_device(q, beta, "beta");
    check_same_cuda_device(q, g, "g");
    check_same_cuda_device(q, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(q, raw_cu_seqlens, "raw_cu_seqlens");
    check_same_b16_dtype(q, k, "k");
    check_same_b16_dtype(q, v, "v");
    check_same_b16_dtype(q, a, "a");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be fp32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(raw_cu_seqlens.dtype() == torch::kInt32, "raw_cu_seqlens int32");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.is_contiguous() &&
                cu_seqlens.numel() >= 2,
                "cu_seqlens must be contiguous int32 [cp_batch+1]");
    TORCH_CHECK(raw_cu_seqlens.dim() == 1 && raw_cu_seqlens.is_contiguous() &&
                raw_cu_seqlens.numel() >= 2,
                "raw_cu_seqlens must be contiguous int32 [raw_batch+1]");
    at::cuda::CUDAGuard device_guard(q.device());

    constexpr int DK = 128, DV = 128, CS = 32;
    int T = q.size(1);
    TORCH_CHECK(T > 0, "FlashQLA does not support an empty token dimension");
    TORCH_CHECK(valid_num_tokens > 0 && valid_num_tokens <= T,
                "valid_num_tokens must be in (0, q.size(1)], got ",
                valid_num_tokens, " for ", T, " physical tokens");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == Hg && q.size(3) == DK,
                "q must be [1,T,Hg,128]");
    TORCH_CHECK(k.size(0) == 1 && k.size(1) == T &&
                k.size(2) == Hg && k.size(3) == DK,
                "k must be [1,T,Hg,128] and match q's token dimension");
    TORCH_CHECK(v.size(0) == 1 && v.size(1) == T &&
                v.size(2) == H && v.size(3) == DV,
                "v must be [1,T,H,128] and match q's token dimension");
    TORCH_CHECK(a.size(0) == 1 && a.size(1) == T &&
                a.size(2) == H && a.size(3) == CS,
                "a must be [1,T,H,32] and match q's token dimension");
    TORCH_CHECK(beta.size(0) == 1 && beta.size(1) == T && beta.size(2) == H,
                "beta must be [1,T,H]");
    TORCH_CHECK(g.size(0) == 1 && g.size(1) == T && g.size(2) == H,
                "g must be [1,T,H]");
    int cp_batch = cu_seqlens.size(0) - 1;
    int raw_batch = raw_cu_seqlens.size(0) - 1;
    const int* cp_seq_map_ptr = nullptr;
    torch::Tensor cp_seq_map;
    if (cp_seq_map_opt.has_value()) {
        cp_seq_map = cp_seq_map_opt.value();
        check_same_cuda_device(q, cp_seq_map, "cp_seq_map");
        TORCH_CHECK(cp_seq_map.dtype() == torch::kInt32,
                    "cp_seq_map must be int32");
        TORCH_CHECK(cp_seq_map.is_contiguous(),
                    "cp_seq_map must be contiguous");
        TORCH_CHECK(cp_seq_map.dim() == 1 && cp_seq_map.numel() == cp_batch,
                    "cp_seq_map must be [cp_batch]");
        cp_seq_map_ptr = cp_seq_map.data_ptr<int>();
    } else {
        TORCH_CHECK(cp_batch == raw_batch,
                    "cp_seq_map may be omitted only for an identity plan");
    }
    TORCH_CHECK(raw_state_opt.has_value() == raw_state_indices_opt.has_value(),
                "raw_state and raw_state_indices must be provided together");
    TORCH_CHECK(num_warmup_opt.has_value() == fallback_mask_opt.has_value(),
                "num_warmup and fallback_mask must be provided together");
    TORCH_CHECK(!num_warmup_opt.has_value() || cp_seq_map_opt.has_value(),
                "overlap warmup metadata requires a CP sequence map");
    TORCH_CHECK(!raw_state_input_opt.has_value() ||
                    (raw_state_opt.has_value() && num_warmup_opt.has_value()),
                "raw_state_input is only valid for indexed overlap warmup");
    TORCH_CHECK(!num_warmup_opt.has_value() || !raw_state_opt.has_value() ||
                    raw_state_input_opt.has_value(),
                "indexed overlap warmup requires a read-only raw_state_input");
    TORCH_CHECK(!num_warmup_opt.has_value() || !h0_opt.has_value() ||
                    h0_opt->numel() == 0,
                "overlap warmup cannot consume precomputed CP h0");

    const int* num_warmup_ptr = nullptr;
    const signed char* fallback_mask_ptr = nullptr;
    torch::Tensor num_warmup;
    torch::Tensor fallback_mask;
    if (num_warmup_opt.has_value()) {
        num_warmup = num_warmup_opt.value();
        fallback_mask = fallback_mask_opt.value();
        check_same_cuda_device(q, num_warmup, "num_warmup");
        check_same_cuda_device(q, fallback_mask, "fallback_mask");
        TORCH_CHECK(num_warmup.dtype() == torch::kInt32 &&
                        num_warmup.is_contiguous() &&
                        num_warmup.numel() == cp_batch * H,
                    "num_warmup must be contiguous int32 [cp_batch*H]");
        TORCH_CHECK(fallback_mask.dtype() == torch::kInt8 &&
                        fallback_mask.is_contiguous() &&
                        fallback_mask.numel() == cp_batch * H,
                    "fallback_mask must be contiguous int8 [cp_batch*H]");
        num_warmup_ptr = num_warmup.data_ptr<int>();
        fallback_mask_ptr = reinterpret_cast<const signed char*>(
            fallback_mask.data_ptr<int8_t>());
    }

    if (g_ppu_lib_show_params) {
        std::cout << generate_nvtx_tag(
            raw_batch, T, DK, DV, (int)Hg, (int)H, get_dtype_string(q),
            false, &raw_cu_seqlens, -5.0f, true) << std::endl;
    }

    auto opt_h = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    // The compute kernels only visit the prefix covered by cu_seqlens. Allocate
    // the physical shape required by the framework; only its padded suffix
    // needs initialization.
    auto o = torch::empty({1, T, (int)H, DV}, opt_h);
    // The final state leaves the kernel in the pool's own dtype: bf16 halves the
    // resident state pool, fp32 keeps the historical contract.
    auto opt_state = torch::TensorOptions()
        .dtype(final_state_bf16 ? torch::kBFloat16 : torch::kFloat32)
        .device(q.device());
    // Indexed mode always updates raw_state in place. In the non-indexed path,
    // do not allocate or globally store a state that the public API will drop.
    auto ht_out = (!raw_state_opt.has_value() && output_final_state)
        ? torch::empty({raw_batch, (int)H, DK, DV}, opt_state)
        : torch::empty({0}, opt_state);

    const void* h0_ptr = nullptr;
    torch::Tensor h0;
    if (h0_opt.has_value() && h0_opt->numel() > 0) {
        h0 = h0_opt.value();
        check_same_cuda_device(q, h0, "h0");
        TORCH_CHECK(h0.dtype() == (final_state_bf16 ? torch::kBFloat16
                                                    : torch::kFloat32),
                    "h0 dtype must match the state pool selected by "
                    "final_state_bf16");
        TORCH_CHECK(h0.is_contiguous(), "h0 must be contiguous");
        TORCH_CHECK(h0.dim() == 4 && h0.size(0) >= cp_batch &&
                    h0.size(1) == H && h0.size(2) == DK && h0.size(3) == DV,
                    "h0 must be [num_states>=cp_batch,H,DK,DV]");
        h0_ptr = h0.data_ptr();
    }

    void* raw_state_ptr = nullptr;
    const void* raw_state_input_ptr = nullptr;
    const int* raw_state_indices_ptr = nullptr;
    torch::Tensor raw_state;
    torch::Tensor raw_state_input;
    torch::Tensor raw_state_indices;
    if (raw_state_opt.has_value()) {
        raw_state = raw_state_opt.value();
        check_same_cuda_device(q, raw_state, "raw_state");
        TORCH_CHECK(raw_state.dtype() == (final_state_bf16 ? torch::kBFloat16
                                                            : torch::kFloat32),
                    "raw_state dtype must match the state pool selected by "
                    "final_state_bf16");
        TORCH_CHECK(raw_state.is_contiguous(), "raw_state must be contiguous");
        TORCH_CHECK(raw_state.dim() == 4 && raw_state.size(0) > 0 &&
                    raw_state.size(1) == H &&
                    raw_state.size(2) == DV && raw_state.size(3) == DK,
                    "raw_state must be [num_slots,H,DV,DK]");
        raw_state_indices = raw_state_indices_opt.value();
        check_same_cuda_device(q, raw_state_indices, "raw_state_indices");
        TORCH_CHECK(raw_state_indices.dtype() == torch::kInt32,
                    "raw_state_indices must be int32");
        TORCH_CHECK(raw_state_indices.is_contiguous(),
                    "raw_state_indices must be contiguous");
        TORCH_CHECK(raw_state_indices.dim() == 1 &&
                    raw_state_indices.numel() == raw_batch,
                    "raw_state_indices must be [raw_batch]");
        raw_state_ptr = raw_state.data_ptr();
        raw_state_indices_ptr = raw_state_indices.data_ptr<int>();
    }
    if (raw_state_input_opt.has_value()) {
        raw_state_input = raw_state_input_opt.value();
        check_same_cuda_device(q, raw_state_input, "raw_state_input");
        TORCH_CHECK(raw_state_input.dtype() ==
                        (final_state_bf16 ? torch::kBFloat16 : torch::kFloat32),
                    "raw_state_input dtype must match raw_state");
        TORCH_CHECK(raw_state_input.is_contiguous(),
                    "raw_state_input must be contiguous");
        TORCH_CHECK(raw_state_input.dim() == 4 &&
                        raw_state_input.size(0) == raw_batch &&
                        raw_state_input.size(1) == H &&
                        raw_state_input.size(2) == DV &&
                        raw_state_input.size(3) == DK,
                    "raw_state_input must be [raw_batch,H,DV,DK]");
        raw_state_input_ptr = raw_state_input.data_ptr();
    }

    const auto device = logical_device_info(q);
    const hggcStream_t stream = pla::current_stream();
    if (valid_num_tokens < T) {
        const int64_t tail_offset = valid_num_tokens * H * DV * q.element_size();
        const int64_t tail_bytes =
            (int64_t(T) - valid_num_tokens) * H * DV * q.element_size();
        auto* tail = reinterpret_cast<unsigned char*>(data_ptr_b16(o)) + tail_offset;
        const hggcError_t status = hggcMemsetAsync(tail, 0, tail_bytes, stream);
        TORCH_CHECK(status == hggcSuccess,
                    "failed to zero FlashQLA padded output tail: ",
                    hggcGetErrorString(status));
    }
    launch_fused_fwd_cp_cute(
        data_ptr_b16(o), data_ptr_b16(q), data_ptr_b16(k), data_ptr_b16(v), data_ptr_b16(a),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        h0_ptr, ht_out.numel() > 0 ? ht_out.data_ptr() : nullptr,
        raw_state_ptr, raw_state_input_ptr, final_state_bf16,
        raw_state_indices_ptr, cu_seqlens.data_ptr<int>(), cp_seq_map_ptr,
        raw_cu_seqlens.data_ptr<int>(), num_warmup_ptr, fallback_mask_ptr,
        cp_batch, T, raw_batch, (int)H, (int)Hg, (float)scale, dtype_tag(q),
        v.stride(1),
        device.index, device.num_cu, device.compute_capability, stream);

    return {o, ht_out};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    (void)pla::bypass::enabled();
    (void)pla::instrument::show_params();


    init_show_params();

    m.def("chunk_local_cumsum", &chunk_local_cumsum,
          "FlashQLA CuTe chunk-local cumsum of gate",
          py::arg("g"), py::arg("cu_seqlens"), py::arg("chunk_indices"));
    m.def("chunk_local_cumsum_qk_norm", &chunk_local_cumsum_qk_norm,
          "FlashQLA fused raw q/k L2 normalization and gate cumsum",
          py::arg("g"), py::arg("cu_seqlens"), py::arg("chunk_indices"),
          py::arg("q"), py::arg("k"));
    m.def("chunk_local_cumsum_qk_norm_gated",
          &chunk_local_cumsum_qk_norm_gated,
          "FlashQLA fused raw GDN gating, cumsum, and q/k L2 normalization",
          py::arg("a"), py::arg("b"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("cu_seqlens"), py::arg("chunk_indices"),
          py::arg("q"), py::arg("k"), py::arg("valid_num_tokens") = -1);
    m.def("get_warmup_chunks", &get_warmup_chunks,
          "FlashQLA CuTe get_warmup_chunks (Task 2)",
          py::arg("g"), py::arg("ht_mask"), py::arg("cu_seqlens"),
          py::arg("threshold") = -10.0);
    m.def("kkt_solve", &kkt_solve,
          "FlashQLA CuTe kkt_solve (Task 3)",
          py::arg("K"), py::arg("beta"), py::arg("cu_seqlens"), py::arg("H"),
          py::arg("chunk_indices") = c10::optional<torch::Tensor>());
    m.def("correct_h0", &correct_h0,
          "FlashQLA CuTe correct_h0; raw_h0_indices must index raw_h0 slots",
          py::arg("ht_buffer"), py::arg("mt_buffer"), py::arg("fallback_mask"),
          py::arg("seq_map_r2c"), py::arg("raw_batch_size"),
          py::arg("raw_h0") = c10::optional<torch::Tensor>(),
          py::arg("raw_h0_indices") = c10::optional<torch::Tensor>(),
          py::arg("raw_h0_k_last") = false,
          py::arg("out_bf16") = false);
    m.def("prepare_h", &prepare_h,
          "FlashQLA CuTe prepare_h (Task 5)",
          py::arg("k"), py::arg("v"), py::arg("a"), py::arg("g"), py::arg("beta"),
          py::arg("cu_seqlens"), py::arg("num_warmup_chunks"),
          py::arg("H"), py::arg("Hg"));
    m.def("fused_fwd", &fused_fwd,
          "FlashQLA CuTe fused_fwd; raw_state and raw_state_indices are paired",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("a"),
          py::arg("beta"), py::arg("g"), py::arg("cu_seqlens"),
          py::arg("cp_seq_map"), py::arg("raw_cu_seqlens"),
          py::arg("valid_num_tokens"), py::arg("H"), py::arg("Hg"),
          py::arg("scale"),
          py::arg("h0") = c10::optional<torch::Tensor>(),
          py::arg("raw_state") = c10::optional<torch::Tensor>(),
          py::arg("raw_state_indices") = c10::optional<torch::Tensor>(),
          py::arg("output_final_state") = true,
          py::arg("final_state_bf16") = false,
          py::arg("raw_state_input") = c10::optional<torch::Tensor>(),
          py::arg("num_warmup") = c10::optional<torch::Tensor>(),
          py::arg("fallback_mask") = c10::optional<torch::Tensor>());
}
