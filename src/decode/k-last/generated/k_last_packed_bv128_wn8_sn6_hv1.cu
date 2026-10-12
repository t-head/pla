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

#include "../fused_sigmoid_gating_k_last_packed_kernel.cuh"

// Wrappers: BV=128, WARP_NUM=8, STAGE_NUM=6, HV_PER_CTA=1

void launch_k_last_packed_bv128_wn8_sn6_hv1_f32_fp16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<float, __half, 8, 128, 1, 6, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<float, __half, 8, 128, 1, 6, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_f32_bf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<float, __ppu_bfloat16, 8, 128, 1, 6, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<float, __ppu_bfloat16, 8, 128, 1, 6, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_fp16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__half, __half, 8, 128, 1, 6, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__half, __half, 8, 128, 1, 6, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_bf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__half, __ppu_bfloat16, 8, 128, 1, 6, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__half, __ppu_bfloat16, 8, 128, 1, 6, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_fp16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __half, 8, 128, 1, 6, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __half, 8, 128, 1, 6, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_bf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __ppu_bfloat16, 8, 128, 1, 6, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __ppu_bfloat16, 8, 128, 1, 6, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_f32_fp16_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<float, __half, 8, 128, 1, 6, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<float, __half, 8, 128, 1, 6, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_f32_bf16_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<float, __ppu_bfloat16, 8, 128, 1, 6, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<float, __ppu_bfloat16, 8, 128, 1, 6, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_fp16_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__half, __half, 8, 128, 1, 6, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__half, __half, 8, 128, 1, 6, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_bf16_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__half, __ppu_bfloat16, 8, 128, 1, 6, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__half, __ppu_bfloat16, 8, 128, 1, 6, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_fp16_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __half, 8, 128, 1, 6, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __half, 8, 128, 1, 6, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __half*>(mixed_qkv),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

void launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_bf16_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta, float softplus_threshold,
    const void* mixed_qkv, const void* b,
    void* o, void* h0_source,
    const int32_t* cu_seqlens, const int32_t* h0_indices,
    float scale, int T_max,
    int stride_a, int stride_qkv, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool use_qk_l2norm, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 8);
    if (is_kda) {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __ppu_bfloat16, 8, 128, 1, 6, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    } else {
    fused_sigmoid_gating_delta_kernel_packed<__ppu_bfloat16, __ppu_bfloat16, 8, 128, 1, 6, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(mixed_qkv),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens, h0_indices,
        scale, T_max,
        stride_a, stride_qkv, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        use_qk_l2norm, is_sglang
    );
    }
}

