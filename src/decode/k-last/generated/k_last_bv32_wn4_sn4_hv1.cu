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

#include "../fused_sigmoid_gating_k_last_kernel.cuh"

// Wrappers: BV=32, WARP_NUM=4, STAGE_NUM=4, HV_PER_CTA=1

void launch_k_last_bv32_wn4_sn4_hv1_f32_fp16_dec(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, true, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, false, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_f32_fp16_mtp(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, true, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, false, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_f32_bf16_dec(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, true, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, false, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_f32_bf16_mtp(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, true, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, false, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_fp16_dec(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, true, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, false, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_fp16_mtp(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, true, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, false, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_bf16_dec(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, true, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, false, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_bf16_mtp(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, true, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, false, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_fp16_dec(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, true, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, false, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_fp16_mtp(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, true, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, false, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_bf16_dec(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, true, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, false, false, false, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_bf16_mtp(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, true, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, false, false, true, true, float>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<float*>(h0_source), cu_seqlens,
        reinterpret_cast<float*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}


#include "../fused_sigmoid_gating_k_last_kernel.cuh"

// Wrappers: BV=32, WARP_NUM=4, STAGE_NUM=4, HV_PER_CTA=1

void launch_k_last_bv32_wn4_sn4_hv1_f32_fp16_dec_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, true, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, false, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_f32_fp16_mtp_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, true, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, true, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __half, 4, 32, 1, 4, false, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_f32_bf16_dec_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, true, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, false, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_f32_bf16_mtp_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, true, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, true, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<float, __ppu_bfloat16, 4, 32, 1, 4, false, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const float*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_fp16_dec_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, true, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, false, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_fp16_mtp_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, true, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, true, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __half, 4, 32, 1, 4, false, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_bf16_dec_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, true, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, false, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_fp16_bf16_mtp_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, true, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, true, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__half, __ppu_bfloat16, 4, 32, 1, 4, false, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __half*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_fp16_dec_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, true, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, false, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_fp16_mtp_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, true, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, true, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __half, 4, 32, 1, 4, false, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __half*>(a),
        reinterpret_cast<const __half*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(k),
        reinterpret_cast<const __half*>(v),
        reinterpret_cast<const __half*>(b),
        reinterpret_cast<__half*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_bf16_dec_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, true, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, false, false, false, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

void launch_k_last_bv32_wn4_sn4_hv1_bf16_bf16_mtp_statebf16(
        const void* A_log, const void* a, const void* dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    const void* q, const void* k, const void* v, const void* b,
    void* o, void* h0_source, const int32_t* cu_seqlens,
    void* intermediate_states_buffer,
    const int32_t* h0_indices, const int32_t* intermediate_state_indices,
    int cache_steps, const int64_t* retrieve_parent_token_ptr,
    int stride_retrieve_parent_token_seq, int stride_retrieve_parent_token_token,
    float scale, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda,
    int shared_mem_size, dim3 grid, hggcStream_t stream
) {
    constexpr int WARP_SIZE = 32;
    auto block = dim3(WARP_SIZE, 4);
    if (is_kda && use_lower_bound) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, true, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else if (is_kda) {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, true, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const float*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    } else {
        fused_sigmoid_gating_delta_kernel<__ppu_bfloat16, __ppu_bfloat16, 4, 32, 1, 4, false, false, true, true, __ppu_bfloat16>
        <<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<const __ppu_bfloat16*>(A_log),
        reinterpret_cast<const __ppu_bfloat16*>(a),
        reinterpret_cast<const __ppu_bfloat16*>(dt_bias),
        softplus_beta_or_lower_bound, softplus_threshold,
        reinterpret_cast<const __ppu_bfloat16*>(q),
        reinterpret_cast<const __ppu_bfloat16*>(k),
        reinterpret_cast<const __ppu_bfloat16*>(v),
        reinterpret_cast<const __ppu_bfloat16*>(b),
        reinterpret_cast<__ppu_bfloat16*>(o),
        reinterpret_cast<__ppu_bfloat16*>(h0_source), cu_seqlens,
        reinterpret_cast<__ppu_bfloat16*>(intermediate_states_buffer),
        h0_indices, intermediate_state_indices,
        cache_steps, retrieve_parent_token_ptr,
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token,
        scale, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang
        );
    }
}

