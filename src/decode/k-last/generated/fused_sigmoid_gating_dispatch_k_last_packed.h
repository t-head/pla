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

#pragma once

#include <hggc_runtime.h>
#include <cstdint>

void launch_k_last_packed_bv16_wn4_sn4_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_f32_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_f32_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_fp16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_bf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_f32_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_f32_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_fp16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);
void launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_bf16_statebf16(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);

using k_last_packed_launcher_t = void(*)(
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
    int shared_mem_size, dim3 grid, hggcStream_t stream);

inline k_last_packed_launcher_t get_k_last_packed_launcher_dispatch(
    int bv, int wn, int sn, int hv_per_cta,
    at::ScalarType scalar_type, at::ScalarType alog_type,
    at::ScalarType state_dtype = at::ScalarType::Float) {
    if (bv == 16 && wn == 4 && sn == 4 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv16_wn4_sn4_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 32 && wn == 4 && sn == 4 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn4_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 32 && wn == 4 && sn == 5 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn5_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 32 && wn == 4 && sn == 6 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn6_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 32 && wn == 4 && sn == 8 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn4_sn8_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 32 && wn == 8 && sn == 4 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv32_wn8_sn4_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 4 && sn == 4 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn4_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 4 && sn == 5 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn5_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 4 && sn == 6 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn6_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 4 && sn == 8 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn4_sn8_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 8 && sn == 4 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn4_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 8 && sn == 5 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn5_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 8 && sn == 6 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn6_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 64 && wn == 8 && sn == 8 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv64_wn8_sn8_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 4 && sn == 4 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn4_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 4 && sn == 5 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn5_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 4 && sn == 6 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn6_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 4 && sn == 8 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn4_sn8_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 8 && sn == 4 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn4_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 8 && sn == 5 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn5_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 8 && sn == 6 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn6_hv1_bf16_bf16_statebf16;
        }
    } else if (bv == 128 && wn == 8 && sn == 8 && hv_per_cta == 1) {
        if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_f32_fp16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_f32_bf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_fp16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_bf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_fp16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::Float) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_bf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_f32_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Float && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_f32_bf16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::Half && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_fp16_bf16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::Half && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_fp16_statebf16;
        } else if (alog_type == at::ScalarType::BFloat16 && scalar_type == at::ScalarType::BFloat16 && state_dtype == at::ScalarType::BFloat16) {
            return launch_k_last_packed_bv128_wn8_sn8_hv1_bf16_bf16_statebf16;
        }
    }
    return nullptr;
}
