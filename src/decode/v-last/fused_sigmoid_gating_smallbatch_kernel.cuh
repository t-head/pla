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
// Fused sigmoid gating delta rule -- v-last (V-major) smallbatch kernel.
//
// Split out of fused_sigmoid_gating_cuda_smallbatch.cu so the template
// instantiations can be spread over one translation unit per H value
// (fused_sigmoid_gating_smallbatch_h*.cu).
//
// The device translation unit uses the torch-free dtype dispatch from
// v_last_args.h and must never include host-only common.cuh.
#include <stdexcept>

#include <hggc_fp16.h>
#include <hggc_bf16.h>

#include "v_last_args.h"
#include <hggc_runtime.h>
#include <hggc_pipeline.h>

// #define GINO_DEBUG

#define ASYNC_COPY_H0

__device__ __forceinline__ static float divide_float(float a, float b) {
  #ifdef __HGGC_ARCH__
    return __fdividef(a, b);
  #else
    return (a / b);
  #endif
}
__device__ __forceinline__ static float divide_by_power_of_two(int divided, int divider) {
#ifdef __HGGC_ARCH__
    return (divided >> __builtin_ctz(divider));
#else
    return divided / divider;
#endif
}

__device__ __forceinline__ uint32_t cvta_sm_addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <int SizeInBytes>
__device__ __forceinline__ void cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
// #if defined (__HGGCCC__)
//     /*
//         kp = 1 (CA)	cache in all level;
//         kp = 2 (CG)	read through L1/L2, cache in LLC;
//         kp = 3 (CV)	read through L1/L2/LLC;
//         kp = 6 (BL)	bypass L1/L2;
//         kp = 7 (BA)	bypass L1/L2/LLC.
//     */
//     if (pred_guard) {
//       __ppu_pipeline_memcpy_async(smem_ptr, global_ptr, SizeInBytes, 0, 3);
//     }

// #else
  // Make sure the size is supported.
  static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
    "Size is not supported");

  uint32_t smem_int_ptr = cvta_sm_addr(smem_ptr);

#if defined (__HGGCCC__)
    // if (pred_guard) {
    //     asm volatile(
    //         "{\n"
    //         "  cp.async.kp3.shared.global [%0], [%1], %2;\n"
    //         "}\n" ::"r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
    // }
  asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  ppu.cmpp.ne.b32 p, %0, 0;\n"
    "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
    // asm volatile(
    // "{\n"
    // "  .reg .pred p;\n"
    // "  ppu.cmpp.ne.b32 p, %0, 0;\n"
    // "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n"
    // "}\n" ::"r"((int)pred_guard),
    // "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
#else
  asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  ppu.cmpp.ne.b32 p, %0, 0;\n"
    "  @p ppu.cp.async.ca.shared.global [%1], [%2], %3;\n"
    "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
#endif
// #endif
}

__device__ __forceinline__ void cp_async_commit()
{
// #ifdef __HGGCCC__
//     __pipeline_commit();
// #else
    asm volatile("ppu.cp.async.commit_group;\n" ::);
// #endif
}


__device__ __forceinline__ void cp_async_wait(int PIPE) {
    switch (PIPE) {
        case 0: asm volatile("ppu.cp.async.wait_group 0;");
                break;
        case 1: asm volatile("ppu.cp.async.wait_group 1;");
                break;
        case 2: asm volatile("ppu.cp.async.wait_group 2;");
                break;
        case 3: asm volatile("ppu.cp.async.wait_group 3;");
                break;
        case 4: asm volatile("ppu.cp.async.wait_group 4;");
                break;
        case 5: asm volatile("ppu.cp.async.wait_group 5;");
                break;
        case 6: asm volatile("ppu.cp.async.wait_group 6;");
                break;
        case 7: asm volatile("ppu.cp.async.wait_group 7;");
                break;
        case 8: asm volatile("ppu.cp.async.wait_group 8;");
                break;
        default: asm volatile("ppu.cp.async.wait_group 0;");
                 break;
    }
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t softplus_impl(scalar_t x, scalar_t beta, scalar_t threshold) {
    float beta_x = float(beta) * x;
    if (beta_x <= threshold) {
        return (1.0f / beta) * logf(1.0f + __expf(beta_x));
    } else {
        return x;
    }
}

__device__ __forceinline__ int swizzle_col_idx(int row_id, int col_id) {
    int vec_id = col_id >> 2;
    return ((((row_id & 0x1f) ^ (vec_id & 0x7)) & 0x7) << 2) | (col_id & ~0x1c);
}

__device__ __forceinline__ int map_to_lane_id(int row_id) {
    return ((row_id >> 4) << 2) | (row_id & 3);
}

__device__ __forceinline__ int map_to_row_id(int warp_id, int lane_id) {
    return ((lane_id >> 2) << 4) | (warp_id << 2) | (lane_id & 3);
}

template<typename scalar_t, int low = 1, int high = 16>
__device__ __forceinline__ float warp_reduce_sum(scalar_t f_val) {
    #pragma unroll
    for (int offset = high; offset >= low; offset /= 2) {
        // f_val += __shfl_down_sync(0xffffffff, f_val, offset);
        f_val += __shfl_xor_sync(0xffffffff, float(f_val), offset); // better perf
    }
    return f_val;
}

// Optimized block reduction using warp-level primitives first
template<int WARP_NUM, typename scalar_t>
__device__ float block_reduce_sum(scalar_t val, float* shared_mem) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int lane_id = tid & 31;  // threadIdx.x % 32
    const int warp_id = tid >> 5;  // threadIdx.x / 32

    // Convert to float for shuffle operations
    float f_val = static_cast<float>(val);

    // Warp-level reduction using shuffle operations
    f_val = warp_reduce_sum(f_val);

    // Write reduced value of each warp to shared memory
    // Only lane 0 of each warp writes
    if (lane_id == 0) {
        shared_mem[warp_id] = f_val;
    }

    // Synchronize so all warp results are written
    __syncthreads();

    // Sequential reduction by thread 0 only
#if 0
    float result = 0.0f;
    if (tid == 0) {
        int num_warps = (block_size + 31) / 32;
        for (int i = 0; i < num_warps; i++) {
            result += shared_mem[i];
        }
        shared_mem[0] = result;
    }
#else
    float result = tid < (block_size + 31) / 32 ? shared_mem[tid] : 0.0f;

    if (warp_id == 0) {
        #pragma unroll
        for (int offset = WARP_NUM / 2; offset > 0; offset /= 2) {
            result += __shfl_down_sync(0xffffffff, result, offset);
        }
        if (lane_id == 0) {
            shared_mem[0] = result;
        }
    }

#endif

    // Synchronize so final result is available to all threads
    __syncthreads();

    // All threads read the final result
    return shared_mem[0];
}

template <typename scalar_t>
__device__ __forceinline__ float2 convert_to_2floats(scalar_t* pTwoVals) {
    ppu_bfloat162 vals = *reinterpret_cast<const ppu_bfloat162*>(pTwoVals);
    return __bfloat1622float2(vals);
}

template<>
__device__ __forceinline__ float2 convert_to_2floats<half>(half* pTwoVals) {
    half2 vals = *reinterpret_cast<const half2*>(pTwoVals);
    return __half22float2(vals);
}

template<int V, int BV>
__device__ __forceinline__ void async_cp_h0(float* p_h0, float* shm_h0, const int h0_offset_v_base, const int row_idx, const int col_idx, const bool pref_pred = true) {

    // PHASE 1: Load K and h0 into shared memory with vector loads
    // Each warp loads 4 rows of 32 values (128 values total per warp)
    // 4 warps read 16 rows

    const int h0_offset = h0_offset_v_base + row_idx * V + col_idx;

    const void* gmem_ptr = &p_h0[h0_offset];
    void* shm_ptr = shm_h0 + row_idx * BV + col_idx;

    #ifdef GINO_DEBUG
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && (threadIdx.x == 0 || threadIdx.x == 8)) {
    //     printf("tid=%d, grp_idx=%d, h0_offset=%d, shm_offset=%d, pref_pred=%d\n", threadIdx.x, grp_idx, h0_offset, row_idx * BV + col_idx, pref_pred);
    // }
    #endif
    cp_async<sizeof(float4)>(shm_ptr, gmem_ptr, pref_pred);
}


// __maxnreg__(128)
template<typename alog_t, typename scalar_t, int K, int V, int H, int HV, int BV, int WARP_NUM>
static __global__ void fused_sigmoid_gating_delta_rule_cuda_kernel(
    const alog_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ dt_bias,
    const float softplus_beta,
    const float softplus_threshold,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ o,
    float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ cu_seqlens,
    const float scale,
    const int T,
    const int B,
    // const int H, const int HV,
    const bool use_initial_state,
    const bool use_qk_l2norm_in_kernel,
    const bool is_varlen
) {
    constexpr int H0_VEC_LEN = sizeof(float4) / sizeof(float);
    constexpr int WARP_SIZE = 32;
    constexpr int LANE_NUM_PER_ROW = BV / H0_VEC_LEN;
    constexpr int ROW_NUM_PER_WARP_PER_GRP = WARP_SIZE / LANE_NUM_PER_ROW; //4-8

    constexpr int GRP_ROW_NUM = ROW_NUM_PER_WARP_PER_GRP * WARP_NUM; //16-32
    constexpr int GRP_ITER_NUM = (K / GRP_ROW_NUM); //8-4
    constexpr int CPY_ROW_PER_WARP = GRP_ROW_NUM / WARP_NUM; //4-8
    constexpr int GROUP_SIZE = HV / H; // divide_by_power_of_two(HV, H);

    // Thread indices matching Triton
    const int i_n = blockIdx.z;  // batch index
    const int i_hv = blockIdx.y; // head V index
    const int i_bv = blockIdx.x; // Block V index
    const int tid = threadIdx.x; // Thread within block
    const int warp_id = __shfl_sync(0xffffffff, tid >> 5, 0);  // threadIdx.x / 32
    const int lane_id = tid & 31;  // threadIdx.x % 32

    // Decompose indices
    const int i_h = i_hv / GROUP_SIZE; // divide_by_power_of_two(i_hv, GROUP_SIZE);

    // Sequence boundary calculations
    int bos, seq_len;
    if (is_varlen) {
        bos = __ldg(&cu_seqlens[i_n]);
        seq_len = __ldg(&cu_seqlens[i_n + 1]) - bos;
    } else {
        bos = i_n * T;
        seq_len = T;
    }

    // Assert T == 1 as required
    if (seq_len != 1) return;

    // Shared memory layout for the new implementation
    extern __shared__ float shared_mem[];

    float* shared_h0 = shared_mem;
    float* shared_h_k_gemv = &shared_mem[K * BV];       // Hidden state (K x BV floats)
    float* shared_out = &shared_mem[K * BV + WARP_NUM * BV];
    float* shared_q = reinterpret_cast<float*>(&shared_mem[K * BV + WARP_NUM * BV * 2]);
    float* shared_k = reinterpret_cast<float*>(&shared_mem[K * BV + WARP_NUM * BV * 2 + K]);
    float* shared_reduction_q = &shared_mem[K * BV + WARP_NUM * BV * 2 + K * 2]; // Reduction buffer (32 floats)
    float* shared_reduction_k = &shared_mem[K * BV + WARP_NUM * BV * 2 + K * 2 + WARP_NUM]; // Reduction buffer (32 floats)

    // Load q and k for this thread's k index
    float b_q_val = float(q[(bos * H + i_h) * K + tid]);
    float b_k_val = float(k[(bos * H + i_h) * K + tid]);

    // Initial state handling
    float* p_h0 = nullptr;
    int idx = -1;
    if (use_initial_state && h0_source != nullptr) {
        idx = h0_indices[i_n];
        if (idx >= 0) {
            p_h0 = h0_source + (int64_t)idx * HV * K * V;
        }
    }
    if (idx < 0) {
        return;
    }

    const int col_idx = (lane_id % (BV / H0_VEC_LEN)) * H0_VEC_LEN;
    const int row_idx_in_warp = (lane_id / (BV / H0_VEC_LEN));
    // float4 h0_f4[GRP_ITER_NUM];

    int v_base = i_bv * BV;
    int v_base_idx = (bos * HV + i_hv) * V + v_base;
    const int h0_offset_v_base = i_hv * K * V + v_base;
    #ifdef ASYNC_COPY_H0
    #pragma unroll
    for (int grp_idx = 0; grp_idx < GRP_ITER_NUM; grp_idx++) {
        const int row_start_per_warp = grp_idx * GRP_ROW_NUM + (warp_id * CPY_ROW_PER_WARP); // Each warp handles 4 rows
        const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading
        async_cp_h0<V, BV>(p_h0, shared_h0, h0_offset_v_base, row_idx, col_idx);
        cp_async_commit();
    }
    #endif

    float b_beta, b_g_exp;
    if (lane_id == 0) {
        // Load gating parameters
        scalar_t b_b = b[bos * HV + i_hv];
        #if defined(__HGGC_ARCH__)
            b_beta = __ppu_sgmdf(float(b_b));
        #else
            b_beta = 1.0f / (1.0f + __expf(float(-b_b)));
        #endif

        float b_A_log = A_log[i_hv];
        scalar_t b_a = a[bos * HV + i_hv];
        scalar_t b_dt_bias = dt_bias[i_hv];

        // Compute gating
        float x = float(b_a) + float(b_dt_bias);
        float b_g = -__expf(b_A_log) * softplus_impl(x, softplus_beta, softplus_threshold);
        b_g_exp = __expf(b_g);

        #ifdef GINO_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            // printf("tid=%d, b_g_exp=%f, b_beta=%f\n", tid, float(b_g_exp), float(b_beta));
        }
        #endif
    }
    b_beta = __shfl_sync(0xFFFFFFFF, b_beta, 0);
    b_g_exp = __shfl_sync(0xFFFFFFFF, b_g_exp, 0);

    // L2 normalization
    if (use_qk_l2norm_in_kernel) {
        // Compute squared sums
        float q_sq = b_q_val * b_q_val;
        float k_sq = b_k_val * b_k_val;

        // Reduce across block
        float sum_q_sq = block_reduce_sum<WARP_NUM>(q_sq, shared_reduction_q);
        float sum_k_sq = block_reduce_sum<WARP_NUM>(k_sq, shared_reduction_k);

        // Normalize
        float q_norm = sqrt(sum_q_sq + float(1e-6));
        float k_norm = sqrt(sum_k_sq + float(1e-6));
        b_q_val = divide_float(b_q_val, q_norm);
        b_k_val = divide_float(b_k_val, k_norm);
    }

    // Apply scaling
    b_q_val = b_q_val * scale;

    // make sure Q and K are written to shared memory
    shared_k[tid] = b_k_val;
    shared_q[tid] = b_q_val;
    __syncthreads();

    #ifdef GINO_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            // printf("tid=%d, b_q=%f, b_k=%f\n", tid, float(b_q_val), float(b_k_val));
        }
    #endif

    // Process each group (matching loop in Triton)
    // #pragma unroll 2
    // for (int gid = 0; gid < GROUP_SIZE; gid++) {
        // int gid =
        // const int i_hv = i_h * GROUP_SIZE + gid;


        // Process V in tiles of BV=32

        // prefetch for V
        // #ifdef __HGGC_ARCH__
        // if (warp_id == 0) {
        //     __ppu_vmem_prefetch_bulk_b32x2(&v[v_base_idx], 1);
        // }
        // #endif
        // #pragma unroll
        // for (int v_base = 0; v_base < V; v_base += BV, v_base_idx += BV) {


            float4 h0_f4[GRP_ITER_NUM];
            float4 h_mul_k_per_warp = {.0f, .0f, .0f, .0f};

            // b_v -= tl.sum(b_h * b_k[:, None], 0)
            // loop in K dimension, each loop handles 32 rows in h0
            #pragma unroll
            for (int grp_idx = 0; grp_idx < GRP_ITER_NUM; grp_idx++) {
                // PHASE 1: Load K and h0 into shared memory with vector loads
                // Each warp loads 4 rows of 32 values (128 values total per warp)
                // 4 warps read 16 rows

                const int row_start_per_warp = grp_idx * GRP_ROW_NUM + (warp_id * CPY_ROW_PER_WARP); // Each warp handles 4 rows
                const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading (4 rows per warp)
                //

                // Apply gating to hidden state: h *= exp(gid)
                // Unroll the loop for better performance


                #ifdef ASYNC_COPY_H0
                cp_async_wait(GRP_ITER_NUM -1 - grp_idx);
                h0_f4[grp_idx] = *reinterpret_cast<float4*>(shared_h0 + row_idx * BV + col_idx);
                #else
                const int h0_offset = h0_offset_v_base + row_idx * V + col_idx;
                h0_f4[grp_idx] = __ldcv(reinterpret_cast<float4*>(&p_h0[h0_offset]));
                #endif

                h0_f4[grp_idx] = {h0_f4[grp_idx].x * float(b_g_exp), h0_f4[grp_idx].y * float(b_g_exp), h0_f4[grp_idx].z * float(b_g_exp), h0_f4[grp_idx].w * float(b_g_exp)};

                // PHASE 2: Compute b_v -= tl.sum(b_h * b_k[:, None], 0)
                // Each loop computes 32 rows and 16 columns of output
                // Need 4x2 such loops to cover all K dimensions
                // int vreg_lane_id = map_to_lane_id(row_idx);
                // float k_per_thread = __shfl_sync(0xFFFFFFFF, b_k_val, vreg_lane_id);
                b_k_val = shared_k[row_idx];
                h_mul_k_per_warp.x += h0_f4[grp_idx].x * b_k_val,
                h_mul_k_per_warp.y += h0_f4[grp_idx].y * b_k_val,
                h_mul_k_per_warp.z += h0_f4[grp_idx].z * b_k_val,
                h_mul_k_per_warp.w += h0_f4[grp_idx].w * b_k_val;

                // #ifdef GINO_DEBUG
                // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 /*&& tid < (BV / H0_VEC_LEN)*/ && gid == 0 && v_base == 0) {
                //     printf("tid=%d, grp_idx=%d, h_mul_k_per_warp=%f, %f, %f, %f, h0_f4=%f, %f, %f, %f, k_per_thread=%f\n", tid, grp_idx,
                //                                                             float(h_mul_k_per_warp.x),
                //                                                             float(h_mul_k_per_warp.y),
                //                                                             float(h_mul_k_per_warp.z),
                //                                                             float(h_mul_k_per_warp.w),
                //                                                             float(h0_f4[grp_idx].x),
                //                                                             float(h0_f4[grp_idx].y),
                //                                                             float(h0_f4[grp_idx].z),
                //                                                             float(h0_f4[grp_idx].w),
                //                                                             k_per_thread);
                // }
                // #endif
                // #ifdef GINO_DEBUG
                // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 /*&& tid < (BV / H0_VEC_LEN)*/ && gid == 0 && v_base == 0) {

                //     float4 reduce_sum_in_cur_loop = h_k_mul_cur_loop;
                //     for (int offset = 16; offset >= BV / H0_VEC_LEN; offset /= 2) {
                //         reduce_sum_in_cur_loop.x += __shfl_xor_sync(0xffffffff, reduce_sum_in_cur_loop.x, offset);
                //         reduce_sum_in_cur_loop.y += __shfl_xor_sync(0xffffffff, reduce_sum_in_cur_loop.y, offset);
                //         reduce_sum_in_cur_loop.z += __shfl_xor_sync(0xffffffff, reduce_sum_in_cur_loop.z, offset);
                //         reduce_sum_in_cur_loop.w += __shfl_xor_sync(0xffffffff, reduce_sum_in_cur_loop.w, offset);
                //     }
                //     printf("tid=%d, grp_idx=%d, h_k_mul_cur_loop=%f, %f, %f, %f, reduce_sum_in_cur_loop=%f, %f, %f, %f\n", tid, grp_idx,
                //                                                             float(h_k_mul_cur_loop.x),
                //                                                             float(h_k_mul_cur_loop.y),
                //                                                             float(h_k_mul_cur_loop.z),
                //                                                             float(h_k_mul_cur_loop.w),
                //                                                             float(reduce_sum_in_cur_loop.x),
                //                                                             float(reduce_sum_in_cur_loop.y),
                //                                                             float(reduce_sum_in_cur_loop.z),
                //                                                             float(reduce_sum_in_cur_loop.w));
                // }
                // #endif
            } // group loop
            // cp_async_commit();

            #pragma unroll
            for (int offset = 16; offset >= BV / H0_VEC_LEN; offset /= 2) {
                // all threads hold acc in column direction
                h_mul_k_per_warp.x += __shfl_xor_sync(0xffffffff, h_mul_k_per_warp.x, offset);
                h_mul_k_per_warp.y += __shfl_xor_sync(0xffffffff, h_mul_k_per_warp.y, offset);
                h_mul_k_per_warp.z += __shfl_xor_sync(0xffffffff, h_mul_k_per_warp.z, offset);
                h_mul_k_per_warp.w += __shfl_xor_sync(0xffffffff, h_mul_k_per_warp.w, offset);
            }

            if (lane_id < (BV / H0_VEC_LEN)) {
                *reinterpret_cast<float4*>(&shared_h_k_gemv[warp_id * BV + lane_id * H0_VEC_LEN]) = h_mul_k_per_warp;
            }
            __syncthreads();
            if (tid < (BV / H0_VEC_LEN)) {
                float4 sum_lane8 = *reinterpret_cast<float4*>(&shared_h_k_gemv[lane_id * H0_VEC_LEN]);
                #pragma unroll
                for (int wid = 1; wid <WARP_NUM; wid++) {
                    float4 sum_per_warp = *reinterpret_cast<float4*>(&shared_h_k_gemv[wid * BV + lane_id * H0_VEC_LEN]);
                    sum_lane8.x += sum_per_warp.x;
                    sum_lane8.y += sum_per_warp.y;
                    sum_lane8.z += sum_per_warp.z;
                    sum_lane8.w += sum_per_warp.w;
                }

                scalar_t v4[H0_VEC_LEN];
                *reinterpret_cast<float2*>(v4) = *reinterpret_cast<const float2*>(&v[v_base_idx + col_idx]);
                float2 xy = convert_to_2floats(v4);
                float2 wz = convert_to_2floats(v4 + H0_VEC_LEN / 2);
                sum_lane8.x = (xy.x - sum_lane8.x) * b_beta;
                sum_lane8.y = (xy.y - sum_lane8.y) * b_beta;
                sum_lane8.z = (wz.x - sum_lane8.z) * b_beta;
                sum_lane8.w = (wz.y - sum_lane8.w) * b_beta;

                *reinterpret_cast<float4*>(&shared_h_k_gemv[lane_id * H0_VEC_LEN]) = sum_lane8;
            }
            __syncthreads();

            #ifdef GINO_DEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid < (BV / H0_VEC_LEN) && gid == 0 && v_base == 0) {
                // printf("tid=%d, shared_h_k_gemv=%f, %f, %f, %f, h_mul_k_per_warp=%f, %f, %f, %f\n", tid,
                //                                                         float(shared_h_k_gemv[lane_id * H0_VEC_LEN]),
                //                                                         float(shared_h_k_gemv[lane_id * H0_VEC_LEN + 1]),
                //                                                         float(shared_h_k_gemv[lane_id * H0_VEC_LEN + 2]),
                //                                                         float(shared_h_k_gemv[lane_id * H0_VEC_LEN + 3]),
                //                                                         float(h_mul_k_per_warp.x),
                //                                                         float(h_mul_k_per_warp.y),
                //                                                         float(h_mul_k_per_warp.z),
                //                                                         float(h_mul_k_per_warp.w));
            }
            #endif

            h_mul_k_per_warp = *reinterpret_cast<float4*>(&shared_h_k_gemv[(tid % (BV / H0_VEC_LEN)) * H0_VEC_LEN]);

            #ifdef GINO_DEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid < (BV / H0_VEC_LEN) && gid == 0 && v_base == 0) {
                // printf("tid=%d, new_v=%f, %f, %f, %f, old_v=%f, %f, %f, %f\n", tid,
                //                                                         float(h_mul_k_per_warp.x),
                //                                                         float(h_mul_k_per_warp.y),
                //                                                         float(h_mul_k_per_warp.z),
                //                                                         float(h_mul_k_per_warp.w),
                //                                                         float(v4[0]),
                //                                                         float(v4[1]),
                //                                                         float(v4[2]),
                //                                                         float(v4[3]));
            }
            #endif
            float4 out_f4 = {.0f, .0f, .0f, .0f};
            #pragma unroll
            for (int grp_idx = 0; grp_idx < GRP_ITER_NUM; grp_idx++) {
                // PHASE 3: Compute b_h = tl.fma(b_k[:, None], b_v[None, :], b_h)
                // And compute b_o = tl.sum(b_h * b_q[:, None], 0) in the same loop

                const int row_start_per_warp = grp_idx * GRP_ROW_NUM + (warp_id * CPY_ROW_PER_WARP); // Each warp handles 4 rows
                const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading (4 rows per warp)
                const int h0_offset = h0_offset_v_base + row_idx * V + col_idx;

                // int vreg_lane_id = map_to_lane_id(row_idx);
                // float k_per_thread = __shfl_sync(0xFFFFFFFF, b_k_val, vreg_lane_id);
                b_k_val = shared_k[row_idx];
                h0_f4[grp_idx].x += h_mul_k_per_warp.x * b_k_val;
                h0_f4[grp_idx].y += h_mul_k_per_warp.y * b_k_val;
                h0_f4[grp_idx].z += h_mul_k_per_warp.z * b_k_val;
                h0_f4[grp_idx].w += h_mul_k_per_warp.w * b_k_val;

                // float q_per_thread = __shfl_sync(0xFFFFFFFF, b_q_val, vreg_lane_id);
                b_q_val = shared_q[row_idx];
                out_f4.x += h0_f4[grp_idx].x * b_q_val;
                out_f4.y += h0_f4[grp_idx].y * b_q_val;
                out_f4.z += h0_f4[grp_idx].z * b_q_val;
                out_f4.w += h0_f4[grp_idx].w * b_q_val;

                // *reinterpret_cast<float4*>(&p_h0[h0_offset]) = h0_f4[grp_idx];
                __stcg(reinterpret_cast<float4*>(&p_h0[h0_offset]), h0_f4[grp_idx]);
            }

            #pragma unroll
            for (int offset = 16; offset >= BV / H0_VEC_LEN; offset /= 2) {
                // all threads hold acc in column direction
                out_f4.x += __shfl_xor_sync(0xffffffff, out_f4.x, offset);
                out_f4.y += __shfl_xor_sync(0xffffffff, out_f4.y, offset);
                out_f4.z += __shfl_xor_sync(0xffffffff, out_f4.z, offset);
                out_f4.w += __shfl_xor_sync(0xffffffff, out_f4.w, offset);
            }


            if (lane_id < (BV / H0_VEC_LEN)) {
                *reinterpret_cast<float4*>(&shared_out[warp_id * BV + lane_id * H0_VEC_LEN]) = out_f4;
            }
            __syncthreads();

            if (tid < (BV / H0_VEC_LEN)) {
                float4 sum_lane8 = *reinterpret_cast<float4*>(&shared_out[lane_id * H0_VEC_LEN]);
                #pragma unroll
                for (int wid = 1; wid <WARP_NUM; wid++) {
                    float4 sum_per_warp = *reinterpret_cast<float4*>(&shared_out[wid * BV + lane_id * H0_VEC_LEN]);
                    sum_lane8.x += sum_per_warp.x;
                    sum_lane8.y += sum_per_warp.y;
                    sum_lane8.z += sum_per_warp.z;
                    sum_lane8.w += sum_per_warp.w;
                }

                scalar_t out_scalar4[H0_VEC_LEN];
                float* out_p = &sum_lane8.x;
                #pragma unroll
                for (int i = 0; i < H0_VEC_LEN; i++) {
                    out_scalar4[i] = scalar_t(out_p[i]);
                }
                *reinterpret_cast<float2*>(&o[v_base_idx + col_idx]) = *reinterpret_cast<float2*>(&out_scalar4[0]);
            }
        // } // for (int v_base = 0; v_base < V; v_base += BV)
    // } // gid
}

template<typename alog_t, typename scalar_t, int K, int V, int H>
static void dispatch_V_Head_Num(
    const alog_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ dt_bias,
    const float softplus_beta,
    const float softplus_threshold,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ o,
    float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ cu_seqlens,
    const float scale,
    const int T,
    const int B,
    const int HV,
    const bool use_initial_state,
    const bool use_qk_l2norm_in_kernel,
    const bool is_varlen,
    const hggcStream_t stream
) {

    // Kernel configuration: 4 warps per block (128 threads)
#if defined (__HGGCCC__)
    constexpr int BV = 32;
#else
    constexpr int BV = 16;
#endif
    const int blocks_per_grid_bv = V / BV;
    const int blocks_per_grid_hv = HV;
    const int blocks_per_grid_n = B;
    const int WARP_NUM = (K / 32); // / 2;
    const int threads_per_block = WARP_NUM * 32;
    dim3 block(threads_per_block);
    dim3 grid(blocks_per_grid_bv, blocks_per_grid_hv, blocks_per_grid_n);

    // Shared memory calculation
    // We need: K (for k) + K (for q) + BV (for new_v & output) + K*BV (for h0) + 32 (for reduction)
    const int shared_mem_size = (K * BV + WARP_NUM * BV * 2 + K * 2 + WARP_NUM*2) * sizeof(float);

#if defined (__HGGCCC__)
#define DISPATCH_HV_WITH_ATTR(HV_VAL) \
    if (HV == HV_VAL) { \
        hggcFuncSetAttribute(fused_sigmoid_gating_delta_rule_cuda_kernel<alog_t, scalar_t, K, V, H, HV_VAL, BV, WARP_NUM>, \
                             hggcFuncAttributeBlockAgeEn, 1); \
        fused_sigmoid_gating_delta_rule_cuda_kernel<alog_t, scalar_t, K, V, H, HV_VAL, BV, WARP_NUM><<<grid, block, shared_mem_size, stream>>>( \
            A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, \
            h0_source, h0_indices, cu_seqlens, scale, T, B, \
            use_initial_state, use_qk_l2norm_in_kernel, is_varlen \
        ); \
        return; \
    }
#else
#define DISPATCH_HV_WITH_ATTR(HV_VAL) \
    if (HV == HV_VAL) { \
        fused_sigmoid_gating_delta_rule_cuda_kernel<alog_t, scalar_t, K, V, H, HV_VAL, BV, WARP_NUM><<<grid, block, shared_mem_size, stream>>>( \
            A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, \
            h0_source, h0_indices, cu_seqlens, scale, T, B, \
            use_initial_state, use_qk_l2norm_in_kernel, is_varlen \
        ); \
        return; \
    }
#endif

    DISPATCH_HV_WITH_ATTR(128);
    DISPATCH_HV_WITH_ATTR(64);
    DISPATCH_HV_WITH_ATTR(48);
    DISPATCH_HV_WITH_ATTR(32);
    DISPATCH_HV_WITH_ATTR(24);
    DISPATCH_HV_WITH_ATTR(16);
    DISPATCH_HV_WITH_ATTR(12);
    DISPATCH_HV_WITH_ATTR(8);
    DISPATCH_HV_WITH_ATTR(6);
    DISPATCH_HV_WITH_ATTR(4);
    DISPATCH_HV_WITH_ATTR(3);
    DISPATCH_HV_WITH_ATTR(2);
    DISPATCH_HV_WITH_ATTR(1);
    throw std::runtime_error(std::string("pla.decode.gdn.v_last smallbatch does not support HV=") + std::to_string(HV));
}

// Local helper of dispatch_V_Head_Num above; do not leak it further.
#undef DISPATCH_HV_WITH_ATTR

// Entry point instantiated once per H by the per-H translation units.
template<int H>
static void smallbatch_launch_for_H(PLA_V_LAST_ENTRY_PARAMS) {
    PLA_V_LAST_DISPATCH_ALOG(args.alog_dtype, ([&] {
        PLA_V_LAST_DISPATCH_SCALAR(args.scalar_dtype, ([&] {
            if (args.Kdim == 128 && args.Vdim == 128) {
                dispatch_V_Head_Num<alog_t, scalar_t, 128, 128, H>(
                    static_cast<const alog_t*>(args.A_log),
                    static_cast<const scalar_t*>(args.a),
                    static_cast<const scalar_t*>(args.dt_bias),
                    args.softplus_beta,
                    args.softplus_threshold,
                    static_cast<const scalar_t*>(args.q),
                    static_cast<const scalar_t*>(args.k),
                    static_cast<const scalar_t*>(args.v),
                    static_cast<const scalar_t*>(args.b),
                    static_cast<scalar_t*>(args.o),
                    args.h0_source,
                    args.h0_indices,
                    args.cu_seqlens,
                    args.scale,
                    args.T, args.B, args.HV,
                    args.use_initial_state,
                    args.use_qk_l2norm_in_kernel,
                    args.is_varlen,
                    args.stream
                );
            }
        }));
    }));
}
