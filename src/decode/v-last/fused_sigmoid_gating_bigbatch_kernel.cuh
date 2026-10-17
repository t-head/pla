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
// Fused sigmoid gating delta rule -- v-last (V-major) bigbatch kernel.
//
// Split out of fused_sigmoid_gating_cuda_bigbatch.cu so the template
// instantiations can be spread over one translation unit per H value
// (fused_sigmoid_gating_bigbatch_h*.cu). The full cascade is
//   3 A_log dtypes x 2 scalar dtypes x 6 H x 12 HV x <=4 hv_per_cta
// which is why compiling it as a single TU took ~25 minutes.
#include <stdexcept>

#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <hggc_pipeline.h>

#ifdef __HGGCCC__
#include <hggc_ad.h>
#endif

// common.cuh was included only for its DISPATCH_TYPES/DISPATCH_ALOG_TYPES
// macros -- PLA_CUSTOM_DEVICE_HELPERS was defined precisely to switch off its
// device-helper half, which this file provides itself. Those macros spoke torch
// dtypes, so the torch-free equivalents in v_last_args.h replace them and the
// torch dependency disappears with the include.
#include "v_last_args.h"
// #define GINO_DEBUG

#define ENABLE_BETA_IN_TSM
// AIU exists on both PPU generations (see ppu_aiu_load below for the per-arch
// encodings).
#if defined (__HGGC_ARCH__) && (__HGGC_ARCH__ == 150 || __HGGC_ARCH__ == 100)
    #define PLA_USE_AIU
#endif

#define BV 32

// Cached CU count; defined once in fused_sigmoid_gating_v_last.cpp.
// MUST have external linkage: the host entry writes it in another
// translation unit while choose_hv_per_cta() below reads it.
extern int g_v_last_cu_count;

inline float calculate_waste_ratio(int batch, int H, int HV, int hv_per_cta, int tb_per_cu) {
    if (hv_per_cta > (HV / H)) {
        return 1.0f;
    }

    float wave = float(batch * (HV / H / hv_per_cta)) / (g_v_last_cu_count * tb_per_cu);
    return (std::ceil(wave) - wave) / std::ceil(wave);
}

inline int choose_hv_per_cta(int batch, int H, int HV) {
    float min_waste_ratio = 1.0f, cur_waste_ratio = 1.0f;
    int hv_per_cta = 1;
    int group_size = (HV / H);
    // try 1
    if ((cur_waste_ratio = calculate_waste_ratio(batch, H, HV, 1, 10)) < min_waste_ratio) {
        hv_per_cta = 1;
        min_waste_ratio = cur_waste_ratio;
    }
    // try 2
    if ((group_size % 2 == 0) && (cur_waste_ratio = calculate_waste_ratio(batch, H, HV, 2, 8)) < min_waste_ratio) {
        hv_per_cta = 2;
        min_waste_ratio = cur_waste_ratio;
    }
    // try 3
    if ((group_size % 3 == 0) && (cur_waste_ratio = calculate_waste_ratio(batch, H, HV, 3, 8)) < min_waste_ratio) {
        hv_per_cta = 3;
        min_waste_ratio = cur_waste_ratio;
    }
    // try 4
    if ((group_size % 4 == 0) && (cur_waste_ratio = calculate_waste_ratio(batch, H, HV, 4, 8)) < min_waste_ratio) {
        hv_per_cta = 4;
        min_waste_ratio = cur_waste_ratio;
    }

    if (g_v_last_cu_count != 39) {
        // only enable this optimization for PPU1.5
        hv_per_cta = HV / H;
    }
    return hv_per_cta;
}

#ifdef __HGGC_ARCH__
#define Global __attribute__((address_space(1)))
typedef float v4float __attribute__((ext_vector_type(4)));
#endif

#define CHECK_CU(err) __check_cuda_driver((err), #err, __FILE__, __LINE__)

inline void __check_cuda_driver(HGresult err, const char* op,
                                  const char* file, int line) {
    if (err != HGGC_SUCCESS) {
        const char* err_name    = nullptr;
        const char* err_message = nullptr;
        hgGetErrorName(err, &err_name);      // get error name
        hgGetErrorString(err, &err_message); // get error description
        fprintf(stderr, "[CUDA Driver Error] %s:%d  %s failed\n"
                        "  Error: %s (%d)\n"
                        "  Info:  %s\n",
                file, line, op, err_name, err, err_message);
        exit(EXIT_FAILURE);
    }
}

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

  // Make sure the size is supported.
  static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
    "Size is not supported");

  uint32_t smem_int_ptr = cvta_sm_addr(smem_ptr);

#if defined (__HGGC_ARCH__)
  /*
       kp = 1 (CA)	cache in all level;
       kp = 2 (CG)	read through L1/L2, cache in LLC;
       kp = 3 (CV)	read through L1/L2/LLC;
       kp = 6 (BL)	bypass L1/L2;
       kp = 7 (BA)	bypass L1/L2/LLC.
  */
  asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  ppu.cmpp.ne.b32 p, %0, 0;\n"
    "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
#else
  asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  ppu.cmpp.ne.b32 p, %0, 0;\n"
    "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));
#endif
// #endif
}

#ifdef PLA_USE_AIU
template <int CUBE_C, int CUBE_W, int CUBE_N, int DIM_C, int DIM_W, int DIM_N, int64_t TENSOR_STRIDE_W, int64_t TENSOR_STRIDE_N>
__device__ __forceinline__ void ppu_aiu_load(void *smem_ptr, const void* gmem_ptr, int coord_c, int coord_w, int coord_n) {
    uint32_t smem_int_ptr = cvta_sm_addr(smem_ptr);
#if defined (__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    // ppu001 spells the same copy as a 3d.linear bulk tensor: the {dim} and
    // {start} triples are h-major, and there are no stride operands at all --
    // the source must be a dense [DIM_N][DIM_W][DIM_C] cube. Assert the strides
    // the caller computed agree with that, otherwise the copy would silently
    // read the wrong rows.
    static_assert(TENSOR_STRIDE_W == (int64_t)DIM_C * 4,
                  "ppu001 AIU needs a contiguous w axis: TENSOR_STRIDE_W must be DIM_C * sizeof(float)");
    static_assert(CUBE_N == 1 || TENSOR_STRIDE_N == (int64_t)DIM_W * DIM_C * 4,
                  "ppu001 AIU needs a contiguous n axis when CUBE_N > 1");
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.3d.b32 [%0], [%1], "
        "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_int_ptr), "l"(gmem_ptr),
          "r"(DIM_N), "r"(DIM_W), "r"(DIM_C),
          "r"(coord_n), "r"(coord_w), "r"(coord_c),
          "r"(CUBE_N), "r"(CUBE_W), "r"(CUBE_C), "r"(1)
    );
#else
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b32"
        "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(smem_int_ptr), "l"(gmem_ptr),
          "r"(DIM_C), "r"(DIM_W), "r"(DIM_N),
          "r"(CUBE_C), "r"(CUBE_W), "r"(CUBE_N),
          "l"(TENSOR_STRIDE_W), "l"(TENSOR_STRIDE_N),
          "r"(coord_c), "r"(coord_w), "r"(coord_n),
          "r"(0)
    );
#endif

#ifdef GINO_DEBUG
    // if (/*blockIdx.x == 0 && (blockIdx.y == 0) && blockIdx.z == 0 &&*/ threadIdx.x == 0) {
    //     printf("blockIdx.xyz=%d, i_hv=%d, dim_c=%d, dim_w=%d, dim_n=%d, coord_c=%d, coord_w=%d, coord_n=%d, tensor_stride_w=%ld, tensor_stride_n=%ld\n",
    //     blockIdx.x, blockIdx.y, blockIdx.z, i_hv, dim_c, dim_w, dim_n, coord_c, coord_w, coord_n, tensor_stride_w, tensor_stride_n);
    // }
#endif
};
#endif

__device__ __forceinline__ void cp_async_commit()
{
#ifdef PLA_USE_AIU
    __pipeline_commit();
#else
    asm volatile("ppu.cp.async.commit_group;\n" ::);
#endif
}

template<int PIPE>
__device__ void cp_async_wait() {
#ifdef PLA_USE_AIU
    __pipeline_wait_prior(PIPE);
#else
  asm volatile("ppu.cp.async.wait_group %0;" :: "n"(PIPE));
#endif
}


template<typename scalar_t>
__device__ __forceinline__ scalar_t softplus_impl(scalar_t x, scalar_t beta, scalar_t threshold) {
    float beta_x = float(beta) * x;
    if (beta_x <= threshold) {
        return (1.0f / beta) * __logf(1.0f + __expf(beta_x));
    } else {
        return x;
    }
}

__device__ __forceinline__ int swizzle_col_idx(int row_id, int col_id) {
    int vec_id = col_id >> 2;
    return ((((row_id & 0x1f) ^ (vec_id & 0x7)) & 0x7) << 2) | (col_id & ~0x1c);
}

__device__ __forceinline__ int map_to_lane_id(int row_id) {
    // return (row_id / 16) * 4 + (row_id % 4);
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
    const int warp_id = __shfl_sync(0xffffffff, tid / 32, 0);  // threadIdx.x / 32

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
__device__ __forceinline__ float2 convert_to_2floats<__half>(__half* pTwoVals) {
    half2 vals = *reinterpret_cast<const half2*>(pTwoVals);
    return __half22float2(vals);
}

template<int GRP_ROW_NUM, int ROW_NUM_PER_WARP_PER_GRP, int V>
__device__ __forceinline__ void async_cp_h0(float* p_h0, float* shm_h0, const int h0_offset_v_head_base, const int row_idx, const int col_idx_in_warp, const bool pref_pred = true) {

    // PHASE 1: Load K and h0 into shared memory with vector loads
    // Each warp loads 4 rows of 32 values (128 values total per warp)
    // 4 warps read 16 rows

    const int h0_offset = h0_offset_v_head_base + row_idx * V + col_idx_in_warp;

    const void* gmem_ptr = &p_h0[h0_offset];
    void* shm_ptr = shm_h0 + row_idx * BV + col_idx_in_warp;

    #ifdef GINO_DEBUG
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && (threadIdx.x == 0 || threadIdx.x == 8)) {
    //     printf("tid=%d, grp_idx=%d, h0_offset=%d, shm_offset=%d, pref_pred=%d\n", threadIdx.x, grp_idx, h0_offset, row_idx * BV + col_idx, pref_pred);
    // }
    #endif
    cp_async<sizeof(float4)>(shm_ptr, gmem_ptr, pref_pred);
}

// __maxnreg__(128)
template<typename alog_t, typename scalar_t, int STAGE_NUM, int K, int V, const int H, const int HV, const int WARP_NUM, const int HV_PER_BLOCK>
__global__ void fused_sigmoid_gating_delta_rule_cuda_kernel(
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

    constexpr int GRP_ROW_NUM = ROW_NUM_PER_WARP_PER_GRP * WARP_NUM; //16
    constexpr int GRP_ITER_NUM = (K / GRP_ROW_NUM); //8
    constexpr int TOTAL_ROW_PER_WARP = K / WARP_NUM; //32
    constexpr int GROUP_SIZE = HV / H;

    // Thread indices matching Triton
    const int i_n = blockIdx.z;
    const int i_h = blockIdx.y / (GROUP_SIZE / HV_PER_BLOCK);
    const int tid = threadIdx.x; // Thread within block
    const int warp_id = __shfl_sync(0xffffffff, tid / 32, 0);  // threadIdx.x / 32
    const int lane_id = tid & 31;  // threadIdx.x % 32

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

    float* shared_h0 = shared_mem; // Hidden state (STAGE_NUM x K x BV floats)
    float* shared_h_k_gemv = &shared_mem[STAGE_NUM * K * BV];
    float* shared_out = &shared_mem[STAGE_NUM * K * BV + WARP_NUM * BV]; // Out buffer (4 x BV foats)
    float* shared_reduction_q = &shared_mem[STAGE_NUM * K * BV + WARP_NUM * BV * 2]; // Reduction buffer (4 floats)
    float* shared_reduction_k = &shared_mem[STAGE_NUM * K * BV + WARP_NUM * BV * 2 + WARP_NUM]; // Reduction buffer (4 floats)

    // may be needed in the future to reduce vreg usage
#ifdef ENABLE_BETA_IN_TSM
    float* shared_beta = &shared_mem[STAGE_NUM * K * BV + WARP_NUM * BV * 2 + 2 * WARP_NUM]; // temp buffer (GROUP_SIZE floats)
    float* shared_g_exp = &shared_mem[STAGE_NUM * K * BV + WARP_NUM * BV * 2 + 2 * WARP_NUM + HV_PER_BLOCK]; // temp buffer (GROUP_SIZE floats)
#endif

    // Load q and k for this thread's k index
    float b_q_val = q[(bos * H + i_h) * K + map_to_row_id(warp_id, lane_id)];
    float b_k_val = k[(bos * H + i_h) * K + map_to_row_id(warp_id, lane_id)];

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

    int cur_stage_seq = 0;
    const int h0_offset_v_head_base = blockIdx.y * HV_PER_BLOCK * K * V;
#ifdef PLA_USE_AIU
    constexpr int CUBE_C = BV;
    constexpr int CUBE_W = ROW_NUM_PER_WARP_PER_GRP; //4
    constexpr int CUBE_N = GRP_ITER_NUM; //8
    constexpr int DIM_C = V;
    constexpr int DIM_W = GRP_ROW_NUM; //16
    constexpr int DIM_N = HV_PER_BLOCK * GRP_ITER_NUM; //2 * 8;
    constexpr int64_t tensor_stride_w = DIM_C * sizeof(float);
    constexpr int64_t tensor_stride_n = DIM_W * DIM_C * sizeof(float);
    const int coord_w = warp_id * ROW_NUM_PER_WARP_PER_GRP;
    int coord_c = 0;
    int coord_n = 0; //i_h * GROUP_SIZE;
    const int shm_offset_per_warp = warp_id * TOTAL_ROW_PER_WARP * BV;
    #pragma unroll
    for (int stage = 0; stage < STAGE_NUM; stage++) {
        ppu_aiu_load<CUBE_C, CUBE_W, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w, tensor_stride_n>
         (shared_h0 + stage * K * BV + shm_offset_per_warp,
          p_h0 + h0_offset_v_head_base,
          coord_c + stage * BV, coord_w, coord_n);
        cp_async_commit();
    }
#else
    #pragma unroll
    for (int stage = 0; stage < STAGE_NUM; stage++) {
        for (int grp_idx = 0; grp_idx < GRP_ITER_NUM; grp_idx++) {
            const int row_start_per_warp = grp_idx * GRP_ROW_NUM + (warp_id * ROW_NUM_PER_WARP_PER_GRP); // Each warp handles 4 rows
            const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading
            async_cp_h0<GRP_ROW_NUM, ROW_NUM_PER_WARP_PER_GRP, V>(p_h0, shared_h0 + stage * K * BV, h0_offset_v_head_base + stage * BV, row_idx, col_idx);
        }
        cp_async_commit();
    }
#endif

#ifndef ENABLE_BETA_IN_TSM
    float b_beta_all[HV_PER_BLOCK], b_g_exp_all[HV_PER_BLOCK];
#endif
    float b_beta, b_g_exp;
#ifndef ENABLE_BETA_IN_TSM
    if (lane_id < HV_PER_BLOCK) {
        const int i_hv = blockIdx.y * HV_PER_BLOCK + lane_id;
#else
    if (tid < HV_PER_BLOCK) {
        const int i_hv = blockIdx.y * HV_PER_BLOCK + tid;
#endif
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

#ifdef ENABLE_BETA_IN_TSM
        shared_beta[tid] = b_beta;
        shared_g_exp[tid] = b_g_exp;
#endif
    }
#ifndef ENABLE_BETA_IN_TSM
    #pragma unroll
    for (int i = 0; i < HV_PER_BLOCK; i++) {
        b_beta_all[i] = __shfl_sync(0xFFFFFFFF, b_beta, i);
        b_g_exp_all[i] = __shfl_sync(0xFFFFFFFF, b_g_exp, i);
    }
#endif

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

    #ifdef GINO_DEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            // printf("tid=%d, b_q=%f, b_k=%f\n", tid, float(b_q_val), float(b_k_val));
        }
    #endif


    // Process each group (matching loop in Triton)
    // #pragma unroll
    for (int gid = 0; gid < HV_PER_BLOCK; gid++) {
        const int i_hv = blockIdx.y * HV_PER_BLOCK + gid;
#ifdef PLA_USE_AIU
        coord_n = gid * GRP_ITER_NUM; //i_hv;
#endif

        // Process V in tiles of BV=32
        int v_base_idx = (bos * HV + i_hv) * V;
        // prefetch for V
        #ifdef __HGGC_ARCH__
        if (warp_id == 0) {
            __ppu_vmem_prefetch_bulk_b32x2(&v[v_base_idx], 1);
        }
        #endif
        #pragma unroll
        for (int v_base = 0; v_base < V; v_base += BV, v_base_idx += BV) {

            // for H0
            constexpr int PREF_BV_DIST = BV * STAGE_NUM;
            const int h0_offset_v_base = i_hv * K * V + v_base;
            int h0_offset_v_base_pref = h0_offset_v_base + PREF_BV_DIST;
#ifdef PLA_USE_AIU
            int v_base_next = v_base + PREF_BV_DIST;
            if (v_base_next >= V) {

                    v_base_next -= V;
                    coord_n = (gid + 1) * GRP_ITER_NUM; //i_hv + 1;
            }
#else
            bool pref_pred = true;
            if ((v_base + PREF_BV_DIST) >= V) {
                if ((gid + 1) < HV_PER_BLOCK) {
                    h0_offset_v_base_pref = h0_offset_v_base_pref - V + K * V;
                } else {
                    pref_pred = false;
                }
            }
#endif

            float4 h0_f4[GRP_ITER_NUM];
            float4 h_mul_k_per_warp = {.0f, .0f, .0f, .0f};

            // b_v -= tl.sum(b_h * b_k[:, None], 0)
            // loop in K dimension, each loop handles 32 rows in h0
#ifdef PLA_USE_AIU
            cp_async_wait<STAGE_NUM - 1>();
            // __syncthreads();

            // first load data from shared memory
            #pragma unroll
            for (int grp_idx = 0; grp_idx < GRP_ITER_NUM; grp_idx++) {
                const int row_start_per_warp = warp_id * TOTAL_ROW_PER_WARP + grp_idx * ROW_NUM_PER_WARP_PER_GRP; // Each warp handles 4 rows
                const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading (4 rows per warp)
                if constexpr (STAGE_NUM > 1) {
                    h0_f4[grp_idx] = *reinterpret_cast<float4*>(shared_h0 + cur_stage_seq * K * BV + row_idx * BV + col_idx);
                } else {
                    h0_f4[grp_idx] = *reinterpret_cast<float4*>(shared_h0 + row_idx * BV + col_idx);
                }

#ifdef GINO_DEBUG
                // if (blockIdx.x == 0 && (blockIdx.y == 0) && blockIdx.z == 0 && threadIdx.x == 0) {
                //     printf("blockIdx.y=%d, load from tsm, i_hv=%d, gid=%d, grp_idx=%d, cur_stage_seq=%d\n",
                //             blockIdx.y, i_hv, gid, grp_idx, cur_stage_seq);
                // }
#endif
            }
#ifdef __HGGC_ARCH__
            __ppu_wait_tsmcnt(0);
#endif

            // third issue aiu load
            coord_c = v_base_next;
            if constexpr (STAGE_NUM > 1) {
                ppu_aiu_load<CUBE_C, CUBE_W, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w, tensor_stride_n>
                            (shared_h0 + ((cur_stage_seq + STAGE_NUM) % STAGE_NUM) * K * BV + shm_offset_per_warp,
                             p_h0 + h0_offset_v_head_base,
                             coord_c, coord_w, coord_n);
            } else {
                ppu_aiu_load<CUBE_C, CUBE_W, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w, tensor_stride_n>
                            (shared_h0 + shm_offset_per_warp,
                             p_h0 + h0_offset_v_head_base,
                             coord_c, coord_w, coord_n);
            }
            cp_async_commit();

            // second use the fresh h0 data
            #pragma unroll
            for (int grp_idx = 0; grp_idx < GRP_ITER_NUM; grp_idx++) {
                const int row_start_per_warp = grp_idx * GRP_ROW_NUM + (warp_id * ROW_NUM_PER_WARP_PER_GRP); // Each warp handles 4 rows
                const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading (4 rows per warp)

#ifndef ENABLE_BETA_IN_TSM
                b_g_exp = b_g_exp_all[gid];
#else
                b_g_exp = shared_g_exp[gid];
#endif
                // float b_g_exp = shared_g_exp[gid];
                h0_f4[grp_idx] = {h0_f4[grp_idx].x * float(b_g_exp), h0_f4[grp_idx].y * float(b_g_exp), h0_f4[grp_idx].z * float(b_g_exp), h0_f4[grp_idx].w * float(b_g_exp)};

                // PHASE 2: Compute b_v -= tl.sum(b_h * b_k[:, None], 0)
                // Each loop computes 32 rows and 16 columns of output
                // Need 4x2 such loops to cover all K dimensions
                int vreg_lane_id = map_to_lane_id(row_idx);
                float k_per_thread = __shfl_sync(0xFFFFFFFF, b_k_val, vreg_lane_id);
                h_mul_k_per_warp.x += h0_f4[grp_idx].x * k_per_thread,
                h_mul_k_per_warp.y += h0_f4[grp_idx].y * k_per_thread,
                h_mul_k_per_warp.z += h0_f4[grp_idx].z * k_per_thread,
                h_mul_k_per_warp.w += h0_f4[grp_idx].w * k_per_thread;
            }

#else
            cp_async_wait<STAGE_NUM - 1>();

            // #pragma hggc dislicm
            {
                #pragma unroll
                for (int grp_idx = 0; grp_idx < GRP_ITER_NUM; grp_idx++) {
                // PHASE 1: Load K and h0 into shared memory with vector loads
                // Each warp loads 4 rows of 32 values (128 values total per warp)
                // 4 warps read 16 rows

                    const int row_start_per_warp = grp_idx * GRP_ROW_NUM + (warp_id * ROW_NUM_PER_WARP_PER_GRP); // Each warp handles 4 rows
                    const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading (4 rows per warp)

                    if constexpr (STAGE_NUM > 1) {
                        h0_f4[grp_idx] = *reinterpret_cast<float4*>(shared_h0 + cur_stage_seq * K * BV + row_idx * BV + col_idx);
#ifdef __HGGC_ARCH__
                        __ppu_wait_tsmcnt(0);
#endif
                        async_cp_h0<GRP_ROW_NUM, ROW_NUM_PER_WARP_PER_GRP, V>(p_h0, shared_h0 + ((cur_stage_seq + STAGE_NUM) % STAGE_NUM) * K * BV, h0_offset_v_base_pref, row_idx, col_idx, pref_pred);
                    } else {
                        h0_f4[grp_idx] = *reinterpret_cast<float4*>(shared_h0 + row_idx * BV + col_idx);
#ifdef __HGGC_ARCH__
                        __ppu_wait_tsmcnt(0);
#endif
                        async_cp_h0<GRP_ROW_NUM, ROW_NUM_PER_WARP_PER_GRP, V>(p_h0, shared_h0, h0_offset_v_base_pref, row_idx, col_idx, pref_pred);
                    }

#ifndef ENABLE_BETA_IN_TSM
                    b_g_exp = b_g_exp_all[gid];
#else
                    b_g_exp = shared_g_exp[gid];
#endif
                    // float b_g_exp = shared_g_exp[gid];
                    h0_f4[grp_idx] = {h0_f4[grp_idx].x * float(b_g_exp), h0_f4[grp_idx].y * float(b_g_exp), h0_f4[grp_idx].z * float(b_g_exp), h0_f4[grp_idx].w * float(b_g_exp)};

                    // PHASE 2: Compute b_v -= tl.sum(b_h * b_k[:, None], 0)
                    // Each loop computes 32 rows and 16 columns of output
                    // Need 4x2 such loops to cover all K dimensions
                    int vreg_lane_id = map_to_lane_id(row_idx);
                    float k_per_thread = __shfl_sync(0xFFFFFFFF, b_k_val, vreg_lane_id);
                    h_mul_k_per_warp.x += h0_f4[grp_idx].x * k_per_thread,
                    h_mul_k_per_warp.y += h0_f4[grp_idx].y * k_per_thread,
                    h_mul_k_per_warp.z += h0_f4[grp_idx].z * k_per_thread,
                    h_mul_k_per_warp.w += h0_f4[grp_idx].w * k_per_thread;
                } // group loop
            }

            cp_async_commit();
#endif

            if constexpr (STAGE_NUM > 1) {
                cur_stage_seq = (cur_stage_seq + 1) % STAGE_NUM;
            }

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
                // #pragma unroll
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
#ifndef ENABLE_BETA_IN_TSM
                float b_beta = b_beta_all[gid];
#else
                float b_beta = shared_beta[gid];
#endif
                // float b_beta = shared_beta[gid];
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

                const int row_start_per_warp = grp_idx * GRP_ROW_NUM + (warp_id * ROW_NUM_PER_WARP_PER_GRP); // Each warp handles 4 rows
                const int row_idx = row_start_per_warp + row_idx_in_warp; // Which row this thread is loading (4 rows per warp)
                const int h0_offset = h0_offset_v_base + row_idx * V + col_idx;

                int vreg_lane_id = map_to_lane_id(row_idx);
                float k_per_thread = __shfl_sync(0xFFFFFFFF, b_k_val, vreg_lane_id);
                h0_f4[grp_idx].x += h_mul_k_per_warp.x * k_per_thread;
                h0_f4[grp_idx].y += h_mul_k_per_warp.y * k_per_thread;
                h0_f4[grp_idx].z += h_mul_k_per_warp.z * k_per_thread;
                h0_f4[grp_idx].w += h_mul_k_per_warp.w * k_per_thread;

                float q_per_thread = __shfl_sync(0xFFFFFFFF, b_q_val, vreg_lane_id);
                out_f4.x += h0_f4[grp_idx].x * q_per_thread;
                out_f4.y += h0_f4[grp_idx].y * q_per_thread;
                out_f4.z += h0_f4[grp_idx].z * q_per_thread;
                out_f4.w += h0_f4[grp_idx].w * q_per_thread;

#ifdef __HGGC_ARCH__
                *((Global v4float*)&p_h0[h0_offset]) = *(v4float*)&h0_f4[grp_idx];
                // if (gid == 1) {
                //     *((Global v4float*)&p_h0[h0_offset]) = *(v4float*)&h0_f4[grp_idx];
                // } else {
                //     __stwt((float4*)(&p_h0[h0_offset]), h0_f4[grp_idx]);
                // }
#else
                __stcg((float4*)(&p_h0[h0_offset]), h0_f4[grp_idx]);
#endif
            }

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
#ifdef __HGGC_ARCH__
            __ppu_barrier_sync_defer(0, WARP_NUM*32, 1);
#else
            __syncthreads();
#endif

            if (tid < (BV / H0_VEC_LEN)) {
                float4 sum_lane8 = *reinterpret_cast<float4*>(&shared_out[lane_id * H0_VEC_LEN]);
                // #pragma unroll
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
        } // for (int v_base = 0; v_base < V; v_base += BV)
    } // gid

    // cp_async_wait<0>();
}

template<typename alog_t, typename scalar_t, int STAGE_NUM, int K, int V, int H, int HV, int WARP_NUM>
static void dispatch_hv_per_cta(
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
    const bool use_initial_state,
    const bool use_qk_l2norm_in_kernel,
    const bool is_varlen,
    const hggcStream_t stream
) {

    // Kernel configuration: 4 warps per block (128 threads)
    const int GROUP_SIZE = HV / H;
    const int blocks_per_grid_k = 1;
    const int blocks_per_grid_n = B;
    const int hv_per_cta = choose_hv_per_cta(B, H, HV);
    const int blocks_per_grid_h = (GROUP_SIZE / hv_per_cta) * H;
    const int threads_per_block = WARP_NUM * 32;
    dim3 block(threads_per_block);
    dim3 grid(blocks_per_grid_k, blocks_per_grid_h, blocks_per_grid_n);

    // Shared memory calculation
    // We need: K (for k) + K (for q) + BV (for new_v & output) + K*BV (for h0) + 32 (for reduction)
    int shared_mem_size = (STAGE_NUM * K * BV + WARP_NUM * BV * 2 + WARP_NUM*2) * sizeof(float);
#ifdef ENABLE_BETA_IN_TSM
    shared_mem_size += 2 * hv_per_cta * sizeof(float);
#endif

#if defined (__HGGCCC__)
    #define KERNEL_LAUNCH(hv_per_cta_v) \
        if ((GROUP_SIZE % hv_per_cta_v) == 0 && hv_per_cta == hv_per_cta_v) { \
            hggcFuncSetAttribute( \
                fused_sigmoid_gating_delta_rule_cuda_kernel< \
                    alog_t, scalar_t, STAGE_NUM, K, V, H, HV, WARP_NUM, hv_per_cta_v>, \
                PLA_FUNC_ATTR_BLOCK_AGE_EN, \
                1); \
            fused_sigmoid_gating_delta_rule_cuda_kernel<alog_t, scalar_t, STAGE_NUM, K, V, H, HV, WARP_NUM, hv_per_cta_v><<<grid, block, shared_mem_size, stream>>>( \
                A_log,  \
                a,  \
                dt_bias, \
                softplus_beta,  \
                softplus_threshold, \
                q,  \
                k,  \
                v,  \
                b,  \
                o,  \
                h0_source, \
                h0_indices, \
                cu_seqlens, \
                scale, \
                T, B, \
                use_initial_state, \
                use_qk_l2norm_in_kernel, \
                is_varlen \
            ); \
            return; \
        }
#else
    #define KERNEL_LAUNCH(hv_per_cta_v) \
    if ((GROUP_SIZE % hv_per_cta_v) == 0 && hv_per_cta == hv_per_cta_v) { \
        fused_sigmoid_gating_delta_rule_cuda_kernel<alog_t, scalar_t, STAGE_NUM, K, V, H, HV, WARP_NUM, hv_per_cta_v><<<grid, block, shared_mem_size, stream>>>( \
            A_log,  \
            a,  \
            dt_bias, \
            softplus_beta,  \
            softplus_threshold, \
            q,  \
            k,  \
            v,  \
            b,  \
            o,  \
            h0_source, \
            h0_indices, \
            cu_seqlens, \
            scale, \
            T, B, \
            use_initial_state, \
            use_qk_l2norm_in_kernel, \
            is_varlen \
        ); \
        return; \
    }
#endif

    KERNEL_LAUNCH(4);
    KERNEL_LAUNCH(3);
    KERNEL_LAUNCH(2);
    KERNEL_LAUNCH(1);
    throw std::runtime_error(
        "pla.decode.gdn.v_last bigbatch has no launch for H=" + std::to_string(H) +
        ", HV=" + std::to_string(HV) +
        ", hv_per_cta=" + std::to_string(hv_per_cta));

#if 0
    auto kernel = fused_sigmoid_gating_delta_rule_cuda_kernel<scalar_t, STAGE_NUM, K, V, H, 8, WARP_NUM>;
    const void *ptrfunc = reinterpret_cast<const void*>(kernel);
    HGfunction cu_func = static_cast<HGfunction>(NULL);
    hggcGetFuncBySymbol(reinterpret_cast<hggcFunction_t*>(&cu_func), ptrfunc);

    // int H_val = H;
    void* kernel_params[] = {
        (void*)&A_log,
        (void*)&a,
        (void*)&dt_bias,
        (void*)&softplus_beta,
        (void*)&softplus_threshold,
        (void*)&q,
        (void*)&k,
        (void*)&v,
        (void*)&b,
        (void*)&o,
        (void*)&h0_source,
        (void*)&h0_indices,
        (void*)&cu_seqlens,
        (void*)&scale,
        (void*)&T,
        (void*)&B,
        // (void*)&H_val,
        // (void*)&HV,
        (void*)&use_initial_state,
        (void*)&use_qk_l2norm_in_kernel,
        (void*)&is_varlen
    };
    HGlaunchConfigAD launchCfg;
    HGlaunchAttributeAD launchAttr = {CUAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE};
    launchAttr.value.schedPreference.blocksPerMultiprocessor = 8;
    launchAttr.value.schedPreference.gridStepX = 1;
    launchAttr.value.schedPreference.gridStepY = 1;
    // enable block age
    launchAttr.value.schedPreference.flags = CUAD_SCHED_PREFER_FIFS;

    launchCfg = {grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_mem_size, stream, &launchAttr, 1};

    CHECK_CU(hgLaunchKernelExAD(&launchCfg, cu_func, kernel_params, nullptr));
    return;
#endif
}

template<typename alog_t, typename scalar_t, int STAGE_NUM, int K, int V, int H>
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
    const int WARP_NUM = 4; //(K / 32); // / 2;
#define DISPATCH_V_HEAD(HV_V) \
    if (HV == HV_V) { \
        dispatch_hv_per_cta<alog_t, scalar_t, STAGE_NUM, K, V, H, HV_V, WARP_NUM>( \
            A_log,  \
            a,  \
            dt_bias, \
            softplus_beta,  \
            softplus_threshold, \
            q,  \
            k,  \
            v,  \
            b,  \
            o,  \
            h0_source, \
            h0_indices, \
            cu_seqlens, \
            scale, \
            T, B, \
            use_initial_state, \
            use_qk_l2norm_in_kernel, \
            is_varlen, \
            stream \
        ); \
        return; \
    }

    DISPATCH_V_HEAD(128);
    DISPATCH_V_HEAD(64);
    DISPATCH_V_HEAD(48);
    DISPATCH_V_HEAD(32);
    DISPATCH_V_HEAD(24);
    DISPATCH_V_HEAD(16);
    DISPATCH_V_HEAD(12);
    DISPATCH_V_HEAD(8);
    DISPATCH_V_HEAD(6);
    DISPATCH_V_HEAD(4);
    DISPATCH_V_HEAD(3);
    DISPATCH_V_HEAD(2);
    throw std::runtime_error(std::string("pla.decode.gdn.v_last bigbatch does not support HV=") + std::to_string(HV));
}

// DISPATCH_V_HEAD is a local helper of dispatch_V_Head_Num above; the H-level
// cascade lives in fused_sigmoid_gating_v_last.cpp and must not inherit it.
#undef DISPATCH_V_HEAD

// Entry point instantiated once per H by the per-H translation units.
template<int H>
static void bigbatch_launch_for_H(PLA_V_LAST_ENTRY_PARAMS) {
    constexpr int STAGE_NUM = 1;
    PLA_V_LAST_DISPATCH_ALOG(args.alog_dtype, ([&] {
        PLA_V_LAST_DISPATCH_SCALAR(args.scalar_dtype, ([&] {
            if (args.Kdim == 128 && args.Vdim == 128) {
                dispatch_V_Head_Num<alog_t, scalar_t, STAGE_NUM, 128, 128, H>(
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
