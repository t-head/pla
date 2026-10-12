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
// Device-side helpers shared by the k_last kernel family.
//
// Split out of common.cuh so that device translation units never see torch: the
// torch headers pull in the CUDA compatibility header tree, and a TU that mixes
// it with the PPU-native tree loses its execution-space qualifiers (the visible
// symptom is "hggc_fp16.h: typedef redefinition ('struct ::__half_raw' vs
// 'struct ::__half_raw')" followed by __cvta_generic_to_shared going missing).
// common.cuh keeps the host-only half: the NVTX/HGTX wrapper, the environment
// flags and op_forge_case.h (which includes <ATen/ATen.h>).
//
// TypeConverter is specialised on the native scalar types rather than
// at::Half / at::BFloat16; the pairs are bit-compatible, and the callers only
// ever reinterpret 32 bits of packed data.

#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>

// =========================================================================
// Shared device helpers for k_last kernel family.
// Files that define their own variants (bigbatch, smallbatch) should
// #define PLA_CUSTOM_DEVICE_HELPERS before including this header.
// Only compiled in CUDA translation units.
// =========================================================================
#if defined(__HGGCCC__) && !defined(PLA_CUSTOM_DEVICE_HELPERS)

#define ENABLE_BETA_IN_TSM
// AIU is available on both PPU generations: ppu0015 (__HGGC_ARCH__ == 150) via
// the 2d.tile form and ppu001 (__HGGC_ARCH__ == 100) via the 3d.linear form.
// ppu_aiu_load below emits the right encoding for each, so the gate covers both.
#if defined (__HGGC_ARCH__) && (__HGGC_ARCH__ == 150 || __HGGC_ARCH__ == 100)
    #define PLA_USE_AIU
#endif

#ifdef __HGGC_ARCH__
#define Global __attribute__((address_space(1)))
typedef float v4float __attribute__((ext_vector_type(4)));
typedef unsigned int v2uint __attribute__((ext_vector_type(2)));
typedef unsigned int v4uint __attribute__((ext_vector_type(4)));
#endif

template<typename T> struct TypeConverter;
template<> struct TypeConverter<__half> {
    static __device__ __forceinline__ float2 convert(uint32_t val) {
        return __half22float2(*reinterpret_cast<const __half2*>(&val));
    }
};
template<> struct TypeConverter<__ppu_bfloat16> {
    static __device__ __forceinline__ float2 convert(uint32_t val) {
        return __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(&val));
    }
};

__device__ __forceinline__ uint32_t cvta_sm_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <int CUBE_C, int CUBE_W, int CUBE_N, int DIM_C, int DIM_W, int DIM_N, int64_t TENSOR_STRIDE_W, int64_t TENSOR_STRIDE_N, bool b32=true, int swzl_bytes=0>
__device__ __forceinline__ void ppu_aiu_load(void *smem_ptr, const void* gmem_ptr, int coord_c, int coord_w, int coord_n) {
    uint32_t smem_int_ptr = cvta_sm_addr(smem_ptr);
#if defined (__HGGC_ARCH__) && __HGGC_ARCH__ == 150
    if constexpr (swzl_bytes == 0) {
        if constexpr (b32) {
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
        } else {
            asm volatile(
                "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
                "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
                :: "r"(smem_int_ptr), "l"(gmem_ptr),
                  "r"(DIM_C), "r"(DIM_W), "r"(DIM_N),
                  "r"(CUBE_C), "r"(CUBE_W), "r"(CUBE_N),
                  "l"(TENSOR_STRIDE_W), "l"(TENSOR_STRIDE_N),
                  "r"(coord_c), "r"(coord_w), "r"(coord_n),
                  "r"(0)
            );
        }
    } else {
        if constexpr (swzl_bytes == 128) {
            asm volatile(
                "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl.b32"
                "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
                :: "r"(smem_int_ptr), "l"(gmem_ptr),
                  "r"(DIM_C), "r"(DIM_W), "r"(DIM_N),
                  "r"(CUBE_C), "r"(CUBE_W), "r"(CUBE_N),
                  "l"(TENSOR_STRIDE_W), "l"(TENSOR_STRIDE_N),
                  "r"(coord_c), "r"(coord_w), "r"(coord_n),
                  "r"(0)
            );
        } else {
            asm volatile(
                "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.swzl.b32"
                "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
                :: "r"(smem_int_ptr), "l"(gmem_ptr),
                  "r"(DIM_C), "r"(DIM_W), "r"(DIM_N),
                  "r"(CUBE_C), "r"(CUBE_W), "r"(CUBE_N),
                  "l"(TENSOR_STRIDE_W), "l"(TENSOR_STRIDE_N),
                  "r"(coord_c), "r"(coord_w), "r"(coord_n),
                  "r"(1)
            );
        }
    }
#elif defined (__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    // ppu001's 3d.linear form carries no stride operands: the source is
    // addressed as a dense [DIM_N][DIM_W][DIM_C] cube, so a caller passing a
    // strided view would be silently mis-copied. Turn that into a build error.
    constexpr int64_t ELEM_BYTES = b32 ? 4 : 2;
    static_assert(swzl_bytes == 0,
                  "ppu001 AIU only implements the linear (non-swizzled) mode");
    static_assert(TENSOR_STRIDE_W == (int64_t)DIM_C * ELEM_BYTES,
                  "ppu001 AIU needs a contiguous w axis: TENSOR_STRIDE_W must be DIM_C * sizeof(elem)");
    // TENSOR_STRIDE_N is only consulted when the copy actually walks the outer
    // axis; single-head callers leave it at an unrelated value.
    static_assert(CUBE_N == 1 || TENSOR_STRIDE_N == (int64_t)DIM_W * DIM_C * ELEM_BYTES,
                  "ppu001 AIU needs a contiguous n axis when CUBE_N > 1");
    if (coord_n < DIM_N) {
        if constexpr (b32) {
            asm volatile(
                "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.3d.b32 [%0], [%1], "
                "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
                :: "r"(smem_int_ptr), "l"(gmem_ptr),
                  "r"(DIM_N), "r"(DIM_W), "r"(DIM_C),
                  "r"(coord_n), "r"(coord_w), "r"(coord_c),
                  "r"(CUBE_N), "r"(CUBE_W), "r"(CUBE_C), "r"(1)
            );
        } else {
            asm volatile(
                "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.3d.b16 [%0], [%1], "
                "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
                :: "r"(smem_int_ptr), "l"(gmem_ptr),
                  "r"(DIM_N), "r"(DIM_W), "r"(DIM_C),
                  "r"(coord_n), "r"(coord_w), "r"(coord_c),
                  "r"(CUBE_N), "r"(CUBE_W), "r"(CUBE_C), "r"(1)
            );
        }
    }
#endif

        #ifdef GINO_DEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
                float4* pVal = reinterpret_cast<float4*>((char*)gmem_ptr + coord_n * TENSOR_STRIDE_N + coord_w * TENSOR_STRIDE_W);
                printf("blockIdx.xyz=%d,%d,%d, coord_w=%d, coord_n=%d, smem_ptr=%p, gmem_ptr=%p \
                        f_val=%f,%f,%f,%f\n",
                        blockIdx.x, blockIdx.y, blockIdx.z, coord_w, coord_n, smem_ptr, pVal, pVal->x, pVal->y, pVal->z, pVal->w);
            }
        #endif
};

template <int STAGES>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr (STAGES <= 0) {
        asm volatile("ppu.cp.async.wait_all;\n");
    } else if constexpr (STAGES == 1) {
        asm volatile("ppu.cp.async.wait_group 1;\n");
    } else if constexpr (STAGES == 2) {
        asm volatile("ppu.cp.async.wait_group 2;\n");
    } else if constexpr (STAGES == 3) {
        asm volatile("ppu.cp.async.wait_group 3;\n");
    } else if constexpr (STAGES == 4) {
        asm volatile("ppu.cp.async.wait_group 4;\n");
    } else if constexpr (STAGES == 5) {
        asm volatile("ppu.cp.async.wait_group 5;\n");
    } else if constexpr (STAGES == 6) {
        asm volatile("ppu.cp.async.wait_group 6;\n");
    } else if constexpr (STAGES == 7) {
        asm volatile("ppu.cp.async.wait_group 7;\n");
    } else {
        asm volatile("ppu.cp.async.wait_all;\n");
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("ppu.cp.async.commit_group;\n");
}

__device__ __forceinline__ float compute_bg(float a,
    float dt,
    float b_A_log,
    float softplus_beta,
    float softplus_threshold) {
    float x = a + dt;
    float beta_x = softplus_beta * x;
    float softplus_x = (beta_x <= softplus_threshold) ?
                       (1.0f / softplus_beta) * __logf(1.0f + __expf(beta_x)) : x;
    return __expf(-__expf(b_A_log) * softplus_x);
}

template <bool USE_LOWER_BOUND>
__device__ __forceinline__ float compute_bg(float a,
    float dt,
    float b_A_log,
    float softplus_beta_or_lower_bound,
    float softplus_threshold) {
    if constexpr (USE_LOWER_BOUND) {
        float x = a + dt;
        float exp_A = __expf(b_A_log);
        // Safe-gate definition:
        // https://github.com/fla-org/flash-linear-attention/pull/701/changes#diff-18dc009e5fff3ccb85310e8bb8e8e9398ee7967e3ebd1d26771339601e22f458R112
#if defined(__HGGC_ARCH__)
        float sigmoid_x = __ppu_sgmdf(exp_A * x);
#else
        float sigmoid_x = 1.0f / (1.0f + __expf(-exp_A * x));
#endif
        return __expf(softplus_beta_or_lower_bound * sigmoid_x);
    }
    return compute_bg(a, dt, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
}

__device__ __forceinline__ static int divide_by_power_of_two(int divided, int divider) {
#ifdef __HGGC_ARCH__
    return (divided >> __builtin_ctz(divider));
#else
    return divided / divider;
#endif
}

template<typename scalar_t, int low = 1, int high = 16>
__device__ __forceinline__ float warp_reduce_sum(scalar_t f_val) {
    #pragma unroll
    for (int offset = high; offset >= low; offset /= 2) {
        f_val += __shfl_xor_sync(0xffffffff, float(f_val), offset);
    }
    return f_val;
}

template<int WARP_NUM, typename scalar_t>
__device__ void block_reduce_sum2(scalar_t val1, scalar_t val2, float* shared_mem1, float* shared_mem2, int lane_id, int warp_id, float* out1, float* out2) {
    float f_val1 = static_cast<float>(val1);
    float f_val2 = static_cast<float>(val2);

    f_val1 = warp_reduce_sum(f_val1);
    f_val2 = warp_reduce_sum(f_val2);

    if (lane_id == 0) {
        shared_mem1[warp_id] = f_val1;
        shared_mem2[warp_id] = f_val2;
    }

    __syncthreads();

    if (warp_id == 0) {
        float result1 = lane_id < WARP_NUM ? shared_mem1[lane_id] : 0.0f;
        float result2 = lane_id < WARP_NUM ? shared_mem2[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WARP_NUM / 2; offset > 0; offset /= 2) {
            result1 += __shfl_down_sync(0xffffffff, result1, offset);
            result2 += __shfl_down_sync(0xffffffff, result2, offset);
        }
        if (lane_id == 0) {
            shared_mem1[0] = result1;
            shared_mem2[0] = result2;
        }
    }

    __syncthreads();

    *out1 = shared_mem1[0];
    *out2 = shared_mem2[0];
}

template<int WARP_NUM, typename scalar_t>
__device__ float block_reduce_sum(scalar_t val, float* shared_mem, int lane_id, int warp_id) {
    float f_val = static_cast<float>(val);

    f_val = warp_reduce_sum(f_val);

    if (lane_id == 0) {
        shared_mem[warp_id] = f_val;
    }

    __syncthreads();

    if (warp_id == 0) {
        float result = lane_id < WARP_NUM ? shared_mem[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = WARP_NUM / 2; offset > 0; offset /= 2) {
            result += __shfl_down_sync(0xffffffff, result, offset);
        }
        if (lane_id == 0) {
            shared_mem[0] = result;
        }
    }

    __syncthreads();

    return shared_mem[0];
}

template<typename scalar_t>
__device__ float block_reduce_sum<1, scalar_t>(scalar_t val, float* shared_mem, int lane_id, int warp_id) {
    float f_val = static_cast<float>(val);
    f_val = warp_reduce_sum(f_val);
    return f_val;
}

__device__ __forceinline__ static float divide_float(float a, float b) {
  #ifdef __HGGC_ARCH__
    return __fdividef(a, b);
  #else
    return (a / b);
  #endif
}

#endif // defined(__HGGCCC__) && !defined(PLA_CUSTOM_DEVICE_HELPERS)
