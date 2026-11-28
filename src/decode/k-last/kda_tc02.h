// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2026 MoonshotAI
// SPDX-License-Identifier: BSD-3-Clause AND MIT
// AIU state transport and half-warp row ownership adapted from FlashKDA.

#pragma once

#include <cstdint>
#include <hggc_runtime.h>
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
#include <hggc_bf16.h>
#endif

namespace pla {

enum class KdaDecodeTc02Plan { Fallback, Small, Rolling, BulkSplit, BulkWhole, SmallBulk };

// Tensor-independent ABI; validation and optional tensor handling stay in the
// existing Torch wrapper.
struct KdaDecodeTc02Args {
    const void *q, *k, *v, *gate, *beta, *dt_bias;
    const float* a_log;
    void *state, *out;
    const int32_t *cu_seqlens, *state_indices;
    int64_t state_stride;
    int q_stride, k_stride, v_stride, gate_stride, beta_stride;
    int batch, heads, state_slots, tokens;
    float scale, softplus_beta_or_lower_bound, softplus_threshold;
    bool use_lower_bound, use_qk_l2norm, is_sglang, bf16_state;
};

int kda_tc02_cu_count(int device);

// Select a compiled transport/row layout from device capacity and logical work.
KdaDecodeTc02Plan kda_decode_tc02_plan(int device, int batch, int heads, bool bf16_state,
                                     int fallback_rows);
void launch_kda_decode_tc02(const KdaDecodeTc02Args& args, KdaDecodeTc02Plan plan, hggcStream_t stream);

// The dense chain owns a private FP32 state; storage dtype applies only to
// initial reads and checkpoint writes. Pool pitches remain caller supplied.
struct KdaVerifyTc02Args {
    const void *q, *k, *v, *gate, *beta;
    const float *a_log, *dt_bias;
    const void *state;
    void *cache, *out;
    const int32_t *cu_seqlens, *state_indices, *cache_indices;
    int64_t state_stride, cache_stride, cache_step_stride;
    int q_stride, k_stride, v_stride, gate_stride;
    int batch, heads, max_steps, tokens, state_slots, cache_slots;
    float scale, lower_bound;
    bool bf16_state;
};

// Zero keeps the established PLA rolling path.
int kda_verify_tc02_splits(int device, int batch, int heads, bool bf16_state);
void launch_kda_verify_tc02(const KdaVerifyTc02Args& args, int splits, hggcStream_t stream);

namespace kda_tc02 {

// Shared vector transport and reductions; request/state policies stay in each kernel.
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
using BF16 = __ppu_bfloat16;
typedef float v4float __attribute__((ext_vector_type(4)));
#define PLA_KDA_TC02_GLOBAL __attribute__((address_space(1)))

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t raw) {
    return __bfloat1622float2(*reinterpret_cast<__ppu_bfloat162 const *>(&raw));
}

__device__ __forceinline__ float4 load_bf16x4(BF16 const *ptr) {
    uint2 raw = *reinterpret_cast<uint2 const *>(ptr);
    float2 lo = bf16x2_to_float2(raw.x);
    float2 hi = bf16x2_to_float2(raw.y);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ void store_float4_global(float *ptr, float4 value) {
#ifdef __HGGC_ARCH__
    *(PLA_KDA_TC02_GLOBAL v4float *)(ptr) = *(v4float *)&value;
#else
    *reinterpret_cast<float4 *>(ptr) = value;
#endif
}

template <typename StateT> __device__ __forceinline__ float4 load_state4(StateT const *ptr) {
    if constexpr (sizeof(StateT) == sizeof(float)) {
        return *reinterpret_cast<float4 const *>(ptr);
    } else {
        static_assert(sizeof(StateT) == sizeof(BF16));
        return load_bf16x4(reinterpret_cast<BF16 const *>(ptr));
    }
}

template <typename StateT> __device__ __forceinline__ void store_state4_global(StateT *ptr, float4 value) {
    if constexpr (sizeof(StateT) == sizeof(float)) {
        store_float4_global(reinterpret_cast<float *>(ptr), value);
    } else {
        static_assert(sizeof(StateT) == sizeof(BF16));
        __ppu_bfloat162 lo = __floats2bfloat162_rn(value.x, value.y);
        __ppu_bfloat162 hi = __floats2bfloat162_rn(value.z, value.w);
        uint2 raw =
            make_uint2(*reinterpret_cast<uint32_t *>(&lo), *reinterpret_cast<uint32_t *>(&hi));
        *reinterpret_cast<uint2 *>(ptr) = raw;
    }
}

__device__ __forceinline__ float warp_sum_all(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset);
    }
    return x;
}

__device__ __forceinline__ float warp_sum_lane0(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

__device__ __forceinline__ void half_warp_sum_all_pair(float &x0, float &x1) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        x0 += __shfl_xor_sync(0xffffffff, x0, offset, 16);
        x1 += __shfl_xor_sync(0xffffffff, x1, offset, 16);
    }
}

__device__ __forceinline__ void half_warp_sum_lane0_pair(float &x0, float &x1) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        x0 += __shfl_down_sync(0xffffffff, x0, offset, 16);
        x1 += __shfl_down_sync(0xffffffff, x1, offset, 16);
    }
}

#undef PLA_KDA_TC02_GLOBAL
#endif

} // namespace kda_tc02

} // namespace pla
