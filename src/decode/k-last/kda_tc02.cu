// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2026 MoonshotAI
// SPDX-License-Identifier: BSD-3-Clause AND MIT
// AIU state transport and half-warp row ownership adapted from FlashKDA.

#include <array>
#include <atomic>
#include <stdexcept>
#include <type_traits>

#include <hggc_bf16.h>
#include "kda_tc02.h"

namespace pla {
namespace {

using namespace kda_tc02;

using BF16 = __ppu_bfloat16;
constexpr int kD = 128, kStages = 4, kStateElems = kD*kD;
constexpr int kTallRowSplits = 4;

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
// Cache-global vector store: keep 16 contiguous bytes per lane while bypassing
// the per-row warp-bulk path and its extra ppu0015 store transaction.
__device__ __forceinline__ void store_float4_global_cg(float *ptr, float4 value) {
#ifdef __HGGC_ARCH__
    __stcg(reinterpret_cast<float4 *>(ptr), value);
#else
    *reinterpret_cast<float4 *>(ptr) = value;
#endif
}

template <bool CacheGlobal = false>
__device__ __forceinline__ void store_bf16_state8_global(BF16 *ptr, float4 lo, float4 hi) {
    __ppu_bfloat162 x0 = __floats2bfloat162_rn(lo.x, lo.y);
    __ppu_bfloat162 x1 = __floats2bfloat162_rn(lo.z, lo.w);
    __ppu_bfloat162 x2 = __floats2bfloat162_rn(hi.x, hi.y);
    __ppu_bfloat162 x3 = __floats2bfloat162_rn(hi.z, hi.w);
    uint4 raw = make_uint4(*reinterpret_cast<uint32_t *>(&x0), *reinterpret_cast<uint32_t *>(&x1),
                           *reinterpret_cast<uint32_t *>(&x2), *reinterpret_cast<uint32_t *>(&x3));
    if constexpr (CacheGlobal) __stcg(reinterpret_cast<uint4 *>(ptr), raw);
    else *reinterpret_cast<uint4 *>(ptr) = raw;
}

// ppu0015 AIU: load 32 contiguous fp32 state rows.  The height dimension moves
// 16 KiB with one warp-level descriptor instead of 1024 lane-level requests.
template <typename StateT, int Rows, int StatePitch = kD, int SourceRows = kD>
__device__ __forceinline__ void aiu_load_rows(StateT *smem, StateT const *gmem, int row_offset = 0) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    constexpr int kDimC = kD;
    constexpr int kDimW = SourceRows;
    constexpr int kDimN = 1;
    constexpr int kCubeC = StatePitch;
    constexpr int kCubeW = Rows;
    constexpr int kCubeN = 1;
    constexpr int64_t kStrideW = kD * sizeof(StateT);
    constexpr int64_t kStrideN = SourceRows * kD * sizeof(StateT);
    if constexpr (sizeof(StateT) == sizeof(float)) {
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b32"
                     "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %12}, %13;\n" ::"r"(smem_addr),
                     "l"(gmem), "r"(kDimC), "r"(kDimW), "r"(kDimN), "r"(kCubeC), "r"(kCubeW),
                     "r"(kCubeN), "l"(kStrideW), "l"(kStrideN), "r"(0), "r"(row_offset), "r"(0),
                     "r"(0));
    } else {
        static_assert(sizeof(StateT) == sizeof(BF16));
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
                     "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %12}, %13;\n" ::"r"(smem_addr),
                     "l"(gmem), "r"(kDimC), "r"(kDimW), "r"(kDimN), "r"(kCubeC), "r"(kCubeW),
                     "r"(kCubeN), "l"(kStrideW), "l"(kStrideN), "r"(0), "r"(row_offset), "r"(0),
                     "r"(0));
    }
}

__device__ __forceinline__ void warp_sum_all_pair(float &x0, float &x1) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x0 += __shfl_xor_sync(0xffffffff, x0, offset);
        x1 += __shfl_xor_sync(0xffffffff, x1, offset);
    }
}

__device__ __forceinline__ void warp_sum_lane0_pair(float &x0, float &x1) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x0 += __shfl_down_sync(0xffffffff, x0, offset);
        x1 += __shfl_down_sync(0xffffffff, x1, offset);
    }
}

__device__ __forceinline__ float decay_factor(float gate, float dt, float exp_A, float lower_bound,
                               bool use_lower_bound, float softplus_beta, float softplus_threshold) {
    float x = gate + dt;
    if (use_lower_bound) {
        // Kimi K3 safe gate: lower_bound * sigmoid(exp(A_log) * x).
        return __expf(lower_bound * __ppu_sgmdf(exp_A * x));
    }
    float bx = softplus_beta * x;
    float softplus = bx <= softplus_threshold ? __logf(1.0f + __expf(bx)) / softplus_beta : x;
    return __expf(-exp_A * softplus);
}

__device__ __forceinline__ void cp_async_fence_all() {
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
}
template <int N> __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("ppu.cp.async.wait_group %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void copy16(void* shared, const void* global) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared));
    asm volatile("ppu.cp.async.cg.shared.global.LLC::128B [%0], [%1], 16;"
                 :: "r"(address), "l"(global));
}

#endif

template <typename StateT, bool kUseBulkAiu, int NumWarps, int RowSplits = 1,
          int StatePitch = kD, bool Normalize = true>
__device__ __forceinline__ void decode_rows(
    const BF16* __restrict__ q_input,
    const BF16* __restrict__ k_input,
    const BF16* __restrict__ v_input,
    const BF16* __restrict__ gate,
    const BF16* __restrict__ beta,
    const BF16* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    StateT* __restrict__ state,
    BF16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ state_indices,
    int64_t state_slot_stride,
    int q_stride,
    int k_stride,
    int v_stride,
    int gate_stride,
    int beta_stride,
    int H,
    int state_slots,
    int tokens,
    float scale,
    float lower_bound,
    float softplus_threshold,
    bool use_lower_bound,
    bool is_sglang) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
    static_assert(StatePitch >= kD && (StatePitch * sizeof(StateT)) % 16 == 0);
    static_assert(kUseBulkAiu || StatePitch == kD);
    // BF16 padded tiles keep the low/high packets in contiguous planes.
    constexpr bool TwoPlaneParameters = sizeof(StateT) == sizeof(BF16) &&
        kUseBulkAiu && NumWarps == 8 && StatePitch == 136;
    constexpr int kWarps = NumWarps;
    constexpr int kRowsPerHead = kD / RowSplits;
    constexpr int kRowsPerWarp = kRowsPerHead / kWarps;
    constexpr bool kHalfWarpRows = sizeof(StateT) == sizeof(BF16) || (kUseBulkAiu && kRowsPerHead <= 16);
    constexpr int kRowsPerBulkGroup = 4 * kWarps;
    static_assert(!kUseBulkAiu || (kHalfWarpRows && kRowsPerHead % kRowsPerBulkGroup == 0));
    constexpr int kBulkGroupsPerHead = kRowsPerHead / kRowsPerBulkGroup;
    int const batch = blockIdx.z;
    int const head = blockIdx.y;
    int const warp = threadIdx.y;
    int const lane = threadIdx.x;
    int const tid = warp * 32 + lane;
    int const k0 = lane * 4;
    // The half-warp consumer owns two four-float packets. Put the low/high
    // packets in separate 64-float planes so each vector load is contiguous
    // across its 16 logical lanes; normalization/gate values stay unchanged.
    int const parameter_k0 = TwoPlaneParameters ? (lane >> 1) * 4 + (lane & 1) * 64 : k0;
    int const row_base = blockIdx.x * kRowsPerHead;
    int const slot = state_indices[batch];
    int const head_offset = head * kD;
    const bool readable_state = (is_sglang ? slot >= 0 : slot > 0) && slot < state_slots;

    extern __shared__ float smem[];
    StateT *sm_state = reinterpret_cast<StateT *>(smem);
    constexpr int kStateRows = kUseBulkAiu ? kRowsPerHead : kStages * kWarps;
    float *sm_q = reinterpret_cast<float *>(sm_state + kStateRows * StatePitch);
    float *sm_k = sm_q + kD;
    BF16 *sm_v = reinterpret_cast<BF16 *>(sm_k + kD);
    float *sm_decay = reinterpret_cast<float *>(sm_v + kD);
    float *sm_beta = sm_decay + kD;

    StateT *state_head = state;
    if (readable_state)
        state_head += int64_t(slot) * state_slot_stride + int64_t(head) * kStateElems;

    // Pool addressing is independent of token addressing. Start the valid
    // state read first, overlapping the device prefix loads and input setup.
    if (readable_state) {
        if constexpr (kUseBulkAiu) {
            // Split the 128 state rows across resident CTAs.  Each CTA issues one
            // contiguous AIU descriptor, halving its shared footprint while
            // preserving full-width row traffic and the total recurrence warps.
            if (warp == 0)
                aiu_load_rows<StateT, kRowsPerHead, StatePitch>(sm_state, state_head, row_base);
            cp_async_fence_all();
        } else {
            // Seed every resident rolling stage with one contiguous AIU transfer.
            if (warp == 0) {
                aiu_load_rows<StateT, kStages * kWarps>(sm_state, state_head, row_base);
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i)
                cp_async_fence_all();
        }

    }

    const int token = cu_seqlens ? cu_seqlens[batch] : batch;
    const int end = cu_seqlens ? cu_seqlens[batch + 1] : token + 1;
    const bool invalid_span = token < 0 || end < token || end - token > 1 || end > tokens;
    if (invalid_span || token == end) {
        // Even an empty or rejected request must drain its early state
        // copy before the CTA releases the shared allocation.
        cp_async_wait_all<0>();
        __ppu_barrier_sync_defer(0, kWarps * 32, 1);
        if (invalid_span) __trap();
        return;
    }

    if (is_sglang ? slot < 0 : slot <= 0) {
        if (tid < kRowsPerHead) {
            out[(token * H + head) * kD + row_base + tid] = BF16(0.0f);
        }
        return;
    }

    if (slot >= state_slots) { __trap(); return; }

    // Four independent producer warps overlap all scalar/vector preparation:
    // Q norm, K norm, KDA decay+beta, and V staging.
    if (warp == 0 || warp == 1) {
        const auto *input = static_cast<const BF16*>(warp == 0 ? q_input : k_input);
        BF16 const *src = input + int64_t(token) * (warp == 0 ? q_stride : k_stride) + head_offset;
        float4 x = load_bf16x4(src + k0);
        float sq = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        float inv_norm = Normalize ? rsqrtf(warp_sum_all(sq) + 1.0e-6f) : 1.f;
        if (warp == 0)
            inv_norm *= scale;
        x.x *= inv_norm;
        x.y *= inv_norm;
        x.z *= inv_norm;
        x.w *= inv_norm;
        *reinterpret_cast<float4 *>((warp == 0 ? sm_q : sm_k) + parameter_k0) = x;
    } else if (warp == 2) {
        float exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);
        float4 g = load_bf16x4(gate + int64_t(token) * gate_stride + head_offset + k0);
        float4 dt = load_bf16x4(dt_bias + head_offset + k0);
        float4 d = make_float4(decay_factor(g.x, dt.x, exp_A, lower_bound, use_lower_bound, lower_bound, softplus_threshold),
                               decay_factor(g.y, dt.y, exp_A, lower_bound, use_lower_bound, lower_bound, softplus_threshold),
                               decay_factor(g.z, dt.z, exp_A, lower_bound, use_lower_bound, lower_bound, softplus_threshold),
                               decay_factor(g.w, dt.w, exp_A, lower_bound, use_lower_bound, lower_bound, softplus_threshold));
        *reinterpret_cast<float4 *>(sm_decay + parameter_k0) = d;
        if (lane == 0) {
            sm_beta[0] = __ppu_sgmdf(float(beta[int64_t(token) * beta_stride + head]));
        }
    } else if (warp == 3 && lane * 4 < kRowsPerHead) {
        int const v_offset = row_base + lane * 4;
        BF16 const *src_v = static_cast<const BF16*>(v_input) + int64_t(token)*v_stride + head_offset + v_offset;
        *reinterpret_cast<uint2 *>(sm_v + v_offset) = *reinterpret_cast<uint2 const *>(src_v);
    }
    if constexpr (kUseBulkAiu) cp_async_wait_all<0>();
    // One join publishes completed bulk copies and the input producer stores.
    __ppu_barrier_sync_defer(0, kWarps * 32, 1);

    float4 q = *reinterpret_cast<float4 const *>(sm_q + k0);
    float4 k = *reinterpret_cast<float4 const *>(sm_k + k0);
    float4 decay = *reinterpret_cast<float4 const *>(sm_decay + k0);
    float beta_value = sm_beta[0];

    if constexpr (kUseBulkAiu) {
        int half = 0;
        int half_lane = 0;
        int col0 = 0;
        float4 q0_invariant, q1_invariant;
        float4 k0_invariant, k1_invariant;
        float4 d0_invariant, d1_invariant;
        constexpr bool kCacheHalfParameters =
            kHalfWarpRows && (RowSplits == 1 || sizeof(StateT) == sizeof(BF16));
        if constexpr (kCacheHalfParameters) {
            half = lane >> 4;
            half_lane = lane & 15;
            col0 = half_lane * 8;
            q0_invariant =
                *reinterpret_cast<float4 const *>(sm_q + (TwoPlaneParameters ? col0 / 2 : col0));
            q1_invariant = *reinterpret_cast<float4 const *>(
                sm_q + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
            k0_invariant =
                *reinterpret_cast<float4 const *>(sm_k + (TwoPlaneParameters ? col0 / 2 : col0));
            k1_invariant = *reinterpret_cast<float4 const *>(
                sm_k + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
            d0_invariant = *reinterpret_cast<float4 const *>(
                sm_decay + (TwoPlaneParameters ? col0 / 2 : col0));
            d1_invariant = *reinterpret_cast<float4 const *>(
                sm_decay + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
        } else if constexpr (kHalfWarpRows) {
            half = lane >> 4;
            half_lane = lane & 15;
            col0 = half_lane * 8;
        }
        #pragma unroll
        for (int group = 0; group < kBulkGroupsPerHead; ++group) {
            if constexpr (kHalfWarpRows) {
                int const row0 = row_base + group * kRowsPerBulkGroup + warp * 2 + half;
                int const row1 = row0 + 2 * kWarps;
                float4 q0, q1, k0v, k1v, d0, d1;
                if constexpr (kCacheHalfParameters) {
                    q0 = q0_invariant;
                    q1 = q1_invariant;
                    k0v = k0_invariant;
                    k1v = k1_invariant;
                    d0 = d0_invariant;
                    d1 = d1_invariant;
                } else {
                    q0 = *reinterpret_cast<float4 const *>(sm_q +
                                                           (TwoPlaneParameters ? col0 / 2 : col0));
                    q1 = *reinterpret_cast<float4 const *>(
                        sm_q + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
                    k0v = *reinterpret_cast<float4 const *>(sm_k +
                                                            (TwoPlaneParameters ? col0 / 2 : col0));
                    k1v = *reinterpret_cast<float4 const *>(
                        sm_k + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
                    d0 = *reinterpret_cast<float4 const *>(sm_decay +
                                                           (TwoPlaneParameters ? col0 / 2 : col0));
                    d1 = *reinterpret_cast<float4 const *>(
                        sm_decay + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
                }
                StateT *state_ptr0 =
                    sm_state + (group * kRowsPerBulkGroup + warp * 2 + half) * StatePitch + col0;
                StateT *state_ptr1 = state_ptr0 + 2 * kWarps * StatePitch;
                float4 h00 = load_state4(state_ptr0);
                float4 h01 = load_state4(state_ptr0 + 4);
                float4 h10 = load_state4(state_ptr1);
                float4 h11 = load_state4(state_ptr1 + 4);

                h00.x *= d0.x;
                h00.y *= d0.y;
                h00.z *= d0.z;
                h00.w *= d0.w;
                h01.x *= d1.x;
                h01.y *= d1.y;
                h01.z *= d1.z;
                h01.w *= d1.w;
                h10.x *= d0.x;
                h10.y *= d0.y;
                h10.z *= d0.z;
                h10.w *= d0.w;
                h11.x *= d1.x;
                h11.y *= d1.y;
                h11.z *= d1.z;
                h11.w *= d1.w;

                float hk0 = h00.x * k0v.x + h00.y * k0v.y + h00.z * k0v.z + h00.w * k0v.w +
                            h01.x * k1v.x + h01.y * k1v.y + h01.z * k1v.z + h01.w * k1v.w;
                float hk1 = h10.x * k0v.x + h10.y * k0v.y + h10.z * k0v.z + h10.w * k0v.w +
                            h11.x * k1v.x + h11.y * k1v.y + h11.z * k1v.z + h11.w * k1v.w;
                half_warp_sum_all_pair(hk0, hk1);
                float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
                float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;

                h00.x += k0v.x * v_new0;
                h00.y += k0v.y * v_new0;
                h00.z += k0v.z * v_new0;
                h00.w += k0v.w * v_new0;
                h01.x += k1v.x * v_new0;
                h01.y += k1v.y * v_new0;
                h01.z += k1v.z * v_new0;
                h01.w += k1v.w * v_new0;
                h10.x += k0v.x * v_new1;
                h10.y += k0v.y * v_new1;
                h10.z += k0v.z * v_new1;
                h10.w += k0v.w * v_new1;
                h11.x += k1v.x * v_new1;
                h11.y += k1v.y * v_new1;
                h11.z += k1v.z * v_new1;
                h11.w += k1v.w * v_new1;

                if constexpr (sizeof(StateT) == sizeof(BF16)) {
                    store_bf16_state8_global(state_head + row0 * kD + col0, h00, h01);
                    store_bf16_state8_global(state_head + row1 * kD + col0, h10, h11);
                } else {
                    store_state4_global(state_head + row0 * kD + col0, h00);
                    store_state4_global(state_head + row0 * kD + col0 + 4, h01);
                    store_state4_global(state_head + row1 * kD + col0, h10);
                    store_state4_global(state_head + row1 * kD + col0 + 4, h11);
                }

                float hq0 = h00.x * q0.x + h00.y * q0.y + h00.z * q0.z + h00.w * q0.w +
                            h01.x * q1.x + h01.y * q1.y + h01.z * q1.z + h01.w * q1.w;
                float hq1 = h10.x * q0.x + h10.y * q0.y + h10.z * q0.z + h10.w * q0.w +
                            h11.x * q1.x + h11.y * q1.y + h11.z * q1.z + h11.w * q1.w;
                half_warp_sum_lane0_pair(hq0, hq1);
                if (half_lane == 0) {
                    out[(token * H + head) * kD + row0] = BF16(hq0);
                    out[(token * H + head) * kD + row1] = BF16(hq1);
                }
            }
        }
    } else {
        #pragma unroll
        for (int v_pair = 0; v_pair < kRowsPerWarp; v_pair += 2) {
            int const stage0 = v_pair % kStages;
            int const stage1 = (v_pair + 1) % kStages;
            int const row0 = row_base + v_pair * kWarps + warp;
            int const row1 = row0 + kWarps;
            cp_async_wait_all<kStages - 2>();
            if (v_pair == 0) {
                __ppu_barrier_sync_defer(0, kWarps * 32, 1);
            }

            float *stage_ptr0 = sm_state + (stage0 * kWarps + warp) * kD + k0;
            float *stage_ptr1 = sm_state + (stage1 * kWarps + warp) * kD + k0;
            float4 h0 = *reinterpret_cast<float4 const *>(stage_ptr0);
            float4 h1 = *reinterpret_cast<float4 const *>(stage_ptr1);
            h0.x *= decay.x;
            h0.y *= decay.y;
            h0.z *= decay.z;
            h0.w *= decay.w;
            h1.x *= decay.x;
            h1.y *= decay.y;
            h1.z *= decay.z;
            h1.w *= decay.w;

            float hk0 = h0.x * k.x + h0.y * k.y + h0.z * k.z + h0.w * k.w;
            float hk1 = h1.x * k.x + h1.y * k.y + h1.z * k.z + h1.w * k.w;
            warp_sum_all_pair(hk0, hk1);
            float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
            float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;

            int const next0 = v_pair + kStages;
            int const next1 = next0 + 1;
            if (next0 < kRowsPerWarp) {
                float const *src0 = state_head + (row_base + next0 * kWarps + warp) * kD + k0;
                copy16(stage_ptr0, src0);
            }
            cp_async_fence_all();
            if (next1 < kRowsPerWarp) {
                float const *src1 = state_head + (row_base + next1 * kWarps + warp) * kD + k0;
                copy16(stage_ptr1, src1);
            }
            cp_async_fence_all();

            h0.x += k.x * v_new0;
            h0.y += k.y * v_new0;
            h0.z += k.z * v_new0;
            h0.w += k.w * v_new0;
            h1.x += k.x * v_new1;
            h1.y += k.y * v_new1;
            h1.z += k.z * v_new1;
            h1.w += k.w * v_new1;
            if constexpr (RowSplits == kTallRowSplits) {
                store_float4_global_cg(state_head + row0 * kD + k0, h0);
                store_float4_global_cg(state_head + row1 * kD + k0, h1);
            } else {
                store_float4_global(state_head + row0 * kD + k0, h0);
                store_float4_global(state_head + row1 * kD + k0, h1);
            }

            float hq0 = h0.x * q.x + h0.y * q.y + h0.z * q.z + h0.w * q.w;
            float hq1 = h1.x * q.x + h1.y * q.y + h1.z * q.z + h1.w * q.w;
            warp_sum_lane0_pair(hq0, hq1);
            if (lane == 0) {
                out[(token * H + head) * kD + row0] = BF16(hq0);
                out[(token * H + head) * kD + row1] = BF16(hq1);
            }
        }
        cp_async_wait_all<0>();
    }
#endif
}

template <typename StateT, bool kUseBulkAiu, int NumWarps, int RowSplits = 1,
          int StatePitch = kD, bool Normalize = true>
__global__ void kda_decode_tc02(
    const BF16* __restrict__ q_input,
    const BF16* __restrict__ k_input,
    const BF16* __restrict__ v_input,
    const BF16* __restrict__ gate,
    const BF16* __restrict__ beta,
    const BF16* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    StateT* __restrict__ state,
    BF16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ state_indices,
    int64_t state_slot_stride,
    int q_stride,
    int k_stride,
    int v_stride,
    int gate_stride,
    int beta_stride,
    int H,
    int state_slots,
    int tokens,
    float scale,
    float lower_bound,
    float softplus_threshold,
    bool use_lower_bound,
    bool is_sglang) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
    decode_rows<StateT, kUseBulkAiu, NumWarps, RowSplits, StatePitch, Normalize>(
        q_input, k_input, v_input, gate, beta, dt_bias, A_log, state, out,
        cu_seqlens, state_indices, state_slot_stride, q_stride, k_stride, v_stride,
        gate_stride, beta_stride, H, state_slots, tokens, scale, lower_bound,
        softplus_threshold, use_lower_bound, is_sglang);
#endif
}

template <typename StateT, bool kUseBulkAiu, int NumWarps, int RowSplits = 1,
          int StatePitch = kD, bool Normalize = true>
__global__ void kda_decode_packed_tc02(
    const BF16* __restrict__ mixed_qkv, const BF16* __restrict__ gate,
    const BF16* __restrict__ beta, const BF16* __restrict__ dt_bias,
    const float* __restrict__ A_log, StateT* __restrict__ state,
    BF16* __restrict__ out, const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ state_indices, int64_t state_slot_stride,
    int H, int state_slots, int tokens, float scale, float gate_parameter, uint32_t flags) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
    // Including the runtime's hidden arguments, this entry fits the 32 input
    // SREG slots. Keep pitches/default softplus attributes out of global km.
    static_assert(Normalize);
    decode_rows<StateT, kUseBulkAiu, NumWarps, RowSplits, StatePitch, true>(
        mixed_qkv, mixed_qkv + H*kD, mixed_qkv + 2*H*kD, gate, beta, dt_bias,
        A_log, state, out, cu_seqlens, state_indices, state_slot_stride,
        3*H*kD, 3*H*kD, 3*H*kD, H*kD, H, H, state_slots, tokens,
        scale, gate_parameter, 20.f, (flags & 1) != 0, (flags & 2) != 0);
#endif
}

template <typename State, bool Bulk, int Warps, int Splits, int Pitch = kD>
void launch_decode(const KdaDecodeTc02Args& args, hggcStream_t stream) {
    constexpr int Rows = Bulk ? kD / Splits : kStages * Warps;
    constexpr int Smem = sizeof(State)*Rows*Pitch + sizeof(float)*(3*kD+1) + sizeof(BF16)*kD;
    const int head_elems = args.heads * kD;
    const uintptr_t q = reinterpret_cast<uintptr_t>(args.q);
    const bool packed = args.q_stride == 3*head_elems && args.k_stride == args.q_stride
        && args.v_stride == args.q_stride && args.gate_stride == head_elems
        && args.beta_stride == args.heads
        && (args.use_lower_bound || (args.softplus_beta_or_lower_bound == 1.f
                                    && args.softplus_threshold == 20.f))
        && reinterpret_cast<uintptr_t>(args.k) == q + head_elems*sizeof(BF16)
        && reinterpret_cast<uintptr_t>(args.v) == q + 2*head_elems*sizeof(BF16);
    auto launch = [&](auto normalize) {
        constexpr bool Normalize = decltype(normalize)::value;
        if constexpr (Normalize) {
            if (packed) {
                kda_decode_packed_tc02<State, Bulk, Warps, Splits, Pitch, true>
                    <<<dim3(Splits, args.heads, args.batch), dim3(32, Warps), Smem, stream>>>(
                        static_cast<const BF16*>(args.q), static_cast<const BF16*>(args.gate),
                        static_cast<const BF16*>(args.beta), static_cast<const BF16*>(args.dt_bias),
                        args.a_log, static_cast<State*>(args.state), static_cast<BF16*>(args.out),
                        args.cu_seqlens, args.state_indices, args.state_stride, args.heads,
                        args.state_slots, args.tokens, args.scale, args.softplus_beta_or_lower_bound,
                        (args.use_lower_bound ? 1u : 0u) | (args.is_sglang ? 2u : 0u));
                return;
            }
        }
        kda_decode_tc02<State, Bulk, Warps, Splits, Pitch, Normalize>
        <<<dim3(Splits, args.heads, args.batch), dim3(32, Warps), Smem, stream>>>(
            static_cast<const BF16*>(args.q),
            static_cast<const BF16*>(args.k),
            static_cast<const BF16*>(args.v),
            static_cast<const BF16*>(args.gate),
            static_cast<const BF16*>(args.beta),
            static_cast<const BF16*>(args.dt_bias),
            args.a_log,
            static_cast<State*>(args.state),
            static_cast<BF16*>(args.out),
            args.cu_seqlens,
            args.state_indices,
            args.state_stride,
            args.q_stride,
            args.k_stride,
            args.v_stride,
            args.gate_stride,
            args.beta_stride,
            args.heads,
            args.state_slots,
            args.tokens,
            args.scale,
            args.softplus_beta_or_lower_bound,
            args.softplus_threshold,
            args.use_lower_bound,
            args.is_sglang);
    };
    if (args.use_qk_l2norm) launch(std::true_type{});
    else launch(std::false_type{});
}

}  // namespace

int kda_tc02_cu_count(int device) {
    static std::array<std::atomic<int>, 64> capacities{};
    if (device < 0 || device >= int(capacities.size())) return 0;
    int capacity = capacities[device].load(std::memory_order_relaxed);
    if (capacity == 0) {
        hggcDeviceProp properties{};
        if (hggcGetDeviceProperties(&properties, device) != hggcSuccess)
            throw std::runtime_error("KDA device properties query failed");
        capacity = properties.major == 8 && properties.minor == 9
            ? properties.multiProcessorCount : -1;
        capacities[device].store(capacity, std::memory_order_relaxed);
    }
    return capacity;
}

// Plans encode a small set of transport/row layouts, not model dimensions.
using DecodePlan = KdaDecodeTc02Plan;

KdaDecodeTc02Plan kda_decode_tc02_plan(int device, int batch, int heads, bool bf16_state,
                                     int fallback_rows) {
    const int capacity = kda_tc02_cu_count(device);
    if (capacity <= 0) return DecodePlan::Fallback;
    const int64_t work = int64_t(batch) * heads;
    if (bf16_state) {
        if (3 * work <= capacity) return DecodePlan::Small;
        return work <= 64 * capacity ? DecodePlan::BulkSplit : DecodePlan::BulkWhole;
    }
    if (2 * work < capacity) return DecodePlan::SmallBulk;
    if (work <= 3 * capacity) return DecodePlan::Small;
    // A saturated FP32 grid benefits from the existing V-tiled launcher:
    // its shorter state rows expose more independent memory operations.
    if (fallback_rows < kD && work >= 24 * capacity) return DecodePlan::Fallback;
    // At intermediate density, rolling rows avoid a second split and
    // keep state traffic independent of the number of heads per slot.
    return DecodePlan::Rolling;
}

void launch_kda_decode_tc02(const KdaDecodeTc02Args& args, KdaDecodeTc02Plan plan, hggcStream_t stream) {
    if (args.bf16_state) {
        if (plan == DecodePlan::Small) launch_decode<BF16, true, 4, 8, 128>(args, stream);
        else if (plan == DecodePlan::BulkSplit) launch_decode<BF16, true, 8, 2, 136>(args, stream);
        else launch_decode<BF16, true, 8, 1, 136>(args, stream);
    } else {
        if (plan == DecodePlan::SmallBulk) launch_decode<float, true, 4, 8, 128>(args, stream);
        else if (plan == DecodePlan::Small) launch_decode<float, false, 4, 4>(args, stream);
        else launch_decode<float, false, 8, 1>(args, stream);
    }
}


// Linear Verify shares the Decode transport helpers.

namespace {


constexpr int kWarps = 4;

template <int Rows> struct VerifyScratch {
    static constexpr int kFloatStride = Rows == 32 ? kD + 4 : kD;
    static constexpr int kPayloadBytes =
        sizeof(float) * (3 * kFloatStride + 4) + sizeof(BF16) * Rows;
    static constexpr int kStrideBytes = (kPayloadBytes + 15) & ~15;
    float *q;
    float *k;
    float *decay;
    float *beta;
    BF16 *v;
};

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150

__device__ __forceinline__ float half_warp_sum_all(float x) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 16);
    }
    return x;
}

__device__ __forceinline__ float half_warp_sum_lane0(float x) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset, 16);
    }
    return x;
}


template <int Rows>
__device__ __forceinline__ VerifyScratch<Rows> get_verify_scratch(unsigned char *raw_smem, int token) {
    unsigned char *base = raw_smem + (token & 1) * VerifyScratch<Rows>::kStrideBytes;
    float *q = reinterpret_cast<float *>(base);
    float *k = q + VerifyScratch<Rows>::kFloatStride;
    float *decay = k + VerifyScratch<Rows>::kFloatStride;
    float *beta = decay + VerifyScratch<Rows>::kFloatStride;
    BF16 *v = reinterpret_cast<BF16 *>(beta + 4);
    return {q, k, decay, beta, v};
}

// State tiles are copied by the ppu0015 AIU into ordinary shared memory.  Do
// not use an AIU swizzled destination here: the recurrent loop needs scalar
// element addressing for the same tile at every verify step.

template <int Rows>
__device__ __forceinline__ void
produce_token(BF16 const *q, BF16 const *k, BF16 const *v, BF16 const *gate, BF16 const *beta,
              float const *dt_bias, float exp_A, float scale, float lower_bound, int token,
              int head, int H, int64_t q_token_stride, int64_t k_token_stride,
              int64_t v_token_stride, int64_t gate_token_stride, int row_base, int warp, int lane,
              float *sm_q, float *sm_k, float *sm_decay, float *sm_beta, BF16 *sm_v) {
    // The caller passes token as the already packed B*T token index.
    int64_t const q_base = int64_t(token) * q_token_stride + head * kD;
    int64_t const k_base = int64_t(token) * k_token_stride + head * kD;
    int64_t const v_base = int64_t(token) * v_token_stride + head * kD;
    int64_t const gate_base = int64_t(token) * gate_token_stride + head * kD;
    if (warp == 0) {
        // Run the independent Q and K reductions in the two 16-lane groups.
        // This frees a second warp for the SFU-heavy gate/decay transform.
        int const half = lane >> 4;
        int const half_lane = lane & 15;
        int const col = half_lane * 8;
        BF16 const *src = half == 0 ? q + q_base : k + k_base;
        float *dst = half == 0 ? sm_q : sm_k;
        float4 x0 = load_bf16x4(src + col);
        float4 x1 = load_bf16x4(src + col + 4);
        float sq = x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w + x1.x * x1.x +
                   x1.y * x1.y + x1.z * x1.z + x1.w * x1.w;
        float inv = rsqrtf(half_warp_sum_all(sq) + 1.0e-6f);
        inv *= half == 0 ? scale : 1.0f;
        x0.x *= inv;
        x0.y *= inv;
        x0.z *= inv;
        x0.w *= inv;
        x1.x *= inv;
        x1.y *= inv;
        x1.z *= inv;
        x1.w *= inv;
        *reinterpret_cast<float4 *>(dst + col) = x0;
        *reinterpret_cast<float4 *>(dst + col + 4) = x1;
    } else if (warp == 1 || warp == 2) {
        int const col = (warp - 1) * 64 + lane * 2;
        float2 a = bf16x2_to_float2(*reinterpret_cast<uint32_t const *>(gate + gate_base + col));
        float2 dt = *reinterpret_cast<float2 const *>(dt_bias + head * kD + col);
        float2 d = make_float2(__expf(lower_bound * __ppu_sgmdf(exp_A * (a.x + dt.x))),
                               __expf(lower_bound * __ppu_sgmdf(exp_A * (a.y + dt.y))));
        *reinterpret_cast<float2 *>(sm_decay + col) = d;
        if (warp == 1 && lane == 0) {
            sm_beta[0] = __ppu_sgmdf(float(beta[token * H + head]));
        }
    } else if (warp == 3) {
        // The scratch tail is 16-byte aligned without increasing its rounded
        // allocation, so copy eight BF16 V values per active lane.
        if (lane < Rows / 8) {
            int const row = lane * 8;
            *reinterpret_cast<uint4 *>(sm_v + row) =
                *reinterpret_cast<uint4 const *>(v + v_base + row_base + row);
        }
    }
}

#endif

template <typename StateT, int RowSplits>
__device__ __forceinline__ void verify_rows(
    const BF16* __restrict__ q,
    const BF16* __restrict__ k,
    const BF16* __restrict__ v,
    const BF16* __restrict__ gate,
    const BF16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const StateT* __restrict__ state,
    StateT* __restrict__ intermediate_states,
    BF16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ intermediate_state_indices,
    int64_t state_slot_stride,
    int64_t intermediate_slot_stride,
    int64_t intermediate_step_stride,
    int q_token_stride,
    int k_token_stride,
    int v_token_stride,
    int gate_token_stride,
    int H,
    int max_steps,
    int tokens,
    int state_slots,
    int intermediate_slots,
    float scale,
    float lower_bound) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
    constexpr int kRows = kD / RowSplits;
    int const batch = blockIdx.z;
    int const head = blockIdx.y;
    int const row_base = blockIdx.x * kRows;
    int const warp = threadIdx.y;
    int const lane = threadIdx.x;
    int const tid = warp * 32 + lane;
    int const col = lane * 4;
    const int token_begin = cu_seqlens ? cu_seqlens[batch] : batch * max_steps;
    const int token_end = cu_seqlens ? cu_seqlens[batch + 1] : token_begin + max_steps;
    const int T = token_end - token_begin;
    if (token_begin < 0 || token_end < token_begin || token_end > tokens || T > max_steps) {
        __trap(); return;
    }
    if (T == 0) return;
    int const slot = state_indices[batch];
    int const intermediate_slot = intermediate_state_indices[batch];
    constexpr bool kDoubleScratch = sizeof(StateT) == sizeof(BF16) && RowSplits == 4;

    extern __shared__ unsigned char raw_smem[];
    StateT *sm_state = reinterpret_cast<StateT *>(raw_smem);
    // The state tile is needed only for the initial AIU load. Once every warp
    // has moved its rows into registers, reuse the allocation for two aligned
    // producer-scratch stages.

    if (slot < 0) {
        for (int t = 0; t < T; ++t) {
            BF16 *out_token = out + ((token_begin + t) * H + head) * kD;
            for (int d = tid; d < kRows; d += kWarps * 32)
                out_token[row_base + d] = BF16(0.0f);
        }
        return;
    }

    if (slot >= state_slots || intermediate_slot >= intermediate_slots) {
        __trap(); return;
    }

    StateT const *state_head =
        state + int64_t(slot) * state_slot_stride + int64_t(head) * kD * kD + row_base * kD;
    if (warp == 0)
        aiu_load_rows<StateT, kRows, kD, kRows>(sm_state, state_head);
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
    asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    __ppu_barrier_sync_defer(0, kWarps * 32, 1);

    if constexpr (sizeof(StateT) == sizeof(BF16) && RowSplits == 16) {
        // The eight-row tile gives each warp exactly two rows. Assign one row
        // to each 16-lane group so the register footprint stays identical to
        // the full-warp path while reductions and BF16 stores become wider.
        int const half = lane >> 4;
        int const half_lane = lane & 15;
        int const col0 = half_lane * 8;
        int const row = warp * 2 + half;
        float4 h0 = load_state4(sm_state + row * kD + col0);
        float4 h1 = load_state4(sm_state + row * kD + col0 + 4);
        __ppu_barrier_sync_defer(0, kWarps * 32, 1);

        float const exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);
        for (int t = 0; t < T; ++t) {
            auto scratch = get_verify_scratch<kRows>(raw_smem, kDoubleScratch ? t : 0);
            float *sm_q = scratch.q;
            float *sm_k = scratch.k;
            float *sm_decay = scratch.decay;
            float *sm_beta = scratch.beta;
            BF16 *sm_v = scratch.v;
            produce_token<kRows>(q, k, v, gate, beta, dt_bias, exp_A, scale, lower_bound,
                                 token_begin + t, head, H, q_token_stride, k_token_stride,
                                 v_token_stride, gate_token_stride, row_base, warp, lane, sm_q,
                                 sm_k, sm_decay, sm_beta, sm_v);
            __ppu_barrier_sync_defer(0, kWarps * 32, 1);

            float4 q0 = *reinterpret_cast<float4 const *>(sm_q + col0);
            float4 q1 = *reinterpret_cast<float4 const *>(sm_q + col0 + 4);
            float4 k0 = *reinterpret_cast<float4 const *>(sm_k + col0);
            float4 k1 = *reinterpret_cast<float4 const *>(sm_k + col0 + 4);
            float4 d0 = *reinterpret_cast<float4 const *>(sm_decay + col0);
            float4 d1 = *reinterpret_cast<float4 const *>(sm_decay + col0 + 4);
            float const beta_value = sm_beta[0];
            h0.x *= d0.x;
            h0.y *= d0.y;
            h0.z *= d0.z;
            h0.w *= d0.w;
            h1.x *= d1.x;
            h1.y *= d1.y;
            h1.z *= d1.z;
            h1.w *= d1.w;
            float hk = h0.x * k0.x + h0.y * k0.y + h0.z * k0.z + h0.w * k0.w + h1.x * k1.x +
                       h1.y * k1.y + h1.z * k1.z + h1.w * k1.w;
            hk = half_warp_sum_all(hk);
            float const v_new = (float(sm_v[row]) - hk) * beta_value;

            h0.x += k0.x * v_new;
            h0.y += k0.y * v_new;
            h0.z += k0.z * v_new;
            h0.w += k0.w * v_new;
            h1.x += k1.x * v_new;
            h1.y += k1.y * v_new;
            h1.z += k1.z * v_new;
            h1.w += k1.w * v_new;
            if (intermediate_slot >= 0) {
                BF16 *checkpoint_head = reinterpret_cast<BF16 *>(
                    intermediate_states + int64_t(intermediate_slot) * intermediate_slot_stride +
                    int64_t(t) * intermediate_step_stride + int64_t(head) * kD * kD +
                    row_base * kD);
                store_bf16_state8_global<true>(checkpoint_head + row * kD + col0, h0, h1);
            }
            float hq = h0.x * q0.x + h0.y * q0.y + h0.z * q0.z + h0.w * q0.w + h1.x * q1.x +
                       h1.y * q1.y + h1.z * q1.z + h1.w * q1.w;
            hq = half_warp_sum_lane0(hq);
            if (half_lane == 0) {
                out[((token_begin + t) * H + head) * kD + row_base + row] = BF16(hq);
            }
            if constexpr (!kDoubleScratch)
                __ppu_barrier_sync_defer(0, kWarps * 32, 1);
        }
    } else if constexpr (sizeof(StateT) == sizeof(BF16) && (RowSplits == 4 || RowSplits == 8)) {
        // Two independent 16-lane groups share each warp. Each lane owns eight
        // contiguous columns for two rows, preserving the original per-thread
        // FP32 state count while halving shuffle depth and issuing natural
        // aligned 16-byte BF16 checkpoint packets.
        constexpr int kRowsPerWarp = kRows / kWarps;
        constexpr int kGroupsPerWarp = kRowsPerWarp / 4;
        int const half = lane >> 4;
        int const half_lane = lane & 15;
        int const col0 = half_lane * 8;
        float4 h00[kGroupsPerWarp];
        float4 h01[kGroupsPerWarp];
        float4 h10[kGroupsPerWarp];
        float4 h11[kGroupsPerWarp];
        #pragma unroll
        for (int group = 0; group < kGroupsPerWarp; ++group) {
            int const row0 = group * 4 * kWarps + warp * 2 + half;
            int const row1 = row0 + 2 * kWarps;
            h00[group] = load_state4(sm_state + row0 * kD + col0);
            h01[group] = load_state4(sm_state + row0 * kD + col0 + 4);
            h10[group] = load_state4(sm_state + row1 * kD + col0);
            h11[group] = load_state4(sm_state + row1 * kD + col0 + 4);
        }
        __ppu_barrier_sync_defer(0, kWarps * 32, 1);

        float const exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);
        for (int t = 0; t < T; ++t) {
            auto scratch = get_verify_scratch<kRows>(raw_smem, kDoubleScratch ? t : 0);
            float *sm_q = scratch.q;
            float *sm_k = scratch.k;
            float *sm_decay = scratch.decay;
            float *sm_beta = scratch.beta;
            BF16 *sm_v = scratch.v;
            produce_token<kRows>(q, k, v, gate, beta, dt_bias, exp_A, scale, lower_bound,
                                 token_begin + t, head, H, q_token_stride, k_token_stride,
                                 v_token_stride, gate_token_stride, row_base, warp, lane, sm_q,
                                 sm_k, sm_decay, sm_beta, sm_v);
            __ppu_barrier_sync_defer(0, kWarps * 32, 1);

            float4 q0 = *reinterpret_cast<float4 const *>(sm_q + col0);
            float4 q1 = *reinterpret_cast<float4 const *>(sm_q + col0 + 4);
            float4 k0 = *reinterpret_cast<float4 const *>(sm_k + col0);
            float4 k1 = *reinterpret_cast<float4 const *>(sm_k + col0 + 4);
            float4 d0 = *reinterpret_cast<float4 const *>(sm_decay + col0);
            float4 d1 = *reinterpret_cast<float4 const *>(sm_decay + col0 + 4);
            float const beta_value = sm_beta[0];
            BF16 *checkpoint_head = nullptr;
            if (intermediate_slot >= 0) {
                checkpoint_head = reinterpret_cast<BF16 *>(
                    intermediate_states + int64_t(intermediate_slot) * intermediate_slot_stride +
                    int64_t(t) * intermediate_step_stride + int64_t(head) * kD * kD +
                    row_base * kD);
            }

            #pragma unroll
            for (int group = 0; group < kGroupsPerWarp; ++group) {
                int const row0 = group * 4 * kWarps + warp * 2 + half;
                int const row1 = row0 + 2 * kWarps;
                float4 &x00 = h00[group];
                float4 &x01 = h01[group];
                float4 &x10 = h10[group];
                float4 &x11 = h11[group];
                x00.x *= d0.x;
                x00.y *= d0.y;
                x00.z *= d0.z;
                x00.w *= d0.w;
                x01.x *= d1.x;
                x01.y *= d1.y;
                x01.z *= d1.z;
                x01.w *= d1.w;
                x10.x *= d0.x;
                x10.y *= d0.y;
                x10.z *= d0.z;
                x10.w *= d0.w;
                x11.x *= d1.x;
                x11.y *= d1.y;
                x11.z *= d1.z;
                x11.w *= d1.w;
                float hk0 = x00.x * k0.x + x00.y * k0.y + x00.z * k0.z + x00.w * k0.w +
                            x01.x * k1.x + x01.y * k1.y + x01.z * k1.z + x01.w * k1.w;
                float hk1 = x10.x * k0.x + x10.y * k0.y + x10.z * k0.z + x10.w * k0.w +
                            x11.x * k1.x + x11.y * k1.y + x11.z * k1.z + x11.w * k1.w;
                half_warp_sum_all_pair(hk0, hk1);
                float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
                float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;

                x00.x += k0.x * v_new0;
                x00.y += k0.y * v_new0;
                x00.z += k0.z * v_new0;
                x00.w += k0.w * v_new0;
                x01.x += k1.x * v_new0;
                x01.y += k1.y * v_new0;
                x01.z += k1.z * v_new0;
                x01.w += k1.w * v_new0;
                x10.x += k0.x * v_new1;
                x10.y += k0.y * v_new1;
                x10.z += k0.z * v_new1;
                x10.w += k0.w * v_new1;
                x11.x += k1.x * v_new1;
                x11.y += k1.y * v_new1;
                x11.z += k1.z * v_new1;
                x11.w += k1.w * v_new1;
                if (checkpoint_head != nullptr) {
                    store_bf16_state8_global<true>(checkpoint_head + row0 * kD + col0, x00, x01);
                    store_bf16_state8_global<true>(checkpoint_head + row1 * kD + col0, x10, x11);
                }
                float hq0 = x00.x * q0.x + x00.y * q0.y + x00.z * q0.z + x00.w * q0.w +
                            x01.x * q1.x + x01.y * q1.y + x01.z * q1.z + x01.w * q1.w;
                float hq1 = x10.x * q0.x + x10.y * q0.y + x10.z * q0.z + x10.w * q0.w +
                            x11.x * q1.x + x11.y * q1.y + x11.z * q1.z + x11.w * q1.w;
                half_warp_sum_lane0_pair(hq0, hq1);
                if (half_lane == 0) {
                    out[((token_begin + t) * H + head) * kD + row_base + row0] = BF16(hq0);
                    out[((token_begin + t) * H + head) * kD + row_base + row1] = BF16(hq1);
                }
            }
            if constexpr (!kDoubleScratch)
                __ppu_barrier_sync_defer(0, kWarps * 32, 1);
        }
    } else {
        // Each warp owns a fixed set of V rows for the complete verify chain.
        // Keeping those rows in FP32 registers removes one shared read and one
        // shared write per state element per draft step. It also matches Triton's
        // recurrence semantics: BF16 is used at pool/checkpoint boundaries, while
        // the private chain state remains FP32 between steps.
        constexpr int kRowsPerWarp = kRows / kWarps;
        float4 h_regs[kRowsPerWarp];
        #pragma unroll
        for (int owned = 0; owned < kRowsPerWarp; ++owned) {
            int const row = warp + owned * kWarps;
            h_regs[owned] = load_state4(sm_state + row * kD + col);
        }
        __ppu_barrier_sync_defer(0, kWarps * 32, 1);

        float const exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);

        // Load the committed state once. Every subsequent step updates only the
        // private register tile, then stores it to the speculative checkpoint cache.
        for (int t = 0; t < T; ++t) {
            auto scratch = get_verify_scratch<kRows>(raw_smem, kDoubleScratch ? t : 0);
            float *sm_q = scratch.q;
            float *sm_k = scratch.k;
            float *sm_decay = scratch.decay;
            float *sm_beta = scratch.beta;
            BF16 *sm_v = scratch.v;
            produce_token<kRows>(q, k, v, gate, beta, dt_bias, exp_A, scale, lower_bound,
                                 token_begin + t, head, H, q_token_stride, k_token_stride,
                                 v_token_stride, gate_token_stride, row_base, warp, lane, sm_q,
                                 sm_k, sm_decay, sm_beta, sm_v);
            __ppu_barrier_sync_defer(0, kWarps * 32, 1);

            float4 qv = *reinterpret_cast<float4 const *>(sm_q + col);
            float4 kv = *reinterpret_cast<float4 const *>(sm_k + col);
            float4 decay = *reinterpret_cast<float4 const *>(sm_decay + col);
            float const beta_value = sm_beta[0];
            StateT *checkpoint_head = nullptr;
            if (intermediate_slot >= 0) {
                checkpoint_head =
                    intermediate_states + int64_t(intermediate_slot) * intermediate_slot_stride +
                    int64_t(t) * intermediate_step_stride + int64_t(head) * kD * kD + row_base * kD;
            }

            // Each warp owns disjoint V rows.  The recurrence for one V row is
            // independent of every other row, so no inter-CTA synchronization is
            // needed between draft steps.
            #pragma unroll
            for (int owned = 0; owned < kRowsPerWarp; ++owned) {
                int const row = warp + owned * kWarps;
                float4 &h = h_regs[owned];
                h.x *= decay.x;
                h.y *= decay.y;
                h.z *= decay.z;
                h.w *= decay.w;
                float hk = h.x * kv.x + h.y * kv.y + h.z * kv.z + h.w * kv.w;
                hk = warp_sum_all(hk);
                float const v_new = (float(sm_v[row]) - hk) * beta_value;

                h.x += kv.x * v_new;
                h.y += kv.y * v_new;
                h.z += kv.z * v_new;
                h.w += kv.w * v_new;
                if (checkpoint_head != nullptr) {
                    store_state4_global(checkpoint_head + row * kD + col, h);
                }
                float hq = h.x * qv.x + h.y * qv.y + h.z * qv.z + h.w * qv.w;
                hq = warp_sum_lane0(hq);
                if (lane == 0) {
                    out[((token_begin + t) * H + head) * kD + row_base + row] = BF16(hq);
                }
            }
            if constexpr (!kDoubleScratch)
                __ppu_barrier_sync_defer(0, kWarps * 32, 1);
        }
    }
#endif
}

template <typename StateT, int RowSplits>
__global__ void kda_verify_tc02(
    const BF16* __restrict__ q,
    const BF16* __restrict__ k,
    const BF16* __restrict__ v,
    const BF16* __restrict__ gate,
    const BF16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const StateT* __restrict__ state,
    StateT* __restrict__ intermediate_states,
    BF16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ intermediate_state_indices,
    int64_t state_slot_stride,
    int64_t intermediate_slot_stride,
    int64_t intermediate_step_stride,
    int q_token_stride,
    int k_token_stride,
    int v_token_stride,
    int gate_token_stride,
    int H,
    int max_steps,
    int tokens,
    int state_slots,
    int intermediate_slots,
    float scale,
    float lower_bound) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
    verify_rows<StateT, RowSplits>(q, k, v, gate, beta, A_log, dt_bias, state,
        intermediate_states, out, cu_seqlens, state_indices, intermediate_state_indices,
        state_slot_stride, intermediate_slot_stride, intermediate_step_stride,
        q_token_stride, k_token_stride, v_token_stride, gate_token_stride,
        H, max_steps, tokens, state_slots, intermediate_slots, scale, lower_bound);
#endif
}

template <typename StateT, int RowSplits>
__global__ void kda_verify_packed_tc02(
    uint32_t shape, uint32_t pool_capacities, float scale, float lower_bound,
    const BF16* __restrict__ mixed_qkv, const BF16* __restrict__ gate,
    const BF16* __restrict__ beta, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, const StateT* __restrict__ state,
    StateT* __restrict__ cache, BF16* __restrict__ out,
    const int32_t* __restrict__ cu_seqlens, const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ cache_indices) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
    // Two metadata words keep the entire ABI in input SREGs. The host checks
    // representability before encoding; larger or ragged layouts use the full
    // entry below. This changes metadata transport, not recurrence precision.
    const int H = shape & UINT16_MAX;
    const int max_steps = (shape >> 16) & UINT8_MAX;
    const int cache_steps = shape >> 24;
    const int state_slots = pool_capacities & UINT16_MAX;
    const int cache_slots = pool_capacities >> 16;
    const int tokens = int(gridDim.z) * max_steps;
    const int64_t state_elems = int64_t(H) * kD * kD;
    verify_rows<StateT, RowSplits>(
        mixed_qkv, mixed_qkv + H*kD, mixed_qkv + 2*H*kD, gate, beta, A_log, dt_bias,
        state, cache, out, cu_seqlens, state_indices, cache_indices,
        state_elems, int64_t(cache_steps)*state_elems, state_elems,
        3*H*kD, 3*H*kD, 3*H*kD, H*kD, H, max_steps, tokens,
        state_slots, cache_slots, scale, lower_bound);
#endif
}

template <typename State, int Splits>
void launch_verify(const KdaVerifyTc02Args& args, hggcStream_t stream) {
    constexpr int Rows = kD / Splits;
    constexpr bool DoubleScratch = sizeof(State) == sizeof(BF16) && Splits == 4;
    constexpr int StateBytes = sizeof(State) * Rows * kD;
    constexpr int ScratchBytes = (DoubleScratch ? 2 : 1) * VerifyScratch<Rows>::kStrideBytes;
    constexpr int Smem = StateBytes > ScratchBytes ? StateBytes : ScratchBytes;
    const int head_elems = args.heads * kD;
    const int64_t state_elems = int64_t(head_elems) * kD;
    const uintptr_t q = reinterpret_cast<uintptr_t>(args.q);
    const bool packed = args.q_stride == 3*head_elems && args.k_stride == args.q_stride
        && args.v_stride == args.q_stride && args.gate_stride == head_elems
        && args.state_stride == state_elems && args.cache_step_stride == state_elems
        && args.cache_stride >= int64_t(args.max_steps)*state_elems
        && args.cache_stride % state_elems == 0
        && args.heads <= UINT16_MAX && args.max_steps <= UINT8_MAX
        && args.cache_stride/state_elems <= UINT8_MAX
        && args.state_slots <= UINT16_MAX && args.cache_slots <= UINT16_MAX
        && int64_t(args.tokens) == int64_t(args.batch)*args.max_steps
        && reinterpret_cast<uintptr_t>(args.k) == q + head_elems*sizeof(BF16)
        && reinterpret_cast<uintptr_t>(args.v) == q + 2*head_elems*sizeof(BF16);
    if (packed) {
        kda_verify_packed_tc02<State, Splits>
            <<<dim3(Splits, args.heads, args.batch), dim3(32, kWarps), Smem, stream>>>(
                uint32_t(args.heads) | (uint32_t(args.max_steps) << 16)
                    | (uint32_t(args.cache_stride/state_elems) << 24),
                uint32_t(args.state_slots) | (uint32_t(args.cache_slots) << 16),
                args.scale, args.lower_bound,
                static_cast<const BF16*>(args.q), static_cast<const BF16*>(args.gate),
                static_cast<const BF16*>(args.beta), args.a_log, args.dt_bias,
                static_cast<const State*>(args.state), static_cast<State*>(args.cache),
                static_cast<BF16*>(args.out), args.cu_seqlens, args.state_indices, args.cache_indices);
        return;
    }
    kda_verify_tc02<State, Splits>
        <<<dim3(Splits, args.heads, args.batch), dim3(32, kWarps), Smem, stream>>>(
            static_cast<const BF16*>(args.q),
            static_cast<const BF16*>(args.k),
            static_cast<const BF16*>(args.v),
            static_cast<const BF16*>(args.gate),
            static_cast<const BF16*>(args.beta),
            args.a_log,
            args.dt_bias,
            static_cast<const State*>(args.state),
            static_cast<State*>(args.cache),
            static_cast<BF16*>(args.out),
            args.cu_seqlens,
            args.state_indices,
            args.cache_indices,
            args.state_stride,
            args.cache_stride,
            args.cache_step_stride,
            args.q_stride,
            args.k_stride,
            args.v_stride,
            args.gate_stride,
            args.heads,
            args.max_steps,
            args.tokens,
            args.state_slots,
            args.cache_slots,
            args.scale,
            args.lower_bound);
}

}  // namespace

int kda_verify_tc02_splits(int device, int batch, int heads, bool bf16_state) {
    const int capacity = kda_tc02_cu_count(device);
    if (capacity <= 0) return 0;
    const int64_t work = int64_t(batch) * heads;
    // FP32 rolling remains effective once the grid has two CU waves. BF16
    // amortizes a register-resident chain across the whole concurrency range.
    if (!bf16_state && work > 2 * capacity) return 0;
    if (3 * work <= capacity) return 16;
    return 2 * work <= 5 * capacity ? 8 : 4;
}

void launch_kda_verify_tc02(const KdaVerifyTc02Args& args, int splits, hggcStream_t stream) {
    if (args.bf16_state) {
        if (splits == 16) launch_verify<BF16, 16>(args, stream);
        else if (splits == 8) launch_verify<BF16, 8>(args, stream);
        else launch_verify<BF16, 4>(args, stream);
    } else {
        if (splits == 16) launch_verify<float, 16>(args, stream);
        else launch_verify<float, 8>(args, stream);
    }
}

}  // namespace pla
