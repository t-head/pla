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
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <stdexcept>

#include "../../pla_device_helpers.cuh"
#include "k_last_shared_layout.h"


// =============================================================================
// State pipeline traits — compile-time constants per state dtype
// =============================================================================
template<typename state_t> struct StatePipeline {
    static constexpr int TASKS_PER_STAGE = 1;   // fp32: 1 task × 16B = 16B per thread per stage
    static constexpr int COPY_BYTES = 16;       // cp.async.cg 16B
};
template<> struct StatePipeline<__ppu_bfloat16> {
    static constexpr int TASKS_PER_STAGE = 2;   // bf16: 2 tasks × 8B = 16B per thread per stage
    static constexpr int COPY_BYTES = 8;        // cp.async.ca 8B
};

// =============================================================================
// Main kernel
// =============================================================================
template <typename alog_t, typename scalar_t, int WARP_NUM, int BV, int HV_PER_CTA, int STAGE_NUM, bool IS_KDA, bool use_initial_state=true, typename state_t=float>
__global__ void fused_sigmoid_gating_delta_kernel_packed(
    const alog_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ dt_bias,
    const float softplus_beta,
    const float softplus_threshold,
    const scalar_t* __restrict__ mixed_qkv,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ o,
    state_t* __restrict__ h0_source,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ h0_indices,
    const float scale,
    const int T_max,
    const int stride_a, const int stride_qkv, const int stride_b,
    const int64_t h0_slot_stride,
    const int B, const int H, const int HV, const int K, const int V,
    const bool use_qk_l2norm,
    const bool is_sglang)
{
    constexpr int V_ITERS = BV / WARP_NUM;
    constexpr int TOTAL_TASKS = HV_PER_CTA * V_ITERS;
    constexpr int TASKS_PER_STAGE = StatePipeline<state_t>::TASKS_PER_STAGE;
    constexpr int COPY_BYTES = StatePipeline<state_t>::COPY_BYTES;

    // =========================================================================
    // PREAMBLE — shared by both state-dtype paths
    // =========================================================================
    int i_n;
    int i_h;
    int hv_per_h;
    if ((H & (H - 1)) == 0) {
        i_n = divide_by_power_of_two(blockIdx.z, H);
        i_h = blockIdx.z & (H - 1);
        hv_per_h = divide_by_power_of_two(HV, H);
    } else {
        // A non-power-of-two head count cannot split batch and head with shift/mask.
        i_n = blockIdx.z / H;
        i_h = blockIdx.z % H;
        hv_per_h = HV / H;
    }
    int hv_base = i_h * hv_per_h + blockIdx.y * HV_PER_CTA;
    int v_base = blockIdx.x * BV;

    int bos = (cu_seqlens != nullptr) ? cu_seqlens[i_n] : (i_n * T_max);
    const int64_t eos = (cu_seqlens != nullptr) ? cu_seqlens[i_n + 1] : (int64_t(bos) + T_max);
    // This executes on graph replay as well. Do not silently truncate corrupt
    // metadata: it would hide incorrect output/state and can overflow MTP smem.
    if (bos < 0 || eos < bos || eos - bos > T_max) {
        __trap();
        return;
    }
    const int T = static_cast<int>(eos - bos);
    // Padded requests have no token or state access, including accepted/slot loads.
    if (T == 0) return;

    int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    int lane_id = threadIdx.x;
    int tid = warp_id * 32 + lane_id;
    int k_idx = lane_id * 4;

    int h0_idx = -1;
    if constexpr (use_initial_state) {
        if (h0_indices != nullptr) {
            h0_idx = h0_indices[i_n];
        }
    }

    if constexpr (use_initial_state) {
        if (is_sglang ? (h0_idx < 0) : (h0_idx <= 0)) {
            #pragma unroll
            for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
                int i_hv = hv_base + hv_step;
                if (tid < BV) {
                    o[bos * HV * 128 + i_hv * 128 + v_base + tid] = static_cast<scalar_t>(0.0f);
                }
            }
            return;
        }
    }

    const scalar_t* p_mixed = mixed_qkv + bos * stride_qkv;
    int q_off = i_h * K;
    int k_off = (H * K) + i_h * K;

    // --- Shared memory layout ---
    // H-staging: STAGE_NUM * TASKS_PER_STAGE * WARP_NUM * 128 elements of state_t
    // (fp32: 4*1*8*128*4B = 16384B; bf16: 4*2*8*128*2B = 16384B — identical)
    extern __shared__ __align__(16) unsigned char smem_bytes[];
    const pla::KLastSharedLayout layout(
        STAGE_NUM, K, WARP_NUM, BV, HV_PER_CTA, IS_KDA, false, T_max, sizeof(scalar_t));
    state_t* smem_h_typed = reinterpret_cast<state_t*>(smem_bytes + layout.h);
    float* smem_q = reinterpret_cast<float*>(smem_bytes + layout.q);
    float* smem_k = reinterpret_cast<float*>(smem_bytes + layout.k);
    scalar_t* smem_bv = reinterpret_cast<scalar_t*>(smem_bytes + layout.bv);
    float* smem_reduction_q = reinterpret_cast<float*>(smem_bytes + layout.reduction_q);
    float* smem_reduction_k = reinterpret_cast<float*>(smem_bytes + layout.reduction_k);
    float* smem_b_beta = reinterpret_cast<float*>(smem_bytes + layout.beta);
    float* smem_bg = reinterpret_cast<float*>(smem_bytes + layout.bg);
    float* smem_bg_scalar = smem_bg;

    state_t* h0_cta_base = nullptr;
    if constexpr (use_initial_state) {
        h0_cta_base = h0_source + (int64_t)h0_idx * h0_slot_stride + hv_base * 16384 + v_base * 128 + k_idx;
    }

    float b_q_val = static_cast<float>(p_mixed[q_off + tid]);
    float b_k_val = static_cast<float>(p_mixed[k_off + tid]);

    // --- V prefetch (identical for both paths) ---
    {
        constexpr int DIM_C = BV;
        constexpr int64_t tensor_stride_w = DIM_C * sizeof(scalar_t);
        constexpr int64_t tensor_stride_n = 128 * sizeof(scalar_t);
        const scalar_t* v_ptr = p_mixed + (2 * H * K) + hv_base * V + v_base;
#ifdef PLA_USE_AIU
        if (warp_id == 0) {
            ppu_aiu_load<DIM_C, 1, HV_PER_CTA, DIM_C, 1, HV_PER_CTA, tensor_stride_w, tensor_stride_n, false>(
                smem_bv, v_ptr, 0, 0, 0);
        }
#else
        constexpr int V_VEC_LEN = sizeof(float4) / sizeof(scalar_t);
        const scalar_t* v_ptr_t = v_ptr + (warp_id * 128) + lane_id * V_VEC_LEN;
        uint32_t smem_bv_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_bv + warp_id * BV + lane_id * V_VEC_LEN));
        if (warp_id < HV_PER_CTA && lane_id < (BV / V_VEC_LEN)) {
            asm volatile(
                "{\n  .reg .pred p;\n  ppu.cmpp.ne.b32 p, %0, 0;\n"
                "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n}\n"
                ::"r"(1), "r"(smem_bv_addr), "l"(v_ptr_t), "n"(16));
        }
#endif
    }

    // --- Head-dependent init (KDA gating / GDN scalar) ---
    if constexpr (IS_KDA) {
        #pragma unroll
        if (warp_id < HV_PER_CTA) {
            int i_hv = hv_base + warp_id;
            float b_A_log = A_log[i_hv];
            uint2 a_raw = *reinterpret_cast<const uint2*>(a + bos * stride_a + i_hv * 128 + k_idx);
            uint2 dt_raw = *reinterpret_cast<const uint2*>(dt_bias + i_hv * 128 + k_idx);
            float2 a01 = TypeConverter<scalar_t>::convert(a_raw.x);
            float2 a23 = TypeConverter<scalar_t>::convert(a_raw.y);
            float2 dt01 = TypeConverter<scalar_t>::convert(dt_raw.x);
            float2 dt23 = TypeConverter<scalar_t>::convert(dt_raw.y);
            *reinterpret_cast<float4*>(smem_bg + warp_id * K + k_idx) = make_float4(
                compute_bg(a01.x, dt01.x, b_A_log, softplus_beta, softplus_threshold),
                compute_bg(a01.y, dt01.y, b_A_log, softplus_beta, softplus_threshold),
                compute_bg(a23.x, dt23.x, b_A_log, softplus_beta, softplus_threshold),
                compute_bg(a23.y, dt23.y, b_A_log, softplus_beta, softplus_threshold));
            if (lane_id == 0) {
                float b_val = static_cast<float>(b[bos * stride_b + i_hv]);
                #if defined(__HGGC_ARCH__)
                    smem_b_beta[warp_id] = __ppu_sgmdf(b_val);
                #else
                    smem_b_beta[warp_id] = 1.0f / (1.0f + __expf(-b_val));
                #endif
            }
        }
    } else {
        if (warp_id < HV_PER_CTA) {
            int i_hv = hv_base + warp_id;
            if (lane_id == 0) {
                float b_A_log = A_log[i_hv];
                float b_val = static_cast<float>(b[bos * stride_b + i_hv]);
                #if defined(__HGGC_ARCH__)
                    smem_b_beta[warp_id] = __ppu_sgmdf(b_val);
                #else
                    smem_b_beta[warp_id] = 1.0f / (1.0f + __expf(-b_val));
                #endif
                float a_val = static_cast<float>(a[bos * stride_a + i_hv]);
                float dt_val = static_cast<float>(dt_bias[i_hv]);
                smem_bg_scalar[warp_id] = compute_bg(a_val, dt_val, b_A_log, softplus_beta, softplus_threshold);
            }
        }
    }

    // --- Q/K normalization ---
    if constexpr (WARP_NUM != 4) {
        static_assert(WARP_NUM >= 2, "warp0/warp1 split requires WARP_NUM >= 2");
        if (warp_id == 0) {
            uint2 q_raw = *reinterpret_cast<const uint2*>(p_mixed + q_off + k_idx);
            float2 q01 = TypeConverter<scalar_t>::convert(q_raw.x);
            float2 q23 = TypeConverter<scalar_t>::convert(q_raw.y);
            if (use_qk_l2norm) {
                float q_sq = q01.x*q01.x + q01.y*q01.y + q23.x*q23.x + q23.y*q23.y;
                #pragma unroll
                for (int off = 16; off > 0; off /= 2) q_sq += __shfl_down_sync(0xffffffff, q_sq, off);
                q_sq = __shfl_sync(0xffffffff, q_sq, 0);
                float n = rsqrtf(q_sq + 1e-6f) * scale;
                q01.x *= n; q01.y *= n; q23.x *= n; q23.y *= n;
            } else { q01.x *= scale; q01.y *= scale; q23.x *= scale; q23.y *= scale; }
            *reinterpret_cast<float4*>(&smem_q[k_idx]) = make_float4(q01.x, q01.y, q23.x, q23.y);
        } else if (warp_id == 1) {
            uint2 k_raw = *reinterpret_cast<const uint2*>(p_mixed + k_off + k_idx);
            float2 k01 = TypeConverter<scalar_t>::convert(k_raw.x);
            float2 k23 = TypeConverter<scalar_t>::convert(k_raw.y);
            if (use_qk_l2norm) {
                float k_sq = k01.x*k01.x + k01.y*k01.y + k23.x*k23.x + k23.y*k23.y;
                #pragma unroll
                for (int off = 16; off > 0; off /= 2) k_sq += __shfl_down_sync(0xffffffff, k_sq, off);
                k_sq = __shfl_sync(0xffffffff, k_sq, 0);
                float n = rsqrtf(k_sq + 1e-6f);
                k01.x *= n; k01.y *= n; k23.x *= n; k23.y *= n;
            }
            *reinterpret_cast<float4*>(&smem_k[k_idx]) = make_float4(k01.x, k01.y, k23.x, k23.y);
        }
    } else {
        if (use_qk_l2norm) {
            float q_sq = b_q_val * b_q_val, k_sq = b_k_val * b_k_val;
            float sum_q, sum_k;
            block_reduce_sum2<WARP_NUM>(q_sq, k_sq, smem_reduction_q, smem_reduction_k, lane_id, warp_id, &sum_q, &sum_k);
            b_q_val *= rsqrtf(sum_q + 1e-6f);
            b_k_val *= rsqrtf(sum_k + 1e-6f);
        }
        b_q_val *= scale;
        smem_q[tid] = b_q_val;
        smem_k[tid] = b_k_val;
    }

    __syncthreads();

    float4 q_vec = *reinterpret_cast<const float4*>(&smem_q[k_idx]);
    float4 k_vec = *reinterpret_cast<const float4*>(&smem_k[k_idx]);

    // =========================================================================
    // PIPELINE — split by state_t specialization
    // =========================================================================

    if constexpr (sizeof(state_t) == 4) {
        // =================================================================
        // FP32 STATE: 1 task per stage, 16B cp.async.cg per thread
        // =================================================================
        uint32_t smem_h_ptr_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_h_typed[warp_id * 128 + k_idx]));

        auto issue_fp32 = [&](int task, int stage, bool pred) {
            int hv_s = task / V_ITERS, v_i = task % V_ITERS;
            state_t* ptr = h0_cta_base + hv_s * 16384 + (v_i * WARP_NUM + warp_id) * 128;
            uint32_t dst = smem_h_ptr_base + (stage * WARP_NUM * 128 * sizeof(state_t));
            asm volatile(
                "{\n  .reg .pred p;\n  ppu.cmpp.ne.b32 p, %0, 0;\n"
                "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n}\n"
                ::"r"((int)pred), "r"(dst), "l"(ptr), "n"(16));
        };

#ifdef PLA_USE_AIU
        {
            constexpr int64_t tsw = 128 * sizeof(state_t);
            constexpr int64_t tsn = 128 * 128 * sizeof(state_t);
            state_t* base = h0_source + (int64_t)h0_idx * h0_slot_stride + hv_base * 16384 + v_base * 128;
            if (warp_id == 0)
                ppu_aiu_load<128, STAGE_NUM*WARP_NUM, HV_PER_CTA, 128, BV, HV_PER_CTA, tsw, tsn>(
                    smem_h_typed, base, 0, 0, 0);
        }
#endif

        if constexpr (use_initial_state) {
            #pragma unroll
            for (int i = 0; i < STAGE_NUM; ++i) {
                #ifndef PLA_USE_AIU
                issue_fp32(i, i, i < TOTAL_TASKS);
                #endif
                cp_async_commit();
            }
        }

        int task_idx = 0;
        #pragma unroll
        for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
            float b_beta = smem_b_beta[hv_step];
            float4 bg_vec; float bg_sc;
            if constexpr (IS_KDA) { bg_vec = *reinterpret_cast<const float4*>(smem_bg + hv_step * K + k_idx); }
            else { bg_sc = smem_bg_scalar[hv_step]; }
            int i_hv = hv_base + hv_step;

            float4 h_val;
            #pragma unroll
            for (int v_i = 0; v_i < V_ITERS; ++v_i) {
                int stage = task_idx % STAGE_NUM;

                if constexpr (use_initial_state) {
                    cp_async_wait<STAGE_NUM - 1>();
                    if (hv_step == 0 && v_i == 0) __ppu_barrier_sync_defer(0, WARP_NUM*32, 1);
                    h_val = *reinterpret_cast<float4*>(&smem_h_typed[stage * WARP_NUM * 128 + warp_id * 128 + k_idx]);
                }

                float v_val = static_cast<float>(smem_bv[hv_step * BV + warp_id + v_i * WARP_NUM]);

                if constexpr (IS_KDA) { h_val.x *= bg_vec.x; h_val.y *= bg_vec.y; h_val.z *= bg_vec.z; h_val.w *= bg_vec.w; }
                else { h_val.x *= bg_sc; h_val.y *= bg_sc; h_val.z *= bg_sc; h_val.w *= bg_sc; }

                float v_delta = h_val.x*k_vec.x + h_val.y*k_vec.y + h_val.z*k_vec.z + h_val.w*k_vec.w;
                #pragma unroll
                for (int off = 16; off > 0; off /= 2) v_delta += __shfl_down_sync(0xffffffff, v_delta, off);
                v_delta = __shfl_sync(0xffffffff, v_delta, 0);

                float v_new = (v_val - v_delta) * b_beta;

                int next = task_idx + STAGE_NUM;
                issue_fp32(next, next % STAGE_NUM, next < TOTAL_TASKS);
                cp_async_commit();

                h_val.x += k_vec.x * v_new; h_val.y += k_vec.y * v_new;
                h_val.z += k_vec.z * v_new; h_val.w += k_vec.w * v_new;

                if constexpr (use_initial_state) {
                    #ifdef __HGGC_ARCH__
                        *(Global v4float*)(h0_cta_base + hv_step * 16384 + (warp_id + v_i * WARP_NUM) * 128) = *(v4float*)&h_val;
                    #else
                        *reinterpret_cast<float4*>(h0_cta_base + hv_step * 16384 + (warp_id + v_i * WARP_NUM) * 128) = h_val;
                    #endif
                }

                float o_val = h_val.x*q_vec.x + h_val.y*q_vec.y + h_val.z*q_vec.z + h_val.w*q_vec.w;
                #pragma unroll
                for (int off = 16; off > 0; off /= 2) o_val += __shfl_down_sync(0xffffffff, o_val, off);
                if (lane_id == 0) o[bos * HV * 128 + i_hv * 128 + v_base + warp_id + v_i * WARP_NUM] = static_cast<scalar_t>(o_val);

                task_idx++;
            }
        }
        cp_async_wait<0>();

    } else {
        // =================================================================
        // BF16 STATE: the two half-warps process two state rows in parallel.
        // Each lane owns 8 adjacent K elements and transfers 16B per stage.
        // Half-warp reductions need four crosslane steps instead of five full-
        // warp steps per row, while retaining the same bytes and arithmetic.
        // =================================================================
        int half_warp = lane_id >> 4;
        int half_lane = lane_id & 15;
        int half_k_idx = half_lane * 8;
        state_t* bf16_h0_cta_base = nullptr;
        if constexpr (use_initial_state) {
            bf16_h0_cta_base = h0_source + (int64_t)h0_idx * h0_slot_stride
                + hv_base * 16384 + v_base * 128 + half_k_idx;
        }

        float4 q_vec_lo = *reinterpret_cast<const float4*>(&smem_q[half_k_idx]);
        float4 q_vec_hi = *reinterpret_cast<const float4*>(&smem_q[half_k_idx + 4]);
        float4 k_vec_lo = *reinterpret_cast<const float4*>(&smem_k[half_k_idx]);
        float4 k_vec_hi = *reinterpret_cast<const float4*>(&smem_k[half_k_idx + 4]);

        // One 16B copy per lane. The two half-warps target adjacent task slots.
        auto issue_pair = [&](int task_base, int stage) {
            int task = task_base + half_warp;
            int hv_s = task / V_ITERS, v_i = task % V_ITERS;
            state_t* ptr = bf16_h0_cta_base + hv_s * 16384 + (v_i * WARP_NUM + warp_id) * 128;
            uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(
                &smem_h_typed[(stage * 2 + half_warp) * WARP_NUM * 128 + warp_id * 128 + half_k_idx]));
            asm volatile(
                "{\n  .reg .pred p;\n  ppu.cmpp.ne.b32 p, %0, 0;\n"
                "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n}\n"
                ::"r"((int)(task < TOTAL_TASKS)), "r"(dst), "l"(ptr), "n"(16));
        };

        // Read one half-warp task from smem -> two float4 values.
        auto read_bf16 = [&](int stage, float4& h_lo, float4& h_hi) {
            int smem_slot = stage * 2 + half_warp;
            uint4 raw = *reinterpret_cast<uint4*>(
                &smem_h_typed[smem_slot * WARP_NUM * 128 + warp_id * 128 + half_k_idx]);
            __ppu_bfloat162 p0 = *reinterpret_cast<__ppu_bfloat162*>(&raw.x);
            __ppu_bfloat162 p1 = *reinterpret_cast<__ppu_bfloat162*>(&raw.y);
            __ppu_bfloat162 p2 = *reinterpret_cast<__ppu_bfloat162*>(&raw.z);
            __ppu_bfloat162 p3 = *reinterpret_cast<__ppu_bfloat162*>(&raw.w);
            float2 f01 = __bfloat1622float2(p0);
            float2 f23 = __bfloat1622float2(p1);
            float2 f45 = __bfloat1622float2(p2);
            float2 f67 = __bfloat1622float2(p3);
            h_lo = make_float4(f01.x, f01.y, f23.x, f23.y);
            h_hi = make_float4(f45.x, f45.y, f67.x, f67.y);
        };

        // Store all 8 BF16 elements as one aligned 16B transaction.
        auto store_bf16 = [&](float4 h_lo, float4 h_hi, int hv_step, int cur_v_i) {
            __ppu_bfloat162 p0 = __float22bfloat162_rn(make_float2(h_lo.x, h_lo.y));
            __ppu_bfloat162 p1 = __float22bfloat162_rn(make_float2(h_lo.z, h_lo.w));
            __ppu_bfloat162 p2 = __float22bfloat162_rn(make_float2(h_hi.x, h_hi.y));
            __ppu_bfloat162 p3 = __float22bfloat162_rn(make_float2(h_hi.z, h_hi.w));
            uint4 packed = make_uint4(
                *reinterpret_cast<uint32_t*>(&p0), *reinterpret_cast<uint32_t*>(&p1),
                *reinterpret_cast<uint32_t*>(&p2), *reinterpret_cast<uint32_t*>(&p3));
            #ifdef __HGGC_ARCH__
                *(Global v4uint*)(bf16_h0_cta_base + hv_step * 16384 + (warp_id + cur_v_i * WARP_NUM) * 128)
                    = *reinterpret_cast<v4uint*>(&packed);
            #else
                *reinterpret_cast<uint4*>(bf16_h0_cta_base + hv_step * 16384 + (warp_id + cur_v_i * WARP_NUM) * 128) = packed;
            #endif
        };

        // --- Prologue: fill STAGE_NUM stages, each with 2 tasks ---
        // This bf16 h0 fill stays on cp.async on every arch: issue_pair() also
        // converts and repacks, so there is no plain tile copy for AIU to take
        // over. (AIU itself does handle 16-bit elements -- see ppu_aiu_load.)
        if constexpr (use_initial_state) {
            #pragma unroll
            for (int s = 0; s < STAGE_NUM; ++s) {
                issue_pair(s * 2, s);
                cp_async_commit();
            }
        }

        // --- Main loop: 2 tasks per wait ---
        int task_idx = 0;
        #pragma unroll
        for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
            float b_beta = smem_b_beta[hv_step];
            float4 bg_vec_lo, bg_vec_hi; float bg_sc;
            if constexpr (IS_KDA) {
                bg_vec_lo = *reinterpret_cast<const float4*>(smem_bg + hv_step * K + half_k_idx);
                bg_vec_hi = *reinterpret_cast<const float4*>(smem_bg + hv_step * K + half_k_idx + 4);
            }
            else { bg_sc = smem_bg_scalar[hv_step]; }
            int i_hv = hv_base + hv_step;

            #pragma unroll
            for (int v_i = 0; v_i < V_ITERS; v_i += 2) {
                int stage = (task_idx / 2) % STAGE_NUM;

                if constexpr (use_initial_state) {
                    cp_async_wait<STAGE_NUM - 1>();
                    if (hv_step == 0 && v_i == 0) __ppu_barrier_sync_defer(0, WARP_NUM*32, 1);
                }

                // Each half-warp handles one row, so the pair advances together
                // with four shuffle steps for each reduction.
                int cur_v_i = v_i + half_warp;
                float4 h_lo, h_hi;
                if constexpr (use_initial_state) {
                    read_bf16(stage, h_lo, h_hi);
                }

                float v_val = static_cast<float>(smem_bv[hv_step * BV + warp_id + cur_v_i * WARP_NUM]);

                if constexpr (IS_KDA) {
                    h_lo.x *= bg_vec_lo.x; h_lo.y *= bg_vec_lo.y;
                    h_lo.z *= bg_vec_lo.z; h_lo.w *= bg_vec_lo.w;
                    h_hi.x *= bg_vec_hi.x; h_hi.y *= bg_vec_hi.y;
                    h_hi.z *= bg_vec_hi.z; h_hi.w *= bg_vec_hi.w;
                } else {
                    h_lo.x *= bg_sc; h_lo.y *= bg_sc; h_lo.z *= bg_sc; h_lo.w *= bg_sc;
                    h_hi.x *= bg_sc; h_hi.y *= bg_sc; h_hi.z *= bg_sc; h_hi.w *= bg_sc;
                }

                float v_delta = h_lo.x*k_vec_lo.x + h_lo.y*k_vec_lo.y
                              + h_lo.z*k_vec_lo.z + h_lo.w*k_vec_lo.w
                              + h_hi.x*k_vec_hi.x + h_hi.y*k_vec_hi.y
                              + h_hi.z*k_vec_hi.z + h_hi.w*k_vec_hi.w;
                #pragma unroll
                for (int off = 8; off > 0; off /= 2)
                    v_delta += __shfl_xor_sync(0xffffffff, v_delta, off, 16);

                float v_new = (v_val - v_delta) * b_beta;
                h_lo.x += k_vec_lo.x * v_new; h_lo.y += k_vec_lo.y * v_new;
                h_lo.z += k_vec_lo.z * v_new; h_lo.w += k_vec_lo.w * v_new;
                h_hi.x += k_vec_hi.x * v_new; h_hi.y += k_vec_hi.y * v_new;
                h_hi.z += k_vec_hi.z * v_new; h_hi.w += k_vec_hi.w * v_new;

                if constexpr (use_initial_state) {
                    store_bf16(h_lo, h_hi, hv_step, cur_v_i);
                }

                float o_val = h_lo.x*q_vec_lo.x + h_lo.y*q_vec_lo.y
                            + h_lo.z*q_vec_lo.z + h_lo.w*q_vec_lo.w
                            + h_hi.x*q_vec_hi.x + h_hi.y*q_vec_hi.y
                            + h_hi.z*q_vec_hi.z + h_hi.w*q_vec_hi.w;
                #pragma unroll
                for (int off = 8; off > 0; off /= 2)
                    o_val += __shfl_down_sync(0xffffffff, o_val, off, 16);
                if (half_lane == 0)
                    o[bos * HV * 128 + i_hv * 128 + v_base + warp_id + cur_v_i * WARP_NUM] = static_cast<scalar_t>(o_val);

                task_idx += 2;

                // Issue next pair into the now-free stage.
                if constexpr (use_initial_state) {
                    int next_base = task_idx + (STAGE_NUM - 1) * 2;
                    issue_pair(next_base, stage);
                    cp_async_commit();
                }
            }
        }
        cp_async_wait<0>();
    }
}
