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
#include <type_traits>

#include "../../pla_device_helpers.cuh"
#include "k_last_shared_layout.h"

// Fast-decode (T=1) pipeline structure:
//   * State staging: per-warp cp.async rolling pipeline (every warp
//     prefetches its OWN staging rows smem_h[warp_id*128 + k_idx] and
//     refills stage (task+SN) right after consuming the task).
//   * The prologue is barrier-free on the GDN path: beta/bg are recomputed
//     per warp into registers (L2/LLC hits), and warp0 alone normalizes q/k
//     into smem — the (WN-1) sibling copies of the l2norm chain were pure
//     issue-bandwidth waste once the kernel turned out to be
//     instruction-budget bound (453M warp instructions / (39 CU x 5.58
//     inst/cycle) == the measured 2.08M cycles; DRAM 85% just rides along).
//     warp0's q/k publication is ordered by the task-0 barrier everyone
//     hits anyway behind the first state wait.
//   * The tiny V tile rides the state cp.async pipeline inside group 0
//     (issued by every warp's lanes into its OWN group, so no cross-warp
//     publication is needed at all).
//   * Issue-budget discipline (Candidate D): per-warp base pointers with
//     fully-unrolled constant offsets (no per-task v.madl/v.madw address
//     MADs), butterfly (XOR) warp reductions (no re-broadcast shuffle),
//     and a full-prefetch specialization (TOTAL_TASKS <= STAGE_NUM, e.g.
//     {32,4,8}/{64,8,8}) that drops the empty commit_group churn and waits
//     exactly on group (task_idx).
//   * KDA keeps its per-K bg vector in smem (published by the same task-0
//     barrier) and, when STAGE data is AIU-bulked (macro undefined), the
//     warp0 bulk + deferred barrier path is preserved.
#define PLA_KLAST_STATE_CPASYNC 1


// dt_bias is read as fp32 only on the KDA MTP-verify path: there the per-K decay
// is recomputed for every draft token, so rounding dt_bias to the activation dtype
// compounds through the recurrence. Pure decode evaluates the gate once, so it
// keeps dt_bias at scalar_t -- which also halves that load. GDN is scalar_t
// throughout, where dt_bias is one scalar per head. Deriving the type here rather
// than adding a template parameter leaves the instantiation count unchanged.
template <bool IS_KDA, bool MTP_VERIFY, typename scalar_t>
using dt_bias_t = std::conditional_t<IS_KDA && MTP_VERIFY, float, scalar_t>;

// State is stored natively as FP32/BF16; the recurrence is always computed in FP32 registers.
template <typename state_t>
__device__ __forceinline__ float4 k_last_load_state4(const state_t* ptr) {
    if constexpr (std::is_same_v<state_t, float>) {
        return *reinterpret_cast<const float4*>(ptr);
    } else {
        uint2 raw = *reinterpret_cast<const uint2*>(ptr);
        float2 lo = TypeConverter<state_t>::convert(raw.x);
        float2 hi = TypeConverter<state_t>::convert(raw.y);
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    }
}

template <typename state_t>
__device__ __forceinline__ void k_last_store_state4(state_t* ptr, float4 value) {
    if constexpr (std::is_same_v<state_t, float>) {
#ifdef __HGGC_ARCH__
        *(Global v4float*)ptr = *reinterpret_cast<v4float*>(&value);
#else
        *reinterpret_cast<float4*>(ptr) = value;
#endif
    } else {
        __ppu_bfloat162 lo = __float22bfloat162_rn(make_float2(value.x, value.y));
        __ppu_bfloat162 hi = __float22bfloat162_rn(make_float2(value.z, value.w));
        uint2 packed = make_uint2(*reinterpret_cast<uint32_t*>(&lo),
                                 *reinterpret_cast<uint32_t*>(&hi));
#ifdef __HGGC_ARCH__
        *(Global v2uint*)ptr = *reinterpret_cast<v2uint*>(&packed);
#else
        *reinterpret_cast<uint2*>(ptr) = packed;
#endif
    }
}

template <typename state_t>
__device__ __forceinline__ void k_last_prefetch_state4(
    uint32_t dst, const state_t* src, bool pred) {
    if constexpr (std::is_same_v<state_t, float>) {
        asm volatile(
            "{\n .reg .pred p;\n ppu.cmpp.ne.b32 p, %0, 0;\n"
            " @p ppu.cp.async.cg.shared.global [%1], [%2], 16;\n}\n"
            :: "r"((int)pred), "r"(dst), "l"(src));
    } else {
        // Each lane moves 4 BF16 values on its own, avoiding cross-lane async read dependencies.
        asm volatile(
            "{\n .reg .pred p;\n ppu.cmpp.ne.b32 p, %0, 0;\n"
            " @p ppu.cp.async.ca.shared.global [%1], [%2], 8;\n}\n"
            :: "r"((int)pred), "r"(dst), "l"(src));
    }
}

template <typename alog_t, typename scalar_t, int WARP_NUM, int BV, int HV_PER_CTA, int STAGE_NUM, bool IS_KDA, bool USE_LOWER_BOUND, bool MTP_VERIFY, bool use_initial_state=true, typename state_t=float>
__global__ void fused_sigmoid_gating_delta_kernel(
    const alog_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const dt_bias_t<IS_KDA, MTP_VERIFY, scalar_t>* __restrict__ dt_bias,
    const float softplus_beta_or_lower_bound,
    const float softplus_threshold,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ o,
    state_t* __restrict__ h0_source,
    const int32_t* __restrict__ cu_seqlens,
    state_t* __restrict__ intermediate_states_buffer,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ intermediate_state_indices,
    const int cache_steps,
    const int64_t* __restrict__ retrieve_parent_token_ptr,
    const int stride_retrieve_parent_token_seq,
    const int stride_retrieve_parent_token_token,
    const float scale,
    const int T_max,
    const int stride_a, const int stride_q, const int stride_k, const int stride_v, const int stride_b,
    const int64_t h0_slot_stride,
    const int B, const int H, const int HV, const int K, const int V,
    const bool cache_intermediate_states,
    const bool has_eagle_tree,
    const bool use_qk_l2norm,
    const bool disable_state_update,
    // vLLM spec decode passes a 2D [N, S] slot table through h0_indices, whose
    // row stride is stride_h0_indices_seq (0 selects the 1D sglang layout);
    // num_accepted_tokens then picks the starting column for each request.
    const int32_t* __restrict__ num_accepted_tokens,
    const int stride_h0_indices_seq,
    // The two frameworks disagree on which slot id means "invalid": sglang's
    // MambaPool allocates an extra scratch row so slot 0 is valid and -1 is the
    // pad id, while vLLM's BlockPool reserves block 0 as null_block.
    const bool is_sglang)
{
    constexpr int V_ITERS = BV / WARP_NUM;
    constexpr int TOTAL_TASKS = HV_PER_CTA * V_ITERS;

    int i_n;
    int i_h;
    int hv_per_h;
    if constexpr (USE_LOWER_BOUND) {
        i_n = blockIdx.z / H;
        i_h = blockIdx.z % H;
        hv_per_h = HV / H;
    } else if ((H & (H - 1)) == 0) {
        i_n = divide_by_power_of_two(blockIdx.z, H);
        i_h = blockIdx.z & (H - 1);
        hv_per_h = divide_by_power_of_two(HV, H);
    } else {
        // H is guaranteed positive by the host; this branch is uniform across the CTA and independent of the gate type.
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

    // Empty requests preserve their state and cache. Uniform across the CTA,
    // and before any asynchronous issue or block-wide synchronization.
    if (T <= 0) return;

    int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    int lane_id = threadIdx.x;
    int tid = warp_id * 32 + lane_id;
    int k_idx = lane_id * 4;

    int h0_idx = -1;
    if constexpr (use_initial_state) {
        if (h0_indices != nullptr) {
            if (stride_h0_indices_seq > 0) {
                // vLLM spec decode: resume from the last accepted token's slot.
                const int accepted = (num_accepted_tokens != nullptr) ? num_accepted_tokens[i_n] : 1;
                // Clamp before subtracting so INT32_MIN has the same padding
                // semantics without signed overflow. Accepted counts refer to
                // prior-step columns, not the current sequence length.
                const int i_t = accepted > 0 ? accepted - 1 : 0;
                h0_idx = h0_indices[i_n * stride_h0_indices_seq + i_t];
            } else {
                h0_idx = h0_indices[i_n];
            }
        }
    }

    // Invalid state slot: emit zeros and bail out, matching the Triton kernels.
    // h0_idx only depends on i_n, so every CTA of a request agrees on this and
    // the early return can never desynchronise a later __syncthreads().
    if constexpr (use_initial_state) {
        if (h0_indices != nullptr && (is_sglang ? (h0_idx < 0) : (h0_idx <= 0))) {
            for (int t = 0; t < T; ++t) {
                #pragma unroll
                for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
                    if (tid < BV) {
                        o[(bos + t) * HV * 128 + (hv_base + hv_step) * 128 + v_base + tid] =
                            static_cast<scalar_t>(0.0f);
                    }
                }
            }
            return;
        }
    }

    int cache_idx = -1;
    if constexpr (MTP_VERIFY) {
        if (cache_intermediate_states && intermediate_state_indices != nullptr) {
            cache_idx = intermediate_state_indices[i_n];
        }
    }

    extern __shared__ __align__(16) unsigned char smem_bytes[];
    const pla::KLastSharedLayout layout(
        STAGE_NUM, K, WARP_NUM, BV, HV_PER_CTA, IS_KDA, MTP_VERIFY, T_max, sizeof(scalar_t));
    state_t* smem_h_flat = reinterpret_cast<state_t*>(smem_bytes + layout.h);
    float* smem_q = reinterpret_cast<float*>(smem_bytes + layout.q);
    float* smem_k = reinterpret_cast<float*>(smem_bytes + layout.k);
    scalar_t* smem_bv = reinterpret_cast<scalar_t*>(smem_bytes + layout.bv);
    float* smem_reduction_q = reinterpret_cast<float*>(smem_bytes + layout.reduction_q);
    float* smem_reduction_k = reinterpret_cast<float*>(smem_bytes + layout.reduction_k);
    float* smem_b_beta = reinterpret_cast<float*>(smem_bytes + layout.beta);
    float* smem_bg = reinterpret_cast<float*>(smem_bytes + layout.bg);
    float* smem_bg_scalar = smem_bg;
    scalar_t* smem_v_mtp = reinterpret_cast<scalar_t*>(smem_bytes + layout.v_mtp);
    float* smem_q_mtp = reinterpret_cast<float*>(smem_bytes + layout.q_mtp);
    float* smem_k_mtp = reinterpret_cast<float*>(smem_bytes + layout.k_mtp);
    scalar_t* smem_q_raw = reinterpret_cast<scalar_t*>(smem_bytes + layout.q_raw);
    scalar_t* smem_k_raw = reinterpret_cast<scalar_t*>(smem_bytes + layout.k_raw);
    float* smem_beta_mtp = reinterpret_cast<float*>(smem_bytes + layout.beta_mtp);
    float* smem_bg_mtp = reinterpret_cast<float*>(smem_bytes + layout.bg_mtp);

    // Streamlined Base Offset bypassing deep inner-loop Integer Multipliers
    state_t* h0_cta_base = nullptr;
    // Per-warp gmem state row (v_i = 0); with the task loops fully unrolled,
    // every in-loop offset below folds to base+const instead of a fresh
    // v.madl/v.madw address MAD — pcsamp parked ~14% of not_selected
    // samples on exactly those address MADs (issue-bandwidth bound kernel).
    state_t* h0_row0 = nullptr;
    // use_initial_state = use_initial_state && h0_idx >= 0;
    if constexpr (use_initial_state) {
        h0_cta_base = h0_source + (int64_t)h0_idx * h0_slot_stride + hv_base * 16384 + v_base * 128 + k_idx;
        h0_row0 = h0_cta_base + warp_id * 128;
    }

    uint32_t smem_h_ptr_per_warp = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_h_flat[warp_id * 128 + k_idx]));
    auto issue_async_stage = [&](int task, int stage, bool pred_guard) {
        int hv_step = task / V_ITERS;
        int v_i = task % V_ITERS;
        // float* h0_ptr = h0_cta_base + hv_step * 16384 + (warp_id * V_ITERS + v_i) * 128;
        state_t* h0_ptr = h0_row0 + hv_step * 16384 + v_i * (WARP_NUM * 128);
        uint32_t smem_h_ptr = smem_h_ptr_per_warp + (stage * WARP_NUM * 128 * sizeof(state_t));
        k_last_prefetch_state4(smem_h_ptr, h0_ptr, pred_guard);
    };


    // NOTE: the fast-decode pipeline no longer preloads V through the AIU
    // here. The V tile (BV*HV_PER_CTA scalar_t, a few hundred bytes) rides the
    // state cp.async pipeline inside group 0 (see the prologue loop below),
    // which removes both the AIU command and the deferred block barrier from
    // the consume side. The MTP_VERIFY pipeline keeps its own V preload.

    // issue prefetch of H0
    auto issue_aiu_ld = [&](int stage, int coord_w, int coord_n) {
        constexpr int DIM_C = 128;  // K
        constexpr int DIM_W = BV;
        constexpr int DIM_N = HV_PER_CTA;
        constexpr int CUBE_C = DIM_C;
        constexpr int CUBE_W = STAGE_NUM * WARP_NUM;
        constexpr int CUBE_W_S1 = WARP_NUM;
        constexpr int CUBE_N = HV_PER_CTA;
        constexpr int64_t tensor_stride_w = DIM_C * sizeof(state_t);
        constexpr int64_t tensor_stride_n = 128 * 128 * sizeof(state_t); // K * V
        state_t* h0_offset_v_head_base = h0_source + (int64_t)h0_idx * h0_slot_stride + hv_base * 16384 + v_base * 128;

        // preload V input
        if (warp_id == 0) {
            if (stage == 1) {
                ppu_aiu_load<CUBE_C, CUBE_W_S1, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w, tensor_stride_n, sizeof(state_t) == 4>(
                    smem_h_flat,
                    h0_offset_v_head_base,
                    0, coord_w, coord_n);
            } else {
                ppu_aiu_load<CUBE_C, CUBE_W, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w, tensor_stride_n, sizeof(state_t) == 4>(
                    smem_h_flat,
                    h0_offset_v_head_base,
                    0, coord_w, coord_n);
            }
        }
    };

    // =========================================================================
    // FAST DECODING PIPELINE (MTP_VERIFY = false, T = 1)
    // =========================================================================
    if constexpr (!MTP_VERIFY) {
#if defined(PLA_USE_AIU) && !defined(PLA_KLAST_STATE_CPASYNC)
        // issue prefetch of H0 for whole STAGE of data in V_ITERS, and commit in group0
        issue_aiu_ld(STAGE_NUM, 0, 0);
#endif
        // 1. Issue exactly Stage 0 exclusively to jump-start the HBM cache.
        //    The V tile rides the SAME group 0 as state stage 0: a handful of
        //    lanes issue 16B-granular cp.async covering BV*HV_PER_CTA, and the
        //    warp-collective commit folds them into the group drained by the
        //    first cp_async_wait of the consume loop — no AIU command, no
        //    deferred block barrier.
        constexpr int V_ELEMS = BV * HV_PER_CTA;
        constexpr int V_LANES = (V_ELEMS * (int)sizeof(scalar_t)) / 16;
        if constexpr (use_initial_state) {
            #pragma unroll
            for (int i = 0; i < STAGE_NUM; ++i) {
#if !defined(PLA_USE_AIU) || defined(PLA_KLAST_STATE_CPASYNC)
                if (i == 0 && lane_id < V_LANES) {
                    int v_off = lane_id * (16 / (int)sizeof(scalar_t));
                    int v_row = v_off / BV;
                    int v_col = v_off % BV;
                    const scalar_t* v_src = v + bos * stride_v + (hv_base + v_row) * 128 + v_base + v_col;
                    uint32_t v_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_bv + v_off));
                    asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16;\n"
                                 :: "r"(v_dst), "l"(v_src), "n"(16));
                }
                issue_async_stage(i, i, i < TOTAL_TASKS);
#endif
                // only warp0 has real aiu ld task
                cp_async_commit();
            }
        } else {
            // No state pipeline: V gets its own per-warp group (every warp
            // issues the same tile into ITS group 0, so each warp's wait<0>
            // covers its own copy — no cross-warp publication needed),
            // drained right here. The block-wide publication of q/k/KDA bg
            // happens at the unified !use_initial_state barrier below.
            if (lane_id < V_LANES) {
                int v_off = lane_id * (16 / (int)sizeof(scalar_t));
                int v_row = v_off / BV;
                int v_col = v_off % BV;
                const scalar_t* v_src = v + bos * stride_v + (hv_base + v_row) * 128 + v_base + v_col;
                uint32_t v_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_bv + v_off));
                asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(v_dst), "l"(v_src), "n"(16));
            }
            cp_async_commit();
            cp_async_wait<0>();
        }

        // Head Dependent Initializations.
        // KDA keeps its per-K bg VECTOR in smem (warp<HPC producer + the barrier
        // below publishes it). beta/bg as per-hv SCALARS move into registers:
        // every warp recomputes them (redundant loads hit L2/LLC — grid.y
        // siblings of the same (i_n,i_h) already touched these lines), which
        // removes the tid<HPC producer thread, the smem publication and the
        // barrier on the GDN path.
        float b_beta_arr[HV_PER_CTA];
        float bg_arr[HV_PER_CTA];
        #pragma unroll
        for (int hs = 0; hs < HV_PER_CTA; ++hs) {
            int i_hv = hv_base + hs;
            float b_A_log = A_log[i_hv];
            float b_val = static_cast<float>(b[bos * stride_b + i_hv]);
            #if defined(__HGGC_ARCH__)
                b_beta_arr[hs] = __ppu_sgmdf(b_val);
            #else
                b_beta_arr[hs] = 1.0f / (1.0f + __expf(-b_val));
            #endif
            float a_val = static_cast<float>(a[bos * stride_a + i_hv]);
            float dt_val = static_cast<float>(dt_bias[i_hv]);
            bg_arr[hs] = compute_bg<USE_LOWER_BOUND>(a_val, dt_val, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
        }
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

                float bg0 = compute_bg<USE_LOWER_BOUND>(a01.x, dt01.x, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                float bg1 = compute_bg<USE_LOWER_BOUND>(a01.y, dt01.y, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                float bg2 = compute_bg<USE_LOWER_BOUND>(a23.x, dt23.x, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                float bg3 = compute_bg<USE_LOWER_BOUND>(a23.y, dt23.y, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);

                *reinterpret_cast<float4*>(smem_bg + warp_id * K + k_idx) = make_float4(bg0, bg1, bg2, bg3);
            }
        }

        // Q/K: warp0 ALONE loads + l2-normalizes (its 32 lanes cover K=128)
        // and publishes to smem_q/smem_k; siblings load them once after the
        // first block barrier they hit anyway (the task-0 barrier under
        // use_initial_state, the unified !use_initial_state barrier below).
        // The per-warp redundant l2norm chain this replaces was pure
        // issue-bandwidth waste — the kernel is instruction-budget bound and
        // the (WN-1) sibling copies of the 10-shuffle reduction never bought
        // anything once the prologue went barrier-free.
        float4 q_vec_decode;
        float4 k_vec_decode;
        if (warp_id == 0) {
            const scalar_t* q_src = q + bos * stride_q + i_h * 128 + k_idx;
            const scalar_t* k_src = k + bos * stride_k + i_h * 128 + k_idx;
            uint2 q_raw = *reinterpret_cast<const uint2*>(q_src);
            uint2 k_raw = *reinterpret_cast<const uint2*>(k_src);
            float2 q01 = TypeConverter<scalar_t>::convert(q_raw.x);
            float2 q23 = TypeConverter<scalar_t>::convert(q_raw.y);
            float2 k01 = TypeConverter<scalar_t>::convert(k_raw.x);
            float2 k23 = TypeConverter<scalar_t>::convert(k_raw.y);
            if (use_qk_l2norm) {
                float q_sq = q01.x*q01.x + q01.y*q01.y + q23.x*q23.x + q23.y*q23.y;
                float k_sq = k01.x*k01.x + k01.y*k01.y + k23.x*k23.x + k23.y*k23.y;
                // Butterfly reduction: all lanes hold the sums, no broadcast.
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    q_sq += __shfl_xor_sync(0xffffffff, q_sq, offset);
                    k_sq += __shfl_xor_sync(0xffffffff, k_sq, offset);
                }
                float q_norm = rsqrtf(q_sq + 1e-6f) * scale;
                float k_norm = rsqrtf(k_sq + 1e-6f);
                q01.x *= q_norm; q01.y *= q_norm; q23.x *= q_norm; q23.y *= q_norm;
                k01.x *= k_norm; k01.y *= k_norm; k23.x *= k_norm; k23.y *= k_norm;
            } else {
                q01.x *= scale; q01.y *= scale; q23.x *= scale; q23.y *= scale;
            }
            *reinterpret_cast<float4*>(&smem_q[k_idx]) = make_float4(q01.x, q01.y, q23.x, q23.y);
            *reinterpret_cast<float4*>(&smem_k[k_idx]) = make_float4(k01.x, k01.y, k23.x, k23.y);
        }
        if constexpr (!use_initial_state) {
            // Unified !use_initial_state publication barrier: V's drain
            // (wait<0> happened earlier), warp0's q/k and KDA's bg vector are
            // all published by this single block barrier.
            __syncthreads();
        }

        int task_idx = 0;
        float4 bg_vec;
        float bg_sc;

        // Per-warp bases (D1): one setup per warp replaces the per-task
        // v.madl/v.madw address MADs. With the task loops fully unrolled,
        // every remaining offset (hv_step, v_i, stage) is a compile-time
        // constant, so each access below is base+const.
        const state_t* smem_h_lane0 = &smem_h_flat[warp_id * 128 + k_idx];
        const scalar_t* smem_bv_lane0 = smem_bv + warp_id;
        // o is [B, HV, V]: the per-hv stride is i_hv*128 with NO qk-head
        // grouping (i_h*128 belongs to the q/k loads only — adding it here
        // shifted every o store to the wrong hv slot).
        scalar_t* o_warp = o + bos * (HV * 128) + v_base + warp_id;

        #pragma unroll
        for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
            float b_beta = b_beta_arr[hv_step];
            int i_hv = hv_base + hv_step;
            scalar_t* o_row = o_warp + i_hv * 128;

            float4 h_val;
            #pragma unroll
            for (int v_i = 0; v_i < V_ITERS; ++v_i) {
                int stage = task_idx % STAGE_NUM;

                if constexpr (use_initial_state) {
                    // Task-0 barrier: every warp just parked on (nearly) the
                    // same DRAM round-trip in the wait above, so arrivals are
                    // already aligned and this barrier costs ~nothing. It
                    // publishes warp0's smem_q/smem_k and KDA's smem_bg (V
                    // itself rides each warp's OWN group 0, issued by that
                    // warp's lanes, so it needs no cross-warp publication).
                    if constexpr (TOTAL_TASKS <= STAGE_NUM) {
                        // Full-prefetch shape (e.g. {32,4,8}/{64,8,8}): all
                        // TOTAL_TASKS groups were committed in the prologue
                        // and this loop commits none, so the per-task
                        // wait-depth chain would need a runtime-dependent
                        // template argument (illegal). Instead one wait<0>
                        // at task-0 drains every group — the per-task waits +
                        // empty commit_group churn it replaces were pure
                        // issue-bandwidth waste, and the extra startup
                        // latency hides behind the ~14 resident CTAs per CU.
                        if (hv_step == 0 && v_i == 0) {
                            cp_async_wait<0>();
                            __syncthreads();
                        }
                    } else {
                        cp_async_wait<STAGE_NUM - 1>();
                        if (hv_step == 0 && v_i == 0) {
                            __syncthreads();
                        }
                    }
                    h_val = k_last_load_state4(smem_h_lane0 + stage * (WARP_NUM * 128));
                }

                if (v_i == 0) {
                    if (hv_step == 0) {
                        // One-time post-barrier inputs: warp0's normalized
                        // q/k (published to smem_q/smem_k ahead of the task-0
                        // barrier under use_initial_state, or ahead of the
                        // unified !use_initial_state prologue barrier).
                        q_vec_decode = *reinterpret_cast<const float4*>(&smem_q[k_idx]);
                        k_vec_decode = *reinterpret_cast<const float4*>(&smem_k[k_idx]);
                    }
                    // KDA's bg vector is the last smem-published input read on
                    // this path: its first read must sit AFTER the task-0
                    // barrier above (use_initial_state) or after the
                    // !use_initial_state prologue barrier. bg_sc reads a
                    // register array and needs no ordering — it moves here
                    // only to keep the two variants symmetric.
                    if constexpr (IS_KDA) {
                        bg_vec = *reinterpret_cast<const float4*>(smem_bg + hv_step * K + k_idx);
                    } else {
                        bg_sc = bg_arr[hv_step];
                    }
                }

                float v_val = static_cast<float>(smem_bv_lane0[hv_step * BV + v_i * WARP_NUM]);

                if constexpr (IS_KDA) {
                    h_val.x *= bg_vec.x; h_val.y *= bg_vec.y; h_val.z *= bg_vec.z; h_val.w *= bg_vec.w;
                } else {
                    h_val.x *= bg_sc; h_val.y *= bg_sc; h_val.z *= bg_sc; h_val.w *= bg_sc;
                }

                float v_delta = .0f;
                v_delta += h_val.x * k_vec_decode.x;
                v_delta += h_val.y * k_vec_decode.y;
                v_delta += h_val.z * k_vec_decode.z;
                v_delta += h_val.w * k_vec_decode.w;

                // Butterfly (XOR) reduction: every lane ends up holding the
                // warp sum, which removes the re-broadcast shuffle (one shfl
                // fewer per task, and the sum feeds k_vec * v_new directly).
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) v_delta += __shfl_xor_sync(0xffffffff, v_delta, offset);

                float v_new = (v_val - v_delta) * b_beta;

                if constexpr (TOTAL_TASKS > STAGE_NUM) {
                    // Rolling-refill shape only: issue prefetch of the next
                    // stage of H0. Full-prefetch shapes (TOTAL_TASKS <=
                    // STAGE_NUM) committed everything in the prologue and
                    // skip this entirely — the empty commit/wait pair was
                    // pure issue-bandwidth waste there.
                    int next_task = task_idx + STAGE_NUM;
                    issue_async_stage(next_task, next_task % STAGE_NUM, next_task < TOTAL_TASKS);
                    cp_async_commit();
                }

                h_val.x += k_vec_decode.x * v_new;
                h_val.y += k_vec_decode.y * v_new;
                h_val.z += k_vec_decode.z * v_new;
                h_val.w += k_vec_decode.w * v_new;

                if constexpr (use_initial_state) {
                    k_last_store_state4(h0_row0 + hv_step * 16384 + v_i * (WARP_NUM * 128), h_val);
                }

                float o_val = .0f;
                o_val += h_val.x * q_vec_decode.x;
                o_val += h_val.y * q_vec_decode.y;
                o_val += h_val.z * q_vec_decode.z;
                o_val += h_val.w * q_vec_decode.w;

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) o_val += __shfl_xor_sync(0xffffffff, o_val, offset);

                if (lane_id == 0) {
                    o_row[v_i * WARP_NUM] = static_cast<scalar_t>(o_val);
                }
                task_idx++;
            }
        }
    }
    // =========================================================================
    // MTP_VERIFY PIPELINE (T > 1) — Optimized: t-loop outside v_i-loop
    // Q/K loaded once per timestep, V prefetched into smem, states in registers
    // =========================================================================
    else {
        // === MTP prologue precompute (computed ONCE per timestep, shared by all warps) ===
        // Eliminates the WARP_NUM-fold redundant l2norm/scale of q,k and __expf/softplus of
        // bg,beta that previously ran per-(hv_step,t) inside every warp. Raw q,k are async
        // prefetched via AIU so their transfer overlaps the beta/bg transcendental compute.
        {
            constexpr int QK_VEC = sizeof(float4) / sizeof(scalar_t);  // 8 for bf16
#ifdef PLA_USE_AIU
            for (int tt = warp_id; tt < T; tt += WARP_NUM) {
                ppu_aiu_load<128, 1, 1, 128, 1, 1, 128 * (int64_t)sizeof(scalar_t), 128 * (int64_t)sizeof(scalar_t), false>(
                    smem_q_raw + tt * 128, q + (bos + tt) * stride_q + i_h * 128, 0, 0, 0);
                ppu_aiu_load<128, 1, 1, 128, 1, 1, 128 * (int64_t)sizeof(scalar_t), 128 * (int64_t)sizeof(scalar_t), false>(
                    smem_k_raw + tt * 128, k + (bos + tt) * stride_k + i_h * 128, 0, 0, 0);
            }
#else
            for (int tt = warp_id; tt < T; tt += WARP_NUM) {
                if (lane_id < 128 / QK_VEC) {
                    const scalar_t* q_src = q + (bos + tt) * stride_q + i_h * 128 + lane_id * QK_VEC;
                    const scalar_t* k_src = k + (bos + tt) * stride_k + i_h * 128 + lane_id * QK_VEC;
                    uint32_t q_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_q_raw + tt * 128 + lane_id * QK_VEC));
                    uint32_t k_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_k_raw + tt * 128 + lane_id * QK_VEC));
                    asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(q_dst), "l"(q_src), "n"(16));
                    asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(k_dst), "l"(k_src), "n"(16));
                }
            }
#endif
            // [ASYNC GROUP A = q/k] Commit immediately after issue to close the q/k group, so its
            // DMA transfer overlaps the beta/bg transcendental compute below.
            //   PRELOAD point : q/k issued just above (AIU or cp.async).
            //   COMMIT point  : here (defines the group boundary).
            //   PAIRED WAIT   : "[DRAIN GROUP A]" below, right before q/k normalization.
            cp_async_commit();
            // Compute beta,bg once per (t,hv_step); overlaps with the q/k async transfer above.
            for (int tt = warp_id; tt < T; tt += WARP_NUM) {
                int tt_idx = bos + tt;
                #pragma unroll
                for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
                    int i_hv = hv_base + hv_step;
                    if constexpr (!IS_KDA) {
                        if (lane_id == 0) {
                            float b_A_log = A_log[i_hv];
                            float b_val = static_cast<float>(b[tt_idx * stride_b + i_hv]);
                            #if defined(__HGGC_ARCH__)
                                smem_beta_mtp[tt * HV_PER_CTA + hv_step] = __ppu_sgmdf(b_val);
                            #else
                                smem_beta_mtp[tt * HV_PER_CTA + hv_step] = 1.0f / (1.0f + __expf(-b_val));
                            #endif
                            float a_val = static_cast<float>(a[tt_idx * stride_a + i_hv]);
                            float dt_val = static_cast<float>(dt_bias[i_hv]);
                            smem_bg_mtp[tt * HV_PER_CTA + hv_step] = compute_bg<USE_LOWER_BOUND>(a_val, dt_val, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                        }
                    } else {
                        float b_A_log = A_log[i_hv];
                        if (lane_id == 0) {
                            float b_val = static_cast<float>(b[tt_idx * stride_b + i_hv]);
                            #if defined(__HGGC_ARCH__)
                                smem_beta_mtp[tt * HV_PER_CTA + hv_step] = __ppu_sgmdf(b_val);
                            #else
                                smem_beta_mtp[tt * HV_PER_CTA + hv_step] = 1.0f / (1.0f + __expf(-b_val));
                            #endif
                        }
                        uint2 a_raw = *reinterpret_cast<const uint2*>(a + tt_idx * stride_a + i_hv * 128 + k_idx);
                        float4 dt = *reinterpret_cast<const float4*>(dt_bias + i_hv * 128 + k_idx);
                        float2 a01 = TypeConverter<scalar_t>::convert(a_raw.x);
                        float2 a23 = TypeConverter<scalar_t>::convert(a_raw.y);
                        float bg0 = compute_bg<USE_LOWER_BOUND>(a01.x, dt.x, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                        float bg1 = compute_bg<USE_LOWER_BOUND>(a01.y, dt.y, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                        float bg2 = compute_bg<USE_LOWER_BOUND>(a23.x, dt.z, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                        float bg3 = compute_bg<USE_LOWER_BOUND>(a23.y, dt.w, b_A_log, softplus_beta_or_lower_bound, softplus_threshold);
                        *reinterpret_cast<float4*>(smem_bg_mtp + (tt * HV_PER_CTA + hv_step) * 128 + k_idx) = make_float4(bg0, bg1, bg2, bg3);
                    }
                }
            }
            // Drain q/k async prefetch before normalizing.
            // [DRAIN A = q/k] cp_async_wait<0> waits until all committed groups (the q/k GROUP A
            // above) complete; the barrier then publishes smem_q_raw/smem_k_raw before normalize reads them.
#ifdef PLA_USE_AIU
            cp_async_wait<0>();
            __ppu_barrier_sync_defer(0, WARP_NUM * 32, 1);
#else
            cp_async_wait<0>();
            __syncthreads();
#endif
            // Normalize + scale q/k from the staged raw tile (one warp per timestep).
            for (int tt = warp_id; tt < T; tt += WARP_NUM) {
                uint2 q_raw = *reinterpret_cast<const uint2*>(smem_q_raw + tt * 128 + k_idx);
                uint2 k_raw = *reinterpret_cast<const uint2*>(smem_k_raw + tt * 128 + k_idx);
                float2 q01 = TypeConverter<scalar_t>::convert(q_raw.x);
                float2 q23 = TypeConverter<scalar_t>::convert(q_raw.y);
                float2 k01 = TypeConverter<scalar_t>::convert(k_raw.x);
                float2 k23 = TypeConverter<scalar_t>::convert(k_raw.y);
                if (use_qk_l2norm) {
                    float q_sq = q01.x*q01.x + q01.y*q01.y + q23.x*q23.x + q23.y*q23.y;
                    float k_sq = k01.x*k01.x + k01.y*k01.y + k23.x*k23.x + k23.y*k23.y;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        q_sq += __shfl_down_sync(0xffffffff, q_sq, offset);
                        k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
                    }
                    q_sq = __shfl_sync(0xffffffff, q_sq, 0);
                    k_sq = __shfl_sync(0xffffffff, k_sq, 0);
                    float q_norm = rsqrtf(q_sq + 1e-6f) * scale;
                    float k_norm = rsqrtf(k_sq + 1e-6f);
                    q01.x *= q_norm; q01.y *= q_norm; q23.x *= q_norm; q23.y *= q_norm;
                    k01.x *= k_norm; k01.y *= k_norm; k23.x *= k_norm; k23.y *= k_norm;
                } else {
                    q01.x *= scale; q01.y *= scale; q23.x *= scale; q23.y *= scale;
                }
                *reinterpret_cast<float4*>(smem_q_mtp + tt * 128 + k_idx) = make_float4(q01.x, q01.y, q23.x, q23.y);
                *reinterpret_cast<float4*>(smem_k_mtp + tt * 128 + k_idx) = make_float4(k01.x, k01.y, k23.x, k23.y);
            }
            __syncthreads();
        }

        // [GROUP B = V preload] Issued below. When use_initial_state, V is NOT committed on its own:
        // it is folded into GROUP C (H0) and committed by the H0 cp_async_commit, then drained together
        // with H0 by [DRAIN C] in the v_i loop (t==0). When !use_initial_state it is committed + drained
        // locally right after issue (see the if constexpr(!use_initial_state) block).
        // Preload all T timesteps x HV_PER_CTA heads of V into shared, to offload
        // per-iteration scalar V loads (sld -> SALU) into shared loads (lds).
#ifdef PLA_USE_AIU
        // Async prefetch via AIU: each warp issues the AIU load for its timestep(s)
        // (loop if WARP_NUM < T). For use_initial_state, these are committed by the
        // H0 cp_async_commit and drained by the existing cp_async_wait + defer barrier
        // before the first V read. For !use_initial_state there is no H0 async path,
        // so commit + drain V explicitly here.
        {
            constexpr int DIM_C = BV;
            constexpr int DIM_W = 1;
            constexpr int DIM_N = HV_PER_CTA;
            constexpr int CUBE_C = DIM_C;
            constexpr int CUBE_W = 1;
            constexpr int CUBE_N = DIM_N;
            constexpr int64_t tensor_stride_w = DIM_C * sizeof(scalar_t);
            constexpr int64_t tensor_stride_n = 128 * sizeof(scalar_t); // K * V
            for (int tt = warp_id; tt < T; tt += WARP_NUM) {
                const scalar_t* v_src = v + (bos + tt) * stride_v + hv_base * 128 + v_base;
                ppu_aiu_load<CUBE_C, CUBE_W, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w, tensor_stride_n, false>(
                    smem_v_mtp + tt * HV_PER_CTA * BV,
                    v_src,
                    0, 0, 0);
            }
            if constexpr (!use_initial_state) {
                cp_async_commit();
                #ifdef __HGGC_ARCH__
                __ppu_barrier_sync_defer(0, WARP_NUM * 32, 1);
                #else
                cp_async_wait<0>();
                __syncthreads();
                #endif
            }
        }
#else
        // Fallback (non-AIU): asynchronous prefetch via cp.async. Issue all rows,
        // then commit + drain before the compute loop (drained here, before the H0
        // cp.async pipeline starts, so no group-count entanglement).
        {
            constexpr int V_VEC_LEN = sizeof(float4) / sizeof(scalar_t); // 8 for bf16/fp16
            constexpr int VECS_PER_ROW = BV / V_VEC_LEN;
            int total_rows = T * HV_PER_CTA;
            for (int row = warp_id; row < total_rows; row += WARP_NUM) {
                int tt = row / HV_PER_CTA;
                int hh = row - tt * HV_PER_CTA;
                if (lane_id < VECS_PER_ROW) {
                    const scalar_t* v_src = v + (bos + tt) * stride_v + (hv_base + hh) * 128 + v_base + lane_id * V_VEC_LEN;
                    uint32_t smem_dst = static_cast<uint32_t>(__cvta_generic_to_shared(
                        smem_v_mtp + (tt * HV_PER_CTA + hh) * BV + lane_id * V_VEC_LEN));
                    asm volatile(
                        "ppu.cp.async.cg.shared.global [%0], [%1], %2;\n" ::
                        "r"(smem_dst), "l"(v_src), "n"(16));
                }
            }
            if constexpr (!use_initial_state) {
                cp_async_commit();
                cp_async_wait<0>();
                __syncthreads();
            }
        }
#endif

        #pragma unroll
        for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
            int i_hv = hv_base + hv_step;

            float4 h_vals[V_ITERS];
            #pragma unroll
            for (int v_i = 0; v_i < V_ITERS; ++v_i) {
                h_vals[v_i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            // probeD: H0 at t=0 no longer goes through the AIU bulk + smem stage ring (the recycle refill
            // when V_ITERS>SN is the second suspect path of this bug); the consumer instead has each warp do a direct synchronous global read
            // (see the t==0 branch of the v_i loop), so H0 no longer occupies any async group.
            // In the original implementation the V preload (GROUP B) was folded into the first H0 commit; here it is closed explicitly
            // and drained with the same discipline as [DRAIN A], so the data is ready before t=0 first reads smem_v_mtp.
            // V (issued unconditionally above) is folded into this commit/drain.
            // probeD replaced the H0 async prefetch with a direct global read, so
            // this drain must NOT be gated on h0_idx>=0: when initial_state_indices
            // is None the host passes a null h0_indices, giving h0_idx<0 with NO
            // early return (the invalid-slot bail-out above requires h0_indices !=
            // nullptr). Gating here would then let the v_i loop read smem_v_mtp
            // (line ~942) before the V AIU/cp.async load has any completion barrier
            // -> race. Every provided-but-invalid slot already early-returns, so
            // reaching here with h0_idx<0 is exactly (and only) the None case;
            // h0_idx>=0 is the sole production path and stays behaviour-identical.
            if constexpr (use_initial_state) {
                cp_async_commit();
                cp_async_wait<0>();
#ifdef __HGGC_ARCH__
                __ppu_barrier_sync_defer(0, WARP_NUM * 32, 1);
#else
                __syncthreads();
#endif
            }

            for (int t = 0; t < T; ++t) {
                int t_idx = bos + t;

                // Resolve where this timestep's state has to land. sglang keeps
                // intermediate states in a dedicated buffer laid out as
                // [cache_idx][step], whereas vLLM hands every draft token its own
                // slot of the main state pool through the 2D h0_indices table.
                // The two buffers have different slot strides: SGLang's
                // intermediate_states_buffer is contiguous (stride = HV*V*K),
                // while vLLM's h0_source is page-packed (stride = h0_slot_stride,
                // which exceeds HV*V*K because conv_state shares the page).
                state_t* step_state_buf = nullptr;
                int64_t step_state_slot = 0;
                int64_t step_state_slot_stride = (int64_t)HV * 16384;
                if (is_sglang) {
                    if (cache_intermediate_states && cache_idx >= 0) {
                        step_state_buf = intermediate_states_buffer;
                        step_state_slot = (int64_t)cache_idx * cache_steps + t;
                        // stride stays HV * 16384 (contiguous buffer)
                    }
                } else if (stride_h0_indices_seq > 0 && h0_indices != nullptr) {
                    int step_slot = h0_indices[i_n * stride_h0_indices_seq + t];
                    if (step_slot > 0) {
                        step_state_buf = h0_source;
                        step_state_slot = step_slot;
                        step_state_slot_stride = h0_slot_stride;
                    }
                }

                // --- Read precomputed normalized q,k from shared (computed once in prologue) ---
                float4 q_vec = *reinterpret_cast<const float4*>(smem_q_mtp + t * 128 + k_idx);
                float4 k_vec = *reinterpret_cast<const float4*>(smem_k_mtp + t * 128 + k_idx);

                // --- Read precomputed beta, bg from shared (computed once in prologue) ---
                float b_beta = smem_beta_mtp[t * HV_PER_CTA + hv_step];
                float bg_sc;
                float4 bg_vec;
                if constexpr (!IS_KDA) {
                    bg_sc = smem_bg_mtp[t * HV_PER_CTA + hv_step];
                } else {
                    bg_vec = *reinterpret_cast<const float4*>(smem_bg_mtp + (t * HV_PER_CTA + hv_step) * 128 + k_idx);
                }

                // --- V is preloaded into smem_v_mtp in prologue; read from smem in v_i loop ---

                // --- Eagle tree: load parent step once per timestep ---
                int parent_step = -1;
                if (has_eagle_tree && t != 0 && cache_idx >= 0) {
                    if (lane_id == 0) parent_step = retrieve_parent_token_ptr[i_n * stride_retrieve_parent_token_seq + t * stride_retrieve_parent_token_token];
                    parent_step = __shfl_sync(0xffffffff, parent_step, 0);
                    if (parent_step >= t) {
                        __trap();
                        return;
                    }
                }

                // Eagle tree: preload intermediate states via async pipeline
                // [GROUP D = eagle states] Issued (AIU or cp.async) and committed in the loop below,
                // in its own group per timestep. Paired drain: [DRAIN D] Eagle tree: drain in v_i loop.
                bool need_eagle_reload = (parent_step >= 0 && parent_step != t - 1);
                state_t* eagle_preload_base = nullptr;
                if (need_eagle_reload) {
                    // Cross-warp / cross-timestep global hazard. The slot we are about
                    // to reload, intermediate_states_buffer[cache_idx][parent_step], was
                    // produced by an EARLIER timestep at the end of the v_i loop (see the
                    // step_state_buf store below), written per-warp: each warp owns the
                    // v-columns v_base+warp_id+v_i*WARP_NUM. Plain timesteps carry no
                    // block barrier, so warps drift; a fast warp can reach this eagle
                    // reload and read columns a lagging warp has not written yet -> the
                    // cp.async/AIU load then pulls stale/partial parent state, producing
                    // run-to-run nondeterminism (KDA amplifies it through the per-K
                    // bg_vec decay). Fence our own global writes to device scope, then a
                    // full block barrier so every warp's writes (and their fences) are
                    // globally visible before ANY warp reloads this slot. need_eagle_reload
                    // is uniform across the CTA (depends only on i_n and t), so the barrier
                    // is not divergent.
                    __threadfence();
                    __syncthreads();
                    int64_t eagle_cache_offset = (int64_t)cache_idx * cache_steps * HV * 16384 + parent_step * HV * 16384 + i_hv * 16384 + v_base * 128;
                    eagle_preload_base = intermediate_states_buffer + eagle_cache_offset + k_idx;
#ifdef PLA_USE_AIU
                    {
                        constexpr int DIM_C = 128;
                        constexpr int DIM_W = BV;
                        constexpr int DIM_N = 1;
                        constexpr int CUBE_C = DIM_C;
                        constexpr int CUBE_W = STAGE_NUM * WARP_NUM;
                        constexpr int CUBE_N = 1;
                        constexpr int64_t tensor_stride_w = DIM_C * sizeof(state_t);
                        constexpr int64_t tensor_stride_n = 128 * 128 * sizeof(state_t);
                        state_t* eagle_offset = intermediate_states_buffer + eagle_cache_offset;
                        if (warp_id == 0) {
                            ppu_aiu_load<CUBE_C, CUBE_W, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w, tensor_stride_n, sizeof(state_t) == 4>(
                                smem_h_flat, eagle_offset, 0, 0, 0);
                        }
                    }
#endif
                    #pragma unroll
                    for (int i = 0; i < STAGE_NUM; ++i) {
#ifndef PLA_USE_AIU
                        {
                            state_t* ptr = eagle_preload_base + (i * WARP_NUM + warp_id) * 128;
                            uint32_t smem_ptr = smem_h_ptr_per_warp + (i * WARP_NUM * 128 * sizeof(state_t));
                            bool pred = i < V_ITERS;
                            k_last_prefetch_state4(smem_ptr, ptr, pred);
                        }
#endif
                        cp_async_commit();
                    }
                }

                // --- Process all V-tiles for this timestep ---
                #pragma unroll
                for (int v_i = 0; v_i < V_ITERS; ++v_i) {
                    float4 h_val;
                    int v_col = v_base + warp_id + v_i * WARP_NUM;

                    // probeD: for the first-step H0 each warp synchronously reads the columns it owns (same addresses as
                    // issue_async_stage), fully bypassing the smem stage ring, AIU bulk,
                    // group-count waits and recycle refill; the eagle path keeps probeC unchanged.
                    if (t == 0 && h0_idx >= 0) {
                        if constexpr (use_initial_state) {
                            h_val = k_last_load_state4(h0_row0 + hv_step * 16384 + v_i * (WARP_NUM * 128));
                        } else {
                            h_val = h_vals[v_i];
                        }
                    } else {
                        h_val = h_vals[v_i];
                    }

                    if (need_eagle_reload) {
                        // probeC: the first round (v_i<SN, no recycling) keeps the original AIU bulk + stage and the
                        // v_i==0 publish barrier; only the recycled part (v_i>=SN) switches to each warp doing a direct
                        // synchronous global read of the parent-state columns, no longer refilling the stage ring.
                        if (v_i < STAGE_NUM) {
#ifdef PLA_USE_AIU
                            // PPU: warp0's single AIU bulk lands in group 0 and the
                            // remaining STAGE_NUM-1 commits are empty, so wait<SN-1>
                            // drains the whole first ring on the first consume.
                            cp_async_wait<STAGE_NUM - 1>();
#else
                            // Non-AIU fallback: each prefetched stage is a distinct
                            // cp.async group. The per-stage commit that used to
                            // advance the wait window was removed, so a fixed
                            // wait<SN-1> would only ever guarantee group 0 and read
                            // stale rows for v_i>=1. Drain every stage once up front.
                            if (v_i == 0) cp_async_wait<0>();
#endif
                            if (v_i == 0) {
#ifdef __HGGC_ARCH__
                                __ppu_barrier_sync_defer(0, WARP_NUM*32, 1);
#endif
                            }
                            h_val = k_last_load_state4(&smem_h_flat[(v_i % STAGE_NUM) * WARP_NUM * 128 + warp_id * 128 + k_idx]);
                        } else {
                            h_val = k_last_load_state4(
                                eagle_preload_base + (v_i * WARP_NUM + warp_id) * 128);
                        }
                    }

                    float v_val = static_cast<float>(smem_v_mtp[(t * HV_PER_CTA + hv_step) * BV + warp_id + v_i * WARP_NUM]);

                    if constexpr (IS_KDA) {
                        h_val.x *= bg_vec.x; h_val.y *= bg_vec.y; h_val.z *= bg_vec.z; h_val.w *= bg_vec.w;
                    } else {
                        h_val.x *= bg_sc; h_val.y *= bg_sc; h_val.z *= bg_sc; h_val.w *= bg_sc;
                    }

                    float v_delta = .0f;
                    v_delta += h_val.x * k_vec.x;
                    v_delta += h_val.y * k_vec.y;
                    v_delta += h_val.z * k_vec.z;
                    v_delta += h_val.w * k_vec.w;

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) v_delta += __shfl_down_sync(0xffffffff, v_delta, offset);
                    v_delta = __shfl_sync(0xffffffff, v_delta, 0);

                    float v_new = (v_val - v_delta) * b_beta;

                    h_val.x += k_vec.x * v_new;
                    h_val.y += k_vec.y * v_new;
                    h_val.z += k_vec.z * v_new;
                    h_val.w += k_vec.w * v_new;

                    if (step_state_buf != nullptr) {
                        int64_t cache_offset = step_state_slot * step_state_slot_stride + i_hv * 16384;
                        k_last_store_state4(step_state_buf + cache_offset + v_col * 128 + k_idx, h_val);
                    }

                    float o_val = .0f;
                    o_val += h_val.x * q_vec.x;
                    o_val += h_val.y * q_vec.y;
                    o_val += h_val.z * q_vec.z;
                    o_val += h_val.w * q_vec.w;

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) o_val += __shfl_down_sync(0xffffffff, o_val, offset);

                    if (lane_id == 0) {
                        o[t_idx * HV * 128 + i_hv * 128 + v_col] = static_cast<scalar_t>(o_val);
                    }

                    h_vals[v_i] = h_val;
                }

                // Flush async pipeline after first timestep or eagle tree reload
                if ((t == 0 && h0_idx >= 0) || need_eagle_reload) {
                    cp_async_wait<0>();
                }
            }

            // Write final states back
            if (!disable_state_update && h0_idx >= 0) {
                #pragma unroll
                for (int v_i = 0; v_i < V_ITERS; ++v_i) {
                    k_last_store_state4(h0_cta_base + hv_step * 16384 + (warp_id + v_i * WARP_NUM) * 128, h_vals[v_i]);
                }
            }
        }
    }
    cp_async_wait<0>();
}
