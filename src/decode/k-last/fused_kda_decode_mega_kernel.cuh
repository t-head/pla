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

/*
 * PPU KDA fused decode kernel — device-side implementation.
 *
 * This header holds the DEVICE-SIDE MAIN BODY of the PPU fused KDA decode
 * kernel (`fused_kda_decode_ppu_kernel`). The host wrapper / launchers live in
 * fused_kda_decode_mega.cu and expose the SAME I/O contract as the NVIDIA
 * reference `fused_kda_decode_ref_forward` (torch.ops._C.fused_kda_decode):
 *
 *   - block = dim3(32, WARP_NUM)  (256 threads for WN8, 128 for WN4)
 *   - each warp owns a k-slice k_idx = lane*4 (float4), and iterates over
 *     V_ITERS = BV / WARP_NUM value-rows;
 *   - the recurrent state H0 is streamed on-chip through a STAGE_NUM-deep
 *     software pipeline (AIU bulk-tensor load on PPU / cp.async fallback),
 *   - Stage A (conv+silu, q/k L2-norm, per-channel decay gate), Stage B
 *     (delta-rule recurrence) and Stage C (gated RMSNorm) all close inside a
 *     single CTA.
 *
 * Portability (task: PPU1.0 + PPU1.5 + generic CUDA):
 *   - PPU1.5 (__HGGC_ARCH__ == 150): AIU 2D-tile bulk load via ppu_aiu_load.
 *   - PPU1.0 (__HGGC_ARCH__ == 100): AIU 3D-linear bulk load via ppu_aiu_load
 *     (the arch-100 asm variant, per the TIX AIU documentation). Both PPU
 *     generations take the AIU path rather than falling back to cp.async.
 *   - Non-PPU CUDA: cp.async multi-stage software pipeline fallback.
 *
 * Only the parts that DIFFER from the original PLA-style interface are adjusted
 * to the vllm reference contract (functional micro-adjustments):
 *   (a) non-power-of-2 head count: grid = dim3(1, H, B), i_n=blockIdx.z,
 *       i_h=i_hv=blockIdx.y  (Kimi K3 has H==HV, H in {6,12,24,48,96});
 *   (b) lower_bound decay variant  exp(lb*sigmoid(exp_a*g)) vs the softplus one;
 *   (c) packed tap-major conv weight [3,W,dim] and conv_state [slot,3*dim,3]
 *       (bit-exact 3-tap shift writeback), instead of channel-major [D,W];
 *   (d) slot strides taken from state.stride(0) / conv_state.stride(0);
 *   (e) the SAME state_indices slot addresses both the recurrent state and the
 *       short-conv state;
 *   (f) dt_bias is a float[dim] indexed by hk = i_h*128 + k;
 *   (g) fuse_output_norm optional (raw o written out when absent);
 *   (h) results written in place to `out` [1, B, H, 128].
 *
 * PPU sigmoid uses the hardware __ppu_sgmdf intrinsic; the sm90 grid-dependency
 * intrinsics of the reference are not used here. The kernel is templated on
 * scalar_t so it instantiates for both __ppu_bfloat16 and
 * __half activations.
 */

#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_runtime.h>

#include "../../pla_device_helpers.cuh"
#include "k_last_shared_layout.h"

// -----------------------------------------------------------------------------
// AIU-enable portability macro.
//
// pla_device_helpers.cuh already gates PLA_USE_AIU on both PPU generations, so
// this is just a local alias kept for readability at the two use sites below. The
// cp.async fallback is gated on its absence so a PPU never double-issues cp.async
// on top of the AIU load.
// -----------------------------------------------------------------------------
#ifdef PLA_USE_AIU
#define PPU_KDA_USE_AIU
#endif

namespace fused_kda_decode_ppu {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kHeadElems = kDimK * kDimV;   // 16384, per-(v-head) state stride
constexpr int kKernelWidth = 4;
constexpr int kConvStateWidth = kKernelWidth - 1;   // 3 history taps

// ----------------------------------------------------------------------------
// Small device helpers (sigmoid uses the PPU hardware op when available).
// ----------------------------------------------------------------------------
__device__ __forceinline__ float ppu_sigmoid(float x) {
#if defined(__HGGC_ARCH__)
  return __ppu_sgmdf(x);
#else
  return 1.0f / (1.0f + __expf(-x));
#endif
}

__device__ __forceinline__ float ppu_silu(float x) { return x * ppu_sigmoid(x); }

__device__ __forceinline__ float ppu_softplus(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

// tap-major conv weight  w[tap*dim + channel]
__device__ __forceinline__ float conv_w(const float* __restrict__ w, int channel,
                                         int tap, int dim) {
  return w[tap * dim + channel];
}

// ----------------------------------------------------------------------------
// The mega decode kernel, adapted to the vllm packed contract.
//
// Layout facts (K3 decode, K==V==128, H==HV):
//   x            : [B, 3*dim] packed q|k|v, row stride = x_row (>= 3*dim)
//                  x_q = base, x_k = base + dim, x_v = base + 2*dim
//   conv_state   : [slots, 3*dim, 3], cs[slot*conv_slot + ch + tap*3*dim]
//                  cs_q = base, cs_k = base + dim, cs_v = base + 2*dim
//   weight       : [3, W, dim] f32, w_q=base, w_k=+W*dim, w_v=+2*W*dim
//   state        : [slots, H, 128, 128] f32, slot stride = state_slot
//   raw_g        : [1, B, H, 128]   -> g[bos*dim + i_hv*128 + k]
//   dt_bias      : [dim] f32        -> dt_bias[i_h*128 + k]
//   raw_beta     : [1, B, H]        -> beta[bos*beta_row + i_hv]
//   output_gate  : [.,B,H,128]      -> og[i_n*onorm_row + i_hv*128 + v]
//   out          : [1, B, H, 128]   -> out[i_n*dim + i_hv*128 + v]
// ----------------------------------------------------------------------------
template <typename scalar_t, int WARP_NUM, int BV, int STAGE_NUM, int CORE_ROWS,
          bool APPLY_ONORM, bool USE_LOWER_BOUND, bool PARALLEL_EPILOGUE>
__global__ void fused_kda_decode_ppu_kernel(
    // gate / decay
    const float* __restrict__ a_log,      // [H]
    const scalar_t* __restrict__ g,       // raw_g [1,B,H,128]
    const float* __restrict__ dt_bias,    // [dim]
    // q/k/v pre-conv projections (packed x)
    const scalar_t* __restrict__ x_q,
    const scalar_t* __restrict__ x_k,
    const scalar_t* __restrict__ x_v,
    const scalar_t* __restrict__ beta,    // raw_beta [1,B,H]
    // short conv
    const float* __restrict__ w_q_t,
    const float* __restrict__ w_k_t,
    const float* __restrict__ w_v_t,
    const float* __restrict__ bias_q,     // may be null
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    scalar_t* __restrict__ cs_q,
    scalar_t* __restrict__ cs_k,
    scalar_t* __restrict__ cs_v,
    // gated RMSNorm epilogue
    const scalar_t* __restrict__ g_out,       // output_gate (may be null)
    const float* __restrict__ o_norm_weight,  // [V] (may be null)
    const float norm_eps,
    // recurrent state + output
    float* __restrict__ h0_source,            // state [slots,H,128,128]
    const int* __restrict__ state_indices,    // [B]
    scalar_t* __restrict__ o,                 // out [1,B,H,128]
    const float scale,
    const float lower_bound,
    const int B, const int H, const int HV,
    const int64_t x_row, const int64_t beta_row, const int64_t onorm_row,
    const int64_t conv_slot, const int64_t state_slot) {
  constexpr int K = kDimK;
  constexpr int V = kDimV;
  constexpr int V_ITERS = BV / WARP_NUM;      // 16 for WN8, 32 for WN4
  constexpr int HV_PER_CTA = 1;
  constexpr int TOTAL_TASKS = HV_PER_CTA * V_ITERS;
  constexpr bool QUAD_ROW_CORE = CORE_ROWS == 4;
  constexpr int PIPE_STAGES = STAGE_NUM;
  static_assert(CORE_ROWS == 2 || CORE_ROWS == 4,
                "KDA recurrent core only supports 2 or 4 rows per iteration");
  static_assert(WARP_NUM == 4 || WARP_NUM == 8,
                "KDA mega kernel only supports WARP_NUM=4 or 8");

  const int i_n = blockIdx.z;                 // batch row
  const int i_h = blockIdx.y;                 // qk head
  const int i_hv = i_h;                        // K3: HV == H
  const int hv_base = i_hv;
  const int bos = i_n;                         // decode: single token
  const int slot = (state_indices != nullptr) ? state_indices[i_n] : i_n;

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  const int tid = warp_id * 32 + lane_id;
  const int k_idx = lane_id * 4;

  const int dim = H * kDimK;                   // key_dim == value_dim
  const int64_t kPackedDim = 3LL * dim;        // conv tap stride
  const int hk_off = i_h * kDimK;
  const int hv_off = i_hv * kDimV;

  // Framework pad slot. sglang marks CUDA-graph padding rows with
  // PAD_SLOT_ID = -1, which has no backing memory, so slot * state_slot would
  // address outside the pool; emit zeros and touch neither state nor conv_state.
  // vLLM instead pads with slot 0, a real reserved null_block, so it stays >= 0
  // and keeps its original behaviour of computing the row as usual -- hence this
  // guard needs no framework flag. slot only depends on i_n, so the whole CTA
  // agrees and the early return cannot desynchronise a later __syncthreads().
  if (slot < 0) {
    if (tid < kDimV) {
      o[i_n * dim + i_hv * kDimV + tid] = static_cast<scalar_t>(0.0f);
    }
    return;
  }

  // ---------------------------- shared memory -----------------------------
  extern __shared__ __align__(16) unsigned char smem_bytes[];
  constexpr pla::KLastSharedLayout shared_layout(
      PIPE_STAGES, K, WARP_NUM, BV, HV_PER_CTA, true, false, 1, sizeof(scalar_t));
  // Stage-A uses the two reduction vectors as one interleaved float2 array.
  static_assert(shared_layout.reduction_k == shared_layout.reduction_q + sizeof(float) * WARP_NUM,
                "paired Q/K reductions must remain adjacent");
  float* smem_h_flat = reinterpret_cast<float*>(smem_bytes + shared_layout.h);
  float* smem_q = reinterpret_cast<float*>(smem_bytes + shared_layout.q);
  float* smem_k = reinterpret_cast<float*>(smem_bytes + shared_layout.k);
  scalar_t* smem_bv = reinterpret_cast<scalar_t*>(smem_bytes + shared_layout.bv);
  float* smem_reduction = reinterpret_cast<float*>(smem_bytes + shared_layout.reduction_q);
  float* smem_reduction2 = reinterpret_cast<float*>(smem_bytes + shared_layout.reduction_k);
  float* smem_b_beta = reinterpret_cast<float*>(smem_bytes + shared_layout.beta);
  float* smem_bg = reinterpret_cast<float*>(smem_bytes + shared_layout.bg);
  float* smem_o = reinterpret_cast<float*>(smem_bytes + shared_layout.total_bytes);

  // Recurrent-state slot base (contiguous [H,128,128] within a slot).
  float* h0_cta_base =
      h0_source + slot * state_slot + hv_base * kHeadElems + k_idx;

  uint32_t smem_h_ptr_per_warp = static_cast<uint32_t>(
      __cvta_generic_to_shared(&smem_h_flat[warp_id * 128 + k_idx]));
  auto issue_async_stage = [&](int task, int stage, bool pred_guard) {
    int v_i = task % V_ITERS;
    int hv_step = task / V_ITERS;
    float* h0_ptr = h0_cta_base + hv_step * kHeadElems + (v_i * WARP_NUM + warp_id) * 128;
    uint32_t smem_h_ptr = smem_h_ptr_per_warp + (stage * WARP_NUM * 128 * sizeof(float));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  ppu.cmpp.ne.b32 p, %0, 0;\n"
        "  @p ppu.cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred_guard),
        "r"(smem_h_ptr), "l"(h0_ptr), "n"(16));
  };

  auto issue_aiu_ld = [&](int coord_w, int coord_n) {
    constexpr int DIM_C = 128;
    constexpr int DIM_W = BV;
    constexpr int DIM_N = HV_PER_CTA;
    constexpr int CUBE_C = DIM_C;
    constexpr int CUBE_W = PIPE_STAGES * WARP_NUM;
    constexpr int CUBE_N = HV_PER_CTA;
    constexpr int64_t tensor_stride_w = DIM_C * sizeof(float);
    constexpr int64_t tensor_stride_n = 128 * 128 * sizeof(float);
    float* h0_head_base = h0_source + slot * state_slot + hv_base * kHeadElems;
    if (warp_id == 0) {
      ppu_aiu_load<CUBE_C, CUBE_W, CUBE_N, DIM_C, DIM_W, DIM_N, tensor_stride_w,
                   tensor_stride_n>(smem_h_flat, h0_head_base, 0, coord_w, coord_n);
    }
  };

  // Kick off the recurrent-state prefetch as early as possible so its DMA
  // overlaps the Stage-A conv / gate transcendental compute below.
#ifdef PPU_KDA_USE_AIU
  issue_aiu_ld(0, 0);
#endif
  #pragma unroll
  for (int i = 0; i < PIPE_STAGES; ++i) {
#ifndef PPU_KDA_USE_AIU
    issue_async_stage(i, i, i < TOTAL_TASKS);
#endif
    cp_async_commit();
  }

  // =========================================================================
  // Stage A: q/k/v conv+silu -> q/k L2norm -> scale ; decay gate ; beta.
  // P9 uses a compile-time WARP_NUM branch:
  //   WN8: warps 0-3 own q+k, while warps 4-7 own v+decay. All eight warps
  //        therefore execute useful Stage-A work concurrently.
  //   WN4: preserve the existing q+k+decay -> reduction -> v schedule so a
  //        later 4-warp production configuration keeps its known behaviour.
  // Every channel is still handled by exactly one thread with the same
  // per-channel arithmetic and bit-exact 3-tap conv-state writeback.
  // =========================================================================
  {
    // Only lane 0 of each warp evaluates the transcendental; broadcast within
    // the warp (mirrors the ref kernel's __shfl_sync(...,0) optimisation).
    // Bit-exact: the same __expf result is broadcast, only instruction count
    // drops from 256 to WARP_NUM per CTA.
    const float exp_a =
        __shfl_sync(0xffffffff, lane_id == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

    float q_out = 0.0f;
    float k_out = 0.0f;

    // q/k ownership stays on warps 0-3 for WN8. This also preserves the P1
    // q/k warp-reduction order; WN8 warps 4-7 contribute exact zeros below.
    const int qk_channel =
        (WARP_NUM == 8 && warp_id >= 4) ? -1 : tid;
    if (qk_channel >= 0 && qk_channel < K) {
      const int k = qk_channel;
      const int hk = hk_off + k;

      float q_acc = (bias_q != nullptr) ? bias_q[hk] : 0.0f;
      float k_acc = (bias_k != nullptr) ? bias_k[hk] : 0.0f;
      scalar_t q_taps[kConvStateWidth];
      scalar_t k_taps[kConvStateWidth];
      #pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int64_t ci = (int64_t)slot * conv_slot + hk + (int64_t)w * kPackedDim;
        q_taps[w] = cs_q[ci];
        k_taps[w] = cs_k[ci];
        q_acc += static_cast<float>(q_taps[w]) * conv_w(w_q_t, hk, w, dim);
        k_acc += static_cast<float>(k_taps[w]) * conv_w(w_k_t, hk, w, dim);
      }
      const scalar_t q_new = x_q[bos * x_row + i_h * kDimK + k];
      const scalar_t k_new = x_k[bos * x_row + i_h * kDimK + k];
      q_acc += static_cast<float>(q_new) * conv_w(w_q_t, hk, kKernelWidth - 1, dim);
      k_acc += static_cast<float>(k_new) * conv_w(w_k_t, hk, kKernelWidth - 1, dim);

      // bit-exact 3-tap shift writeback: [t1,t2,new]  (per-channel independent)
      const int64_t base = (int64_t)slot * conv_slot + hk;
      cs_q[base] = q_taps[1];
      cs_q[base + kPackedDim] = q_taps[2];
      cs_q[base + 2 * kPackedDim] = q_new;
      cs_k[base] = k_taps[1];
      cs_k[base + kPackedDim] = k_taps[2];
      cs_k[base + 2 * kPackedDim] = k_new;

      q_out = ppu_silu(q_acc);
      k_out = ppu_silu(k_acc);

      // WN4 keeps the pre-P9 q+k+decay ownership and instruction order.
      if constexpr (WARP_NUM == 4) {
        const float g_raw =
            static_cast<float>(g[bos * dim + i_hv * kDimK + k]) + dt_bias[hk];
        float decay;
        if constexpr (USE_LOWER_BOUND) {
          decay = __expf(lower_bound * ppu_sigmoid(exp_a * g_raw));
        } else {
          decay = __expf(-exp_a * ppu_softplus(g_raw));
        }
        smem_bg[k] = decay;
      }
    }

    // WN8 Stage-A upper half: one value channel and one decay channel per
    // thread. This runs concurrently with q+k on warps 0-3.
    if constexpr (WARP_NUM == 8) {
      if (warp_id >= 4) {
        const int ch = (warp_id - 4) * 32 + lane_id;
        const int hvv = hv_off + ch;
        float v_acc = (bias_v != nullptr) ? bias_v[hvv] : 0.0f;
        scalar_t v_taps[kConvStateWidth];
        #pragma unroll
        for (int w = 0; w < kConvStateWidth; ++w) {
          const int64_t ci =
              (int64_t)slot * conv_slot + hvv + (int64_t)w * kPackedDim;
          v_taps[w] = cs_v[ci];
          v_acc += static_cast<float>(v_taps[w]) * conv_w(w_v_t, hvv, w, dim);
        }
        const scalar_t v_new = x_v[bos * x_row + i_hv * kDimV + ch];
        v_acc += static_cast<float>(v_new) *
                 conv_w(w_v_t, hvv, kKernelWidth - 1, dim);
        const int64_t base = (int64_t)slot * conv_slot + hvv;
        cs_v[base] = v_taps[1];
        cs_v[base + kPackedDim] = v_taps[2];
        cs_v[base + 2 * kPackedDim] = v_new;
        smem_bv[ch] = static_cast<scalar_t>(ppu_silu(v_acc));

        const int hk = hk_off + ch;
        const float g_raw =
            static_cast<float>(g[bos * dim + i_hv * kDimK + ch]) + dt_bias[hk];
        float decay;
        if constexpr (USE_LOWER_BOUND) {
          decay = __expf(lower_bound * ppu_sigmoid(exp_a * g_raw));
        } else {
          decay = __expf(-exp_a * ppu_softplus(g_raw));
        }
        smem_bg[ch] = decay;
      }
    }

    // ------------------- q/k L2 normalisation, block-wide -------------------
    // WN8 only uses warps 0-3 for q/k.  Keep the upper four warps out of the
    // shuffle/re-sum/rsqrt path instead of publishing and consuming zero
    // partials.  WN4 keeps its original four-warp reduction unchanged.
    constexpr int QK_WARPS = (WARP_NUM == 8) ? 4 : WARP_NUM;
    // P6: pack the q/k warp partials as float2 so the block-wide re-sum issues
    // QK_WARPS vectorised 8-byte smem loads instead of 2*QK_WARPS scalar loads.
    // smem_reduction and smem_reduction2 are contiguous (2*WARP_NUM floats,
    // 8-byte aligned), so this float2 view spans exactly that region. The P3
    // single-barrier design is preserved (still one __syncthreads).
    float2* smem_qk_pair = reinterpret_cast<float2*>(smem_reduction);
    if (qk_channel >= 0 && qk_channel < K) {
      float q_sq = q_out * q_out;
      float k_sq = k_out * k_out;
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        q_sq += __shfl_down_sync(0xffffffff, q_sq, offset);
        k_sq += __shfl_down_sync(0xffffffff, k_sq, offset);
      }
      if (lane_id == 0) {
        smem_qk_pair[warp_id] = make_float2(q_sq, k_sq);
      }
    }
    __syncthreads();

    // Only q/k owner threads need the norm.  With WN8 this reduces thread-level
    // float2 loads from 256*8 to 128*4 and removes four warps' shuffle/rsqrt
    // work, while preserving the partial-sum order (the omitted terms were +0).
    if (qk_channel >= 0 && qk_channel < K) {
      float total_q = 0.0f;
      float total_k = 0.0f;
      #pragma unroll
      for (int w = 0; w < QK_WARPS; ++w) {
        float2 p = smem_qk_pair[w];
        total_q += p.x;
        total_k += p.y;
      }
      const float q_norm = rsqrtf(total_q + 1.0e-6f) * scale;
      const float k_norm = rsqrtf(total_k + 1.0e-6f);
      smem_q[qk_channel] = q_out * q_norm;
      smem_k[qk_channel] = k_out * k_norm;
    }

    // WN4 retains the original post-normalisation v-conv phase. With BV=128,
    // each of the 128 threads executes exactly one loop iteration.
    if constexpr (WARP_NUM == 4) {
      for (int ch = tid; ch < BV; ch += WARP_NUM * 32) {
        const int hvv = hv_off + ch;
        float v_acc = (bias_v != nullptr) ? bias_v[hvv] : 0.0f;
        scalar_t v_taps[kConvStateWidth];
        #pragma unroll
        for (int w = 0; w < kConvStateWidth; ++w) {
          const int64_t ci =
              (int64_t)slot * conv_slot + hvv + (int64_t)w * kPackedDim;
          v_taps[w] = cs_v[ci];
          v_acc += static_cast<float>(v_taps[w]) * conv_w(w_v_t, hvv, w, dim);
        }
        const scalar_t v_new = x_v[bos * x_row + i_hv * kDimV + ch];
        v_acc += static_cast<float>(v_new) *
                 conv_w(w_v_t, hvv, kKernelWidth - 1, dim);
        const int64_t base = (int64_t)slot * conv_slot + hvv;
        cs_v[base] = v_taps[1];
        cs_v[base + kPackedDim] = v_taps[2];
        cs_v[base + 2 * kPackedDim] = v_new;
        smem_bv[ch] = static_cast<scalar_t>(ppu_silu(v_acc));
      }
    }

    if (tid == 0) {
      const float b_val = static_cast<float>(beta[bos * beta_row + i_hv]);
      smem_b_beta[0] = ppu_sigmoid(b_val);
    }
  }

  __syncthreads();

  float4 q_vec = *reinterpret_cast<const float4*>(&smem_q[k_idx]);
  float4 k_vec = *reinterpret_cast<const float4*>(&smem_k[k_idx]);

  // =========================================================================
  // Stage B (recurrent core) + Stage C (gated RMSNorm), single v-head.
  // =========================================================================
  int task_idx = 0;
  #pragma unroll
  for (int hv_step = 0; hv_step < HV_PER_CTA; ++hv_step) {
    float b_beta = smem_b_beta[hv_step];
    float4 bg_vec = *reinterpret_cast<const float4*>(smem_bg + hv_step * K + k_idx);
    int cur_hv = hv_base + hv_step;

    float o_reg[V_ITERS];
    if constexpr (QUAD_ROW_CORE) {
      // Four independent state rows expose enough arithmetic to overlap the
      // four-stage refill and halve the number of wait points.  Every row
      // keeps the original FMA and xor-butterfly order.
      #pragma unroll
      for (int v_quad = 0; v_quad < V_ITERS; v_quad += 4) {
        cp_async_wait<0>();
        if (hv_step == 0 && v_quad == 0) {
#ifdef __HGGC_ARCH__
          __ppu_barrier_sync_defer(0, WARP_NUM * 32, 1);
#endif
        }

        float4 h[4];
        float v_val[4];
        float vd[4];
        float v_new[4];
        float o_val[4];

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          int task = task_idx + r;
          int stage = task % PIPE_STAGES;
          h[r] = *reinterpret_cast<float4*>(
              &smem_h_flat[stage * WARP_NUM * 128 + warp_id * 128 + k_idx]);
          v_val[r] = static_cast<float>(
              smem_bv[hv_step * BV + warp_id + (v_quad + r) * WARP_NUM]);
          h[r].x *= bg_vec.x; h[r].y *= bg_vec.y;
          h[r].z *= bg_vec.z; h[r].w *= bg_vec.w;
          vd[r] = h[r].x * k_vec.x + h[r].y * k_vec.y +
                  h[r].z * k_vec.z + h[r].w * k_vec.w;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
          #pragma unroll
          for (int r = 0; r < 4; ++r) {
            vd[r] += __shfl_xor_sync(0xffffffff, vd[r], offset);
          }
        }

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          v_new[r] = (v_val[r] - vd[r]) * b_beta;
          int next = task_idx + r + PIPE_STAGES;
          issue_async_stage(next, next % PIPE_STAGES, next < TOTAL_TASKS);
          cp_async_commit();
        }

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          h[r].x += k_vec.x * v_new[r]; h[r].y += k_vec.y * v_new[r];
          h[r].z += k_vec.z * v_new[r]; h[r].w += k_vec.w * v_new[r];
          float* dst = h0_cta_base + hv_step * kHeadElems +
                       (warp_id + (v_quad + r) * WARP_NUM) * 128;
#ifdef __HGGC_ARCH__
          *(Global v4float*)(dst) = *(v4float*)&h[r];
#else
          *reinterpret_cast<float4*>(dst) = h[r];
#endif
          o_val[r] = h[r].x * q_vec.x + h[r].y * q_vec.y +
                     h[r].z * q_vec.z + h[r].w * q_vec.w;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
          #pragma unroll
          for (int r = 0; r < 4; ++r) {
            o_val[r] += __shfl_xor_sync(0xffffffff, o_val[r], offset);
          }
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          if constexpr (PARALLEL_EPILOGUE) {
            if (lane_id == 0) {
              smem_o[warp_id + (v_quad + r) * WARP_NUM] = o_val[r];
            }
          }
          o_reg[v_quad + r] = o_val[r];
        }
        task_idx += 4;
      }
    } else {
    // 2-rows-per-iteration recurrent core (mirrors ref warp_reduce_sum_pair):
    // each iteration processes two consecutive value-rows a/b so the two
    // k.S (and q.S) dot products share a single __shfl_xor_sync butterfly.
    // xor-reduction leaves the full sum in ALL lanes, so the separate
    // __shfl_sync(...,0) broadcast used by the old 1-row path is dropped.
    #pragma unroll
    for (int v_pair = 0; v_pair < V_ITERS; v_pair += 2) {
      int task_a = task_idx;
      int task_b = task_idx + 1;
      int stage_a = task_a % STAGE_NUM;
      int stage_b = task_b % STAGE_NUM;

      cp_async_wait<STAGE_NUM - 2>();
      if (hv_step == 0 && v_pair == 0) {
#ifdef __HGGC_ARCH__
        __ppu_barrier_sync_defer(0, WARP_NUM * 32, 1);
#endif
      }
      float4 h_a = *reinterpret_cast<float4*>(
          &smem_h_flat[stage_a * WARP_NUM * 128 + warp_id * 128 + k_idx]);
      float4 h_b = *reinterpret_cast<float4*>(
          &smem_h_flat[stage_b * WARP_NUM * 128 + warp_id * 128 + k_idx]);

      float v_val_a = static_cast<float>(smem_bv[hv_step * BV + warp_id + v_pair * WARP_NUM]);
      float v_val_b = static_cast<float>(smem_bv[hv_step * BV + warp_id + (v_pair + 1) * WARP_NUM]);

      // S = S * exp(g)  (per-channel diagonal decay), both rows
      h_a.x *= bg_vec.x; h_a.y *= bg_vec.y; h_a.z *= bg_vec.z; h_a.w *= bg_vec.w;
      h_b.x *= bg_vec.x; h_b.y *= bg_vec.y; h_b.z *= bg_vec.z; h_b.w *= bg_vec.w;

      // v_delta = k . S  (both rows, paired butterfly -> full sum in all lanes)
      float vd_a = h_a.x * k_vec.x + h_a.y * k_vec.y + h_a.z * k_vec.z + h_a.w * k_vec.w;
      float vd_b = h_b.x * k_vec.x + h_b.y * k_vec.y + h_b.z * k_vec.z + h_b.w * k_vec.w;
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        vd_a += __shfl_xor_sync(0xffffffff, vd_a, offset);
        vd_b += __shfl_xor_sync(0xffffffff, vd_b, offset);
      }

      // v_new = beta * (v - k.S)
      float v_new_a = (v_val_a - vd_a) * b_beta;
      float v_new_b = (v_val_b - vd_b) * b_beta;

      // prefetch the next two H0 stages before the rank-1 update math
      int next_a = task_a + STAGE_NUM;
      int next_b = task_b + STAGE_NUM;
      issue_async_stage(next_a, next_a % STAGE_NUM, next_a < TOTAL_TASKS);
      cp_async_commit();
      issue_async_stage(next_b, next_b % STAGE_NUM, next_b < TOTAL_TASKS);
      cp_async_commit();

      // S = S + k x v_new  (both rows)
      h_a.x += k_vec.x * v_new_a; h_a.y += k_vec.y * v_new_a;
      h_a.z += k_vec.z * v_new_a; h_a.w += k_vec.w * v_new_a;
      h_b.x += k_vec.x * v_new_b; h_b.y += k_vec.y * v_new_b;
      h_b.z += k_vec.z * v_new_b; h_b.w += k_vec.w * v_new_b;

      // persist both updated recurrent-state rows
      {
        float* dst_a = h0_cta_base + hv_step * kHeadElems + (warp_id + v_pair * WARP_NUM) * 128;
        float* dst_b = h0_cta_base + hv_step * kHeadElems + (warp_id + (v_pair + 1) * WARP_NUM) * 128;
#ifdef __HGGC_ARCH__
        *(Global v4float*)(dst_a) = *(v4float*)&h_a;
        *(Global v4float*)(dst_b) = *(v4float*)&h_b;
#else
        *reinterpret_cast<float4*>(dst_a) = h_a;
        *reinterpret_cast<float4*>(dst_b) = h_b;
#endif
      }

      // o = q . S  (both rows, paired butterfly -> full sum in all lanes)
      float o_a = h_a.x * q_vec.x + h_a.y * q_vec.y + h_a.z * q_vec.z + h_a.w * q_vec.w;
      float o_b = h_b.x * q_vec.x + h_b.y * q_vec.y + h_b.z * q_vec.z + h_b.w * q_vec.w;
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        o_a += __shfl_xor_sync(0xffffffff, o_a, offset);
        o_b += __shfl_xor_sync(0xffffffff, o_b, offset);
      }

      // The P7 path stages outputs for a 128-way epilogue.  The P6 path keeps
      // them in lane-0 registers and avoids the extra TSM round trip; archived
      // sweeps show that each path wins at different grid sizes.
      if constexpr (PARALLEL_EPILOGUE) {
        if (lane_id == 0) {
          smem_o[warp_id + v_pair * WARP_NUM] = o_a;
          smem_o[warp_id + (v_pair + 1) * WARP_NUM] = o_b;
        }
      }
      o_reg[v_pair] = o_a;         // full sum in all lanes
      o_reg[v_pair + 1] = o_b;
      task_idx += 2;
    }
    }

    // --------------------- Stage C: gated RMSNorm ------------------------
    // P3: down from 3 block barriers to 1. After warps publish their o^2
    // partials via a single __syncthreads, every thread redundantly sums the
    // 8 partials in registers (cheap, no second barrier). HV_PER_CTA == 1
    // means there is no next iteration, so the trailing "reuse smem" barrier
    // is dropped as well.
    float local_sumsq = 0.0f;
    if (lane_id == 0) {
      #pragma unroll
      for (int v_i = 0; v_i < V_ITERS; ++v_i) local_sumsq += o_reg[v_i] * o_reg[v_i];
      smem_reduction[warp_id] = local_sumsq;
    }
    __syncthreads();

    float rstd = 1.0f;
    if constexpr (APPLY_ONORM) {
      float total = 0.0f;
      #pragma unroll
      for (int w = 0; w < WARP_NUM; ++w) total += smem_reduction[w];
      rstd = rsqrtf(total / static_cast<float>(V) + norm_eps);
    }

    if constexpr (PARALLEL_EPILOGUE) {
      // P7: 128-way parallel output write. Each tid<BV thread reads its staged
      // output and writes exactly one gated-RMSNorm element.
      if (tid < BV) {
        const int v_col = tid;
        float out_val;
        if constexpr (APPLY_ONORM) {
          float w_norm = o_norm_weight[v_col];
          float g_val = static_cast<float>(
              g_out[i_n * onorm_row + cur_hv * kDimV + v_col]);
          float gate = ppu_sigmoid(g_val);
          out_val = smem_o[v_col] * rstd * w_norm * gate;
        } else {
          out_val = smem_o[v_col];
        }
        o[i_n * dim + cur_hv * kDimV + v_col] =
            static_cast<scalar_t>(out_val);
      }
    } else {
      // P6: one lane per warp writes its 16 resident rows. This avoids the
      // smem_o store/load path and is faster around the 384/768-CTA wave edges.
      if (lane_id == 0) {
        #pragma unroll
        for (int v_i = 0; v_i < V_ITERS; ++v_i) {
          const int v_col = warp_id + v_i * WARP_NUM;
          float out_val;
          if constexpr (APPLY_ONORM) {
            float w_norm = o_norm_weight[v_col];
            float g_val = static_cast<float>(
                g_out[i_n * onorm_row + cur_hv * kDimV + v_col]);
            float gate = ppu_sigmoid(g_val);
            out_val = o_reg[v_i] * rstd * w_norm * gate;
          } else {
            out_val = o_reg[v_i];
          }
          o[i_n * dim + cur_hv * kDimV + v_col] =
              static_cast<scalar_t>(out_val);
        }
      }
    }
    // HV_PER_CTA == 1: no next hv_step, so no trailing __syncthreads() needed.
  }

  cp_async_wait<0>();
}

}  // namespace fused_kda_decode_ppu
