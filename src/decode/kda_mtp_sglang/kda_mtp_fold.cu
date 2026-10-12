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

// PPU1.5 port of kda_replayssm_exact_fold_kernel + conv_window_scatter
// (SGLang KDA commit-after-verify). Ported in structure from the H20 CUDA C++
// version (csrc/_h20_kda_fold.cu.orig); pure CUDA-core, register-resident
// state, warp-shuffle reductions. Device intrinsics used are pla's standard
// __expf / rsqrtf / __bfloat16* conversions -- there is no per-namespace
// compat shim.
//
// Arithmetic order is the recurrent baseline's IS_KDA branch verbatim:
//   division-form L2 norm with eps inside the sqrt, per-K gate decay,
//   h *= exp(gk), then decay -> delta -> rank-1.
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <cstdint>

#include "kda_mtp_abi.h"

namespace {

using BF16 = __ppu_bfloat16;

constexpr int KDIM = 128;
#ifndef KDA_FOLD_BV
#define KDA_FOLD_BV 32
#endif
constexpr int BV = KDA_FOLD_BV;
#ifndef KDA_FOLD_THREADS
// PPU1.5 sweep (card 6, THREADS x BV over 64/128/256 x 16/32/64): THREADS=256,
// BV=32 (=> KGRPS=8, K_PER_THREAD=16, F4_PER_THREAD=4) is the consistent best
// (bs=1/8/64: 62.6/675.2/5240us vs 73.8/765.7/5890 at THREADS=128). Doubling
// threads/v-column halves the register-resident state tile per thread, lifting
// occupancy on this state-bandwidth-bound kernel. (H20's best was 128; PPU differs.)
#define KDA_FOLD_THREADS 256
#endif
constexpr int THREADS = KDA_FOLD_THREADS;
constexpr int KGRPS_CHK = THREADS / KDA_FOLD_BV;
static_assert(THREADS % 32 == 0, "THREADS must be whole warps");
static_assert(THREADS % KDA_FOLD_BV == 0, "BV must divide THREADS");
static_assert(KGRPS_CHK >= 1 && KGRPS_CHK <= 32 && 32 % KGRPS_CHK == 0,
              "KGRPS must divide the 32 float4 of a K row and fit in a warp");
constexpr int KGRPS = THREADS / BV;                  // threads per v column
constexpr int K_PER_THREAD = KDIM / KGRPS;
constexpr int VEC = 4;                               // float4
constexpr int F4_PER_THREAD = K_PER_THREAD / VEC;

// Interleaved float4 ownership: thread k_grp takes float4 index k_grp + KGRPS*i,
// so the KGRPS threads of one v column (consecutive tids -> same warp) cover
// contiguous bytes per iteration and consecutive i walks the row forward.
__device__ __forceinline__ int f4_index(int k_grp, int i) {
  return k_grp + KGRPS * i;
}
constexpr float L2_EPS = 1e-6f;

__device__ __forceinline__ float warp_bfly_add(float x, int width) {
  for (int off = width >> 1; off >= 1; off >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, off, 32);
  return x;
}

constexpr int K_PER_LANE = KDIM / 32;
static_assert(KDIM % 32 == 0, "K must spread evenly over a warp");

// Stage the whole accepted window once, up front (H20 "attempt C"): barriers
// drop from 3*n_commit to exactly one. Each warp takes whole steps so its ksq
// reduction is warp-local; the single __syncthreads only publishes the table.
__device__ __forceinline__ void stage_window(
    float* __restrict__ s_kk, float* __restrict__ s_dc, int warp, int lane,
    int nwarps, int n_commit, const BF16* __restrict__ rawk_p,
    const float* __restrict__ gk_p) {
  for (int t = warp; t < n_commit; t += nwarps) {
    const BF16* rk = rawk_p + (long)t * KDIM;
    const float* gk = gk_p + (long)t * KDIM;
    float kv[K_PER_LANE];
    float sq = 0.f;
#pragma unroll
    for (int m = 0; m < K_PER_LANE; ++m) {
      kv[m] = __bfloat162float(rk[lane + m * 32]);
      sq += kv[m] * kv[m];
    }
    sq = warp_bfly_add(sq, 32);
    const float kinv = 1.0f / sqrtf(sq + L2_EPS);
    float* kk_t = s_kk + (long)t * KDIM;
    float* dc_t = s_dc + (long)t * KDIM;
#pragma unroll
    for (int m = 0; m < K_PER_LANE; ++m) {
      const int k = lane + m * 32;
      kk_t[k] = kv[m] * kinv;
      dc_t[k] = __expf(gk[k]);
    }
  }
  __syncthreads();
}

// h0     [layers, slots, HV, V, K] fp32, folded in place
// rawv   [layers, slots, HV, L, V] bf16
// rawk   [layers, slots, H,  L, K] bf16
// gk     [layers, slots, HV, L, K] fp32
// beta   [layers, slots, HV, L]    fp32
__global__ __launch_bounds__(THREADS) void fold_kernel(
    float* __restrict__ h0, const BF16* __restrict__ rawv,
    const BF16* __restrict__ rawk, const float* __restrict__ gkc,
    const float* __restrict__ betac, const int32_t* __restrict__ slot_idx,
    const int32_t* __restrict__ accept_lens,
    const int32_t* __restrict__ track_idx,
    const int32_t* __restrict__ track_step, long stride_h_layer,
    long stride_h_slot, long stride_rawv_layer, long stride_rawv_slot,
    long stride_rawk_layer, long stride_rawk_slot, long stride_gk_layer,
    long stride_gk_slot, long stride_beta_layer, long stride_beta_slot, int H,
    int HV, int V, int L, int null_block_id, bool has_track) {
  const int i_vtile = blockIdx.x;
  const int i_n = blockIdx.y;
  const long i_hvl = blockIdx.z;
  const long i_layer = i_hvl / HV;
  const int i_hv = (int)(i_hvl % HV);
  const int i_h = i_hv / (HV / H);

  const int slot = slot_idx[i_n];
  if (slot <= null_block_id) return;
  const int n_commit = accept_lens[i_n];
  if (n_commit <= 0) return;

  const int tid = threadIdx.x;
  const int v_local = tid / (THREADS / BV);
  const int k_grp = tid % (THREADS / BV);
  const int v = i_vtile * BV + v_local;
  const bool v_ok = v < V;

  float* h_base = h0 + i_layer * stride_h_layer + slot * stride_h_slot +
                      (long)i_hv * V * KDIM;
  // element (v, k) lives at v*K + k -- the Triton tile is [K, V] transposed
  float* p_h_row = h_base + (long)v * KDIM;

  const BF16* rawk_p =
      rawk + i_layer * stride_rawk_layer + slot * stride_rawk_slot +
      (long)i_h * L * KDIM;
  const BF16* rawv_p =
      rawv + i_layer * stride_rawv_layer + slot * stride_rawv_slot +
      (long)i_hv * L * V;
  const float* gk_p = gkc + i_layer * stride_gk_layer + slot * stride_gk_slot +
                      (long)i_hv * L * KDIM;
  const float* beta_p = betac + i_layer * stride_beta_layer +
                        slot * stride_beta_slot + (long)i_hv * L;

  extern __shared__ char smem_raw[];
  float* const s_kk = reinterpret_cast<float*>(smem_raw);
  float* const s_dc = s_kk + (long)L * KDIM;

  const float4* p_h4 = reinterpret_cast<const float4*>(p_h_row);
  float4 st[F4_PER_THREAD];
#pragma unroll
  for (int i = 0; i < F4_PER_THREAD; ++i)
    st[i] = v_ok ? p_h4[f4_index(k_grp, i)] : make_float4(0.f, 0.f, 0.f, 0.f);
  stage_window(s_kk, s_dc, tid >> 5, tid & 31, THREADS / 32, n_commit, rawk_p,
               gk_p);

  const int t_track = has_track ? track_step[i_n] : -1;
  const int slot_track = has_track ? track_idx[i_n] : null_block_id;

  for (int t = 0; t < n_commit; ++t) {
    const float4* sh_kk4 =
        reinterpret_cast<const float4*>(s_kk + (long)t * KDIM);
    const float4* sh_dc4 =
        reinterpret_cast<const float4*>(s_dc + (long)t * KDIM);

    float shk = 0.f;
#pragma unroll
    for (int i = 0; i < F4_PER_THREAD; ++i) {
      const int f4 = f4_index(k_grp, i);
      const float4 k4 = sh_kk4[f4];
      const float4 d4 = sh_dc4[f4];
      st[i].x *= d4.x; st[i].y *= d4.y; st[i].z *= d4.z; st[i].w *= d4.w;
      shk += st[i].x * k4.x + st[i].y * k4.y + st[i].z * k4.z +
             st[i].w * k4.w;
    }
    shk = warp_bfly_add(shk, KGRPS);

    const float vv = v_ok ? __bfloat162float(rawv_p[(long)t * V + v]) : 0.f;
    const float vn = (vv - shk) * beta_p[t];

#pragma unroll
    for (int i = 0; i < F4_PER_THREAD; ++i) {
      const float4 k4 = sh_kk4[f4_index(k_grp, i)];
      st[i].x = fmaf(vn, k4.x, st[i].x);
      st[i].y = fmaf(vn, k4.y, st[i].y);
      st[i].z = fmaf(vn, k4.z, st[i].z);
      st[i].w = fmaf(vn, k4.w, st[i].w);
    }

    if (has_track && t == t_track && slot_track > null_block_id && v_ok) {
      float4* p_tr = reinterpret_cast<float4*>(
          h0 + i_layer * stride_h_layer + (long)slot_track * stride_h_slot +
          (long)i_hv * V * KDIM + (long)v * KDIM);
#pragma unroll
      for (int i = 0; i < F4_PER_THREAD; ++i) p_tr[f4_index(k_grp, i)] = st[i];
    }
  }

  if (v_ok) {
    float4* o4 = reinterpret_cast<float4*>(p_h_row);
#pragma unroll
    for (int i = 0; i < F4_PER_THREAD; ++i) o4[f4_index(k_grp, i)] = st[i];
  }
}

// dst [layers, cache, dim, KM1] contiguous;
// src [layers, spec, steps, dim, KM1] overlapping strided view.
// Walk d (not the flat element) so KM1 is a template param, the w loop unrolls,
// no integer division is emitted, and each thread owns D_PER_THREAD rows.
template <typename T, int KM1, int D_PER_THREAD>
__global__ void conv_scatter_kernel(
    const T* __restrict__ src, T* __restrict__ dst,
    const int32_t* __restrict__ dst_idx, const int32_t* __restrict__ step_idx,
    int dim, long s_layer, long s_req, long s_step, long s_dim, long s_win,
    long d_layer, long d_req, long d_dim, long d_win,
    int src_req_size, int src_step_size,
    int dst_req_size) {
  const int i_req = blockIdx.x;
  const long i_layer = blockIdx.y;
  const int step = step_idx[i_req];
  if (step < 0) return;
  const int d_i = dst_idx[i_req];
  if (!(d_i >= 0 && d_i < dst_req_size && i_req < src_req_size &&
        step < src_step_size))
    return;

  const long src_base = i_layer * s_layer + (long)i_req * s_req +
                        (long)step * s_step;
  const long dst_base = i_layer * d_layer + (long)d_i * d_req;
  const int d0 = blockIdx.z * (blockDim.x * D_PER_THREAD) + threadIdx.x;

#pragma unroll
  for (int r = 0; r < D_PER_THREAD; ++r) {
    const int d = d0 + r * (int)blockDim.x;
    if (d >= dim) break;
    const T* sp = src + src_base + (long)d * s_dim;
    T* dp = dst + dst_base + (long)d * d_dim;
#pragma unroll
    for (int w = 0; w < KM1; ++w) dp[(long)w * d_win] = sp[(long)w * s_win];
  }
}

}  // namespace

namespace pla {
namespace kda_mtp {

void launch_fold(
    float* h0, const void* rawv, const void* rawk,
    const float* gk, const float* beta,
    const int32_t* slot_idx, const int32_t* accept_lens,
    const int32_t* track_idx, const int32_t* track_step,
    int64_t h0_s0, int64_t h0_s1,
    int64_t rawv_s0, int64_t rawv_s1,
    int64_t rawk_s0, int64_t rawk_s1,
    int64_t gk_s0, int64_t gk_s1,
    int64_t beta_s0, int64_t beta_s1,
    int layers, int B, int num_k_heads, int HV, int V, int L,
    int null_block_id, bool has_track, hggcStream_t stream) {
  dim3 grid((V + BV - 1) / BV, B, static_cast<unsigned long>(HV) * layers);
  const int smem = static_cast<int>(sizeof(float)) * 2 * L * KDIM;
  if (smem > 48 * 1024) {
    hggcFuncSetAttribute(fold_kernel,
                         hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  fold_kernel<<<grid, THREADS, smem, stream>>>(
      h0, static_cast<const BF16*>(rawv), static_cast<const BF16*>(rawk),
      gk, beta, slot_idx, accept_lens, track_idx, track_step,
      h0_s0, h0_s1, rawv_s0, rawv_s1, rawk_s0, rawk_s1,
      gk_s0, gk_s1, beta_s0, beta_s1,
      num_k_heads, HV, V, L, null_block_id, has_track);
}

namespace {
constexpr int kConvBlock = 256;
constexpr int kConvDPerThread = 4;

template <typename T, int KM1>
void launch_conv_scatter_typed(
    const void* src, void* dst,
    const int32_t* dst_idx, const int32_t* step_idx,
    int layers, int dim, int reqs,
    int64_t src_s0, int64_t src_s1, int64_t src_s2,
    int64_t src_s3, int64_t src_s4,
    int64_t dst_s0, int64_t dst_s1, int64_t dst_s2, int64_t dst_s3,
    int src_reqs, int src_steps, int dst_reqs,
    hggcStream_t stream) {
  const int per_block = kConvBlock * kConvDPerThread;
  dim3 grid(reqs, layers, (dim + per_block - 1) / per_block);
  conv_scatter_kernel<T, KM1, kConvDPerThread>
      <<<grid, kConvBlock, 0, stream>>>(
          static_cast<const T*>(src), static_cast<T*>(dst), dst_idx, step_idx,
          dim, src_s0, src_s1, src_s2, src_s3, src_s4, dst_s0, dst_s1,
          dst_s2, dst_s3, src_reqs, src_steps, dst_reqs);
}

template <typename T>
void dispatch_conv_window(
    int km1, const void* src, void* dst,
    const int32_t* dst_idx, const int32_t* step_idx,
    int layers, int dim, int reqs,
    int64_t src_s0, int64_t src_s1, int64_t src_s2,
    int64_t src_s3, int64_t src_s4,
    int64_t dst_s0, int64_t dst_s1, int64_t dst_s2, int64_t dst_s3,
    int src_reqs, int src_steps, int dst_reqs,
    hggcStream_t stream) {
#define PLA_CONV_LAUNCH(KM1)                                                   \
  launch_conv_scatter_typed<T, KM1>(                                          \
      src, dst, dst_idx, step_idx, layers, dim, reqs,                         \
      src_s0, src_s1, src_s2, src_s3, src_s4, dst_s0, dst_s1,                 \
      dst_s2, dst_s3, src_reqs, src_steps, dst_reqs, stream)
  switch (km1) {
    case 1: PLA_CONV_LAUNCH(1); break;
    case 2: PLA_CONV_LAUNCH(2); break;
    case 3: PLA_CONV_LAUNCH(3); break;
    case 4: PLA_CONV_LAUNCH(4); break;
  }
#undef PLA_CONV_LAUNCH
}
}  // namespace

void launch_conv_scatter(
    ConvDType dtype, int km1,
    const void* src, void* dst,
    const int32_t* dst_idx, const int32_t* step_idx,
    int layers, int dim, int reqs,
    int64_t src_s0, int64_t src_s1, int64_t src_s2,
    int64_t src_s3, int64_t src_s4,
    int64_t dst_s0, int64_t dst_s1, int64_t dst_s2, int64_t dst_s3,
    int src_reqs, int src_steps, int dst_reqs,
    hggcStream_t stream) {
#define PLA_CONV_ARGS                                                          \
  km1, src, dst, dst_idx, step_idx, layers, dim, reqs,                        \
      src_s0, src_s1, src_s2, src_s3, src_s4, dst_s0, dst_s1,                 \
      dst_s2, dst_s3, src_reqs, src_steps, dst_reqs, stream
  if (dtype == ConvDType::kBFloat16) {
    dispatch_conv_window<BF16>(PLA_CONV_ARGS);
  } else {
    dispatch_conv_window<float>(PLA_CONV_ARGS);
  }
#undef PLA_CONV_ARGS
}

}  // namespace kda_mtp
}  // namespace pla
