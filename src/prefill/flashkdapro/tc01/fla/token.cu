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

#include "heads.h"
#include "fla_activation.h"
/* Fixed batch-1 FLA token-intra stage for tc01, H=8, K=128, BT=64, BC=16. */
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <cstdint>
#include <utility>

#include "cute/tensor.hpp"

#ifndef FLA_TOKEN_INTRA_PUBLISH_QG
#define FLA_TOKEN_INTRA_PUBLISH_QG 0
#endif

// Keep q*scale in FP32 until Qg publication; an intermediate BF16 rounding
// loses accuracy in cancellation-sensitive outputs. Zero is a negative control.
#ifndef FLA_QG_FP32_SCALED_Q
#define FLA_QG_FP32_SCALED_Q 1
#endif
#if FLA_QG_FP32_SCALED_Q != 0 && FLA_QG_FP32_SCALED_Q != 1
#error "FLA_QG_FP32_SCALED_Q must be 0 or 1"
#endif

namespace fla_glm53_token_intra_cute {

#if FLA_HEADS > 0
constexpr int kH = FLA_HEADS;
#else
#define kH FLA_DEVICE_HEADS(y)
#endif
constexpr int kK = 128;
constexpr int kBT = 64;
constexpr int kBC = 16;
constexpr int kThreads = 32;

union alignas(8) Bf16x4 {
  uint2 vector;
  __ppu_bfloat162 pairs[2];
};

__device__ __forceinline__ void load_bf16x4(
    const __ppu_bfloat16* address, float (&values)[4]) {
  Bf16x4 packed;
  packed.vector = *reinterpret_cast<const uint2*>(address);
#pragma unroll
  for (int pair_index = 0; pair_index < 2; ++pair_index) {
    const float2 pair = __bfloat1622float2(packed.pairs[pair_index]);
    values[pair_index * 2] = pair.x;
    values[pair_index * 2 + 1] = pair.y;
  }
}

__device__ __forceinline__ void load_float4(
    const float* address, float (&values)[4]) {
  const float4 packed = *reinterpret_cast<const float4*>(address);
  values[0] = packed.x;
  values[1] = packed.y;
  values[2] = packed.z;
  values[3] = packed.w;
}

__device__ __forceinline__ float local_sum4(float const (&x)[4]) {
  // Match the frozen LLIR's lane-local reduction before the five shuffles:
  // (((x0 + x1) + x2) + x3).
  const float sum01 = x[0] + x[1];
  const float sum012 = sum01 + x[2];
  return sum012 + x[3];
}

__device__ __forceinline__ float full_warp_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
  }
  return value;
}

#if FLA_TOKEN_INTRA_PUBLISH_QG
__device__ __forceinline__ void publish_qg_vec4(
    float const (&q_i)[4], float const (&g_i)[4],
    __ppu_bfloat16* __restrict__ qg, int64_t global_i, int head, int first_d,
    float scale) {
  float rounded_q[4];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
#if FLA_QG_FP32_SCALED_Q
    rounded_q[item] = __fmul_rn(q_i[item], scale);
#else
    rounded_q[item] = __bfloat162float(__float2bfloat16_rn(q_i[item] * scale));
#endif
  }
  constexpr float kLog2E = 0x1.715476p+0f;
  Bf16x4 result;
#pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    const int low = 2 * pair;
    const int high = low + 1;
    result.pairs[pair] = __floats2bfloat162_rn(
        rounded_q[low] * __exp2f(g_i[low] * kLog2E),
        rounded_q[high] * __exp2f(g_i[high] * kLog2E));
  }
  const int64_t offset =
      (global_i * static_cast<int64_t>(kH) + head) * kK + first_d;
  *reinterpret_cast<uint2*>(qg + offset) = result.vector;
}
#endif

extern "C" __global__ __launch_bounds__(32, 1)
__block_size__((32, 1, 1))
void token_intra_bf16_d128_bt64_bc16_h8_kernel(
    const __ppu_bfloat16* __restrict__ q,
    const __ppu_bfloat16* __restrict__ k,
    const float* __restrict__ g,
    const __ppu_bfloat16* __restrict__ beta_raw,
    int64_t beta_stride,
    __ppu_bfloat16* __restrict__ aqk,
    float* __restrict__ akk,
    __ppu_bfloat16* __restrict__ qg,
    float scale,
    const int32_t* __restrict__ cu_seqlens,
    int n,
    int total_tokens) {
  using namespace cute;

  // Dynamic extents and strides are int64 so address formation does not rely
  // on an implicit int32 product bound. Coordinates are supplied to CuTe;
  // there is no parallel hand-flattened indexing path for the six tensors.
  const int64_t total64 = static_cast<int64_t>(total_tokens);
  Tensor gQ = make_tensor(
      make_gmem_ptr(q),
      make_layout(make_shape(total64, FLA_CUTE_INT(kH), Int<kK>{}),
                  make_stride(FLA_CUTE_INT(kH * kK), Int<kK>{}, Int<1>{})));
  Tensor gK = make_tensor(
      make_gmem_ptr(k),
      make_layout(make_shape(total64, FLA_CUTE_INT(kH), Int<kK>{}),
                  make_stride(FLA_CUTE_INT(kH * kK), Int<kK>{}, Int<1>{})));
  Tensor gG = make_tensor(
      make_gmem_ptr(g),
      make_layout(make_shape(total64, FLA_CUTE_INT(kH), Int<kK>{}),
                  make_stride(FLA_CUTE_INT(kH * kK), Int<kK>{}, Int<1>{})));
  Tensor gBeta = make_tensor(
      make_gmem_ptr(beta_raw),
      make_layout(make_shape(total64, FLA_CUTE_INT(kH)),
                  make_stride(beta_stride, Int<1>{})));
  Tensor gAqk = make_tensor(
      make_gmem_ptr(aqk),
      make_layout(make_shape(total64, FLA_CUTE_INT(kH), Int<kBT>{}),
                  make_stride(FLA_CUTE_INT(kH * kBT), Int<kBT>{}, Int<1>{})));
  Tensor gAkk = make_tensor(
      make_gmem_ptr(akk),
      make_layout(make_shape(total64, FLA_CUTE_INT(kH), Int<kBC>{}),
                  make_stride(FLA_CUTE_INT(kH * kBC), Int<kBC>{}, Int<1>{})));
  Tensor gCu = make_tensor(
      make_gmem_ptr(cu_seqlens), make_layout(make_shape(n + 1)));

  const int global_i = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x) & 31;

  // Preserve the frozen IS_VARLEN specialization's bounded 20-step device
  // binary search even though this first public contract admits only N=1.
  int left = 0;
  int right = n;
#pragma unroll
  for (int step = 0; step < 20; ++step) {
    if (left < right) {
      const int mid = (left + right) >> 1;
      if (global_i < gCu(mid + 1)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
  }
  const int sequence = left;
  if (sequence >= n) return;
  const int bos = gCu(sequence);
  const int eos = gCu(sequence + 1);
  if (bos < 0 || eos > total_tokens) return;
  const int sequence_t = eos - bos;
  const int i = global_i - bos;
  if (i < 0 || i >= sequence_t) return;

  const int chunk_start = (i / kBT) * kBT;
  const int subchunk_start = chunk_start + ((i % kBT) / kBC) * kBC;
  const int first_d = lane * 4;
  float q_i[4];
  float k_i_beta[4];
  float g_i[4];
  const float beta_i = fla_sigmoid_bf16(gBeta(static_cast<int64_t>(global_i), head));
  load_bf16x4(&gQ(static_cast<int64_t>(global_i), head, first_d), q_i);
  load_bf16x4(&gK(static_cast<int64_t>(global_i), head, first_d), k_i_beta);
  load_float4(&gG(static_cast<int64_t>(global_i), head, first_d), g_i);
#if FLA_TOKEN_INTRA_PUBLISH_QG
  publish_qg_vec4(q_i, g_i, qg, static_cast<int64_t>(global_i), head,
                  first_d, scale);
#endif
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    k_i_beta[r] *= beta_i;
  }

  const int last_j = i < subchunk_start + kBC ? i : subchunk_start + kBC - 1;
  for (int j = subchunk_start; j <= last_j && j < sequence_t; ++j) {
    const int64_t global_j = static_cast<int64_t>(bos) + j;
    float k_j[4];
    float g_j[4];
    float aq_terms[4];
    float ak_terms[4];
    load_bf16x4(&gK(global_j, head, first_d), k_j);
    load_float4(&gG(global_j, head, first_d), g_j);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const float kg_j = k_j[r] * expf(g_i[r] - g_j[r]);
      aq_terms[r] = q_i[r] * kg_j;
      ak_terms[r] = k_i_beta[r] * kg_j;
    }
    const float aq = full_warp_sum(local_sum4(aq_terms));
    const float ak = full_warp_sum(local_sum4(ak_terms));
    if (lane == 0) {
      gAqk(static_cast<int64_t>(global_i), head, j % kBT) =
          __float2bfloat16_rn(aq * scale);
      gAkk(static_cast<int64_t>(global_i), head, j - subchunk_start) =
          j < i ? ak : 0.0f;
    }
  }
}

// N=1 chunk-tiled variant: one CTA owns one BT chunk of one head and two
// warps own each BC subchunk. Every (i, j) dot product keeps the per-token
// kernel's lane mapping (lane owns d = 4*lane..4*lane+3), lane-local terms and
// xor-butterfly tree. The butterfly is evaluated in transposed form: a batch
// of P pairs carries 2P partial sums, and each shuffle level halves the
// values a lane holds, so one shuffle retires one level of two reductions.
// A warp caches only eight j rows, which keeps four CTAs resident per CU.
constexpr int kChunkWarps = 2 * kBT / kBC;
constexpr int kChunkThreads = kChunkWarps * 32;
constexpr int kRows = kBC / 2;

// Reduces K lane partials with the masks 16, 8, 4, 2, 1. Levels with more
// than one value per lane exchange halves; remaining levels are the plain
// butterfly. Lane l returns value l / (32 / K).
template <int K>
__device__ __forceinline__ float transposed_warp_sum(float (&v)[K], int lane) {
  static_assert(K == 8 || K == 16 || K == 32, "unsupported batch width");
#pragma unroll
  for (int level = 0; level < 5; ++level) {
    const int mask = 16 >> level;
    const int count = K >> level;
    if (count > 1) {
      const bool upper = (lane & mask) != 0;
#pragma unroll
      for (int t = 0; t < count / 2; ++t) {
        const float keep = upper ? v[t + count / 2] : v[t];
        const float send = upper ? v[t] : v[t + count / 2];
        v[t] = keep + __shfl_xor_sync(0xffffffffu, send, mask, 32);
      }
    } else {
      v[0] += __shfl_xor_sync(0xffffffffu, v[0], mask, 32);
    }
  }
  return v[0];
}

struct ChunkTokenContext {
  const __ppu_bfloat16* __restrict__ q;
  const __ppu_bfloat16* __restrict__ k;
  const float* __restrict__ g;
  __ppu_bfloat16* __restrict__ aqk;
  float* __restrict__ akk;
  __ppu_bfloat16* __restrict__ qg;
  float scale;
  int64_t bos;
  int subchunk_start;
  int sequence_t;
  int head;
  int heads;
  int lane;
  float beta_lane;
};

// Caches the lane's slice of subchunk rows [R0, R0 + kRows) as j operands.
template <int R0>
__device__ __forceinline__ void chunk_load_rows(
    ChunkTokenContext const& c, float (&k_rows)[kRows][4],
    float (&g_rows)[kRows][4]) {
  const int first_d = c.lane * 4;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int j = c.subchunk_start + R0 + r;
    if (j < c.sequence_t) {
      const int64_t row = (c.bos + j) * c.heads + c.head;
      load_bf16x4(c.k + row * kK + first_d, k_rows[r]);
      load_float4(c.g + row * kK + first_d, g_rows[r]);
    } else {
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        k_rows[r][d] = 0.0f;
        g_rows[r][d] = 0.0f;
      }
    }
  }
}

template <bool Publish, int A>
__device__ __forceinline__ void chunk_token_operands(
    ChunkTokenContext const& c, float (&q_i)[4], float (&k_i_beta)[4],
    float (&g_i)[4]) {
  const int i = c.subchunk_start + A;
  const float beta_i = __shfl_sync(0xffffffffu, c.beta_lane, A, 32);
  const int first_d = c.lane * 4;
  if (i < c.sequence_t) {
    const int64_t global_i = c.bos + i;
    const int64_t row = global_i * c.heads + c.head;
    load_bf16x4(c.q + row * kK + first_d, q_i);
    load_bf16x4(c.k + row * kK + first_d, k_i_beta);
    load_float4(c.g + row * kK + first_d, g_i);
#if FLA_TOKEN_INTRA_PUBLISH_QG
    if constexpr (Publish)
      publish_qg_vec4(q_i, g_i, c.qg, global_i, c.head, first_d, c.scale);
#endif
  } else {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      q_i[r] = 0.0f;
      k_i_beta[r] = 0.0f;
      g_i[r] = 0.0f;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    k_i_beta[r] *= beta_i;
  }
}

template <int R0, int J, int P, int Slot>
__device__ __forceinline__ void chunk_pair_terms(
    float const (&k_rows)[kRows][4], float const (&g_rows)[kRows][4],
    float const (&q_i)[4], float const (&k_i_beta)[4], float const (&g_i)[4],
    float (&v)[2 * P]) {
  static_assert(J >= R0 && J < R0 + kRows, "j row is not cached");
  float kg_j[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    kg_j[r] = __fmul_rn(k_rows[J - R0][r],
                        expf(__fsub_rn(g_i[r], g_rows[J - R0][r])));
  }
  // local_sum4 leaves contraction to the compiler, which fuses a different
  // term depending on the surrounding code. Pin the per-token kernel's
  // compiled form: fma(x0, round(x1)), then fuse x2 and x3.
  const auto contracted_sum4 = [&](float const (&a)[4]) {
    const float sum01 = __fmaf_rn(a[0], kg_j[0], __fmul_rn(a[1], kg_j[1]));
    const float sum012 = __fmaf_rn(a[2], kg_j[2], sum01);
    return __fmaf_rn(a[3], kg_j[3], sum012);
  };
  v[Slot] = contracted_sum4(q_i);
  v[P + Slot] = contracted_sum4(k_i_beta);
}

// Slots [0, N1) hold token A1 with j = J1 + slot; slots [N1, N1 + N2) hold
// token A2 with j = J2 + slot - N1. Publish selects the batch that owns the
// token's Qg row.
template <bool Publish, int R0, int A1, int J1, int N1, int A2, int J2, int N2,
          int... Slots>
__device__ __forceinline__ void chunk_token_batch(
    ChunkTokenContext const& c, float const (&k_rows)[kRows][4],
    float const (&g_rows)[kRows][4], std::integer_sequence<int, Slots...>) {
  constexpr int P = N1 + N2;
  static_assert(sizeof...(Slots) == P, "batch size");
  if (c.subchunk_start + A1 >= c.sequence_t &&
      (N2 == 0 || c.subchunk_start + A2 >= c.sequence_t))
    return;
  float q1[4], kb1[4], g1[4], q2[4], kb2[4], g2[4];
  chunk_token_operands<Publish, A1>(c, q1, kb1, g1);
  if constexpr (N2 > 0)
    chunk_token_operands<Publish, A2>(c, q2, kb2, g2);
  float v[2 * P];
  (chunk_pair_terms<R0, (Slots < N1 ? J1 + Slots : J2 + Slots - N1), P, Slots>(
       k_rows, g_rows, Slots < N1 ? q1 : q2, Slots < N1 ? kb1 : kb2,
       Slots < N1 ? g1 : g2, v),
   ...);
  const float total = transposed_warp_sum<2 * P>(v, c.lane);
  constexpr int kSpread = 32 / (2 * P);
  if (c.lane % kSpread != 0) return;
  const int value = c.lane / kSpread;
  const int slot = value % P;
  const bool first = slot < N1;
  const int a = first ? A1 : A2;
  const int j = first ? J1 + slot : J2 + slot - N1;
  const int i = c.subchunk_start + a;
  if (i >= c.sequence_t) return;
  const int64_t row = (c.bos + i) * c.heads + c.head;
  if (value < P) {
    c.aqk[row * kBT + (c.subchunk_start + j) % kBT] =
        __float2bfloat16_rn(total * c.scale);
  } else {
    c.akk[row * kBC + j] = j < a ? total : 0.0f;
  }
}

__device__ __forceinline__ void token_intra_chunk_body(
    const __ppu_bfloat16* q,
    const __ppu_bfloat16* k,
    const float* __restrict__ g,
    const __ppu_bfloat16* __restrict__ beta_raw,
    int64_t beta_stride,
    __ppu_bfloat16* __restrict__ aqk,
    float* __restrict__ akk,
    __ppu_bfloat16* __restrict__ qg,
    float scale,
    const int32_t* __restrict__ cu_seqlens,
    int total_tokens) {
  const int bos = cu_seqlens[0];
  const int eos = cu_seqlens[1];
  if (bos < 0 || eos > total_tokens) return;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  ChunkTokenContext c{q, k, g, aqk, akk, qg, scale, bos,
                      static_cast<int>(blockIdx.x) * kBT + (warp >> 1) * kBC,
                      eos - bos, static_cast<int>(blockIdx.y), kH,
                      static_cast<int>(threadIdx.x) & 31, 0.0f};
  if (c.subchunk_start >= c.sequence_t) return;
  const int beta_token = c.subchunk_start + (c.lane & (kBC - 1));
  if (beta_token < c.sequence_t)
    c.beta_lane = fla_sigmoid_bf16(beta_raw[(c.bos + beta_token) * beta_stride + c.head]);

  using S8 = std::make_integer_sequence<int, 8>;
  using S4 = std::make_integer_sequence<int, 4>;
  float k_rows[kRows][4];
  float g_rows[kRows][4];
  chunk_load_rows<0>(c, k_rows, g_rows);
  if ((warp & 1) == 0) {
    // Tokens 0..7 against j 0..7, then tokens 8..11 against j 0..7.
    chunk_token_batch<true, 0, 0, 0, 1, 6, 0, 7>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 1, 0, 2, 5, 0, 6>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 2, 0, 3, 4, 0, 5>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 7, 0, 8, 7, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 3, 0, 4, 3, 0, 0>(c, k_rows, g_rows, S4{});
    chunk_token_batch<true, 0, 8, 0, 8, 8, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 9, 0, 8, 9, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 10, 0, 8, 10, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 11, 0, 8, 11, 0, 0>(c, k_rows, g_rows, S8{});
  } else {
    // Tokens 12..15 against j 0..7, then tokens 8..15 against j 8..15.
    chunk_token_batch<true, 0, 12, 0, 8, 12, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 13, 0, 8, 13, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 14, 0, 8, 14, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<true, 0, 15, 0, 8, 15, 0, 0>(c, k_rows, g_rows, S8{});
    chunk_load_rows<kRows>(c, k_rows, g_rows);
    chunk_token_batch<false, 8, 8, 8, 1, 14, 8, 7>(c, k_rows, g_rows, S8{});
    chunk_token_batch<false, 8, 9, 8, 2, 13, 8, 6>(c, k_rows, g_rows, S8{});
    chunk_token_batch<false, 8, 10, 8, 3, 12, 8, 5>(c, k_rows, g_rows, S8{});
    chunk_token_batch<false, 8, 15, 8, 8, 15, 8, 0>(c, k_rows, g_rows, S8{});
    chunk_token_batch<false, 8, 11, 8, 4, 11, 8, 0>(c, k_rows, g_rows, S4{});
  }
}

__global__ __launch_bounds__(kChunkThreads, 4)
__block_size__((kChunkThreads, 1, 1))
void token_intra_chunk_bf16_d128_bt64_bc16_kernel(
    const __ppu_bfloat16* __restrict__ q,
    const __ppu_bfloat16* __restrict__ k,
    const float* __restrict__ g,
    const __ppu_bfloat16* __restrict__ beta_raw,
    int64_t beta_stride,
    __ppu_bfloat16* __restrict__ aqk,
    float* __restrict__ akk,
    __ppu_bfloat16* __restrict__ qg,
    float scale,
    const int32_t* __restrict__ cu_seqlens,
    int total_tokens) {
  token_intra_chunk_body(q, k, g, beta_raw, beta_stride, aqk, akk, qg, scale,
                         cu_seqlens, total_tokens);
}

// Fused L2 variant: before the token stage, the CTA normalizes the q and k
// rows of its own chunk and head (physical rows, as the standalone L2 stage
// does) into qn/kn, then consumes them after a CTA barrier. The per-row math
// is l2norm2's: a 16-lane halfwarp owns one row, a lane owns eight contiguous
// D values, sequential squares, xor 8/4/2/1 butterfly, one division and one
// BF16 rounding, so qn/kn are bit-identical to the separate launch.
constexpr int kL2RowsPerPass = kChunkThreads / 16;
constexpr int kL2Passes = 2 * kBT / kL2RowsPerPass;
constexpr float kL2Eps = 1.0e-6f;

union alignas(16) Bf16x8 {
  uint4 vector;
  __ppu_bfloat162 pairs[4];
};

// On return packed[pass] holds this thread's rounded qn (pass < 4) or kn
// (pass >= 4) slice of chunk row (pass % 4) * 16 + threadIdx.x / 16.
__device__ __forceinline__ void chunk_l2_rows(
    const __ppu_bfloat16* __restrict__ raw_q,
    const __ppu_bfloat16* __restrict__ raw_k,
    __ppu_bfloat16* qn, __ppu_bfloat16* kn, int total_tokens,
    Bf16x8 (&packed)[kL2Passes]) {
  const int row_in_pass = static_cast<int>(threadIdx.x) >> 4;
  const int first_d = (static_cast<int>(threadIdx.x) & 15) * 8;
  const int head = static_cast<int>(blockIdx.y);
  const int chunk_start = static_cast<int>(blockIdx.x) * kBT;
  // Issue every load before the reductions. Rows past the physical extent
  // still join the full-mask shuffles but neither load nor store.
#pragma unroll
  for (int pass = 0; pass < kL2Passes; ++pass) {
    constexpr int kPassesEach = kL2Passes / 2;
    const int token = chunk_start + (pass % kPassesEach) * kL2RowsPerPass + row_in_pass;
    const __ppu_bfloat16* x = pass < kPassesEach ? raw_q : raw_k;
    packed[pass].vector = make_uint4(0u, 0u, 0u, 0u);
    if (token < total_tokens)
      packed[pass].vector = *reinterpret_cast<const uint4*>(
          x + (static_cast<int64_t>(token) * kH + head) * kK + first_d);
  }
#pragma unroll
  for (int pass = 0; pass < kL2Passes; ++pass) {
    constexpr int kPassesEach = kL2Passes / 2;
    const int token = chunk_start + (pass % kPassesEach) * kL2RowsPerPass + row_in_pass;
    __ppu_bfloat16* y = pass < kPassesEach ? qn : kn;
    float values[8];
    float sum_sq = 0.0f;
#pragma unroll
    for (int pair_index = 0; pair_index < 4; ++pair_index) {
      const float2 pair = __bfloat1622float2(packed[pass].pairs[pair_index]);
      values[pair_index * 2] = pair.x;
      values[pair_index * 2 + 1] = pair.y;
      sum_sq += pair.x * pair.x;
      sum_sq += pair.y * pair.y;
    }
#pragma unroll
    for (int lane_mask = 8; lane_mask > 0; lane_mask >>= 1) {
      sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, lane_mask, 16);
    }
    const float denominator = sqrtf(sum_sq + kL2Eps);
    Bf16x8 packed_output;
#pragma unroll
    for (int pair_index = 0; pair_index < 4; ++pair_index) {
      packed_output.pairs[pair_index] = __float22bfloat162_rn(make_float2(
          values[pair_index * 2] / denominator,
          values[pair_index * 2 + 1] / denominator));
    }
    packed[pass].vector = packed_output.vector;
    if (token < total_tokens)
      *reinterpret_cast<uint4*>(
          y + (static_cast<int64_t>(token) * kH + head) * kK + first_d) =
          packed_output.vector;
  }
}

// Fused inter-prepare phase: the same values the inter stage's
// prepare_inter_operands writes, produced by the CTA that owns the chunk.
// Gated operands for subchunks r = 1..3 go to the w/u/kg scratch tiles
// (overwritten later by recompute) and the FP32 16x16 diagonal blocks of akk
// are solved in place. Arithmetic, rounding and write domains match the
// separate kernel element for element; only the thread mapping differs.
// The tiles are built from the L2 phase's registers: a thread owns the same
// row and eight D values of every subchunk, so qn/kn need no reload. N=1
// guarantees bos == 0, which makes the physical L2 rows the chunk rows.
__device__ __forceinline__ float prepared_exp(float value) {
  return __exp2f(__fmul_rn(value, __uint_as_float(0x3fb8aa3bu)));
}

__device__ __forceinline__ float prepared_component(Bf16x8 const& packed,
                                                    int component) {
  const float2 pair = __bfloat1622float2(packed.pairs[component >> 1]);
  return component & 1 ? pair.y : pair.x;
}

__device__ __forceinline__ void load_gate8(const float* address,
                                           float (&values)[8]) {
  const float4 low = *reinterpret_cast<const float4*>(address);
  const float4 high = *reinterpret_cast<const float4*>(address + 4);
  values[0] = low.x; values[1] = low.y; values[2] = low.z; values[3] = low.w;
  values[4] = high.x; values[5] = high.y; values[6] = high.z; values[7] = high.w;
}

__device__ __forceinline__ void store_prepared(cute::bfloat16_t* address,
                                               cute::bfloat16_t const (&v)[8]) {
  *reinterpret_cast<uint4*>(address) = make_uint4(
      uint32_t(v[0].raw()) | (uint32_t(v[1].raw()) << 16),
      uint32_t(v[2].raw()) | (uint32_t(v[3].raw()) << 16),
      uint32_t(v[4].raw()) | (uint32_t(v[5].raw()) << 16),
      uint32_t(v[6].raw()) | (uint32_t(v[7].raw()) << 16));
}

__device__ __forceinline__ cute::bfloat16_t* prepared_tile(
    cute::bfloat16_t* w, cute::bfloat16_t* u, cute::bfloat16_t* kg,
    int64_t chunk, int head, int tile) {
  cute::bfloat16_t* base = tile < 4 ? w : tile < 8 ? u : kg;
  return base + (chunk * kH + head) * 8192 + (tile % 4) * 2048;
}

__device__ __forceinline__ bool prepared_chunk(
    const int32_t* __restrict__ cu_seqlens, int total_tokens, int64_t& start,
    int64_t& end) {
  const int64_t bos = cu_seqlens[0];
  end = cu_seqlens[1];
  start = bos + int64_t(blockIdx.x) * kBT;
  return bos == 0 && end <= total_tokens && start < end;
}

__device__ __forceinline__ void chunk_prepare_tiles(
    Bf16x8 const (&packed)[kL2Passes], const float* __restrict__ g,
    cute::bfloat16_t* w, cute::bfloat16_t* u, cute::bfloat16_t* kg,
    const int32_t* __restrict__ cu_seqlens, int total_tokens) {
  int64_t start, end;
  if (!prepared_chunk(cu_seqlens, total_tokens, start, end)) return;
  const int64_t chunk = blockIdx.x;
  const int head = static_cast<int>(blockIdx.y);
  const int row = static_cast<int>(threadIdx.x) >> 4;
  const int d = (static_cast<int>(threadIdx.x) & 15) * 8;
  const int linear = row * kK + d;
  constexpr int kKeyPass = kL2Passes / 2;
  // Source-major order loads each gate row once: subchunk s supplies the
  // q/k operands of r = s and the k side of every b tile (r > s, s).
  const auto first_tile = [](int r) { return r == 1 ? 0 : r == 2 ? 3 : 7; };
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int64_t source = start + s * kBC;
    float gate[8] = {};
    if (source + row < end)
      load_gate8(g + ((source + row) * kH + head) * kK + d, gate);
    if (s > 0) {
      float anchor[8] = {};
      cute::bfloat16_t qv[8] = {}, kv[8] = {};
      if (source < end) load_gate8(g + (source * kH + head) * kK + d, anchor);
      if (source + row < end) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const float e = prepared_exp(__fsub_rn(gate[c], anchor[c]));
          qv[c] = cute::bfloat16_t(__fmul_rn(prepared_component(packed[s], c), e));
          kv[c] = cute::bfloat16_t(
              __fmul_rn(prepared_component(packed[kKeyPass + s], c), e));
        }
      }
      store_prepared(prepared_tile(w, u, kg, chunk, head, first_tile(s)) + linear, qv);
      store_prepared(prepared_tile(w, u, kg, chunk, head, first_tile(s) + 1) + linear, kv);
    }
#pragma unroll
    for (int r = s + 1; r < 4; ++r) {
      const int64_t target = start + r * kBC;
      cute::bfloat16_t bv[8] = {};
      if (target < end) {
        float anchor[8];
        load_gate8(g + (target * kH + head) * kK + d, anchor);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const float e = prepared_exp(__fsub_rn(anchor[c], gate[c]));
          bv[c] = cute::bfloat16_t(
              __fmul_rn(prepared_component(packed[kKeyPass + s], c), e));
        }
      }
      store_prepared(
          prepared_tile(w, u, kg, chunk, head, first_tile(r) + 2 + s) + linear, bv);
    }
  }
}

__device__ __forceinline__ void chunk_solve_diagonal(
    float* akk, const int32_t* __restrict__ cu_seqlens, int total_tokens) {
  int64_t start, end;
  if (threadIdx.x >= 64 || !prepared_chunk(cu_seqlens, total_tokens, start, end))
    return;
  const int head = static_cast<int>(blockIdx.y);
  const int column = threadIdx.x & 15;
  const int64_t base = start + (threadIdx.x >> 4) * kBC;
  float solved[16] = {};
#pragma unroll
  for (int row = 0; row < kBC; ++row) {
    float* entry = akk + ((base + row) * kH + head) * kBC + column;
    const float coefficient = base + row < end && column < row ? -*entry : 0.0f;
    float value = column == row ? 1.0f : 0.0f;
#pragma unroll
    for (int previous = 0; previous < row; ++previous) {
      const float broadcast = __shfl_sync(0xffffffffu, coefficient, previous, 16);
      value = __fadd_rn(value, __fmul_rn(broadcast, solved[previous]));
    }
    solved[row] = value;
    if (base + row < end) *entry = value;
  }
}

// Clears the chunk's live Aqk rows from the diagonal tile rightwards with
// 16-byte runs, replacing a workspace memset: row block R owns columns
// [16R, 64). The token body overwrites each diagonal tile's lower triangle
// after the CTA barrier that follows; the rest stays +0 for CP consumers
// that read full-chunk Aqk unmasked.
__device__ __forceinline__ void chunk_zero_aqk_upper(
    __ppu_bfloat16* aqk, const int32_t* __restrict__ cu_seqlens,
    int total_tokens) {
  int64_t start, end;
  if (!prepared_chunk(cu_seqlens, total_tokens, start, end)) return;
  const int head = static_cast<int>(blockIdx.y);
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int i = static_cast<int>(threadIdx.x) + kChunkThreads * pass;
    if (i >= 320) break;
    int row, col;
    if (i < 128) { row = i / 8; col = 8 * (i % 8); }
    else if (i < 224) { row = 16 + (i - 128) / 6; col = 16 + 8 * ((i - 128) % 6); }
    else if (i < 288) { row = 32 + (i - 224) / 4; col = 32 + 8 * ((i - 224) % 4); }
    else { row = 48 + (i - 288) / 2; col = 48 + 8 * ((i - 288) % 2); }
    if (start + row < end)
      *reinterpret_cast<uint4*>(aqk + ((start + row) * kH + head) * kBT + col) =
          make_uint4(0, 0, 0, 0);
  }
}

__global__ __launch_bounds__(kChunkThreads, 4)
__block_size__((kChunkThreads, 1, 1))
void token_intra_chunk_l2_bf16_d128_bt64_bc16_kernel(
    const __ppu_bfloat16* __restrict__ raw_q,
    const __ppu_bfloat16* __restrict__ raw_k,
    __ppu_bfloat16* qn,
    __ppu_bfloat16* kn,
    const float* __restrict__ g,
    const __ppu_bfloat16* __restrict__ beta_raw,
    int64_t beta_stride,
    __ppu_bfloat16* __restrict__ aqk,
    float* akk,
    __ppu_bfloat16* __restrict__ qg,
    cute::bfloat16_t* w,
    cute::bfloat16_t* u,
    cute::bfloat16_t* kg,
    float scale,
    const int32_t* __restrict__ cu_seqlens,
    int total_tokens) {
  Bf16x8 packed[kL2Passes];
  chunk_l2_rows(raw_q, raw_k, qn, kn, total_tokens, packed);
  if (w) chunk_prepare_tiles(packed, g, w, u, kg, cu_seqlens, total_tokens);
  if (w) chunk_zero_aqk_upper(aqk, cu_seqlens, total_tokens);
  // The barrier publishes this CTA's qn/kn rows and Aqk zeros; no other CTA
  // reads them.
  __syncthreads();
  token_intra_chunk_body(qn, kn, g, beta_raw, beta_stride, aqk, akk, qg, scale,
                         cu_seqlens, total_tokens);
  if (w) {
    // Publishes the chunk's akk rows to the diagonal solve.
    __syncthreads();
    chunk_solve_diagonal(akk, cu_seqlens, total_tokens);
  }
}

}  // namespace fla_glm53_token_intra_cute

// N=1 launcher for the fused L2 + token stage; qn/kn are still published for
// the inter, recompute and output stages. Non-null w/u/kg also fuse the inter
// stage's operand preparation and diagonal solve.
extern "C" hggcError_t FLA_JOIN_HEAD(launch_fla_token_l2, FLA_HEADS)(
    const void* raw_q, const void* raw_k, void* qn, void* kn,
    const float* g, const uint16_t* beta_raw, int64_t beta_stride,
    void* aqk, float* akk, void* qg, void* w, void* u, void* kg, float scale,
    const int32_t* cu_seqlens, int total_tokens, int heads,
    hggcStream_t stream) {
  if (total_tokens == 0) return hggcSuccess;
  const dim3 grid(static_cast<unsigned>(
                      (total_tokens + fla_glm53_token_intra_cute::kBT - 1) /
                      fla_glm53_token_intra_cute::kBT),
                  heads);
  fla_glm53_token_intra_cute::token_intra_chunk_l2_bf16_d128_bt64_bc16_kernel
      <<<grid, fla_glm53_token_intra_cute::kChunkThreads, 0, stream>>>(
          static_cast<const __ppu_bfloat16*>(raw_q),
          static_cast<const __ppu_bfloat16*>(raw_k),
          static_cast<__ppu_bfloat16*>(qn), static_cast<__ppu_bfloat16*>(kn), g,
          reinterpret_cast<const __ppu_bfloat16*>(beta_raw), beta_stride,
          static_cast<__ppu_bfloat16*>(aqk), akk,
          static_cast<__ppu_bfloat16*>(qg), static_cast<cute::bfloat16_t*>(w),
          static_cast<cute::bfloat16_t*>(u), static_cast<cute::bfloat16_t*>(kg),
          scale, cu_seqlens, total_tokens);
  return hggcGetLastError();
}

hggcError_t launch_fla_glm53_token_intra_bf16_d128_bt64_bc16_h8(
    const void* q,
    const void* k,
    const float* g,
    const uint16_t* beta_raw,
    int64_t beta_stride,
    void* aqk,
    float* akk,
    void* qg,
    float scale,
    const int32_t* cu_seqlens,
    int n,
    int total_tokens, int heads,
    hggcStream_t stream) {
  if (total_tokens == 0) return hggcSuccess;
  if (n == 1) {
    // Chunk-tiled N=1 path; the effective length is read on device, so the
    // grid covers the physical token extent.
    const dim3 grid(static_cast<unsigned>(
                        (total_tokens + fla_glm53_token_intra_cute::kBT - 1) /
                        fla_glm53_token_intra_cute::kBT),
                    heads);
    fla_glm53_token_intra_cute::token_intra_chunk_bf16_d128_bt64_bc16_kernel
        <<<grid, fla_glm53_token_intra_cute::kChunkThreads, 0, stream>>>(
            static_cast<const __ppu_bfloat16*>(q),
            static_cast<const __ppu_bfloat16*>(k), g,
            reinterpret_cast<const __ppu_bfloat16*>(beta_raw), beta_stride,
            static_cast<__ppu_bfloat16*>(aqk), akk,
            static_cast<__ppu_bfloat16*>(qg), scale, cu_seqlens, total_tokens);
    return hggcGetLastError();
  }
  const dim3 grid(static_cast<unsigned>(total_tokens),
                  heads);
  const dim3 block(fla_glm53_token_intra_cute::kThreads);
  fla_glm53_token_intra_cute::token_intra_bf16_d128_bt64_bc16_h8_kernel
      <<<grid, block, 0, stream>>>(
          static_cast<const __ppu_bfloat16*>(q),
          static_cast<const __ppu_bfloat16*>(k), g,
          reinterpret_cast<const __ppu_bfloat16*>(beta_raw), beta_stride,
          static_cast<__ppu_bfloat16*>(aqk), akk,
          static_cast<__ppu_bfloat16*>(qg), scale, cu_seqlens, n,
          total_tokens);
  return hggcGetLastError();
}
