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

}  // namespace fla_glm53_token_intra_cute

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
