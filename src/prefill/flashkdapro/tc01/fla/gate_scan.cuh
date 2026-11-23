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
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include "cute/tensor.hpp"

namespace fla_gate {
union alignas(8) Bf16x4 {
  uint2 vector;
  __ppu_bfloat162 pairs[2];
};

template <bool DefaultBound, class Heads>
__device__ __forceinline__ void scan(
    const __ppu_bfloat16* __restrict__ input,
    const float* __restrict__ a_log, const float* __restrict__ bias,
    float* __restrict__ output, int tokens, const int32_t* cu, const int32_t* indices,
    int n, Heads heads, float lower_bound, int cta_chunk, int cta_column,
    int head, float* carry_storage) {
  using namespace cute;
  const int sequence = indices[2 * cta_chunk];
  const int chunk = indices[2 * cta_chunk + 1];
  if (sequence < 0 || sequence >= n || chunk < 0) return;
  const int bos = cu[sequence], eos = cu[sequence + 1];
  const int64_t first = int64_t(bos) + int64_t(chunk) * 64;
  if (bos < 0 || eos > tokens || first >= eos) return;
  auto layout = make_layout(make_shape(int64_t(tokens), heads, Int<128>{}),
                            make_stride(heads * cute::Int<128>{}, Int<128>{}, Int<1>{}));
  auto gX = make_tensor(make_gmem_ptr(input), layout);
  auto gO = make_tensor(make_gmem_ptr(output), layout);
  auto carry = make_tensor(make_smem_ptr(carry_storage),
                           make_layout(make_shape(Int<16>{}, Int<32>{}),
                                       make_stride(Int<32>{}, Int<1>{})));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int time_lane = lane >> 3;
  const int feature = (lane & 7) * 4;
  const int column = cta_column * 32 + feature;
  const float a = expf(a_log[head]);

  float channel_bias[4];
  const float* bias_address = bias + head * 128 + column;
  const float4 packed_bias = *reinterpret_cast<const float4*>(bias_address);
  channel_bias[0] = packed_bias.x;
  channel_bias[1] = packed_bias.y;
  channel_bias[2] = packed_bias.z;
  channel_bias[3] = packed_bias.w;

  float values[2][4];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int64_t row = first + half * 32 + warp * 4 + time_lane;
    float raw[4] = {};
    if (row < eos) {
      const auto* address = &gX(row, head, column);
      Bf16x4 packed;
      packed.vector = *reinterpret_cast<const uint2*>(address);
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const float2 converted = __bfloat1622float2(packed.pairs[pair]);
        raw[pair * 2] = converted.x;
        raw[pair * 2 + 1] = converted.y;
      }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float z = a * (raw[c] + channel_bias[c]);
      float value = row < eos ? (DefaultBound ? -5.0f : lower_bound) / (1.0f + expf(-z)) : 0.0f;
#pragma unroll
      for (int step = 1; step <= 2; step *= 2) {
        const float prior = __shfl_up_sync(0xffffffffu, value, step * 8);
        if (time_lane >= step) value += prior;
      }
      values[half][c] = value;
      if (time_lane == 3) carry(half * 8 + warp, feature + c) = value;
    }
  }

  __syncthreads();
  if (warp == 0) {
    float prefix = 0.0f;
#pragma unroll
    for (int segment = 0; segment < 16; ++segment) {
      const float total = carry(segment, lane);
      carry(segment, lane) = prefix;
      prefix += total;
    }
  }
  __syncthreads();

#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int64_t row = first + half * 32 + warp * 4 + time_lane;
    if (row < eos) {
      float result[4];
#pragma unroll
      for (int c = 0; c < 4; ++c)
        result[c] = values[half][c] + carry(half * 8 + warp, feature + c);
      *reinterpret_cast<float4*>(&gO(row, head, column)) =
          make_float4(result[0], result[1], result[2], result[3]);
    }
  }
}

}  // namespace fla_gate
