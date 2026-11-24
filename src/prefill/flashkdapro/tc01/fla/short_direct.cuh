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

// A warp or half warp owns one value row and keeps its state in registers.
#if !FLA_RECURRENCE_BODY_ONLY && FLA_HEADS == 0
#include <type_traits>

namespace fla_short_prefill {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
union alignas(8) Packed4 {
  uint2 vector;
  uint16_t element[4];
};
union alignas(16) Packed8 {
  uint4 vector;
  uint16_t element[8];
};

template <int Lanes>
CUTE_DEVICE float dot(const float (&a)[128 / Lanes], const float (&b)[128 / Lanes]) {
  float result;
  if constexpr (Lanes == 32) {
    result = __fmul_rn(a[3], b[3]);
    CUTE_UNROLL
    for (int i = 2; i >= 0; --i) result = __fmaf_rn(a[i], b[i], result);
  } else {
    float lo = __fmaf_rn(a[0], b[0], __fmul_rn(a[1], b[1]));
    float hi = __fmaf_rn(a[4], b[4], __fmul_rn(a[5], b[5]));
    lo = __fmaf_rn(a[2], b[2], lo);
    hi = __fmaf_rn(a[6], b[6], hi);
    lo = __fmaf_rn(a[3], b[3], lo);
    hi = __fmaf_rn(a[7], b[7], hi);
    result = __fadd_rn(lo, hi);
  }
  CUTE_UNROLL
  for (int stride = Lanes / 2; stride > 0; stride >>= 1)
    result = __fadd_rn(result, __shfl_xor_sync(0xffffffffu, result, stride, Lanes));
  return result;
}

template <int Elements>
CUTE_DEVICE void unpack(const uint16_t* address, float (&output)[Elements]) {
  using Packed = std::conditional_t<Elements == 4, Packed4, Packed8>;
  using Vector = std::conditional_t<Elements == 4, uint2, uint4>;
  Packed p;
  p.vector = *reinterpret_cast<const Vector*>(address);
  CUTE_UNROLL
  for (int i = 0; i < Elements; ++i) output[i] = fla_recurrence::load_bf16(p.element[i]);
}
#endif

template <bool Prepared, int Lanes>
__global__ __launch_bounds__(128,1) __block_size__((128,1,1))
void direct(const uint16_t* q,const uint16_t* k,const uint16_t* v,
            const void* gate,const void* beta,int beta_stride,
            const float* a_log,const float* bias,uint16_t* out,int tokens,
            const void* initial,void* final,int fp32,const int32_t* indices,
            float lower,float scale) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  const int heads=gridDim.y,head=blockIdx.y;
  constexpr int Elements = 128 / Lanes;
  const int lane = threadIdx.x & (Lanes - 1);
  const int value = Elements * blockIdx.x + threadIdx.x / Lanes;
  const int column = Elements * lane;
  const int slot=indices ? indices[0] : 0;
  const int64_t offset=((int64_t(slot)*heads+head)*128+value)*128+column;
  float a = 0.0f;
  float4 channel_bias{};
  if constexpr (!Prepared) {
    a = expf(a_log[head]);
    channel_bias = *reinterpret_cast<const float4*>(bias + head * 128 + column);
  }
  const float channel[4] = {channel_bias.x, channel_bias.y, channel_bias.z, channel_bias.w};
  float state[Elements]{};
  if(initial) {
    CUTE_UNROLL
    for (int i = 0; i < Elements; ++i)
      state[i]=fp32
          ? fla_recurrence::load_bf16(fla_recurrence::round_bf16(static_cast<const float*>(initial)[offset+i]))
          : fla_recurrence::load_bf16(static_cast<const uint16_t*>(initial)[offset+i]);
  }
  CUTE_NO_UNROLL
  for(int token=0;token<tokens;++token) {
    const int base=(token*heads+head)*128;
    float keys[Elements], queries[Elements];
    unpack(k+base+column,keys);
    unpack(q+base+column,queries);
    float v_t=fla_recurrence::load_bf16(v[base+value]);
    float b;
    if constexpr (Prepared) {
      b = static_cast<const float*>(beta)[token * heads + head];
      const auto* decay = reinterpret_cast<const float4*>(static_cast<const float*>(gate) + base + column);
      CUTE_UNROLL
      for (int index = 0; index < Elements; index += 4) {
        const auto d = decay[index / 4];
        state[index] = __fmul_rn(state[index], d.x);
        state[index + 1] = __fmul_rn(state[index + 1], d.y);
        state[index + 2] = __fmul_rn(state[index + 2], d.z);
        state[index + 3] = __fmul_rn(state[index + 3], d.w);
      }
    } else {
      static_assert(Lanes == 32);
      float raw_gate[4];
      unpack(static_cast<const uint16_t*>(gate) + base + column, raw_gate);
      const float b_t = fla_recurrence::load_bf16(static_cast<const uint16_t*>(beta)[int64_t(token) * beta_stride + head]);
      b = 1.0f / (1.0f + expf(-b_t));
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        const float decay = lower / (1.0f + expf(-a * (raw_gate[i] + channel[i])));
        state[i] = __fmul_rn(state[i], expf(decay));
      }
    }
    const float residual=__fmul_rn(__fsub_rn(v_t,dot<Lanes>(state,keys)),b);
    CUTE_UNROLL
    for (int i = 0; i < Elements; ++i) state[i]=__fmaf_rn(keys[i],residual,state[i]);
    const float projected=__fmul_rn(dot<Lanes>(state,queries),scale);
    if(lane==0) out[base+value]=fla_recurrence::round_bf16(projected);
  }
  if(final) {
    CUTE_UNROLL
    for (int i = 0; i < Elements; ++i) {
      auto bits=fla_recurrence::round_bf16(state[i]);
      if(fp32) static_cast<float*>(final)[offset+i]=fla_recurrence::load_bf16(bits);
      else static_cast<uint16_t*>(final)[offset+i]=bits;
    }
  }
#else
  __trap();
#endif
}
}  // namespace fla_short_prefill

extern "C" hggcError_t launch_fla_short_direct(
    const void* q,const void* k,const void* v,const void* gate,const void* beta,
    int beta_stride,const float* a_log,const float* bias,void* out,int tokens,int heads,
    const void* initial,void* final,int fp32,const int32_t* indices,
    float lower,float scale,bool prepared,bool half_warp,hggcStream_t stream) {
  if (tokens < 1 || tokens > (prepared ? 218 : 64) || heads < 1 || (half_warp && !prepared))
    return hggcErrorInvalidValue;
  using U=uint16_t;
  auto kernel = half_warp ? fla_short_prefill::direct<true, 16>
      : prepared ? fla_short_prefill::direct<true, 32> : fla_short_prefill::direct<false, 32>;
  kernel<<<dim3(half_warp ? 16 : 32, heads),dim3(128),0,stream>>>(
      static_cast<const U*>(q),static_cast<const U*>(k),static_cast<const U*>(v),
      gate,beta,beta_stride,a_log,bias,
      static_cast<U*>(out),tokens,initial,final,fp32,indices,lower,scale);
  return hggcGetLastError();
}
#endif
