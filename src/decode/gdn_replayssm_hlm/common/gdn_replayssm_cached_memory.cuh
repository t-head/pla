// SPDX-License-Identifier: Apache-2.0
#pragma once

template <class View>
__device__ __forceinline__ void cached_copy_bf8(const View& t, int64_t offset, int valid, BF* dst) {
  if (valid == 0) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) dst[e] = BF(0.f);
    return;
  }
  const BF* src = static_cast<const BF*>(t.data) + offset;
  if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    *reinterpret_cast<uint4*>(dst) = raw;
  } else {
    #pragma unroll
    for (int e = 0; e < 8; ++e) dst[e] = src[e];
  }
}

__device__ __forceinline__ uint32_t cached_bf16_pair(float first, float second) {
  const uint32_t low = __bfloat16_as_ushort(BF(first));
  const uint32_t high = __bfloat16_as_ushort(BF(second));
  return low | (high << 16);
}

template <class S, int N, class View>
__device__ __forceinline__ void cached_fold_store_vector(const View& t, int64_t offset,
                                                          const float (&values)[N]) {
  static_assert((std::is_same_v<S, float> && N == 4) || (std::is_same_v<S, BF> && (N == 4 || N == 8)),
                "Fold vector stores require FP32x4, BF16x4, or BF16x8");
  S* dst = static_cast<S*>(t.data) + offset;
  if ((reinterpret_cast<uintptr_t>(dst) & (N * sizeof(S) - 1)) == 0) {
    if constexpr (std::is_same_v<S,float>) {
      *reinterpret_cast<float4*>(dst) = float4{values[0],values[1],values[2],values[3]};
    } else if constexpr (N == 4) {
      *reinterpret_cast<uint2*>(dst) = uint2{cached_bf16_pair(values[0], values[1]),
                                             cached_bf16_pair(values[2], values[3])};
    } else {
      *reinterpret_cast<uint4*>(dst) = uint4{cached_bf16_pair(values[0], values[1]),
                                             cached_bf16_pair(values[2], values[3]),
                                             cached_bf16_pair(values[4], values[5]),
                                             cached_bf16_pair(values[6], values[7])};
    }
  } else {
    #pragma unroll
    for (int e = 0; e < N; ++e) write<S>(t,offset+e,values[e]);
  }
}
