// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace cached_tsm {

template <bool Transpose>
__device__ __forceinline__ uint4 load4(const void* row_address) {
  uint4 d;
  const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(row_address));
  if constexpr (Transpose) {
    asm volatile("ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "r"(addr) : "memory");
  } else {
    asm volatile("ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "r"(addr) : "memory");
  }
  return d;
}

__device__ __forceinline__ uint2 load2(const void* row_address) {
  uint2 d;
  const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(row_address));
  asm volatile("ppu.tc02.ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(d.x), "=r"(d.y) : "r"(addr) : "memory");
  return d;
}

}  // namespace cached_tsm
