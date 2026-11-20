// SPDX-License-Identifier: Apache-2.0
// PPU1.5 BF16 operand register order verified by probe_operands.cu.
template <class Frag>
__device__ __forceinline__ void load_a16(Frag& out, const BF* base, int stride, int lane) {
  static_assert(sizeof(out.x) == sizeof(uint4));
  const int row = (lane & 7) + (lane & 8), col = (lane & 16) / 2;
  const uint4 packed = cached_tsm::load4<false>(base + row*stride + col);
  __builtin_memcpy(out.x, &packed, sizeof(packed));
}

template <class Frag>
__device__ __forceinline__ void load_b16(Frag& out, const BF* base, int stride, int lane) {
  static_assert(sizeof(out.x) == sizeof(uint4));
  // B swaps the two middle quadrants in ldmatrix's address groups.
  const int row = (lane & 7) + (lane & 16)/2, col = lane & 8;
  const uint4 packed = cached_tsm::load4<false>(base + row*stride + col);
  __builtin_memcpy(out.x, &packed, sizeof(packed));
}

template <bool Transpose, class Frag>
__device__ __forceinline__ void load_a32(Frag& out, const BF* base, int stride, int lane) {
  static_assert(sizeof(out.x) == 2*sizeof(uint4));
  const int row = (lane & 7) + (Transpose ? (lane & 16)/2 : lane & 8);
  const int col = Transpose ? lane & 8 : (lane & 16)/2;
  const BF* address = base + row*stride + col;
  const uint4 lower = cached_tsm::load4<Transpose>(address);
  const uint4 upper = cached_tsm::load4<Transpose>(address + (Transpose ? 16 : 16*stride));
  __builtin_memcpy(out.x, &lower, sizeof(lower));
  __builtin_memcpy(out.x + 8, &upper, sizeof(upper));
}

template <class Frag>
__device__ __forceinline__ void load_b8(Frag& out, const BF* base, int stride, int lane) {
  static_assert(sizeof(out.x) == sizeof(uint2));
  const uint2 packed = cached_tsm::load2(base + (lane & 7)*stride + (lane & 8));
  __builtin_memcpy(out.x, &packed, sizeof(packed));
}
