// SPDX-License-Identifier: Apache-2.0
// PPU1.5 fragment coordinates are checked by probe_fold_matrix.cu.
template <bool OperandA, class Fragment>
__device__ __forceinline__ void load_fold_operand(Fragment& fragment, const BF* base,
                                                     int stride, int lane) {
  static_assert(sizeof(fragment.x) == sizeof(uint4));
  const int row = (lane & 7) + (OperandA ? (lane & 16)/2 : lane & 8);
  const int column = OperandA ? lane & 8 : (lane & 16)/2;
  const auto address = static_cast<uint32_t>(__cvta_generic_to_shared(base + row*stride + column));
  uint4 packed;
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w) : "r"(address) : "memory");
  __builtin_memcpy(fragment.x, &packed, sizeof(packed));
}
