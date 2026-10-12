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

/*
 * FlashKDA PPU1.5 — f16->f16 accumulate MMA atom (project-specific).
 *
 * The reusable flashkda_cute_traits.cuh (borrowed PPU base component) only
 * provides the f16->f32 FIXED atom (PPU_16x16x16_F32F16F16F32_TN_FIXED).
 * FlashKDA's blueprint (torch_ref / fwd_kernel1.cuh neumann_inv_fused_1warp,
 * L = tril(k_decayed @ k_inv^T)) deliberately uses fp16 ACCUMULATE MMA
 * (SM80_16x8x16_F16F16F16F16_TN / cuBLAS COMPUTE_16F). To bit-align that
 * numeric behavior on PPU1.5 we add the f16->f16 accumulate atom here.
 *
 * Hardware instruction (TIX 9.3.10):
 *   ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16
 *     D{4xf16x2}, A{4xf16x2}, B{4xf16x2}, C{4xf16x2}
 * (validated present as cute::PPU0015_16x16x16_F16F16F16F16_TN).
 *
 * Mirrors the FIXED-atom pattern of the borrowed traits: raw "r"-constraint
 * asm + a cute::gemm-bypassing mma_unpack override + MMA_Traits inherited
 * from the ppu0015 f16-accumulate traits (ValType* = half_t, C layout
 * = PPU0015_16x16_Row).
 */
#pragma once

// flashkda_arch.cuh dispatches the actlize traits by __HGGC_ARCH__, so include
// it before any cute/cutlass header and let it pick the trait family.
#include "flashkda_arch.cuh"

namespace flashkda {

using namespace cute;

// Raw f16->f16 accumulate 16x16x16 MMA (D = A @ B^T + C, TN), "r" constraints
// for the HGGC backend. All of A/B/C/D are 4x f16x2 (uint32) registers, so
// the same asm string works on tc01 and tc02 -- only the ARCH guard differs.
__forceinline__ __device__ void ppu_mma_f16acc_impl(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
    uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
    uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
    uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3) {
#if defined(FLASHKDA_ARCH_150) || defined(FLASHKDA_ARCH_100)
  asm volatile(
      "ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9, %10, %11}, "
      "{%12, %13, %14, %15};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1), "r"(b2), "r"(b3),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#else
  (void)d0; (void)d1; (void)d2; (void)d3;
  (void)a0; (void)a1; (void)a2; (void)a3;
  (void)b0; (void)b1; (void)b2; (void)b3;
  (void)c0; (void)c1; (void)c2; (void)c3;
#endif
}

}  // namespace flashkda

namespace cute {

// FIXED f16-accumulate atom (bypasses the assert(0) generic PPU stub).
struct PPU_16x16x16_F16F16F16F16_TN_FIXED {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[4];

  __forceinline__ __device__ static void
  fma(uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3) {
    flashkda::ppu_mma_f16acc_impl(d0, d1, d2, d3, a0, a1, a2, a3,
                                  b0, b1, b2, b3, c0, c1, c2, c3);
  }
};

// Inherit thread/value layout from the arch-dispatched f16-accumulate atom
// (ValType* = half_t, CLayout follows the target chip's fragment tiling).
template <>
struct MMA_Traits<PPU_16x16x16_F16F16F16F16_TN_FIXED>
    : MMA_Traits<flashkda::PpuF16AccMMA> {};

// Bypass cute::gemm dispatch chain (detail::explode), call fma directly.
template <class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<PPU_16x16x16_F16F16F16F16_TN_FIXED> const&,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

  Tensor rA = recast<uint32_t>(A);
  Tensor rB = recast<uint32_t>(B);
  Tensor rD = recast<uint32_t>(D);
  Tensor rC = recast<uint32_t>(C);

  CUTE_STATIC_ASSERT_V(size(rA) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<4>{});

  flashkda::ppu_mma_f16acc_impl(
      rD(0), rD(1), rD(2), rD(3),
      rA(0), rA(1), rA(2), rA(3),
      rB(0), rB(1), rB(2), rB(3),
      rC(0), rC(1), rC(2), rC(3));
}

}  // namespace cute

namespace flashkda {

using namespace cute;

// Single-warp (32-thread) 16x16x16 f16-accumulate TiledMMA (one atom).
using TiledMma1WarpF16acc = decltype(cute::make_tiled_mma(
    cute::MMA_Atom<cute::PPU_16x16x16_F16F16F16F16_TN_FIXED>{},
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));

// Allocate a 16x16 f16 C accumulator (4x uint32 = 8 halfs) for single-warp MMA.
CUTE_DEVICE auto make_acc16_f16acc_1warp(int lane) {
  TiledMma1WarpF16acc mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::_16{}, cute::_1{})));
  return thr_mma.partition_fragment_C(dummy);
}

// C[16,16] = A[16,16] @ B[16,16]^T (TN), fp16 inputs, fp16 ACCUMULATE. Clears tC.
// sA,sB: (16,16) row-major fp16 smem views. lane in [0,32).
template <typename SmemA, typename SmemB, typename TC>
CUTE_DEVICE void gemm16_ss_1warp_f16acc(
    SmemA const& sA, SmemB const& sB, int lane, TC& tC) {
  TiledMma1WarpF16acc mma;
  auto thr_mma = mma.get_thread_slice(lane);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::DefaultCopy, cute::half_t>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(
      cute::Copy_Atom<cute::DefaultCopy, cute::half_t>{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(lane);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  clear(tC);
  Tensor rC = recast<uint32_t>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(rA_full); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16acc_impl(
            rC(0, m, n), rC(1, m, n), rC(2, m, n), rC(3, m, n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC(0, m, n), rC(1, m, n), rC(2, m, n), rC(3, m, n));
      }
    }
  }
}

// Store a 16x16 f16 accumulator to fp16 smem (row-major, leading dim ldc).
template <typename TC>
CUTE_DEVICE void store_acc16_f16acc_1warp(
    TC const& tC, __half* dst, int ldc, int lane) {
  TiledMma1WarpF16acc mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(dst)),
      cute::make_layout(cute::make_shape(16, 16), cute::make_stride(ldc, 1)));
  Tensor tCsC = thr_mma.partition_C(sOut);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v)
        tCsC(v, m, n) = tC(v, m, n);
}

}  // namespace flashkda
