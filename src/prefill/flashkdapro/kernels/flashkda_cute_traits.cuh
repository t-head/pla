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
 * FlashQLA CuTe MMA Infrastructure for PPU1.5
 *
 * Replaces AWMMA API with CuTe MMA_Atom + TiledMMA.
 * Uses PPU_16x16x16_F32F16F16F32_TN_FIXED workaround for PPU hggccc backend.
 *
 * Key patterns (from gdn_prefill ppu_mma_atom_fixed.hpp & flat_common.hpp):
 *   - All asm operands use "r" constraint (uint32 reg class)
 *   - cute::gemm() dispatch is bypassed; fma_fixed_impl called directly
 *   - B operand loaded lazily per K-slice to reduce register pressure
 *
 * TN MMA semantics:
 *   partition_fragment_A(sA[M,K]) + partition_fragment_B(sB[N,K])
 *   => computes C[M,N] = A[M,K] × B[N,K]^T
 *   So passing row-major A[M,K] and row-major B[N,K] gives A × B^T.
 *   For C = A × B (non-transposed), pass select_tensor<1,0>(B) as the B operand.
 */

#pragma once

// Keep the target generation and cutlass trait family consistent before any
// CuTe header selects its inline-asm implementation.
#include "flashkda_arch.cuh"

#include "cute/arch/copy_ppu.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_ppu.hpp"
#include "cute/arch/mma_ppu.hpp"
#include "cute/atom/mma_traits_ppu.hpp"

namespace flashqla {

using namespace cute;

// ============================================================================
// fp16 MMA atom raw asm — PPU1.5 FIXED workaround
// ============================================================================

__forceinline__ __device__ void ppu_mma_f16_fixed_impl(
    float& d0, float& d1, float& d2, float& d3,
    float& d4, float& d5, float& d6, float& d7,
    uint32_t const& a0, uint32_t const& a1,
    uint32_t const& a2, uint32_t const& a3,
    uint32_t const& b0, uint32_t const& b1,
    uint32_t const& b2, uint32_t const& b3,
    float const& c0, float const& c1, float const& c2, float const& c3,
    float const& c4, float const& c5, float const& c6, float const& c7) {
#if defined(FLASHKDA_ARCH_150)
  asm volatile(
    "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7}, "
    "{%8, %9, %10, %11}, {%12, %13, %14, %15}, "
    "{%16, %17, %18, %19, %20, %21, %22, %23};\n"
    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3),
      "=r"(d4), "=r"(d5), "=r"(d6), "=r"(d7)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1), "r"(b2), "r"(b3),
      "r"(c0), "r"(c1), "r"(c2), "r"(c3),
      "r"(c4), "r"(c5), "r"(c6), "r"(c7));
#elif defined(FLASHKDA_ARCH_100)
  asm volatile(
    "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7}, "
    "{%8, %9, %10, %11}, {%12, %13, %14, %15}, "
    "{%16, %17, %18, %19, %20, %21, %22, %23};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
      "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1), "r"(b2), "r"(b3),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3),
      "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#endif
}

}  // namespace flashqla

namespace cute {

// ============================================================================
// PPU fp16 FIXED MMA atom struct
// ============================================================================

struct PPU_16x16x16_F32F16F16F32_TN_FIXED {
  using DRegisters = float[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[8];

  __forceinline__ __device__ static void
  fma(float& d0, float& d1, float& d2, float& d3,
      float& d4, float& d5, float& d6, float& d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& b2, uint32_t const& b3,
      float const& c0, float const& c1, float const& c2, float const& c3,
      float const& c4, float const& c5, float const& c6, float const& c7) {
#if defined(FLASHKDA_ARCH_150) || defined(FLASHKDA_ARCH_100)
    flashqla::ppu_mma_f16_fixed_impl(
        d0, d1, d2, d3, d4, d5, d6, d7,
        a0, a1, a2, a3,
        b0, b1, b2, b3,
        c0, c1, c2, c3, c4, c5, c6, c7);
#else
    CUTE_INVALID_CONTROL_PATH(
        "PPU_16x16x16_F32F16F16F32_TN_FIXED requires HGGC_ARCH 100 or 150");
#endif
  }
};

// Inherit the physical fragment layout selected for the target generation.
template <>
struct MMA_Traits<PPU_16x16x16_F32F16F16F32_TN_FIXED>
    : MMA_Traits<flashkda::PpuF16MMA> {};

// Custom mma_unpack: bypass cute::gemm dispatch chain (detail::explode)
template <class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<PPU_16x16x16_F32F16F16F32_TN_FIXED> const&,
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
  Tensor rD = recast<float>(D);
  Tensor rC = recast<float>(C);

  CUTE_STATIC_ASSERT_V(size(rA) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<8>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<8>{});

  flashqla::ppu_mma_f16_fixed_impl(
      rD(0), rD(1), rD(2), rD(3), rD(4), rD(5), rD(6), rD(7),
      rA(0), rA(1), rA(2), rA(3),
      rB(0), rB(1), rB(2), rB(3),
      rC(0), rC(1), rC(2), rC(3), rC(4), rC(5), rC(6), rC(7));
}

}  // namespace cute

namespace flashqla {

// ============================================================================
// TiledMMA configurations
// ============================================================================

// fp16 MMA atom (FIXED variant for PPU)
using MmaAtomF16 = cute::MMA_Atom<cute::PPU_16x16x16_F32F16F16F32_TN_FIXED>;

// 4-warp TiledMMA: covers M=64, N=16, K=16 per atom step
// For a (M,N,K) problem: M_tiles = M/64, N_tiles = N/16, K_tiles = K/16
using TiledMma4Warp = decltype(cute::make_tiled_mma(
    MmaAtomF16{},
    cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>{}));

// 8-warp TiledMMA: covers M=64, N=32, K=16 per atom step (N tiled x2).
// Used by the non-WS 16-warp/CTA variant: each DV-group uses 256 threads
// (8 warps) instead of 128 (4 warps), doubling active warps per CU with the
// SAME smem footprint. Warp layout: warp_m = warp%4 selects the 16 A-rows,
// warp_n = warp/4 selects the N-half. Per-thread C-fragment halves in N, so
// make_acc<M,N> here is 8*(M/64)*(N/32) fp32 regs (half of the 4-warp count),
// which also relieves register pressure / spill.
// 4-warp M=32 TiledMMA: covers M=32, N=32, K=16 per atom step (warps = 2m x 2n).
// Used by the CS=32 DV-sharded variant: each DV-group is 128 threads (4 warps),
// block = 2 groups = 256 threads. Fits every kernel shape exactly with no
// over-tiling: GEMM4 [CS=32,CS=32] -> 1x1 atom (8 regs); GEMM1/2/3/5 [CS=32,DVh=64]
// -> 1x2 (16 regs); register-h [DK=128,DVh=64] -> 4x2 (64 regs). At 256 thr/block
// the 2-block register cap is 128K/(256*2)=256 regs/thread, so h_reg never spills.
using TiledMmaM32 = decltype(cute::make_tiled_mma(
    MmaAtomF16{},
    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{}));

using TiledMma8Warp = decltype(cute::make_tiled_mma(
    MmaAtomF16{},
    cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>{}));

// 8-warp TiledMMA, N-heavy layout: covers M=32, N=64, K=16 per atom step
// (warp_m = warp%2 selects the 16 A-rows over 2 warps, warp_n = warp/2 spreads
// the N=64 output over 4 warps). This is the "N=64 -> 4 warps in N" variant used
// by the FF_WG8 8-warp/WG fused_fwd path for the O-chain GEMMs [CS=32, DVh=64]:
// M=32 -> 2 warp_m (exact), N=64 -> 4 warp_n (exact), zero over-tiling, 8 fp32
// C-regs/thread. Pairs with TiledMma8Warp (<4,2,1>, M=128 -> 4 warp_m) for the
// GEMM6 state update [DK=128, DVh=64].
using TiledMma8WarpN = decltype(cute::make_tiled_mma(
    MmaAtomF16{},
    cute::Layout<cute::Shape<cute::_2, cute::_4, cute::_1>>{}));

// SmemCopyAtom: plain shared memory load (no LDSM, no TSM_LD_SWZL)
using SmemCopyAtomF16 = cute::Copy_Atom<cute::DefaultCopy, cute::half_t>;

// Transposing smem load for the B operand via hardware ldmatrix.trans.
// NVIDIA-native mnemonic (ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16) is
// accepted by the PPU backend (the PPU-specific ppu.tc02.ldmatrix mnemonic is
// NOT). Feed B as a transposed layout view over natural [K,N] smem; the .trans
// reads the physical 8x8 core matrices and transposes into the tc02 B fragment.
using SmemCopyAtomBTrans = cute::Copy_Atom<cute::PPU_U16x8_LDSM_T, cute::half_t>;

// ============================================================================
// SmemLayout helpers
// ============================================================================

// Row-major smem layout atom (8 rows × K columns, no swizzle)
// Used for constructing CuTe tensor views over flat smem buffers.
template <int K>
using SmemLayoutAtom = cute::Layout<
    cute::Shape<cute::_8, cute::Int<K>>,
    cute::Stride<cute::Int<K>, cute::_1>>;

// Build a full smem layout for (M, K) tile
template <int M, int K>
using SmemLayout2D = decltype(cute::tile_to_shape(
    SmemLayoutAtom<K>{},
    cute::make_shape(cute::Int<M>{}, cute::Int<K>{})));

// Swizzled smem layout for the transposing ldmatrix B operand (bank-conflict-free).
// Atom = Swizzle<3,3,3> o (8, 64):(64,1). Swizzle<3,3,3> maps 8-half chunk k of
// row r to chunk (k XOR r), so a ldmatrix.trans phase reading 8 consecutive rows
// (K/contraction) at a fixed 8-half column block lands on 8 distinct bank groups.
// Physical shape is (Rows=K, Cols=N); Rows must be a multiple of 8, Cols of 64.
template <int Rows, int Cols>
using SmemLayoutSwzB = decltype(cute::tile_to_shape(
    cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::_64>,
                     cute::Stride<cute::_64, cute::_1>>{}),
    cute::make_shape(cute::Int<Rows>{}, cute::Int<Cols>{})));

// Transposed (Cols, Rows) view over the swizzled buffer -> the mma B operand (N, K).
template <int Rows, int Cols>
using SmemLayoutSwzB_T = decltype(cute::composition(
    SmemLayoutSwzB<Rows, Cols>{},
    cute::make_layout(cute::Shape<cute::Int<Cols>, cute::Int<Rows>>{},
                      cute::GenRowMajor{})));

// Swizzled A-operand layout for the [FF_CS=32, FF_CS=32] Ag buffer used by the
// non-WASP fused_fwd (FF_A_SWZL). This matches the PPU1.5 TSM_LD_SWZL64 hardware
// swizzle mode for 64-byte-wide (= 32-half) rows.
//
// Parameter choice: Swizzle<B=2, M=3, S=3>.
//   * M=3 preserves bits [0, 3) (the 8-half = 16B int4 vec chunk offset), so
//     16B-aligned int4 stores keep all 8 halfs physically contiguous inside one
//     chunk. This is the safety condition for vectorized gate-fused writes.
//   * B=2 XORs 2 bits, which is exactly the 2-bit chunk-col index space
//     (32-half row / 8-half chunk = 4 chunks per row).
//   * S=3 satisfies CuTe's |S| >= B constraint and pulls row bits [M+S, M+S+B)
//     = [6, 8) -- the 2 high bits of the 8-row atom index -- to XOR into the
//     chunk-col index at bits [3, 5).
//
// Effective permutation: chunk_col_new(r) = chunk_col_orig XOR ((r >> 1) & 3),
// so rows {0,1,2,3,4,5,6,7} of an ldmatrix.x4 8-row phase land on chunk-cols
// {0,0,1,1,2,2,3,3}. Combined with the 32-half row stride (rows 0,2,4,6 sit on
// banks 0..15 half; rows 1,3,5,7 sit on banks 16..31 half), the 8 threads
// distribute to disjoint 4-bank groups: {0-3, 16-19, 4-7, 20-23, 8-11, 24-27,
// 12-15, 28-31} -- FULL bank-conflict-free ldmatrix read.
//
// Prior wrong choice (fixed 2026-07-14): Swizzle<3,2,3>. M=2 makes the XOR
// span bit 2, splitting the int4 vec's 8 halfs across two swizzle slots and
// delivering wrong data on ldmatrix (5/5 NaN in test_fused_fwd). WASP's
// Swizzle<3,3,3> works only on CS=64 rows (need 8-half chunk x 8 chunk-cols =
// 3-bit chunk index) and cannot compose with a 32-half row.
template <int Rows, int Cols>
using SmemLayoutSwzA_cs32 = decltype(cute::tile_to_shape(
    cute::composition(
        cute::Swizzle<2, 3, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::_32>,
                     cute::Stride<cute::_32, cute::_1>>{}),
    cute::make_shape(cute::Int<Rows>{}, cute::Int<Cols>{})));

// ============================================================================
// AIU .swzl helpers for the CS=32 non-WASP fused_fwd (TiledMmaM32, 2x2 warps)
// ----------------------------------------------------------------------------
// The WASP fused_fwd uses CUBE_H=CUBE_W=64 (swzl_mode=0, "swizzle128" atom)
// because its FF_CS=64 fills one physical cube row-block. The non-WASP fused_fwd
// runs FF_CS=32, so the matching AIU atom is CUBE_H=CUBE_W=32 -- the "swizzle64"
// f16 atom (CUBE_W*sizeof(half)=64B, so swzl_mode=1 is picked automatically by
// PPU0015_TSM_LD_SWZL::swzl_mode = CUBE_W*sizeof(Element)==128 ? 0 : 1).
// Physical layout of a [CS=32, DK=128] swzl tile: DK/32 = 4 contiguous
// 32x32 col-cubes, each a 32B row × 32 rows 128B-XOR tile (mode 1).
//
// TiledMmaM32 = Layout<Shape<_2,_2,_1>> (col-major default): warp_m = warp%2
// selects the 16 A-rows within a 32-row M-tile, warp_n = warp/2 selects the
// 16 B-rows within a 32-row N-tile. M_tiles = M/32, N_tiles = N/32, K_tiles = K/16.
// KT_PER_CUBE = CUBE_W/16 = 2 MMA k-tiles per physical col-cube; INST = K/32
// col-cubes total.
// ============================================================================

// FF_SWZL_READ_CVTA: convert the base pointer handed to the AIU .swzl READ atom
// (PPU0015_TSM_LD_SWZL) from a raw generic smem pointer to a CTA-relative
// .shared byte offset via __cvta_generic_to_shared (default 1). MUST match the
// .swzl WRITE side (FF_SWZL_CVTA): the write feeds the DMA a cvta'd .shared
// address, but the read atom internally computes tsm_add=(uintptr_t)base/16 on
// whatever base it is given. If the read starts from the RAW generic pointer it
// carries the per-CTA shared-window base, so under 2 blk/CU co-residency the read
// addresses a different TSM slice than the write wrote -> A/B corruption. Passing
// the cvta'd shared offset here strips the window base so read and write address
// the identical TSM location. Set 0 to reproduce the old raw-generic read.

// Map a generic smem base pointer to the .swzl READ atom's expected base.
CUTE_DEVICE __half* ff_swzl_read_base(const __half* p) {
#if defined(__HGGC_ARCH__)
  return reinterpret_cast<__half*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(p)));
#else
  return const_cast<__half*>(p);
#endif
}

// flashqla_gemm_ss_swzlA_m32: like flashqla_gemm_ss but A is read from an AIU
// .swzl-written smem buffer (physical [M,K], M/32 col-cubes of 32x32 128B-XOR,
// swzl_mode=1) via TSM_LD_SWZL non-trans, and B is plain-loaded from smem.
// Clears tC. Used for GEMM1 (K.A + hstmT.B plain) and GEMM3 (Q.A + hstmT.B).
// Requires M<=32, K a multiple of 32.
template <int M, int K, typename SmemB, typename TC>
CUTE_DEVICE void flashqla_gemm_ss_swzlA_m32(
    const __half* Aswz, SmemB const& sB, int tidx, TC& tC) {
  constexpr int CH = 32;             // cube height (rows) == FF_CS
  constexpr int CW = 32;             // cube width  (cols) -> swzl_mode=1
  constexpr int KT_PER_CUBE = CW / 16;   // = 2 MMA k-tiles per physical col-cube
  constexpr int INST = K / CW;           // number of col-cubes
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      cute::half_t, CH, CW, /*Swap=*/false, /*Trans=*/false, /*InstNum=*/INST>;

  TiledMmaM32 mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;       // 2x2 layout: rows within the M-tile
  // (warp_n contributes to B side; A is warp_m-only.)

  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      SmemLayout2D<M, K>{});
  Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rA_full); ++m) {
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      const int cube = k / KT_PER_CUBE;
      const int col_in_cube = (k % KT_PER_CUBE) * 16;
      const int row = warp_m * 16 + m * 32;   // M-tile stride = 32 (was 64 in WASP)
      Tensor sliceA = rA_full(_, m, k);
      Tensor rA = recast<uint32_t>(sliceA);
      int frag[4] = {0, 0, 0, 0};
      TsmA::copy(frag, reinterpret_cast<void*>(ff_swzl_read_base(Aswz)),
                 /*coord_h=*/col_in_cube, /*coord_w=*/row,
                 /*cube_in_stage=*/cube, /*stage=*/0);
      rA(0) = static_cast<uint32_t>(frag[0]);
      rA(1) = static_cast<uint32_t>(frag[1]);
      rA(2) = static_cast<uint32_t>(frag[2]);
      rA(3) = static_cast<uint32_t>(frag[3]);
    }
  }

  // Partition B: plain lazy load per K-slice.
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomF16{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  clear(tC);
  Tensor rC_full = recast<float>(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_ss_swzlAB_m32: GEMM4 (P = Q @ K^T) variant where BOTH operands
// are read from AIU .swzl-written smem buffers. A (Q) is warp_m-tiled non-trans
// swzl, B (K) is warp_n-tiled non-trans swzl with the TN-atom fragment reg
// permutation (b0,b2,b1,b3). Clears tC. Requires M,N<=32, K a multiple of 32.
template <int M, int N, int K, typename TC>
CUTE_DEVICE void flashqla_gemm_ss_swzlAB_m32(
    const __half* Aswz, const __half* Bswz, int tidx, TC& tC) {
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;   // = 2
  constexpr int INST = K / CW;
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      cute::half_t, CH, CW, /*Swap=*/false, /*Trans=*/false, /*InstNum=*/INST>;
  using TsmB = cute::PPU0015_TSM_LD_SWZL<
      cute::half_t, CH, CW, /*Swap=*/false, /*Trans=*/false, /*InstNum=*/INST>;

  TiledMmaM32 mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;
  const int warp_n = warp / 2;

  // A fragment (Q, warp_m-tiled).
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      SmemLayout2D<M, K>{});
  Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rA_full); ++m) {
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      const int cube = k / KT_PER_CUBE;
      const int col_in_cube = (k % KT_PER_CUBE) * 16;
      const int row = warp_m * 16 + m * 32;
      Tensor sliceA = rA_full(_, m, k);
      Tensor rA = recast<uint32_t>(sliceA);
      int frag[4] = {0, 0, 0, 0};
      TsmA::copy(frag, reinterpret_cast<void*>(ff_swzl_read_base(Aswz)),
                 /*coord_h=*/col_in_cube, /*coord_w=*/row,
                 /*cube_in_stage=*/cube, /*stage=*/0);
      rA(0) = static_cast<uint32_t>(frag[0]);
      rA(1) = static_cast<uint32_t>(frag[1]);
      rA(2) = static_cast<uint32_t>(frag[2]);
      rA(3) = static_cast<uint32_t>(frag[3]);
    }
  }

  // B fragment (K, warp_n-tiled): fill lazily via non-trans swzl read.
  auto dummyB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      SmemLayout2D<N, K>{});
  Tensor rB_full = thr_mma.partition_fragment_B(dummyB);

  clear(tC);
  Tensor rC_full = recast<float>(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    const int cube = k / KT_PER_CUBE;
    const int col_in_cube = (k % KT_PER_CUBE) * 16;
    CUTE_UNROLL
    for (int n = 0; n < cute::size<1>(rB_full); ++n) {
      const int row = warp_n * 16 + n * 32;   // N-tile stride = 32 (was n*16 in WASP TiledMma4Warp)
      uint32_t bfrag[4] = {0, 0, 0, 0};
      TsmB::copy(reinterpret_cast<int*>(bfrag),
                 reinterpret_cast<void*>(ff_swzl_read_base(Bswz)),
                 /*coord_h=*/col_in_cube, /*coord_w=*/row,
                 /*cube_in_stage=*/cube, /*stage=*/0);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tC); ++m) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        // TN atom: B operand fragment = A-read fragment with reg 1<->2 swapped.
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            bfrag[0], bfrag[2], bfrag[1], bfrag[3],
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_ss_swzlA_trans_acc_m32: GEMM6 (h += K^T @ gVd) variant.
// A operand K^T[M=DK, K=CS] is read DIRECTLY from the AIU .swzl-written K buffer
// (physical [CS=32, DK=128], 4 col-cubes of 32x32) via Trans=true TSM_LD_SWZL,
// eliminating the KT_buf scalar transpose. B (gVd) is plain-loaded from an
// [N=DVh, K=CS] smem view (matches the caller's transposed store convention in
// the non-WASP kernel -- gVdswz_g is written as (dv,cs) row-major, which is
// exactly the plain (N,K) B operand shape). Accumulates (no clear(tC)).
// M = DK (must be multiple of 32 -> M/32 col-cubes = INST), K = CS <= 32.
template <int M, int K, typename SmemB, typename TC>
CUTE_DEVICE void flashqla_gemm_ss_swzlA_trans_acc_m32(
    const __half* Aswz, SmemB const& sB, int tidx, TC& tC) {
  constexpr int CH = 32;   // cube height (rows = cs)
  constexpr int CW = 32;   // cube width  (cols = dk_local) -> swzl_mode=1
  using TsmTrans = cute::PPU0015_TSM_LD_SWZL<
      cute::half_t, CH, CW, /*Swap=*/false, /*Trans=*/true, /*InstNum=*/1>;

  TiledMmaM32 mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;

  // A fragment shape (u32x4, M_tiles, K_tiles) via dummy tensor.
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      SmemLayout2D<M, K>{});
  Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rA_full); ++m) {
    const int m_global = warp_m * 16 + m * 32;   // K^T row (= dk) 0..M-1
    const int cube = m_global / 32;              // which physical col-cube of K
    const int dk_in_cube = m_global % 32;        // row inside the transposed tile
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      const int cs_col = k * 16;                 // contraction offset inside cube
      int afrag[4] = {0, 0, 0, 0};
      TsmTrans::copy(afrag, reinterpret_cast<void*>(ff_swzl_read_base(Aswz)),
                     /*coord_h=*/cs_col, /*coord_w=*/dk_in_cube,
                     /*cube_in_stage=*/0, /*stage=*/cube);
      Tensor sliceA = rA_full(_, m, k);
      Tensor rA = recast<uint32_t>(sliceA);
      rA(0) = static_cast<uint32_t>(afrag[0]);
      rA(1) = static_cast<uint32_t>(afrag[1]);
      rA(2) = static_cast<uint32_t>(afrag[2]);
      rA(3) = static_cast<uint32_t>(afrag[3]);
    }
  }

  // Partition B: plain lazy load per K-slice (matches _acc convention).
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomF16{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  // NOTE: no clear(tC) -- accumulate into caller's h_reg.
  Tensor rC_full = recast<float>(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// ============================================================================
// GEMM helper: both operands from shared memory
// A loaded fully to registers, B loaded lazily per K-slice.
// ============================================================================

// flashqla_gemm_ss: Shared × Shared GEMM with zero-acc
//   sA: smem tensor shape (M, K) — will be partition_fragment_A
//   sB: smem tensor shape (N, K) — will be partition_fragment_B (lazy)
//   tC: output accumulator (cleared before GEMM)
//   tidx: thread index within the 128-thread group
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp>
CUTE_DEVICE void flashqla_gemm_ss(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomF16{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  // Load all K-slices of A
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  // Partition B: lazy load per K-slice
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomF16{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  // Clear accumulator
  clear(tC);
  Tensor rC_full = recast<float>(tC);

  // K-loop: load B lazily, call MMA directly
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_ss_acc: identical to flashqla_gemm_ss (plain DefaultCopy for BOTH
// A and B, B provided as a plain (N,K) row-major smem tensor) but does NOT clear
// the accumulator — it accumulates tC += A@B. Used to bypass the swizzle +
// ldmatrix.trans B path: the caller pre-transposes the B operand into a plain
// [N,K] buffer, so no hardware transpose is needed. CS-agnostic and correct for
// any tile shape the plain (N,K) B path supports (same path as GEMM1/GEMM3).
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp>
CUTE_DEVICE void flashqla_gemm_ss_acc(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomF16{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  // Partition B: plain lazy load per K-slice (B is plain (N,K) row-major)
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomF16{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  // NOTE: NO clear(tC) — accumulate into the existing accumulator.
  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_ss_ldsmB: same as flashqla_gemm_ss but the B operand is loaded
// with the hardware transposing ldmatrix (PPU_U16x8_LDSM_T). Caller must pass
// sB as a transposed layout VIEW over natural [K,N] smem (shape (N,K), stride
// (1, ldK)), so the .trans reads the physical 8x8 core matrices and delivers
// them already transposed into the tc02 B fragment (no elementwise transpose).
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp>
CUTE_DEVICE void flashqla_gemm_ss_ldsmB(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomF16{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  // Partition B: transposing ldmatrix load per K-slice
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomBTrans{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  clear(tC);
  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_ss_ldsmB_acc: same as flashqla_gemm_ss_ldsmB (hardware-transpose
// B via ldmatrix.trans) but does NOT clear the accumulator (accumulates:
// tC += A@B). Used by GEMM6 to update register-h in place
// (h_reg += K^T@(g_rev*Vd)), avoiding a separate 128-reg h_acc frame.
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp>
CUTE_DEVICE void flashqla_gemm_ss_ldsmB_acc(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomF16{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  // Partition B: transposing ldmatrix load per K-slice
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomBTrans{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  // NOTE: no clear(tC) — accumulate into the caller's accumulator.
  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_ss_acc: same as above but does NOT clear accumulator (accumulates)
template <typename SmemA, typename SmemB, typename TC>
CUTE_DEVICE void flashqla_gemm_ss_acc(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  TiledMma4Warp mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomF16{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomF16{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_lazy_b: A already in registers, B loaded lazily from smem
//   tA: register fragment (atom, M_tiles, K_tiles)
//   sB: smem tensor (N, K)
//   tC: accumulator (NOT cleared — caller must clear if needed)
template <typename TA, typename SmemB, typename TC>
CUTE_DEVICE void flashqla_gemm_lazy_b(
    TA const& tA, SmemB const& sB,
    int tidx, TC& tC) {
  TiledMma4Warp mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomF16{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = tA(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_lazy_b_ldsmB: same as flashqla_gemm_lazy_b (A already in
// registers) but the B operand is loaded with the transposing ldmatrix
// (PPU_U16x8_LDSM_T). Caller passes sB as the transposed swizzled read view
// (shape (N,K)) over a Swizzle<3,3,3> buffer; tC is NOT cleared (caller clears).
template <typename TA, typename SmemB, typename TC, typename Mma = TiledMma4Warp>
CUTE_DEVICE void flashqla_gemm_lazy_b_ldsmB(
    TA const& tA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomBTrans{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = tA(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// flashqla_gemm_ss_regB: A from smem, B ALREADY in registers (a B-fragment
// produced by an upstream GEMM's transposed output). No smem round-trip for B.
// tC is cleared. rB_full shape: (atom, N_tiles, K_tiles).
template <typename SmemA, typename RegB, typename TC>
CUTE_DEVICE void flashqla_gemm_ss_regB(
    SmemA const& sA, RegB const& rB_full,
    int tidx, TC& tC) {
  TiledMma4Warp mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomF16{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  clear(tC);
  Tensor rC_full = recast<float>(tC);

  // B is already resident in registers (rB_full): no smem load needed.
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// ============================================================================
// Utility: allocate register accumulator for (M, N) output
// ============================================================================

template <int M, int N, typename Mma = TiledMma4Warp>
CUTE_DEVICE auto make_acc(int tidx) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  // Use a dummy smem tensor just for shape — partition_fragment_C only
  // reads the layout, not the data pointer.
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      SmemLayout2D<M, N>{});
  return thr_mma.partition_fragment_C(dummy);
}

// Allocate a register B-fragment for a (N, K) operand — used to receive an
// upstream GEMM's transposed output as the next GEMM's B operand (register
// hand-off, no smem). Layout matches partition_fragment_B over row-major (N,K).
template <int N, int K>
CUTE_DEVICE auto make_bfrag(int tidx) {
  TiledMma4Warp mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      SmemLayout2D<N, K>{});
  return thr_mma.partition_fragment_B(dummy);
}

// ============================================================================
// Utility: store fp32 accumulator back to fp16 smem
// ============================================================================

// Store MMA accumulator (fp32) to shared memory as fp16.
// tC shape: (atom=8, M_tiles, N_tiles)
// dst: flat __half* pointer to output buffer, row-major (M, N) with stride ldc
template <typename TC>
CUTE_DEVICE void store_acc_to_smem_fp16(
    TC const& tC, __half* dst, int M, int N, int ldc, int tidx) {
  // Use the MMA's thread-value mapping to determine which elements this thread owns.
  // For PPU 16x16x16 TN atom with AtomLayout<4,1,1>:
  //   Each thread owns 8 elements per (M_tile, N_tile) atom.
  //   Thread mapping within a 16x16 tile:
  //     row = (tid/4)*2 + (elem/4) within the 16-row tile
  //     col = (tid%4)*4 + (elem%4) ... actually this depends on the atom layout.
  //
  // Simpler approach: use CuTe's partition to figure out the mapping.
  TiledMma4Warp mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Create a smem tensor with the target layout
  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(dst),
      cute::make_layout(cute::make_shape(M, N), cute::make_stride(ldc, 1)));

  // Get the C partition's coordinate mapping
  Tensor tCsC = thr_mma.partition_C(sOut);

  // Copy from register accumulator to smem via the partition mapping
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n) {
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v) {
        tCsC(v, m, n) = __float2half(rC(v, m, n));
      }
    }
  }
}

// Store accumulator to fp32 smem
// Mma defaults to TiledMma4Warp (existing call sites unchanged); pass TiledMmaM32
// when the accumulator was produced by an M32 (<2,2,1>) tiled MMA so the C
// partition mapping matches the register fragment ownership.
template <typename Mma = TiledMma4Warp, typename TC>
CUTE_DEVICE void store_acc_to_smem_fp32(
    TC const& tC, float* dst, int M, int N, int ldc, int tidx) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(dst),
      cute::make_layout(cute::make_shape(M, N), cute::make_stride(ldc, 1)));

  Tensor tCsC = thr_mma.partition_C(sOut);
  Tensor rC = recast<float>(tC);

  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n) {
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v) {
        tCsC(v, m, n) = rC(v, m, n);
      }
    }
  }
}

// Fused Gram-store + beta-scale + M-build (kkt_solve).  Takes the fp32 Gram
// accumulator tC (CSxCS, produced by an M32 <2,2,1> tiled MMA) and, entirely
// from registers, computes the unit-lower-triangular M matrix
//   M[i,j] = 1 (i==j); -KKT[i,j]*beta[j] (i>j, both < actual); 0 otherwise
// writing fp16 M to Mh and zeroing Ah.  This ELIMINATES the intermediate fp32
// KKT_buf (4KB) and the store→reload __syncthreads: M is a pure elementwise
// transform of KKT, so no cross-thread data movement is needed.  The 4-warp
// <2,2,1> C partition covers the full CSxCS exactly once (4*32*8 = 1024 = 32^2),
// and partition_C over an identity tensor recovers each fragment element's (i,j).
template <int CS, typename Mma, typename TC>
CUTE_DEVICE void store_gram_build_M_m32(
    TC const& tC, __half* Mh, __half* Ah, const float* beta,
    int actual, int tidx) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto shapeCC = cute::make_shape(cute::Int<CS>{}, cute::Int<CS>{});
  auto sM = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(Mh)),
      cute::make_layout(shapeCC, cute::make_stride(cute::Int<CS>{}, cute::Int<1>{})));
  auto sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(Ah)),
      cute::make_layout(shapeCC, cute::make_stride(cute::Int<CS>{}, cute::Int<1>{})));
  auto cId = cute::make_identity_tensor(shapeCC);
  Tensor tMsM = thr_mma.partition_C(sM);
  Tensor tAsA = thr_mma.partition_C(sA);
  Tensor tCcC = thr_mma.partition_C(cId);
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tMsM); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tMsM); ++n) {
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tMsM); ++v) {
        auto coord = tCcC(v, m, n);
        const int i = cute::get<0>(coord);
        const int j = cute::get<1>(coord);
        const float scaled = rC(v, m, n) * beta[j];
        const float mval = (i == j) ? 1.0f
                         : ((i > j && i < actual && j < actual) ? -scaled : 0.0f);
        tMsM(v, m, n) = cute::half_t(mval);
        tAsA(v, m, n) = cute::half_t(0.0f);
      }
    }
  }
}

// ============================================================================
// Utility: make CuTe smem tensor from flat pointer + dimensions
// ============================================================================

// Create a row-major (M, K) CuTe tensor view over a flat smem pointer
template <int M, int K>
CUTE_DEVICE auto make_smem_tensor(cute::half_t* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(ptr),
      SmemLayout2D<M, K>{});
}

template <int M, int K>
CUTE_DEVICE auto make_smem_tensor(const cute::half_t* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(ptr),
      SmemLayout2D<M, K>{});
}

// Overload for __half pointers (reinterpret as cute::half_t)
template <int M, int K>
CUTE_DEVICE auto make_smem_tensor_h(__half* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(ptr)),
      SmemLayout2D<M, K>{});
}

template <int M, int K>
CUTE_DEVICE auto make_smem_tensor_h(const __half* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<const cute::half_t*>(ptr)),
      SmemLayout2D<M, K>{});
}

// Swizzled (Rows=K, Cols=N) WRITE-view over a flat smem buffer. Write by (k, n)
// coordinate; the swizzle relocates banks so a subsequent ldmatrix.trans read is
// bank-conflict-free. Pair with make_smem_swzB_T for the transposed read view.
template <int Rows, int Cols>
CUTE_DEVICE auto make_smem_swzB(__half* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(ptr)),
      SmemLayoutSwzB<Rows, Cols>{});
}

// Transposed (Cols=N, Rows=K) READ-view over the same swizzled buffer: the mma
// B operand (N, K), to be loaded with the transposing ldmatrix copy atom.
template <int Rows, int Cols>
CUTE_DEVICE auto make_smem_swzB_T(const __half* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<const cute::half_t*>(ptr)),
      SmemLayoutSwzB_T<Rows, Cols>{});
}

// Swizzled A-operand VIEW for the non-WASP Ag buffer (Rows = FF_CS = 32,
// Cols = FF_CS = 32). Both the int4 vec-store writer (WG-cooperative gating)
// and the ldmatrix A-read (GEMM2) index this view by (row, col); the
// Swizzle<2,3,3> layout composes the physical offset. See SmemLayoutSwzA_cs32.
template <int Rows, int Cols>
CUTE_DEVICE auto make_smem_swzA_cs32(__half* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(ptr)),
      SmemLayoutSwzA_cs32<Rows, Cols>{});
}

template <int Rows, int Cols>
CUTE_DEVICE auto make_smem_swzA_cs32(const __half* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<const cute::half_t*>(ptr)),
      SmemLayoutSwzA_cs32<Rows, Cols>{});
}

// ============================================================================
// Utility: transpose smem tensor (swap first two dimensions)
// For TN MMA: to compute C = A @ B (not A @ B^T), pass transposed B.
// ============================================================================

template <typename Tensor>
CUTE_DEVICE auto transpose_tensor(Tensor&& t) {
  // Swap dimensions 0 and 1: (M, K) -> (K, M)
  auto layout = t.layout();
  auto new_layout = cute::make_layout(
      cute::make_shape(cute::get<1>(layout.shape()), cute::get<0>(layout.shape())),
      cute::make_stride(cute::get<1>(layout.stride()), cute::get<0>(layout.stride())));
  return cute::make_tensor(t.data(), new_layout);
}

// ============================================================================
// Single-warp (32-thread) 16x16x16 block GEMM helpers.
// Used by kkt_solve's blocked triangular inverse: each warp independently
// computes one 16x16 block MMA (PPU1.5 supports subset-warp tc02 MMA atoms).
// TiledMma1Warp covers exactly one atom: M=16, N=16, K=16.
// ============================================================================

using TiledMma1Warp = decltype(cute::make_tiled_mma(
    MmaAtomF16{},
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));

// Allocate a 16x16 fp32 register accumulator for single-warp MMA. lane in [0,32).
CUTE_DEVICE auto make_acc16_1warp(int lane) {
  TiledMma1Warp mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      SmemLayout2D<16, 16>{});
  return thr_mma.partition_fragment_C(dummy);
}

// 2-WARP swzl Gram (kkt_solve 2-task interleave).  ONE warp-pair (64 threads)
// computes the full 32x32 Gram KKT = K@K^T from the .swzl-written K buffer and
// stores it fp32 to KKT_dst (row-major, leading dim ldc).  The pair-local warp
// (local_tid/32 ∈ {0,1}) owns output row-block m_block and iterates BOTH column
// blocks n∈{0,1}, so 2 warps cover all four 16x16 output blocks (each does 2).
// The swzl reads use the SAME atom + coords as flashqla_gemm_ss_swzlAB_m32
// (Swap=false, Trans=false); A rows = m_block*16 (shared across n), B rows =
// n*16.  Accumulators use the TiledMma1Warp C layout and are stored via the
// validated store_acc_to_smem_fp32<TiledMma1Warp> path.  Kdim must be a
// multiple of 32 (col-cubes = Kdim/32).
template <int Kdim>
CUTE_DEVICE void flashqla_gemm_ss_swzl_gram_2warp(
    const __half* Kswz, int local_tid, float* KKT_dst, int ldc) {
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;     // = 2
  constexpr int INST = Kdim / CW;          // col-cubes
  using Tsm = cute::PPU0015_TSM_LD_SWZL<
      cute::half_t, CH, CW, /*Swap=*/false, /*Trans=*/false, /*InstNum=*/INST>;

  const int local_warp = local_tid / 32;   // 0 or 1
  const int lane       = local_tid & 31;
  const int m_block    = local_warp;        // output row-block owned by this warp
  void* base = reinterpret_cast<void*>(ff_swzl_read_base(Kswz));

  auto accN0 = make_acc16_1warp(lane); clear(accN0);
  auto accN1 = make_acc16_1warp(lane); clear(accN1);
  Tensor rC0 = recast<float>(accN0);
  Tensor rC1 = recast<float>(accN1);

  CUTE_UNROLL
  for (int k = 0; k < INST * KT_PER_CUBE; ++k) {
    const int cube = k / KT_PER_CUBE;
    const int col_in_cube = (k % KT_PER_CUBE) * 16;
    // A fragment: rows [m_block*16, +16), shared by both n-blocks.
    int afrag[4] = {0, 0, 0, 0};
    Tsm::copy(afrag, base, /*coord_h=*/col_in_cube, /*coord_w=*/m_block * 16,
              /*cube_in_stage=*/cube, /*stage=*/0);
    // n_block = 0 (columns [0,16))
    {
      int bfrag[4] = {0, 0, 0, 0};
      Tsm::copy(bfrag, base, /*coord_h=*/col_in_cube, /*coord_w=*/0,
                /*cube_in_stage=*/cube, /*stage=*/0);
      ppu_mma_f16_fixed_impl(
          rC0(0,0,0), rC0(1,0,0), rC0(2,0,0), rC0(3,0,0),
          rC0(4,0,0), rC0(5,0,0), rC0(6,0,0), rC0(7,0,0),
          (uint32_t)afrag[0], (uint32_t)afrag[1], (uint32_t)afrag[2], (uint32_t)afrag[3],
          (uint32_t)bfrag[0], (uint32_t)bfrag[2], (uint32_t)bfrag[1], (uint32_t)bfrag[3],
          rC0(0,0,0), rC0(1,0,0), rC0(2,0,0), rC0(3,0,0),
          rC0(4,0,0), rC0(5,0,0), rC0(6,0,0), rC0(7,0,0));
    }
    // n_block = 1 (columns [16,32))
    {
      int bfrag[4] = {0, 0, 0, 0};
      Tsm::copy(bfrag, base, /*coord_h=*/col_in_cube, /*coord_w=*/16,
                /*cube_in_stage=*/cube, /*stage=*/0);
      ppu_mma_f16_fixed_impl(
          rC1(0,0,0), rC1(1,0,0), rC1(2,0,0), rC1(3,0,0),
          rC1(4,0,0), rC1(5,0,0), rC1(6,0,0), rC1(7,0,0),
          (uint32_t)afrag[0], (uint32_t)afrag[1], (uint32_t)afrag[2], (uint32_t)afrag[3],
          (uint32_t)bfrag[0], (uint32_t)bfrag[2], (uint32_t)bfrag[1], (uint32_t)bfrag[3],
          rC1(0,0,0), rC1(1,0,0), rC1(2,0,0), rC1(3,0,0),
          rC1(4,0,0), rC1(5,0,0), rC1(6,0,0), rC1(7,0,0));
    }
  }
  // Store both 16x16 blocks (fp32) to KKT_dst at (m_block,0) and (m_block,1).
  store_acc_to_smem_fp32<TiledMma1Warp>(
      accN0, KKT_dst + m_block * 16 * ldc + 0,  16, 16, ldc, lane);
  store_acc_to_smem_fp32<TiledMma1Warp>(
      accN1, KKT_dst + m_block * 16 * ldc + 16, 16, 16, ldc, lane);
}

// Make a 16x16 fp16 block view of a row-major buffer with leading dim `LD`
// (compile-time). Block (bi,bj) starts at base + bi*16*LD + bj*16. The layout
// must be fully static so CuTe's MMA partition_fragment works.
template <int LD>
CUTE_DEVICE auto block16_view(__half* base, int bi, int bj) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(base + bi * 16 * LD + bj * 16)),
      cute::make_layout(
          cute::make_shape(cute::Int<16>{}, cute::Int<16>{}),
          cute::make_stride(cute::Int<LD>{}, cute::Int<1>{})));
}

// Single-warp GEMM (accumulating): tC += A[16,16] . B[16,16]^T  (TN). No clear.
//   sA: (16,16) view;  sB: (16,16) view;  lane in [0,32)
// For NN (tC += A.B) pass sB as transpose_tensor(B).
template <typename SmemA, typename SmemB, typename TC>
CUTE_DEVICE void gemm16_ss_1warp_acc(
    SmemA const& sA, SmemB const& sB, int lane, TC& tC) {
  TiledMma1Warp mma;
  auto thr_mma = mma.get_thread_slice(lane);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomF16{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomF16{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(lane);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor sliceB = rB_full(_, n, k);
        Tensor rA = recast<uint32_t>(sliceA);
        Tensor rB = recast<uint32_t>(sliceB);
        ppu_mma_f16_fixed_impl(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// Single-warp GEMM: C[16,16] = A[16,16] . B[16,16]^T  (TN semantics). Clears tC.
//   sA: (16,16) row-major smem view;  sB: (16,16) row-major smem view
//   lane: threadIdx within the warp, [0,32)
// To compute C = A . B (NN) instead, pass sB as a transposed view of B
// (transpose_tensor(B) turns a (K,N) view into the (N,K) operand).
template <typename SmemA, typename SmemB, typename TC>
CUTE_DEVICE void gemm16_ss_1warp(
    SmemA const& sA, SmemB const& sB, int lane, TC& tC) {
  clear(tC);
  gemm16_ss_1warp_acc(sA, sB, lane, tC);
}

// Store a 16x16 fp32 accumulator to fp16 smem (row-major, leading dim ldc).
template <typename TC>
CUTE_DEVICE void store_acc16_fp16_1warp(
    TC const& tC, __half* dst, int ldc, int lane) {
  TiledMma1Warp mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(dst)),
      cute::make_layout(cute::make_shape(16, 16), cute::make_stride(ldc, 1)));
  Tensor tCsC = thr_mma.partition_C(sOut);
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v)
        tCsC(v, m, n) = cute::half_t(rC(v, m, n));
}

}  // namespace flashqla
