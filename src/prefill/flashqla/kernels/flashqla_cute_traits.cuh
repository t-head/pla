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

// flashqla_arch.cuh is the single point of truth for __HGGC_ARCH__ dispatch. It
// selects the matching actlize ppu0010 / ppu0015 arch/atom/copy headers
// (only the family for the target chip is pulled in) and exposes the
// arch-invariant flashqla::Ppu* alias family. It MUST come before any
// other actlize include so the trait family is fixed before they are parsed.
#include "flashqla_arch.cuh"

#include "cute/arch/copy_ppu.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_ppu.hpp"

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
#if defined(FLASHQLA_ARCH_150)
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
#elif defined(FLASHQLA_ARCH_100)
  // Same mnemonic on tc01; the PPU1.0/1.1 backend takes the f32 accumulator in
  // the float register class. Mirrors actlize
  // PPU0010_16x16x16_F32F16F16F32_TN.
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
#if defined(FLASHQLA_ARCH_150) || defined(FLASHQLA_ARCH_100)
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

// Inherit thread/value layout from the arch-dispatched fp16 atom traits
template <>
struct MMA_Traits<PPU_16x16x16_F32F16F16F32_TN_FIXED>
    : MMA_Traits<flashqla::PpuF16MMA> {};

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
// Element-dispatched MMA/copy traits
// ============================================================================

template <typename Element>
struct ElementTraits;

template <>
struct ElementTraits<__half> {
  using CuteType = cute::half_t;
  using MmaAtom = cute::MMA_Atom<cute::PPU_16x16x16_F32F16F16F32_TN_FIXED>;
  CUTE_DEVICE static float to_float(__half x) { return __half2float(x); }
  CUTE_DEVICE static __half from_float(float x) { return __float2half(x); }
  CUTE_DEVICE static void mma(
      float& d0, float& d1, float& d2, float& d3,
      float& d4, float& d5, float& d6, float& d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const& c0, float const& c1, float const& c2, float const& c3,
      float const& c4, float const& c5, float const& c6, float const& c7) {
    ppu_mma_f16_fixed_impl(d0, d1, d2, d3, d4, d5, d6, d7,
        a0, a1, a2, a3, b0, b1, b2, b3,
        c0, c1, c2, c3, c4, c5, c6, c7);
  }
};

template <>
struct ElementTraits<__ppu_bfloat16> {
  using CuteType = cute::bfloat16_t;
  using MmaAtom = cute::MMA_Atom<flashqla::PpuBf16MMA>;
  CUTE_DEVICE static float to_float(__ppu_bfloat16 x) { return __bfloat162float(x); }
  CUTE_DEVICE static __ppu_bfloat16 from_float(float x) { return __float2bfloat16(x); }
  CUTE_DEVICE static void mma(
      float& d0, float& d1, float& d2, float& d3,
      float& d4, float& d5, float& d6, float& d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const& c0, float const& c1, float const& c2, float const& c3,
      float const& c4, float const& c5, float const& c6, float const& c7) {
    flashqla::PpuBf16MMA::fma(
        d0, d1, d2, d3, d4, d5, d6, d7,
        a0, a1, a2, a3, b0, b1, b2, b3,
        c0, c1, c2, c3, c4, c5, c6, c7);
  }
};

template <typename Element>
CUTE_DEVICE float element_to_float(Element x) { return ElementTraits<Element>::to_float(x); }

template <typename Element>
CUTE_DEVICE Element float_to_element(float x) { return ElementTraits<Element>::from_float(x); }

template <typename Element>
CUTE_DEVICE void ppu_mma_impl(
    float& d0, float& d1, float& d2, float& d3,
    float& d4, float& d5, float& d6, float& d7,
    uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
    uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
    float const& c0, float const& c1, float const& c2, float const& c3,
    float const& c4, float const& c5, float const& c6, float const& c7) {
  ElementTraits<Element>::mma(d0, d1, d2, d3, d4, d5, d6, d7,
      a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7);
}

template <typename Element>
using MmaAtom = typename ElementTraits<Element>::MmaAtom;
using MmaAtomF16 = MmaAtom<__half>;
using MmaAtomBF16 = MmaAtom<__ppu_bfloat16>;

// ============================================================================
// Recurrent-state pool element
// ============================================================================
// The pool holding h0 / ht / the indexed state is either fp32 or bf16. The
// recurrence itself always runs in fp32 registers and only reaches memory at
// the two kernel boundaries, so the pool dtype merely selects how those
// boundary accesses convert.
//
// The dtype is a compile-time parameter, not a runtime flag. The boundary
// loops are fully unrolled over every state register, so a runtime select
// duplicates the unrolled body and costs the fp32 path 0.3-0.8% end to end;
// with the type fixed at compile time the fp32 instantiation reduces to the
// plain fp32 load/store it replaces.
template <bool BF16>
struct StatePoolElement { using type = float; };
template <>
struct StatePoolElement<true> { using type = __ppu_bfloat16; };
template <bool BF16>
using StatePool = typename StatePoolElement<BF16>::type;

CUTE_DEVICE float state_to_f32(float x) { return x; }
CUTE_DEVICE float state_to_f32(__ppu_bfloat16 x) { return __bfloat162float(x); }
CUTE_DEVICE void state_from_f32(float& dst, float x) { dst = x; }
CUTE_DEVICE void state_from_f32(__ppu_bfloat16& dst, float x) {
  dst = __float2bfloat16(x);
}

// Scalar accessors for the CP correction kernel, which runs once per CP call
// on a state-sized tile and is far off the critical path: there a runtime
// dtype flag is cheaper than doubling its instantiations.
CUTE_DEVICE float load_state_f32(const void* p, int64_t off, bool is_bf16) {
  return is_bf16
      ? __bfloat162float(reinterpret_cast<const __ppu_bfloat16*>(p)[off])
      : reinterpret_cast<const float*>(p)[off];
}

CUTE_DEVICE void store_state_f32(void* p, int64_t off, float x, bool is_bf16) {
  if (is_bf16) {
    reinterpret_cast<__ppu_bfloat16*>(p)[off] = __float2bfloat16(x);
  } else {
    reinterpret_cast<float*>(p)[off] = x;
  }
}

// ============================================================================
// TiledMMA configurations
// ============================================================================

// 4-warp TiledMMA: covers M=64, N=16, K=16 per atom step
// For a (M,N,K) problem: M_tiles = M/64, N_tiles = N/16, K_tiles = K/16
template <typename Element = __half>
using TiledMma4WarpT = decltype(cute::make_tiled_mma(
    MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>{}));
using TiledMma4Warp = TiledMma4WarpT<__half>;

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
template <typename Element = __half>
using TiledMmaM32T = decltype(cute::make_tiled_mma(
    MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{}));
using TiledMmaM32 = TiledMmaM32T<__half>;

template <typename Element = __half>
using TiledMma8WarpT = decltype(cute::make_tiled_mma(
    MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>{}));
using TiledMma8Warp = TiledMma8WarpT<__half>;

// 8-warp TiledMMA, N-heavy layout: covers M=32, N=64, K=16 per atom step
// (warp_m = warp%2 selects the 16 A-rows over 2 warps, warp_n = warp/2 spreads
// the N=64 output over 4 warps). This is the "N=64 -> 4 warps in N" variant used
// by the FF_WG8 8-warp/WG fused_fwd path for the O-chain GEMMs [CS=32, DVh=64]:
// M=32 -> 2 warp_m (exact), N=64 -> 4 warp_n (exact), zero over-tiling, 8 fp32
// C-regs/thread. Pairs with TiledMma8Warp (<4,2,1>, M=128 -> 4 warp_m) for the
// GEMM6 state update [DK=128, DVh=64].
template <typename Element = __half>
using TiledMma8WarpNT = decltype(cute::make_tiled_mma(
    MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_2, cute::_4, cute::_1>>{}));
using TiledMma8WarpN = TiledMma8WarpNT<__half>;

// SmemCopyAtom: plain shared memory load (no LDSM, no TSM_LD_SWZL)
template <typename Element = __half>
using SmemCopyAtomT = cute::Copy_Atom<cute::DefaultCopy, typename ElementTraits<Element>::CuteType>;
using SmemCopyAtomF16 = SmemCopyAtomT<__half>;

// Transposing smem load for the B operand via hardware ldmatrix.trans.
// NVIDIA-native mnemonic (ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16) is
// accepted by the PPU backend (the PPU-specific ppu.tc02.ldmatrix mnemonic is
// NOT). Feed B as a transposed layout view over natural [K,N] smem; the .trans
// reads the physical 8x8 core matrices and transposes into the tc02 B fragment.
template <typename Element = __half>
using SmemCopyAtomBTransT = cute::Copy_Atom<cute::PPU_U16x8_LDSM_T, typename ElementTraits<Element>::CuteType>;
using SmemCopyAtomBTrans = SmemCopyAtomBTransT<__half>;

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
template <typename Element>
CUTE_DEVICE Element* ff_swzl_read_base(const Element* p) {
#if defined(__HGGC_ARCH__)
  return reinterpret_cast<Element*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(p)));
#else
  return const_cast<Element*>(p);
#endif
}

// flashqla_gemm_ss_swzlA_m32: like flashqla_gemm_ss but A is read from an AIU
// .swzl-written smem buffer (physical [M,K], M/32 col-cubes of 32x32 128B-XOR,
// swzl_mode=1) via TSM_LD_SWZL non-trans. B uses the caller-selected copy atom
// (plain by default). Clears tC. Used for GEMM1 (K.A + hstmT.B) and GEMM3
// (Q.A + hstmT.B).
// Requires M<=32, K a multiple of 32.
#if defined(FLASHQLA_ARCH_100)
// Forward declarations: the tc01 trans-GEMM bodies below delegate to these
// generic helpers / view builder, which are defined later in this header.
template <typename SmemA, typename SmemB, typename TC, typename Mma, typename Element>
CUTE_DEVICE void flashqla_gemm_ss_acc(SmemA const& sA, SmemB const& sB, int tidx, TC& tC);
template <int N, int K, typename SmemA, typename TC, typename Element>
CUTE_DEVICE void flashqla_gemm_ss_regB_acc_m32(
    SmemA const& sA, const uint32_t b_regs[2][2][4], int tidx, TC& tC);

// ---------------------------------------------------------------------------
// PPU1.0/1.1 shared AIU-native tile traits + inlined GEMM bodies.
//
// On tc01 the swzl A/B operands are written by the closed actlize pair
// (PPU0010_AIU_LOAD, non-swzl, AIU-native cube-major SmemLayout) at the
// kernel's load sites, and read back here through the matching
// PPU0010_TSM_LD_SWZL atom. These helpers give the four flashqla_gemm_*
// swzl entry points a single tc01 body so every call site in fused_fwd /
// prepare_h / kkt_solve is served without changing the call sites.
//
// BLK_MN/BLK_K parametrise the tile: (32,128) for K/Q/V, (32,32) for A. The
// AiuContElemSize (128B cap) and InstNum are derived exactly as in
// ppu_mma_builder.inl.
// ---------------------------------------------------------------------------
template <typename Element, int BLK_MN, int BLK_K>
struct FqAiuTileTc01 {
  using CuteElem = typename ElementTraits<Element>::CuteType;
  static constexpr int BlockContSize   = BLK_K * static_cast<int>(sizeof(CuteElem));
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  static constexpr int AiuContElemSize = AiuContByteSize / static_cast<int>(sizeof(CuteElem));
  static constexpr int InstNum         = BLK_K / AiuContElemSize;

  using SmemLayoutAtom = cute::Layout<
      cute::Shape<cute::_8, cute::Int<AiuContElemSize>>,
      cute::Stride<cute::Int<AiuContElemSize>, cute::_1>>;
  using SmemLayout = decltype(cute::tile_to_shape(
      SmemLayoutAtom{},
      cute::make_shape(cute::Int<BLK_MN>{}, cute::Int<BLK_K>{})));

  using SmemCopyOp = cute::PPU0010_TSM_LD_SWZL<
      CuteElem, BLK_MN, AiuContElemSize, /*Swap=*/false, /*Trans=*/false, InstNum>;
  using SmemCopyAtom = cute::Copy_Atom<SmemCopyOp, CuteElem>;

  CUTE_DEVICE static auto view(const Element* buf) {
    // Non-const CuteElem* on purpose: Copy_Traits<PPU0010_TSM_LD_SWZL>
    // copy_unpack_ takes a non-const void* src, so a const smem tensor's
    // iterator would fail to bind. The buffer is only read here regardless.
    return cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<CuteElem*>(const_cast<Element*>(buf))),
        SmemLayout{});
  }
};

// A from AIU-native swzl buffer, B from packed smem. Clears tC.
template <int M, int K, typename SmemB, typename TC, typename Element>
CUTE_DEVICE void fq_gemm_aiuA_ss_tc01(
    const Element* Aswz, SmemB const& sB, int tidx, TC& tC) {
  using Traits = FqAiuTileTc01<Element, M, K>;
  using cute::_;
  auto sA = Traits::view(Aswz);
  TiledMmaM32T<Element> mma;
  auto thr_m = mma.get_thread_slice(tidx);
  auto rA_full = thr_m.partition_fragment_A(sA);
  auto scpA = cute::make_tiled_copy_A(typename Traits::SmemCopyAtom{}, mma);
  auto tcpA = scpA.get_thread_slice(tidx);
  auto tAsA = tcpA.partition_S(cute::make_mix_tensor_like(sA));
  auto tArA = tcpA.retile_D(rA_full);
  auto rB_full = thr_m.partition_fragment_B(sB);
  auto scpB = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
  auto tcpB = scpB.get_thread_slice(tidx);
  auto tBsB = tcpB.partition_S(sB);
  auto tBrB = tcpB.retile_D(rB_full);
  cute::clear(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(scpA, tAsA(_,_,k), tArA(_,_,k));
    cute::copy(scpB, tBsB(_,_,k), tBrB(_,_,k));
    cute::gemm(mma, rA_full(_,_,k), rB_full(_,_,k), tC);
  }
}

// Q@h and K@h have the same B operand. Keep one h fragment load per K slice
// and consume it with Q and K in sequence; the two accumulators and MMA order
// remain identical to two fq_gemm_aiuA_ss_tc01 calls.
template <int M, int K, typename SmemB, typename TC0, typename TC1,
          typename Element>
CUTE_DEVICE void fq_gemm_qk_aiuA_ss_tc01(
    const Element* Qswz, const Element* Kswz, SmemB const& sB, int tidx,
    TC0& qh_acc, TC1& kh_acc) {
  using Traits = FqAiuTileTc01<Element, M, K>;
  using cute::_;
  auto sQ = Traits::view(Qswz);
  auto sK = Traits::view(Kswz);
  TiledMmaM32T<Element> mma;
  auto thr_m = mma.get_thread_slice(tidx);
  auto rA_full = thr_m.partition_fragment_A(sQ);
  auto scpA = cute::make_tiled_copy_A(typename Traits::SmemCopyAtom{}, mma);
  auto tcpA = scpA.get_thread_slice(tidx);
  auto tQsA = tcpA.partition_S(cute::make_mix_tensor_like(sQ));
  auto tKsA = tcpA.partition_S(cute::make_mix_tensor_like(sK));
  auto tArA = tcpA.retile_D(rA_full);
  auto rB_full = thr_m.partition_fragment_B(sB);
  auto scpB = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
  auto tcpB = scpB.get_thread_slice(tidx);
  auto tBsB = tcpB.partition_S(sB);
  auto tBrB = tcpB.retile_D(rB_full);
  cute::clear(qh_acc);
  cute::clear(kh_acc);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(scpB, tBsB(_,_,k), tBrB(_,_,k));
    cute::copy(scpA, tQsA(_,_,k), tArA(_,_,k));
    cute::gemm(mma, rA_full(_,_,k), rB_full(_,_,k), qh_acc);
    cute::copy(scpA, tKsA(_,_,k), tArA(_,_,k));
    cute::gemm(mma, rA_full(_,_,k), rB_full(_,_,k), kh_acc);
  }
}

// Both A and B from AIU-native swzl buffers (GEMM4: P = Q @ K^T). Clears tC.
template <int M, int N, int K, typename TC, typename Element>
CUTE_DEVICE void fq_gemm_aiuAB_tc01(
    const Element* Aswz, const Element* Bswz, int tidx, TC& tC) {
  using TrA = FqAiuTileTc01<Element, M, K>;
  using TrB = FqAiuTileTc01<Element, N, K>;
  using cute::_;
  auto sA = TrA::view(Aswz);
  auto sB = TrB::view(Bswz);
  TiledMmaM32T<Element> mma;
  auto thr_m = mma.get_thread_slice(tidx);
  auto rA_full = thr_m.partition_fragment_A(sA);
  auto scpA = cute::make_tiled_copy_A(typename TrA::SmemCopyAtom{}, mma);
  auto tcpA = scpA.get_thread_slice(tidx);
  auto tAsA = tcpA.partition_S(cute::make_mix_tensor_like(sA));
  auto tArA = tcpA.retile_D(rA_full);
  auto rB_full = thr_m.partition_fragment_B(sB);
  auto scpB = cute::make_tiled_copy_B(typename TrB::SmemCopyAtom{}, mma);
  auto tcpB = scpB.get_thread_slice(tidx);
  auto tBsB = tcpB.partition_S(cute::make_mix_tensor_like(sB));
  auto tBrB = tcpB.retile_D(rB_full);
  cute::clear(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(scpA, tAsA(_,_,k), tArA(_,_,k));
    cute::copy(scpB, tBsB(_,_,k), tBrB(_,_,k));
    cute::gemm(mma, rA_full(_,_,k), rB_full(_,_,k), tC);
  }
}
#endif  // FLASHQLA_ARCH_100

template <int M, int K, typename SmemB, typename TC, typename Element = __half,
          typename SmemCopyB = SmemCopyAtomT<Element>>
CUTE_DEVICE void flashqla_gemm_ss_swzlA_m32(
    const Element* Aswz, SmemB const& sB, int tidx, TC& tC,
    SmemCopyB smem_copy_b = {}) {
#if defined(FLASHQLA_ARCH_100)
  (void)smem_copy_b;
  fq_gemm_aiuA_ss_tc01<M, K, SmemB, TC, Element>(Aswz, sB, tidx, tC);
#else
  constexpr int CH = 32;             // cube height (rows) == FF_CS
  constexpr int CW = 32;             // cube width  (cols) -> swzl_mode=1
  constexpr int KT_PER_CUBE = CW / 16;   // = 2 MMA k-tiles per physical col-cube
  constexpr int INST = K / CW;           // number of col-cubes
  using TsmA = flashqla::PpuTsmLdSwzl<
      typename ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false, /*Trans=*/false, /*InstNum=*/INST>;

  TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;       // 2x2 layout: rows within the M-tile
  // (warp_n contributes to B side; A is warp_m-only.)

  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
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
      TsmA::copy(frag, reinterpret_cast<void*>(ff_swzl_read_base<Element>(Aswz)),
                 /*coord_h=*/col_in_cube, /*coord_w=*/row,
                 /*cube_in_stage=*/cube, /*stage=*/0);
      rA(0) = static_cast<uint32_t>(frag[0]);
      rA(1) = static_cast<uint32_t>(frag[1]);
      rA(2) = static_cast<uint32_t>(frag[2]);
      rA(3) = static_cast<uint32_t>(frag[3]);
    }
  }

  // Partition B: lazy load per K-slice with the caller-selected copy atom.
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(smem_copy_b, mma);
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
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
#endif  // FLASHQLA_ARCH_100
}

// flashqla_gemm_ss_swzlAB_m32: GEMM4 (P = Q @ K^T) variant where BOTH operands
// are read from AIU .swzl-written smem buffers. A (Q) is warp_m-tiled non-trans
// swzl, B (K) is warp_n-tiled non-trans swzl read with Swap=true, which swaps
// the atom's lbo/sbo so the fragment already lands in TN B-register order.
// Clears tC. Requires M,N<=32, K a multiple of 32.
template <int M, int N, int K, typename TC, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss_swzlAB_m32(
    const Element* Aswz, const Element* Bswz, int tidx, TC& tC) {
#if defined(FLASHQLA_ARCH_100)
  fq_gemm_aiuAB_tc01<M, N, K, TC, Element>(Aswz, Bswz, tidx, tC);
#else
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;   // = 2
  constexpr int INST = K / CW;
  using TsmA = flashqla::PpuTsmLdSwzl<
      typename ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false, /*Trans=*/false, /*InstNum=*/INST>;
  using TsmB = flashqla::PpuTsmLdSwzl<
      typename ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/true, /*Trans=*/false, /*InstNum=*/INST>;

  TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;
  const int warp_n = warp / 2;

  // A fragment (Q, warp_m-tiled).
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
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
      TsmA::copy(frag, reinterpret_cast<void*>(ff_swzl_read_base<Element>(Aswz)),
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
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
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
                 reinterpret_cast<void*>(ff_swzl_read_base<Element>(Bswz)),
                 /*coord_h=*/col_in_cube, /*coord_w=*/row,
                 /*cube_in_stage=*/cube, /*stage=*/0);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tC); ++m) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        // Swap=true already delivered the B operand register order.
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            bfrag[0], bfrag[1], bfrag[2], bfrag[3],
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
#endif  // FLASHQLA_ARCH_100
}

// flashqla_gemm_ss_swzlA_trans_acc_m32: GEMM6 (h += K^T @ gVd) variant.
// A operand K^T[M=DK, K=CS] is read DIRECTLY from the AIU .swzl-written K buffer
// (physical [CS=32, DK=128], 4 col-cubes of 32x32) via Trans=true TSM_LD_SWZL,
// eliminating the KT_buf scalar transpose. B (gVd) is plain-loaded from an
// [N=DVh, K=CS] smem view (matches the caller's transposed store convention in
// the non-WASP kernel -- gVdswz_g is written as (dv,cs) row-major, which is
// exactly the plain (N,K) B operand shape). Accumulates (no clear(tC)).
// M = DK (must be multiple of 32 -> M/32 col-cubes = INST), K = CS <= 32.
template <int M, int K, typename SmemB, typename TC, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss_swzlA_trans_acc_m32(
    const Element* Aswz, SmemB const& sB, int tidx, TC& tC) {
#if defined(FLASHQLA_ARCH_100)
  // tc01: read K^T DIRECTLY from the swizzled K buffer via Trans=true TSM_LD_SWZL
  // (validated byte-exact against a CPU reference). No KT_buf materialisation.
  // The swzl K buffer has CUBE_H=K(=CS) rows and CUBE_W-wide col-cubes
  // (AiuContElemSize for BLK_K=M=DK); dk row -> cube dk/CW at offset dk%CW,
  // contraction cs -> coord_h=k*16.
  using CE = typename ElementTraits<Element>::CuteType;
  constexpr int CH = K;                                       // cs rows in a cube
  constexpr int AiuByte = (M * (int)sizeof(CE) > 128) ? 128 : M * (int)sizeof(CE);
  constexpr int CW = AiuByte / (int)sizeof(CE);               // 64 for DK=128,b16
  constexpr int INST = M / CW;
  using TsmTrans = flashqla::PpuTsmLdSwzl<CE, CH, CW, /*Swap=*/false, /*Trans=*/true, /*InstNum=*/INST>;

  TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;

  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CE*>(0x10)), SmemLayout2D<M, K>{});
  Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rA_full); ++m) {
    const int m_global = warp_m * 16 + m * 32;   // dk (K^T row) 0..M-1
    const int cube = m_global / CW;
    const int dk_in_cube = m_global % CW;
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      const int cs_col = k * 16;
      int afrag[4] = {0, 0, 0, 0};
      TsmTrans::copy(afrag, reinterpret_cast<void*>(ff_swzl_read_base<Element>(Aswz)),
                     /*coord_h=*/cs_col, /*coord_w=*/dk_in_cube,
                     /*cube_in_stage=*/cube, /*stage=*/0);
      Tensor rA = recast<uint32_t>(rA_full(_, m, k));
      rA(0) = static_cast<uint32_t>(afrag[0]);
      rA(1) = static_cast<uint32_t>(afrag[1]);
      rA(2) = static_cast<uint32_t>(afrag[2]);
      rA(3) = static_cast<uint32_t>(afrag[3]);
    }
  }
  // B: plain lazy load per K-slice; accumulate into caller's tC (no clear).
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k)
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::gemm(mma, rA_full(_, _, k), rB_full(_, _, k), tC);
#else
  constexpr int CH = 32;   // cube height (rows = cs)
  constexpr int CW = 32;   // cube width  (cols = dk_local) -> swzl_mode=1
  using TsmTrans = flashqla::PpuTsmLdSwzl<
      typename ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false, /*Trans=*/true, /*InstNum=*/1>;

  TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;

  // A fragment shape (u32x4, M_tiles, K_tiles) via dummy tensor.
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
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
      TsmTrans::copy(afrag, reinterpret_cast<void*>(ff_swzl_read_base<Element>(Aswz)),
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
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
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
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
#endif  // FLASHQLA_ARCH_100
}

template <int M, int K, typename TC, typename Element = __half, int N = 64>
CUTE_DEVICE void flashqla_gemm_swzlA_trans_regB_acc_m32(
    const Element* Aswz, const uint32_t b_regs[2][2][4], int tidx, TC& tC) {
#if defined(FLASHQLA_ARCH_100)
  // tc01: read K^T DIRECTLY from the swizzled K buffer via Trans=true TSM_LD_SWZL
  // (validated). B lives in registers (b_regs). See swzlA_trans_acc_m32 for the
  // coord scheme.
  using CE = typename ElementTraits<Element>::CuteType;
  constexpr int CH = K;
  constexpr int AiuByte = (M * (int)sizeof(CE) > 128) ? 128 : M * (int)sizeof(CE);
  constexpr int CW = AiuByte / (int)sizeof(CE);
  constexpr int INST = M / CW;
  using TsmTrans = flashqla::PpuTsmLdSwzl<CE, CH, CW, /*Swap=*/false, /*Trans=*/true, /*InstNum=*/INST>;

  TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp   = tidx / 32;
  const int warp_m = warp % 2;

  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CE*>(0x10)), SmemLayout2D<M, K>{});
  Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rA_full); ++m) {
    const int m_global = warp_m * 16 + m * 32;
    const int cube = m_global / CW;
    const int dk_in_cube = m_global % CW;
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      const int cs_col = k * 16;
      int afrag[4] = {0, 0, 0, 0};
      TsmTrans::copy(afrag, reinterpret_cast<void*>(ff_swzl_read_base<Element>(Aswz)),
                     /*coord_h=*/cs_col, /*coord_w=*/dk_in_cube,
                     /*cube_in_stage=*/cube, /*stage=*/0);
      Tensor rA = recast<uint32_t>(rA_full(_, m, k));
      rA(0) = static_cast<uint32_t>(afrag[0]);
      rA(1) = static_cast<uint32_t>(afrag[1]);
      rA(2) = static_cast<uint32_t>(afrag[2]);
      rA(3) = static_cast<uint32_t>(afrag[3]);
    }
  }
  auto dummyB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CE*>(0x10)), SmemLayout2D<N, K>{});
  Tensor rB_full = thr_mma.partition_fragment_B(dummyB);
  Tensor rC_full = recast<float>(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            b_regs[k][n][0], b_regs[k][n][1], b_regs[k][n][2], b_regs[k][n][3],
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
#else
  constexpr int CH = 32;
  constexpr int CW = 32;
  using TsmTrans = flashqla::PpuTsmLdSwzl<
      typename ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false, /*Trans=*/true, /*InstNum=*/1>;

  TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp = tidx / 32;
  const int warp_m = warp % 2;

  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
      SmemLayout2D<M, K>{});
  Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rA_full); ++m) {
    const int m_global = warp_m * 16 + m * 32;
    const int cube = m_global / 32;
    const int dk_in_cube = m_global % 32;
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      const int cs_col = k * 16;
      int afrag[4] = {0, 0, 0, 0};
      TsmTrans::copy(afrag, reinterpret_cast<void*>(ff_swzl_read_base<Element>(Aswz)),
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

  auto dummyB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
      SmemLayout2D<64, K>{});
  Tensor rB_full = thr_mma.partition_fragment_B(dummyB);
  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            b_regs[k][n][0], b_regs[k][n][1], b_regs[k][n][2], b_regs[k][n][3],
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
#endif  // FLASHQLA_ARCH_100
}

template <int N, int K, typename SmemA, typename TC, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss_regB_acc_m32(
    SmemA const& sA, const uint32_t b_regs[2][2][4], int tidx, TC& tC) {
  TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  auto dummyB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
      SmemLayout2D<N, K>{});
  Tensor rB_full = thr_mma.partition_fragment_B(dummyB);
  Tensor rC_full = recast<float>(tC);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            b_regs[k][n][0], b_regs[k][n][1], b_regs[k][n][2], b_regs[k][n][3],
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
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
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
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  // Clear accumulator
  clear(tC);

#if defined(FLASHQLA_ARCH_100)
  // tc01: use cute::gemm (see flashqla_gemm_ss_acc comment for rationale).
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    cute::gemm(mma, rA_full(cute::_, cute::_, k), rB_full(cute::_, cute::_, k), tC);
  }
#else
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
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
#endif
}

// flashqla_gemm_ss_acc: identical to flashqla_gemm_ss (plain DefaultCopy for BOTH
// A and B, B provided as a plain (N,K) row-major smem tensor) but does NOT clear
// the accumulator — it accumulates tC += A@B. Used to bypass the swizzle +
// ldmatrix.trans B path: the caller pre-transposes the B operand into a plain
// [N,K] buffer, so no hardware transpose is needed. CS-agnostic and correct for
// any tile shape the plain (N,K) B path supports (same path as GEMM1/GEMM3).
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss_acc(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  // Partition B: plain lazy load per K-slice (B is plain (N,K) row-major)
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  // NOTE: NO clear(tC) — accumulate into the existing accumulator.
#if defined(FLASHQLA_ARCH_100)
  // tc01: use cute::gemm which correctly handles the tc01 fragment layout
  // (ALayout/CLayout differ from tc02; the manual ppu_mma_impl register ordering
  // produces a subtle ~3% error at warp_m=1 rows). cute::gemm dispatches through
  // mma_unpack(PPU_16x16x16_F32F16F16F32_TN_FIXED) which is validated byte-exact.
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rB_full); ++k) {
    cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    cute::gemm(mma, rA_full(cute::_, cute::_, k), rB_full(cute::_, cute::_, k), tC);
  }
#else
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
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
#endif
}

// flashqla_gemm_ss_ldsmB: same as flashqla_gemm_ss but the B operand is loaded
// with the hardware transposing ldmatrix (PPU_U16x8_LDSM_T). Caller must pass
// sB as a transposed layout VIEW over natural [K,N] smem (shape (N,K), stride
// (1, ldK)), so the .trans reads the physical 8x8 core matrices and delivers
// them already transposed into the tc02 B fragment (no elementwise transpose).
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss_ldsmB(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  // Partition B: transposing ldmatrix load per K-slice
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomBTransT<Element>{}, mma);
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
        ppu_mma_impl<Element>(
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
template <typename SmemA, typename SmemB, typename TC, typename Mma = TiledMma4Warp, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss_ldsmB_acc(
    SmemA const& sA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(tidx);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  // Partition B: transposing ldmatrix load per K-slice
  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomBTransT<Element>{}, mma);
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
        ppu_mma_impl<Element>(
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
template <typename TA, typename SmemB, typename TC, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_lazy_b(
    TA const& tA, SmemB const& sB,
    int tidx, TC& tC) {
  TiledMma4WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
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
        ppu_mma_impl<Element>(
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
template <typename TA, typename SmemB, typename TC, typename Mma = TiledMma4Warp, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_lazy_b_ldsmB(
    TA const& tA, SmemB const& sB,
    int tidx, TC& tC) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomBTransT<Element>{}, mma);
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
        ppu_mma_impl<Element>(
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
template <typename SmemA, typename RegB, typename TC, typename Element = __half>
CUTE_DEVICE void flashqla_gemm_ss_regB(
    SmemA const& sA, RegB const& rB_full,
    int tidx, TC& tC) {
  TiledMma4WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);

  // Partition A: load all from smem to registers (plain copy)
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
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
        ppu_mma_impl<Element>(
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

template <int M, int N, typename Mma = TiledMma4Warp, typename Element = __half>
CUTE_DEVICE auto make_acc(int tidx) {
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  // Use a dummy smem tensor just for shape — partition_fragment_C only
  // reads the layout, not the data pointer.
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
      SmemLayout2D<M, N>{});
  return thr_mma.partition_fragment_C(dummy);
}

// Allocate a register B-fragment for a (N, K) operand — used to receive an
// upstream GEMM's transposed output as the next GEMM's B operand (register
// hand-off, no smem). Layout matches partition_fragment_B over row-major (N,K).
template <typename Element = __half, int N, int K>
CUTE_DEVICE auto make_bfrag(int tidx) {
  TiledMma4WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
      SmemLayout2D<N, K>{});
  return thr_mma.partition_fragment_B(dummy);
}

// ============================================================================
// Utility: store fp32 accumulator back to fp16 smem
// ============================================================================

// Store MMA accumulator (fp32) to shared memory as fp16.
// tC shape: (atom=8, M_tiles, N_tiles)
// dst: flat __half* pointer to output buffer, row-major (M, N) with stride ldc
template <typename Element = __half, typename TC>
CUTE_DEVICE void store_acc_to_smem_fp16(
    TC const& tC, Element* dst, int M, int N, int ldc, int tidx) {
  // Use the MMA's thread-value mapping to determine which elements this thread owns.
  // For PPU 16x16x16 TN atom with AtomLayout<4,1,1>:
  //   Each thread owns 8 elements per (M_tile, N_tile) atom.
  //   Thread mapping within a 16x16 tile:
  //     row = (tid/4)*2 + (elem/4) within the 16-row tile
  //     col = (tid%4)*4 + (elem%4) ... actually this depends on the atom layout.
  //
  // Simpler approach: use CuTe's partition to figure out the mapping.
  TiledMma4WarpT<Element> mma;
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
        tCsC(v, m, n) = typename ElementTraits<Element>::CuteType(rC(v, m, n));
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

// ============================================================================
// Utility: make CuTe smem tensor from flat pointer + dimensions
// ============================================================================

// Create a row-major (M, K) CuTe tensor view over a flat smem pointer
template <int M, int K, typename Element>
CUTE_DEVICE auto make_smem_tensor_h(Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(ptr)),
      SmemLayout2D<M, K>{});
}

template <int M, int K, typename Element>
CUTE_DEVICE auto make_smem_tensor_h(const Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<const typename ElementTraits<Element>::CuteType*>(ptr)),
      SmemLayout2D<M, K>{});
}

// Row-major (M, K) view with an explicitly padded leading dimension.  The
// logical tensor shape remains (M, K); only the physical distance between
// rows changes.  This is useful for long-lived state snapshots whose K=128
// rows otherwise begin on the same TSM bank set.
template <int M, int K, int LD, typename Element>
CUTE_DEVICE auto make_smem_tensor_h_ld(Element* ptr) {
  static_assert(LD >= K, "shared-memory leading dimension must cover K");
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(ptr)),
      cute::make_layout(
          cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
          cute::make_stride(cute::Int<LD>{}, cute::_1{})));
}

template <int M, int K, int LD, typename Element>
CUTE_DEVICE auto make_smem_tensor_h_ld(const Element* ptr) {
  static_assert(LD >= K, "shared-memory leading dimension must cover K");
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<const typename ElementTraits<Element>::CuteType*>(ptr)),
      cute::make_layout(
          cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
          cute::make_stride(cute::Int<LD>{}, cute::_1{})));
}

// Swizzled (Rows=K, Cols=N) WRITE-view over a flat smem buffer. Write by (k, n)
// coordinate; the swizzle relocates banks so a subsequent ldmatrix.trans read is
// bank-conflict-free. Pair with make_smem_swzB_T for the transposed read view.
template <int Rows, int Cols, typename Element>
CUTE_DEVICE auto make_smem_swzB(Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(ptr)),
      SmemLayoutSwzB<Rows, Cols>{});
}

// Transposed (Cols=N, Rows=K) READ-view over the same swizzled buffer: the mma
// B operand (N, K), to be loaded with the transposing ldmatrix copy atom.
template <int Rows, int Cols, typename Element>
CUTE_DEVICE auto make_smem_swzB_T(const Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<const typename ElementTraits<Element>::CuteType*>(ptr)),
      SmemLayoutSwzB_T<Rows, Cols>{});
}

// Swizzled A-operand VIEW for the non-WASP Ag buffer (Rows = FF_CS = 32,
// Cols = FF_CS = 32). Both the int4 vec-store writer (WG-cooperative gating)
// and the ldmatrix A-read (GEMM2) index this view by (row, col); the
// Swizzle<2,3,3> layout composes the physical offset. See SmemLayoutSwzA_cs32.
template <int Rows, int Cols, typename Element>
CUTE_DEVICE auto make_smem_swzA_cs32(Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(ptr)),
      SmemLayoutSwzA_cs32<Rows, Cols>{});
}

template <int Rows, int Cols, typename Element>
CUTE_DEVICE auto make_smem_swzA_cs32(const Element* ptr) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<const typename ElementTraits<Element>::CuteType*>(ptr)),
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

template <typename Element = __half>
using TiledMma1WarpT = decltype(cute::make_tiled_mma(
    MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));
using TiledMma1Warp = TiledMma1WarpT<__half>;

CUTE_DEVICE uint32_t pack_b16x2_u32(uint32_t lo, uint32_t hi) {
  return (lo & 0xffffu) | ((hi & 0xffffu) << 16);
}

CUTE_DEVICE uint32_t get_b16_from_pack(uint32_t packed, int half_idx) {
  return half_idx ? (packed >> 16) : (packed & 0xffffu);
}

template <typename Element>
CUTE_DEVICE uint32_t element_bits16(Element x) {
  union Bits { Element e; unsigned short u; } bits;
  bits.e = x;
  return static_cast<uint32_t>(bits.u);
}

template <typename Element>
CUTE_DEVICE Element element_from_bits16(uint32_t x) {
  union Bits { Element e; unsigned short u; } bits;
  bits.u = static_cast<unsigned short>(x);
  return bits.e;
}

CUTE_DEVICE void cfrag_to_bfrag_swizzle_u32(
    const uint32_t c_regs[4], uint32_t b_regs[4], int lane) {
#if defined(FLASHQLA_ARCH_100)
  // tc01's PPU0010 C/B lane layouts differ from tc02's PPU0015
  // layouts.  These source coordinates are the closed-form result of matching
  // partition_C(identity) against transposed partition_B(identity) for the
  // tc01 16x16 atom.  Eight direct shuffles keep the register path cheap.
  uint32_t b_half[8];
  CUTE_UNROLL
  for (int dst_slot = 0; dst_slot < 8; ++dst_slot) {
    int src_lane = ((lane & 3) << 3) + ((dst_slot & 1) << 2)
                 + ((lane >> 2) & 3);
    int src_slot = ((dst_slot & 2) << 1) + ((dst_slot & 4) >> 1)
                 + (lane >> 4);
    uint32_t src_pack = __shfl_sync(0xffffffff, c_regs[src_slot >> 1], src_lane);
    b_half[dst_slot] = get_b16_from_pack(src_pack, src_slot & 1);
  }
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    b_regs[i] = pack_b16x2_u32(b_half[2 * i], b_half[2 * i + 1]);
  }
#else
  // Original tc02 PPU0015 mapping.
  uint32_t b_half[8];
  CUTE_UNROLL
  for (int dst_slot = 0; dst_slot < 8; ++dst_slot) {
    int src_lane = ((lane & 3) << 3) + ((dst_slot & 1) << 2) + (lane >> 3);
    int src_slot = (dst_slot & ~1) + ((lane >> 2) & 1);
    uint32_t src_pack = __shfl_sync(0xffffffff, c_regs[src_slot >> 1], src_lane);
    b_half[dst_slot] = get_b16_from_pack(src_pack, src_slot & 1);
  }
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    b_regs[i] = pack_b16x2_u32(b_half[2 * i], b_half[2 * i + 1]);
  }
#endif
}

// Convert one 16x16 accumulator tile into the A-fragment layout for the same
// (row, column) coordinates.  PPU0015 has identical A/C layouts, whereas
// PPU0010 uses
//
//   C: ((4,8),(4,2)):((16,1),(64,8))
//   A: ((4,8),(2,2,2)):((32,1),(16,128,8)).
//
// The closed form below is partition_C(identity) -> partition_A(identity).
// It is the non-transposing companion to cfrag_to_bfrag_swizzle_u32 and lets
// tc01 keep accumulator-produced state/intermediates in a register chain.
CUTE_DEVICE void cfrag_to_afrag_u32(
    const uint32_t c_regs[4], uint32_t a_regs[4], int lane) {
#if defined(FLASHQLA_ARCH_100)
  uint32_t a_half[8];
  CUTE_UNROLL
  for (int dst_slot = 0; dst_slot < 8; ++dst_slot) {
    const int a_v0 = dst_slot & 1;
    const int a_v1 = (dst_slot >> 1) & 1;
    const int a_v2 = dst_slot >> 2;
    const int col = 2 * (lane & 3) + a_v0 + 8 * a_v1;
    const int src_lane = (col & 3) + 4 * (lane >> 2);
    const int src_slot = (col >> 2) + 4 * a_v2;
    const uint32_t src_pack =
        __shfl_sync(0xffffffff, c_regs[src_slot >> 1], src_lane);
    a_half[dst_slot] = get_b16_from_pack(src_pack, src_slot & 1);
  }
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    a_regs[i] = pack_b16x2_u32(a_half[2 * i], a_half[2 * i + 1]);
#else
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) a_regs[i] = c_regs[i];
#endif
}

// MMA-remap companion to cfrag_to_afrag_u32 (tc01, bf16 and fp16). The packed
// data stays on the A operand, a per-lane permutation fragment sits on B,
// and one 16-bit-output MMA (zero accumulator) performs the register remap.
// On tc01 the 16-bit D register layout matches the A-operand layout for both
// instructions (device-measured; the same hardware fact the FlashMLA and
// flash-attention tc01 converters rely on). The C and A layouts assign
// different columns to the same physical slot — n_C=(l&3)+4*(s&3) vs
// n_A=2*(l&3)+(s&1)+8*((s>>1)&1) — so B carries the composed permutation
// W[k,n] = (h(k)==n) with h(k) = ((k>>1)&3) + 4*(k&1) + 8*(k>>3), not the
// identity. This is arithmetic, not a bitwise shuffle: a NaN/Inf element
// pollutes its whole 16-wide row through the zero-weight products and -0.0
// may lose its sign, so callers must guarantee finite inputs (the same
// upstream contract the C-to-B raw candidate uses). The cutlass F16-output
// atom wrapper is an assert(0) shell on tc01, but the underlying
// f16.f16.f16.f16 instruction assembles and is probe-verified bit-exact
// against the shuffle reference over all 63488 finite fp16 patterns
// (profile/kda-c2a-mma-f16-v1-20260923/harness/probe_c2a_mma_f16.cu).
template <typename Element>
CUTE_DEVICE void cfrag_to_afrag_mma_u32(
    const uint32_t c_regs[4], uint32_t a_regs[4], int lane) {
#if defined(FLASHQLA_ARCH_100)
  static_assert(std::is_same_v<Element, __ppu_bfloat16> ||
                std::is_same_v<Element, __half>);
  constexpr uint32_t kOneLo =
      std::is_same_v<Element, __ppu_bfloat16> ? 0x3f80u : 0x3c00u;
  constexpr uint32_t kOneHi =
      std::is_same_v<Element, __ppu_bfloat16> ? 0x3f800000u : 0x3c000000u;
  uint32_t w[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    const int n = (lane >> 2) + 8 * (i >> 1);
    const int k = 2 * (lane & 3) + 8 * (i & 1);
    const int h0 = ((k >> 1) & 3) + 4 * (k & 1) + 8 * (k >> 3);
    const int h1 = (((k + 1) >> 1) & 3) + 4 * ((k + 1) & 1) + 8 * ((k + 1) >> 3);
    w[i] = (h0 == n ? kOneLo : 0u) | (h1 == n ? kOneHi : 0u);
  }
  float out[4];
  const float zero = __uint_as_float(0u);
  if constexpr (std::is_same_v<Element, __ppu_bfloat16>) {
    asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%12,%12,%12};\n"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3])
        : "f"(__uint_as_float(c_regs[0])), "f"(__uint_as_float(c_regs[1])),
          "f"(__uint_as_float(c_regs[2])), "f"(__uint_as_float(c_regs[3])),
          "f"(__uint_as_float(w[0])), "f"(__uint_as_float(w[1])),
          "f"(__uint_as_float(w[2])), "f"(__uint_as_float(w[3])), "f"(zero));
  } else {
    asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%12,%12,%12};\n"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3])
        : "f"(__uint_as_float(c_regs[0])), "f"(__uint_as_float(c_regs[1])),
          "f"(__uint_as_float(c_regs[2])), "f"(__uint_as_float(c_regs[3])),
          "f"(__uint_as_float(w[0])), "f"(__uint_as_float(w[1])),
          "f"(__uint_as_float(w[2])), "f"(__uint_as_float(w[3])), "f"(zero));
  }
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) a_regs[i] = __float_as_uint(out[i]);
  return;
#else
  cfrag_to_afrag_u32(c_regs, a_regs, lane);
#endif
}

template <typename Element = __half, typename TC>
CUTE_DEVICE void acc16_to_bfrag_swizzle_u32(TC const& tC, uint32_t b_regs[4], int lane) {
  Tensor rC = recast<float>(tC);
  uint32_t c_half[8];
  int pos = 0;
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rC); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(rC); ++n) {
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(rC); ++v) {
        c_half[pos] = element_bits16<Element>(float_to_element<Element>(rC(v, m, n)));
        ++pos;
      }
    }
  }
  uint32_t c_regs[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    c_regs[i] = pack_b16x2_u32(c_half[2 * i], c_half[2 * i + 1]);
  }
  cfrag_to_bfrag_swizzle_u32(c_regs, b_regs, lane);
}

CUTE_DEVICE uint32_t ldmatrix_x1_trans_u32(uint32_t smem_addr) {
#if defined(FLASHQLA_ARCH_150)
  uint32_t dst;
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
      : "=r"(dst)
      : "r"(smem_addr));
  return dst;
#elif defined(FLASHQLA_ARCH_100)
  // tc01 exposes the same m8n8.x1.trans.shared.b16 primitive (TIX section
  // 9.3.9), only under the tc01 prefix.
  uint32_t dst;
  asm volatile(
      "ppu.tc01.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
      : "=r"(dst)
      : "r"(smem_addr));
  return dst;
#else
  (void)smem_addr;
  return 0u;
#endif
}

CUTE_DEVICE uint32_t movmatrix_u32(uint32_t src, uint32_t* scratch, int lane) {
  scratch[lane] = src;
  __syncwarp();
  const int row = lane & 7;
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(scratch + row * 4));
  uint32_t dst = ldmatrix_x1_trans_u32(smem_addr);
  __syncwarp();
  return dst;
}

template <typename Element = __half, typename TC>
CUTE_DEVICE void acc16_to_bfrag_movmatrix_u32(
    TC const& tC, uint32_t b_regs[4], int lane, uint32_t* scratch) {
  Tensor rC = recast<float>(tC);
  uint32_t c_half[8];
  int pos = 0;
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rC); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(rC); ++n) {
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(rC); ++v) {
        c_half[pos] = element_bits16<Element>(float_to_element<Element>(rC(v, m, n)));
        ++pos;
      }
    }
  }
  uint32_t c_regs[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    c_regs[i] = pack_b16x2_u32(c_half[2 * i], c_half[2 * i + 1]);
  }
#if defined(FLASHQLA_ARCH_100)
  // ldmatrix.trans alone assumes tc02's accumulator lane distribution.
  (void)scratch;
  cfrag_to_bfrag_swizzle_u32(c_regs, b_regs, lane);
#else
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    b_regs[i] = movmatrix_u32(c_regs[i], scratch, lane);
  }
#endif
}

// Allocate a 16x16 fp32 register accumulator for single-warp MMA. lane in [0,32).
template <typename Element = __half>
CUTE_DEVICE auto make_acc16_1warp(int lane) {
  TiledMma1WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(0x10)),
      SmemLayout2D<16, 16>{});
  return thr_mma.partition_fragment_C(dummy);
}

template <typename Element = __half, typename TC>
CUTE_DEVICE void scale_acc16_rows(TC& tC, const float* row_scale, int row_base, int lane) {
  TiledMma1WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto cId = cute::make_identity_tensor(cute::make_shape(cute::Int<16>{}, cute::Int<16>{}));
  auto tc = thr_mma.partition_C(cId);
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tc); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tc); ++n) {
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tc); ++v) {
        auto crd = tc(v, m, n);
        int row = cute::get<0>(crd);
        rC(v, m, n) *= row_scale[row_base + row];
      }
    }
  }
}

// Make a 16x16 fp16 block view of a row-major buffer with leading dim `LD`
// (compile-time). Block (bi,bj) starts at base + bi*16*LD + bj*16. The layout
// must be fully static so CuTe's MMA partition_fragment works.
template <int LD, typename Element>
CUTE_DEVICE auto block16_view(Element* base, int bi, int bj) {
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(base + bi * 16 * LD + bj * 16)),
      cute::make_layout(
          cute::make_shape(cute::Int<16>{}, cute::Int<16>{}),
          cute::make_stride(cute::Int<LD>{}, cute::Int<1>{})));
}

template <typename SmemA, typename TC, typename Element = __half>
CUTE_DEVICE void gemm16_sr_1warp_acc(
    SmemA const& sA, const uint32_t b_regs[4], int lane, TC& tC) {
  TiledMma1WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(lane);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  }

  Tensor rC_full = recast<float>(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(rA_full); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(rC_full); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        ppu_mma_impl<Element>(
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            b_regs[0], b_regs[1], b_regs[2], b_regs[3],
            rC_full(0,m,n), rC_full(1,m,n), rC_full(2,m,n), rC_full(3,m,n),
            rC_full(4,m,n), rC_full(5,m,n), rC_full(6,m,n), rC_full(7,m,n));
      }
    }
  }
}

// Single-warp GEMM (accumulating): tC += A[16,16] . B[16,16]^T  (TN). No clear.
//   sA: (16,16) view;  sB: (16,16) view;  lane in [0,32)
// For NN (tC += A.B) pass sB as transpose_tensor(B).
template <typename SmemA, typename SmemB, typename TC, typename Element = __half>
CUTE_DEVICE void gemm16_ss_1warp_acc(
    SmemA const& sA, SmemB const& sB, int lane, TC& tC) {
  TiledMma1WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(lane);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(SmemCopyAtomT<Element>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(SmemCopyAtomT<Element>{}, mma);
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
        ppu_mma_impl<Element>(
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
template <typename SmemA, typename SmemB, typename TC, typename Element = __half>
CUTE_DEVICE void gemm16_ss_1warp(
    SmemA const& sA, SmemB const& sB, int lane, TC& tC) {
  clear(tC);
  gemm16_ss_1warp_acc(sA, sB, lane, tC);
}

// Store a 16x16 fp32 accumulator to fp16 smem (row-major, leading dim ldc).
template <typename Element = __half, typename TC>
CUTE_DEVICE void store_acc16_fp16_1warp(
    TC const& tC, Element* dst, int ldc, int lane) {
  TiledMma1WarpT<Element> mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename ElementTraits<Element>::CuteType*>(dst)),
      cute::make_layout(cute::make_shape(16, 16), cute::make_stride(ldc, 1)));
  Tensor tCsC = thr_mma.partition_C(sOut);
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v)
        tCsC(v, m, n) = typename ElementTraits<Element>::CuteType(rC(v, m, n));
}

}  // namespace flashqla
