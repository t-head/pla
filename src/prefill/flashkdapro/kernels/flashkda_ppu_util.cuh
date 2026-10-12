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
 * FlashKDA PPU1.5 — numeric helpers + bf16->f32 accumulate MMA atom.
 *
 * Ports the blueprint's csrc/smxx/utils.cuh transcendental helpers to PPU1.5
 * intrinsics (TIX 5.11/5.12/6.x):
 *   ex2.approx.ftz.f32   -> ppu.ex2.approx.ftz.f32
 *   sgmdf sigmoid       -> __ppu_sgmdf
 *   cvt.f32.bf16        -> ppu.cvt.f32.bf16
 * These match the fp32 approximations the torch reference (tests/torch_ref.py)
 * is defined against (2-ulp), so gate/exp math aligns within the E2E tolerance.
 *
 * Also adds the bf16->f32 accumulate 16x16x16 MMA FIXED atom + single-warp
 * helpers, used for L = k_decayed @ k_inv^T and Mqk = q_decayed @ k_inv^T
 * (blueprint uses SM80_16x8x16_F32BF16BF16F32_TN, bf16 in / f32 acc).
 */
#pragma once

// This must precede every CuTe/cutlass header so the selected trait family and
// inline-asm register constraints agree with the physical PPU generation.
#include "flashkda_arch.cuh"
#include "prefill/flashkdapro/common/workspace_abi.h"

namespace flashkda {

using namespace cute;
using BF16 = cutlass::bfloat16_t;

// Type tag for dispatching a template kernel on a runtime-selected element type
// (e.g. the recurrent-state dtype: bf16 vs fp32). Lets a launcher pick StateT at
// runtime while keeping the kernel a compile-time template.
template <typename T>
struct TypeTag { using type = T; };

// The shared K1<->K2 slot byte image and CP segment descriptor live in
// common/workspace_abi.h.  Keep architecture-specific copy/load helpers here,
// but never duplicate host-visible offsets in a kernel-only header.

// kr AIU swzl permutation (probe-verified, bit-exact). For merged smem/global
// linear slot j into the CUBE_H=16/CUBE_W=32 swizzled kr image, the source
// row-major (row r, col c) of the pre-swizzle kr tile is:
//   cube = j>>9; r = (j&511)>>5; k = j&31; c = cube*32 + (k ^ (((r>>1)&3)<<3))
// so o_kr[j] = kr[r*128 + c]. Helper returns the source row-major index.
__forceinline__ __host__ __device__ int ws_kr_swzl_src_index(int j) {
  int cube = j >> 9;           // j / 512
  int local = j & 511;         // j % 512
  int r = local >> 5;          // source row 0..15
  int k = local & 31;          // 0..31
  int shift = ((r >> 1) & 3) << 3;  // {0,8,16,24}
  int c = cube * 32 + (k ^ shift);
  return r * 128 + c;
}

// ---- transcendental helpers (PPU intrinsics, match torch_ref semantics) ----
__forceinline__ __device__ float ex2_approx_ftz_f32(float x) {
#if defined(__HGGC_ARCH__)
  float r;
  asm("ppu.ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
#else
  return exp2f(x);
#endif
}

__forceinline__ __device__ float tanh_approx_f32(float x) {
#if defined(__HGGC_ARCH__)
  float r;
  asm("ppu.tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
#else
  return tanhf(x);
#endif
}

// PPU has a native sigmoid intrinsic; non-PPU fallback keeps the tanh identity.
__forceinline__ __device__ float sigmoid_tanh_approx_f32(float x) {
#if defined(__HGGC_ARCH__)
  return __ppu_sgmdf(x);
#else
  float th = tanh_approx_f32(x * 0.5f);
  return th * 0.5f + 0.5f;
#endif
}

__forceinline__ __device__ float bf16_to_f32(BF16 x) {
#if defined(__HGGC_ARCH__)
  float r;
  asm("ppu.cvt.f32.bf16 %0, %1;\n" : "=f"(r) : "h"(x.storage));
  return r;
#else
  return float(x);
#endif
}

}  // namespace flashkda

namespace cute {

// bf16->f32 accumulate 16x16x16 MMA (D = A @ B^T + C, TN). D/C 8x f32,
// A/B 4x bf16x2 (uint32). tc02 represents f32 accumulators in the general
// register class; tc01 uses the float register class.
struct PPU_16x16x16_F32BF16BF16F32_TN_FIXED {
  using DRegisters = float[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[8];

  __forceinline__ __device__ static void
  fma(float& d0, float& d1, float& d2, float& d3,
      float& d4, float& d5, float& d6, float& d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      float const& c0, float const& c1, float const& c2, float const& c3,
      float const& c4, float const& c5, float const& c6, float const& c7) {
#if defined(FLASHKDA_ARCH_150)
    asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
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
        "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, {%12, %13, %14, %15}, "
        "{%16, %17, %18, %19, %20, %21, %22, %23};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "f"(c4), "f"(c5), "f"(c6), "f"(c7));
#else
    (void)d0;(void)d1;(void)d2;(void)d3;(void)d4;(void)d5;(void)d6;(void)d7;
    (void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;(void)b2;(void)b3;
    (void)c0;(void)c1;(void)c2;(void)c3;(void)c4;(void)c5;(void)c6;(void)c7;
#endif
  }
};

template <>
struct MMA_Traits<PPU_16x16x16_F32BF16BF16F32_TN_FIXED>
    : MMA_Traits<flashkda::PpuBf16MMA> {};

template <class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<PPU_16x16x16_F32BF16BF16F32_TN_FIXED> const&,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  Tensor rA = recast<uint32_t>(A);
  Tensor rB = recast<uint32_t>(B);
  Tensor rD = recast<float>(D);
  Tensor rC = recast<float>(C);
  CUTE_STATIC_ASSERT_V(size(rA) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<8>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<8>{});
  PPU_16x16x16_F32BF16BF16F32_TN_FIXED::fma(
      rD(0), rD(1), rD(2), rD(3), rD(4), rD(5), rD(6), rD(7),
      rA(0), rA(1), rA(2), rA(3),
      rB(0), rB(1), rB(2), rB(3),
      rC(0), rC(1), rC(2), rC(3), rC(4), rC(5), rC(6), rC(7));
}

// Native fp16-input, fp32-accumulate counterpart used by tc02 CP when K1
// materializes the private Kd workspace field as fp16. Keep a KDA-specific
// type name because kernel3 also includes FlashQLA's fixed atom definition.
struct PPU_16x16x16_F32F16F16F32_TN_FIXED_KDA {
  using DRegisters = float[8];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[8];

  __forceinline__ __device__ static void
  fma(float& d0, float& d1, float& d2, float& d3,
      float& d4, float& d5, float& d6, float& d7,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
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
#else
    (void)d0;(void)d1;(void)d2;(void)d3;(void)d4;(void)d5;(void)d6;(void)d7;
    (void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;(void)b2;(void)b3;
    (void)c0;(void)c1;(void)c2;(void)c3;(void)c4;(void)c5;(void)c6;(void)c7;
#endif
  }
};

template <>
struct MMA_Traits<PPU_16x16x16_F32F16F16F32_TN_FIXED_KDA>
    : MMA_Traits<flashkda::PpuF16MMA> {};

template <class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr void
mma_unpack(MMA_Traits<PPU_16x16x16_F32F16F16F32_TN_FIXED_KDA> const&,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  Tensor rA = recast<uint32_t>(A);
  Tensor rB = recast<uint32_t>(B);
  Tensor rD = recast<float>(D);
  Tensor rC = recast<float>(C);
  CUTE_STATIC_ASSERT_V(size(rA) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<4>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<8>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<8>{});
  PPU_16x16x16_F32F16F16F32_TN_FIXED_KDA::fma(
      rD(0), rD(1), rD(2), rD(3), rD(4), rD(5), rD(6), rD(7),
      rA(0), rA(1), rA(2), rA(3),
      rB(0), rB(1), rB(2), rB(3),
      rC(0), rC(1), rC(2), rC(3), rC(4), rC(5), rC(6), rC(7));
}

}  // namespace cute

namespace flashkda {

using namespace cute;

// Single-warp 16x16x16 bf16->f32 TiledMMA (one atom).
using TiledMma1WarpBf16 = decltype(cute::make_tiled_mma(
    cute::MMA_Atom<cute::PPU_16x16x16_F32BF16BF16F32_TN_FIXED>{},
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));
using TiledMma1WarpF16 = decltype(cute::make_tiled_mma(
    cute::MMA_Atom<cute::PPU_16x16x16_F32F16F16F32_TN_FIXED_KDA>{},
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>{}));

// Allocate a 16x16 f32 C accumulator for single-warp bf16 MMA.
CUTE_DEVICE auto make_acc16_bf16_1warp(int lane) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<BF16*>(0x10)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::_16{}, cute::_1{})));
  return thr_mma.partition_fragment_C(dummy);
}

CUTE_DEVICE auto make_acc16_f16_1warp(int lane) {
  TiledMma1WarpF16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto dummy = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cute::half_t*>(0x10)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::_16{}, cute::_1{})));
  return thr_mma.partition_fragment_C(dummy);
}

CUTE_DEVICE uint32_t pack_half2_u32(uint32_t lo, uint32_t hi) {
  return (lo & 0xffffu) | (hi << 16);
}

CUTE_DEVICE uint32_t get_half_from_pack(uint32_t packed, int half_idx) {
  return half_idx ? (packed >> 16) : (packed & 0xffffu);
}

CUTE_DEVICE void bf16_cfrag_to_bfrag_swizzle_u32(
    const uint32_t c_regs[4], uint32_t b_regs[4], int lane) {
  uint32_t b_half[8];
  CUTE_UNROLL
  for (int dst_slot = 0; dst_slot < 8; ++dst_slot) {
    int src_lane = ((lane & 3) << 3) + ((dst_slot & 1) << 2) + (lane >> 3);
    int src_slot = (dst_slot & ~1) + ((lane >> 2) & 1);
    uint32_t src_pack = __shfl_sync(0xffffffff, c_regs[src_slot >> 1], src_lane);
    b_half[dst_slot] = get_half_from_pack(src_pack, src_slot & 1);
  }
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    b_regs[i] = pack_half2_u32(b_half[2 * i], b_half[2 * i + 1]);
}

CUTE_DEVICE uint32_t ldmatrix_x1_trans_u32(uint32_t smem_addr) {
#if defined(FLASHKDA_ARCH_150)
  uint32_t dst;
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
      : "=r"(dst)
      : "r"(smem_addr));
  return dst;
#elif defined(FLASHKDA_ARCH_100)
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

#if defined(FLASHKDA_ARCH_100)
// Derive tc01's C-fragment -> transposed B-fragment mapping from the selected
// MMA layouts. tc02's movmatrix mapping is not bit-compatible with the
// PPU0010 fragment lane distribution.
CUTE_DEVICE void bf16_cfrag_to_bfrag_layout_u32(
    const uint32_t c_regs[4], uint32_t b_regs[4], int lane) {
  // The PPU0010 atom has fixed layouts:
  //
  //   C: ((4,8),(4,2)):((16,1),(64,8))
  //   B: ((4,8),(2,2,2)):((32,1),(16,128,8))
  //
  // Solving those layouts once gives the transposed-B source directly.  For
  // destination lane `lane`, both halves of one B register come from lanes
  // `src_lane` and `src_lane + 4`; destination lanes 16--31 select the high
  // bf16 half of the same source register.  The B-register order maps to C
  // registers {0,2,1,3}.  This replaces the old runtime search over all
  // 32 lanes and all 8 half-registers (512 shuffles plus 2048 comparisons per
  // conversion) with exactly two shuffles per output register.
  const int src_lane = ((lane & 3) << 3) | ((lane >> 2) & 3);
  const int half_shift = (lane & 16) ? 16 : 0;
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    const int src_reg = ((i & 1) << 1) | (i >> 1);  // 0, 2, 1, 3
    const uint32_t lo_pair =
        __shfl_sync(0xffffffff, c_regs[src_reg], src_lane);
    const uint32_t hi_pair =
        __shfl_sync(0xffffffff, c_regs[src_reg], src_lane + 4);
    const uint32_t lo = (lo_pair >> half_shift) & 0xffffu;
    const uint32_t hi = (hi_pair >> half_shift) & 0xffffu;
    b_regs[i] = lo | (hi << 16);
  }
}
#endif

CUTE_DEVICE void bf16_cfrag_to_bfrag_movmatrix_u32(
    const uint32_t c_regs[4], uint32_t b_regs[4], int lane, uint32_t* scratch) {
#if defined(FLASHKDA_ARCH_150)
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    b_regs[i] = movmatrix_u32(c_regs[i], scratch, lane);
#else
  (void)scratch;
  bf16_cfrag_to_bfrag_layout_u32(c_regs, b_regs, lane);
#endif
}

// Same mapping as four independent movmatrix_u32 calls, but reserve one
// 32-lane scratch plane per register.  Publishing all four planes together
// reduces the conversion from 8 warp synchronizations to 2 without changing
// any ldmatrix address or fragment bit pattern.
CUTE_DEVICE void bf16_cfrag_to_bfrag_movmatrix_batched_u32(
    const uint32_t c_regs[4], uint32_t b_regs[4], int lane, uint32_t* scratch) {
#if defined(FLASHKDA_ARCH_150)
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) scratch[i * 32 + lane] = c_regs[i];
  __syncwarp();
  const int row = lane & 7;
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(scratch + i * 32 + row * 4));
    b_regs[i] = ldmatrix_x1_trans_u32(smem_addr);
  }
  __syncwarp();
#else
  (void)scratch;
  bf16_cfrag_to_bfrag_layout_u32(c_regs, b_regs, lane);
#endif
}

template <typename TC>
CUTE_DEVICE void bf16_acc_to_bfrag_swizzle_u32(TC const& tC,
                                               uint32_t b_regs[4],
                                               int lane) {
  Tensor rC = recast<float>(tC);
  uint32_t c_half[8];
  int pos = 0;
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(rC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(rC); ++v) {
        c_half[pos] = BF16(rC(v, m, n)).storage;
        ++pos;
      }
  uint32_t c_regs[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    c_regs[i] = pack_half2_u32(c_half[2 * i], c_half[2 * i + 1]);
  bf16_cfrag_to_bfrag_swizzle_u32(c_regs, b_regs, lane);
}

template <typename TC>
CUTE_DEVICE void bf16_acc_to_bfrag_movmatrix_u32(TC const& tC,
                                                 uint32_t b_regs[4],
                                                 int lane,
                                                 uint32_t* scratch) {
  Tensor rC = recast<float>(tC);
  uint32_t c_half[8];
  int pos = 0;
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(rC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(rC); ++v) {
        c_half[pos] = BF16(rC(v, m, n)).storage;
        ++pos;
      }
  uint32_t c_regs[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    c_regs[i] = pack_half2_u32(c_half[2 * i], c_half[2 * i + 1]);
  bf16_cfrag_to_bfrag_movmatrix_u32(c_regs, b_regs, lane, scratch);
}

template <typename TC>
CUTE_DEVICE void bf16_acc_to_bfrag_movmatrix_batched_u32(
    TC const& tC, uint32_t b_regs[4], int lane, uint32_t* scratch) {
  Tensor rC = recast<float>(tC);
  uint32_t c_half[8];
  int pos = 0;
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(rC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(rC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(rC); ++v) {
        c_half[pos] = BF16(rC(v, m, n)).storage;
        ++pos;
      }
  uint32_t c_regs[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    c_regs[i] = pack_half2_u32(c_half[2 * i], c_half[2 * i + 1]);
  bf16_cfrag_to_bfrag_movmatrix_batched_u32(
      c_regs, b_regs, lane, scratch);
}

// C[16,16] = A[16,16] @ B[16,16]^T (TN), bf16 in, f32 accumulate. Clears tC.
// sA,sB: (16,16) row-major bf16 smem views. lane in [0,32). For C = A@B (NN),
// pass sB as a transposed (K,N) view.
template <typename SmemA, typename SmemB, typename TC>
CUTE_DEVICE void gemm16_ss_1warp_bf16(
    SmemA const& sA, SmemB const& sB, int lane, TC& tC) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(lane);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  clear(tC);
  Tensor rC = recast<float>(tC);
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
        cute::PPU_16x16x16_F32BF16BF16F32_TN_FIXED::fma(
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n));
      }
    }
  }
}

// Accumulating variant of gemm16_ss_1warp_bf16: tC += A @ B^T (no clear).
// Used to contract over multiple 16-wide k-blocks (e.g. L over D=128).
template <typename SmemA, typename SmemB, typename TC>
CUTE_DEVICE void gemm16_ss_1warp_bf16_acc(
    SmemA const& sA, SmemB const& sB, int lane, TC& tC) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));

  Tensor rB_full = thr_mma.partition_fragment_B(sB);
  auto smem_copy_B = cute::make_tiled_copy_B(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(lane);
  auto sB_part = thr_copy_B.partition_S(sB);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  Tensor rC = recast<float>(tC);  // NO clear(tC): accumulate
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
        cute::PPU_16x16x16_F32BF16BF16F32_TN_FIXED::fma(
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n));
      }
    }
  }
}

template <typename SmemA, typename TC>
CUTE_DEVICE void gemm16_sr_1warp_f16_acc(
    SmemA const& sA, const uint32_t b_regs[4], int lane, TC& tC) {
  TiledMma1WarpF16 mma;
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

  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(rA_full); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(rC); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        cute::PPU_16x16x16_F32F16F16F32_TN_FIXED_KDA::fma(
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            b_regs[0], b_regs[1], b_regs[2], b_regs[3],
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n));
      }
    }
  }
}

template <typename SmemA, typename TC>
CUTE_DEVICE void gemm16_sr_1warp_bf16_acc(
    SmemA const& sA, const uint32_t b_regs[4], int lane, TC& tC) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);

  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));

  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(rA_full); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(rC); ++n) {
        Tensor sliceA = rA_full(_, m, k);
        Tensor rA = recast<uint32_t>(sliceA);
        cute::PPU_16x16x16_F32BF16BF16F32_TN_FIXED::fma(
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            b_regs[0], b_regs[1], b_regs[2], b_regs[3],
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n));
      }
    }
  }
}

// Split form of the ordinary row-major A load used by gemm16_sr above.  This
// lets a warp issue the synchronous TSM->register transfer in an earlier,
// otherwise-idle scheduling phase and execute only the register-register tc02
// MMA once its dependent B fragment becomes ready.
template <typename SmemA>
CUTE_DEVICE void load16_A_1warp_bf16(
    SmemA const& sA, int lane, uint32_t a_regs[4]) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  Tensor rA_full = thr_mma.partition_fragment_A(sA);
  auto smem_copy_A = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(sA);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  Tensor rA = recast<uint32_t>(rA_full);
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) a_regs[i] = rA(i);
}

// ---------------------------------------------------------------------------
// AIU .swzl write + Trans=true swzl read for the kr^T @ U delta path.
//
// kr is stored physically as [CS=16, DK=128] row-major (one chunk of decayed
// K). The recurrence needs delta = kr^T @ U, i.e. A = kr^T[DK, CS] as the MMA A
// operand. Previously this was a strided DefaultCopy smem read (inner stride
// DK*2 = 256B, heavily bank-conflicted). Instead we AIU .swzl-write kr into a
// CUBE_H=16 / CUBE_W=32 cube tile (DK/32 = 4 col-cubes of 16x32) and read the
// transposed A fragment directly with a Trans=true TSM_LD_SWZL atom -- no
// explicit transpose. Layout + coordinate conventions are the probe-verified
// (cos=1.0) ones from flashqla_gemm_ss_swzlA_trans_acc_m16.

// Map a generic smem base pointer to the .swzl READ atom's expected base
// (CTA-relative .shared byte offset; required for correctness under >=2 blk/CU).
CUTE_DEVICE BF16* k2_swzl_read_base(const BF16* p) {
#if defined(__HGGC_ARCH__)
  return reinterpret_cast<BF16*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(p)));
#else
  return const_cast<BF16*>(p);
#endif
}

// AIU .swzl 2D write of a [rows<=16, DK=128] gmem tile into a CUBE_H=16/CUBE_W=32
// swizzled smem tile (4 contiguous 16x32 col-cubes). .padz fills partial rows.
__forceinline__ __device__ void k2_aiu_swzl_write_dk128(
    BF16* smem, const BF16* gmem, int row_stride_elems, int rows) {
#if defined(__HGGC_ARCH__)
  using AiuWriteCube = flashkda::PpuAiuLoad<
      /*NumBits=*/16 * 32 * 16, /*Element=*/cute::bfloat16_t,
      /*Trans=*/false, /*CUBE_H=*/16, /*CUBE_W=*/32>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t*>(gmem), /*h=*/rows, /*w=*/128,
            /*stride_w=*/row_stride_elems);
  CUTE_UNROLL
  for (int cube = 0; cube < 128 / 32; ++cube) {
    uint32_t dst_shared = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem + cube * 16 * 32));
    AiuWriteCube::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst_shared)),
                       reinterpret_cast<const void*>(gmem), desc,
                       /*coord_w=*/cube * 32, /*coord_h=*/0);
  }
#else
  (void)smem; (void)gmem; (void)row_stride_elems; (void)rows;
#endif
}

// Split load/MMA form used to overlap the U producer with workers' independent
// kr^T A-fragment loads.  The composed helper below retains the old API.
CUTE_DEVICE void load16_swzlA_trans_1warp_bf16(
    const BF16* Aswz, int mb, int lane, uint32_t a_regs[4]) {
#if defined(FLASHKDA_ARCH_150)
  using TsmTrans = flashkda::PpuTsmLdSwzl<
      cute::bfloat16_t, /*CH=*/16, /*CW=*/32, /*Swap=*/false, /*Trans=*/true,
      /*InstNum=*/1>;
  const int m_global   = mb * 16;          // dk-row base of this A-tile
  const int cube       = m_global / 32;    // physical col-cube of kr
  const int dk_in_cube = m_global % 32;    // {0,16}: row inside transposed tile
  int afrag[4] = {0, 0, 0, 0};
  TsmTrans::copy(afrag, reinterpret_cast<void*>(k2_swzl_read_base(Aswz)),
                 /*coord_h=*/0, /*coord_w=*/dk_in_cube,
                 /*cube_in_stage=*/0, /*stage=*/cube);
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) a_regs[i] = static_cast<uint32_t>(afrag[i]);
#elif defined(FLASHKDA_ARCH_100)
  // tc01 K1 stores this field as plain row-major [CS=16, DK=128]. Build the
  // logical kr^T[DK,CS] tile rooted at dk=mb*16 and let the selected
  // PPU0010 MMA layout derive its A fragment directly.
  auto A = cute::make_tensor(
      cute::make_smem_ptr(const_cast<BF16*>(Aswz + mb * 16)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::_1{}, cute::Int<128>{})));
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  Tensor rA_full = thr_mma.partition_fragment_A(A);
  auto smem_copy_A = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(A);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  Tensor packed = recast<uint32_t>(rA_full);
  CUTE_STATIC_ASSERT_V(cute::size(packed) == cute::Int<4>{});
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) a_regs[i] = packed(i);
#else
  (void)Aswz; (void)mb; (void)lane; (void)a_regs;
#endif
}

// tc01 experimental workspace format: kr^T is already materialized as
// row-major [DK,CH], so one state-update A tile is contiguous in TSM.
CUTE_DEVICE void load16_contig_krT_1warp_bf16(
    const BF16* KrT, int mb, int lane, uint32_t a_regs[4]) {
  auto A = cute::make_tensor(
      cute::make_smem_ptr(const_cast<BF16*>(KrT + mb * 16 * 16)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::_16{}, cute::_1{})));
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  Tensor rA_full = thr_mma.partition_fragment_A(A);
  auto smem_copy_A = cute::make_tiled_copy_A(
      cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
  auto thr_copy_A = smem_copy_A.get_thread_slice(lane);
  auto sA_part = thr_copy_A.partition_S(A);
  auto rA_retile = thr_copy_A.retile_D(rA_full);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k)
    cute::copy(smem_copy_A, sA_part(_, _, k), rA_retile(_, _, k));
  Tensor packed = recast<uint32_t>(rA_full);
  CUTE_STATIC_ASSERT_V(cute::size(packed) == cute::Int<4>{});
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) a_regs[i] = packed(i);
}

template <typename TC>
CUTE_DEVICE void gemm16_rr_1warp_bf16_acc(
    const uint32_t a_regs[4], const uint32_t b_regs[4], TC& tC) {
#if defined(__HGGC_ARCH__)
  Tensor rC = recast<float>(tC);
  cute::PPU_16x16x16_F32BF16BF16F32_TN_FIXED::fma(
      rC(0, 0, 0), rC(1, 0, 0), rC(2, 0, 0), rC(3, 0, 0),
      rC(4, 0, 0), rC(5, 0, 0), rC(6, 0, 0), rC(7, 0, 0),
      a_regs[0], a_regs[1], a_regs[2], a_regs[3],
      b_regs[0], b_regs[1], b_regs[2], b_regs[3],
      rC(0, 0, 0), rC(1, 0, 0), rC(2, 0, 0), rC(3, 0, 0),
      rC(4, 0, 0), rC(5, 0, 0), rC(6, 0, 0), rC(7, 0, 0));
#else
  (void)a_regs; (void)b_regs; (void)tC;
#endif
}

// tC += kr^T_tile @ U, where kr^T_tile = A[16(dk), 16(cs)] is the mb-th 16-row
// slice of kr^T read via Trans=true swzl from the CUBE_H=16 kr buffer, and U is
// supplied as pre-swizzled B-fragment registers (b_regs). Single warp, one MMA.
template <typename TC>
CUTE_DEVICE void gemm16_sr_swzlA_trans_1warp_bf16_acc(
    const BF16* Aswz, int mb, const uint32_t b_regs[4], int lane, TC& tC) {
  uint32_t a_regs[4];
  load16_swzlA_trans_1warp_bf16(Aswz, mb, lane, a_regs);
  gemm16_rr_1warp_bf16_acc(a_regs, b_regs, tC);
}

template <typename TC>
CUTE_DEVICE void gemm16_sr_contig_krT_1warp_bf16_acc(
    const BF16* KrT, int mb, const uint32_t b_regs[4], int lane, TC& tC) {
  uint32_t a_regs[4];
  load16_contig_krT_1warp_bf16(KrT, mb, lane, a_regs);
  gemm16_rr_1warp_bf16_acc(a_regs, b_regs, tC);
}

// Store a 16x16 f32 accumulator (bf16 TiledMMA C layout) to smem as a chosen type.
template <typename T, typename TC>
CUTE_DEVICE void store_acc16_bf16mma(TC const& tC, T* dst, int ldc, int lane) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(dst),
      cute::make_layout(cute::make_shape(16, 16), cute::make_stride(ldc, 1)));
  Tensor tCsC = thr_mma.partition_C(sOut);
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v)
        tCsC(v, m, n) = T(rC(v, m, n));
}

// Store the same logical C[r,c] tile as dst[c,r].  The affine K1 producer uses
// this to materialize P directly as contiguous row-major [DK,CH], avoiding a
// second scalar transpose without changing the BF16 conversion point.
template <typename T, typename TC>
CUTE_DEVICE void store_acc16_bf16mma_transposed(
    TC const& tC, T* dst, int lane) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(dst),
      cute::make_layout(cute::make_shape(16, 16), cute::make_stride(1, 16)));
  Tensor tCsC = thr_mma.partition_C(sOut);
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v)
        tCsC(v, m, n) = T(rC(v, m, n));
}

// Add a 16x16 f32 accumulator into an existing bf16 smem tile.
template <typename TC>
CUTE_DEVICE void store_acc16_add_bf16mma(TC const& tC, BF16* dst, int ldc, int lane) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  auto sOut = cute::make_tensor(
      cute::make_smem_ptr(dst),
      cute::make_layout(cute::make_shape(16, 16), cute::make_stride(ldc, 1)));
  Tensor tCsC = thr_mma.partition_C(sOut);
  Tensor rC = recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tCsC); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tCsC); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(tCsC); ++v) {
        float base = bf16_to_f32(tCsC(v, m, n));
        BF16 add = BF16(rC(v, m, n));
        tCsC(v, m, n) = BF16(base + bf16_to_f32(add));
      }
}

// Generic multi-warp TN GEMM over smem: C[M,N] = A[M,K] @ B[N,K]^T, bf16 in,
// f32 accumulate, output written to sC (row-major, ldc=N) as type OutT. All
// `nthreads` threads (nthreads/32 warps) cooperate; each warp owns a subset of
// the (M/16)*(N/16) 16x16 output blocks, contracting over K/16 k-blocks.
// A,B,C are flat row-major smem pointers. M,N,K multiples of 16.
template <int M, int N, int K, typename OutT>
CUTE_DEVICE void gemm_tn_bf16_smem(const BF16* sA, const BF16* sB, OutT* sC,
                                   int tid, int nthreads) {
  constexpr int MB = M / 16, NB = N / 16, KB = K / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  auto Av = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sA)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Bv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sB)),
      cute::make_layout(cute::make_shape(cute::Int<N>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  for (int blk = warp; blk < MB * NB; blk += nwarps) {
    int mb = blk / NB, nb = blk % NB;
    auto acc = make_acc16_bf16_1warp(lane);
    cute::clear(acc);
    CUTE_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      auto Ablk = cute::local_tile(Av, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(mb, kb));
      auto Bblk = cute::local_tile(Bv, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(nb, kb));
      gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
    }
    store_acc16_bf16mma<OutT>(acc, sC + mb * 16 * N + nb * 16, N, lane);
  }
}

// C[M,N] = A[M,K] @ B[K,N]. B is physically row-major [K,N], but the MMA helper
// sees a logical [N,K] view, avoiding an explicit B transpose in shared memory.
template <int M, int N, int K, typename OutT>
CUTE_DEVICE void gemm_nn_bf16_smem(const BF16* sA, const BF16* sBkn, OutT* sC,
                                   int tid, int nthreads) {
  constexpr int MB = M / 16, NB = N / 16, KB = K / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  auto Av = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sA)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Bv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sBkn)),
      cute::make_layout(cute::make_shape(cute::Int<N>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<N>{})));
  for (int blk = warp; blk < MB * NB; blk += nwarps) {
    int mb = blk / NB, nb = blk % NB;
    auto acc = make_acc16_bf16_1warp(lane);
    cute::clear(acc);
    CUTE_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      auto Ablk = cute::local_tile(Av, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(mb, kb));
      auto Bblk = cute::local_tile(Bv, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(nb, kb));
      gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
    }
    store_acc16_bf16mma<OutT>(acc, sC + mb * 16 * N + nb * 16, N, lane);
  }
}

// C[M,N] += A[M,K] @ B[K,N]. Used to fuse K2 Mqk@U into the O1 tile.
template <int M, int N, int K>
CUTE_DEVICE void gemm_nn_bf16_smem_add(const BF16* sA, const BF16* sBkn, BF16* sC,
                                       int tid, int nthreads) {
  constexpr int MB = M / 16, NB = N / 16, KB = K / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  auto Av = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sA)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Bv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sBkn)),
      cute::make_layout(cute::make_shape(cute::Int<N>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<N>{})));
  for (int blk = warp; blk < MB * NB; blk += nwarps) {
    int mb = blk / NB, nb = blk % NB;
    auto acc = make_acc16_bf16_1warp(lane);
    cute::clear(acc);
    CUTE_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      auto Ablk = cute::local_tile(Av, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(mb, kb));
      auto Bblk = cute::local_tile(Bv, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(nb, kb));
      gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
    }
    store_acc16_add_bf16mma(acc, sC + mb * 16 * N + nb * 16, N, lane);
  }
}

// Two independent NN GEMMs sharing the same B operand:
//   C0[M,N] = A0[M,K] @ B[K,N]
//   C1[M,N] = A1[M,K] @ B[K,N]
// For the Kernel2 CHxDVT path this lets warp 0/1 compute C0 while warp 2/3
// compute C1, instead of running the two small GEMMs serially on warp 0/1.
template <int M, int N, int K, typename OutT>
CUTE_DEVICE void gemm_nn2_bf16_smem(const BF16* sA0, const BF16* sA1,
                                    const BF16* sBkn, OutT* sC0, OutT* sC1,
                                    int tid, int nthreads) {
  constexpr int MB = M / 16, NB = N / 16, KB = K / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  auto A0v = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sA0)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto A1v = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sA1)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Bv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sBkn)),
      cute::make_layout(cute::make_shape(cute::Int<N>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<N>{})));
  constexpr int blocks_per_gemm = MB * NB;
  for (int blk = warp; blk < 2 * blocks_per_gemm; blk += nwarps) {
    bool second = blk >= blocks_per_gemm;
    int local_blk = second ? (blk - blocks_per_gemm) : blk;
    int mb = local_blk / NB, nb = local_blk % NB;
    auto acc = make_acc16_bf16_1warp(lane);
    cute::clear(acc);
    CUTE_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      auto Ablk = second
          ? cute::local_tile(A1v, cute::make_shape(cute::_16{}, cute::_16{}), cute::make_coord(mb, kb))
          : cute::local_tile(A0v, cute::make_shape(cute::_16{}, cute::_16{}), cute::make_coord(mb, kb));
      auto Bblk = cute::local_tile(Bv, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(nb, kb));
      gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
    }
    OutT* sC = second ? sC1 : sC0;
    store_acc16_bf16mma<OutT>(acc, sC + mb * 16 * N + nb * 16, N, lane);
  }
}

// C[M,N] = A[K,M]^T @ B[K,N]. Both operands are kept in their natural row-major
// physical layout and only exposed to MMA through logical transposed views.
template <int M, int N, int K, typename OutT>
CUTE_DEVICE void gemm_at_b_bf16_smem(const BF16* sAkm, const BF16* sBkn, OutT* sC,
                                     int tid, int nthreads) {
  constexpr int MB = M / 16, NB = N / 16, KB = K / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  auto Av = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sAkm)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<M>{})));
  auto Bv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sBkn)),
      cute::make_layout(cute::make_shape(cute::Int<N>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<N>{})));
  for (int blk = warp; blk < MB * NB; blk += nwarps) {
    int mb = blk / NB, nb = blk % NB;
    auto acc = make_acc16_bf16_1warp(lane);
    cute::clear(acc);
    CUTE_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      auto Ablk = cute::local_tile(Av, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(mb, kb));
      auto Bblk = cute::local_tile(Bv, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(nb, kb));
      gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
    }
    store_acc16_bf16mma<OutT>(acc, sC + mb * 16 * N + nb * 16, N, lane);
  }
}

// Kernel2 fused path for CHxDVT U consumers:
//   U      = Inv @ vbeta        is produced in each consuming warp as registers.
//   sO1   += Mqk @ U            is emitted by the first NB warps.
//   sDelta = kr^T @ U           consumes U as pre-swizzled B-fragment registers.
// This avoids routing U through shared memory for the downstream GEMMs.
template <int M, int N, int K>
CUTE_DEVICE void k2_fused_u_o_delta_bf16_smem(
    const BF16* sInv, const BF16* sVbeta, const BF16* sMqk,
    const BF16* sKr, BF16* sO1, BF16* sDelta, int tid, int nthreads) {
  static_assert(K == 16, "K2 fused U path expects one 16-wide reduction tile");
  constexpr int MB = M / 16, NB = N / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  int nb = warp % NB;

  auto Inv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sInv)),
      cute::make_layout(cute::make_shape(cute::Int<K>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Vbeta = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sVbeta)),
      cute::make_layout(cute::make_shape(cute::Int<N>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<N>{})));
  auto Mqk = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sMqk)),
      cute::make_layout(cute::make_shape(cute::Int<K>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Kr = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sKr)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<M>{})));

  auto u_acc = make_acc16_bf16_1warp(lane);
  cute::clear(u_acc);
  auto InvBlk = cute::local_tile(Inv, cute::make_shape(cute::_16{}, cute::_16{}),
                                 cute::make_coord(0, 0));
  auto VbetaBlk = cute::local_tile(Vbeta, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(nb, 0));
  gemm16_ss_1warp_bf16_acc(InvBlk, VbetaBlk, lane, u_acc);

  uint32_t u_bregs[4];
  bf16_acc_to_bfrag_swizzle_u32(u_acc, u_bregs, lane);

  if (warp < NB) {
    auto o_acc = make_acc16_bf16_1warp(lane);
    cute::clear(o_acc);
    auto MqkBlk = cute::local_tile(Mqk, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(0, 0));
    gemm16_sr_1warp_bf16_acc(MqkBlk, u_bregs, lane, o_acc);
    store_acc16_add_bf16mma(o_acc, sO1 + nb * 16, N, lane);
  }

  for (int blk = warp; blk < MB * NB; blk += nwarps) {
    int mb = blk / NB;
    auto delta_acc = make_acc16_bf16_1warp(lane);
    cute::clear(delta_acc);
    auto KrBlk = cute::local_tile(Kr, cute::make_shape(cute::_16{}, cute::_16{}),
                                  cute::make_coord(mb, 0));
    gemm16_sr_1warp_bf16_acc(KrBlk, u_bregs, lane, delta_acc);
    store_acc16_bf16mma<BF16>(delta_acc, sDelta + mb * 16 * N + nb * 16, N, lane);
  }
}

template <int N>
CUTE_DEVICE void k2_vbeta_bfrag_from_smem_movmatrix_u32(
    const BF16* sV, const BF16* sVsub, const float* sBeta,
    int nb, int lane, uint32_t b_regs[4], uint32_t* scratch) {
  TiledMma1WarpBf16 mma;
  auto thr_mma = mma.get_thread_slice(lane);
  const int col0 = nb * 16;
  auto V = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sV + col0)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::Int<N>{}, cute::_1{})));
  auto Vsub = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sVsub + col0)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::Int<N>{}, cute::_1{})));
  auto Beta = cute::make_tensor(cute::make_smem_ptr(const_cast<float*>(sBeta)),
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                        cute::make_stride(cute::_1{}, cute::_0{})));
  auto v_part = thr_mma.partition_C(V);
  auto vsub_part = thr_mma.partition_C(Vsub);
  auto beta_part = thr_mma.partition_C(Beta);

  uint32_t c_half[8];
  int pos = 0;
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(v_part); ++m)
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(v_part); ++n)
      CUTE_UNROLL
      for (int v = 0; v < cute::size<0>(v_part); ++v) {
        float val = (bf16_to_f32(v_part(v, m, n)) -
                     bf16_to_f32(vsub_part(v, m, n))) * beta_part(v, m, n);
        c_half[pos] = BF16(val).storage;
        ++pos;
      }
  uint32_t c_regs[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i)
    c_regs[i] = pack_half2_u32(c_half[2 * i], c_half[2 * i + 1]);
  bf16_cfrag_to_bfrag_movmatrix_u32(c_regs, b_regs, lane, scratch);
}

// Variant that computes each U tile only once per CTA and stores the swizzled
// B-fragment registers into a temporary shared scratch. For DVT=32 this reduces
// Inv@vbeta from 8 warp-level GEMMs to 2, trading it for a small shared exchange.
template <int M, int N, int K>
CUTE_DEVICE void k2_fused_u_o_delta_cached_bf16_smem(
    const BF16* sInv, const BF16* sVbeta, const BF16* sMqk,
    const BF16* sKr, BF16* sO1, BF16* sDelta, BF16* sUScratch,
    int tid, int nthreads) {
  static_assert(K == 16, "K2 fused U path expects one 16-wide reduction tile");
  constexpr int MB = M / 16, NB = N / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  int nb = warp % NB;
  uint32_t* sURegs = reinterpret_cast<uint32_t*>(sUScratch);
  uint32_t u_bregs[4] = {0, 0, 0, 0};

  auto Inv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sInv)),
      cute::make_layout(cute::make_shape(cute::Int<K>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Vbeta = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sVbeta)),
      cute::make_layout(cute::make_shape(cute::Int<N>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<N>{})));
  auto Mqk = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sMqk)),
      cute::make_layout(cute::make_shape(cute::Int<K>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Kr = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sKr)),
      cute::make_layout(cute::make_shape(cute::Int<M>{}, cute::Int<K>{}),
                        cute::make_stride(cute::_1{}, cute::Int<M>{})));

  if (warp < NB) {
    auto u_acc = make_acc16_bf16_1warp(lane);
    cute::clear(u_acc);
    auto InvBlk = cute::local_tile(Inv, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(0, 0));
    auto VbetaBlk = cute::local_tile(Vbeta, cute::make_shape(cute::_16{}, cute::_16{}),
                                     cute::make_coord(warp, 0));
    gemm16_ss_1warp_bf16_acc(InvBlk, VbetaBlk, lane, u_acc);
    bf16_acc_to_bfrag_swizzle_u32(u_acc, u_bregs, lane);
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i)
      sURegs[(warp * 32 + lane) * 4 + i] = u_bregs[i];

    auto o_acc = make_acc16_bf16_1warp(lane);
    cute::clear(o_acc);
    auto MqkBlk = cute::local_tile(Mqk, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(0, 0));
    gemm16_sr_1warp_bf16_acc(MqkBlk, u_bregs, lane, o_acc);
    store_acc16_add_bf16mma(o_acc, sO1 + warp * 16, N, lane);
  }

  __syncthreads();
  if (warp >= NB) {
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i)
      u_bregs[i] = sURegs[(nb * 32 + lane) * 4 + i];
  }

  for (int blk = warp; blk < MB * NB; blk += nwarps) {
    int mb = blk / NB;
    auto delta_acc = make_acc16_bf16_1warp(lane);
    cute::clear(delta_acc);
    auto KrBlk = cute::local_tile(Kr, cute::make_shape(cute::_16{}, cute::_16{}),
                                  cute::make_coord(mb, 0));
    gemm16_sr_1warp_bf16_acc(KrBlk, u_bregs, lane, delta_acc);
    store_acc16_bf16mma<BF16>(delta_acc, sDelta + mb * 16 * N + nb * 16, N, lane);
  }
}

// Direct-vbeta + movmatrix variant. Producer warps generate vbeta as register
// B-fragments via ldmatrix.trans scratch, then cache U B-fragments for consumers.
template <int M, int N, int K>
CUTE_DEVICE void k2_fused_u_o_delta_cached_vbeta_movmatrix_bf16_smem(
    const BF16* sInv, const BF16* sV, const BF16* sVsub, const float* sBeta,
    const BF16* sMqk, const BF16* sKr, BF16* sO1, float* sDelta,
    BF16* sUScratch, int tid, int nthreads) {
  static_assert(K == 16, "K2 direct vbeta path expects one 16-wide reduction tile");
  constexpr int MB = M / 16, NB = N / 16;
  const int warp = tid / 32, lane = tid % 32, nwarps = nthreads / 32;
  int nb = warp % NB;
  uint32_t* sURegs = reinterpret_cast<uint32_t*>(sUScratch);
  uint32_t* warpScratch = sURegs + warp * 32 * 4;
  uint32_t u_bregs[4] = {0, 0, 0, 0};

  auto Inv = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sInv)),
      cute::make_layout(cute::make_shape(cute::Int<K>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  auto Mqk = cute::make_tensor(cute::make_smem_ptr(const_cast<BF16*>(sMqk)),
      cute::make_layout(cute::make_shape(cute::Int<K>{}, cute::Int<K>{}),
                        cute::make_stride(cute::Int<K>{}, cute::_1{})));
  // NOTE: sKr is AIU .swzl-written (CUBE_H=16/CUBE_W=32) in this path, so the
  // delta = kr^T @ U read below goes through gemm16_sr_swzlA_trans_1warp_bf16_acc
  // instead of a logical-transpose smem view.

  // Stage A — producer warps (warp<NB): compute u_bregs = Inv @ vbeta and
  // cache into sURegs for consumers. P1-D: NO O update here (moved to B2).
  if (warp < NB) {
    uint32_t vbeta_bregs[4];
    k2_vbeta_bfrag_from_smem_movmatrix_u32<N>(
        sV, sVsub, sBeta, warp, lane, vbeta_bregs, warpScratch);

    auto u_acc = make_acc16_bf16_1warp(lane);
    cute::clear(u_acc);
    auto InvBlk = cute::local_tile(Inv, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(0, 0));
    gemm16_sr_1warp_bf16_acc(InvBlk, vbeta_bregs, lane, u_acc);
    bf16_acc_to_bfrag_movmatrix_u32(u_acc, u_bregs, lane, warpScratch);
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i)
      sURegs[(warp * 32 + lane) * 4 + i] = u_bregs[i];
  }

  __syncthreads();
  // Stage B1 — consumer warps read u_bregs for own nb; producers keep register.
  if (warp >= NB) {
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i)
      u_bregs[i] = sURegs[(nb * 32 + lane) * 4 + i];
  }

  // Stage B2 — P1-D: O update offloaded to warps NB..2*NB-1 (i.e., warps 2 & 3
  // for NB=2). Removes one MMA + one add-store from warp 0/1's critical path.
  // Runs in the same warp before the delta loop; not in parallel per-warp, but
  // NOT on warps 0/1 anymore. Warps 4..7 skip B2 and go straight to delta.
  if (warp >= NB && warp < 2 * NB) {
    auto o_acc = make_acc16_bf16_1warp(lane);
    cute::clear(o_acc);
    auto MqkBlk = cute::local_tile(Mqk, cute::make_shape(cute::_16{}, cute::_16{}),
                                   cute::make_coord(0, 0));
    gemm16_sr_1warp_bf16_acc(MqkBlk, u_bregs, lane, o_acc);
    store_acc16_add_bf16mma(o_acc, sO1 + nb * 16, N, lane);
  }

  // Stage C — delta. All 8 warps, 2 iterations each.
  for (int blk = warp; blk < MB * NB; blk += nwarps) {
    int mb = blk / NB;
    auto delta_acc = make_acc16_bf16_1warp(lane);
    cute::clear(delta_acc);
    gemm16_sr_swzlA_trans_1warp_bf16_acc(sKr, mb, u_bregs, lane, delta_acc);
    store_acc16_bf16mma<float>(delta_acc, sDelta + mb * 16 * N + nb * 16, N, lane);
  }
}

}  // namespace flashkda
