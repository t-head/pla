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
 * FlashQLA PPU1.5 CuTe rewrite — Kernel (5) fused_fwd_cp  [NON-WARP-SPECIALIZED]
 *
 * Symmetric data-parallel variant. Rationale: fused_fwd is memory-bound and on
 * 39-CU PPU1.5 warp-specialization is structurally disadvantaged (no register
 * realloc -> the producer WG wastes the uniform NREG budget; synchronous mma
 * negates producer/consumer async overlap). Here ALL threads run the same
 * program (bulk-synchronous), so every warp carries useful compute and the
 * active-warp count per scheduler is maximized.
 *
 * DV-COLUMN SHARDING: each 128-thread group owns one 64-column DV half and a
 * register-h shard h[DK=128,DVh=64]. The launcher uses either two groups in one
 * CTA or two one-group CTAs along grid.z. The two-group form avoids duplicating
 * shared input work; the one-group form raises CTA parallelism when all of its
 * split tasks fit in the 3-CTA/CU residency available at the smaller smem size.
 * No producer warpgroup or per-buffer mbarriers are used; phases are separated
 * by block or 128-thread group barriers.
 *
 * Five DV-sharded GEMMs plus one DV-agnostic P GEMM per chunk. In the two-group
 * form, P/Pg is computed once by the otherwise-idle second group while the
 * first group builds g/Ag, then both groups consume the shared Pg tile:
 *   GEMM1 U      = K  @ h_stm[DVh]        [CS,DVh]  W = V[:,half] - g_exp*U
 *   GEMM2 Vd     = Ag @ W                 [CS,DVh]  (Ag = gated WY factor, shared)
 *   GEMM3 O_int  = Q  @ h_stm[DVh]        [CS,DVh]  O_int *= scale*g_exp
 *   GEMM4 P      = Q  @ K^T               [CS,CS ]  Pg = scale*tri(exp(gi-gj))*P
 *   GEMM5 O_intr = Pg @ Vd                [CS,DVh]  O = O_int + O_intr
 *   GEMM6 h     += K^T @ (g_rev*Vd)       [DK,DVh]  (after h *= exp(g_last))
 *
 * Shared (computed once per CTA, then a __syncthreads): the input tiles
 * K/V/Q/A/g/b, the gated Ag[CS,CS] (in place over A_buf), Pg[CS,CS], and
 * g_exp/g_rev. The register chain keeps h/W/Vd/gVd private to each warp; its
 * W smem tile is used only to coalesce the NGRP=1 output store.
 *
 * Grid: cp_batch * H * (2 / groups_per_CTA).
 */
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <hggc_awbarrier.h>

#include "flashqla_cute_traits.cuh"
// NOTE: no <cutlass/barrier.h>. All synchronisation here is awbar / __ppu_barrier
// / raw ppu.bar.sync asm; cutlass::Barrier and NamedBarrier are never used.

namespace flashqla_cute {

namespace fq = flashqla;

#ifndef FF_TC01_DV_SPLIT
#define FF_TC01_DV_SPLIT 0
#endif

// Production build flags (all default ON via build system unless noted):
//   FF_K_SWZL / FF_Q_SWZL: AIU .swzl 32x32 atom writes for K/Q → zero bank conflicts
//   FF_SWZL_CVTA: CTA-relative .shared dst for .swzl (2 blk/CU correctness)
//   FF_A_SWZL: Ag gated factor → Swizzle<2,3,3> buffer, int4 vec stores
//   Gate ratios use exp(g_r-g_c) directly to avoid overflow for large decay.
//   FF_GRP_BAR: intra-group named barrier (ppu.bar.sync id, 128) for W→GEMM2 / Pg→GEMM5
//   FF_GEMM_REORDER: GEMM schedule 4→3→1→2→6→5 (runtime guard below)
#ifndef FF_GEMM_REORDER
#define FF_GEMM_REORDER 1
#endif

#ifndef FF_GVD_REG_B
// The C-fragment to B-fragment conversion is architecture-dispatched.
#define FF_GVD_REG_B 1
#endif

#ifndef FF_VD_REG_B
#define FF_VD_REG_B_TC01 1
#define FF_VD_REG_B_TC02 0
#else
#define FF_VD_REG_B_TC01 FF_VD_REG_B
#define FF_VD_REG_B_TC02 FF_VD_REG_B
#endif
#if !defined(FF_VD_REG_B)
#if defined(FLASHQLA_ARCH_100)
#define FF_VD_REG_B FF_VD_REG_B_TC01
#else
#define FF_VD_REG_B FF_VD_REG_B_TC02
#endif
#endif

#if !defined(FF_REG_CHAIN)
#define FF_REG_CHAIN 1
#endif

constexpr int FF_DK    = 128;
constexpr int FF_DV    = 128;
constexpr int FF_DVH   = 64;
constexpr int FF_CS    = 32;
constexpr int FF_WG    = 128;   // 4-warp MMA group = one DV-half group
using MmaO = fq::TiledMmaM32;
using MmaH = fq::TiledMmaM32;
constexpr int FF_NGRP_TC01 = 2;
constexpr int FF_NGRP_TC02 = 2;
constexpr float FF_LOG2E = 1.4426950408889634f;

// Keep one 16-row DV tile on the same warp throughout the transposed
// state/output chain. On tc02 an accumulator becomes the next A fragment with
// a lane-local pack; tc01 uses cfrag_to_afrag_u32 for the differing A/C layout.
template <typename Element>
using FfMmaM4T = decltype(cute::make_tiled_mma(
    fq::MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>{}));

#if defined(FLASHQLA_ARCH_150)
// Retained for the existing helper definitions below.  The transposed hot path
// uses FfMmaM4T; keeping this alias avoids coupling the experiment to removal of
// the previously validated N-major helpers.
template <typename Element>
using FfMmaN4T = decltype(cute::make_tiled_mma(
    fq::MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_1, cute::_4, cute::_1>>{}));
#endif

// Store one warp-owned [DV16, CS32] O^T accumulator tile directly as the
// row-major [CS32, DV16] output.  cfrag_to_bfrag_swizzle_u32 maps each 16x16
// C tile to the tc02 B[N=CS,K=DV] layout, which is exactly the transpose needed
// here.  Its four b32 values are naturally coalesced across each four-lane
// group, so no TSM staging or block/group barrier is required.
template <typename Element, typename TC>
__device__ __forceinline__ void ff_store_ot_register_direct(
    TC const& o_acc, Element* out, int64_t out_row_stride,
    int dv_base, int tidx, int actual) {
  const int lane = tidx & 31;
  const int warp = tidx >> 5;
  const int cs8 = lane >> 2;
  const int dv_pair = (lane & 3) * 2;
  cute::Tensor rO = cute::recast<float>(o_acc);

  CUTE_UNROLL
  for (int n = 0; n < 2; ++n) {
    uint32_t c_regs[4];
    uint32_t b_regs[4];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      const uint32_t lo = fq::element_bits16<Element>(
          fq::float_to_element<Element>(rO(2 * i, 0, n)));
      const uint32_t hi = fq::element_bits16<Element>(
          fq::float_to_element<Element>(rO(2 * i + 1, 0, n)));
      c_regs[i] = fq::pack_b16x2_u32(lo, hi);
    }
    fq::cfrag_to_bfrag_swizzle_u32(c_regs, b_regs, lane);

    const int cs0 = n * 16 + cs8;
    const int cs1 = cs0 + 8;
    const int dv = dv_base + warp * 16 + dv_pair;
    if (cs0 < actual) {
      *reinterpret_cast<uint32_t*>(
          out + (int64_t)cs0 * out_row_stride + dv) = b_regs[0];
      *reinterpret_cast<uint32_t*>(
          out + (int64_t)cs0 * out_row_stride + dv + 8) = b_regs[1];
    }
    if (cs1 < actual) {
      *reinterpret_cast<uint32_t*>(
          out + (int64_t)cs1 * out_row_stride + dv) = b_regs[2];
      *reinterpret_cast<uint32_t*>(
          out + (int64_t)cs1 * out_row_stride + dv + 8) = b_regs[3];
    }
  }
}
// Gate ratio exp(g_r - g_c) for r >= c, from the log-domain g snapshot. Computing
// the difference before the exp keeps it in range: the old form exp(g_r)*exp(-g_c)
// underflows one factor to 0 and overflows the other to +inf under strong decay,
// and 0 * inf is NaN.
//
// GDN supplies a non-positive gate and chunk_local_cumsum accumulates each head
// serially, so every stored prefix is the preceding prefix plus a non-positive
// fp32 value.  It therefore cannot increase, including after rounding.  All
// callers below also guard r >= c, which makes g[r] - g[c] <= 0 without a clamp.
// Keeping that invariant explicit avoids an fminf in both hot Ag/Pg loops.
__device__ __forceinline__ float ff_gate_ratio(const float* g_log, int r, int c) {
  return exp2f((g_log[r] - g_log[c]) * FF_LOG2E);
}

#if defined(FLASHQLA_ARCH_150)
// Q @ h and K @ h share h. Consume one h B fragment twice before advancing to
// the next K tile, while reusing one A fragment for Q and K in sequence.
template <typename Element, typename SmemB, typename TC0, typename TC1,
          typename SmemCopyB>
__device__ __forceinline__ void ff_gemm_qk_h(
    const Element* Qswz, const Element* Kswz, SmemB const& sH, int tidx,
    TC0& qh_acc, TC1& kh_acc, SmemCopyB smem_copy_b) {
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;
  constexpr int INST = FF_DK / CW;
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false,
      /*Trans=*/false, /*InstNum=*/INST>;

  fq::TiledMmaM32T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  const int warp_m = (tidx / 32) % 2;
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<FF_CS, FF_DK>{});
  cute::Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  cute::Tensor rB_full = thr_mma.partition_fragment_B(sH);
  auto smem_copy_B = cute::make_tiled_copy_B(smem_copy_b, mma);
  auto thr_copy_B = smem_copy_B.get_thread_slice(tidx);
  auto sB_part = thr_copy_B.partition_S(sH);
  auto rB_retile = thr_copy_B.retile_D(rB_full);

  cute::clear(qh_acc);
  cute::clear(kh_acc);
  cute::Tensor rQh = cute::recast<float>(qh_acc);
  cute::Tensor rKh = cute::recast<float>(kh_acc);
  CUTE_UNROLL
  for (int kt = 0; kt < cute::size<2>(rB_full); ++kt) {
    cute::copy(smem_copy_B,
               sB_part(cute::_, cute::_, kt),
               rB_retile(cute::_, cute::_, kt));
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(qh_acc); ++m) {
      const int cube = kt / KT_PER_CUBE;
      const int col_in_cube = (kt % KT_PER_CUBE) * 16;
      const int row = warp_m * 16 + m * 32;
      cute::Tensor rA = cute::recast<uint32_t>(rA_full(cute::_, m, kt));
      int frag[4] = {0, 0, 0, 0};

      TsmA::copy(frag,
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Qswz)),
                 col_in_cube, row, cube, 0);
      rA(0) = static_cast<uint32_t>(frag[0]);
      rA(1) = static_cast<uint32_t>(frag[1]);
      rA(2) = static_cast<uint32_t>(frag[2]);
      rA(3) = static_cast<uint32_t>(frag[3]);
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        cute::Tensor rB =
            cute::recast<uint32_t>(rB_full(cute::_, n, kt));
        fq::ppu_mma_impl<Element>(
            rQh(0,m,n), rQh(1,m,n), rQh(2,m,n), rQh(3,m,n),
            rQh(4,m,n), rQh(5,m,n), rQh(6,m,n), rQh(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rQh(0,m,n), rQh(1,m,n), rQh(2,m,n), rQh(3,m,n),
            rQh(4,m,n), rQh(5,m,n), rQh(6,m,n), rQh(7,m,n));
      }

      TsmA::copy(frag,
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Kswz)),
                 col_in_cube, row, cube, 0);
      rA(0) = static_cast<uint32_t>(frag[0]);
      rA(1) = static_cast<uint32_t>(frag[1]);
      rA(2) = static_cast<uint32_t>(frag[2]);
      rA(3) = static_cast<uint32_t>(frag[3]);
      CUTE_UNROLL
      for (int n = 0; n < cute::size<1>(rB_full); ++n) {
        cute::Tensor rB =
            cute::recast<uint32_t>(rB_full(cute::_, n, kt));
        fq::ppu_mma_impl<Element>(
            rKh(0,m,n), rKh(1,m,n), rKh(2,m,n), rKh(3,m,n),
            rKh(4,m,n), rKh(5,m,n), rKh(6,m,n), rKh(7,m,n),
            rA(0), rA(1), rA(2), rA(3),
            rB(0), rB(1), rB(2), rB(3),
            rKh(0,m,n), rKh(1,m,n), rKh(2,m,n), rKh(3,m,n),
            rKh(4,m,n), rKh(5,m,n), rKh(6,m,n), rKh(7,m,n));
      }
    }
  }
}

// Register-B variant for the 1x4 warp layout. A is a natural [M,K] matrix in
// the same 32x32 AIU-swizzled format used by Q/K/Ag/Pg. B is already arranged
// as tc02 fragments by the producer accumulator in the same warp.
template <int M, int K, typename Element, typename TC>
__device__ __forceinline__ void ff_gemm_swzlA_regB_n4(
    const Element* Aswz, const uint32_t b_regs[K / 16][1][4], int tidx,
    TC& tC, bool clear_acc = true) {
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;
  constexpr int INST = K / CW;
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false,
      /*Trans=*/false, /*InstNum=*/INST>;

  FfMmaN4T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<M, K>{});
  cute::Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  if (clear_acc) cute::clear(tC);
  cute::Tensor rC = cute::recast<float>(tC);

  CUTE_UNROLL
  for (int kt = 0; kt < cute::size<2>(rA_full); ++kt) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      const int cube = kt / KT_PER_CUBE;
      const int col_in_cube = (kt % KT_PER_CUBE) * 16;
      const int row = m * 16;
      int frag[4] = {0, 0, 0, 0};
      TsmA::copy(frag,
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Aswz)),
                 col_in_cube, row, cube, 0);
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tC); ++n) {
        fq::ppu_mma_impl<Element>(
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n),
            static_cast<uint32_t>(frag[0]), static_cast<uint32_t>(frag[1]),
            static_cast<uint32_t>(frag[2]), static_cast<uint32_t>(frag[3]),
            b_regs[kt][n][0], b_regs[kt][n][1],
            b_regs[kt][n][2], b_regs[kt][n][3],
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n));
      }
    }
  }
}

// Q@h and K@h share the register-resident h fragments. Each warp owns one DV
// tile, so no shared-memory publication is required between the two GEMMs.
template <typename Element, typename TC0, typename TC1>
__device__ __forceinline__ void ff_gemm_qk_h_reg_n4(
    const Element* Qswz, const Element* Kswz,
    const uint32_t h_regs[FF_DK / 16][1][4], int tidx,
    TC0& qh_acc, TC1& kh_acc) {
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;
  constexpr int INST = FF_DK / CW;
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false,
      /*Trans=*/false, /*InstNum=*/INST>;

  FfMmaN4T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<FF_CS, FF_DK>{});
  cute::Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  cute::clear(qh_acc);
  cute::clear(kh_acc);
  cute::Tensor rQh = cute::recast<float>(qh_acc);
  cute::Tensor rKh = cute::recast<float>(kh_acc);

  CUTE_UNROLL
  for (int kt = 0; kt < cute::size<2>(rA_full); ++kt) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(qh_acc); ++m) {
      const int cube = kt / KT_PER_CUBE;
      const int col_in_cube = (kt % KT_PER_CUBE) * 16;
      const int row = m * 16;
      int frag[4] = {0, 0, 0, 0};
      TsmA::copy(frag,
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Qswz)),
                 col_in_cube, row, cube, 0);
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(qh_acc); ++n) {
        fq::ppu_mma_impl<Element>(
            rQh(0,m,n), rQh(1,m,n), rQh(2,m,n), rQh(3,m,n),
            rQh(4,m,n), rQh(5,m,n), rQh(6,m,n), rQh(7,m,n),
            static_cast<uint32_t>(frag[0]), static_cast<uint32_t>(frag[1]),
            static_cast<uint32_t>(frag[2]), static_cast<uint32_t>(frag[3]),
            h_regs[kt][n][0], h_regs[kt][n][1],
            h_regs[kt][n][2], h_regs[kt][n][3],
            rQh(0,m,n), rQh(1,m,n), rQh(2,m,n), rQh(3,m,n),
            rQh(4,m,n), rQh(5,m,n), rQh(6,m,n), rQh(7,m,n));
      }
      TsmA::copy(frag,
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Kswz)),
                 col_in_cube, row, cube, 0);
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(kh_acc); ++n) {
        fq::ppu_mma_impl<Element>(
            rKh(0,m,n), rKh(1,m,n), rKh(2,m,n), rKh(3,m,n),
            rKh(4,m,n), rKh(5,m,n), rKh(6,m,n), rKh(7,m,n),
            static_cast<uint32_t>(frag[0]), static_cast<uint32_t>(frag[1]),
            static_cast<uint32_t>(frag[2]), static_cast<uint32_t>(frag[3]),
            h_regs[kt][n][0], h_regs[kt][n][1],
            h_regs[kt][n][2], h_regs[kt][n][3],
            rKh(0,m,n), rKh(1,m,n), rKh(2,m,n), rKh(3,m,n),
            rKh(4,m,n), rKh(5,m,n), rKh(6,m,n), rKh(7,m,n));
      }
    }
  }
}

// K^T @ gVd updates h in the same 1x4 warp layout. K is stored as swizzled
// [CS,DK], so the AIU read is transposed into [DK,CS].
template <typename Element, typename TC>
__device__ __forceinline__ void ff_gemm_kt_regB_n4_acc(
    const Element* Kswz, const uint32_t b_regs[FF_CS / 16][1][4], int tidx,
    TC& h_acc) {
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, 32, 32, /*Swap=*/false,
      /*Trans=*/true, /*InstNum=*/1>;
  FfMmaN4T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<FF_DK, FF_CS>{});
  cute::Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  cute::Tensor rH = cute::recast<float>(h_acc);

  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(h_acc); ++m) {
    const int m_global = m * 16;
    const int cube = m_global / 32;
    const int row_in_cube = m_global % 32;
    CUTE_UNROLL
    for (int kt = 0; kt < cute::size<2>(rA_full); ++kt) {
      int frag[4] = {0, 0, 0, 0};
      TsmA::copy(frag,
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Kswz)),
                 kt * 16, row_in_cube, 0, cube);
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(h_acc); ++n) {
        fq::ppu_mma_impl<Element>(
            rH(0,m,n), rH(1,m,n), rH(2,m,n), rH(3,m,n),
            rH(4,m,n), rH(5,m,n), rH(6,m,n), rH(7,m,n),
            static_cast<uint32_t>(frag[0]), static_cast<uint32_t>(frag[1]),
            static_cast<uint32_t>(frag[2]), static_cast<uint32_t>(frag[3]),
            b_regs[kt][n][0], b_regs[kt][n][1],
            b_regs[kt][n][2], b_regs[kt][n][3],
            rH(0,m,n), rH(1,m,n), rH(2,m,n), rH(3,m,n),
            rH(4,m,n), rH(5,m,n), rH(6,m,n), rH(7,m,n));
      }
    }
  }
}

#endif  // FLASHQLA_ARCH_150 tc02-only N-major helpers

// Transposed register-chain helpers for the 4x1 warp layout. The caller
// supplies architecture-correct A fragments. Swizzled B loads use the native
// copy traits on tc01 and the compact direct TSM sequence on tc02.
template <typename Element, typename TC0, typename TC1>
__device__ __forceinline__ void ff_gemm_regA_qkB_m4(
    const uint32_t h_regs[FF_DK / 16][1][4],
    const Element* Qswz, const Element* Kswz, int tidx,
    TC0& hq_acc, TC1& hk_acc) {
#if defined(FLASHQLA_ARCH_100)
  using Traits = fq::FqAiuTileTc01<Element, FF_CS, FF_DK>;
  using Mma = FfMmaM4T<Element>;
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto sQ = Traits::view(Qswz);
  auto sK = Traits::view(Kswz);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<FF_DVH, FF_DK>{});
  auto rA_full = thr_mma.partition_fragment_A(dummyA);
  auto rB_full = thr_mma.partition_fragment_B(sQ);
  auto scpB = cute::make_tiled_copy_B(typename Traits::SmemCopyAtom{}, mma);
  auto tcpB = scpB.get_thread_slice(tidx);
  auto tQsB = tcpB.partition_S(cute::make_mix_tensor_like(sQ));
  auto tKsB = tcpB.partition_S(cute::make_mix_tensor_like(sK));
  auto tBrB = tcpB.retile_D(rB_full);

  CUTE_UNROLL
  for (int kt = 0; kt < FF_DK / 16; ++kt) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(rA_full); ++m) {
      auto rA = cute::recast<uint32_t>(rA_full(cute::_, m, kt));
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) rA(i) = h_regs[kt][m][i];
    }
  }

  cute::clear(hq_acc);
  cute::clear(hk_acc);
  CUTE_UNROLL
  for (int kt = 0; kt < FF_DK / 16; ++kt) {
    cute::copy(scpB, tQsB(cute::_, cute::_, kt),
               tBrB(cute::_, cute::_, kt));
    cute::gemm(mma, rA_full(cute::_, cute::_, kt),
               rB_full(cute::_, cute::_, kt), hq_acc);
    cute::copy(scpB, tKsB(cute::_, cute::_, kt),
               tBrB(cute::_, cute::_, kt));
    cute::gemm(mma, rA_full(cute::_, cute::_, kt),
               rB_full(cute::_, cute::_, kt), hk_acc);
  }
#else
  (void)tidx;
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;
  constexpr int INST = FF_DK / CW;
  using TsmB = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/true,
      /*Trans=*/false, /*InstNum=*/INST>;

  cute::clear(hq_acc);
  cute::clear(hk_acc);
  cute::Tensor rHq = cute::recast<float>(hq_acc);
  cute::Tensor rHk = cute::recast<float>(hk_acc);
  CUTE_UNROLL
  for (int kt = 0; kt < FF_DK / 16; ++kt) {
    const int cube = kt / KT_PER_CUBE;
    const int col_in_cube = (kt % KT_PER_CUBE) * 16;
    // Pull the independent Q/K B fragments into registers together.  Besides
    // increasing TSM-load-to-MMA distance, alternating the two accumulator
    // chains below gives the scheduler useful work between RAW-dependent MMA
    // updates without changing the arithmetic order within either chain.
    uint32_t qfrag[FF_CS / 16][4];
    uint32_t kfrag[FF_CS / 16][4];
    CUTE_UNROLL
    for (int n = 0; n < FF_CS / 16; ++n) {
      const int row = n * 16;
      TsmB::copy(reinterpret_cast<int*>(qfrag[n]),
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Qswz)),
                 col_in_cube, row, cube, 0);
      TsmB::copy(reinterpret_cast<int*>(kfrag[n]),
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Kswz)),
                 col_in_cube, row, cube, 0);
    }
    CUTE_UNROLL
    for (int n = 0; n < FF_CS / 16; ++n) {
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(hq_acc); ++m) {
        fq::ppu_mma_impl<Element>(
            rHq(0,m,n), rHq(1,m,n), rHq(2,m,n), rHq(3,m,n),
            rHq(4,m,n), rHq(5,m,n), rHq(6,m,n), rHq(7,m,n),
            h_regs[kt][m][0], h_regs[kt][m][1],
            h_regs[kt][m][2], h_regs[kt][m][3],
            qfrag[n][0], qfrag[n][1], qfrag[n][2], qfrag[n][3],
            rHq(0,m,n), rHq(1,m,n), rHq(2,m,n), rHq(3,m,n),
            rHq(4,m,n), rHq(5,m,n), rHq(6,m,n), rHq(7,m,n));
        fq::ppu_mma_impl<Element>(
            rHk(0,m,n), rHk(1,m,n), rHk(2,m,n), rHk(3,m,n),
            rHk(4,m,n), rHk(5,m,n), rHk(6,m,n), rHk(7,m,n),
            h_regs[kt][m][0], h_regs[kt][m][1],
            h_regs[kt][m][2], h_regs[kt][m][3],
            kfrag[n][0], kfrag[n][1], kfrag[n][2], kfrag[n][3],
            rHk(0,m,n), rHk(1,m,n), rHk(2,m,n), rHk(3,m,n),
            rHk(4,m,n), rHk(5,m,n), rHk(6,m,n), rHk(7,m,n));
      }
    }
  }
#endif
}

template <int N, int K, typename Element, typename TC>
__device__ __forceinline__ void ff_gemm_regA_swzlB_m4(
    const uint32_t a_regs[K / 16][1][4], const Element* Bswz, int tidx,
    TC& tC, bool clear_acc = true) {
  static_assert(N % 16 == 0);
#if defined(FLASHQLA_ARCH_100)
  using Mma = FfMmaM4T<Element>;
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  // Ag/Pg are produced on chip.  Publish them in the software-swizzled
  // row-major layout used by the validated tc01 fallback instead of treating
  // them as untouched AIU-load output.
  auto sB = fq::make_smem_swzA_cs32<N, K>(Bswz);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<FF_DVH, K>{});
  auto rA_full = thr_mma.partition_fragment_A(dummyA);
  auto rB_full = thr_mma.partition_fragment_B(sB);
  auto scpB = cute::make_tiled_copy_B(fq::SmemCopyAtomT<Element>{}, mma);
  auto tcpB = scpB.get_thread_slice(tidx);
  auto tBsB = tcpB.partition_S(sB);
  auto tBrB = tcpB.retile_D(rB_full);
  CUTE_UNROLL
  for (int kt = 0; kt < K / 16; ++kt) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(rA_full); ++m) {
      auto rA = cute::recast<uint32_t>(rA_full(cute::_, m, kt));
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) rA(i) = a_regs[kt][m][i];
    }
  }
  if (clear_acc) cute::clear(tC);
  CUTE_UNROLL
  for (int kt = 0; kt < K / 16; ++kt) {
    cute::copy(scpB, tBsB(cute::_, cute::_, kt),
               tBrB(cute::_, cute::_, kt));
    cute::gemm(mma, rA_full(cute::_, cute::_, kt),
               rB_full(cute::_, cute::_, kt), tC);
  }
#else
  (void)tidx;
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;
  constexpr int INST = K / CW;
  // Swap=true only exchanges the atom's lbo/sbo operands, which for 16-bit
  // elements swaps the two middle 8x8 sub-tiles of the x4 load: the fragment
  // lands in TN B-operand order directly, so the MMA below needs no 1<->2
  // register permutation (and therefore no copies).
  using TsmB = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/true,
      /*Trans=*/false, /*InstNum=*/INST>;

  if (clear_acc) cute::clear(tC);
  cute::Tensor rC = cute::recast<float>(tC);
  // K is only 32 in the register chain.  Preloading its two B slices makes the
  // TSM loads independent of the immediately following MMA while adding only
  // eight uint32 registers over the one-slice schedule.
  uint32_t bfrag[K / 16][N / 16][4];
  CUTE_UNROLL
  for (int kt = 0; kt < K / 16; ++kt) {
    const int cube = kt / KT_PER_CUBE;
    const int col_in_cube = (kt % KT_PER_CUBE) * 16;
    CUTE_UNROLL
    for (int n = 0; n < N / 16; ++n) {
      const int row = n * 16;
      TsmB::copy(reinterpret_cast<int*>(bfrag[kt][n]),
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Bswz)),
                 col_in_cube, row, cube, 0);
    }
  }
  CUTE_UNROLL
  for (int kt = 0; kt < K / 16; ++kt) {
    CUTE_UNROLL
    for (int n = 0; n < N / 16; ++n) {
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tC); ++m) {
        fq::ppu_mma_impl<Element>(
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n),
            a_regs[kt][m][0], a_regs[kt][m][1],
            a_regs[kt][m][2], a_regs[kt][m][3],
            bfrag[kt][n][0], bfrag[kt][n][1],
            bfrag[kt][n][2], bfrag[kt][n][3],
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n));
      }
    }
  }
#endif
}

// GEMM6^T: h^T += (g_rev*Vd)^T @ K.  The physical K buffer is [CS,DK]; a
// transposed swizzle read exposes logical [DK,CS].  Swap=true flips the atom's
// lbo/sbo on top of Trans, which lands the fragment in TN B-operand order
// directly instead of the loader's A-register order.
template <typename Element, typename TC>
__device__ __forceinline__ void ff_gemm_regA_ktB_m4_acc(
    const uint32_t a_regs[FF_CS / 16][1][4], const Element* Kswz, int tidx,
    TC& ht_acc) {
#if defined(FLASHQLA_ARCH_100)
  using TsmTransB = fq::PpuTsmLdSwzl<
      typename fq::ElementTraits<Element>::CuteType, 32, 64, /*Swap=*/false,
      /*Trans=*/true, /*InstNum=*/2>;
  using Mma = FfMmaM4T<Element>;
  Mma mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<FF_DVH, FF_CS>{});
  auto dummyB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<FF_DK, FF_CS>{});
  auto rA_full = thr_mma.partition_fragment_A(dummyA);
  auto rB_full = thr_mma.partition_fragment_B(dummyB);
  CUTE_UNROLL
  for (int kt = 0; kt < FF_CS / 16; ++kt) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(rA_full); ++m) {
      auto rA = cute::recast<uint32_t>(rA_full(cute::_, m, kt));
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) rA(i) = a_regs[kt][m][i];
    }
    CUTE_UNROLL
    for (int n = 0; n < FF_DK / 16; ++n) {
      const int dk = n * 16;
      uint32_t bfrag[4] = {0, 0, 0, 0};
      TsmTransB::copy(
          reinterpret_cast<int*>(bfrag),
          reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Kswz)),
          kt * 16, dk % 64, dk / 64, 0);
      auto rB = cute::recast<uint32_t>(rB_full(cute::_, n, kt));
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) rB(i) = bfrag[i];
    }
    cute::gemm(mma, rA_full(cute::_, cute::_, kt),
               rB_full(cute::_, cute::_, kt), ht_acc);
  }
#else
  (void)tidx;
  using TsmTransB = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, 32, 32, /*Swap=*/true,
      /*Trans=*/true, /*InstNum=*/1>;

  cute::Tensor rH = cute::recast<float>(ht_acc);
  CUTE_UNROLL
  for (int kt = 0; kt < FF_CS / 16; ++kt) {
    CUTE_UNROLL
    // Four independent DK output tiles amortize each TSM load burst and leave
    // more instructions between dependent updates of the same h accumulator.
    // A depth of four is deliberate: depth eight raises the live range by 24
    // more registers and risks dropping the current two-block residency.
    for (int n0 = 0; n0 < FF_DK / 16; n0 += 4) {
      uint32_t bfrag[4][4];
      CUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        const int dk = (n0 + j) * 16;
        const int cube = dk / 32;
        const int row_in_cube = dk % 32;
        TsmTransB::copy(
            reinterpret_cast<int*>(bfrag[j]),
            reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Kswz)),
            kt * 16, row_in_cube, 0, cube);
      }
      CUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        const int n = n0 + j;
        CUTE_UNROLL
        for (int m = 0; m < cute::size<1>(ht_acc); ++m) {
          fq::ppu_mma_impl<Element>(
              rH(0,m,n), rH(1,m,n), rH(2,m,n), rH(3,m,n),
              rH(4,m,n), rH(5,m,n), rH(6,m,n), rH(7,m,n),
              a_regs[kt][m][0], a_regs[kt][m][1],
              a_regs[kt][m][2], a_regs[kt][m][3],
              bfrag[j][0], bfrag[j][1], bfrag[j][2], bfrag[j][3],
              rH(0,m,n), rH(1,m,n), rH(2,m,n), rH(3,m,n),
              rH(4,m,n), rH(5,m,n), rH(6,m,n), rH(7,m,n));
        }
      }
    }
  }
#endif
}

// ---- AIU plain (linear) 2D b16 async load + commit/wait (ports .promoted) ----
// 2D AIU load. `dim_c` is the contiguous width, `dim_w` the number of rows that
// are actually valid in global memory.
//
// `cube_w` is the tile height the engine writes into smem. It defaults to dim_w,
// which transfers exactly the valid rows and leaves anything past them
// untouched. Passing a larger value makes the tile overhang the tensor bound and
// the `padz` in the mnemonic zero-fills the overhang -- the same mechanism that
// already keeps K/Q clean on a partial chunk, where the cube atom's CUBE_H is a
// fixed 32 while AiuDesc::dim_h is the valid row count. Use it instead of
// zeroing smem by hand: the fill costs nothing and overlaps the transfer,
// whereas a manual memset can only run after the wait.
__forceinline__ __device__ void ff_aiu_load_2d_b16(
    void* smem_ptr, const void* gmem_ptr,
    int dim_c, int dim_w, int64_t stride_w_bytes, int cube_w = -1) {
#if defined(FLASHQLA_ARCH_150)
  if (cube_w < 0) cube_w = dim_w;
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
      " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
      :: "r"(smem_addr), "l"(gmem_ptr),
         "r"(dim_c), "r"(dim_w), "r"(1),      // tensor bounds: dim_w valid rows
         "r"(dim_c), "r"(cube_w), "r"(1),     // tile: cube_w rows, overhang padz'd
         "l"(stride_w_bytes), "l"(int64_t(0)),
         "r"(0), "r"(0), "r"(0),
         "r"(0)
      : "memory");
#elif defined(FLASHQLA_ARCH_100)
  // tc01: this helper is only used for the tiny g/b scalar tiles now (K/V/Q/A
  // go through the actlize AIU pipeline). g/b are fp32[rows] gathered with a
  // per-token stride and stored packed into smem. dim_c=2 means "2 b16 = 1
  // fp32 per row". A single-thread synchronous fp32 copy is simplest and
  // provably correct (the issuing thread is tx==0/lane==0; the subsequent
  // ff_aiu_wait_all + __syncthreads make the write visible before any reader).
  (void)cube_w;
  float* dst = reinterpret_cast<float*>(smem_ptr);
  const float* src = reinterpret_cast<const float*>(gmem_ptr);
  const int stride_floats = static_cast<int>(stride_w_bytes >> 2);
  for (int i = 0; i < dim_w; ++i) dst[i] = src[static_cast<int64_t>(i) * stride_floats];
  (void)dim_c;
#else
  (void)smem_ptr; (void)gmem_ptr; (void)dim_c; (void)dim_w; (void)stride_w_bytes;
  (void)cube_w;
#endif
}
__forceinline__ __device__ void ff_aiu_commit() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}
__forceinline__ __device__ void ff_aiu_wait_all() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}
// wait_group<N>: drain until ≤ N commit groups remain in this thread's FIFO.
template <int N>
__forceinline__ __device__ void ff_aiu_wait_group() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// Named group barrier: syncs only the FF_WG threads of one DV group (bar id = gid+1).
__forceinline__ __device__ void ff_group_bar(int gid) {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.bar.sync %0, %1;\n" :: "r"(gid + 1), "n"(FF_WG) : "memory");
#else
  (void)gid;
  __syncthreads();
#endif
}

// AIU .swzl 2D write: K or Q [rows, DK] → 4-cube swizzled smem tile (.padz fills partial).
#if defined(FLASHQLA_ARCH_100)
// tc01 forward decls: the actual loaders are defined after the traits block
// below (they need FfQKVTraitsTc01 / FfATraitsTc01).
template <typename Element>
__forceinline__ __device__ void ff_load_qkv_tile_tc01(
    Element* dst, const Element* src, int row_stride_elems, int rows);
template <typename Element>
__forceinline__ __device__ void ff_load_qk_tile_tc01(
    Element* dst, const Element* src, int row_stride_elems, int rows);
template <typename Element>
__forceinline__ __device__ void ff_load_a_tile_tc01(
    Element* dst, const Element* src, int row_stride_elems, int rows);
#endif
template <typename Element>
__forceinline__ __device__ void ff_aiu_swzl_write_kq_cs32(
    Element* smem, const Element* gmem, int row_stride_elems, int rows) {
#if defined(FLASHQLA_ARCH_100)
  // tc01: Q/K land SWIZZLED in the AIU-native cube-major SmemLayout (FfQKTraitsTc01)
  // via the closed AIU_LOAD(Swzl=true)/TSM_LD_SWZL pair; the downstream GEMMs read
  // them back through the matching swizzled TSM_LD_SWZL atom. row_stride_elems is
  // the gmem row pitch (Hg*FF_DK), FF_DK the channel width.
  ff_load_qk_tile_tc01<Element>(smem, gmem, row_stride_elems, rows);
#elif defined(__HGGC_ARCH__)
  using AiuWriteCube = flashqla::PpuAiuLoad<
      32 * 32 * 16, typename fq::ElementTraits<Element>::CuteType, /*Trans=*/false,
      /*CUBE_H=*/32, /*CUBE_W=*/32, /*Swzl=*/true>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t*>(gmem), /*h=*/rows, /*w=*/FF_DK,
            /*stride_w=*/row_stride_elems);
  CUTE_UNROLL
  for (int cube = 0; cube < FF_DK / 32; ++cube) {
    uint32_t dst_shared = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem + cube * 32 * 32));  // CTA-relative .shared addr
    AiuWriteCube::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst_shared)),
                       reinterpret_cast<const void*>(gmem), desc,
                       /*coord_w=*/cube * 32, /*coord_h=*/0);
  }
#else
  (void)smem; (void)gmem; (void)row_stride_elems; (void)rows;
#endif
}

// Issue one chunk as a single AIU commit group (tx==0 path).  The state-only
// specialization skips Q because its output half reloads Q independently.
template <typename Element>
__device__ __forceinline__ void ff_issue_chunk_aiu(
    Element* Kd, Element* Vd, Element* Qd, Element* Ad,
    const Element* K_src, const Element* V_src,
    const Element* Q_src, const Element* A_src,
    int H, int Hg, int64_t v_token_stride, int actual = FF_CS,
    float* gd = nullptr, float* bd = nullptr,
    const float* g_src = nullptr, const float* b_src = nullptr) {
  ff_aiu_swzl_write_kq_cs32<Element>(
      Qd, Q_src, /*row_stride=*/Hg * FF_DK, /*rows=*/actual);
  ff_aiu_swzl_write_kq_cs32<Element>(
      Kd, K_src, /*row_stride=*/Hg * FF_DK, /*rows=*/actual);
  ff_aiu_load_2d_b16(reinterpret_cast<Element*>(gd),
                     reinterpret_cast<const __half*>(g_src),
                     /*dim_c=*/2, /*dim_w=*/actual, /*stride=*/(int64_t)H * 4);
  // V and A get cube_w=FF_CS so the engine zero-fills rows [actual, FF_CS) on a
  // partial chunk. V needs it: its tail rows are read unmasked by GEMM1's
  // `W = V - g_exp*U` epilogue, so stale bits (a NaN pattern in untouched smem)
  // would reach W and then survive GEMM2 as 0 * NaN. A does not strictly need it
  // -- its reader assigns a literal 0 outside [0,actual) -- but padding it costs
  // nothing and removes the dependence on that mask staying in place.
  // g/b deliberately keep the default: their reader substitutes g_last_w, not 0.
  //
  // Verified by planting NaN in rows [actual, FF_CS) just before this issue: the
  // tail cases in e2e_acc_full.py Section 5 fail without cube_w and pass with it,
  // i.e. the engine really does overwrite the overhang rather than skip them.
#if defined(FLASHQLA_ARCH_100)
  // tc01: V/A land in AIU-native SmemLayout (FfQKVTraitsTc01 / FfATraitsTc01),
  // read back by the paired TSM_LD_SWZL atom in the GEMMs. .padz zero-fills the
  // [actual, FF_CS) overhang exactly like the tc02 cube_w path.
  ff_load_qkv_tile_tc01<Element>(Vd, V_src, v_token_stride, actual);
  ff_load_a_tile_tc01<Element>(Ad, A_src, H * FF_CS, actual);
#else
  ff_aiu_load_2d_b16(
      Vd, V_src, FF_DV, actual, v_token_stride * 2, /*cube_w=*/FF_CS);
  ff_aiu_load_2d_b16(Ad, A_src, FF_CS, actual, (int64_t)H * FF_CS * 2, /*cube_w=*/FF_CS);
#endif
  ff_aiu_load_2d_b16(reinterpret_cast<Element*>(bd),
                     reinterpret_cast<const __half*>(b_src),
                     /*dim_c=*/2, /*dim_w=*/actual, /*stride=*/(int64_t)H * 4);
  ff_aiu_commit();
}

// Smem element counts (needed by preload helper below and smem layout)
constexpr int FF_KELEMS  = FF_CS * FF_DK;
constexpr int FF_VELEMS  = FF_CS * FF_DV;
constexpr int FF_AELEMS  = FF_CS * FF_CS;

#if defined(FLASHQLA_ARCH_100)
constexpr int FF_NBUF = FF_REG_CHAIN ? 2 : 1;
#else
constexpr int FF_NBUF = 2;
#endif

// ---- Multi-warp split AIU preload helper ----
template <typename Element, int H, int Hg, int NGRP>
__device__ __forceinline__ void ff_preload_chunks_split_aiu(
    int n_chunks, int start, int first_slot, int seq_end,
    int warp, int lane,
    Element* Q_all, Element* K_all, Element* V_all, Element* A_all,
    float* g_all, float* b_all,
    const Element* q, const Element* k, const Element* v, const Element* a,
    const float* g, const float* beta,
    int kv_head_idx, int head_idx, int64_t v_token_stride) {
  for (int i = 0; i < n_chunks; ++i) {
    const int s = start + i * FF_CS;
    if (min(FF_CS, seq_end - s) < FF_CS) return;
#if defined(FLASHQLA_ARCH_100) && !FF_REG_CHAIN
    const int slot = 0;
#else
    const int slot = (first_slot + i) % FF_NBUF;
#endif
    const Element* Qs = q + (int64_t)s * Hg * FF_DK + kv_head_idx * FF_DK;
    const Element* Ks = k + (int64_t)s * Hg * FF_DK + kv_head_idx * FF_DK;
    const Element* Vs =
        v + (int64_t)s * v_token_stride + head_idx * FF_DV;
    const Element* As = a + (int64_t)s * H  * FF_CS + head_idx    * FF_CS;
    const float*  gs = g    + (int64_t)s * H + head_idx;
    const float*  bs = beta + (int64_t)s * H + head_idx;
#if defined(FLASHQLA_ARCH_100) && FF_TC01_DV_SPLIT
    if (warp == 0 && lane == 0) {
      ff_aiu_swzl_write_kq_cs32<Element>(
          Q_all + slot * FF_KELEMS, Qs, Hg * FF_DK, FF_CS);
      ff_aiu_swzl_write_kq_cs32<Element>(
          K_all + slot * FF_KELEMS, Ks, Hg * FF_DK, FF_CS);
    }
    if (warp == 1 && lane == 0) {
      ff_load_qkv_tile_tc01<Element>(
          V_all + slot * FF_VELEMS, Vs, v_token_stride, FF_CS);
      ff_load_a_tile_tc01<Element>(A_all + slot * FF_AELEMS, As, H * FF_CS, FF_CS);
    }
    if (warp == 2)
      g_all[slot * FF_CS + lane] = gs[(int64_t)lane * H];
    if (warp == 3)
      b_all[slot * FF_CS + lane] = bs[(int64_t)lane * H];
#else
    constexpr int k_warp = NGRP == 1 ? 1 : 4;
    constexpr int v_warp = NGRP == 1 ? 2 : 1;
    constexpr int a_warp = NGRP == 1 ? 2 : 5;
    constexpr int g_warp = NGRP == 1 ? 3 : 2;
    constexpr int b_warp = NGRP == 1 ? 3 : 6;
    if (warp == 0 && lane == 0)
      ff_aiu_swzl_write_kq_cs32<Element>(
          Q_all + slot * FF_KELEMS, Qs, Hg * FF_DK, FF_CS);
    if (warp == k_warp && lane == 0)
      ff_aiu_swzl_write_kq_cs32<Element>(K_all + slot * FF_KELEMS, Ks, Hg * FF_DK, FF_CS);
    if (warp == v_warp && lane == 0)
#if defined(FLASHQLA_ARCH_100)
      ff_load_qkv_tile_tc01<Element>(
          V_all + slot * FF_VELEMS, Vs, v_token_stride, FF_CS);
#else
      ff_aiu_load_2d_b16(V_all + slot * FF_VELEMS, Vs, FF_DV, FF_CS,
                         v_token_stride * 2);
#endif
    if (warp == a_warp && lane == 0)
#if defined(FLASHQLA_ARCH_100)
      ff_load_a_tile_tc01<Element>(A_all + slot * FF_AELEMS, As, H * FF_CS, FF_CS);
#else
      ff_aiu_load_2d_b16(A_all + slot * FF_AELEMS, As, FF_CS, FF_CS,
                         (int64_t)H * FF_CS * 2);
#endif
    if (warp == g_warp)
#if defined(FLASHQLA_ARCH_100)
      g_all[slot * FF_CS + lane] = gs[(int64_t)lane * H];
#else
    if (lane == 0)
      ff_aiu_load_2d_b16(reinterpret_cast<Element*>(g_all + slot * FF_CS),
                         reinterpret_cast<const __half*>(gs), 2, FF_CS,
                         (int64_t)H * 4);
#endif
    if (warp == b_warp)
#if defined(FLASHQLA_ARCH_100)
      b_all[slot * FF_CS + lane] = bs[(int64_t)lane * H];
#else
    if (lane == 0)
      ff_aiu_load_2d_b16(reinterpret_cast<Element*>(b_all + slot * FF_CS),
                         reinterpret_cast<const __half*>(bs), 2, FF_CS,
                         (int64_t)H * 4);
#endif
#endif
    ff_aiu_commit();
  }
}

// ---- Architecture-specific shared-memory layout ----
#if defined(FLASHQLA_ARCH_100)
// Shift successive DV rows across tc01 TSM bank groups.  The logical state
// tile is still [DVH, DK]; eight padding halfs are never read or written.
constexpr int FF_HLD     = FF_DK + 8;
// W/Vd are physically [DVH, CS].  Their 32-half rows otherwise repeat every
// 64 B, so use the same bank-rotation distance that won for hstmT.
constexpr int FF_WLD     = FF_CS + 8;
// Pg is logically [CS, CS] and has the same 32-half row alias.
constexpr int FF_PGLD    = FF_CS + 8;
#else
constexpr int FF_HLD     = FF_DK;
constexpr int FF_WLD     = FF_CS;
constexpr int FF_PGLD    = FF_CS;
#endif
constexpr int FF_HELEMS  = FF_DVH * FF_HLD;
constexpr int FF_WELEMS  = FF_DVH * FF_WLD;
constexpr int FF_PGELEMS = FF_CS * FF_PGLD;
template <int NGRP, int DVH = FF_DVH,
          bool EnableTc02RegChainPrune = false>
struct FfSmemLayout {
  static constexpr bool IsTc01 =
#if defined(FLASHQLA_ARCH_100)
      true;
#else
      false;
#endif
  // Register-chain kernels keep h/W/Vd/gVd in registers. Ag/Pg and the
  // g-expansion data are independent of the DV-half group and can be shared.
  static constexpr bool PruneRegChain = FF_REG_CHAIN &&
      (IsTc01 || EnableTc02RegChainPrune);
  static constexpr int SharedGroups = PruneRegChain ? 1 : NGRP;
  static constexpr int AgBytes =
      PruneRegChain ? FF_CS * FF_CS * 2
                    : (IsTc01 && FF_VD_REG_B && FF_GVD_REG_B
                           ? 0 : NGRP * FF_CS * FF_CS * 2);
  static constexpr int HBytes =
      PruneRegChain ? 0 : NGRP * DVH * FF_HLD * 2;
  static constexpr int WBytes =
      PruneRegChain ? 0 : NGRP * DVH * FF_WLD * 2;
  static constexpr int VdBytes =
      PruneRegChain || (IsTc01 && FF_VD_REG_B)
          ? 0 : NGRP * DVH * FF_WLD * 2;
  static constexpr int GVdBytes =
      PruneRegChain || (IsTc01 && FF_GEMM_REORDER && FF_GVD_REG_B)
          ? 0 : NGRP * DVH * FF_WLD * 2;
  static constexpr int PgBytes =
      PruneRegChain ? FF_CS * FF_CS * 2
                    : NGRP * FF_PGELEMS * 2;
  static constexpr int K       = 0;
  static constexpr int V       = K + FF_NBUF * FF_KELEMS * 2;
  static constexpr int Q       = V + FF_NBUF * FF_VELEMS * 2;
  static constexpr int A       = Q + FF_NBUF * FF_KELEMS * 2;
  static constexpr int KT      = A + FF_NBUF * FF_AELEMS * 2;
  static constexpr int AGSWZ   = KT;
  static constexpr int HSTMT   = AGSWZ + AgBytes;
  static constexpr int WSWZ    = HSTMT + HBytes;
  static constexpr int VDSWZ   = WSWZ + WBytes;
  static constexpr int GVDSWZ  = VDSWZ + VdBytes;
  static constexpr int PG      = GVDSWZ + GVdBytes;
  static constexpr int G       = PG + PgBytes;
  static constexpr int B       = G + FF_NBUF * FF_CS * 4;
  static constexpr int GEXP    = B + FF_NBUF * FF_CS * 4;
  static constexpr int GREV    = GEXP + SharedGroups * FF_CS * 4;
  static constexpr int GLOG    = GREV + SharedGroups * FF_CS * 4;
  static constexpr int TOTAL   = GLOG + SharedGroups * FF_CS * 4;
};

// nvcc emits both ppu001 and ppu0015 device images for the shared sm_80
// gencode.  The host pass is compiled only once, so it cannot use
// __HGGC_ARCH__ to size a launch.  Keep both layout totals available and pick
// the one matching the active device at runtime.
constexpr int ff_total_bytes_for_arch(bool arch100, int dvh = FF_DVH) {
  const int ngrp = arch100 ? FF_NGRP_TC01 : FF_NGRP_TC02;
  const bool prune_tc01_reg_chain = arch100 && FF_REG_CHAIN;
  const int shared_groups = prune_tc01_reg_chain ? 1 : ngrp;
  const int nbuf = prune_tc01_reg_chain ? 2 : (arch100 ? 1 : 2);
  const int hld = arch100 ? FF_DK + 8 : FF_DK;
  const int wld = arch100 ? FF_CS + 8 : FF_CS;
  const int pgld = arch100 ? FF_CS + 8 : FF_CS;
  const int helems = dvh * hld;
  const int welems = dvh * wld;
  const int pgelems = FF_CS * pgld;
  const int vd_reg_b = arch100 ? FF_VD_REG_B_TC01 : FF_VD_REG_B_TC02;
  const int agswz_bytes = prune_tc01_reg_chain
      ? FF_CS * FF_CS * 2
      : (arch100 && vd_reg_b && FF_GVD_REG_B
             ? 0 : ngrp * FF_CS * FF_CS * 2);
  const int h_bytes = prune_tc01_reg_chain ? 0 : ngrp * helems * 2;
  const int w_bytes = prune_tc01_reg_chain ? 0 : ngrp * welems * 2;
  const int vdswz_bytes = prune_tc01_reg_chain || (arch100 && vd_reg_b)
      ? 0 : ngrp * welems * 2;
  const int gvdswz_bytes = prune_tc01_reg_chain ||
      (arch100 && FF_GEMM_REORDER && FF_GVD_REG_B)
      ? 0 : ngrp * welems * 2;
  const int pg_bytes = prune_tc01_reg_chain
      ? FF_CS * FF_CS * 2 : ngrp * pgelems * 2;
  return nbuf * FF_KELEMS * 2
       + nbuf * FF_VELEMS * 2
       + nbuf * FF_KELEMS * 2
       + nbuf * FF_AELEMS * 2
       + agswz_bytes
       + h_bytes
       + w_bytes
       + vdswz_bytes
       + gvdswz_bytes
       + pg_bytes
       + 2 * nbuf * FF_CS * 4
       + 3 * shared_groups * FF_CS * 4;
}

constexpr int FF_TOTAL_TC01 = ff_total_bytes_for_arch(true);
constexpr int FF_TOTAL_TC01_D32 = ff_total_bytes_for_arch(true, 32);
constexpr int FF_TOTAL_TC02 = FfSmemLayout<FF_NGRP_TC02>::TOTAL;
constexpr int FF_TOTAL_TC02_PRUNED =
    FfSmemLayout<FF_NGRP_TC02, FF_DVH, true>::TOTAL;
#if defined(FLASHQLA_ARCH_100)
static_assert(FfSmemLayout<FF_NGRP_TC01>::TOTAL == FF_TOTAL_TC01);
static_assert(FfSmemLayout<FF_NGRP_TC01, 32>::TOTAL == FF_TOTAL_TC01_D32);
#elif defined(FLASHQLA_ARCH_150)
static_assert(FF_TOTAL_TC02 == 120064);
static_assert(FfSmemLayout<1, FF_DVH, true>::TOTAL == 58240);
static_assert(FF_TOTAL_TC02_PRUNED == 58240);
#endif

// ============================================================================
// PPU1.0/1.1 (tc01) Q/K/V/A pipeline traits + gmem->smem loaders + views.
//
// Third and final application of the closed actlize pair established in
// kkt_solve (milestone 1) and prepare_h (milestone 2):
//   PPU0010_AIU_LOAD (gmem->smem)  +  PPU0010_TSM_LD_SWZL (smem->reg)
// with an AIU-native cube-major SmemLayoutAtom<_8, AiuContElemSize>.
//
//   Q/K/V: BLK_MN=FF_CS=32, BLK_K=FF_DK=128 -> cube_w=64, InstNum=2
//   A:     BLK_MN=FF_CS=32, BLK_K=FF_CS=32  -> cube_w=32, InstNum=1
//
// The smem footprint per buffer is unchanged (tile_to_shape adds no padding
// beyond the atom), so the FF_{K,V,Q,A} double-buffer offsets above still
// describe the same allocations; only the internal element order differs.
// ============================================================================
#if defined(FLASHQLA_ARCH_100)
template <typename Element, int BLK_MN_, int BLK_K_, bool Swzl_ = false>
struct FfBufTraitsTc01 {
  using CuteElem = typename fq::ElementTraits<Element>::CuteType;
  static constexpr int BLK_MN = BLK_MN_;
  static constexpr int BLK_K  = BLK_K_;

  static constexpr int BlockContSize   = BLK_K * static_cast<int>(sizeof(CuteElem));
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  static constexpr int AiuContElemSize = AiuContByteSize / static_cast<int>(sizeof(CuteElem));
  static constexpr int InstNum         = BLK_K / AiuContElemSize;
  static constexpr int BitsPerAIU      = BLK_MN * AiuContElemSize *
                                         static_cast<int>(sizeof(CuteElem)) * 8;

  using CopyInst = cute::PPU0010_AIU_LOAD<
      cute::C<BitsPerAIU>, CuteElem, /*Trans=*/false, /*Swzl=*/Swzl_>;

  using GmemTiledCopy = decltype(cute::make_tiled_copy(
      cute::Copy_Atom<CopyInst, CuteElem>{},
      cute::Layout<cute::Shape <cute::_1, cute::_1>,
                   cute::Stride<cute::_1, cute::_1>>{},
      cute::Layout<cute::Shape<cute::Int<BLK_MN>, cute::Int<AiuContElemSize>>>{}));

  using SmemLayoutAtom = cute::Layout<
      cute::Shape<cute::_8, cute::Int<AiuContElemSize>>,
      cute::Stride<cute::Int<AiuContElemSize>, cute::_1>>;
  using SmemLayout = decltype(cute::tile_to_shape(
      SmemLayoutAtom{},
      cute::make_shape(cute::Int<BLK_MN>{}, cute::Int<BLK_K>{})));

  using SmemCopyOp = cute::PPU0010_TSM_LD_SWZL<
      CuteElem, BLK_MN, AiuContElemSize, /*Swap=*/false, /*Trans=*/false, InstNum>;
  using SmemCopyAtom = cute::Copy_Atom<SmemCopyOp, CuteElem>;
};

template <typename Element>
using FfQKVTraitsTc01 = FfBufTraitsTc01<Element, FF_CS, FF_DK>;
template <typename Element>
using FfATraitsTc01   = FfBufTraitsTc01<Element, FF_CS, FF_CS>;
// Q and K are consumed by TSM_LD_SWZL (GEMM1/GEMM3/GEMM4 non-trans + GEMM6
// Trans=true), so their AIU write MUST be swizzled to match the reader. V stays
// linear (read elementwise via ff_qkv_view_tc01 for the W = V - g_exp*U epilogue).
template <typename Element>
using FfQKTraitsTc01  = FfBufTraitsTc01<Element, FF_CS, FF_DK, /*Swzl=*/true>;

// Issue one AIU-driven load of a [rows<=BLK_MN, BLK_K] gmem tile. .padz
// zero-fills rows past `rows`. Caller must be a single thread (AIU needs one
// issuer) and must commit afterwards.
template <typename Traits, typename Element>
__forceinline__ __device__ void ff_load_tile_tc01_(
    Element* dst_smem, const Element* src_gmem, int row_stride_elems, int rows) {
  using CuteElem = typename Traits::CuteElem;
  auto gA = cute::make_mix_tensor(
      cute::make_gmem_ptr(reinterpret_cast<const CuteElem*>(src_gmem)),
      cute::make_layout(
          cute::make_shape(cute::Int<Traits::BLK_MN>{}, cute::Int<Traits::BLK_K>{}),
          cute::make_counting_stride<2, 1, 1>()));
  auto sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElem*>(dst_smem)),
      typename Traits::SmemLayout{});
  typename Traits::GmemTiledCopy tiled_copy;
  tiled_copy.desc_.template init<CuteElem, /*Trans=*/false,
                                  Traits::BLK_MN, Traits::BLK_K>(
      reinterpret_cast<const uint8_t*>(src_gmem),
      /*MN=*/rows, /*K=*/Traits::BLK_K,
      cute::make_stride(row_stride_elems, cute::_1{}));
  auto thr = tiled_copy.get_slice(0);
  auto tAgA = thr.partition_S(gA);
  auto tAsA = thr.partition_D(sA);
  cute::copy(tiled_copy, tAgA, tAsA);
}

template <typename Element>
__forceinline__ __device__ void ff_load_qkv_tile_tc01(
    Element* dst, const Element* src, int row_stride_elems, int rows) {
  ff_load_tile_tc01_<FfQKVTraitsTc01<Element>, Element>(dst, src, row_stride_elems, rows);
}
template <typename Element>
__forceinline__ __device__ void ff_load_qk_tile_tc01(
    Element* dst, const Element* src, int row_stride_elems, int rows) {
  ff_load_tile_tc01_<FfQKTraitsTc01<Element>, Element>(dst, src, row_stride_elems, rows);
}
template <typename Element>
__forceinline__ __device__ void ff_load_a_tile_tc01(
    Element* dst, const Element* src, int row_stride_elems, int rows) {
  ff_load_tile_tc01_<FfATraitsTc01<Element>, Element>(dst, src, row_stride_elems, rows);
}

// SmemLayout tensor views over the AIU-native buffers, for elementwise readers
// and for the inlined GEMM A-operand partitioning.
template <typename Element>
CUTE_DEVICE auto ff_qkv_view_tc01(Element* buf) {
  using Traits = FfQKVTraitsTc01<Element>;
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename Traits::CuteElem*>(buf)),
      typename Traits::SmemLayout{});
}
template <typename Element>
CUTE_DEVICE auto ff_a_view_tc01(Element* buf) {
  using Traits = FfATraitsTc01<Element>;
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename Traits::CuteElem*>(buf)),
      typename Traits::SmemLayout{});
}

// Inlined "A from AIU-native smem, B from packed smem" GEMM. Mirrors the
// milestone-1/2 bodies: A fragments come through the paired TSM_LD_SWZL atom
// (its Copy_Traits reads src.data().coord_, hence make_mix_tensor_like), B
// fragments through the ordinary ldmatrix atom. Clear controls whether tC is
// zeroed first (matching the tc02 helper being replaced at each call site).
template <bool Clear, typename ATraits, typename SmemA, typename SmemB,
          typename TC, typename Element>
CUTE_DEVICE void ff_gemm_aiuA_ss_tc01(
    SmemA const& sA, SmemB const& sB, int tidx, TC& tC) {
  using cute::_;
  fq::TiledMmaM32T<Element> tmma;
  auto thr_m = tmma.get_thread_slice(tidx);

  auto rA_full = thr_m.partition_fragment_A(sA);
  auto smem_cp_A = cute::make_tiled_copy_A(typename ATraits::SmemCopyAtom{}, tmma);
  auto thr_cp_A = smem_cp_A.get_thread_slice(tidx);
  auto tAsA = thr_cp_A.partition_S(cute::make_mix_tensor_like(sA));
  auto tArA = thr_cp_A.retile_D(rA_full);

  auto rB_full = thr_m.partition_fragment_B(sB);
  auto smem_cp_B = cute::make_tiled_copy_B(fq::SmemCopyAtomT<Element>{}, tmma);
  auto thr_cp_B = smem_cp_B.get_thread_slice(tidx);
  auto tBsB = thr_cp_B.partition_S(sB);
  auto tBrB = thr_cp_B.retile_D(rB_full);

  if constexpr (Clear) cute::clear(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_cp_A, tAsA(_,_,k), tArA(_,_,k));
    cute::copy(smem_cp_B, tBsB(_,_,k), tBrB(_,_,k));
  }
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::gemm(tmma, rA_full(_,_,k), rB_full(_,_,k), tC);
  }
}

// Variant where BOTH operands live in AIU-native smem (GEMM4: P = Q @ K^T).
template <bool Clear, typename ATraits, typename SmemA, typename SmemB,
          typename TC, typename Element>
CUTE_DEVICE void ff_gemm_aiuAB_tc01(
    SmemA const& sA, SmemB const& sB, int tidx, TC& tC) {
  using cute::_;
  fq::TiledMmaM32T<Element> tmma;
  auto thr_m = tmma.get_thread_slice(tidx);

  auto rA_full = thr_m.partition_fragment_A(sA);
  auto smem_cp_A = cute::make_tiled_copy_A(typename ATraits::SmemCopyAtom{}, tmma);
  auto thr_cp_A = smem_cp_A.get_thread_slice(tidx);
  auto tAsA = thr_cp_A.partition_S(cute::make_mix_tensor_like(sA));
  auto tArA = thr_cp_A.retile_D(rA_full);

  auto rB_full = thr_m.partition_fragment_B(sB);
  auto smem_cp_B = cute::make_tiled_copy_B(typename ATraits::SmemCopyAtom{}, tmma);
  auto thr_cp_B = smem_cp_B.get_thread_slice(tidx);
  auto tBsB = thr_cp_B.partition_S(cute::make_mix_tensor_like(sB));
  auto tBrB = thr_cp_B.retile_D(rB_full);

  if constexpr (Clear) cute::clear(tC);
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::copy(smem_cp_A, tAsA(_,_,k), tArA(_,_,k));
    cute::copy(smem_cp_B, tBsB(_,_,k), tBrB(_,_,k));
  }
  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(rA_full); ++k) {
    cute::gemm(tmma, rA_full(_,_,k), rB_full(_,_,k), tC);
  }
}
#endif  // FLASHQLA_ARCH_100

// ==========================================================================
// Templated on H (V-heads) and Hg (Q/K-heads). H/Hg become constexpr inside
// the kernel body so every H*..., Hg*..., H/Hg divide, and stride computation
// is constant-folded by the compiler. Runtime dispatch table lives in the
// host launcher below. blockIdx.x/y select head/sequence; blockIdx.z selects
// the DV tile for a one-group CTA. PPU1.0/1.1 uses one CTA/CU because its
// materialised K transpose aliases the state tile within the TSM window;
// PPU1.5 keeps the original two-CTA residency target.
#if defined(FLASHQLA_ARCH_100)
#  define FF_MIN_BLOCKS 1
#else
#  define FF_MIN_BLOCKS 2
#endif
enum class FfStateMode : int {
  Legacy,
  Indexed,
  Runtime,
};

// The hot statement body is expanded directly into three independent globals.
// tc01 dispatches the narrow Legacy ABI when no indexed state pool is present,
// otherwise the Indexed ABI. tc02 keeps Runtime by default; the launcher uses
// Indexed only for the measured low-batch H=24 long-state range.
//
// STATE_BF16 selects the recurrent-state pool element. It has to be a template
// parameter rather than a runtime flag: the state boundary loops are fully
// unrolled over every accumulator register, so a runtime select duplicates the
// unrolled body and measurably slows the fp32 path down. With the dtype fixed
// here the fp32 instantiation generates exactly the fp32-only code it replaces.
template <typename Element, int H_TPL, int Hg_TPL, int NGRP,
          int DVH_TPL = FF_DVH, bool AUX_P_TPL = false,
          bool STATE_BF16 = false>
__global__ void __launch_bounds__(FF_WG * (AUX_P_TPL ? 2 : NGRP), FF_MIN_BLOCKS)
fused_fwd_cp_cute_kernel(
    Element* __restrict__ o,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ a,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const fq::StatePool<STATE_BF16>* __restrict__ h0,
    fq::StatePool<STATE_BF16>* __restrict__ ht_out,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cp_seq_map,
    const int* __restrict__ raw_cu_seqlens,
    int batch_size, int num_tokens, int raw_batch_size,
    float scale, int64_t v_token_stride)
{
#define FF_BODY_LEGACY_TC01 1
#include "fused_fwd_cute_body.inc"
#undef FF_BODY_LEGACY_TC01
}

template <typename Element, int H_TPL, int Hg_TPL, int NGRP,
          int DVH_TPL = FF_DVH, bool AUX_P_TPL = false,
          bool STATE_BF16 = false>
__global__ void __launch_bounds__(FF_WG * (AUX_P_TPL ? 2 : NGRP), FF_MIN_BLOCKS)
fused_fwd_cp_cute_indexed_kernel(
    Element* __restrict__ o,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ a,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const fq::StatePool<STATE_BF16>* __restrict__ h0,
    fq::StatePool<STATE_BF16>* __restrict__ ht_out,
    fq::StatePool<STATE_BF16>* __restrict__ raw_state,
    const int* __restrict__ raw_state_indices,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cp_seq_map,
    const int* __restrict__ raw_cu_seqlens,
    int batch_size, int num_tokens, int raw_batch_size,
    float scale, int64_t v_token_stride)
{
#define FF_BODY_INDEXED_TC01 1
#include "fused_fwd_cute_body.inc"
#undef FF_BODY_INDEXED_TC01
}

template <typename Element, int H_TPL, int Hg_TPL, int NGRP,
          int DVH_TPL = FF_DVH, bool AUX_P_TPL = false,
          bool STATE_BF16 = false>
__global__ void __launch_bounds__(FF_WG * (AUX_P_TPL ? 2 : NGRP), FF_MIN_BLOCKS)
fused_fwd_cp_cute_runtime_kernel(
    Element* __restrict__ o,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ a,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const fq::StatePool<STATE_BF16>* __restrict__ h0,
    fq::StatePool<STATE_BF16>* __restrict__ ht_out,
    fq::StatePool<STATE_BF16>* __restrict__ raw_state,
    const int* __restrict__ raw_state_indices,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cp_seq_map,
    const int* __restrict__ raw_cu_seqlens,
    int batch_size, int num_tokens, int raw_batch_size,
    float scale, int64_t v_token_stride)
{
#include "fused_fwd_cute_body.inc"
}

template <typename Element, int H_TPL, int Hg_TPL, int NGRP,
          int DVH_TPL = FF_DVH, bool AUX_P_TPL = false,
          bool STATE_BF16 = false>
__global__ void __launch_bounds__(FF_WG * (AUX_P_TPL ? 2 : NGRP), FF_MIN_BLOCKS)
fused_fwd_cp_cute_runtime_head_major_kernel(
    Element* __restrict__ o,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ a,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const fq::StatePool<STATE_BF16>* __restrict__ h0,
    fq::StatePool<STATE_BF16>* __restrict__ ht_out,
    fq::StatePool<STATE_BF16>* __restrict__ raw_state,
    const int* __restrict__ raw_state_indices,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cp_seq_map,
    const int* __restrict__ raw_cu_seqlens,
    int batch_size, int num_tokens, int raw_batch_size,
    float scale, int64_t v_token_stride)
{
#define FF_BODY_HEAD_MAJOR 1
#include "fused_fwd_cute_body.inc"
#undef FF_BODY_HEAD_MAJOR
}

}  // namespace flashqla_cute

// ============================================================
// Host launch
// ============================================================
namespace {

constexpr int kFusedMaxCachedDevices = 64;

template <flashqla_cute::FfStateMode StateMode,
          typename Element, int H, int Hg, int NGRP, int DVH, bool AUX_P,
          bool SBF16>
void configure_fused_kernel_smem_once(int device_index, int smem_cap) {
  auto configure = [smem_cap]() {
    if constexpr (StateMode == flashqla_cute::FfStateMode::Legacy) {
      hggcFuncSetAttribute(
          flashqla_cute::fused_fwd_cp_cute_kernel<
              Element, H, Hg, NGRP, DVH, AUX_P, SBF16>,
          hggcFuncAttributeMaxDynamicSharedMemorySize, smem_cap);
    } else if constexpr (StateMode == flashqla_cute::FfStateMode::Indexed) {
      hggcFuncSetAttribute(
          flashqla_cute::fused_fwd_cp_cute_indexed_kernel<
              Element, H, Hg, NGRP, DVH, AUX_P, SBF16>,
          hggcFuncAttributeMaxDynamicSharedMemorySize, smem_cap);
    } else {
      hggcFuncSetAttribute(
          flashqla_cute::fused_fwd_cp_cute_runtime_kernel<
              Element, H, Hg, NGRP, DVH, AUX_P, SBF16>,
          hggcFuncAttributeMaxDynamicSharedMemorySize, smem_cap);
    }
    hggcGetLastError();
  };
  if (device_index >= 0 && device_index < kFusedMaxCachedDevices) {
    // Function-local storage is independently instantiated for every mode,
    // dtype, shape, group count, DV tile, AUX specialization, state dtype, and
    // device.
    static std::once_flag configured[kFusedMaxCachedDevices];
    std::call_once(configured[device_index], configure);
  } else {
    configure();
  }
}

template <typename Element, int H, int Hg, int NGRP, int DVH, bool AUX_P,
          bool SBF16>
void configure_fused_head_major_kernel_smem_once(
    int device_index, int smem_cap) {
  auto configure = [smem_cap]() {
    hggcFuncSetAttribute(
        flashqla_cute::fused_fwd_cp_cute_runtime_head_major_kernel<
            Element, H, Hg, NGRP, DVH, AUX_P, SBF16>,
        hggcFuncAttributeMaxDynamicSharedMemorySize, smem_cap);
    hggcGetLastError();
  };
  if (device_index >= 0 && device_index < kFusedMaxCachedDevices) {
    static std::once_flag configured[kFusedMaxCachedDevices];
    std::call_once(configured[device_index], configure);
  } else {
    configure();
  }
}

}  // namespace

void launch_fused_fwd_cp_cute(
    void* o, const void* q, const void* k, const void* v,
    const void* a, const float* beta, const float* g,
    const void* h0, void* ht_out, void* raw_state,
    bool state_bf16,
    const int* raw_state_indices,
    const int* cu_seqlens, const int* cp_seq_map, const int* raw_cu_seqlens,
    int batch_size, int num_tokens, int raw_batch_size,
    int H, int Hg, float scale, int dtype_tag,
    int64_t v_token_stride,
    int device_index, int num_cu, int compute_capability,
    hggcStream_t stream)
{
  if (compute_capability != 80 && compute_capability != 89) {
    throw std::runtime_error(
        "launch_fused_fwd_cp_cute: unsupported compute capability");
  }
  const int FORCE_1BLK_SMEM = 130 * 1024;
  const bool is_tc01 = compute_capability == 80;
  // NGRP=2 keeps both D64 value shards in one CTA and shares Q/K/A/P and input
  // loads between them. With the register chain this is 3-13% faster across
  // single-sequence H=8..32 once at least two chunks are present, despite the
  // smaller CTA grid: NGRP=1 duplicates more work than its extra grid
  // parallelism repays. Retain NGRP=1 only for the measured sub-two-chunk
  // low-head boundary, where both variants are launch dominated and the split
  // form is up to 0.7% faster. Packed and CP launches always use NGRP=2.
  const bool tiny_split =
      (H <= 8 && num_tokens < 64) || (H <= 16 && num_tokens < 48);
  const bool use_one_group = !is_tc01 &&
                             raw_batch_size == 1 && batch_size == 1 &&
                             tiny_split;
  // D32 exposes four independent DV CTAs per value head on tc01.  Public-API
  // A/B was uniformly faster for single sequences through H=16 and unsplit
  // B=2/H<=8.
  // Denser packed grids already fill the 64 CUs and only duplicate common work.
  // The DV-major register chain uses a 4x1 warp layout and therefore owns one
  // complete D64 tile.  The older D32 launch deliberately over-partitioned DV
  // for occupancy, but its 2-warp logical tile is not compatible with M4.
  const bool use_tc01_d32 = is_tc01 && !FF_REG_CHAIN &&
      batch_size == raw_batch_size &&
      ((raw_batch_size == 1 && H <= 16) ||
       (raw_batch_size == 2 && H <= 8));
  // The pruned register-chain layout has the same footprint for NGRP=1/2, so
  // retain both launch shapes in one binary.  Two groups share Q/K/A/Ag/Pg;
  // one group retains grid parallelism for low-head launches.  The environment
  // override is used by the reproducible A/B harness and does not change math.
  bool use_tc01_ngrp2 = is_tc01 && !use_tc01_d32;
  if (const char* ng2 = std::getenv("FF_TC01_NGRP2")) {
    if (ng2[0] == '0') use_tc01_ngrp2 = false;
    if (ng2[0] == '1') use_tc01_ngrp2 = true;
  }
  const bool use_tc01_aux_p = use_tc01_d32 && raw_batch_size == 1 &&
      batch_size == 1 && H <= 12;
  const int ngrp = is_tc01 ? (use_tc01_ngrp2 ? 2 : 1)
                           : (use_one_group ? 1 : 2);
  const int dvh = use_tc01_d32 ? 32 : flashqla_cute::FF_DVH;
  const bool use_tc02_h24_smem_prune = !is_tc01 && H == 24;
  int smem_bytes = is_tc01
      ? (use_tc01_d32 ? flashqla_cute::FF_TOTAL_TC01_D32
                      : flashqla_cute::FF_TOTAL_TC01)
      : (use_tc02_h24_smem_prune
             ? flashqla_cute::FF_TOTAL_TC02_PRUNED
             : (use_one_group ? flashqla_cute::FfSmemLayout<1>::TOTAL
                              : flashqla_cute::FF_TOTAL_TC02));
  bool force_1blk = false;
  {
    const char* f2 = std::getenv("FF_FORCE_2BLK");
    if (f2 && f2[0] == '1') force_1blk = false;
    const char* f1 = std::getenv("FF_FORCE_1BLK");
    if (f1 && f1[0] == '1') force_1blk = true;
  }
  if (force_1blk && smem_bytes < FORCE_1BLK_SMEM) smem_bytes = FORCE_1BLK_SMEM;
  int pad_cap = FORCE_1BLK_SMEM;
  {
    const char* pb = std::getenv("FF_PAD_BYTES");
    if (pb) {
      int pad = std::atoi(pb);
      if (pad > 0) {
        smem_bytes += pad;
        if (smem_bytes > pad_cap) pad_cap = smem_bytes;
      }
    }
  }
  // Indexed state makes the final-state destination known at launch time, so
  // this specialization removes the per-element raw_state/ht_out store branch.
  // Same-card complete-API ABBA measurements were uniformly positive for the
  // low-batch H=24 range from 6K tokens; shorter, denser, and other-head grids
  // retain the runtime ABI because they did not show the same stable gain.
  const bool use_tc02_h24_indexed_state =
      !is_tc01 && H == 24 && raw_batch_size <= 3 && num_tokens >= 6144 &&
      raw_state != nullptr;
  // Head-major launch order reduces variable-length tail waves by distributing
  // adjacent CTA IDs across heads of the same request. Keep the existing long
  // H=32/Hg=8/B=3 specialization and extend it to the full-sweep-safe packed
  // range H=32/B=6..25. CP segments retain sequence-major order.
  const bool use_tc02_h32_b3_head_major =
      !is_tc01 && H == 32 && Hg == 8 && raw_batch_size == 3 &&
      batch_size == 3 && num_tokens >= 16384;
  const bool use_tc02_h32_varlen_head_major =
      !is_tc01 && cp_seq_map == nullptr && H == 32 &&
      batch_size >= 6 && batch_size <= 25;
  const bool use_tc02_head_major =
      use_tc02_h32_b3_head_major || use_tc02_h32_varlen_head_major;
  dim3 grid = use_tc02_head_major
      ? dim3(H, batch_size, flashqla_cute::FF_DV / (ngrp * dvh))
      : dim3(batch_size, H, flashqla_cute::FF_DV / (ngrp * dvh));
  dim3 block(flashqla_cute::FF_WG, ngrp);

// The state pool dtype is a kernel template parameter, so each ABI expands
// into a pair of instantiations and the launcher picks one at runtime. The
// _S forms take the dtype as a macro argument; the public forms branch.
#define FF_LAUNCH_LEGACY_S(ElementType, HV, HGV, NG, DVH, AUX, BLOCK, SBF16)     \
    do {                                                                         \
      using SPool = flashqla::StatePool<(SBF16)>;                                      \
      configure_fused_kernel_smem_once<                                          \
          flashqla_cute::FfStateMode::Legacy,                                    \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>(                \
              device_index, pad_cap);                                            \
      flashqla_cute::fused_fwd_cp_cute_kernel<                                   \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>                 \
          <<<grid, (BLOCK), smem_bytes, stream>>>(                               \
              reinterpret_cast<ElementType*>(o),                                 \
              reinterpret_cast<const ElementType*>(q),                           \
              reinterpret_cast<const ElementType*>(k),                           \
              reinterpret_cast<const ElementType*>(v),                           \
              reinterpret_cast<const ElementType*>(a), beta, g,                  \
              reinterpret_cast<const SPool*>(h0),                                \
              reinterpret_cast<SPool*>(ht_out),                                  \
              cu_seqlens, cp_seq_map, raw_cu_seqlens,                            \
              batch_size, num_tokens, raw_batch_size, scale, v_token_stride);    \
    } while (0)

#define FF_LAUNCH_LEGACY(ElementType, HV, HGV, NG, DVH, AUX, BLOCK)              \
    do {                                                                         \
      if (state_bf16) {                                                          \
        FF_LAUNCH_LEGACY_S(ElementType, HV, HGV, NG, DVH, AUX, BLOCK, true);     \
      } else {                                                                   \
        FF_LAUNCH_LEGACY_S(ElementType, HV, HGV, NG, DVH, AUX, BLOCK, false);    \
      }                                                                          \
    } while (0)

#define FF_LAUNCH_INDEXED_S(ElementType, HV, HGV, NG, DVH, AUX, BLOCK, SBF16)    \
    do {                                                                         \
      using SPool = flashqla::StatePool<(SBF16)>;                                      \
      configure_fused_kernel_smem_once<                                          \
          flashqla_cute::FfStateMode::Indexed,                                   \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>(                \
              device_index, pad_cap);                                            \
      flashqla_cute::fused_fwd_cp_cute_indexed_kernel<                           \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>                 \
          <<<grid, (BLOCK), smem_bytes, stream>>>(                               \
              reinterpret_cast<ElementType*>(o),                                 \
              reinterpret_cast<const ElementType*>(q),                           \
              reinterpret_cast<const ElementType*>(k),                           \
              reinterpret_cast<const ElementType*>(v),                           \
              reinterpret_cast<const ElementType*>(a), beta, g,                  \
              reinterpret_cast<const SPool*>(h0),                                \
              reinterpret_cast<SPool*>(ht_out),                                  \
              reinterpret_cast<SPool*>(raw_state), raw_state_indices,            \
              cu_seqlens, cp_seq_map, raw_cu_seqlens,                            \
              batch_size, num_tokens, raw_batch_size, scale, v_token_stride);    \
    } while (0)

#define FF_LAUNCH_INDEXED(ElementType, HV, HGV, NG, DVH, AUX, BLOCK)             \
    do {                                                                         \
      if (state_bf16) {                                                          \
        FF_LAUNCH_INDEXED_S(ElementType, HV, HGV, NG, DVH, AUX, BLOCK, true);    \
      } else {                                                                   \
        FF_LAUNCH_INDEXED_S(ElementType, HV, HGV, NG, DVH, AUX, BLOCK, false);   \
      }                                                                          \
    } while (0)

#define FF_LAUNCH_RUNTIME_S(ElementType, HV, HGV, NG, DVH, AUX, BLOCK, SBF16)    \
    do {                                                                         \
      using SPool = flashqla::StatePool<(SBF16)>;                                \
      configure_fused_kernel_smem_once<                                          \
          flashqla_cute::FfStateMode::Runtime,                                   \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>(               \
              device_index, pad_cap);                                            \
      flashqla_cute::fused_fwd_cp_cute_runtime_kernel<                           \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>                \
          <<<grid, (BLOCK), smem_bytes, stream>>>(                               \
              reinterpret_cast<ElementType*>(o),                                 \
              reinterpret_cast<const ElementType*>(q),                           \
              reinterpret_cast<const ElementType*>(k),                           \
              reinterpret_cast<const ElementType*>(v),                           \
              reinterpret_cast<const ElementType*>(a), beta, g,                  \
              reinterpret_cast<const SPool*>(h0),                                \
              reinterpret_cast<SPool*>(ht_out),                                  \
              reinterpret_cast<SPool*>(raw_state), raw_state_indices,            \
              cu_seqlens, cp_seq_map, raw_cu_seqlens,                            \
              batch_size, num_tokens, raw_batch_size, scale, v_token_stride);    \
    } while (0)

#define FF_LAUNCH_RUNTIME(ElementType, HV, HGV, NG, DVH, AUX, BLOCK)             \
    do {                                                                         \
      if (state_bf16) {                                                          \
        FF_LAUNCH_RUNTIME_S(                                                     \
            ElementType, HV, HGV, NG, DVH, AUX, BLOCK, true);                    \
      } else {                                                                   \
        FF_LAUNCH_RUNTIME_S(                                                     \
            ElementType, HV, HGV, NG, DVH, AUX, BLOCK, false);                   \
      }                                                                          \
    } while (0)

#define FF_LAUNCH_RUNTIME_HEAD_MAJOR_S(                                          \
    ElementType, HV, HGV, NG, DVH, AUX, BLOCK, SBF16)                            \
    do {                                                                         \
      using SPool = flashqla::StatePool<(SBF16)>;                                \
      configure_fused_head_major_kernel_smem_once<                               \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>(               \
              device_index, pad_cap);                                            \
      flashqla_cute::fused_fwd_cp_cute_runtime_head_major_kernel<                \
          ElementType, (HV), (HGV), (NG), (DVH), (AUX), (SBF16)>                \
          <<<grid, (BLOCK), smem_bytes, stream>>>(                               \
              reinterpret_cast<ElementType*>(o),                                 \
              reinterpret_cast<const ElementType*>(q),                           \
              reinterpret_cast<const ElementType*>(k),                           \
              reinterpret_cast<const ElementType*>(v),                           \
              reinterpret_cast<const ElementType*>(a), beta, g,                  \
              reinterpret_cast<const SPool*>(h0),                                \
              reinterpret_cast<SPool*>(ht_out),                                  \
              reinterpret_cast<SPool*>(raw_state), raw_state_indices,            \
              cu_seqlens, cp_seq_map, raw_cu_seqlens,                            \
              batch_size, num_tokens, raw_batch_size, scale, v_token_stride);    \
    } while (0)

#define FF_LAUNCH_RUNTIME_HEAD_MAJOR(                                            \
    ElementType, HV, HGV, NG, DVH, AUX, BLOCK)                                  \
    do {                                                                         \
      if (state_bf16) {                                                          \
        FF_LAUNCH_RUNTIME_HEAD_MAJOR_S(                                          \
            ElementType, HV, HGV, NG, DVH, AUX, BLOCK, true);                    \
      } else {                                                                   \
        FF_LAUNCH_RUNTIME_HEAD_MAJOR_S(                                          \
            ElementType, HV, HGV, NG, DVH, AUX, BLOCK, false);                   \
      }                                                                          \
    } while (0)

#define FF_LAUNCH_TC01(ElementType, HV, HGV, NG, DVH, AUX, BLOCK)                \
    do {                                                                         \
      if (raw_state != nullptr) {                                                 \
        FF_LAUNCH_INDEXED(ElementType, HV, HGV, NG, DVH, AUX, BLOCK);            \
      } else {                                                                   \
        FF_LAUNCH_LEGACY(ElementType, HV, HGV, NG, DVH, AUX, BLOCK);             \
      }                                                                          \
      return;                                                                    \
    } while (0)

#define FF_LAUNCH_AUX_P(ElementType, HV, HGV)                                   \
    do {                                                                         \
      dim3 aux_block(flashqla_cute::FF_WG, 2);                                  \
      FF_LAUNCH_TC01(ElementType, HV, HGV, 1, 32, true, aux_block);             \
    } while (0)

#define FF_DISPATCH_AUX_P(ElementType, HV, HGV)                                 \
    if (H == (HV) && Hg == (HGV)) { FF_LAUNCH_AUX_P(ElementType, HV, HGV); }

#define FF_DISPATCH_AUX_P_ALL(ElementType) \
  FF_DISPATCH_AUX_P(ElementType, 12, 4) \
  FF_DISPATCH_AUX_P(ElementType,  8, 4) \
  FF_DISPATCH_AUX_P(ElementType,  8, 2) \
  FF_DISPATCH_AUX_P(ElementType,  6, 2) \
  FF_DISPATCH_AUX_P(ElementType,  4, 2)

  if (use_tc01_aux_p) {
    if (dtype_tag == 0) { FF_DISPATCH_AUX_P_ALL(__half) }
    else { FF_DISPATCH_AUX_P_ALL(__ppu_bfloat16) }
  }

#undef FF_DISPATCH_AUX_P_ALL
#undef FF_DISPATCH_AUX_P
#undef FF_LAUNCH_AUX_P

#define FF_LAUNCH_E(ElementType, HV, HGV, NG, DVH)                              \
    do {                                                                         \
      if (is_tc01) {                                                             \
        FF_LAUNCH_TC01(ElementType, HV, HGV, NG, DVH, false, block);            \
      } else {                                                                   \
        if constexpr ((HV) == 24) {                                              \
          if (use_tc02_h24_indexed_state) {                                      \
            FF_LAUNCH_INDEXED(ElementType, HV, HGV, NG, DVH, false, block);     \
          } else {                                                               \
            FF_LAUNCH_RUNTIME(ElementType, HV, HGV, NG, DVH, false, block);     \
          }                                                                      \
        } else if constexpr ((HV) == 32) {                                       \
          if (use_tc02_head_major) {                                             \
            FF_LAUNCH_RUNTIME_HEAD_MAJOR(                                        \
                ElementType, HV, HGV, NG, DVH, false, block);                    \
          } else {                                                               \
            FF_LAUNCH_RUNTIME(ElementType, HV, HGV, NG, DVH, false, block);      \
          }                                                                      \
        } else {                                                                 \
          FF_LAUNCH_RUNTIME(ElementType, HV, HGV, NG, DVH, false, block);        \
        }                                                                        \
        return;                                                                  \
      }                                                                          \
    } while (0)

#define FF_DISPATCH_E(ElementType, HV, HGV)                                      \
    if (H == (HV) && Hg == (HGV)) {                                              \
      if (use_tc01_d32) { FF_LAUNCH_E(ElementType, HV, HGV, 1, 32); }            \
      else if (ngrp == 1) { FF_LAUNCH_E(ElementType, HV, HGV, 1, 64); }          \
      else { FF_LAUNCH_E(ElementType, HV, HGV, 2, 64); }                         \
    }

#define FF_DISPATCH_ALL(ElementType) \
  FF_DISPATCH_E(ElementType, 128, 16) \
  FF_DISPATCH_E(ElementType,  64, 16) \
  FF_DISPATCH_E(ElementType,  64,  8) \
  FF_DISPATCH_E(ElementType,  48, 16) \
  FF_DISPATCH_E(ElementType,  32, 16) \
  FF_DISPATCH_E(ElementType,  32,  8) \
  FF_DISPATCH_E(ElementType,  32,  4) \
  FF_DISPATCH_E(ElementType,  24,  8) \
  FF_DISPATCH_E(ElementType,  16,  8) \
  FF_DISPATCH_E(ElementType,  16,  4) \
  FF_DISPATCH_E(ElementType,  16,  2) \
  FF_DISPATCH_E(ElementType,  12,  4) \
  FF_DISPATCH_E(ElementType,   8,  4) \
  FF_DISPATCH_E(ElementType,   8,  2) \
  FF_DISPATCH_E(ElementType,   6,  2) \
  FF_DISPATCH_E(ElementType,   4,  2)

  if (dtype_tag == 0) { FF_DISPATCH_ALL(__half) }
  else { FF_DISPATCH_ALL(__ppu_bfloat16) }

#undef FF_DISPATCH_ALL
#undef FF_DISPATCH_E
#undef FF_LAUNCH_E
#undef FF_LAUNCH_TC01
#undef FF_LAUNCH_RUNTIME_HEAD_MAJOR
#undef FF_LAUNCH_RUNTIME_HEAD_MAJOR_S
#undef FF_LAUNCH_RUNTIME
#undef FF_LAUNCH_RUNTIME_S
#undef FF_LAUNCH_INDEXED
#undef FF_LAUNCH_INDEXED_S
#undef FF_LAUNCH_LEGACY
#undef FF_LAUNCH_LEGACY_S

  fprintf(stderr,
      "[fused_fwd_cp_cute] FATAL: unsupported (H=%d, Hg=%d) — add a FF_DISPATCH "
      "entry in launch_fused_fwd_cp_cute (fused_fwd_cute.cu) and rebuild.\n",
      H, Hg);
  std::abort();
}
