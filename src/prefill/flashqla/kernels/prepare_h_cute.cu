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
 * FlashQLA PPU1.5 CuTe — Kernel (3) prepare_h_cp  [register-h + K/A swzl port]
 *
 * Same recurrence as fused_fwd (K/V/A -> U/W/Vd -> h += K^T @ g_rev*Vd), but
 * runs BACKWARDS over the warmup chunks and produces the additive state ht
 * plus the homogeneous transition mt (no output o).  This revision ports the
 * fused_fwd K/A swizzle optimizations:
 *
 *   PH_K_SWZL=1 : K is AIU .swzl-written into a 128B-XOR'd smem tile, then
 *                 GEMM1 reads it via flashqla_gemm_ss_swzlA_m32 and GEMM3
 *                 (state update) reads K^T via flashqla_gemm_ss_swzlA_trans_acc_m32.
 *                 The explicit KT smem transpose vanishes, KT_buf is unallocated.
 *
 *   PH_A_SWZL=1 : the gated WY factor Ag is written into a Swizzle<2,3,3>
 *                 Ag_swz[CS,CS] buffer via int4 vec stores (8-half chunks;
 *                 Swizzle<2,3,3> preserves the low 3 bits so int4 alignment
 *                 holds).  GEMM2 wraps Ag_swz with make_smem_swzA_cs32 so
 *                 ldmatrix reads bank-conflict-halved.
 *
 * Smem (CS=32, PH_K_SWZL=1, PH_XY_FACTOR=1):
 *   double-buffered K 16KB + V 16KB + A 4KB + swizzled X^T 8KB
 *   + scalars 0.625KB ≈ 44.625KB. The four state groups keep h and Y in
 *   warp-owned register fragments; neither value is published through shared
 *   memory. While they process one chunk, AIU loads the next chunk into the
 *   alternate input slot.
 *
 * ht and mt are produced by one 512-thread CTA. Groups 0-1 compute the
 * additive state ht, while groups 2-3 run the same recurrence homogeneously
 * (V contribution forced to 0, state initialized to identity) to produce mt.
 * The four groups share each K/A/g load and Ag construction. The chunk
 * recurrence is affine in the state (h_out = ht + mt @ h_in), so mt is yielded
 * column-by-column (the DV axis is reinterpreted as the DK' index; DV == DK ==
 * 128). See numpy validation mt_math_check.py. correct_h0 then applies
 * h[i+1] = ht[i] + mt[i] @ h[i] on CP-fallback segments.
 *
 * Grid: batch * H, Block: 512 (2 ht + 2 mt groups, each 128 threads)
 */
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "flashqla_cute_traits.cuh"

// K .swzl port default ON (matches fused_fwd's FF_K_SWZL default).  Eliminates
// KT_buf and the CTA-wide K->KT smem transpose loop.
#ifndef PH_K_SWZL
#define PH_K_SWZL 1
#endif

// A .swzl port default ON.  Adds Ag_swz[CS,CS] Swizzle<2,3,3> buffer written
// via int4 vec stores; GEMM2 reads it with make_smem_swzA_cs32.
#ifndef PH_A_SWZL
#define PH_A_SWZL 1
#endif

#ifndef PH_REG_GVD_RECOMPUTE
#define PH_REG_GVD_RECOMPUTE 1
#endif

// Upstream FlashQLA factorization of the chunk state update:
//
//   X = -diag(beta) @ A^T @ K
//   Y = exp(g_last) * (K @ h) - exp(g_last - g) * V
//   h = exp(g_last) * h + X^T @ Y
//
// X is independent of the state and DV tile, so the combined ht/mt kernel
// computes it once and shares it across all four 128-thread state groups.
#if defined(FLASHQLA_ARCH_100)
#if defined(PH_XY_FACTOR)
#undef PH_XY_FACTOR
#endif
#define PH_XY_FACTOR 0
#else
#ifndef PH_XY_FACTOR
#define PH_XY_FACTOR 1
#endif
#endif

namespace flashqla_cute {

namespace fq = flashqla;

constexpr int PH_DK  = 128;
constexpr int PH_DV  = 128;
constexpr int PH_DVH = 64;    // DV half per group
constexpr int PH_CS  = 32;    // chunk size (matches kkt A [T,H,32])
constexpr int PH_WG  = 128;   // threads per DV-group (4-warp M32 MMA)
constexpr int PH_NBUF = 2;    // overlap next K/V/A load with current compute
constexpr float PH_LOG2E = 1.4426950408889634f;

// Each warp owns one fixed 16-column state tile. This layout lets a warp turn
// the accumulator holding h or Y directly into the next TN MMA's B fragment;
// the M32 layout remains in use for the independent X^T construction below.
template <typename Element>
using PhMmaN4T = decltype(cute::make_tiled_mma(
    fq::MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_1, cute::_4, cute::_1>>{}));

template <typename Element>
using PhMmaM4T = decltype(cute::make_tiled_mma(
    fq::MmaAtom<Element>{},
    cute::Layout<cute::Shape<cute::_4, cute::_1, cute::_1>>{}));

// ---- smem byte layout (single-buffer) --------------------------------------
constexpr int PH_ELEM_K   = PH_CS * PH_DK;    // K element count       (8KB)
constexpr int PH_ELEM_V   = PH_CS * PH_DV;    // V element count       (8KB)
constexpr int PH_ELEM_A   = PH_CS * PH_CS;    // A element count       (2KB)
constexpr int PH_ELEM_HT  = PH_DVH * PH_DK;   // per-group hstmT       (16KB / 2)
constexpr int PH_ELEM_WT  = PH_DVH * PH_CS;   // per-group WT / gVdT   (4KB / 2)
constexpr int PH_ELEM_XT  = PH_DK * PH_CS;    // swizzled X^T           (8KB)

// KT_buf only when PH_K_SWZL=0 (else swzlA_trans_acc_m32 reads K^T directly).
#if PH_K_SWZL
constexpr int PH_KT_BYTES = 0;
#else
constexpr int PH_KT_BYTES = PH_DK * PH_CS * 2;  // KT_buf[DK,CS] fp16
#endif

// Ag_swz single-buffer 2KB (CS=32) only when PH_A_SWZL=1.
#if PH_A_SWZL
constexpr int PH_AGSWZ_BYTES = PH_CS * PH_CS * 2;
#else
constexpr int PH_AGSWZ_BYTES = 0;
#endif

constexpr int PH_K     = 0;                                       // K_buf  [CS,DK] fp16
#if PH_XY_FACTOR
constexpr int PH_V     = PH_K   + PH_NBUF * PH_ELEM_K * 2;        // V_buf[2][CS,DV]
constexpr int PH_A     = PH_V   + PH_NBUF * PH_ELEM_V * 2;        // A_buf[2][CS,CS]
constexpr int PH_KT    = PH_A   + PH_ELEM_A * 2;
constexpr int PH_AGSWZ = PH_KT;
#else
constexpr int PH_V     = PH_K   + PH_ELEM_K * 2;                  // V_buf  [CS,DV] fp16
constexpr int PH_A     = PH_V   + PH_ELEM_V * 2;                  // A_buf  [CS,CS] fp16 (input landing)
constexpr int PH_KT    = PH_A   + PH_ELEM_A * 2;                  // KT_buf [DK,CS] (0 if PH_K_SWZL)
constexpr int PH_AGSWZ = PH_KT  + PH_KT_BYTES;                    // Ag_swz [CS,CS] (0 if !PH_A_SWZL)
#endif
template <bool PH_COMBINED, bool PH_SPLIT_DV = false, bool PrecomputeX = false>
struct PhSmemLayout {
  static constexpr int NGRP = PH_SPLIT_DV ? 1 : (PH_COMBINED ? 4 : 2);
  static constexpr int NTHREADS = PH_WG * NGRP;
#if PH_XY_FACTOR
  static constexpr int XT = PrecomputeX ? PH_A : PH_A + PH_NBUF * PH_ELEM_A * 2;
  static constexpr int GSH = XT + (PrecomputeX ? PH_NBUF : 1) * PH_ELEM_XT * 2;
  static constexpr int BSH = GSH + PH_NBUF * PH_CS * 4;
  static constexpr int GREV = BSH + PH_NBUF * PH_CS * 4;
  static constexpr int TOTAL = GREV + PH_CS * 4;
#else
  static constexpr int HSTMT = PH_AGSWZ + PH_AGSWZ_BYTES;
  static constexpr int WT = HSTMT + NGRP * PH_ELEM_HT * 2;
  static constexpr int GVDT = WT + NGRP * PH_ELEM_WT * 2;
  static constexpr int GSH = GVDT + NGRP * PH_ELEM_WT * 2;
  static constexpr int BSH = GSH + PH_CS * 4;
  static constexpr int GEXP = BSH + PH_CS * 4;
  static constexpr int GREV = GEXP + PH_CS * 4;
  static constexpr int TOTAL = GREV + PH_CS * 4;
#endif
};

constexpr int PH_SMEM_TC01 = 70144;
constexpr int PH_SMEM_TC01_SPLIT_DV = 45568;
constexpr int PH_SMEM_TC02_COMBINED = 45696;
#if defined(FLASHQLA_ARCH_100)
static_assert(PhSmemLayout<false>::TOTAL == PH_SMEM_TC01);
static_assert(PhSmemLayout<false, true>::TOTAL == PH_SMEM_TC01_SPLIT_DV);
#else
static_assert(PhSmemLayout<true>::TOTAL == PH_SMEM_TC02_COMBINED);
#endif

// ============================================================================
// PPU1.0/1.1 (tc01) K/V/A pipeline traits + gmem->smem loaders.
//
// Same shape story as KktKTraitsTc01: closed actlize pair
//   (PPU0010_AIU_LOAD, PPU0010_TSM_LD_SWZL)
// with an AIU-native cube-major SmemLayoutAtom<_8, AiuContElemSize>. K/V share
// one traits (BLK_MN=32, BLK_K=128, cube_w=64, InstNum=2); A has BLK_K=32 so
// cube_w=32 and InstNum=1 (one AIU instruction covers the full tile).
//
// smem footprint is unchanged vs the packed layout (tile_to_shape introduces
// no padding beyond the atom), so PH_ELEM_{K,V,A} and the PH_{K,V,A} byte
// offsets above still describe the same allocations. Downstream readers use
// the SmemLayout via cute::make_tensor(smem_ptr, SmemLayout{}) rather than
// raw index arithmetic -- see the per-GEMM sites below.
// ============================================================================
#if defined(FLASHQLA_ARCH_100)
template <typename Element, int BLK_MN_, int BLK_K_, bool Swzl_ = false>
struct PhBufTraitsTc01 {
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
using PhKVTraitsTc01 = PhBufTraitsTc01<Element, PH_CS, PH_DK>;
template <typename Element>
using PhATraitsTc01  = PhBufTraitsTc01<Element, PH_CS, PH_CS>;
// K is consumed by TSM_LD_SWZL (GEMM1 non-trans + state-GEMM Trans=true), so its
// AIU write MUST be swizzled to match the reader. V/A are read elementwise via
// the plain SmemLayout view, so they stay linear (PhKVTraitsTc01 / PhATraitsTc01).
template <typename Element>
using PhKTraitsTc01  = PhBufTraitsTc01<Element, PH_CS, PH_DK, /*Swzl=*/true>;

// Issue one AIU-driven load of a [rows<=BLK_MN, BLK_K] gmem tile into a smem
// buffer laid out per Traits::SmemLayout. .padz zero-fills rows past `rows`.
// Single thread (tx==0) issues the copy; caller commits.
template <typename Traits, typename Element>
__forceinline__ __device__ void ph_load_tile_tc01_(
    Element* dst_smem, const Element* src_gmem,
    int row_stride_elems, int rows, int tx) {
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
  if (tx == 0) {
    auto thr = tiled_copy.get_slice(0);
    auto tAgA = thr.partition_S(gA);
    auto tAsA = thr.partition_D(sA);
    cute::copy(tiled_copy, tAgA, tAsA);
  }
}

template <typename Element>
__forceinline__ __device__ void ph_load_k_tile_tc01(
    Element* K_buf, const Element* K_src, int row_stride_elems, int rows, int tx) {
  ph_load_tile_tc01_<PhKTraitsTc01<Element>, Element>(
      K_buf, K_src, row_stride_elems, rows, tx);
}
template <typename Element>
__forceinline__ __device__ void ph_load_v_tile_tc01(
    Element* V_buf, const Element* V_src, int row_stride_elems, int rows, int tx) {
  ph_load_tile_tc01_<PhKVTraitsTc01<Element>, Element>(
      V_buf, V_src, row_stride_elems, rows, tx);
}
template <typename Element>
__forceinline__ __device__ void ph_load_a_tile_tc01(
    Element* A_buf, const Element* A_src, int row_stride_elems, int rows, int tx) {
  ph_load_tile_tc01_<PhATraitsTc01<Element>, Element>(
      A_buf, A_src, row_stride_elems, rows, tx);
}

// SmemLayout tensor views for the AIU-native buffers -- used by every reader
// (KT_buf materialise loop, elementwise A gate/beta, elementwise V read, MMA
// A-fragment fill) so downstream code never sees the cube-major internals.
template <typename Element>
CUTE_DEVICE auto ph_k_view_tc01(Element* K_buf) {
  using Traits = PhKVTraitsTc01<Element>;
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename Traits::CuteElem*>(K_buf)),
      typename Traits::SmemLayout{});
}
template <typename Element>
CUTE_DEVICE auto ph_v_view_tc01(Element* V_buf) {
  using Traits = PhKVTraitsTc01<Element>;
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename Traits::CuteElem*>(V_buf)),
      typename Traits::SmemLayout{});
}
template <typename Element>
CUTE_DEVICE auto ph_a_view_tc01(Element* A_buf) {
  using Traits = PhATraitsTc01<Element>;
  return cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<typename Traits::CuteElem*>(A_buf)),
      typename Traits::SmemLayout{});
}
#endif  // FLASHQLA_ARCH_100

// ---- AIU 2D b16 async load ---------------------------------------------------
__forceinline__ __device__ void ph_aiu_load_2d_b16(
    void* smem_ptr, const void* gmem_ptr,
    int dim_c, int dim_w, int64_t stride_w_bytes, int cube_w = -1) {
#if defined(FLASHQLA_ARCH_150)
  if (cube_w < 0) cube_w = dim_w;
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
      " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
      :: "r"(smem_addr), "l"(gmem_ptr),
         "r"(dim_c), "r"(dim_w), "r"(1),
         "r"(dim_c), "r"(cube_w), "r"(1),
         "l"(stride_w_bytes), "l"(int64_t(0)),
         "r"(0), "r"(0), "r"(0),
         "r"(0)
      : "memory");
#elif defined(FLASHQLA_ARCH_100)
  // See ff_aiu_load_2d_b16 in fused_fwd_cute.cu for the tc01 axis-remap
  // rationale (no explicit stride operand; stride-in-elements rides in dim_c).
  (void)cube_w;
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  int32_t dim_c_stride = static_cast<int32_t>(stride_w_bytes >> 1);
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b16"
      " [%0], [%1], {%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
      :: "r"(smem_addr), "l"(gmem_ptr),
         "r"(1), "r"(dim_w), "r"(dim_c_stride),
         "r"(0), "r"(0), "r"(0),
         "r"(1), "r"(dim_w), "r"(dim_c), "r"(1)
      : "memory");
#else
  (void)smem_ptr; (void)gmem_ptr; (void)dim_c; (void)dim_w; (void)stride_w_bytes;
  (void)cube_w;
#endif
}
__forceinline__ __device__ void ph_aiu_commit() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}
__forceinline__ __device__ void ph_aiu_wait_all() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}

// ---- AIU .swzl 2D write of a K chunk [rows, PH_DK] into an PH_DK/32 = 4
//      col-cube (CUBE_H=CUBE_W=32) 128B-XOR-swizzled smem tile.  Pairs with
//      fq::flashqla_gemm_ss_swzlA_m32 / _swzlA_trans_acc_m32.  Full (rows==32)
//      and partial (rows<32) chunks both OK: the .padz atom auto zero-fills
//      cube rows [rows, 32). ----
template <typename Element>
__forceinline__ __device__ void ph_aiu_swzl_write_k_cs32(
    Element* smem, const Element* gmem, int row_stride_elems, int rows) {
#if defined(__HGGC_ARCH__)
  using AiuWriteCube = flashqla::PpuAiuLoad<
      32 * 32 * 16, typename fq::ElementTraits<Element>::CuteType, /*Trans=*/false,
      /*CUBE_H=*/32, /*CUBE_W=*/32, /*Swzl=*/true>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t*>(gmem), /*h=*/rows, /*w=*/PH_DK,
            /*stride_w=*/row_stride_elems);
  CUTE_UNROLL
  for (int cube = 0; cube < PH_DK / 32; ++cube) {
    AiuWriteCube::copy(reinterpret_cast<void*>(smem + cube * 32 * 32),
                       reinterpret_cast<const void*>(gmem), desc,
                       /*coord_w=*/cube * 32, /*coord_h=*/0);
  }
#else
  (void)smem; (void)gmem; (void)row_stride_elems; (void)rows;
#endif
}

// Lane i of a full warp holds g[i] of one chunk (0 past `actual`). Returns
// g_rev[i] = exp2(g_last - g_i) with the tail clamped to g_last, matching the
// shared-memory formulation element for element.
__forceinline__ __device__ float ph_g_rev_lane(float gv, int actual, int lane) {
  const float g_last = __shfl_sync(0xffffffff, gv, actual > 0 ? actual - 1 : 0);
  const float gi = lane < actual ? gv : g_last;
  return exp2f((g_last - gi) * PH_LOG2E);
}

// ---- named group barrier: sync only the PH_WG (128) threads of one DV group.
//      Uses barrier resource id (gid+1) in {1,2} so the two groups get distinct
//      hardware barriers (id 0 is what __syncthreads uses). ----
__forceinline__ __device__ void ph_group_bar(int gid) {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.bar.sync %0, %1;\n" :: "r"(gid + 1), "n"(PH_WG) : "memory");
#else
  (void)gid;
  __syncthreads();
#endif
}

// The register-chain helpers name tc02-only PPU0015 types directly.
// Keep them out of every tc01 device pass even if a global build flag tries
// to enable the optimization.
#if defined(FLASHQLA_ARCH_150) && PH_XY_FACTOR
// Round two floats to the 16-bit element type with one paired conversion.
// Bit-for-bit equal to packing two scalar round-to-nearest conversions with
// `lo` in the low half, but avoids the separate mask/shift/or sequence.
template <typename Element>
__device__ __forceinline__ uint32_t ph_pack_rn_b16x2(float lo, float hi) {
  if constexpr (std::is_same_v<Element, __ppu_bfloat16>) {
    union { __ppu_bfloat162 v; uint32_t u; } bits;
    bits.v = __floats2bfloat162_rn(lo, hi);
    return bits.u;
  } else {
    union { __half2 v; uint32_t u; } bits;
    bits.v = __floats2half2_rn(lo, hi);
    return bits.u;
  }
}

// Exact widening of a packed b16x2 register (`lo` from the low half).
template <typename Element>
__device__ __forceinline__ void ph_unpack_b16x2(uint32_t u, float& lo, float& hi) {
  if constexpr (std::is_same_v<Element, __ppu_bfloat16>) {
    lo = __uint_as_float(u << 16);
    hi = __uint_as_float(u & 0xffff0000u);
  } else {
    union { uint32_t u; __half2 v; } bits;
    bits.u = u;
    const float2 f = __half22float2(bits.v);
    lo = f.x;
    hi = f.y;
  }
}

// Transposed recurrence: each warp owns DV16 throughout h^T -> U^T -> Y^T.
// K is the existing physical [CS,DK] swizzled tile. Swap=true reads its
// natural TN B operand; C->A conversion is lane-local on tc02.
template <typename Element, typename TH, typename TC>
__device__ __forceinline__ void ph_gemm_state_kB_m4(
    TH const& h, const Element* Kswz, int tidx, TC& u) {
  using TsmB = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, 32, 32,
      /*Swap=*/true, /*Trans=*/false, /*InstNum=*/PH_DK / 32>;
  cute::clear(u);
  auto rH = cute::recast<float>(h);
  auto rU = cute::recast<float>(u);
  CUTE_UNROLL
  for (int kt = 0; kt < PH_DK / 16; ++kt) {
    uint32_t b[PH_CS / 16][4];
    CUTE_UNROLL
    for (int n = 0; n < PH_CS / 16; ++n) {
      TsmB::copy(reinterpret_cast<int*>(b[n]),
          reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Kswz)),
          (kt % 2) * 16, n * 16, kt / 2, 0);
    }
    uint32_t c[4], a[4];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      c[i] = ph_pack_rn_b16x2<Element>(rH(2 * i, 0, kt), rH(2 * i + 1, 0, kt));
    }
    fq::cfrag_to_afrag_mma_u32<Element>(c, a, tidx & 31);
    CUTE_UNROLL
    for (int n = 0; n < PH_CS / 16; ++n) {
      fq::ppu_mma_impl<Element>(
          rU(0,0,n), rU(1,0,n), rU(2,0,n), rU(3,0,n),
          rU(4,0,n), rU(5,0,n), rU(6,0,n), rU(7,0,n),
          a[0], a[1], a[2], a[3], b[n][0], b[n][1], b[n][2], b[n][3],
          rU(0,0,n), rU(1,0,n), rU(2,0,n), rU(3,0,n),
          rU(4,0,n), rU(5,0,n), rU(6,0,n), rU(7,0,n));
    }
  }
}

// Scratch remains X^T[DK,CS] with four row cubes. Reading it as a natural
// TN B operand computes Y^T @ X without changing scratch or precompute order.
template <typename Element, int PreloadDepth = 4, typename TH>
__device__ __forceinline__ void ph_gemm_yA_xtB_m4_acc(
    const uint32_t y[PH_CS / 16][1][4], const Element* XTswz, TH& h) {
  using TsmB = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, 32, 32,
      /*Swap=*/true, /*Trans=*/false, /*InstNum=*/1>;
  auto rH = cute::recast<float>(h);
  CUTE_UNROLL
  for (int kt = 0; kt < PH_CS / 16; ++kt) {
    CUTE_UNROLL
    for (int n0 = 0; n0 < PH_DK / 16; n0 += PreloadDepth) {
      uint32_t b[PreloadDepth][4];
      CUTE_UNROLL
      for (int j = 0; j < PreloadDepth; ++j) {
        const int dk = (n0 + j) * 16;
        TsmB::copy(reinterpret_cast<int*>(b[j]),
            reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(XTswz)),
            kt * 16, dk % 32, 0, dk / 32);
      }
      CUTE_UNROLL
      for (int j = 0; j < PreloadDepth; ++j) {
        const int n = n0 + j;
        fq::ppu_mma_impl<Element>(
            rH(0,0,n), rH(1,0,n), rH(2,0,n), rH(3,0,n),
            rH(4,0,n), rH(5,0,n), rH(6,0,n), rH(7,0,n),
            y[kt][0][0], y[kt][0][1], y[kt][0][2], y[kt][0][3],
            b[j][0], b[j][1], b[j][2], b[j][3],
            rH(0,0,n), rH(1,0,n), rH(2,0,n), rH(3,0,n),
            rH(4,0,n), rH(5,0,n), rH(6,0,n), rH(7,0,n));
      }
    }
  }
}

// A is a natural [M,K] matrix in the same 32x32 AIU-swizzled format as K.
// B is already a warp-owned tc02 fragment produced from the state accumulator.
// The 1x4 warp layout keeps every 16-column DV tile on one warp, so K@h does
// not need to publish h to the other warps through shared memory.
template <int M, int K, typename Element, typename TC>
__device__ __forceinline__ void ph_gemm_swzlA_regB_n4(
    const Element* Aswz, const uint32_t b_regs[K / 16][1][4], int tidx,
    TC& tC, bool clear_acc = true) {
  constexpr int CH = 32;
  constexpr int CW = 32;
  constexpr int KT_PER_CUBE = CW / 16;
  constexpr int INST = K / CW;
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, CH, CW, /*Swap=*/false,
      /*Trans=*/false, /*InstNum=*/INST>;

  PhMmaN4T<Element> mma;
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

// Stream one state tile into the B fragment immediately before its MMA use.
// This preserves the conversion boundary and reduction order while shortening
// the lifetime of the 32-word packed-state snapshot.
template <int M, int K, typename Element, bool DecayState,
          typename TC, typename TH>
__device__ __forceinline__ void ph_gemm_swzlA_state_n4(
    const Element* Aswz, int tidx, TC& tC, TH& h, float decay) {
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, 32, 32, false, false, K / 32>;
  PhMmaN4T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<M, K>{});
  auto rA_full = thr_mma.partition_fragment_A(dummyA);
  cute::clear(tC);
  auto rC = cute::recast<float>(tC);
  auto rH = cute::recast<float>(h);
  const int lane = tidx & 31;
  CUTE_UNROLL
  for (int kt = 0; kt < cute::size<2>(rA_full); ++kt) {
    uint32_t packed[4], b[4];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      const uint32_t lo = fq::element_bits16<Element>(
          fq::float_to_element<Element>(rH(2 * i, kt, 0)));
      const uint32_t hi = fq::element_bits16<Element>(
          fq::float_to_element<Element>(rH(2 * i + 1, kt, 0)));
      packed[i] = fq::pack_b16x2_u32(lo, hi);
    }
    fq::cfrag_to_bfrag_swizzle_u32(packed, b, lane);
    if constexpr (DecayState) {
      CUTE_UNROLL
      for (int i = 0; i < 8; ++i) rH(i, kt, 0) *= decay;
    }
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tC); ++m) {
      int a[4] = {0, 0, 0, 0};
      TsmA::copy(a,
          reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(Aswz)),
          (kt % 2) * 16, m * 16, kt / 2, 0);
      fq::ppu_mma_impl<Element>(
          rC(0,m,0), rC(1,m,0), rC(2,m,0), rC(3,m,0),
          rC(4,m,0), rC(5,m,0), rC(6,m,0), rC(7,m,0),
          static_cast<uint32_t>(a[0]), static_cast<uint32_t>(a[1]),
          static_cast<uint32_t>(a[2]), static_cast<uint32_t>(a[3]),
          b[0], b[1], b[2], b[3],
          rC(0,m,0), rC(1,m,0), rC(2,m,0), rC(3,m,0),
          rC(4,m,0), rC(5,m,0), rC(6,m,0), rC(7,m,0));
    }
  }
}

// X^T consists of four independently swizzled 32x32 row cubes. Read those
// cubes with TSM_LD_SWZL and accumulate X^T@Y directly into the persistent
// [DK,DVh] state. `stage` selects the DK row cube; no plain shared-memory copy
// or cross-warp transpose remains on this path.
template <typename Element, typename TC>
__device__ __forceinline__ void ph_gemm_xt_swzl_regB_n4_acc(
    const Element* XTswz,
    const uint32_t b_regs[PH_CS / 16][1][4], int tidx, TC& tC) {
  using TsmA = cute::PPU0015_TSM_LD_SWZL<
      typename fq::ElementTraits<Element>::CuteType, 32, 32, /*Swap=*/false,
      /*Trans=*/false, /*InstNum=*/1>;
  PhMmaN4T<Element> mma;
  auto thr_mma = mma.get_thread_slice(tidx);
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<
          typename fq::ElementTraits<Element>::CuteType*>(0x10)),
      fq::SmemLayout2D<PH_DK, PH_CS>{});
  cute::Tensor rA_full = thr_mma.partition_fragment_A(dummyA);
  cute::Tensor rC = cute::recast<float>(tC);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tC); ++m) {
    const int row = m * 16;
    const int stage = row / 32;
    const int row_in_cube = row % 32;
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rA_full); ++k) {
      int frag[4] = {0, 0, 0, 0};
      TsmA::copy(frag,
                 reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(XTswz)),
                 k * 16, row_in_cube, 0, stage);
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tC); ++n) {
        fq::ppu_mma_impl<Element>(
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n),
            static_cast<uint32_t>(frag[0]), static_cast<uint32_t>(frag[1]),
            static_cast<uint32_t>(frag[2]), static_cast<uint32_t>(frag[3]),
            b_regs[k][n][0], b_regs[k][n][1],
            b_regs[k][n][2], b_regs[k][n][3],
            rC(0,m,n), rC(1,m,n), rC(2,m,n), rC(3,m,n),
            rC(4,m,n), rC(5,m,n), rC(6,m,n), rC(7,m,n));
      }
    }
  }
}
#endif

#if defined(FLASHQLA_ARCH_150) && PH_XY_FACTOR
// Construct the state-independent product on a chunk-parallel grid. Scratch
// slots reserve one extra chunk per segment so unaligned boundaries cannot
// alias the preceding segment's final padded tile.
template <typename Element>
__global__ void __launch_bounds__(128, 6) ph_precompute_x_kernel(
    const Element* k, const Element* a, const float* beta, Element* x,
    const int* cu_seqlens, int batch_size, int H, int Hg) {
  int chunk = blockIdx.x;
  const int head = blockIdx.y;
  int seq = 0, start = 0, end = 0;
  for (; seq < batch_size; ++seq) {
    start = cu_seqlens[seq];
    end = cu_seqlens[seq + 1];
    const int chunks = (end - start + PH_CS - 1) / PH_CS;
    if (chunk < chunks) break;
    chunk -= chunks;
  }
  if (seq == batch_size) return;
  const int cts = start + chunk * PH_CS;
  const int actual = min(PH_CS, end - cts);
  const int tx = threadIdx.x;
  __shared__ Element K_buf[PH_ELEM_K];
  __shared__ Element A_buf[PH_ELEM_A];
  __shared__ float b[PH_CS];
  if (tx == 0) {
    ph_aiu_swzl_write_k_cs32<Element>(K_buf,
        k + (int64_t)cts * Hg * PH_DK + (head / (H / Hg)) * PH_DK,
        Hg * PH_DK, actual);
    ph_aiu_load_2d_b16(A_buf,
        a + (int64_t)cts * H * PH_CS + head * PH_CS,
        PH_CS, actual, (int64_t)H * PH_CS * sizeof(Element), PH_CS);
    ph_aiu_commit();
  }
  if (tx < PH_CS) b[tx] = tx < actual
      ? beta[(int64_t)(cts + tx) * H + head] : 0.0f;
  ph_aiu_wait_all();
  __syncthreads();
  fq::TiledMmaM32T<Element> mma;
  auto thr = mma.get_thread_slice(tx);
  auto id = cute::make_identity_tensor(cute::make_shape(
      cute::Int<PH_CS>{}, cute::Int<PH_CS>{}));
  auto coords = thr.partition_C(id);
  auto sA = fq::make_smem_tensor_h<PH_CS, PH_CS>(A_buf);
  auto sAT = fq::transpose_tensor(sA);
  const int64_t base = ((int64_t)(cts / PH_CS + seq) * H + head) * PH_DK * PH_CS;
  CUTE_UNROLL
  for (int cube = 0; cube < PH_DK / PH_CS; ++cube) {
    auto acc = fq::make_acc<PH_CS, PH_CS, fq::TiledMmaM32T<Element>, Element>(tx);
    cute::clear(acc);
    fq::flashqla_gemm_ss_swzlA_trans_acc_m32<PH_CS, PH_CS,
        decltype(sAT), decltype(acc), Element>(
            K_buf + cube * PH_ELEM_A, sAT, tx, acc);
    auto r = cute::recast<float>(acc);
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(coords); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(coords); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(coords); ++vv) {
          auto crd = coords(vv, m, n);
          const int dk = cube * PH_CS + cute::get<0>(crd);
          const int cs = cute::get<1>(crd);
          x[base + dk * PH_CS + cs] = fq::float_to_element<Element>(
              -b[cs] * r(vv, m, n));
        }
      }
    }
  }
}

// Grouped variant: one CTA per (chunk, key head) runs PH_X_GROUP 128-thread
// groups, one per value head sharing the key head. K is loaded once, the
// A^T operand is fetched into registers once per group, and each group stages
// its rounded X tile in padded smem so the global write is a contiguous
// 16B-vector copy instead of scattered 2B stores. Arithmetic, accumulation
// order and rounding match ph_precompute_x_kernel element for element.
constexpr int PH_X_GROUP = 3;
constexpr int PH_X_STAGE_LD = PH_CS + 8;

template <typename Element>
__global__ void __launch_bounds__(PH_WG * PH_X_GROUP, 4)
ph_precompute_x_group_kernel(
    const Element* k, const Element* a, const float* beta, Element* x,
    const int* cu_seqlens, int batch_size, int H, int Hg) {
  using CE = typename fq::ElementTraits<Element>::CuteType;
  int chunk = blockIdx.x;
  const int key_head = blockIdx.y;
  int seq = 0, start = 0, end = 0;
  for (; seq < batch_size; ++seq) {
    start = cu_seqlens[seq];
    end = cu_seqlens[seq + 1];
    const int chunks = (end - start + PH_CS - 1) / PH_CS;
    if (chunk < chunks) break;
    chunk -= chunks;
  }
  if (seq == batch_size) return;
  const int cts = start + chunk * PH_CS;
  const int actual = min(PH_CS, end - cts);
  const int group = threadIdx.x / PH_WG;
  const int tx = threadIdx.x % PH_WG;
  const int head = key_head * PH_X_GROUP + group;
  __shared__ Element K_buf[PH_ELEM_K];
  __shared__ Element A_buf[PH_X_GROUP][PH_ELEM_A];
  __shared__ float b[PH_X_GROUP][PH_CS];
  __shared__ __align__(16) Element stage[PH_X_GROUP][PH_DK * PH_X_STAGE_LD];
  if (threadIdx.x == 0) {
    ph_aiu_swzl_write_k_cs32<Element>(K_buf,
        k + (int64_t)cts * Hg * PH_DK + key_head * PH_DK, Hg * PH_DK, actual);
    CUTE_UNROLL
    for (int g = 0; g < PH_X_GROUP; ++g) {
      ph_aiu_load_2d_b16(A_buf[g],
          a + (int64_t)cts * H * PH_CS + (key_head * PH_X_GROUP + g) * PH_CS,
          PH_CS, actual, (int64_t)H * PH_CS * sizeof(Element), PH_CS);
    }
    ph_aiu_commit();
  }
  if (tx < PH_CS) b[group][tx] = tx < actual
      ? beta[(int64_t)(cts + tx) * H + head] : 0.0f;
  ph_aiu_wait_all();
  __syncthreads();

  using TsmTrans = fq::PpuTsmLdSwzl<CE, 32, 32, /*Swap=*/false, /*Trans=*/true, /*InstNum=*/1>;
  fq::TiledMmaM32T<Element> mma;
  auto thr = mma.get_thread_slice(tx);
  auto id = cute::make_identity_tensor(cute::make_shape(
      cute::Int<PH_CS>{}, cute::Int<PH_CS>{}));
  auto coords = thr.partition_C(id);
  auto sA = fq::make_smem_tensor_h<PH_CS, PH_CS>(A_buf[group]);
  auto sAT = fq::transpose_tensor(sA);
  auto rB_full = thr.partition_fragment_B(sAT);
  {
    auto copy_B = cute::make_tiled_copy_B(fq::SmemCopyAtomT<Element>{}, mma);
    auto thr_copy_B = copy_B.get_thread_slice(tx);
    auto sB_part = thr_copy_B.partition_S(sAT);
    auto rB_retile = thr_copy_B.retile_D(rB_full);
    CUTE_UNROLL
    for (int kk = 0; kk < cute::size<2>(rB_full); ++kk)
      cute::copy(copy_B, sB_part(cute::_, cute::_, kk), rB_retile(cute::_, cute::_, kk));
  }
  auto dummyA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CE*>(0x10)),
      fq::SmemLayout2D<PH_CS, PH_CS>{});
  auto rA_full = thr.partition_fragment_A(dummyA);
  const int dk_in_cube = (tx / 32) % 2 * 16;
  Element* tile = stage[group];
  CUTE_UNROLL
  for (int cube = 0; cube < PH_DK / PH_CS; ++cube) {
    auto acc = fq::make_acc<PH_CS, PH_CS, fq::TiledMmaM32T<Element>, Element>(tx);
    cute::clear(acc);
    auto r = cute::recast<float>(acc);
    CUTE_UNROLL
    for (int kk = 0; kk < cute::size<2>(rA_full); ++kk) {
      int afrag[4] = {0, 0, 0, 0};
      TsmTrans::copy(afrag, reinterpret_cast<void*>(
                         fq::ff_swzl_read_base<Element>(K_buf + cube * PH_ELEM_A)),
                     kk * 16, dk_in_cube, 0, 0);
      auto rA = cute::recast<uint32_t>(rA_full(cute::_, 0, kk));
      rA(0) = static_cast<uint32_t>(afrag[0]);
      rA(1) = static_cast<uint32_t>(afrag[1]);
      rA(2) = static_cast<uint32_t>(afrag[2]);
      rA(3) = static_cast<uint32_t>(afrag[3]);
    }
    CUTE_UNROLL
    for (int kk = 0; kk < cute::size<2>(rB_full); ++kk) {
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(acc); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<1>(rB_full); ++n) {
          auto rA = cute::recast<uint32_t>(rA_full(cute::_, m, kk));
          auto rB = cute::recast<uint32_t>(rB_full(cute::_, n, kk));
          fq::ppu_mma_impl<Element>(
              r(0,m,n), r(1,m,n), r(2,m,n), r(3,m,n),
              r(4,m,n), r(5,m,n), r(6,m,n), r(7,m,n),
              rA(0), rA(1), rA(2), rA(3),
              rB(0), rB(1), rB(2), rB(3),
              r(0,m,n), r(1,m,n), r(2,m,n), r(3,m,n),
              r(4,m,n), r(5,m,n), r(6,m,n), r(7,m,n));
        }
      }
    }
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(coords); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(coords); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(coords); ++vv) {
          auto crd = coords(vv, m, n);
          const int dk = cube * PH_CS + cute::get<0>(crd);
          const int cs = cute::get<1>(crd);
          tile[dk * PH_X_STAGE_LD + cs] = fq::float_to_element<Element>(
              -b[group][cs] * r(vv, m, n));
        }
      }
    }
  }
  ph_group_bar(group);
  constexpr int VEC = 16 / sizeof(Element);
  constexpr int ROW_VECS = PH_CS / VEC;
  uint4* dst = reinterpret_cast<uint4*>(
      x + ((int64_t)(cts / PH_CS + seq) * H + head) * PH_DK * PH_CS);
  CUTE_UNROLL
  for (int i = tx; i < PH_DK * ROW_VECS; i += PH_WG) {
    const int row = i / ROW_VECS;
    const int col = (i % ROW_VECS) * VEC;
    dst[i] = *reinterpret_cast<const uint4*>(tile + row * PH_X_STAGE_LD + col);
  }
}

template <typename Element>
__device__ __forceinline__ void ph_load_precomputed_x(
    Element* smem, const Element* gmem) {
  using Copy = fq::PpuAiuLoad<32 * 32 * 16,
      typename fq::ElementTraits<Element>::CuteType, false, 32, 32, true>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t*>(gmem), PH_DK, PH_CS, PH_CS);
  CUTE_UNROLL
  for (int cube = 0; cube < PH_DK / PH_CS; ++cube) {
    Copy::copy(smem + cube * PH_ELEM_A, gmem, desc, 0, cube * PH_CS);
  }
}
#endif

// PH_MIN_BLOCKS retains the original two-pass kernel's register-pressure
// tuning. The combined four-group kernel uses one block per CU because its
// persistent state requires 192 registers per thread.
// PH_COMPUTE_MT: when true, the kernel is reused to compute the per-segment
// state-transition operator mt[DK,DK] instead of the additive state ht[DK,DV].
// The chunk recurrence is affine in the state: h_out = ht + mt @ h_in. Running
// the SAME (probe-verified) GEMM chain with the V contribution forced to 0 and
// the register state initialized to the identity yields mt column-by-column
// (the DV axis is reinterpreted as the DK' index of mt; DV == DK == 128).
// Verified in numpy (mt_math_check.py): prepare_h's own affine decomposition
// reproduces the recurrence to rel ~4e-16.
template <typename Element, int PH_MIN_BLOCKS, bool PH_COMPUTE_MT = false,
          bool PH_COMBINED = false, bool PH_SPLIT_DV = false,
          int PH_STREAM_H = 0, int PH_TRANSPOSED = 0>
__global__ void __launch_bounds__(PhSmemLayout<PH_COMBINED, PH_SPLIT_DV,
                                           (PH_STREAM_H >= 3)>::NTHREADS,
                                  PH_MIN_BLOCKS)
prepare_h_cp_cute_kernel(
    const Element* __restrict__ k,              // [1, T, Hg, DK] fp16
    const Element* __restrict__ v,              // [1, T, H, DV] fp16
    const Element* __restrict__ a,              // [1, T, H, CS] fp16
    const float* __restrict__ g,               // [1, T, H] fp32
    const float* __restrict__ beta,            // [1, T, H] fp32
    Element* __restrict__ ht,                 // [batch*H, DK, DV] out
    Element* __restrict__ mt,                   // [batch*H, DK, DK] fp16 out (zeroed)
    const int* __restrict__ cu_seqlens,        // [batch+1]
    const int* __restrict__ num_warmup_chunks, // [batch*H]
    int batch_size, int num_tokens, int H, int Hg,
    int64_t v_token_stride, const Element* __restrict__ precomputed_x)
{
  using Layout = PhSmemLayout<PH_COMBINED, PH_SPLIT_DV, (PH_STREAM_H >= 3)>;
  static_assert(PH_TRANSPOSED >= 0 && PH_TRANSPOSED <= 2);
  static_assert(!PH_TRANSPOSED || (PH_COMBINED && PH_STREAM_H == 3));
  const int bid = blockIdx.x;
  const int batch_idx = bid / H;
  const int head_idx  = bid % H;
  const int kv_head_idx = head_idx / (H / Hg);
  const int tx = threadIdx.x;

  const int gid = tx / PH_WG;
  const int l   = tx % PH_WG;          // 0..127 : 4-warp M32 MMA group-local tid
  const int state_gid = PH_SPLIT_DV ? blockIdx.y :
      (PH_COMBINED ? (gid & 1) : gid);
  const int dv_base = state_gid * PH_DVH;
  const bool compute_mt = PH_COMPUTE_MT || (PH_COMBINED && gid >= 2);

  const int seq_start = cu_seqlens[batch_idx];
  const int seq_end   = cu_seqlens[batch_idx + 1];
  const int seg_len   = seq_end - seq_start;
  const int total_chunks = (seg_len + PH_CS - 1) / PH_CS;

  // Clamp the standalone scan result before using it in pointer arithmetic.
  // An invalid count could otherwise move
  // iter_seq_start below seq_start and issue out-of-bounds K/V/A loads.
  int num_iters = num_warmup_chunks[bid];
  if (num_iters < 0) num_iters = 0;
  if (num_iters > total_chunks) num_iters = total_chunks;
  // mt is needed only when decay never crosses -10 and the incoming state must
  // propagate through the entire CP segment. When the scan finds a shorter
  // suffix, fallback_mask is false and correct_h0 consumes ht without reading
  // mt. The host allocates mt with torch::empty, so those inactive tiles are
  // intentionally left unwritten rather than assumed to contain zero.
  const bool ph_full_warmup = (num_iters > 0) && (num_iters >= total_chunks);

  extern __shared__ char smem[];
  Element* K_all   = reinterpret_cast<Element*>(smem + PH_K);
  Element* V_all   = reinterpret_cast<Element*>(smem + PH_V);
  Element* A_all   = reinterpret_cast<Element*>(smem + PH_A);
#if !PH_K_SWZL
  Element* KT_buf  = reinterpret_cast<Element*>(smem + PH_KT);
#endif
#if PH_A_SWZL && !PH_XY_FACTOR
  Element* Ag_swz  = reinterpret_cast<Element*>(smem + PH_AGSWZ);
#endif
#if PH_XY_FACTOR
  Element* XT_buf  = reinterpret_cast<Element*>(smem + Layout::XT);
  float*  g_all    = reinterpret_cast<float*>(smem + Layout::GSH);
  float*  b_all    = reinterpret_cast<float*>(smem + Layout::BSH);
  float*  g_rev    = reinterpret_cast<float*>(smem + Layout::GREV);
#else
  Element* K_buf   = K_all;
  Element* V_buf   = V_all;
  Element* A_buf   = A_all;
  Element* hstmT_a = reinterpret_cast<Element*>(smem + Layout::HSTMT);
  Element* WT_a    = reinterpret_cast<Element*>(smem + Layout::WT);
  Element* gVdT_a  = reinterpret_cast<Element*>(smem + Layout::GVDT);
  float*  g_shared = reinterpret_cast<float*>(smem + Layout::GSH);
  float*  b_shared = reinterpret_cast<float*>(smem + Layout::BSH);
  float*  g_exp    = reinterpret_cast<float*>(smem + Layout::GEXP);
  float*  g_rev    = reinterpret_cast<float*>(smem + Layout::GREV);
#endif

  // Per-group intermediate buffers are needed only by the original
  // shared-memory recurrence retained under PH_XY_FACTOR=0.
#if !PH_XY_FACTOR
  Element* hstmT_g = hstmT_a + gid * PH_ELEM_HT;
  Element* WT_g    = WT_a    + gid * PH_ELEM_WT;
  Element* gVdT_g  = gVdT_a  + gid * PH_ELEM_WT;
#endif

  // The factorized recurrence keeps one DV16 tile per warp. The transposed
  // variant uses M4 and lane-local C->A packing; the original N4 uses C->B.
  // The unfactorized recurrence retains its M32 layout.
#if PH_XY_FACTOR
  std::conditional_t<(PH_TRANSPOSED != 0), PhMmaM4T<Element>, PhMmaN4T<Element>> state_mma;
#else
  fq::TiledMmaM32T<Element> state_mma;
#endif
  auto state_thr = state_mma.get_thread_slice(l);
  auto cId_csdv = cute::make_identity_tensor(
      cute::make_shape(cute::Int<PH_TRANSPOSED ? PH_DVH : PH_CS>{},
                       cute::Int<PH_TRANSPOSED ? PH_CS : PH_DVH>{}));
  auto tc_csdv = state_thr.partition_C(cId_csdv);    // [CS,DVh] or [DVh,CS]
#if PH_XY_FACTOR
  fq::TiledMmaM32T<Element> x_mma;
  auto x_thr = x_mma.get_thread_slice(l);
  auto cId_csx = cute::make_identity_tensor(
      cute::make_shape(cute::Int<PH_CS>{}, cute::Int<PH_CS>{}));
  auto tc_csx = x_thr.partition_C(cId_csx);          // C coords for one [CS,32] X slice
#endif
  auto cId_dkdv = cute::make_identity_tensor(
      cute::make_shape(cute::Int<PH_TRANSPOSED ? PH_DVH : PH_DK>{},
                       cute::Int<PH_TRANSPOSED ? PH_DK : PH_DVH>{}));
  auto tc_dkdv = state_thr.partition_C(cId_dkdv);    // [DK,DVh] or [DVh,DK]

  // ---- register-resident state h[DK, DVh] for this group, init 0. ----
#if PH_XY_FACTOR
  auto h_reg = fq::make_acc<PH_TRANSPOSED ? PH_DVH : PH_DK,
                            PH_TRANSPOSED ? PH_DK : PH_DVH,
                            decltype(state_mma), Element>(l);
#else
  auto h_reg = fq::make_acc<PH_DK, PH_DVH,
                            fq::TiledMmaM32T<Element>, Element>(l);
#endif
  cute::clear(h_reg);
  cute::Tensor rHreg = cute::recast<float>(h_reg);

  // mt-pass safety guard (mirrors the wmma prepare_mt policy): mt is only
  // mathematically well-defined when the warmup covers the FULL segment, so
  // that prepare_h implements the segment recurrence from state=0. When only
  // the tail is covered (num_iters < total_chunks), writing a partial-tail
  // transition matrix into mt would be applied naively by correct_h0 and
  // hurts accuracy. correct_h0 checks the same fallback mask before reading mt,
  // so an inactive transition tile is never consumed and needs no store.
  if (!PH_COMBINED && PH_COMPUTE_MT && !ph_full_warmup) {
    return;
  }
  const bool active_group = !compute_mt || ph_full_warmup;

  // mt mode: initialize the state to the DK x DK identity. The DV axis (width
  // PH_DVH per group, 2 groups -> PH_DV == PH_DK == 128) is reinterpreted as
  // the DK' column index of mt; global column = dv_base + dv.
  if (compute_mt && active_group) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
          auto crd = tc_dkdv(vv, m, n);
          int dk = cute::get<PH_TRANSPOSED ? 1 : 0>(crd);
                    int dv = cute::get<PH_TRANSPOSED ? 0 : 1>(crd);
          rHreg(vv, m, n) = (dk == (dv_base + dv)) ? 1.0f : 0.0f;
        }
      }
    }
  }

  // Anchor the warmup window on the segment's own chunk grid, which starts at
  // seq_start: A and the cumulative gate are built on that grid, so a segment
  // whose length is not a multiple of PH_CS carries its partial chunk at the
  // END. Counting back from seq_end instead both misaligns every chunk against
  // A/g and pushes the first read below seq_start, which is negative for the
  // first segment and silently reads the previous segment otherwise.
  const int iter_seq_start = seq_start + (total_chunks - num_iters) * PH_CS;

  const int64_t K_stride_w = (int64_t)Hg * PH_DK * 2;  // bytes
  const int64_t V_stride_w = v_token_stride * 2;
  const int64_t A_stride_w = (int64_t)H  * PH_CS * 2;

#if PH_XY_FACTOR
  // Prime slot 0 before entering the recurrence. Later iterations consume the
  // slot made ready at the previous chunk fence, so only this first load sits
  // wholly on the critical path.
  if (num_iters > 0) {
    const int cts0 = iter_seq_start;
    const int actual0 = min(PH_CS, seq_end - cts0);
    if (tx == 0 && actual0 > 0) {
      const Element* K_src =
          k + (int64_t)cts0 * Hg * PH_DK + kv_head_idx * PH_DK;
      const Element* V_src =
          v + (int64_t)cts0 * v_token_stride + head_idx * PH_DV;
      const Element* A_src =
          a + (int64_t)cts0 * H * PH_CS + head_idx * PH_CS;
#if PH_K_SWZL
      ph_aiu_swzl_write_k_cs32<Element>(
          K_all, K_src, /*row_stride=*/Hg * PH_DK, /*rows=*/actual0);
#else
      ph_aiu_load_2d_b16(K_all, K_src, PH_DK, actual0, K_stride_w);
#endif
      if constexpr (PH_TRANSPOSED) {
        // The transposed recurrence reads V^T fragments with the same
        // swizzled cube layout as K (see the V^T load below).
        ph_aiu_swzl_write_k_cs32<Element>(
            V_all, V_src, /*row_stride=*/(int)v_token_stride, /*rows=*/actual0);
      } else {
        ph_aiu_load_2d_b16(V_all, V_src, PH_DV, actual0, V_stride_w,
                           /*cube_w=*/PH_CS);
      }
      if constexpr (PH_STREAM_H >= 3) {
        ph_load_precomputed_x<Element>(XT_buf,
            precomputed_x + ((int64_t)(cts0 / PH_CS + batch_idx) * H + head_idx) * PH_ELEM_XT);
      } else {
        ph_aiu_load_2d_b16(A_all, A_src, PH_CS, actual0, A_stride_w,
                           /*cube_w=*/PH_CS);
      }
      ph_aiu_commit();
    }
    if constexpr (PH_STREAM_H >= 3) {
      // Precomputed X already carries beta, so the beta slots hold the
      // chunk's g_rev instead and no in-loop barrier is needed to publish it.
      if (tx < PH_CS) {
        const int pos = cts0 + tx;
        const float gv = pos < num_tokens && tx < actual0
            ? g[(int64_t)pos * H + head_idx] : 0.0f;
        g_all[tx] = gv;
        b_all[tx] = ph_g_rev_lane(gv, actual0, tx);
      }
    } else {
      for (int idx = tx; idx < PH_CS; idx += Layout::NTHREADS) {
        const int pos = cts0 + idx;
        const bool ok = pos < num_tokens && idx < actual0;
        g_all[idx] = ok ? g[(int64_t)pos * H + head_idx] : 0.0f;
        b_all[idx] = ok ? beta[(int64_t)pos * H + head_idx] : 0.0f;
      }
    }
    ph_aiu_wait_all();
    __syncthreads();
  }
#endif

  for (int i_s = 0; i_s < num_iters; ++i_s) {
    const int cts = iter_seq_start + i_s * PH_CS;      // chunk token start
    const int actual = (cts < seq_end) ? min(PH_CS, seq_end - cts) : 0;

#if PH_XY_FACTOR
    const int p = i_s & 1;
    Element* K_buf = K_all + p * PH_ELEM_K;
    Element* V_buf = V_all + p * PH_ELEM_V;
    Element* A_buf = A_all + p * PH_ELEM_A;
    float* g_shared = g_all + p * PH_CS;
    float* b_shared = b_all + p * PH_CS;
    float* g_rev_p = PH_STREAM_H >= 3 ? b_shared : g_rev;

    // Start the next input transfer before any current-chunk MMA. The alternate
    // slot prevents the AIU writes from racing K/V/A reads below. Scalar loads
    // use only the first 32 threads and land in registers; they are written to
    // the alternate slot just before the final chunk fence, so their global
    // latency no longer delays warp 0 at the chunk's first block barrier.
    float next_g_val = 0.0f, next_b_val = 0.0f;
    if (i_s + 1 < num_iters) {
      const int np = p ^ 1;
      const int next_cts = cts + PH_CS;
      const int next_actual = min(PH_CS, seq_end - next_cts);
      Element* next_K = K_all + np * PH_ELEM_K;
      Element* next_V = V_all + np * PH_ELEM_V;
      Element* next_A = A_all + np * PH_ELEM_A;
      if (tx == 0 && next_actual > 0) {
        const Element* K_src =
            k + (int64_t)next_cts * Hg * PH_DK + kv_head_idx * PH_DK;
        const Element* V_src =
            v + (int64_t)next_cts * v_token_stride + head_idx * PH_DV;
        const Element* A_src =
            a + (int64_t)next_cts * H * PH_CS + head_idx * PH_CS;
#if PH_K_SWZL
        ph_aiu_swzl_write_k_cs32<Element>(
            next_K, K_src, /*row_stride=*/Hg * PH_DK, /*rows=*/next_actual);
#else
        ph_aiu_load_2d_b16(next_K, K_src, PH_DK, next_actual, K_stride_w);
#endif
        if constexpr (PH_TRANSPOSED) {
          ph_aiu_swzl_write_k_cs32<Element>(
              next_V, V_src, /*row_stride=*/(int)v_token_stride,
              /*rows=*/next_actual);
        } else {
          ph_aiu_load_2d_b16(next_V, V_src, PH_DV, next_actual, V_stride_w,
                             /*cube_w=*/PH_CS);
        }
        if constexpr (PH_STREAM_H >= 3) {
          ph_load_precomputed_x<Element>(XT_buf + np * PH_ELEM_XT,
              precomputed_x + ((int64_t)(next_cts / PH_CS + batch_idx) * H + head_idx) * PH_ELEM_XT);
        } else {
          ph_aiu_load_2d_b16(next_A, A_src, PH_CS, next_actual, A_stride_w,
                             /*cube_w=*/PH_CS);
        }
        ph_aiu_commit();
      }
      static_assert(PH_CS <= Layout::NTHREADS);
      if (tx < PH_CS) {
        const int pos = next_cts + tx;
        if (pos < num_tokens && tx < next_actual) {
          next_g_val = g[(int64_t)pos * H + head_idx];
          if constexpr (PH_STREAM_H < 3)
            next_b_val = beta[(int64_t)pos * H + head_idx];
        }
      }
    }
#else
    // ---- PHASE A: AIU load K/V/A (.padz), cooperative scalar g/beta ----
    if (tx == 0 && actual > 0) {
      const Element* K_src = k + (int64_t)cts * Hg * PH_DK + kv_head_idx * PH_DK;
      const Element* V_src =
          v + (int64_t)cts * v_token_stride + head_idx * PH_DV;
      const Element* A_src = a + (int64_t)cts * H  * PH_CS + head_idx    * PH_CS;
#if defined(FLASHQLA_ARCH_100)
      // tc01: closed actlize pair (AIU_LOAD + TSM_LD_SWZL). Load into the
      // AIU-native cube-major SmemLayout defined in PhKVTraitsTc01 / PhATraitsTc01.
      // Each ph_load_*_tile_tc01 wraps make_mix_tensor + tiled_copy internally
      // (see helpers above); each helper's cute::copy call expands into InstNum
      // AIU instructions (K/V: 2 slices of 64 elems, A: 1 slice of 32 elems).
      ph_load_k_tile_tc01<Element>(K_buf, K_src, Hg * PH_DK, actual, tx);
      if (!PH_COMPUTE_MT || PH_COMBINED) {
        ph_load_v_tile_tc01<Element>(
            V_buf, V_src, v_token_stride, actual, tx);
      }
      ph_load_a_tile_tc01<Element>(A_buf, A_src, H  * PH_CS, actual, tx);
#elif PH_K_SWZL
      // K goes through the 32x32 AIU .swzl atom (with .padz zero-fill on partial).
      ph_aiu_swzl_write_k_cs32<Element>(K_buf, K_src, /*row_stride=*/Hg * PH_DK, /*rows=*/actual);
#else
      ph_aiu_load_2d_b16(K_buf, K_src, PH_DK, actual, K_stride_w);
#endif
#if !defined(FLASHQLA_ARCH_100)
      if (!PH_COMPUTE_MT || PH_COMBINED) {
        ph_aiu_load_2d_b16(V_buf, V_src, PH_DV, actual, V_stride_w,
                           /*cube_w=*/PH_CS);
      }
      ph_aiu_load_2d_b16(A_buf, A_src, PH_CS, actual, A_stride_w,
                         /*cube_w=*/PH_CS);
#endif
      ph_aiu_commit();
    }
    for (int idx = tx; idx < PH_CS; idx += Layout::NTHREADS) {
      int pos = cts + idx;
      bool ok = (pos >= 0 && pos < num_tokens && idx < actual);
      g_shared[idx] = ok ? g[(int64_t)pos * H + head_idx] : 0.0f;
      b_shared[idx] = ok ? beta[(int64_t)pos * H + head_idx] : 0.0f;
    }
    ph_aiu_wait_all();
    __syncthreads();
#endif

#if PH_XY_FACTOR
    // The upstream factorization removes the per-state Ag@W GEMM. All four
    // groups compute disjoint X^T slices; the following block barrier publishes
    // that one shared product. Persistent state stays in registers.
    const float g_last = g_shared[actual > 0 ? actual - 1 : 0];
    const float decay = exp2f(g_last * PH_LOG2E);
    if constexpr (PH_STREAM_H < 3) {
      for (int idx = tx; idx < PH_CS; idx += Layout::NTHREADS) {
        const float gi = idx < actual ? g_shared[idx] : g_last;
        g_rev[idx] = exp2f((g_last - gi) * PH_LOG2E);
      }
    }

    // X has four independent 32-column DK slices. Computing the whole
    // [CS,DK] product in gid 0 serializes every chunk: gids 1-3 can only wait at
    // the publication barrier even though their MMA resources are idle. Give
    // each 128-thread group one [CS,32] slice instead. The total MMA work and
    // numerical order along the reduction axis stay unchanged, while the X
    // phase moves off the single-group critical path.
    // Compute the stored form directly:
    //
    //   X^T = -K^T @ A @ diag(beta).
    //
    // K_buf already contains four contiguous swizzled 32x32 DK cubes, one per
    // group. The Trans=true TSM load used by the state update can therefore read
    // each K^T slice directly. This removes the duplicate 8-KiB plain-K AIU load
    // and its shared-memory footprint. Passing A^T as the TN MMA B operand makes
    // B^T equal A; beta is applied column-wise while publishing X^T.
    if constexpr (PH_STREAM_H < 3) {
      auto sA = fq::make_smem_tensor_h<PH_CS, PH_CS>(A_buf);
      auto sAT = fq::transpose_tensor(sA);
      auto x_acc = fq::make_acc<PH_CS, PH_CS,
                                fq::TiledMmaM32T<Element>, Element>(l);
      cute::clear(x_acc);
      fq::flashqla_gemm_ss_swzlA_trans_acc_m32<PH_CS, PH_CS,
          decltype(sAT), decltype(x_acc), Element>(
              K_buf + gid * PH_ELEM_A, sAT, l, x_acc);
      auto rX = cute::recast<float>(x_acc);
      auto sXT_write = fq::make_smem_swzA_cs32<PH_CS, PH_CS>(
          XT_buf + gid * PH_ELEM_A);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csx); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csx); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csx); ++vv) {
            auto crd = tc_csx(vv, m, n);
            const int dk = cute::get<0>(crd);
            const int cs = cute::get<1>(crd);
            sXT_write(dk, cs) =
                typename fq::ElementTraits<Element>::CuteType(
                    -b_shared[cs] * rX(vv, m, n));
          }
        }
      }
      __syncthreads();
    }

    if (active_group) {
      const int lane = l & 31;
      if constexpr (PH_TRANSPOSED) {
        // V does not depend on the state. Reading it ahead of K@h lets the
        // shared-memory latency overlap the first GEMM instead of stalling
        // the Y conversion. Each [DV16,CS16] tile of V^T is read as an MMA A
        // fragment straight from the swizzled V cubes. On tc02 the A layout
        // is the packed C layout, so register i holds the V values paired
        // with U^T elements 2i (low half) and 2i+1 (high half).
        static_assert(PH_DV == PH_DK, "V reuses the K cube writer");
        uint32_t v_pre[PH_CS / 16][4];
        if (!compute_mt) {
          using TsmVT = fq::PpuTsmLdSwzl<
              typename fq::ElementTraits<Element>::CuteType, 32, 32,
              /*Swap=*/false, /*Trans=*/true, /*InstNum=*/1>;
          const int dv_row = dv_base + (l / 32) * 16;
          CUTE_UNROLL
          for (int kt = 0; kt < PH_CS / 16; ++kt) {
            TsmVT::copy(reinterpret_cast<int*>(v_pre[kt]),
                reinterpret_cast<void*>(fq::ff_swzl_read_base<Element>(V_buf)),
                /*coord_h=*/kt * 16, /*coord_w=*/dv_row % 32,
                /*cube_in_stage=*/0, /*stage=*/dv_row / 32);
          }
        }
        auto u = fq::make_acc<PH_DVH, PH_CS, PhMmaM4T<Element>, Element>(l);
        ph_gemm_state_kB_m4<Element>(h_reg, K_buf, l, u);
        auto rU = cute::recast<float>(u);
        uint32_t y[PH_CS / 16][1][4];
        CUTE_UNROLL
        for (int kt = 0; kt < PH_CS / 16; ++kt) {
          uint32_t c[4];
          CUTE_UNROLL
          for (int i = 0; i < 4; ++i) {
            auto crd0 = tc_csdv(2 * i, 0, kt);
            auto crd1 = tc_csdv(2 * i + 1, 0, kt);
            const int cs0 = cute::get<1>(crd0);
            const int cs1 = cute::get<1>(crd1);
            float v0 = 0.0f, v1 = 0.0f;
            if (!compute_mt) ph_unpack_b16x2<Element>(v_pre[kt][i], v0, v1);
            c[i] = ph_pack_rn_b16x2<Element>(
                decay * rU(2 * i, 0, kt) - g_rev_p[cs0] * v0,
                decay * rU(2 * i + 1, 0, kt) - g_rev_p[cs1] * v1);
          }
          fq::cfrag_to_afrag_mma_u32<Element>(c, y[kt][0], lane);
        }
        CUTE_UNROLL
        for (int n = 0; n < PH_DK / 16; ++n) {
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) rHreg(i, 0, n) *= decay;
        }
        ph_gemm_yA_xtB_m4_acc<Element, PH_TRANSPOSED == 2 ? 2 : 4>(
            y, XT_buf + p * PH_ELEM_XT, h_reg);
      } else {
      // Convert each warp-local 16x16 tile of h to the tc02 B layout consumed
      // by K@h. The BF16/FP16 conversion is the same numerical boundary as the
      // removed shared-memory store/load path.
      // U = K@h; Y = decay*U - exp(g_last-g)*V. Convert Y directly to the B
      // fragments consumed by X^T@Y, preserving the old BF16/FP16 truncation
      // without a shared-memory round trip or a group barrier.
      auto u_acc = fq::make_acc<PH_CS, PH_DVH,
                                PhMmaN4T<Element>, Element>(l);
#if PH_K_SWZL
      if constexpr (PH_STREAM_H) {
        ph_gemm_swzlA_state_n4<PH_CS, PH_DK, Element, (PH_STREAM_H == 2)>(
            K_buf, l, u_acc, h_reg, decay);
      } else {
        uint32_t h_b_regs[PH_DK / 16][1][4];
        CUTE_UNROLL
        for (int kt = 0; kt < PH_DK / 16; ++kt) {
          uint32_t c_regs[4];
          CUTE_UNROLL
          for (int i = 0; i < 4; ++i) {
            const uint32_t lo = fq::element_bits16<Element>(
                fq::float_to_element<Element>(rHreg(2 * i, kt, 0)));
            const uint32_t hi = fq::element_bits16<Element>(
                fq::float_to_element<Element>(rHreg(2 * i + 1, kt, 0)));
            c_regs[i] = fq::pack_b16x2_u32(lo, hi);
          }
          fq::cfrag_to_bfrag_swizzle_u32(c_regs, h_b_regs[kt][0], lane);
        }
        ph_gemm_swzlA_regB_n4<PH_CS, PH_DK, Element>(
            K_buf, h_b_regs, l, u_acc);
      }
#else
#error "PH_XY_FACTOR register chain requires PH_K_SWZL"
#endif
      uint32_t y_b_regs[PH_CS / 16][1][4];
      {
        auto rU = cute::recast<float>(u_acc);
        CUTE_UNROLL
        for (int kt = 0; kt < PH_CS / 16; ++kt) {
          uint32_t c_regs[4];
          CUTE_UNROLL
          for (int i = 0; i < 4; ++i) {
            auto crd0 = tc_csdv(2 * i, kt, 0);
            auto crd1 = tc_csdv(2 * i + 1, kt, 0);
            const int cs0 = cute::get<0>(crd0);
            const int dv0 = cute::get<1>(crd0);
            const int cs1 = cute::get<0>(crd1);
            const int dv1 = cute::get<1>(crd1);
            const float v0 = compute_mt
                ? 0.0f
                : fq::element_to_float<Element>(
                      V_buf[cs0 * PH_DV + dv_base + dv0]);
            const float v1 = compute_mt
                ? 0.0f
                : fq::element_to_float<Element>(
                      V_buf[cs1 * PH_DV + dv_base + dv1]);
            const uint32_t lo = fq::element_bits16<Element>(
                fq::float_to_element<Element>(
                    decay * rU(2 * i, kt, 0) - g_rev_p[cs0] * v0));
            const uint32_t hi = fq::element_bits16<Element>(
                fq::float_to_element<Element>(
                    decay * rU(2 * i + 1, kt, 0) - g_rev_p[cs1] * v1));
            c_regs[i] = fq::pack_b16x2_u32(lo, hi);
          }
          fq::cfrag_to_bfrag_swizzle_u32(
              c_regs, y_b_regs[kt][0], lane);
        }
      }

      if constexpr (PH_STREAM_H != 2) {
        CUTE_UNROLL
        for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
          CUTE_UNROLL
          for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
            CUTE_UNROLL
            for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
              rHreg(vv, m, n) *= decay;
            }
          }
        }
      }
      ph_gemm_xt_swzl_regB_n4_acc<Element>(
          XT_buf + (PH_STREAM_H >= 3 ? p * PH_ELEM_XT : 0), y_b_regs, l, h_reg);
      }
    }
#else
    // g_exp = exp2(g), g_rev = exp2(g_last - g); decay = exp2(g_last).
    float g_last = g_shared[actual > 0 ? actual - 1 : 0];
    float decay  = exp2f(g_last * PH_LOG2E);
    for (int idx = tx; idx < PH_CS; idx += Layout::NTHREADS) {
      float gi = idx < actual ? g_shared[idx] : g_last;
      g_exp[idx] = exp2f(gi * PH_LOG2E);
      g_rev[idx] = exp2f((g_last - gi) * PH_LOG2E);
    }

    // ---- PHASE B: build KT (only if !PH_K_SWZL), snapshot hstmT (register),
    //      build Ag (either into Ag_swz swzl vec-store, or in-place over A_buf).
    //      One __syncthreads publishes them all. ----
#if !PH_K_SWZL
    // KT[dk,cs] = K[cs,dk]  (only needed when K is plain-linear)
#if defined(FLASHQLA_ARCH_100)
    // tc01: K_buf is AIU-native cube-major; read via the SmemLayout tensor
    // view so the transpose sees the logical [CS, DK] element even though the
    // physical layout is a tile_to_shape of 8x64 atoms.
    {
      auto sK = ph_k_view_tc01<Element>(K_buf);
      for (int idx = tx; idx < PH_DK * PH_CS; idx += Layout::NTHREADS) {
        int dk = idx / PH_CS, cs = idx % PH_CS;
        KT_buf[dk * PH_CS + cs] = fq::float_to_element<Element>(
            fq::element_to_float<Element>(*reinterpret_cast<const Element*>(&sK(cs, dk))));
      }
    }
#else
    for (int idx = tx; idx < PH_DK * PH_CS; idx += Layout::NTHREADS) {
      int dk = idx / PH_CS, cs = idx % PH_CS;
      KT_buf[dk * PH_CS + cs] = K_buf[cs * PH_DK + dk];
    }
#endif
#endif
#if PH_K_SWZL && defined(FLASHQLA_ARCH_150)
    auto sH_write = fq::make_smem_swzB<PH_DK, PH_DVH>(hstmT_g);
#endif
    // hstmT_g[dv,dk] = fp16(h[dk,dv])   (pre-decay snapshot from register-h)
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
          auto crd = tc_dkdv(vv, m, n);
          int dk = cute::get<PH_TRANSPOSED ? 1 : 0>(crd);
                    int dv = cute::get<PH_TRANSPOSED ? 0 : 1>(crd);
#if PH_K_SWZL && defined(FLASHQLA_ARCH_150)
          sH_write(dk, dv) =
              typename fq::ElementTraits<Element>::CuteType(rHreg(vv, m, n));
#else
          hstmT_g[dv * PH_DK + dk] = fq::float_to_element<Element>(rHreg(vv, m, n));
#endif
        }
      }
    }
#if PH_A_SWZL
    // Ag_swz: Swizzle<2,3,3> [CS,CS] fp16 via int4 (8-half) vec stores.
    {
      auto sAg_sw = fq::make_smem_swzA_cs32<PH_CS, PH_CS>(Ag_swz);
      constexpr int GRP_PER_ROW = PH_CS / 8;                 // 4 int4 groups per row
      for (int grp = tx; grp < PH_CS * GRP_PER_ROW; grp += Layout::NTHREADS) {
        const int r  = grp / GRP_PER_ROW;
        const int c0 = (grp % GRP_PER_ROW) * 8;
        Element packed[8];
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) {
          const int c2 = c0 + j;
          if (r >= c2 && r < actual && c2 < actual) {
            float av  = fq::element_to_float<Element>(A_buf[r * PH_CS + c2]);
            float gij = exp2f((g_shared[r] - g_shared[c2]) * PH_LOG2E);
            packed[j] = fq::float_to_element<Element>(av * gij * b_shared[c2]);
          } else {
            packed[j] = fq::float_to_element<Element>(0.0f);
          }
          A_buf[r * PH_CS + c2] = packed[j];
        }
        // Swizzle<2,3,3> preserves bits [0,3): &sAg_sw(r,c0) is 16B-aligned.
        *reinterpret_cast<int4*>(&sAg_sw(r, c0)) =
            *reinterpret_cast<const int4*>(packed);
      }
    }
#else
    // Ag[r,c] = A[r,c]*exp2(g[r]-g[c])*beta[c], tril; in-place on A_buf.
#if defined(FLASHQLA_ARCH_100)
    // tc01: A_buf is AIU-native cube-major. Read/write through the SmemLayout
    // view so both idx=r*PH_CS+c decomposition and the physical location match.
    {
      auto sA = ph_a_view_tc01<Element>(A_buf);
      for (int idx = tx; idx < PH_CS * PH_CS; idx += Layout::NTHREADS) {
        int r = idx / PH_CS, c = idx % PH_CS;
        Element* slot = reinterpret_cast<Element*>(&sA(r, c));
        if (r >= c && r < actual && c < actual) {
          float av  = fq::element_to_float<Element>(*slot);
          float gij = exp2f((g_shared[r] - g_shared[c]) * PH_LOG2E);
          *slot = fq::float_to_element<Element>(av * gij * b_shared[c]);
        } else {
          *slot = fq::float_to_element<Element>(0.0f);
        }
      }
    }
#else
    for (int idx = tx; idx < PH_CS * PH_CS; idx += Layout::NTHREADS) {
      int r = idx / PH_CS, c = idx % PH_CS;
      if (r >= c && r < actual && c < actual) {
        float av  = fq::element_to_float<Element>(A_buf[idx]);
        float gij = exp2f((g_shared[r] - g_shared[c]) * PH_LOG2E);
        A_buf[idx] = fq::float_to_element<Element>(av * gij * b_shared[c]);
      } else {
        A_buf[idx] = fq::float_to_element<Element>(0.0f);
      }
    }
#endif  // FLASHQLA_ARCH_100
#endif  // PH_A_SWZL
    __syncthreads();   // K(_swz)/KT / hstmT_g / Ag(_swz) / g_exp / g_rev visible

    if (active_group) {
      // ---- GEMM1: U[CS,DVh] = K[CS,DK] @ hstmT_g[DVh,DK]  ->  WT_g = (V - g_exp*U)^T
      {
      auto u_acc = fq::make_acc<PH_CS, PH_DVH, fq::TiledMmaM32T<Element>, Element>(l);
#if PH_K_SWZL
#if defined(FLASHQLA_ARCH_100)
      auto sH = fq::make_smem_tensor_h<PH_DVH, PH_DK>(hstmT_g);
      fq::flashqla_gemm_ss_swzlA_m32<PH_CS, PH_DK,
          decltype(sH), decltype(u_acc), Element>(K_buf, sH, l, u_acc);
#else
      auto sH = fq::make_smem_swzB_T<PH_DK, PH_DVH>(hstmT_g);
      // K is AIU .swzl-written: read A via TSM_LD_SWZL non-trans (bank-conflict-
      // free); B (hstmT) uses the matching transposed swizzle load. Clears u_acc.
      fq::flashqla_gemm_ss_swzlA_m32<PH_CS, PH_DK, decltype(sH), decltype(u_acc), Element>(
          K_buf, sH, l, u_acc, fq::SmemCopyAtomBTransT<Element>{});
#endif
#else
      auto sH = fq::make_smem_tensor_h<PH_DVH, PH_DK>(hstmT_g);
#if defined(FLASHQLA_ARCH_100)
      // tc01: K_buf is AIU-native cube-major; drive A via TSM_LD_SWZL smem-copy
      // atom (the paired reader) and B (hstmT) via the packed row-major view.
      // Inlined so K_buf can stay in its actlize SmemLayout.
      {
        using KTraits = PhKVTraitsTc01<Element>;
        using CuteElem = typename KTraits::CuteElem;
        using cute::_;
        auto sK = ph_k_view_tc01<Element>(K_buf);
        fq::TiledMmaM32T<Element> tmma;
        auto thr_m = tmma.get_thread_slice(l);

        auto rA_full = thr_m.partition_fragment_A(sK);
        auto smem_cp_A = cute::make_tiled_copy_A(typename KTraits::SmemCopyAtom{}, tmma);
        auto thr_cp_A = smem_cp_A.get_thread_slice(l);
        auto tAsA = thr_cp_A.partition_S(cute::make_mix_tensor_like(sK));
        auto tArA = thr_cp_A.retile_D(rA_full);

        // B side (hstmT) is a packed row-major smem, use the generic ldmatrix
        // atom via the default fq::SmemCopyAtomT for this element.
        auto rB_full = thr_m.partition_fragment_B(sH);
        auto smem_cp_B = cute::make_tiled_copy_B(fq::SmemCopyAtomT<Element>{}, tmma);
        auto thr_cp_B = smem_cp_B.get_thread_slice(l);
        auto tBsB = thr_cp_B.partition_S(sH);
        auto tBrB = thr_cp_B.retile_D(rB_full);

        cute::clear(u_acc);
        CUTE_UNROLL
        for (int k = 0; k < cute::size<2>(rA_full); ++k) {
          cute::copy(smem_cp_A, tAsA(_,_,k), tArA(_,_,k));
          cute::copy(smem_cp_B, tBsB(_,_,k), tBrB(_,_,k));
        }
        CUTE_UNROLL
        for (int k = 0; k < cute::size<2>(rA_full); ++k) {
          cute::gemm(tmma, rA_full(_,_,k), rB_full(_,_,k), u_acc);
        }
      }
#else
      auto sK = fq::make_smem_tensor_h<PH_CS, PH_DK>(K_buf);
      fq::flashqla_gemm_ss<decltype(sK), decltype(sH), decltype(u_acc), fq::TiledMmaM32T<Element>, Element>(
          sK, sH, l, u_acc);
#endif  // FLASHQLA_ARCH_100
#endif  // PH_K_SWZL
      auto rU = cute::recast<float>(u_acc);
#if defined(FLASHQLA_ARCH_100)
      // tc01: V_buf is AIU-native cube-major; read through the SmemLayout view.
      auto sV = ph_v_view_tc01<Element>(V_buf);
#endif
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
            float vval = compute_mt
                ? 0.0f
#if defined(FLASHQLA_ARCH_100)
                : fq::element_to_float<Element>(
                      *reinterpret_cast<const Element*>(&sV(cs, dv_base + dv)));
#else
                : fq::element_to_float<Element>(V_buf[cs * PH_DV + (dv_base + dv)]);
#endif
            float w = vval - g_exp[cs] * rU(vv, m, n);
            WT_g[dv * PH_CS + cs] = fq::float_to_element<Element>(w);   // transposed [DVh,CS]
          }
        }
      }
      }
      ph_group_bar(gid);   // WT_g visible within group before GEMM2

#if PH_REG_GVD_RECOMPUTE && PH_K_SWZL
    // ---- GEMM2→state-GEMM register-B path: each state warp recomputes the two
    //      16x16 gVd B-fragments it consumes, avoiding a gVdT_g smem round trip.
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv)
          rHreg(vv, m, n) *= decay;
      }
    }
    {
      uint32_t gvd_b_regs[2][2][4];
      const int warp = l / 32;
      const int warp_n = warp / 2;
      const int lane = l & 31;
      CUTE_UNROLL
      for (int k_half = 0; k_half < 2; ++k_half) {
        CUTE_UNROLL
        for (int n_tile = 0; n_tile < 2; ++n_tile) {
          const int dv_block = warp_n + n_tile * 2;
          auto vd16_acc = fq::make_acc16_1warp<Element>(lane);
          cute::clear(vd16_acc);
          CUTE_UNROLL
          for (int kk = 0; kk < 2; ++kk) {
            auto sAg16 = fq::block16_view<PH_CS>(A_buf, k_half, kk);
            auto sWT16 = fq::block16_view<PH_CS>(WT_g, dv_block, kk);
            fq::gemm16_ss_1warp_acc<decltype(sAg16), decltype(sWT16), decltype(vd16_acc), Element>(
                sAg16, sWT16, lane, vd16_acc);
          }
          fq::scale_acc16_rows<Element>(vd16_acc, g_rev, k_half * 16, lane);
          fq::acc16_to_bfrag_swizzle_u32<Element>(vd16_acc, gvd_b_regs[k_half][n_tile], lane);
        }
      }
      fq::flashqla_gemm_swzlA_trans_regB_acc_m32<PH_DK, PH_CS, decltype(h_reg), Element>(
          K_buf, gvd_b_regs, l, h_reg);
    }
#else
    // ---- GEMM2: Vd[CS,DVh] = Ag[CS,CS] @ WT_g[DVh,CS]  ->  gVdT_g = (g_rev*Vd)^T
    {
#if PH_A_SWZL
      auto sAg = fq::make_smem_swzA_cs32<PH_CS, PH_CS>(Ag_swz);
#else
      auto sAg = fq::make_smem_tensor_h<PH_CS, PH_CS>(A_buf);
#endif
      auto sWT = fq::make_smem_tensor_h<PH_DVH, PH_CS>(WT_g);
      auto vd_acc = fq::make_acc<PH_CS, PH_DVH, fq::TiledMmaM32T<Element>, Element>(l);
#if defined(FLASHQLA_ARCH_100)
      // Ag_swz is a software-swizzled buffer written elementwise above, not an
      // AIU swizzled tile. Reading A_buf through TSM_LD_SWZL paired a linear
      // AIU write with a swizzled read and completely scrambled prepare_h.
      // Use the ordinary ldmatrix/CuTe path over the actual Ag_swz view, exactly
      // like fused_fwd GEMM2. The tc01 plain helper dispatches via cute::gemm.
      fq::flashqla_gemm_ss<decltype(sAg), decltype(sWT), decltype(vd_acc),
                            fq::TiledMmaM32T<Element>, Element>(
          sAg, sWT, l, vd_acc);
#else
      fq::flashqla_gemm_ss<decltype(sAg), decltype(sWT), decltype(vd_acc), fq::TiledMmaM32T<Element>, Element>(
          sAg, sWT, l, vd_acc);
#endif
      auto rVd = cute::recast<float>(vd_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
            gVdT_g[dv * PH_CS + cs] = fq::float_to_element<Element>(g_rev[cs] * rVd(vv, m, n));
          }
        }
      }
    }
    ph_group_bar(gid);   // gVdT_g visible within group before state-GEMM

    // ---- decay register-h, then state-GEMM: h += K^T @ (g_rev*Vd) ----
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv)
          rHreg(vv, m, n) *= decay;
      }
    }
    {
      auto sVg = fq::make_smem_tensor_h<PH_DVH, PH_CS>(gVdT_g);
#if PH_K_SWZL
      // Read K^T directly from the .swzl K buffer via TSM_LD_SWZL Trans=true --
      // no KT_buf, no scalar transpose.  Accumulates into h_reg.
      fq::flashqla_gemm_ss_swzlA_trans_acc_m32<PH_DK, PH_CS, decltype(sVg), decltype(h_reg), Element>(
          K_buf, sVg, l, h_reg);
#else
      auto sKT = fq::make_smem_tensor_h<PH_DK, PH_CS>(KT_buf);
      fq::flashqla_gemm_ss_acc<decltype(sKT), decltype(sVg), decltype(h_reg), fq::TiledMmaM32T<Element>, Element>(
          sKT, sVg, l, h_reg);
#endif
    }
#endif
    }
#endif  // PH_XY_FACTOR
#if PH_XY_FACTOR
    if (i_s + 1 < num_iters) {
      if (tx < PH_CS) {
        g_all[(p ^ 1) * PH_CS + tx] = next_g_val;
        if constexpr (PH_STREAM_H >= 3) {
          b_all[(p ^ 1) * PH_CS + tx] = ph_g_rev_lane(
              next_g_val, min(PH_CS, seq_end - (cts + PH_CS)), tx);
        } else {
          b_all[(p ^ 1) * PH_CS + tx] = next_b_val;
        }
      }
      ph_aiu_wait_all();
    }
#endif
    // Completes all current shared reads and publishes the prefetched alternate
    // slot before the next loop iteration rebinds K_buf/V_buf/A_buf.
    __syncthreads();
  }

  // ---- store final state: ht (additive) or mt (transition operator) ----
  Element* out_ptr = compute_mt ? mt : ht;
  const int out_w  = compute_mt ? PH_DK : PH_DV;   // both == 128
  int64_t out_base = (int64_t)bid * PH_DK * out_w;
  if (active_group) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
          auto crd = tc_dkdv(vv, m, n);
          int dk = cute::get<PH_TRANSPOSED ? 1 : 0>(crd);
                    int dv = cute::get<PH_TRANSPOSED ? 0 : 1>(crd);
          out_ptr[out_base + dk * out_w + dv_base + dv] =
              fq::float_to_element<Element>(rHreg(vv, m, n));
        }
      }
    }
  }
}

}  // namespace flashqla_cute

// ============================================================
// Host launch
// ============================================================
namespace {

constexpr int PH_MAX_CACHED_DEVICES = 64;

template <typename Element, int MinBlocks, bool ComputeMT,
          bool Combined, bool SplitDV, int StreamH = 0, int Transposed = 0>
void prepare_h_configure_dynamic_smem(int device_index, int smem_bytes) {
  auto configure = [smem_bytes]() {
    if (hggcFuncSetAttribute(
            flashqla_cute::prepare_h_cp_cute_kernel<
                Element, MinBlocks, ComputeMT, Combined, SplitDV, StreamH, Transposed>,
            hggcFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes) != hggcSuccess) {
      hggcGetLastError();
    }
  };
  if (device_index < 0 || device_index >= PH_MAX_CACHED_DEVICES) {
    configure();
    return;
  }
  // This static belongs to exactly one kernel specialization. Indexing by
  // logical device makes the first-launch configuration race-free without a
  // runtime API call on warmed launches.
  static std::once_flag configured[PH_MAX_CACHED_DEVICES];
  std::call_once(configured[device_index], configure);
}

}  // namespace

void launch_prepare_h_cp_cute(
    const void* k, const void* v, const void* a,
    const float* g, const float* beta,
    void* ht, void* mt,
    const int* cu_seqlens, const int* num_warmup_chunks,
    int batch_size, int num_tokens, int H, int Hg, int dtype_tag,
    int64_t v_token_stride,
    int device_index, int num_cu, int compute_capability,
    void* precomputed_x, hggcStream_t stream)
{
  if (compute_capability != 80 && compute_capability != 89) {
    throw std::runtime_error(
        "launch_prepare_h_cp_cute: unsupported compute capability");
  }
  int grid_size = batch_size * H;
  dim3 grid(grid_size);
  dim3 tc01_grid(grid_size, 2);
  const bool is_tc01 = compute_capability == 80;
  const bool stream_target = !is_tc01 && num_cu == 39 && precomputed_x &&
      num_tokens == 8192 && ((H == 12 && Hg == 4 && batch_size == 6) ||
                             (H == 6 && Hg == 2 && batch_size == 12));
  const int stream_mode = stream_target ? 3 : 0;
#if defined(FLASHQLA_ARCH_150) && PH_XY_FACTOR
  if (stream_mode >= 3 && H == flashqla_cute::PH_X_GROUP * Hg) {
    dim3 x_grid((num_tokens + 31) / 32 + batch_size, Hg);
    constexpr int x_threads = flashqla_cute::PH_WG * flashqla_cute::PH_X_GROUP;
    if (dtype_tag == 0) {
      flashqla_cute::ph_precompute_x_group_kernel<__half><<<x_grid, x_threads, 0, stream>>>(
          reinterpret_cast<const __half*>(k), reinterpret_cast<const __half*>(a), beta,
          reinterpret_cast<__half*>(precomputed_x), cu_seqlens, batch_size, H, Hg);
    } else {
      flashqla_cute::ph_precompute_x_group_kernel<__ppu_bfloat16><<<x_grid, x_threads, 0, stream>>>(
          reinterpret_cast<const __ppu_bfloat16*>(k), reinterpret_cast<const __ppu_bfloat16*>(a), beta,
          reinterpret_cast<__ppu_bfloat16*>(precomputed_x), cu_seqlens, batch_size, H, Hg);
    }
  } else if (stream_mode >= 3) {
    dim3 x_grid((num_tokens + 31) / 32 + batch_size, H);
    if (dtype_tag == 0) {
      flashqla_cute::ph_precompute_x_kernel<__half><<<x_grid, 128, 0, stream>>>(
          reinterpret_cast<const __half*>(k), reinterpret_cast<const __half*>(a), beta,
          reinterpret_cast<__half*>(precomputed_x), cu_seqlens, batch_size, H, Hg);
    } else {
      flashqla_cute::ph_precompute_x_kernel<__ppu_bfloat16><<<x_grid, 128, 0, stream>>>(
          reinterpret_cast<const __ppu_bfloat16*>(k), reinterpret_cast<const __ppu_bfloat16*>(a), beta,
          reinterpret_cast<__ppu_bfloat16*>(precomputed_x), cu_seqlens, batch_size, H, Hg);
    }
  }
#endif

#define PH_LAUNCH_TC01(ElementType, MinBlocks, ComputeMT) \
  do { \
    prepare_h_configure_dynamic_smem< \
        ElementType, MinBlocks, ComputeMT, false, true>( \
            device_index, flashqla_cute::PH_SMEM_TC01_SPLIT_DV); \
    flashqla_cute::prepare_h_cp_cute_kernel< \
        ElementType, MinBlocks, ComputeMT, false, true> \
        <<<tc01_grid, dim3(128), flashqla_cute::PH_SMEM_TC01_SPLIT_DV, stream>>>( \
        reinterpret_cast<const ElementType*>(k), reinterpret_cast<const ElementType*>(v), \
        reinterpret_cast<const ElementType*>(a), g, beta, \
        reinterpret_cast<ElementType*>(ht), reinterpret_cast<ElementType*>(mt), \
        cu_seqlens, num_warmup_chunks, \
        batch_size, num_tokens, H, Hg, v_token_stride, \
        reinterpret_cast<const ElementType*>(precomputed_x)); \
  } while (0)

#define PH_LAUNCH_COMBINED_T(ElementType, StreamH, Transposed) \
  do { \
    constexpr int MinBlocks = StreamH ? 2 : 1; \
    constexpr int SmemBytes = flashqla_cute::PhSmemLayout<true, false, (StreamH >= 3)>::TOTAL; \
    prepare_h_configure_dynamic_smem< \
        ElementType, MinBlocks, false, true, false, StreamH, Transposed>( \
            device_index, SmemBytes); \
    flashqla_cute::prepare_h_cp_cute_kernel<ElementType, MinBlocks, false, true, false, StreamH, Transposed> \
        <<<grid, dim3(512), SmemBytes, stream>>>( \
        reinterpret_cast<const ElementType*>(k), reinterpret_cast<const ElementType*>(v), \
        reinterpret_cast<const ElementType*>(a), g, beta, \
        reinterpret_cast<ElementType*>(ht), reinterpret_cast<ElementType*>(mt), \
        cu_seqlens, num_warmup_chunks, \
        batch_size, num_tokens, H, Hg, v_token_stride, \
        reinterpret_cast<const ElementType*>(precomputed_x)); \
  } while (0)

  if (is_tc01) {
    constexpr int PH_CU_COUNT = 39;
    if (dtype_tag == 0) {
      if (grid_size > PH_CU_COUNT * 2) {
        PH_LAUNCH_TC01(__half, 3, false);
        PH_LAUNCH_TC01(__half, 3, true);
      } else {
        PH_LAUNCH_TC01(__half, 2, false);
        PH_LAUNCH_TC01(__half, 2, true);
      }
    } else {
      if (grid_size > PH_CU_COUNT * 2) {
        PH_LAUNCH_TC01(__ppu_bfloat16, 3, false);
        PH_LAUNCH_TC01(__ppu_bfloat16, 3, true);
      } else {
        PH_LAUNCH_TC01(__ppu_bfloat16, 2, false);
        PH_LAUNCH_TC01(__ppu_bfloat16, 2, true);
      }
    }
  } else if (dtype_tag == 0) {
    if (stream_target) PH_LAUNCH_COMBINED_T(__half, 3, 2);
    else PH_LAUNCH_COMBINED_T(__half, 0, 0);
  } else {
    if (stream_target) PH_LAUNCH_COMBINED_T(__ppu_bfloat16, 3, 2);
    else PH_LAUNCH_COMBINED_T(__ppu_bfloat16, 0, 0);
  }

#undef PH_LAUNCH_COMBINED_T
#undef PH_LAUNCH_TC01
}
