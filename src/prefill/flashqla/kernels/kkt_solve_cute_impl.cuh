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
 * FlashQLA dual-architecture PPU CuTe — Kernel (1) kkt_solve
 *
 * Computes per (chunk, value head) the inverse of the unit lower-triangular
 * matrix built from the beta-scaled Gram matrix:
 *
 *     KKT = K @ K^T               (CS x CS, fp32 accumulate)
 *     KKT[i,j] *= beta[j]
 *     M[i,j]  = 1 (i==j), -KKT[i,j] (i>j), 0 (otherwise)
 *     A = M^{-1}                  (blocked recursive triangular inverse)
 *
 * Execution model: PERSISTENT GRID + AIU ASYNC DOUBLE-BUFFER
 *   - Grid is persistent: a fixed number of blocks grid-stride over the flat
 *     (chunk, head) task list.
 *   - Block = 128 threads (4 warps, ALL doing compute — no warp-spec).
 *   - K/beta are DOUBLE-BUFFERED: while all 4 warps compute the current task,
 *     the AIU hardware engine asynchronously fetches the next task's K tile
 *     into the alternate buffer (true HW-level overlap, no thread budget lost).
 *   - Sync: __syncthreads() only (no mbarrier, no NamedBarrier).
 *
 * Triangular inverse: sm90 blocked-recursive scheme (4x4 blocks, distance
 * unrolled) using single-warp 16x16 tc02 MMA.
 *
 * This is an internal implementation header.  It is instantiated once for
 * each host launch policy by kkt_solve_cute_tc01.cu and
 * kkt_solve_cute_tc02.cu.  The PPU compiler still selects the matching device
 * implementation from __HGGC_ARCH__ for each image in the fat binary.
 */
#ifndef FLASHQLA_KKT_NAMESPACE
#error "Define FLASHQLA_KKT_NAMESPACE before including kkt_solve_cute_impl.cuh"
#endif
#ifndef FLASHQLA_KKT_KERNEL
#error "Define FLASHQLA_KKT_KERNEL before including kkt_solve_cute_impl.cuh"
#endif
#ifndef FLASHQLA_KKT_LAUNCHER
#error "Define FLASHQLA_KKT_LAUNCHER before including kkt_solve_cute_impl.cuh"
#endif
#ifndef FLASHQLA_KKT_HOST_ARCH_100
#error "Define FLASHQLA_KKT_HOST_ARCH_100 before including kkt_solve_cute_impl.cuh"
#endif

#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <cstdint>
#include <hggc_awbarrier.h>
#include <mutex>

#include "flashqla_cute_traits.cuh"

namespace FLASHQLA_KKT_NAMESPACE {

namespace fq = flashqla;

// ---- KKT_K_SWZL: route the K chunk through the 32x32 AIU .swzl atom (128B-XOR
//      swizzled smem) and compute the Gram matrix KKT = K@K^T with the dual-swzl
//      TiledMmaM32 helper (flashqla_gemm_ss_swzlAB_m32), mirroring the
//      FF_K_SWZL/FF_Q_SWZL experience in fused_fwd. Bank-conflict-free ldmatrix
//      reads on both operands, no KT_buf transpose. Default OFF (A/B gated):
//      kkt_solve is IALU-bound (tri-inverse), so the Gram GEMM speedup may not
//      move E2E; measured empirically. cvta on the .swzl write dst is MANDATORY
//      here — kkt_solve runs at ~3 blk/CU, so a raw generic ptr would clobber a
//      sibling CTA's TSM slice (same hazard class as fused_fwd's FF_SWZL_CVTA). ----

constexpr int KKT_CS = 32;    // chunk size
constexpr int KKT_DK = 128;   // key head dim
constexpr int KKT_NThreads = 128;  // 4 warps, all compute

#if defined(FLASHQLA_ARCH_100)
// A 32-float row is exactly 128 B, so every row of the tc01 fp32 M/A tiles
// starts on the same TSM bank.  One padding float rotates successive rows
// through all 32 banks while leaving the logical matrix 32x32.
constexpr int KKT_FP32_LD = KKT_CS + 1;
#else
constexpr int KKT_FP32_LD = KKT_CS;
#endif

// ---- double-buffered smem byte layout (~25KB < 256KB TSM) ------------------
//   KKT_buf (fp32 Gram, 4KB) REINSTATED (2026-07-24): the register-fused
//   Gram→beta→M build (store_gram_build_M_m32) was measured to REGRESS the
//   kernel 60% (300µs→488µs, CU 72.7%→45.4%) with NO register spill (vreg=32,
//   STACK=0) and NO dynamic layouts — the partition_C scattered fp16 smem write
//   + per-fragment-element beta[j] gather defeats the SIMD-friendly linear
//   store_acc_to_smem_fp32 + strided M-build loop.  Since the head phase is NOT
//   on the critical path (serial tri-inverse tail dominates), the fused build
//   buys no wall-clock and costs 60% — reverted to the known-good 300µs path.
constexpr int KKT_OFF_K0   = 0;                                   // K_buf[0] 8KB
constexpr int KKT_OFF_K1   = KKT_OFF_K0  + KKT_CS * KKT_DK * 2;    // K_buf[1] 8KB
constexpr int KKT_OFF_KKT  = KKT_OFF_K1  + KKT_CS * KKT_DK * 2;    // KKT_buf 4KB
#if defined(FLASHQLA_ARCH_100)
constexpr int KKT_OFF_MH   = KKT_OFF_KKT + KKT_CS * KKT_FP32_LD * 4; // fp32 inverse
constexpr int KKT_OFF_AH   = KKT_OFF_MH;                              // same tile
constexpr int KKT_OFF_SBUF = KKT_OFF_AH  + KKT_CS * KKT_FP32_LD * 4;
// The tc01 fp32 solve does not materialize M or use the tc02 blocked-MMA
// scratch. It reconstructs M[row,k] from Gram[row,k] * beta[row] inside the
// recurrence, preserving Gram for all value heads mapped to one key head.
constexpr int KKT_SBUF_BYTES = 0;
#else
constexpr int KKT_OFF_MH   = KKT_OFF_KKT + KKT_CS * KKT_CS * 4;    // Mh 2KB
constexpr int KKT_OFF_AH   = KKT_OFF_MH  + KKT_CS * KKT_CS * 2;    // Ah 2KB
constexpr int KKT_OFF_SBUF = KKT_OFF_AH  + KKT_CS * KKT_CS * 2;    // Sbuf 2KB
constexpr int KKT_SBUF_BYTES = 4 * 16 * 16 * 2;
#endif
constexpr int KKT_OFF_B0   = KKT_OFF_SBUF + KKT_SBUF_BYTES;         // beta[0]
constexpr int KKT_OFF_B1   = KKT_OFF_B0  + KKT_CS * 4;             // beta[1]
constexpr int KKT_SMEM     = KKT_OFF_B1  + KKT_CS * 4;
constexpr int KKT_SMEM_TC01 = 25088;
constexpr int KKT_SMEM_TC02 = 26880;
#if defined(FLASHQLA_ARCH_100)
static_assert(KKT_SMEM == KKT_SMEM_TC01);
#else
static_assert(KKT_SMEM == KKT_SMEM_TC02);
#endif

constexpr int KKT_PAIR_SMEM_BYTES = 24576;
constexpr int KKT_QUAD_SMEM_BYTES = 29824;
constexpr int KKT_OCT_SMEM_BYTES = 47232;

#if defined(FLASHQLA_ARCH_100)
// Four adjacent value heads can share one K Gram tile when they map to the
// same K head.  Each warp owns one fp32 inverse, so the four recurrences run
// concurrently without the row-by-row block barriers required when two heads
// share one inverse tile.
constexpr int KKT_QUAD_OFF_K = 0;
constexpr int KKT_QUAD_OFF_GRAM = KKT_QUAD_OFF_K + KKT_CS * KKT_DK * 2;
constexpr int KKT_QUAD_OFF_A =
    KKT_QUAD_OFF_GRAM + KKT_CS * KKT_FP32_LD * 4;
constexpr int KKT_QUAD_A_BYTES = KKT_CS * KKT_FP32_LD * 4;
constexpr int KKT_QUAD_OFF_BETA = KKT_QUAD_OFF_A + 4 * KKT_QUAD_A_BYTES;
constexpr int KKT_QUAD_SMEM = KKT_QUAD_OFF_BETA + 4 * KKT_CS * 4;
static_assert(KKT_QUAD_SMEM == KKT_QUAD_SMEM_BYTES);
constexpr int KKT_OCT_OFF_K = 0;
constexpr int KKT_OCT_OFF_GRAM = KKT_OCT_OFF_K + KKT_CS * KKT_DK * 2;
constexpr int KKT_OCT_OFF_A =
    KKT_OCT_OFF_GRAM + KKT_CS * KKT_FP32_LD * 4;
constexpr int KKT_OCT_OFF_BETA = KKT_OCT_OFF_A + 8 * KKT_QUAD_A_BYTES;
constexpr int KKT_OCT_SMEM = KKT_OCT_OFF_BETA + 8 * KKT_CS * 4;
static_assert(KKT_OCT_SMEM == KKT_OCT_SMEM_BYTES);
#endif

#if defined(FLASHQLA_ARCH_150)
// tc02 can solve two adjacent value heads from one shared K Gram tile whenever
// both heads map to the same K head.  M/A remain private to each two-warp group;
// A0/A1 reuse the Gram tile after both groups finish materializing M, reducing
// the layout to 24KB and permitting ten resident CTAs per CU on 890P. tc01 already
// reuses one Gram tile across every value head mapped to a K head above and must
// keep its separate fp32 recurrence/layout.
constexpr int KKT_PAIR_WG       = 64;
constexpr int KKT_PAIR_OFF_K0   = 0;
constexpr int KKT_PAIR_OFF_K1   = KKT_PAIR_OFF_K0 + KKT_CS * KKT_DK * 2;
constexpr int KKT_PAIR_OFF_KKT  = KKT_PAIR_OFF_K1 + KKT_CS * KKT_DK * 2;
constexpr int KKT_PAIR_OFF_M0   = KKT_PAIR_OFF_KKT + KKT_CS * KKT_CS * 4;
constexpr int KKT_PAIR_OFF_M1   = KKT_PAIR_OFF_M0 + KKT_CS * KKT_CS * 2;
constexpr int KKT_PAIR_OFF_A0   = KKT_PAIR_OFF_KKT;
constexpr int KKT_PAIR_OFF_A1   = KKT_PAIR_OFF_A0 + KKT_CS * KKT_CS * 2;
constexpr int KKT_PAIR_SMEM     = KKT_PAIR_OFF_M1 + KKT_CS * KKT_CS * 2;
static_assert(KKT_PAIR_SMEM == KKT_PAIR_SMEM_BYTES);
#endif

__device__ __forceinline__ int kkt_min(int a, int b) { return a < b ? a : b; }

#if defined(FLASHQLA_ARCH_150)
__forceinline__ __device__ void kkt_pair_group_bar(int group) {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.bar.sync %0, %1;\n" :: "r"(group + 1), "n"(KKT_PAIR_WG) : "memory");
#else
  (void)group;
  __syncthreads();
#endif
}
#endif

template <bool PerValueHead>
__device__ __forceinline__ void kkt_task_coords(
    const int* cu_seqlens, const int* chunk_indices, int H, int Hg, int task,
    int& seq_start, int& seq_end, int& local_chunk, int& head_idx) {
#if defined(FLASHQLA_ARCH_100)
    const int task_heads = PerValueHead ? H : Hg;
    const int global_chunk = task / task_heads;
    const int task_head = task % task_heads;
    head_idx = PerValueHead ? task_head : task_head * (H / Hg);
#else
    (void)PerValueHead;
    const int global_chunk = task / H;
    head_idx = task % H;
#endif
    const int seq_idx = chunk_indices ? chunk_indices[global_chunk * 2] : 0;
    local_chunk = chunk_indices ? chunk_indices[global_chunk * 2 + 1] : global_chunk;
    seq_start = cu_seqlens[seq_idx];
    seq_end = cu_seqlens[seq_idx + 1];
}

#if defined(FLASHQLA_ARCH_100)
__device__ __forceinline__ void kkt_quad_task_coords(
    const int* cu_seqlens, const int* chunk_indices, int H, int task,
    int& seq_start, int& seq_end, int& local_chunk, int& head0) {
    const int head_quads = H / 4;
    const int global_chunk = task / head_quads;
    head0 = (task % head_quads) * 4;
    const int seq_idx = chunk_indices ? chunk_indices[global_chunk * 2] : 0;
    local_chunk = chunk_indices ? chunk_indices[global_chunk * 2 + 1] : global_chunk;
    seq_start = cu_seqlens[seq_idx];
    seq_end = cu_seqlens[seq_idx + 1];
}
#endif

#if defined(FLASHQLA_ARCH_150)
__device__ __forceinline__ void kkt_pair_task_coords(
    const int* cu_seqlens, const int* chunk_indices, int H, int task,
    int& seq_start, int& seq_end, int& local_chunk, int& head0) {
    const int head_pairs = H / 2;
    const int global_chunk = task / head_pairs;
    head0 = (task % head_pairs) * 2;
    const int seq_idx = chunk_indices ? chunk_indices[global_chunk * 2] : 0;
    local_chunk = chunk_indices ? chunk_indices[global_chunk * 2 + 1] : global_chunk;
    seq_start = cu_seqlens[seq_idx];
    seq_end = cu_seqlens[seq_idx + 1];
}
#endif

// ============================================================================
// PPU1.0/1.1 (tc01) K-chunk pipeline traits.
//
// Replaces the tc02 swzl AIU write + swzl TSM read pair (which is unavailable
// on tc01, see docs/flashqla_ppu1.0_port_notes.md) with the closed actlize
// pair (PPU0010_AIU_LOAD, PPU0010_TSM_LD_SWZL) that both
// ppu_mma_builder.inl and ppu_mma_aiu_multistage.hpp use for tc01 GEMMs.
//
// Shapes (KKT_CS=32, KKT_DK=128, b16):
//   Block_MN         = KKT_CS = 32
//   Block_K          = KKT_DK = 128
//   BlockContSize    = 128 * 2 = 256B  (> 128B cap)
//   AiuContByteSize  = 128B (capped)
//   AiuContElemSize  = 64 elements
//   InstNum          = Block_K / AiuContElemSize = 2
//   bits_per_aiu     = Block_MN * AiuContElemSize * 16 = 32768
//
// smem for K_buf is *the same 8KB* as the pre-refactor packed tile; only the
// internal layout differs (AIU-native cube-major via tile_to_shape). K_buf0 /
// K_buf1 double-buffer offsets in the smem plan below are unchanged.
// ============================================================================
#if defined(FLASHQLA_ARCH_100)
template <typename Element>
struct KktKTraitsTc01 {
  using CuteElem = typename fq::ElementTraits<Element>::CuteType;
  static constexpr int BLK_MN = KKT_CS;
  static constexpr int BLK_K  = KKT_DK;

  static constexpr int BlockContSize   = BLK_K * static_cast<int>(sizeof(CuteElem));
  static constexpr int AiuContByteSize = BlockContSize > 128 ? 128 : BlockContSize;
  static constexpr int AiuContElemSize = AiuContByteSize / static_cast<int>(sizeof(CuteElem));
  static constexpr int InstNum         = BLK_K / AiuContElemSize;
  static constexpr int BitsPerAIU      = BLK_MN * AiuContElemSize *
                                         static_cast<int>(sizeof(CuteElem)) * 8;

  // Swzl=true: the K tile is read back through the swizzled TSM_LD_SWZL atom
  // (ppu.ldmatrix...swzl), so the AIU write MUST be swizzled too -- this is the
  // closed pair cutlass' ppu_mma_builder uses (its AIU_LOAD defaults Swzl=true).
  // A linear (Swzl=false) write makes the plain SmemLayout view look correct but
  // feeds the swzl reader scrambled data (Gram wrong from A(1,0)).
  using CopyInst = cute::PPU0010_AIU_LOAD<
      cute::C<BitsPerAIU>, CuteElem, /*Trans=*/false, /*Swzl=*/true>;

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

// The K_buf physical footprint under the AIU-native SmemLayout: number of
// Element slots. tile_to_shape does not introduce padding beyond what the
// atom shape already implies, so this equals BLK_MN * BLK_K.
template <typename Element>
static constexpr int kkt_k_buf_elems() {
  return cute::size(typename KktKTraitsTc01<Element>::SmemLayout{});
}

// Issue one AIU-driven load of a [rows<=KKT_CS, KKT_DK] gmem tile into K_buf.
// The .padz atom zero-fills rows [rows, KKT_CS) automatically. Single thread
// issues the copy (AIU only needs one issuer); the caller commits.
template <typename Element>
__forceinline__ __device__ void kkt_load_k_tile_tc01(
    Element* K_dst, const Element* K_src, int row_stride_elems, int rows,
    int tx) {
  using Traits = KktKTraitsTc01<Element>;
  using CuteElem = typename Traits::CuteElem;

  // gmem mix tensor over the CTA tile (32 rows, KKT_DK cols) with strided
  // access to the full matrix. dim_h -> KKT_CS, dim_w-in-desc is the row
  // pitch in elements (rides in stride.get<0>). counting_stride encodes
  // per-element coords so partition_S can feed the atom's copy_unpack.
  auto gA = cute::make_mix_tensor(
      cute::make_gmem_ptr(reinterpret_cast<const CuteElem*>(K_src)),
      cute::make_layout(
          cute::make_shape(cute::Int<Traits::BLK_MN>{}, cute::Int<Traits::BLK_K>{}),
          cute::make_counting_stride<2, 1, 1>()));

  auto sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElem*>(K_dst)),
      typename Traits::SmemLayout{});

  typename Traits::GmemTiledCopy tiled_copy;
  // Descriptor: full-tile shape + real row pitch (in elements). The atom uses
  // this together with per-copy coords from partition_S to place InstNum AIU
  // instructions correctly. `rows` is passed as the tile bound so .padz zero-
  // fills the overhang when rows < KKT_CS.
  tiled_copy.desc_.template init<CuteElem, /*Trans=*/false,
                                  Traits::BLK_MN, Traits::BLK_K>(
      reinterpret_cast<const uint8_t*>(K_src),
      /*MN=*/rows, /*K=*/Traits::BLK_K,
      cute::make_stride(row_stride_elems, cute::_1{}));

  if (tx == 0) {
    auto thr = tiled_copy.get_slice(0);
    auto tAgA = thr.partition_S(gA);
    auto tAsA = thr.partition_D(sA);
    cute::copy(tiled_copy, tAgA, tAsA);
  }
}
#endif  // FLASHQLA_ARCH_100

// ---- AIU 2D b16 async load (padz: rows < dim_w auto zero-padded) ------------
__forceinline__ __device__ void kkt_aiu_load_2d_b16(
    void* smem_ptr, const void* gmem_ptr,
    int dim_c, int dim_w, int64_t stride_w_bytes) {
#if defined(FLASHQLA_ARCH_150)
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
      " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
      :: "r"(smem_addr), "l"(gmem_ptr),
         "r"(dim_c), "r"(dim_w), "r"(1),
         "r"(dim_c), "r"(dim_w), "r"(1),
         "l"(stride_w_bytes), "l"(int64_t(0)),
         "r"(0), "r"(0), "r"(0),
         "r"(0)
      : "memory");
#elif defined(FLASHQLA_ARCH_100)
  // See ff_aiu_load_2d_b16 in fused_fwd_cute.cu for the tc01 axis-remap
  // rationale (no explicit stride operand; stride-in-elements rides in dim_c).
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
#endif
}
__forceinline__ __device__ void kkt_aiu_commit() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}
__forceinline__ __device__ void kkt_aiu_wait_all() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}
// wait_group<N>: drain this thread's cp.async FIFO until ≤ N committed groups
// remain (keeps N prefetches in flight). Mirrors fused_fwd's ff_aiu_wait_group.
template <int N>
__forceinline__ __device__ void kkt_aiu_wait_group() {
#if defined(__HGGC_ARCH__)
  asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// ---- AIU .swzl 2D write of a K chunk [rows, KKT_DK] into KKT_DK/32 = 4 col
//      cubes (CUBE_H=CUBE_W=32) of 128B-XOR-swizzled smem. Pairs with
//      fq::flashqla_gemm_ss_swzlAB_m32 (both Gram operands read swzl). .padz
//      auto zero-fills partial cube rows [rows,32). Issued by a SINGLE thread
//      and shares the current cp.async group (committed by kkt_aiu_commit).
//      NOTE: row_stride is in ELEMENTS (Hg*KKT_DK), matching AiuDesc.init. ----
template <typename Element>
__forceinline__ __device__ void kkt_aiu_swzl_write_k_cs32(
    Element* smem, const Element* gmem, int row_stride_elems, int rows) {
#if defined(__HGGC_ARCH__)
  using AiuWriteCube = flashqla::PpuAiuLoad<
      32 * 32 * 16, typename fq::ElementTraits<Element>::CuteType, /*Trans=*/false,
      /*CUBE_H=*/32, /*CUBE_W=*/32, /*Swzl=*/true>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t*>(gmem), /*h=*/rows, /*w=*/KKT_DK,
            /*stride_w=*/row_stride_elems);
  CUTE_UNROLL
  for (int cube = 0; cube < KKT_DK / 32; ++cube) {
    // MANDATORY cvta at ~3 blk/CU: strip the per-CTA shared-window base so each
    // co-resident CTA's .swzl DMA targets its OWN TSM slice (see FF_SWZL_CVTA).
    uint32_t dst_shared = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem + cube * 32 * 32));
    AiuWriteCube::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst_shared)),
                       reinterpret_cast<const void*>(gmem), desc,
                       /*coord_w=*/cube * 32, /*coord_h=*/0);
  }
#else
  (void)smem; (void)gmem; (void)row_stride_elems; (void)rows;
#endif
}

// ---- Per-cube .swzl write: write ONLY column-cube `cube` of the K chunk.
//      Splitting the KKT_DK/32=4 cubes across the 4 warps (warp w issues cube w)
//      means each warp's cp.async FIFO tracks only ITS OWN DMA, so wait_group<N>
//      per-warp drains just that warp's cube. The prior single-thread variant
//      serialized all 4 cubes and left 3 warps' FIFOs empty (breaking the
//      per-thread wait_group discipline the fused_fwd pipeline relies on). ----
template <typename Element>
__forceinline__ __device__ void kkt_aiu_swzl_write_k_cube(
    Element* smem, const Element* gmem, int row_stride_elems, int rows, int cube) {
#if defined(__HGGC_ARCH__)
  using AiuWriteCube = flashqla::PpuAiuLoad<
      32 * 32 * 16, typename fq::ElementTraits<Element>::CuteType, /*Trans=*/false,
      /*CUBE_H=*/32, /*CUBE_W=*/32, /*Swzl=*/true>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t*>(gmem), /*h=*/rows, /*w=*/KKT_DK,
            /*stride_w=*/row_stride_elems);
  uint32_t dst_shared = static_cast<uint32_t>(
      __cvta_generic_to_shared(smem + cube * 32 * 32));
  AiuWriteCube::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst_shared)),
                     reinterpret_cast<const void*>(gmem), desc,
                     /*coord_w=*/cube * 32, /*coord_h=*/0);
#else
  (void)smem; (void)gmem; (void)row_stride_elems; (void)rows; (void)cube;
#endif
}

// ---- off-diagonal block: A[i][j] = -Dinv[i] . sum_{k=j}^{i-1} M[i][k].A[k][j]
template <typename Element>
__device__ __forceinline__ void kkt_offdiag_block(
    Element* Mh, Element* Ah, Element* Sw, int i, int j, int lane) {
    using namespace cute;
    auto accS = fq::make_acc16_1warp<Element>(lane);
    clear(accS);
    for (int k = j; k < i; ++k) {
        auto sM = fq::block16_view<KKT_CS>(Mh, i, k);
        auto sA = fq::transpose_tensor(fq::block16_view<KKT_CS>(Ah, k, j));
        fq::gemm16_ss_1warp_acc<decltype(sM), decltype(sA), decltype(accS), Element>(sM, sA, lane, accS);
    }
    auto rS = recast<float>(accS);
    CUTE_UNROLL
    for (int e = 0; e < size(rS); ++e) rS(e) = -rS(e);
    uint32_t s_b_regs[4];
    fq::acc16_to_bfrag_swizzle_u32<Element>(accS, s_b_regs, lane);
    auto accA = fq::make_acc16_1warp<Element>(lane);
    clear(accA);
    auto sD = fq::block16_view<KKT_CS>(Ah, i, i);
    fq::gemm16_sr_1warp_acc<decltype(sD), decltype(accA), Element>(sD, s_b_regs, lane, accA);
    fq::store_acc16_fp16_1warp<Element>(accA, Ah + i * 16 * KKT_CS + j * 16, KKT_CS, lane);
    (void)Sw;
}

// ---- HEAD: K@K^T + beta scale + build M. On tc02 this consumes K_buf and
//      beta_buf. tc01 preserves Gram and retains beta through its recurrence;
//      its one-task-per-CTA launch therefore never takes the p2-prefetch path. -
template <typename Element>
__device__ __forceinline__ void kkt_compute_head(
    Element* K_buf, float* beta_buf, float* KKT_buf, Element* Mh, Element* Ah,
    int actual, int tx, int warp, int lane) {
    using namespace cute;

    // Step2: KKT = K @ K^T  (fp32 accumulate into KKT_buf).
    // K_buf is .swzl-written: both Gram operands read from the swizzled buffer
    // via the TN-atom dual-swzl helper (clears tC internally).
    if (tx < KKT_NThreads) {
        auto tC = fq::make_acc<KKT_CS, KKT_CS, fq::TiledMmaM32T<Element>, Element>(tx);
#if defined(FLASHQLA_ARCH_100)
        // tc01: K_buf is written by the paired PPU0010 AIU_LOAD tiled
        // copy into an AIU-native cube-major SmemLayout (see kkt_load_k_tile_tc01
        // + KktKTraitsTc01 above). Read both Gram operands through the matching
        // TSM_LD_SWZL smem-copy atom -- the hardware unpacks the AIU layout
        // into the MMA fragment. Manual clear(tC) here because we're inlining
        // the gemm.
        {
            using Traits = KktKTraitsTc01<Element>;
            using CuteElem = typename Traits::CuteElem;
            using cute::_;

            auto sK = cute::make_tensor(
                cute::make_smem_ptr(reinterpret_cast<CuteElem*>(K_buf)),
                typename Traits::SmemLayout{});

            fq::TiledMmaM32T<Element> mma;
            auto thr_mma = mma.get_thread_slice(tx);

            // A: partition sK as A[M=KKT_CS, K=KKT_DK], fill A fragments via
            // TSM_LD_SWZL smem->reg copy. Copy_Traits<TSM_LD_SWZL> requires
            // its source tensor to carry per-element coords (it reads
            // src.data().coord_ in copy_unpack), so wrap sK with
            // make_mix_tensor_like before partition_S -- matches the pattern
            // in ppu_mma_aiu_multistage.hpp:281.
            auto rA_full = thr_mma.partition_fragment_A(sK);
            auto smem_copy_A = cute::make_tiled_copy_A(typename Traits::SmemCopyAtom{}, mma);
            auto thr_copy_A = smem_copy_A.get_thread_slice(tx);
            auto tAsA = thr_copy_A.partition_S(cute::make_mix_tensor_like(sK));
            auto tArA = thr_copy_A.retile_D(rA_full);

            // B: partition sK as B[N=KKT_CS, K=KKT_DK]. TN atom means the
            // second operand is a [N, K] view (same underlying data for K@K^T).
            auto rB_full = thr_mma.partition_fragment_B(sK);
            auto smem_copy_B = cute::make_tiled_copy_B(typename Traits::SmemCopyAtom{}, mma);
            auto thr_copy_B = smem_copy_B.get_thread_slice(tx);
            auto tBsB = thr_copy_B.partition_S(cute::make_mix_tensor_like(sK));
            auto tBrB = thr_copy_B.retile_D(rB_full);

            cute::clear(tC);
            // Stream one K-slice at a time. The MMA accumulation order is
            // unchanged, while fragment live ranges are substantially shorter.
            CUTE_UNROLL
            for (int k = 0; k < cute::size<2>(rA_full); ++k) {
                cute::copy(smem_copy_A, tAsA(_,_,k), tArA(_,_,k));
                cute::copy(smem_copy_B, tBsB(_,_,k), tBrB(_,_,k));
                cute::gemm(mma, rA_full(_,_,k), rB_full(_,_,k), tC);
            }
        }
#else
        fq::flashqla_gemm_ss_swzlAB_m32<KKT_CS, KKT_CS, KKT_DK, decltype(tC), Element>(
            K_buf, K_buf, tx, tC);
#endif
        fq::store_acc_to_smem_fp32<fq::TiledMmaM32T<Element>>(
            tC, KKT_buf, KKT_CS, KKT_CS, KKT_FP32_LD, tx);
    }
    __syncthreads();

    // Step2b+Step3 merged: fold the per-ROW beta scale into the M-build loop.
    // The same thread owns the same idx in both, so the beta multiply is applied
    // inline.  Simple strided (idx=tx, +128) coalesced smem access — this linear
    // path beats the register-fused partition_C variant on PPU1.5 (see layout
    // comment above).  Last consumer of KKT_buf + beta_buf.
    //
    // WY-factor convention (matches upstream FlashQLA torch_kkt_fwd + torch_solve):
    //   L[i,j] = beta_i * <k_i,k_j>   for i>j   (beta on ROW i, NOT column j)
    //   A = M^{-1},  M = I + tril(L, -1)         (delta-rule "I + L", NOT "I - L")
    // The gate exp(g_i-g_j) is applied later in fused_fwd as a diagonal
    // conjugation of A (D_g A D_g^{-1}), and the per-column beta_d is applied
    // there too (u = A @ (beta*v)); both conjugate cleanly onto this row-beta,
    // I+L inverse.  The previous "-scaled"/beta_buf[j] build produced (I - L)^{-1}
    // with column-beta, which is the mirror-image WY factor and diverged from the
    // official reference (o_cos ~0.95).  See docs analysis for the derivation.
#if defined(FLASHQLA_ARCH_100)
    // Keep Gram intact. tc01 reconstructs the lower-triangular M coefficients
    // from Gram and the current value head's beta inside kkt_compute_tri_tc01.
#else
    for (int idx = tx; idx < KKT_CS * KKT_CS; idx += KKT_NThreads) {
        int i = idx / KKT_CS, j = idx % KKT_CS;
        const int pidx = i * KKT_FP32_LD + j;
        float scaled = KKT_buf[pidx] * beta_buf[i];
        float m = (i == j) ? 1.0f
                : ((i > j && i < actual && j < actual) ? scaled : 0.0f);
        Mh[idx] = fq::float_to_element<Element>(m);
        Ah[idx] = fq::float_to_element<Element>(0.0f);
    }
    __syncthreads();
#endif
}

#if defined(FLASHQLA_ARCH_100)
// tc01 correctness path: invert the full 32x32 unit-lower matrix in fp32.
// A_fp32 occupies one padded tile adjacent to the preserved Gram tile. Each row is
// solved by 32 four-lane subgroups across all four warps: one subgroup per
// output column and a four-way reduction over k. This removes the ACU-observed
// state where three warps waited at the block barrier for warp 0's scalar tail.
template <typename Element>
__device__ __forceinline__ void kkt_compute_tri_tc01(
    Element* A_out, const float* Gram, const float* beta_buf, float* A_fp32,
    int chunk_start, int actual, int head_idx, int num_tokens, int H,
    int tx, int warp, int lane) {
    for (int idx = tx; idx < KKT_CS * KKT_FP32_LD; idx += KKT_NThreads)
        A_fp32[idx] = 0.0f;
    __syncthreads();

    const int col = warp * 8 + (lane >> 2);
    const int subgroup_lane = lane & 3;
    // A column never migrates between warps: one four-lane subgroup owns the
    // whole forward substitution for that column.  Have that owner publish its
    // diagonal so the row recurrence has no cross-warp dependency.
    if (subgroup_lane == 0)
        A_fp32[col * KKT_FP32_LD + col] = 1.0f;
    __syncwarp();

    for (int row = 0; row < actual; ++row) {
        float sum = 0.0f;
        if (col < row) {
            for (int k = col + subgroup_lane; k < row; k += 4)
                sum += Gram[row * KKT_FP32_LD + k] *
                       A_fp32[k * KKT_FP32_LD + col];
        }
        sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 1, 4);
        if (subgroup_lane == 0 && col < row)
            A_fp32[row * KKT_FP32_LD + col] = -(sum * beta_buf[row]);
        // Only the owning warp consumes this column on the next row.  A block
        // barrier here serialized all four independent column groups 32 times.
        __syncwarp();
    }

    // Each warp owns eight complete columns, so it can start publishing them as
    // soon as its recurrence finishes instead of waiting for the heaviest warp.
    // Map the 32 lanes to a 4x8 tile: each eight-lane group writes one contiguous
    // 16-byte row segment, while the outer loop covers all 32 rows.
    for (int row_base = 0; row_base < actual; row_base += 4) {
        const int row = row_base + (lane >> 3);
        const int store_col = warp * 8 + (lane & 7);
        if (row < actual && (chunk_start + row) < num_tokens) {
            A_out[((int64_t)(chunk_start + row)) * H * KKT_CS +
                  head_idx * KKT_CS + store_col] =
                fq::float_to_element<Element>(A_fp32[row * KKT_FP32_LD + store_col]);
        }
    }
    __syncthreads();
}

// Two GVA value heads share Gram but have different beta rows. Pack the first
// inverse in the lower triangle of A_fp32 and the second inverse transposed in
// the upper triangle. The common unit diagonal is shared. Warps 0/1 solve head
// 0 while warps 2/3 solve head 1, retaining the four-lane reduction order of
// the single-head path without allocating a second fp32 tile.
template <typename Element>
__device__ __forceinline__ void kkt_compute_tri_pair_tc01(
    Element* A_out, const float* Gram,
    const float* beta0, const float* beta1, float* A_fp32,
    int chunk_start, int actual, int head0, int num_tokens, int H,
    int tx, int warp, int lane) {
    for (int idx = tx; idx < KKT_CS * KKT_FP32_LD; idx += KKT_NThreads)
        A_fp32[idx] = 0.0f;
    __syncthreads();
    if (tx < KKT_CS)
        A_fp32[tx * KKT_FP32_LD + tx] = 1.0f;
    __syncthreads();

    const int head_slot = warp >> 1;
    const int local_warp = warp & 1;
    const int subgroup_lane = lane & 3;
    const float* beta_buf = head_slot ? beta1 : beta0;

    for (int row = 0; row < actual; ++row) {
        CUTE_UNROLL
        for (int col_group = 0; col_group < 2; ++col_group) {
            const int col = local_warp * 8 + (lane >> 2) + col_group * 16;
            float sum = 0.0f;
            if (col < row) {
                for (int k = col + subgroup_lane; k < row; k += 4) {
                    const float a = head_slot
                        ? A_fp32[col * KKT_FP32_LD + k]
                        : A_fp32[k * KKT_FP32_LD + col];
                    sum += Gram[row * KKT_FP32_LD + k] * a;
                }
            }
            sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
            sum += __shfl_down_sync(0xffffffff, sum, 1, 4);
            if (subgroup_lane == 0 && col < row) {
                const int aidx = head_slot
                    ? col * KKT_FP32_LD + row
                    : row * KKT_FP32_LD + col;
                A_fp32[aidx] = -(sum * beta_buf[row]);
            }
        }
        // The two inverses occupy mirrored halves of one TSM tile. tc01 needs
        // a block-wide publication point before the next row when independent
        // warps write those halves concurrently; warp-only sync produced rare
        // late-row corruption (first visible at rows 27/29/31).
        __syncthreads();
    }

    const int head_lane = local_warp * 32 + lane;
    for (int idx = head_lane; idx < actual * KKT_CS; idx += 64) {
        const int row = idx / KKT_CS;
        const int col = idx % KKT_CS;
        if ((chunk_start + row) < num_tokens) {
            const float a = (col <= row)
                ? (head_slot
                    ? A_fp32[col * KKT_FP32_LD + row]
                    : A_fp32[row * KKT_FP32_LD + col])
                : 0.0f;
            A_out[((int64_t)(chunk_start + row)) * H * KKT_CS +
                  (head0 + head_slot) * KKT_CS + col] =
                fq::float_to_element<Element>(a);
        }
    }
    __syncthreads();
}

// One warp solves all 32 columns for one value head.  Its eight four-lane
// subgroups cover eight columns at a time; four column groups complete a row.
// Every dependency is within the same inverse and warp, so no block-wide
// synchronization is needed during the recurrence.
template <typename Element>
__device__ __forceinline__ void kkt_compute_tri_quad_tc01(
    Element* A_out, const float* Gram, const float* beta,
    float* A_fp32, int chunk_start, int actual, int head,
    int num_tokens, int H, int lane) {
    // Active column groups overwrite every strict-lower element before use.
    A_fp32[lane * KKT_FP32_LD + lane] = 1.0f;
    __syncwarp();

    const int subgroup_lane = lane & 3;
    CUTE_UNROLL
    for (int row = 0; row < KKT_CS; ++row) {
        if (row < actual) {
            CUTE_UNROLL
            for (int col_group = 0; col_group < (row + 7) / 8; ++col_group) {
                const int col = (lane >> 2) + col_group * 8;
                float sum = 0.0f;
                if (col < row) {
                    for (int k = col + subgroup_lane; k < row; k += 4)
                        sum += Gram[row * KKT_FP32_LD + k] *
                               A_fp32[k * KKT_FP32_LD + col];
                }
                sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
                sum += __shfl_down_sync(0xffffffff, sum, 1, 4);
                if (subgroup_lane == 0 && col < row)
                    A_fp32[row * KKT_FP32_LD + col] = -(sum * beta[row]);
            }
            __syncwarp();
        }
    }

    for (int idx = lane; idx < actual * KKT_CS; idx += 32) {
        const int row = idx / KKT_CS;
        const int col = idx % KKT_CS;
        if ((H != 32 || col <= row) &&
            (chunk_start + row) < num_tokens) {
            A_out[((int64_t)(chunk_start + row)) * H * KKT_CS +
                  head * KKT_CS + col] =
                fq::float_to_element<Element>(A_fp32[row * KKT_FP32_LD + col]);
        }
    }
}
#endif

// ---- TAIL: blocked-recursive triangular inverse + store A.  Reads Mh, writes
//      Ah, stores A_out.  This is the long serial critical path (warps 2/3 idle
//      on CS=32) that the just-issued task_{ti+2} DMA hides under. -----------
template <typename Element>
__device__ __forceinline__ void kkt_compute_tri(
    Element* A_out, Element* Mh, Element* Ah, Element* Sbuf,
    int chunk_start, int actual, int head_idx, int num_tokens, int H,
    int tx, int warp, int lane) {
    using namespace cute;

    // Step4a: invert diagonal 16x16 blocks (KKT_CS=32 -> 2 blocks, warps 0,1)
    if (lane < 16 && warp < (KKT_CS / 16)) {
        const int base = warp * 16;
        const int l = lane;
        Ah[(base + l) * KKT_CS + (base + l)] = fq::float_to_element<Element>(1.0f);
        for (int r = l + 1; r < 16; ++r) {
            float sum = 0.0f;
            for (int k = l; k < r; ++k)
                sum += fq::element_to_float<Element>(Mh[(base + r) * KKT_CS + (base + k)]) *
                       fq::element_to_float<Element>(Ah[(base + k) * KKT_CS + (base + l)]);
            Ah[(base + r) * KKT_CS + (base + l)] = fq::float_to_element<Element>(-sum);
        }
    }
    __syncthreads();

    // Step4b: off-diagonal by distance (KKT_CS=32 -> only (i=1,j=0) survives)
    Element* Sw = Sbuf + warp * 16 * 16;
#if (KKT_CS / 16) >= 3
    { int i=-1,j=-1;
      if(warp==0){i=1;j=0;} else if(warp==1){i=2;j=1;} else if(warp==2){i=3;j=2;}
      if(i>=0) kkt_offdiag_block<Element>(Mh,Ah,Sw,i,j,lane); }
    __syncthreads();
    { int i=-1,j=-1;
      if(warp==0){i=2;j=0;} else if(warp==1){i=3;j=1;}
      if(i>=0) kkt_offdiag_block<Element>(Mh,Ah,Sw,i,j,lane); }
    __syncthreads();
    if(warp==0) kkt_offdiag_block<Element>(Mh,Ah,Sw,3,0,lane);
    __syncthreads();
    // Step5: store A (unchanged for CS>=48)
    for (int idx = tx; idx < KKT_CS * KKT_CS; idx += KKT_NThreads) {
        int i = idx / KKT_CS, j = idx % KKT_CS;
        if (i < actual && (chunk_start + i) < num_tokens) {
            A_out[((int64_t)(chunk_start + i)) * H * KKT_CS + head_idx * KKT_CS + j] =
                Ah[i * KKT_CS + j];
        }
    }
    __syncthreads();
#else
    if (warp == 0) kkt_offdiag_block<Element>(Mh, Ah, Sw, 1, 0, lane);
    __syncthreads();
    // Step5: store A
    for (int idx = tx; idx < KKT_CS * KKT_CS; idx += KKT_NThreads) {
        int i = idx / KKT_CS, j = idx % KKT_CS;
        if (i < actual && (chunk_start + i) < num_tokens) {
            A_out[((int64_t)(chunk_start + i)) * H * KKT_CS + head_idx * KKT_CS + j] =
                Ah[i * KKT_CS + j];
        }
    }
    __syncthreads();
#endif
}

#if defined(FLASHQLA_ARCH_150)
// tc02 paired path: all four warps build one K Gram tile, then warps 0/1 and
// 2/3 independently solve the two value heads.  The launcher enables it only
// when each adjacent pair is guaranteed to remain within one K-head group.
template <typename Element>
__device__ __forceinline__ void kkt_compute_pair_head(
    Element* K_buf, const float* beta, float* KKT_buf,
    Element* M0, Element* A0, Element* M1, Element* A1,
    int chunk_start, int head0, int actual, int H, int tx) {
    using namespace cute;

    {
        auto tC = fq::make_acc<KKT_CS, KKT_CS, fq::TiledMmaM32T<Element>, Element>(tx);
        fq::flashqla_gemm_ss_swzlAB_m32<KKT_CS, KKT_CS, KKT_DK, decltype(tC), Element>(
            K_buf, K_buf, tx, tC);
        fq::store_acc_to_smem_fp32<fq::TiledMmaM32T<Element>>(
            tC, KKT_buf, KKT_CS, KKT_CS, KKT_CS, tx);
    }
    __syncthreads();

    const int group = tx / KKT_PAIR_WG;
    const int local_thread = tx % KKT_PAIR_WG;
    Element* Mh = group ? M1 : M0;
    Element* Ah = group ? A1 : A0;
    const int row = local_thread >> 1;
    const int col0 = (local_thread & 1) * 16;
    const float beta_row = (row < actual)
        ? beta[(int64_t)(chunk_start + row) * H + head0 + group]
        : 0.0f;
    CUTE_UNROLL
    for (int dj = 0; dj < 16; ++dj) {
        const int col = col0 + dj;
        if (row > col) {
            const int idx = row * KKT_CS + col;
            const float m = (row < actual)
                ? KKT_buf[idx] * beta_row : 0.0f;
            Mh[idx] = fq::float_to_element<Element>(m);
        }
    }
    (void)Ah;
}

template <typename Element>
__device__ __forceinline__ void kkt_compute_pair_tri(
    Element* A_out, Element* M0, Element* A0, Element* M1, Element* A1,
    int chunk_start, int actual, int head0, int num_tokens, int H,
    int tx) {
    const int group = tx / KKT_PAIR_WG;
    const int local_thread = tx % KKT_PAIR_WG;
    const int local_warp = local_thread >> 5;
    const int lane = local_thread & 31;
    Element* Mh = group ? M1 : M0;
    Element* Ah = group ? A1 : A0;
    const int head = head0 + group;

    // The off-diagonal MMA reads the complete A00/A11 tiles. Initialize only
    // their upper triangles; the diagonal solve below overwrites every diagonal
    // and lower entry, while A01 is never consumed.
    if (local_warp < 2) {
        const int base = local_warp * 16;
        CUTE_UNROLL
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            const int row = idx / 16;
            const int col = idx % 16;
            if (row < col) {
                Ah[(base + row) * KKT_CS + base + col] =
                    fq::float_to_element<Element>(0.0f);
            }
        }
    }

    if (lane < 16 && local_warp < 2) {
        const int base = local_warp * 16;
        const int col = lane;
        Ah[(base + col) * KKT_CS + (base + col)] =
            fq::float_to_element<Element>(1.0f);
        for (int row = col + 1; row < 16; ++row) {
            float sum = 0.0f;
            for (int k = col; k < row; ++k) {
                sum += fq::element_to_float<Element>(
                           Mh[(base + row) * KKT_CS + (base + k)]) *
                       fq::element_to_float<Element>(
                           Ah[(base + k) * KKT_CS + (base + col)]);
            }
            Ah[(base + row) * KKT_CS + (base + col)] =
                fq::float_to_element<Element>(-sum);
        }
    }
    kkt_pair_group_bar(group);

    if (local_warp == 0 && actual > 16)
        kkt_offdiag_block<Element>(Mh, Ah, nullptr, 1, 0, lane);
    kkt_pair_group_bar(group);

    for (int idx = local_thread; idx < KKT_CS * KKT_CS; idx += KKT_PAIR_WG) {
        const int row = idx / KKT_CS;
        const int col = idx % KKT_CS;
        if (row < actual && (chunk_start + row) < num_tokens) {
            Element value;
            if (col < row) {
                value = Ah[idx];
            } else {
                value = fq::float_to_element<Element>(
                    col == row ? 1.0f : 0.0f);
            }
            A_out[((int64_t)(chunk_start + row)) * H * KKT_CS +
                  head * KKT_CS + col] = value;
        }
    }
}
#endif

// ---- Issue one task's inputs into a double-buffer slot (split AIU + beta).
//      The K chunk's 4 column-cubes are split across the 4 warps: warp w's
//      lane0 fires cube w's .swzl DMA. ALL 128 threads then call commit once,
//      so every thread's cp.async FIFO advances by exactly one group per task
//      — the invariant wait_group<N> relies on (a warp only ever waits on its
//      OWN cube). beta is a plain cooperative smem fill (not AIU); it is fenced
//      by the main loop's post-wait __syncthreads before any compute reads it. --
template <typename Element, bool PerValueHead>
__device__ __forceinline__ void kkt_issue_task(
    Element* K_dst, float* beta_dst,
    const Element* K, const float* beta,
    const int* cu_seqlens, const int* chunk_indices,
    int H, int Hg, int task,
    int warp, int lane, int tx) {
    int seq_start, seq_end, c, h;
    kkt_task_coords<PerValueHead>(
        cu_seqlens, chunk_indices, H, Hg, task,
        seq_start, seq_end, c, h);
    const int hg = h / (H / Hg);
    const int cs = seq_start + c * KKT_CS;
    const int actual = (cs < seq_end) ? kkt_min(KKT_CS, seq_end - cs) : 0;
#if defined(FLASHQLA_ARCH_100)
    if (actual > 0) {
        const Element* src = K + (int64_t)cs * Hg * KKT_DK + hg * KKT_DK;
        // tc01: closed actlize pair. One AIU-tiled_copy call issued by tx==0
        // covers InstNum=2 AIU instructions (Block_K=128 split into 2x64 elem
        // slices), writes K_buf in AIU-native cube-major layout to be read by
        // the matching TSM_LD_SWZL smem-copy atom during kkt_compute_head.
        kkt_load_k_tile_tc01<Element>(K_dst, src, Hg * KKT_DK, actual, tx);
    }
#else
    if (actual > 0 && lane == 0 && warp < (KKT_DK / 32)) {
        const Element* src = K + (int64_t)cs * Hg * KKT_DK + hg * KKT_DK;
        kkt_aiu_swzl_write_k_cube<Element>(K_dst, src, Hg * KKT_DK, actual, warp);
    }
#endif
    kkt_aiu_commit();
    for (int i = tx; i < KKT_CS; i += KKT_NThreads) {
        int pos = cs + i;
        beta_dst[i] = (i < actual) ? beta[(int64_t)pos * H + h] : 0.0f;
    }
}

#if defined(FLASHQLA_ARCH_150)
template <typename Element>
__device__ __forceinline__ void kkt_issue_pair_task(
    Element* K_dst, const Element* K,
    const int* cu_seqlens, const int* chunk_indices,
    int H, int Hg, int task,
    int warp, int lane) {
    int seq_start, seq_end, local_chunk, head0;
    kkt_pair_task_coords(
        cu_seqlens, chunk_indices, H, task,
        seq_start, seq_end, local_chunk, head0);
    const int key_head = head0 / (H / Hg);
    const int chunk_start = seq_start + local_chunk * KKT_CS;
    const int actual = (chunk_start < seq_end)
        ? kkt_min(KKT_CS, seq_end - chunk_start) : 0;
    if (actual > 0 && lane == 0 && warp < (KKT_DK / 32)) {
        const Element* src = K +
            (int64_t)chunk_start * Hg * KKT_DK + key_head * KKT_DK;
        kkt_aiu_swzl_write_k_cube<Element>(
            K_dst, src, Hg * KKT_DK, actual, warp);
    }
    kkt_aiu_commit();
}
#endif

// ========================================================================
// Kernel: persistent grid-stride + AIU async double-buffer pipeline
// ========================================================================
template <typename Element, bool PerValueHead = false>
__global__ void __launch_bounds__(KKT_NThreads, 6)
FLASHQLA_KKT_KERNEL(
    Element* __restrict__ A_out,
    const Element* __restrict__ K,
    const float* __restrict__ beta,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens
) {
    const int tx   = threadIdx.x;
    const int warp = tx >> 5;
    const int lane = tx & 31;
#if defined(FLASHQLA_ARCH_100)
    const int total_tasks = num_chunks * (PerValueHead ? H : Hg);
#else
    const int total_tasks = num_chunks * H;
#endif

    extern __shared__ char smem[];
    Element* K_buf0   = reinterpret_cast<Element*>(smem + KKT_OFF_K0);
    Element* K_buf1   = reinterpret_cast<Element*>(smem + KKT_OFF_K1);
    float*  KKT_buf  = reinterpret_cast<float*>(smem + KKT_OFF_KKT);
    Element* Mh       = reinterpret_cast<Element*>(smem + KKT_OFF_MH);
    Element* Ah       = reinterpret_cast<Element*>(smem + KKT_OFF_AH);
    Element* Sbuf     = reinterpret_cast<Element*>(smem + KKT_OFF_SBUF);
    float*  beta_buf0 = reinterpret_cast<float*>(smem + KKT_OFF_B0);
    float*  beta_buf1 = reinterpret_cast<float*>(smem + KKT_OFF_B1);

    const int bx = (int)blockIdx.x;
    const int gx = (int)gridDim.x;

    // ─── Prologue: seed BOTH double-buffer slots (task0→slot0, task1→slot1)
    //     via per-warp split-issue, then DRAIN THE FIRST group so task0's K +
    //     beta are resident before the loop (the user's "preload beta + wait
    //     group 1 in the prologue" step).  task1 stays in flight and is drained
    //     at the tail of iteration 0. ───
    const bool p_have0 = bx < total_tasks;
    const bool p_have1 = bx + gx < total_tasks;
    if (p_have0)
        kkt_issue_task<Element, PerValueHead>(K_buf0, beta_buf0, K, beta,
                       cu_seqlens, chunk_indices, H, Hg,
                       bx, warp, lane, tx);
    if (p_have1)
        kkt_issue_task<Element, PerValueHead>(K_buf1, beta_buf1, K, beta,
                       cu_seqlens, chunk_indices, H, Hg,
                       bx + gx, warp, lane, tx);
    if (p_have0) {
        if (p_have1) kkt_aiu_wait_group<1>();  // 2 in flight → keep task1
        else         kkt_aiu_wait_group<0>();  // only task0 → drain fully
        __syncthreads();                        // publish task0's 4 cubes + beta
    }

    // ─── Main pipeline loop: EARLY-ISSUE / LATE-WAIT ───
    //   head (consumes K+beta) → fire task_{ti+2} prefetch into the freed slot
    //   → tri-inverse (DMA hides here) → tail wait for task_{ti+1}.
    int ti = 0;
    for (int task = bx; task < total_tasks; task += gx, ++ti) {
        const int slot = ti & 1;
        Element* K_buf    = slot ? K_buf1 : K_buf0;
        float*  beta_buf = slot ? beta_buf1 : beta_buf0;

        int seq_start, seq_end, chunk_idx, head_idx;
        kkt_task_coords<PerValueHead>(
            cu_seqlens, chunk_indices, H, Hg, task,
            seq_start, seq_end, chunk_idx, head_idx);
        const int chunk_start = seq_start + chunk_idx * KKT_CS;
        const int actual = (chunk_start < seq_end)
                         ? kkt_min(KKT_CS, seq_end - chunk_start) : 0;

        const int task_p2 = task + 2 * gx;
        const bool have_p2 = task_p2 < total_tasks;

        if (actual > 0) {
            // HEAD — K_buf is dead on both architectures. beta_buf is also
            // dead on tc02; tc01 retains it for the recurrence below.
            kkt_compute_head<Element>(
                             K_buf, beta_buf, KKT_buf, Mh, Ah,
                             actual, tx, warp, lane);

            // EARLY prefetch: task_{ti+2} reuses THIS slot (now free, since
            // (ti+2)&1 == ti&1).  Fired BEFORE the tri-inverse so the ~26KB
            // K DMA + beta fill fully overlap the serial triangular solve.
            if (have_p2)
                kkt_issue_task<Element, PerValueHead>(K_buf, beta_buf, K, beta,
                               cu_seqlens, chunk_indices,
                               H, Hg, task_p2, warp, lane, tx);

            // TAIL — long serial critical path; DMA above hides under it.
#if defined(FLASHQLA_ARCH_100)
            if constexpr (PerValueHead) {
                kkt_compute_tri_tc01<Element>(
                    A_out, KKT_buf, beta_buf, reinterpret_cast<float*>(Mh),
                    chunk_start, actual, head_idx, num_tokens, H,
                    tx, warp, lane);
            } else {
                const int heads_per_k = H / Hg;
                if (heads_per_k == 1) {
                    kkt_compute_tri_tc01<Element>(
                        A_out, KKT_buf, beta_buf, reinterpret_cast<float*>(Mh),
                        chunk_start, actual, head_idx, num_tokens, H,
                        tx, warp, lane);
                } else {
                    for (int i = tx; i < KKT_CS; i += KKT_NThreads) {
                        const int pos = chunk_start + i;
                        beta_buf1[i] = (i < actual)
                            ? beta[(int64_t)pos * H + head_idx + 1] : 0.0f;
                    }
                    __syncthreads();
                    kkt_compute_tri_pair_tc01<Element>(
                        A_out, KKT_buf, beta_buf, beta_buf1,
                        reinterpret_cast<float*>(Mh), chunk_start, actual,
                        head_idx, num_tokens, H, tx, warp, lane);
                }
                for (int value_head = 2; value_head < heads_per_k; value_head += 2) {
                    const int next_head = head_idx + value_head;
                    const bool have_pair = value_head + 1 < heads_per_k;
                    for (int i = tx; i < KKT_CS; i += KKT_NThreads) {
                        const int pos = chunk_start + i;
                        beta_buf[i] = (i < actual)
                            ? beta[(int64_t)pos * H + next_head] : 0.0f;
                        beta_buf1[i] = (i < actual && have_pair)
                            ? beta[(int64_t)pos * H + next_head + 1] : 0.0f;
                    }
                    __syncthreads();
                    if (have_pair) {
                        kkt_compute_tri_pair_tc01<Element>(
                            A_out, KKT_buf, beta_buf, beta_buf1,
                            reinterpret_cast<float*>(Mh), chunk_start, actual,
                            next_head, num_tokens, H, tx, warp, lane);
                    } else {
                        kkt_compute_tri_tc01<Element>(
                            A_out, KKT_buf, beta_buf, reinterpret_cast<float*>(Mh),
                            chunk_start, actual, next_head, num_tokens, H,
                            tx, warp, lane);
                    }
                }
            }
#else
            kkt_compute_tri<Element>(A_out, Mh, Ah, Sbuf, chunk_start, actual,
                            head_idx, num_tokens, H, tx, warp, lane);
#endif
        } else if (have_p2) {
            // Empty task (partial-chunk edge): the slot was never read, so it
            // is trivially free — still fire the prefetch to keep the pipeline
            // group-count balanced for the tail wait.
            kkt_issue_task<Element, PerValueHead>(K_buf, beta_buf, K, beta,
                           cu_seqlens, chunk_indices,
                           H, Hg, task_p2, warp, lane, tx);
        }

        // LATE wait: guarantee task_{ti+1}'s slot is resident for the next
        // iteration.  If we just fired a task_{ti+2} prefetch it is still in
        // flight (wait_group<1>); otherwise task_{ti+1} is the only group left
        // (wait_group<0>).  Every warp waits only on the cube IT issued, so one
        // __syncthreads after the wait publishes all 4 cubes + beta.
        if (task + gx < total_tasks) {
            if (have_p2) kkt_aiu_wait_group<1>();
            else         kkt_aiu_wait_group<0>();
            __syncthreads();
        }
    }
}

#if defined(FLASHQLA_ARCH_100)
template <typename Element>
__global__ void __launch_bounds__(KKT_NThreads, 3)
kkt_solve_quad_tc01_kernel(
    Element* A_out,
    const Element* K,
    const float* beta,
    const int* cu_seqlens,
    const int* chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens) {
    const int tx = threadIdx.x;
    const int warp = tx >> 5;
    const int lane = tx & 31;
    const int total_tasks = num_chunks * (H / 4);
    const int task = blockIdx.x;
    if (task >= total_tasks) return;

    int seq_start, seq_end, local_chunk, head0;
    kkt_quad_task_coords(
        cu_seqlens, chunk_indices, H, task,
        seq_start, seq_end, local_chunk, head0);
    const int chunk_start = seq_start + local_chunk * KKT_CS;
    const int actual = chunk_start < seq_end
        ? kkt_min(KKT_CS, seq_end - chunk_start) : 0;
    if (actual <= 0) return;

    extern __shared__ char smem[];
    Element* K_buf = reinterpret_cast<Element*>(smem + KKT_QUAD_OFF_K);
    float* Gram = reinterpret_cast<float*>(smem + KKT_QUAD_OFF_GRAM);
    float* A_fp32 = reinterpret_cast<float*>(
        smem + KKT_QUAD_OFF_A + warp * KKT_QUAD_A_BYTES);
    float* beta_buf = reinterpret_cast<float*>(
        smem + KKT_QUAD_OFF_BETA + warp * KKT_CS * 4);

    const int key_head = head0 / (H / Hg);
    const Element* K_src = K +
        (int64_t)chunk_start * Hg * KKT_DK + key_head * KKT_DK;
    kkt_load_k_tile_tc01<Element>(
        K_buf, K_src, Hg * KKT_DK, actual, tx);
    kkt_aiu_commit();
    kkt_aiu_wait_group<0>();
    __syncthreads();

    // kkt_compute_head materializes Gram on tc01; its M/A arguments are not
    // used by that architecture-specific branch.
    kkt_compute_head<Element>(
        K_buf, beta_buf, Gram, reinterpret_cast<Element*>(A_fp32),
        reinterpret_cast<Element*>(A_fp32), actual, tx, warp, lane);

    const int head = head0 + warp;
    beta_buf[lane] = lane < actual
        ? beta[(int64_t)(chunk_start + lane) * H + head] : 0.0f;
    __syncwarp();
    kkt_compute_tri_quad_tc01<Element>(
        A_out, Gram, beta_buf, A_fp32, chunk_start, actual, head,
        num_tokens, H, lane);
}

template <typename Element>
__global__ void __launch_bounds__(256, 3)
kkt_solve_oct_tc01_kernel(
    Element* A_out,
    const Element* K,
    const float* beta,
    const int* cu_seqlens,
    const int* chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens) {
    const int tx = threadIdx.x;
    const int warp = tx >> 5;
    const int lane = tx & 31;
    const int total_tasks = num_chunks * Hg;
    const int task = blockIdx.x;
    if (task >= total_tasks) return;

    const int global_chunk = task / Hg;
    const int key_head = task % Hg;
    const int seq_idx = chunk_indices ? chunk_indices[global_chunk * 2] : 0;
    const int local_chunk = chunk_indices
        ? chunk_indices[global_chunk * 2 + 1] : global_chunk;
    const int seq_start = cu_seqlens[seq_idx];
    const int seq_end = cu_seqlens[seq_idx + 1];
    const int chunk_start = seq_start + local_chunk * KKT_CS;
    const int actual = chunk_start < seq_end
        ? kkt_min(KKT_CS, seq_end - chunk_start) : 0;
    if (actual <= 0) return;

    extern __shared__ char smem[];
    Element* K_buf = reinterpret_cast<Element*>(smem + KKT_OCT_OFF_K);
    float* Gram = reinterpret_cast<float*>(smem + KKT_OCT_OFF_GRAM);
    float* A_fp32 = reinterpret_cast<float*>(
        smem + KKT_OCT_OFF_A + warp * KKT_QUAD_A_BYTES);
    float* beta_buf = reinterpret_cast<float*>(
        smem + KKT_OCT_OFF_BETA + warp * KKT_CS * 4);

    const Element* K_src = K +
        (int64_t)chunk_start * Hg * KKT_DK + key_head * KKT_DK;
    kkt_load_k_tile_tc01<Element>(
        K_buf, K_src, Hg * KKT_DK, actual, tx);
    kkt_aiu_commit();
    kkt_aiu_wait_group<0>();
    __syncthreads();
    kkt_compute_head<Element>(
        K_buf, beta_buf, Gram, reinterpret_cast<Element*>(A_fp32),
        reinterpret_cast<Element*>(A_fp32), actual, tx, warp, lane);

    const int head = key_head * 8 + warp;
    beta_buf[lane] = lane < actual
        ? beta[(int64_t)(chunk_start + lane) * H + head] : 0.0f;
    __syncwarp();
    kkt_compute_tri_quad_tc01<Element>(
        A_out, Gram, beta_buf, A_fp32, chunk_start, actual, head,
        num_tokens, H, lane);
}
#else
template <typename Element>
__global__ void __launch_bounds__(KKT_NThreads, 3)
kkt_solve_quad_tc01_kernel(
    Element*, const Element*, const float*, const int*, const int*,
    int, int, int, int) {}
template <typename Element>
__global__ void __launch_bounds__(256, 3)
kkt_solve_oct_tc01_kernel(
    Element*, const Element*, const float*, const int*, const int*,
    int, int, int, int) {}
#endif

#if defined(FLASHQLA_ARCH_150)
template <typename Element>
__global__ void __launch_bounds__(KKT_NThreads, 10)
kkt_solve_pair_cute_kernel(
    Element* __restrict__ A_out,
    const Element* __restrict__ K,
    const float* __restrict__ beta,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens) {
    const int tx = threadIdx.x;
    const int warp = tx >> 5;
    const int lane = tx & 31;
    const int total_tasks = num_chunks * (H / 2);

    extern __shared__ char smem[];
    Element* K_buf0 = reinterpret_cast<Element*>(smem + KKT_PAIR_OFF_K0);
    Element* K_buf1 = reinterpret_cast<Element*>(smem + KKT_PAIR_OFF_K1);
    float* KKT_buf = reinterpret_cast<float*>(smem + KKT_PAIR_OFF_KKT);
    Element* M0 = reinterpret_cast<Element*>(smem + KKT_PAIR_OFF_M0);
    Element* A0 = reinterpret_cast<Element*>(smem + KKT_PAIR_OFF_A0);
    Element* M1 = reinterpret_cast<Element*>(smem + KKT_PAIR_OFF_M1);
    Element* A1 = reinterpret_cast<Element*>(smem + KKT_PAIR_OFF_A1);

    const int block = static_cast<int>(blockIdx.x);
    const int grid = static_cast<int>(gridDim.x);
    const bool have0 = block < total_tasks;
    const bool have1 = block + grid < total_tasks;
    if (have0) {
        kkt_issue_pair_task<Element>(
            K_buf0, K, cu_seqlens, chunk_indices,
            H, Hg, block, warp, lane);
    }
    if (have1) {
        kkt_issue_pair_task<Element>(
            K_buf1, K, cu_seqlens, chunk_indices,
            H, Hg, block + grid, warp, lane);
    }
    if (have0) {
        if (have1) kkt_aiu_wait_group<1>();
        else       kkt_aiu_wait_group<0>();
        __syncthreads();
    }

    int iteration = 0;
    for (int task = block; task < total_tasks; task += grid, ++iteration) {
        Element* K_buf = (iteration & 1) ? K_buf1 : K_buf0;

        int seq_start, seq_end, local_chunk, head0;
        kkt_pair_task_coords(
            cu_seqlens, chunk_indices, H, task,
            seq_start, seq_end, local_chunk, head0);
        const int chunk_start = seq_start + local_chunk * KKT_CS;
        const int actual = (chunk_start < seq_end)
            ? kkt_min(KKT_CS, seq_end - chunk_start) : 0;
        const int task_p2 = task + 2 * grid;
        const bool have_p2 = task_p2 < total_tasks;

        if (actual > 0) {
            kkt_compute_pair_head<Element>(
                K_buf, beta, KKT_buf, M0, A0, M1, A1,
                chunk_start, head0, actual, H, tx);

            if (have_p2) {
                kkt_issue_pair_task<Element>(
                    K_buf, K, cu_seqlens, chunk_indices,
                    H, Hg, task_p2, warp, lane);
            }

            // Both groups have finished every Gram read and published M0/M1.
            // The next-task K prefetch above touches only K_buf, so it may stay
            // overlapped before this lifetime fence releases A0/A1 to reuse Gram.
            __syncthreads();

            kkt_compute_pair_tri<Element>(
                A_out, M0, A0, M1, A1, chunk_start, actual,
                head0, num_tokens, H, tx);
        } else if (have_p2) {
            kkt_issue_pair_task<Element>(
                K_buf, K, cu_seqlens, chunk_indices,
                H, Hg, task_p2, warp, lane);
        }

        if (task + grid < total_tasks) {
            if (have_p2) kkt_aiu_wait_group<1>();
            else         kkt_aiu_wait_group<0>();
            __syncthreads();
        }
    }
}
#else
// The PPU runtime registers one host stub against every image in a multi-arch
// object.  tc01 never dispatches this kernel, but its image must still carry
// the entry so the tc02 implementation can be resolved on PPU1.5.
template <typename Element>
__global__ void __launch_bounds__(KKT_NThreads, 10)
kkt_solve_pair_cute_kernel(
    Element* __restrict__, const Element* __restrict__,
    const float* __restrict__, const int* __restrict__,
    const int* __restrict__,
    int, int, int, int) {}
#endif

}  // namespace FLASHQLA_KKT_NAMESPACE

// ============================================================
// Host launch
// ============================================================
namespace {

constexpr int KKT_MAX_CACHED_DEVICES = 64;

template <typename Configure>
void kkt_configure_dynamic_smem_once(
    std::once_flag (&configured)[KKT_MAX_CACHED_DEVICES],
    int device_index, Configure configure) {
    auto invoke = [&]() {
        if (configure() != hggcSuccess) hggcGetLastError();
    };
    if (device_index < 0 || device_index >= KKT_MAX_CACHED_DEVICES) {
        invoke();
        return;
    }
    std::call_once(configured[device_index], invoke);
}

template <typename Element, bool PerValueHead>
void kkt_configure_main_dynamic_smem(int device_index, int smem_bytes) {
    // One cache per Element x dispatch specialization x logical device.
    static std::once_flag configured[KKT_MAX_CACHED_DEVICES];
    kkt_configure_dynamic_smem_once(
        configured, device_index, [smem_bytes]() {
            return hggcFuncSetAttribute(
                FLASHQLA_KKT_NAMESPACE::FLASHQLA_KKT_KERNEL<
                    Element, PerValueHead>,
                hggcFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        });
}

#if FLASHQLA_KKT_HOST_ARCH_100
template <typename Element>
void kkt_configure_quad_dynamic_smem(int device_index, int smem_bytes) {
    static std::once_flag configured[KKT_MAX_CACHED_DEVICES];
    kkt_configure_dynamic_smem_once(
        configured, device_index, [smem_bytes]() {
            return hggcFuncSetAttribute(
                FLASHQLA_KKT_NAMESPACE::kkt_solve_quad_tc01_kernel<Element>,
                hggcFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        });
}

template <typename Element>
void kkt_configure_oct_dynamic_smem(int device_index, int smem_bytes) {
    static std::once_flag configured[KKT_MAX_CACHED_DEVICES];
    kkt_configure_dynamic_smem_once(
        configured, device_index, [smem_bytes]() {
            return hggcFuncSetAttribute(
                FLASHQLA_KKT_NAMESPACE::kkt_solve_oct_tc01_kernel<Element>,
                hggcFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        });
}
#else
template <typename Element>
void kkt_configure_pair_dynamic_smem(int device_index, int smem_bytes) {
    static std::once_flag configured[KKT_MAX_CACHED_DEVICES];
    kkt_configure_dynamic_smem_once(
        configured, device_index, [smem_bytes]() {
            return hggcFuncSetAttribute(
                FLASHQLA_KKT_NAMESPACE::kkt_solve_pair_cute_kernel<Element>,
                hggcFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        });
}
#endif

}  // namespace

void FLASHQLA_KKT_LAUNCHER(
    const void* K,
    const float* beta,
    void* A,
    const int* cu_seqlens,
    const int* chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens, int dtype_tag,
    int device_index, int num_cu,
    hggcStream_t stream)
{
    // The host pass has no __HGGC_ARCH__.  Each wrapper supplies its host
    // policy explicitly while device code remains selected per embedded PPU
    // image.
    constexpr int smem_bytes = FLASHQLA_KKT_HOST_ARCH_100
        ? FLASHQLA_KKT_NAMESPACE::KKT_SMEM_TC01
        : FLASHQLA_KKT_NAMESPACE::KKT_SMEM_TC02;
#if !FLASHQLA_KKT_HOST_ARCH_100
    constexpr int pair_smem_bytes =
        FLASHQLA_KKT_NAMESPACE::KKT_PAIR_SMEM_BYTES;
#else
    constexpr int quad_smem_bytes =
        FLASHQLA_KKT_NAMESPACE::KKT_QUAD_SMEM_BYTES;
    constexpr int oct_smem_bytes =
        FLASHQLA_KKT_NAMESPACE::KKT_OCT_SMEM_BYTES;
#endif

    const int nsm = num_cu > 0 ? num_cu : 39;
    const int heads_per_k = H / Hg;
    const int64_t shared_tasks =
        static_cast<int64_t>(num_chunks) * Hg;
    const int64_t per_head_tasks =
        static_cast<int64_t>(num_chunks) * H;
    const int64_t shared_waves = (shared_tasks + nsm - 1) / nsm;
    const int64_t per_head_waves = (per_head_tasks + nsm - 1) / nsm;
    const bool single_sequence = chunk_indices == nullptr;

    // Sharing one Gram matrix reduces K reads, while assigning one value head
    // per CTA exposes the triangular solve across more CUs.  Keep the existing
    // underfilled-grid rule for ratios >= 3.  Ratio-2 duplicates the Gram work,
    // so use it only when the original wave-saving condition holds and the
    // chunk count lies in the same-card r201 crossover windows.  For ratio >=
    // 8, measured tc01 cases retain a benefit a little beyond one shared-task
    // wave, but only for a bounded chunk count; longer sequences become K-read
    // dominated again.
    const bool underfilled_shared_grid =
        heads_per_k >= 3 && shared_tasks < nsm;
    const bool ratio2_chunk_gate =
        (H == 8 && num_chunks <= 24) ||
        (H == 16 &&
         (num_chunks <= 12 ||
          (num_chunks >= 27 && num_chunks <= 36 &&
           (num_chunks % 4 == 0 || num_chunks % 4 == 3)))) ||
        (H == 32 && num_chunks <= 6);
    const bool two_head_wave_saving =
        single_sequence && heads_per_k == 2 && ratio2_chunk_gate &&
        per_head_waves < shared_waves * heads_per_k;
    const bool high_ratio_extension = single_sequence && heads_per_k >= 8 &&
        ((H <= 16 && num_chunks <= 128) || shared_tasks < 2 * nsm);
    const int64_t quad_tasks =
        static_cast<int64_t>(num_chunks) * (H / 4);
    const int64_t oct_tasks =
        static_cast<int64_t>(num_chunks) * Hg;
    // Four independent warps solve four adjacent value heads after sharing one
    // K@K^T tile.  On tc01 this removes the block-wide barrier from every row
    // of the triangular recurrence.  The crossover depends on both GQA ratio
    // and grid size: ratio-8 already saves repeated Gram work after one full CU
    // wave, while ratio-4 keeps the same Gram count as the shared kernel and
    // needs four waves once H >= 32 to amortize the larger four-output tile.
    // Small ratio-4 head groups have a shorter publication path and cross over
    // at one wave.  These bounds cover the supported (H, Hg) dispatch matrix
    // and intentionally leave partial waves on the established kernels.
    const int64_t quad_min_tasks =
        heads_per_k >= 8 || H <= 16 ? nsm : 4LL * nsm;
    // Packed ratio-4 has one measured occupancy hole: more than 2.5 CU waves
    // but no more than three waves plus one H/4 task group.  Keep those ten
    // inventory cases on the shared-Gram kernel; packed quad is faster outside
    // that interval in same-process BF/CF r201 measurements.
    const bool packed_quad_hole =
        !single_sequence && heads_per_k == 4 &&
        2LL * quad_tasks > 5LL * nsm &&
        quad_tasks <= 3LL * nsm + H / 4;
    const bool use_oct = FLASHQLA_KKT_HOST_ARCH_100 &&
        heads_per_k == 8 && oct_tasks >= (nsm + 1) / 2 &&
        (single_sequence || H == 16);
    const bool use_quad = !use_oct && FLASHQLA_KKT_HOST_ARCH_100 &&
        (single_sequence || !packed_quad_hole) &&
        heads_per_k % 4 == 0 && quad_tasks >= quad_min_tasks;
    const bool use_per_value_head = !use_oct && !use_quad &&
        FLASHQLA_KKT_HOST_ARCH_100 &&
        (underfilled_shared_grid || two_head_wave_saving ||
         high_ratio_extension);
    const int total_tasks = num_chunks *
        (use_oct ? Hg
                  : (use_quad ? H / 4
                  : (use_per_value_head ? H
                     : (FLASHQLA_KKT_HOST_ARCH_100 ? Hg : H))));
    int persist = nsm * 9;
#if FLASHQLA_KKT_HOST_ARCH_100
    // tc01's fp32 triangular recurrence makes the persistent-loop publication
    // barrier more expensive than relaunching the next resident wave.  Give
    // each CTA exactly one task: the loop-exit path then omits the task-to-task
    // __syncthreads at line 718 entirely.
    int grid_x = total_tasks;
#else
    int grid_x = total_tasks < persist ? total_tasks : persist;
    // Adjacent value heads share one K head exactly when the value-to-key head
    // ratio is even.  Keep odd ratios on the one-head kernel so a pair never
    // crosses a K-head boundary.
    const bool use_pair = H % (2 * Hg) == 0;
    const int pair_tasks = num_chunks * (H / 2);
    const int pair_persist = nsm * 10;
    int pair_grid_x = pair_tasks < pair_persist ? pair_tasks : pair_persist;
    if (pair_grid_x < 1) pair_grid_x = 1;
#endif
    if (grid_x < 1) grid_x = 1;

    dim3 grid(grid_x);
#if !FLASHQLA_KKT_HOST_ARCH_100
    dim3 pair_grid(pair_grid_x);
#endif
    dim3 block(FLASHQLA_KKT_NAMESPACE::KKT_NThreads);

#if FLASHQLA_KKT_HOST_ARCH_100
#define KKT_LAUNCH(ElementType, TAG) \
    do { \
      if (use_oct) { \
        kkt_configure_oct_dynamic_smem<ElementType>( \
            device_index, oct_smem_bytes); \
        dim3 oct_block(256); \
        FLASHQLA_KKT_NAMESPACE::kkt_solve_oct_tc01_kernel<ElementType> \
            <<<grid, oct_block, oct_smem_bytes, stream>>>( \
                reinterpret_cast<ElementType*>(A), \
                reinterpret_cast<const ElementType*>(K), beta, cu_seqlens, \
                chunk_indices, H, Hg, num_chunks, num_tokens); \
      } else if (use_quad) { \
        kkt_configure_quad_dynamic_smem<ElementType>( \
            device_index, quad_smem_bytes); \
        FLASHQLA_KKT_NAMESPACE::kkt_solve_quad_tc01_kernel<ElementType> \
            <<<grid, block, quad_smem_bytes, stream>>>( \
                reinterpret_cast<ElementType*>(A), \
                reinterpret_cast<const ElementType*>(K), beta, cu_seqlens, \
                chunk_indices, H, Hg, num_chunks, num_tokens); \
      } else if (use_per_value_head) { \
        kkt_configure_main_dynamic_smem<ElementType, true>( \
            device_index, smem_bytes); \
        FLASHQLA_KKT_NAMESPACE::FLASHQLA_KKT_KERNEL<ElementType, true> \
            <<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<ElementType*>(A), \
                reinterpret_cast<const ElementType*>(K), beta, cu_seqlens, \
                chunk_indices, H, Hg, num_chunks, num_tokens); \
      } else { \
        kkt_configure_main_dynamic_smem<ElementType, false>( \
            device_index, smem_bytes); \
        FLASHQLA_KKT_NAMESPACE::FLASHQLA_KKT_KERNEL<ElementType, false> \
            <<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<ElementType*>(A), \
                reinterpret_cast<const ElementType*>(K), beta, cu_seqlens, \
                chunk_indices, H, Hg, num_chunks, num_tokens); \
      } \
    } while (0)
#else
#define KKT_LAUNCH(ElementType, TAG) \
    do { \
      if (use_pair) { \
        kkt_configure_pair_dynamic_smem<ElementType>( \
            device_index, pair_smem_bytes); \
        FLASHQLA_KKT_NAMESPACE::kkt_solve_pair_cute_kernel<ElementType> \
            <<<pair_grid, block, pair_smem_bytes, stream>>>( \
                reinterpret_cast<ElementType*>(A), \
                reinterpret_cast<const ElementType*>(K), beta, cu_seqlens, \
                chunk_indices, H, Hg, num_chunks, num_tokens); \
      } else { \
        kkt_configure_main_dynamic_smem<ElementType, false>( \
            device_index, smem_bytes); \
        FLASHQLA_KKT_NAMESPACE::FLASHQLA_KKT_KERNEL<ElementType> \
            <<<grid, block, smem_bytes, stream>>>( \
                reinterpret_cast<ElementType*>(A), \
                reinterpret_cast<const ElementType*>(K), beta, cu_seqlens, \
                chunk_indices, H, Hg, num_chunks, num_tokens); \
      } \
    } while (0)
#endif

    if (dtype_tag == 0) { KKT_LAUNCH(__half, 0); }
    else { KKT_LAUNCH(__ppu_bfloat16, 1); }

#undef KKT_LAUNCH
}
