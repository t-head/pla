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
 * FlashFlashKDA PPU1.5 — Kernel 2 (recurrence, FAITHFUL non-WS register-resident port).
 *
 * This is a design-faithful PPU reproduction of Flash-Flash-KDA's
 * csrc/smxx/fwd_kernel2_nows.cuh (`_flash_kda_fwd_recurrence_nows`):
 *
 *   - grid = (N, H): ONE CTA owns a whole (sequence, head) and ALL DV=128 value
 *     columns (vs the baseline PPU kernel's grid=(N,H,DV/DVT) column split).
 *   - Runtime-selected 128 threads (4 warps x 32 value columns) or 256
 *     threads (8 warps x 16 columns).  The 256-thread template is used for
 *     Batch=1 grids that fit in one two-block-per-CU resident wave.
 *   - The recurrent state slice St_w[dk=128, dv=32] is kept RESIDENT IN BF16 C
 *     FRAGMENTS for the entire chunk loop: rstate[8 dk-row-blocks][2 dv-col-blocks],
 *     8 BF16 each. No per-chunk smem round-trip of the state (that is the whole
 *     point of the faithful port — trade smem traffic for register pressure).
 *   - Per chunk the same 6-phase math as the baseline runs, but the SM80
 *     `movm_transpose_c_to_b_16x16` C->B transpose is mapped to the probe-verified
 *     PPU primitive `bf16_cfrag_to_bfrag_movmatrix_u32`, and the SM80
 *     `SM80_16x8x16_F32BF16BF16F32_TN` MMA maps to PPU_16x16x16_F32BF16BF16F32_TN_FIXED.
 *
 * Because the PPU MMA C-fragment lane layout differs from SM80's, a byte-for-byte
 * port is impossible; the element<->(row,col) mapping is recovered generically at
 * runtime via `thr_mma.partition_C(make_identity_tensor(...))` so beta/g_total
 * indexing stays layout-agnostic.
 *
 * Correctness-first: single-buffered staging (AIU 2D loads issued by the leader).
 * Selected via env FLASHKDA_K2_NOWS in launch_flashkda_recurrence (kernel2_recurrence.cu).
 *
 * ---------------------------------------------------------------------------
 * Context parallelism (CP), two extra template modes
 * ---------------------------------------------------------------------------
 * The chunk recurrence is affine in the carried state:
 *
 *     St' = A_t . St + P_t . v_t,   A_t = diag(g_t) - P_t . kd_t,
 *                                  P_t = kr_t^T . INV_t . diag(beta_t)
 *
 * so a tile chain can be cut into segments whose affine transfer functions are
 * composed afterwards.  Two template flags turn this kernel into all three CP
 * passes without duplicating the register-resident mainloop:
 *
 *   UseSegTable - blockIdx.x indexes a segment descriptor (tok_begin, seg_len,
 *     ws_tile_base) instead of a sequence, and init/final state are indexed by
 *     segment.  This is the segment-parallel pass: identical math, S x more
 *     concurrent chains.
 *   PreScan - grid.z is doubled.  The lower half carries the bias chain B
 *     (zero-initialised, real v) and the upper half the transition chain M
 *     (identity-initialised, v := 0), which is the same recurrence because
 *     A_t acts on the LEFT and therefore treats every column of the carried
 *     matrix independently -- exactly the property the DV split already
 *     exploits.  Output stores are skipped; the two chains land in
 *     final_state[chain][seg].
 *
 * A serial merge (kernel3_cp.cu) then composes carries[i+1] = M_i.carries[i]
 * + B_i, and the segment-parallel pass consumes those carries as init_state.
 */
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <cutlass/bfloat16.h>
#include <type_traits>
#include "prefill/flashkdapro/common/launch_api.h"
#include "prefill/flashkdapro/common/runtime_arch.h"
#include "flashkda_ppu_util.cuh"
#include "prefill/flashkdapro/tc01/kernel2_output_tc01.cuh"

// -----------------------------------------------------------------------------
// cfrag->bfrag transpose backend for the nows chunk loop: movmatrix (smem
// scratch + ldmatrix.trans + 2x __syncwarp), the shipped path.
// -----------------------------------------------------------------------------
#define NOWS_CFRAG_TO_BFRAG(c_regs, b_regs, lane, scratch) do { \
    if constexpr (DKSplitWarps == 1) \
        fk::bf16_cfrag_to_bfrag_movmatrix_batched_u32( \
            (c_regs), (b_regs), (lane), (scratch)); \
    else \
        fk::bf16_cfrag_to_bfrag_movmatrix_u32( \
            (c_regs), (b_regs), (lane), (scratch)); \
} while (0)
#define NOWS_ACC_TO_BFRAG(u_acc, b_regs, lane, scratch) do { \
    if constexpr (DKSplitWarps == 1) \
        fk::bf16_acc_to_bfrag_movmatrix_batched_u32( \
            (u_acc), (b_regs), (lane), (scratch)); \
    else \
        fk::bf16_acc_to_bfrag_movmatrix_u32( \
            (u_acc), (b_regs), (lane), (scratch)); \
} while (0)

// The default output write path remains the shipped fast path: each lane
// converts its 8 owned fp32 accumulator elements to bf16 and stores directly to
// global, packing the two elements of each tc02 C-fragment pair into one aligned
// b32 store.  A tc01-only opt-in below reuses DK1's existing warp scratch for an
// A/B of row-vectorized output stores.

// P5/P6 prefetch (always on, gated to the dvsplit weak region): exact single-warp
// self-pipelined 2-stage AIU prefetch.  The chunk inputs (workspace slot, V, beta)
// depend only on the token index, entirely independent of the loop-carried
// recurrent state, so issuing chunk t+1's async loads before computing chunk t is
// bit-exact -- no numerics change, only latency hiding.  A single compute warp runs
// serially (free WAR safety between the two smem buffers) and the AIU DMA overlaps
// its copies with the register-resident GEMM/state-update math.  It is gated to the
// low-occupancy dvsplit paths (CtaDV < DV, the H12/S8192-style weak region P5
// measured -4.9%/-7.0% on) via K2NSmem::Prefetch; the full-DV main path (CtaDV == DV)
// keeps the single-buffer layout so its occupancy never regresses.  When active,
// K2NSmem double-buffers {ws,v,beta} (out/scr stay single) and the mainloop issues
// t+1 loads + ppu.cp.async.wait_group<1> instead of the single-buffer wait_group<0>.

namespace flashkda_cute_nows {

namespace fk = flashkda;
using cute::_0;
using cute::_1;
using cute::_16;
using cute::Int;
using cute::make_coord;
using cute::make_identity_tensor;
using cute::make_layout;
using cute::make_shape;
using cute::make_smem_ptr;
using cute::make_stride;
using cute::make_tensor;
using cute::local_tile;
using cute::recast;
using BF16 = cutlass::bfloat16_t;
using FP16 = __half;

__device__ __forceinline__ float k2n_state_to_f32(BF16 x) {
    return fk::bf16_to_f32(x);
}

__device__ __forceinline__ float k2n_state_to_f32(FP16 x) {
    return __half2float(x);
}

template <typename StateT>
__device__ __forceinline__ StateT k2n_state_from_f32(float x);

template <>
__device__ __forceinline__ BF16 k2n_state_from_f32<BF16>(float x) {
    return BF16(x);
}

template <>
__device__ __forceinline__ FP16 k2n_state_from_f32<FP16>(float x) {
    return __float2half(x);
}

__device__ __forceinline__ uint16_t k2n_state_as_bf16_bits(BF16 x) {
    return x.storage;
}

__device__ __forceinline__ uint16_t k2n_state_as_bf16_bits(FP16 x) {
    return BF16(__half2float(x)).storage;
}

__device__ __forceinline__ uint16_t k2n_state_bits(BF16 x) {
    return x.storage;
}

__device__ __forceinline__ uint16_t k2n_state_bits(FP16 x) {
    return __half_as_ushort(x);
}

template <bool Fp16State>
__device__ __forceinline__ auto k2n_make_phase1_acc(int lane) {
    if constexpr (Fp16State)
        return fk::make_acc16_f16_1warp(lane);
    else
        return fk::make_acc16_bf16_1warp(lane);
}

static constexpr int CH = 16;
static constexpr int DK = 128;
static constexpr int DV = 128;
static constexpr int WARP = 32;
static constexpr int DKB = DK / 16;          // 8 dk row-blocks

template <bool HalfInterleaved>
__device__ __forceinline__ float* k2n_reduce_vec4(
    float* base, int slot, int half, int lane) {
    if constexpr (HalfInterleaved)
        return base + ((slot * 2 + half) * WARP + lane) * 4;
    else
        return base + (slot * WARP + lane) * 8 + half * 4;
}

template <bool HalfInterleaved>
__device__ __forceinline__ const float* k2n_reduce_vec4(
    const float* base, int slot, int half, int lane) {
    if constexpr (HalfInterleaved)
        return base + ((slot * 2 + half) * WARP + lane) * 4;
    else
        return base + (slot * WARP + lane) * 8 + half * 4;
}

template <int NThreads, int CtaDV = DV, int DKSplitWarps = 1,
          bool DedicatedLeader = false>
struct K2NConfig {
    static_assert(NThreads == 32 || NThreads == 64
                  || NThreads == 128 || NThreads == 160 || NThreads == 256,
                  "nows supports 32, 64, 128, 160, or 256 compute threads");
    static_assert(CtaDV == 16 || CtaDV == 32 || CtaDV == DV,
                  "nows CTA value tile must be 16, 32, or the full DV");
    static constexpr int NWARPS = NThreads / WARP;
    static_assert(DKSplitWarps == 1 || DKSplitWarps == 2
                  || DKSplitWarps == 4 || DKSplitWarps == 8,
                  "DK split supports one, two, four, or eight warps per value tile");
    static constexpr int WORKER_WARPS = NWARPS - (DedicatedLeader ? 1 : 0);
    static_assert(WORKER_WARPS % DKSplitWarps == 0,
                  "DK worker warps must divide into whole DK-split groups");
    static_assert(!DedicatedLeader || DKSplitWarps > 1,
                  "a dedicated leader is useful only for a DK-split CTA");
    static constexpr int DVWARPS = WORKER_WARPS / DKSplitWarps;
    static constexpr int DVT = CtaDV / DVWARPS;
    static constexpr int COLB = DVT / 16;
    static_assert(CtaDV % DVWARPS == 0 && DVT % 16 == 0,
                  "each DK-split warp group must own whole 16-column MMA blocks");
    static_assert(COLB >= 1 && COLB <= 2,
                  "the register mainloop supports one or two blocks per warp");
};

// ---- AIU async load helpers (same intrinsics as the baseline kernel) ----
__forceinline__ __device__ void k2n_aiu_load_2d_b16(
    void* smem_ptr, const void* gmem_ptr, int dim_c, int dim_w, int64_t stride_w_bytes) {
#if defined(FLASHKDA_ARCH_150)
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
#elif defined(FLASHKDA_ARCH_100)
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
__forceinline__ __device__ void k2n_aiu_commit() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N>
__forceinline__ __device__ void k2n_aiu_wait_group() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

static constexpr int A128(int x) { return (x + 127) & ~127; }

// Single-buffer shared storage. All 4 warps share the merged ws slot + v/out/beta.
// When Prefetch is active (dvsplit weak region), {ws,v,beta} become a two-buffer
// prefetch group (buffer b at base + b*PF_STRIDE) with out/scr single-buffered
// after them; otherwise the layout is byte-identical to the shipped baseline.
template <int NThreads, int CtaDV = DV, int DKSplitWarps = 1,
          bool DedicatedLeader = false>
struct K2NSmem {
    using C = K2NConfig<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    // Gate: only the dvsplit paths (CtaDV < DV) double-buffer; the full-DV main
    // path stays single-buffered so its occupancy matches the shipped baseline.
    // The original one-warp dvsplit needs two stages to hide AIU latency.  A
    // DK-parallel CTA already has 2/4/8 compute warps and an fp32 reduction
    // scratch; retaining the second 14 KiB workspace stage can cut residency
    // without adding useful overlap, so those experimental templates use the
    // compact single-buffer layout.
    static constexpr bool Prefetch = (CtaDV < DV);

    static constexpr int O_ws    = 0;                              // merged slot
    static constexpr int O_v     = O_ws  + A128(fk::WS_SLOT_BYTES);// 14080

    // Single-buffer (shipped) ordering: {ws, v, out, beta, scr}.
    static constexpr int _sb_out  = O_v     + A128(CH * CtaDV * 2);
    static constexpr int _sb_beta = _sb_out + A128(CH * CtaDV * 2);// f32[CH]
    static constexpr int _sb_scr  = _sb_beta + A128(CH * 4);       // movm scratch

    // Prefetch (double-buffer) ordering: {ws, v, beta} group x2, then out, scr.
    static constexpr int PF_STAGES = Prefetch ? 2 : 1;
    static constexpr int _pf_beta   = O_v + A128(CH * CtaDV * 2);
    static constexpr int _pf_stride = _pf_beta + A128(CH * 4);     // {ws,v,beta}
    static constexpr int _pf_out    = PF_STAGES * _pf_stride;
    static constexpr int _pf_scr    = _pf_out + A128(CH * CtaDV * 2);

    static constexpr int O_beta    = Prefetch ? _pf_beta   : _sb_beta;
    static constexpr int O_out     = Prefetch ? _pf_out    : _sb_out;
    static constexpr int O_scr     = Prefetch ? _pf_scr    : _sb_scr;
    static constexpr int PF_STRIDE = Prefetch ? _pf_stride : 0;
    static constexpr int SCR_WORDS =
        (DKSplitWarps == 1 ? C::NWARPS * 4 : C::NWARPS) * WARP;
    static constexpr int O_dkreduce =
        O_scr + A128(SCR_WORDS * int(sizeof(uint32_t)));
    // Per DK-split warp: two fp32 C fragments (vsub and o1), indexed in native
    // lane/slot order.  The same storage is recycled for the U B-fragment
    // broadcast after the leader warp has consumed all partials.
    static constexpr int DKREDUCE_BYTES = DKSplitWarps > 1
        ? A128(C::NWARPS * 2 * WARP * 8 * int(sizeof(float))) : 0;
    static constexpr int SMEM_BYTES = O_dkreduce + DKREDUCE_BYTES;

    // field byte offsets within one merged ws slot block
    static constexpr int WOFF_KD  = fk::WS_OFF_KD_U  * 2;   // 0
    static constexpr int WOFF_QD  = fk::WS_OFF_QD_U  * 2;   // 4096
    static constexpr int WOFF_KR  = fk::WS_OFF_KR_U  * 2;   // 8192 (swzl buffer)
    static constexpr int WOFF_INV = fk::WS_OFF_INV_U * 2;   // 12288
    static constexpr int WOFF_MQK = fk::WS_OFF_MQK_U * 2;   // 12800
    static constexpr int WOFF_GT  = fk::WS_OFF_GT_U  * 2;   // 13312 (f32[DK])
    static constexpr int WOFF_BETA = fk::WS_OFF_BETA_U * 2; // sigmoid(beta) f32[CH]
};

// Load one chunk of {ws-slot, V, beta} into shared memory via the AIU.  DeferWait
// lets the caller own cp.async group draining (P5 prefetch issues t+1 then a
// single wait_group<1>); otherwise this does the blocking single-buffer drain.
template <int NThreads, int CtaDV = DV, bool DeferWait = false>
__device__ __forceinline__ void k2n_load_chunk(
    int actual, int ws_idx, int64_t tok0, int H, int T_total, int head,
    int dv_base, int tid,
    const BF16* __restrict__ v, const BF16* __restrict__ beta,
    const BF16* __restrict__ ws,
    BF16* sWs, BF16* sV, float* sBeta, bool skip_v = false) {
    // One contiguous slot (kd|qd|kr_swzl|inv|mqk|gt) = fk::WS_SLOT_UNITS bf16.
    const BF16* g_slot = ws + int64_t(ws_idx) * fk::WS_SLOT_UNITS;
    if (tid == 0) {
        // Single AIU linear bulk load of the whole 14080B slot (dim_c=128, dim_w=55).
        // kr is pre-swizzled by Kernel1 -> byte image matches the old swzl write.
        // The CP pre_scan never reads qd/mqk, but splitting this into two loads to
        // skip those 4096B was measured to change nothing (H=6/12, T=8K/32K):
        // pre_scan is latency/compute bound, not ws-bandwidth bound.
        k2n_aiu_load_2d_b16(sWs, g_slot, DK, fk::WS_SLOT_UNITS / DK,
                            int64_t(DK) * sizeof(BF16));
        if (actual == CH && !skip_v) {
            const BF16* v_src =
                v + tok0 * int64_t(H) * DV + int64_t(head) * DV + dv_base;
            k2n_aiu_load_2d_b16(
                sV, v_src, CtaDV, CH, int64_t(H) * DV * sizeof(BF16));
        }
        k2n_aiu_commit();
    }
    if (actual != CH && !skip_v) {
        for (int idx = tid; idx < CH * CtaDV; idx += NThreads) {
            int r = idx / CtaDV, c = idx % CtaDV;
            sV[idx] = (r < actual)
                ? v[(tok0 + r) * int64_t(H) * DV
                    + int64_t(head) * DV + dv_base + c] : BF16(0);
        }
    }
    (void)beta;
    (void)sBeta;
    // DeferWait: the caller owns the cp.async group draining (P5 prefetch issues
    // t+1 then a single wait_group<1>).  Otherwise this is the shipping
    // single-buffer blocking drain.
    if constexpr (!DeferWait) {
        if (tid == 0) k2n_aiu_wait_group<0>();
    }
}

// Ordinary register-resident nows targets two resident CTAs/CU.  Register
// allocation remains benchmark- and ACU-gated.
template <int NComputeThreads>
struct K2NLaunchConfig {
    static constexpr int NThreads = NComputeThreads;
    static constexpr int MinBlocksPerCU = 2;
};

// State access is specialized as part of the kernel type.  The compact legacy
// accessor contains one typed BF16 pointer; the generic accessor alone carries
// index and dtype metadata.  This keeps those values (and their branches) out
// of the register-resident recurrence when all active state endpoints use the
// historical compact-BF16 layout.
template <bool CompactBf16>
struct K2NStateInAccessor;

template <>
struct K2NStateInAccessor<true> {
    const BF16* ptr;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return logical_slot;
    }
    __device__ __forceinline__ const BF16* bf16_ptr(int64_t offset) const {
        return ptr + offset;
    }
};

template <>
struct K2NStateInAccessor<false> {
    const void* ptr;
    const int32_t* indices;
    bool fp32;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return indices ? indices[logical_slot] : logical_slot;
    }
    __device__ __forceinline__ bool is_fp32() const { return fp32; }
    __device__ __forceinline__ const BF16* bf16_ptr(int64_t offset) const {
        return reinterpret_cast<const BF16*>(ptr) + offset;
    }
    __device__ __forceinline__ const float* fp32_ptr(int64_t offset) const {
        return reinterpret_cast<const float*>(ptr) + offset;
    }
};

template <bool CompactBf16>
struct K2NStateOutAccessor;

template <>
struct K2NStateOutAccessor<true> {
    BF16* ptr;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return logical_slot;
    }
    __device__ __forceinline__ BF16* bf16_ptr(int64_t offset) const {
        return ptr + offset;
    }
};

template <>
struct K2NStateOutAccessor<false> {
    void* ptr;
    const int32_t* indices;
    bool fp32;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return indices ? indices[logical_slot] : logical_slot;
    }
    __device__ __forceinline__ bool is_fp32() const { return fp32; }
    __device__ __forceinline__ BF16* bf16_ptr(int64_t offset) const {
        return reinterpret_cast<BF16*>(ptr) + offset;
    }
    __device__ __forceinline__ float* fp32_ptr(int64_t offset) const {
        return reinterpret_cast<float*>(ptr) + offset;
    }
};

template <bool CompactBf16>
inline K2NStateInAccessor<CompactBf16> make_k2n_state_in_accessor(
    const void* ptr, const int32_t* indices, bool is_fp32) {
    if constexpr (CompactBf16)
        return {reinterpret_cast<const BF16*>(ptr)};
    else
        return {ptr, indices, is_fp32};
}

template <bool CompactBf16>
inline K2NStateOutAccessor<CompactBf16> make_k2n_state_out_accessor(
    void* ptr, const int32_t* indices, bool is_fp32) {
    if constexpr (CompactBf16)
        return {reinterpret_cast<BF16*>(ptr)};
    else
        return {ptr, indices, is_fp32};
}

// Shared BF16 load/store helpers are used by both accessor specializations.
// Only the small generic selection remains dynamic; the recurrence body itself
// is not duplicated.
template <typename StateT, int NThreads, int CtaDV, int DKSplitWarps,
          bool DedicatedLeader>
__device__ __forceinline__ void k2n_load_state(
    const StateT* isp, int dv_base, int dv0, int dk_warp, int tid,
    bool is_dk_worker, const int* crow, const int* ccol,
    StateT* rstate, unsigned char* smem) {
    using C = K2NConfig<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    using S = K2NSmem<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    constexpr int COLB = C::COLB;
    constexpr int LOCAL_DKB = DKB / DKSplitWarps;
    StateT* sState = reinterpret_cast<StateT*>(smem + S::O_ws);
    if constexpr (CtaDV < DV) {
        if (tid == 0) {
            k2n_aiu_load_2d_b16(
                sState, isp + int64_t(dv_base) * DK,
                DK, CtaDV, int64_t(DK) * sizeof(StateT));
            k2n_aiu_commit();
            k2n_aiu_wait_group<0>();
        }
        __syncthreads();
    }
    if (is_dk_worker) {
        CUTE_UNROLL
        for (int ml = 0; ml < LOCAL_DKB; ++ml)
            CUTE_UNROLL
            for (int bi = 0; bi < COLB; ++bi)
                CUTE_UNROLL
                for (int e = 0; e < 8; ++e) {
                    int m = dk_warp * LOCAL_DKB + ml;
                    int dk = m * 16 + crow[e];
                    int dv = dv0 + bi * 16 + ccol[e];
                    if constexpr (CtaDV < DV)
                        rstate[(ml * COLB + bi) * 8 + e] =
                            sState[int64_t(dv - dv_base) * DK + dk];
                    else
                        rstate[(ml * COLB + bi) * 8 + e] =
                            isp[int64_t(dv) * DK + dk];
                }
    }
}

template <typename StateT, int NThreads, int CtaDV, int DKSplitWarps,
          bool DedicatedLeader>
__device__ __forceinline__ void k2n_store_state(
    StateT* fsp, int dv_base, int dv0, int dk_warp, int tid,
    bool is_dk_worker, const int* crow, const int* ccol,
    const StateT* rstate, unsigned char* smem) {
    using C = K2NConfig<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    using S = K2NSmem<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    constexpr int COLB = C::COLB;
    constexpr int LOCAL_DKB = DKB / DKSplitWarps;
    StateT* sState = reinterpret_cast<StateT*>(smem + S::O_ws);
    if (is_dk_worker) {
        CUTE_UNROLL
        for (int ml = 0; ml < LOCAL_DKB; ++ml)
            CUTE_UNROLL
            for (int bi = 0; bi < COLB; ++bi)
                CUTE_UNROLL
                for (int e = 0; e < 8; ++e) {
                    int m = dk_warp * LOCAL_DKB + ml;
                    int dk = m * 16 + crow[e];
                    int dv = dv0 + bi * 16 + ccol[e];
                    if constexpr (CtaDV < DV)
                        sState[int64_t(dv - dv_base) * DK + dk] =
                            rstate[(ml * COLB + bi) * 8 + e];
                    else
                        fsp[int64_t(dv) * DK + dk] =
                            rstate[(ml * COLB + bi) * 8 + e];
                }
    }
    if constexpr (CtaDV < DV) {
        __syncthreads();
        constexpr int NVEC =
            CtaDV * DK * int(sizeof(StateT)) / int(sizeof(uint4));
        const uint4* src4 = reinterpret_cast<const uint4*>(sState);
        uint4* dst4 = reinterpret_cast<uint4*>(fsp + int64_t(dv_base) * DK);
        for (int i = tid; i < NVEC; i += NThreads) dst4[i] = src4[i];
    }
}

template <typename StateT, int NThreads, int CtaDV, int DKSplitWarps,
          bool DedicatedLeader>
__device__ __forceinline__ void k2n_store_bf16_state_converted(
    BF16* fsp, int dv_base, int dv0, int dk_warp, int tid,
    bool is_dk_worker, const int* crow, const int* ccol,
    const StateT* rstate, unsigned char* smem) {
    using C = K2NConfig<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    using S = K2NSmem<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    constexpr int COLB = C::COLB;
    constexpr int LOCAL_DKB = DKB / DKSplitWarps;
    BF16* sState = reinterpret_cast<BF16*>(smem + S::O_ws);
    if (is_dk_worker) {
        CUTE_UNROLL
        for (int ml = 0; ml < LOCAL_DKB; ++ml)
            CUTE_UNROLL
            for (int bi = 0; bi < COLB; ++bi)
                CUTE_UNROLL
                for (int e = 0; e < 8; ++e) {
                    int m = dk_warp * LOCAL_DKB + ml;
                    int dk = m * 16 + crow[e];
                    int dv = dv0 + bi * 16 + ccol[e];
                    BF16 value(k2n_state_to_f32(
                        rstate[(ml * COLB + bi) * 8 + e]));
                    if constexpr (CtaDV < DV)
                        sState[int64_t(dv - dv_base) * DK + dk] = value;
                    else
                        fsp[int64_t(dv) * DK + dk] = value;
                }
    }
    if constexpr (CtaDV < DV) {
        __syncthreads();
        constexpr int NVEC = CtaDV * DK * int(sizeof(BF16)) / int(sizeof(uint4));
        const uint4* src4 = reinterpret_cast<const uint4*>(sState);
        uint4* dst4 = reinterpret_cast<uint4*>(fsp + int64_t(dv_base) * DK);
        for (int i = tid; i < NVEC; i += NThreads) dst4[i] = src4[i];
    }
}

template <int NThreads, bool IsVarlen, bool HasStateIn, bool HasStateOut,
          bool CompactBf16State,
          int CtaDV = DV, int DKSplitWarps = 1,
          bool DedicatedLeader = false,
          bool UseSegTable = false, bool PreScan = false,
          bool Tc01KrTranspose = false,
          bool Tc01VectorOutput = false,
          bool Fp16State = false>
__global__ void __launch_bounds__(
    K2NLaunchConfig<NThreads>::NThreads,
    K2NLaunchConfig<NThreads>::MinBlocksPerCU)
flashkda_recurrence_nows_kernel(
    BF16* __restrict__ o,             // [T_total, H, DV]
    const BF16* __restrict__ v,       // [T_total, H, DV]
    const BF16* __restrict__ beta,    // [H*T_total]
    const BF16* __restrict__ ws,      // merged workspace [nht, WS_SLOT_UNITS]
    K2NStateInAccessor<CompactBf16State> init_state,
    K2NStateOutAccessor<CompactBf16State> final_state,
    int T_total, int H, int N, int total_tiles,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    // CP segment descriptors, stride KDA_SEG_STRIDE int32 per segment:
    //   [0] tok_begin (absolute token index of the segment's first token)
    //   [1] seg_len   (tokens in the segment; only a sequence's last segment
    //                  may end on a partial CH tile)
    //   [2] ws_tile_base (absolute workspace tile index of the first tile)
    //   [3] reserved
    const int* __restrict__ seg_tbl = nullptr)
{
    static_assert(DV % CtaDV == 0, "CTA value tile must divide DV");
    static_assert(DKB % DKSplitWarps == 0,
                  "DK blocks must divide evenly across split warps");
    static_assert(!PreScan || UseSegTable,
                  "pre_scan is only meaningful over CP segments");
    static_assert(!PreScan || !HasStateIn,
                  "pre_scan bias chain always starts from zero; the real "
                  "initial_state is injected by the serial merge");
    static_assert(!PreScan || !DedicatedLeader,
                  "pre_scan is incompatible with the affine dedicated-leader "
                  "template (its leader warp reads V unconditionally and K1 "
                  "folds INV@diag(beta) away); CP always pairs with plain K1");
    static_assert(!Tc01VectorOutput
                      || (IsVarlen && NThreads == 128 && CtaDV == DV
                          && DKSplitWarps == 1 && !DedicatedLeader
                          && !UseSegTable && !PreScan),
                  "tc01 vector output is restricted to the ordinary varlen "
                  "BV128/NT128/DK1 specialization");
    using C = K2NConfig<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    using S = K2NSmem<NThreads, CtaDV, DKSplitWarps, DedicatedLeader>;
    constexpr int DVT = C::DVT;
    constexpr int COLB = C::COLB;
    constexpr int LOCAL_DKB = DKB / DKSplitWarps;
    constexpr int ZSPLIT = DV / CtaDV;
    using StateT = typename std::conditional<Fp16State, FP16, BF16>::type;
    using KdT = typename std::conditional<Fp16State, cute::half_t, BF16>::type;
    using QdT = typename std::conditional<Fp16State, cute::half_t, BF16>::type;
    extern __shared__ __align__(128) unsigned char smem[];
    uint32_t* sScr = reinterpret_cast<uint32_t*>(smem + S::O_scr);

    // pre_scan doubles grid.z: [0,ZSPLIT) carries the bias chain B, [ZSPLIT,2*ZSPLIT)
    // the transition chain M (identity init, v := 0).
    const bool chain_m = PreScan && int(blockIdx.z) >= ZSPLIT;
    int seq = blockIdx.x, head = blockIdx.y;
    int dv_tile = PreScan ? (int(blockIdx.z) % ZSPLIT) : int(blockIdx.z);
#if defined(FLASHKDA_ARCH_150)
    // Order the flattened CTA grid by request, head, then DV slice on tc02.
    // Segment/scan and fixed-length paths keep their original mapping.
    if constexpr (IsVarlen && !UseSegTable) {
        if (N > 1) {
            const int task = (int(blockIdx.z) * H + int(blockIdx.y)) * N
                           + int(blockIdx.x);
            const int request_head = task / ZSPLIT;
            dv_tile = task % ZSPLIT;
            head = request_head % H;
            seq = request_head / H;
        }
    }
#endif
    const int dv_base = dv_tile * CtaDV;
    const int block_tid = threadIdx.x;

    int64_t bos, eos; int tile_base;
    int state_out_slot;
    if constexpr (UseSegTable) {
        const int* d = seg_tbl + seq * fk::KDA_SEG_STRIDE;
        bos = d[fk::KDA_SEG_TOK];
        eos = bos + d[fk::KDA_SEG_LEN];
        tile_base = d[fk::KDA_SEG_WSBASE];
        if constexpr (PreScan) {
            // out_slot >= 0 marks a sequence's LAST segment.  The merge only
            // composes segments 0..cnt-2, so that segment's two chains have no
            // consumer at all -- drop the whole CTA instead of carrying them.
            // The test is CTA-uniform and precedes every barrier.
            if (d[fk::KDA_SEG_OUTSLOT] >= 0) return;
            // pre_scan owns a [2, n_seg] output buffer: bias chain then
            // transition chain.
            state_out_slot = int(chain_m) * N + seq;
        } else {
            // The segment pass writes final_state only where a consumer exists,
            // which saves (n_seg - N) x H x 32 KiB of HBM traffic.
            state_out_slot = d[fk::KDA_SEG_OUTSLOT];
        }
    } else if constexpr (IsVarlen) {
        bos = cu_seqlens[seq]; eos = cu_seqlens[seq + 1];
        state_out_slot = seq;
        if (tile_prefix) {
            tile_base = tile_prefix[seq];
        } else {
            tile_base = 0;
            for (int i = 0; i < seq; ++i)
                tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CH - 1) / CH;
        }
    } else {
        int T_seq = T_total / N;
        bos = int64_t(seq) * T_seq; eos = bos + T_seq;
        tile_base = seq * ((T_seq + CH - 1) / CH);
        state_out_slot = seq;
    }
    int seq_len = int(eos - bos);
    int t_tiles = (seq_len + CH - 1) / CH;

    const int tid = block_tid;
    const int warp = tid / WARP, lane = tid % WARP;
    const int worker_warp = warp - (DedicatedLeader ? 1 : 0);
    const bool is_dk_worker = !DedicatedLeader || worker_warp >= 0;
    const int dv_warp = worker_warp >= 0 ? worker_warp / DKSplitWarps : 0;
    const int dk_warp = worker_warp >= 0 ? worker_warp % DKSplitWarps : 0;
    const bool is_group_leader = DedicatedLeader ? warp == 0 : dk_warp == 0;
    const int dv_local = dv_warp * DVT;
    const int dv0 = dv_base + dv_local;          // global first value column
    uint32_t* warpScr;
    if constexpr (DKSplitWarps == 1)
        warpScr = sScr + warp * 4 * WARP;
    else
        warpScr = sScr + warp * WARP;

    // ---- MMA thread slice + generic C-fragment (row,col) map (layout-agnostic) ----
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    int crow[8], ccol[8];
    {
        auto cC = thr_mma.partition_C(make_identity_tensor(make_shape(_16{}, _16{})));
        int pos = 0;
        CUTE_UNROLL
        for (int m = 0; m < cute::size<1>(cC); ++m)
            CUTE_UNROLL
            for (int n = 0; n < cute::size<2>(cC); ++n)
                CUTE_UNROLL
                for (int vv = 0; vv < cute::size<0>(cC); ++vv) {
                    auto co = cC(vv, m, n);
                    crow[pos] = cute::get<0>(co);
                    ccol[pos] = cute::get<1>(co);
                    ++pos;
                }
    }

    // ---- Register-resident state: rstate[dk-block m][dv-block bi][8 elems] ----
    StateT rstate[LOCAL_DKB][COLB][8];
    if constexpr (HasStateIn) {
        // init_state[slot,h,dv,dk] with element [dv,dk]; St[dk,dv] slice for this warp.
        const int state_slot = init_state.slot(seq);
        const int64_t state_off = (int64_t(state_slot) * H + head) * DV * DK;
        // The value-split paths otherwise issue 2048/4096 scattered scalar
        // global loads per CTA.  Reuse the first workspace staging buffer before
        // the mainloop: one AIU 2D load brings the contiguous [CtaDV,DK] slice to
        // shared memory, after which the probe-derived C-fragment gather is local.
        // Full-DV nows cannot fit the 32 KiB state tile in its 13.5 KiB ws slot,
        // so it retains the original direct path.
        if constexpr (CompactBf16State) {
            k2n_load_state<StateT,
                NThreads, CtaDV, DKSplitWarps, DedicatedLeader>(
                reinterpret_cast<const StateT*>(init_state.bf16_ptr(state_off)),
                dv_base, dv0, dk_warp, tid, is_dk_worker, crow, ccol,
                &rstate[0][0][0], smem);
        } else if (!init_state.is_fp32()) {
            k2n_load_state<StateT,
                NThreads, CtaDV, DKSplitWarps, DedicatedLeader>(
                reinterpret_cast<const StateT*>(init_state.bf16_ptr(state_off)),
                dv_base, dv0, dk_warp, tid, is_dk_worker, crow, ccol,
                &rstate[0][0][0], smem);
        } else if (is_dk_worker) {
            const float* isp = init_state.fp32_ptr(state_off);
            CUTE_UNROLL
            for (int ml = 0; ml < LOCAL_DKB; ++ml)
                CUTE_UNROLL
                for (int bi = 0; bi < COLB; ++bi)
                    CUTE_UNROLL
                    for (int e = 0; e < 8; ++e) {
                        int m = dk_warp * LOCAL_DKB + ml;
                        int dk = m * 16 + crow[e];
                        int dv = dv0 + bi * 16 + ccol[e];
                        rstate[ml][bi][e] =
                            k2n_state_from_f32<StateT>(isp[int64_t(dv) * DK + dk]);
                    }
        }
        // Keep the state gather ahead of the chunk-0 AIU write to the same
        // physical smem region.
        if constexpr (CtaDV < DV) __syncthreads();
    } else if constexpr (PreScan) {
        // Bias chain B starts at zero; transition chain M starts at the
        // identity, which is what makes the same recurrence propagate the
        // segment's affine operator (cuLA's M/B duality).
        CUTE_UNROLL
        for (int ml = 0; ml < LOCAL_DKB; ++ml)
            CUTE_UNROLL
            for (int bi = 0; bi < COLB; ++bi)
                CUTE_UNROLL
                for (int e = 0; e < 8; ++e) {
                    int m = dk_warp * LOCAL_DKB + ml;
                    int dk = m * 16 + crow[e];
                    int col = dv0 + bi * 16 + ccol[e];
                    rstate[ml][bi][e] = k2n_state_from_f32<StateT>(
                        (chain_m && dk == col) ? 1.f : 0.f);
                }
    } else {
        CUTE_UNROLL
        for (int ml = 0; ml < LOCAL_DKB; ++ml)
            CUTE_UNROLL
            for (int bi = 0; bi < COLB; ++bi)
                CUTE_UNROLL
                for (int e = 0; e < 8; ++e)
                    rstate[ml][bi][e] = k2n_state_from_f32<StateT>(0.f);
    }

    for (int t = 0; t < t_tiles; ++t) {
        int actual = min(CH, seq_len - t * CH);
        int ws_idx = head * total_tiles + tile_base + t;
        int64_t tok0 = bos + int64_t(t) * CH;

        BF16* sWs;
        BF16* sV;
        float* sBeta;
        if constexpr (S::Prefetch) {
            // P5 exact self-pipeline: double-buffered {ws,v,beta}.  Issue chunk
            // t+1's async loads before computing chunk t, then a single
            // wait_group<1> keeps t+1 in flight while draining t.  Bit-exact:
            // the chunk inputs are state-independent, so only latency changes.
            // WAR safety comes from the loop-close __syncthreads (it fences
            // iteration t-1's reads of buf[(t-1)&1] before iteration t writes
            // the same physical buffer buf[(t+1)&1]).  Gated to CtaDV<DV so
            // only the low-occupancy dvsplit paths pay the doubled smem.
            const int cur = t % S::PF_STAGES;
            sWs   = reinterpret_cast<BF16*>(smem + S::O_ws   + cur * S::PF_STRIDE);
            sV    = reinterpret_cast<BF16*>(smem + S::O_v    + cur * S::PF_STRIDE);
            sBeta = reinterpret_cast<float*>(smem + S::O_beta + cur * S::PF_STRIDE);
            if (t == 0) {
                // prologue: prime chunk 0 into buffer 0
                k2n_load_chunk<NThreads, CtaDV, /*DeferWait=*/true>(
                    actual, ws_idx, tok0, H, T_total, head, dv_base, tid,
                    v, beta, ws, sWs, sV, sBeta, /*skip_v=*/chain_m);
            }
            if (t + 1 < t_tiles) {
                const int nxt = (t + 1) % S::PF_STAGES;
                int actual_n = min(CH, seq_len - (t + 1) * CH);
                int ws_idx_n = head * total_tiles + tile_base + (t + 1);
                int64_t tok0_n = bos + int64_t(t + 1) * CH;
                BF16*  sWs_n   = reinterpret_cast<BF16*>(smem + S::O_ws   + nxt * S::PF_STRIDE);
                BF16*  sV_n    = reinterpret_cast<BF16*>(smem + S::O_v    + nxt * S::PF_STRIDE);
                float* sBeta_n = reinterpret_cast<float*>(smem + S::O_beta + nxt * S::PF_STRIDE);
                k2n_load_chunk<NThreads, CtaDV, /*DeferWait=*/true>(
                    actual_n, ws_idx_n, tok0_n, H, T_total, head, dv_base, tid,
                    v, beta, ws, sWs_n, sV_n, sBeta_n, /*skip_v=*/chain_m);
                if (tid == 0) k2n_aiu_wait_group<1>();
            } else {
                if (tid == 0) k2n_aiu_wait_group<0>();
            }
            __syncthreads();
        } else {
            sWs = reinterpret_cast<BF16*>(smem + S::O_ws);
            sV = reinterpret_cast<BF16*>(smem + S::O_v);
            sBeta = reinterpret_cast<float*>(smem + S::O_beta);
            k2n_load_chunk<NThreads, CtaDV>(
                actual, ws_idx, tok0, H, T_total, head, dv_base, tid,
                v, beta, ws, sWs, sV, sBeta, /*skip_v=*/chain_m);
            __syncthreads();
        }

        // K1 stores these private fields as FP16 exactly when the CP state is
        // FP16. The slot remains a raw 16-bit byte image, so offsets and AIU
        // staging are unchanged.
        KdT* sKd = reinterpret_cast<KdT*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_KD);
        sBeta = reinterpret_cast<float*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_BETA);
        QdT* sQd = reinterpret_cast<QdT*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_QD);
        BF16* sKr = reinterpret_cast<BF16*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_KR);
        BF16* sInv = reinterpret_cast<BF16*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_INV);
        BF16* sMqk = reinterpret_cast<BF16*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_MQK);
        float* sGt = reinterpret_cast<float*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_GT);

        auto Kd = make_tensor(make_smem_ptr(sKd),
            make_layout(make_shape(_16{}, Int<DK>{}), make_stride(Int<DK>{}, _1{})));
        auto Qd = make_tensor(make_smem_ptr(sQd),
            make_layout(make_shape(_16{}, Int<DK>{}), make_stride(Int<DK>{}, _1{})));
        auto Inv = make_tensor(make_smem_ptr(sInv),
            make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{})));
        auto Mqk = make_tensor(make_smem_ptr(sMqk),
            make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{})));
        auto Beta = make_tensor(make_smem_ptr(sBeta),
            make_layout(make_shape(_16{}, _16{}), make_stride(_1{}, _0{})));
        auto beta_part = thr_mma.partition_C(Beta);

        // The dedicated warp has no Phase-1 state slice, so use that otherwise
        // idle interval to fetch its V C-fragment.  Keeping the eight bf16
        // values packed in four registers hides the shared-memory reads behind
        // the workers' Kd/Qd MMAs without changing the subsequent fp32
        // subtraction, rounding, or movmatrix order.
        uint32_t leader_v_packed[4];
        uint32_t leader_mqk_a[4];
        if constexpr (DedicatedLeader) {
            if (is_group_leader) {
                auto V = make_tensor(make_smem_ptr(sV + dv_local),
                    make_layout(
                        make_shape(_16{}, _16{}),
                        make_stride(Int<CtaDV>{}, _1{})));
                auto v_part = thr_mma.partition_C(V);
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    leader_v_packed[i] = fk::pack_half2_u32(
                        v_part(2 * i).storage, v_part(2 * i + 1).storage);
                }
                fk::load16_A_1warp_bf16(Mqk, lane, leader_mqk_a);
            }
        }

        // ===== Phase 1: vsub[bi] = kd @ St[bi] ; o1[bi] = qd @ St[bi] =====
        auto vsub0 = k2n_make_phase1_acc<Fp16State>(lane);
        auto vsub1 = k2n_make_phase1_acc<Fp16State>(lane);
        auto o1_0  = k2n_make_phase1_acc<Fp16State>(lane);
        auto o1_1  = k2n_make_phase1_acc<Fp16State>(lane);
        cute::clear(vsub0); cute::clear(vsub1);
        cute::clear(o1_0);  cute::clear(o1_1);

        if (is_dk_worker) {
            CUTE_UNROLL
            for (int ml = 0; ml < LOCAL_DKB; ++ml) {
                int m = dk_warp * LOCAL_DKB + ml;
                auto KdBlk = local_tile(Kd, make_shape(_16{}, _16{}), make_coord(0, m));
                auto QdBlk = local_tile(Qd, make_shape(_16{}, _16{}), make_coord(0, m));
                // bi=0
                {
                    uint32_t c_regs[4], b_regs[4];
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        if constexpr (Fp16State) {
                            c_regs[i] = fk::pack_half2_u32(
                                k2n_state_bits(rstate[ml][0][2 * i]),
                                k2n_state_bits(rstate[ml][0][2 * i + 1]));
                        } else {
                            c_regs[i] = fk::pack_half2_u32(
                                k2n_state_as_bf16_bits(rstate[ml][0][2 * i]),
                                k2n_state_as_bf16_bits(rstate[ml][0][2 * i + 1]));
                        }
                    }
                    NOWS_CFRAG_TO_BFRAG(c_regs, b_regs, lane, warpScr);
                    if constexpr (Fp16State) {
                        fk::gemm16_sr_1warp_f16_acc(
                            KdBlk, b_regs, lane, vsub0);
                    } else {
                        fk::gemm16_sr_1warp_bf16_acc(
                            KdBlk, b_regs, lane, vsub0);
                    }
                    // pre_scan carries state only; O is produced by the segment
                    // pass, so its qd@St / Mqk@U branch is elided entirely.
                    if constexpr (!PreScan) {
                        if constexpr (Fp16State) {
                            fk::gemm16_sr_1warp_f16_acc(
                                QdBlk, b_regs, lane, o1_0);
                        } else {
                            fk::gemm16_sr_1warp_bf16_acc(
                                QdBlk, b_regs, lane, o1_0);
                        }
                    }
                }
                // bi=1 exists only in the 128-thread / 32-column-per-warp path.
                if constexpr (COLB > 1) {
                    uint32_t c_regs[4], b_regs[4];
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        if constexpr (Fp16State) {
                            c_regs[i] = fk::pack_half2_u32(
                                k2n_state_bits(rstate[ml][1][2 * i]),
                                k2n_state_bits(rstate[ml][1][2 * i + 1]));
                        } else {
                            c_regs[i] = fk::pack_half2_u32(
                                k2n_state_as_bf16_bits(rstate[ml][1][2 * i]),
                                k2n_state_as_bf16_bits(rstate[ml][1][2 * i + 1]));
                        }
                    }
                    NOWS_CFRAG_TO_BFRAG(c_regs, b_regs, lane, warpScr);
                    if constexpr (Fp16State) {
                        fk::gemm16_sr_1warp_f16_acc(
                            KdBlk, b_regs, lane, vsub1);
                    } else {
                        fk::gemm16_sr_1warp_bf16_acc(
                            KdBlk, b_regs, lane, vsub1);
                    }
                    if constexpr (!PreScan) {
                        if constexpr (Fp16State) {
                            fk::gemm16_sr_1warp_f16_acc(
                                QdBlk, b_regs, lane, o1_1);
                        } else {
                            fk::gemm16_sr_1warp_bf16_acc(
                                QdBlk, b_regs, lane, o1_1);
                        }
                    }
                }
            }
        }

        // ===== Phases 2-4 per bi: vbeta -> U -> out += Mqk@U, plus cache U^T =====
        uint32_t u_bregs[COLB][4];
        auto rv0 = recast<float>(vsub0);
        auto rv1 = recast<float>(vsub1);
        float* sDKReduce = reinterpret_cast<float*>(smem + S::O_dkreduce);
        if constexpr (DKSplitWarps > 1) {
            constexpr bool SplitVOReduction =
                DedicatedLeader && DKSplitWarps == 4;
            static_assert(COLB == 1,
                          "DK-split candidate currently targets one BV16 tile");
            auto ro1_partial = recast<float>(o1_0);
            if (is_dk_worker) {
                float* vdst0 = k2n_reduce_vec4<SplitVOReduction>(
                    sDKReduce, warp * 2 + 0, 0, lane);
                float* vdst1 = k2n_reduce_vec4<SplitVOReduction>(
                    sDKReduce, warp * 2 + 0, 1, lane);
                float* odst0 = k2n_reduce_vec4<SplitVOReduction>(
                    sDKReduce, warp * 2 + 1, 0, lane);
                float* odst1 = k2n_reduce_vec4<SplitVOReduction>(
                    sDKReduce, warp * 2 + 1, 1, lane);
                *reinterpret_cast<float4*>(vdst0) =
                    make_float4(rv0(0), rv0(1), rv0(2), rv0(3));
                *reinterpret_cast<float4*>(vdst1) =
                    make_float4(rv0(4), rv0(5), rv0(6), rv0(7));
                *reinterpret_cast<float4*>(odst0) = make_float4(
                    ro1_partial(0), ro1_partial(1), ro1_partial(2), ro1_partial(3));
                *reinterpret_cast<float4*>(odst1) = make_float4(
                    ro1_partial(4), ro1_partial(5), ro1_partial(6), ro1_partial(7));
            }
            __syncthreads();
            // Reduce the independent output branch on worker warp 1 while the
            // dedicated leader reduces vsub.  Publish the fp32 C fragment into
            // warp-0's otherwise-unused o-partial slot; the existing U barrier
            // is also the visibility fence, so this adds no synchronization and
            // preserves the accumulation order 1,2,3,4.
            if constexpr (SplitVOReduction) {
                if (warp == 1) {
                    const float* osrc00 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, 1 * 2 + 1, 0, lane);
                    const float* osrc01 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, 1 * 2 + 1, 1, lane);
                    float4 os0 = *reinterpret_cast<const float4*>(osrc00);
                    float4 os1 = *reinterpret_cast<const float4*>(osrc01);
                    CUTE_UNROLL
                    for (int w = 2; w <= DKSplitWarps; ++w) {
                        const float* osrc0 = k2n_reduce_vec4<SplitVOReduction>(
                            sDKReduce, w * 2 + 1, 0, lane);
                        const float* osrc1 = k2n_reduce_vec4<SplitVOReduction>(
                            sDKReduce, w * 2 + 1, 1, lane);
                        float4 ot0 = *reinterpret_cast<const float4*>(osrc0);
                        float4 ot1 = *reinterpret_cast<const float4*>(osrc1);
                        os0.x += ot0.x; os0.y += ot0.y;
                        os0.z += ot0.z; os0.w += ot0.w;
                        os1.x += ot1.x; os1.y += ot1.y;
                        os1.z += ot1.z; os1.w += ot1.w;
                    }
                    float* odst0 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, 0 * 2 + 1, 0, lane);
                    float* odst1 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, 0 * 2 + 1, 1, lane);
                    *reinterpret_cast<float4*>(odst0) = os0;
                    *reinterpret_cast<float4*>(odst1) = os1;
                }
            }
            if (is_group_leader) {
                int warp0_v = (DedicatedLeader ? 1 : 0)
                    + dv_warp * DKSplitWarps;
                int warp0_o = warp0_v;
                const float* vsrc00 = k2n_reduce_vec4<SplitVOReduction>(
                    sDKReduce, warp0_v * 2 + 0, 0, lane);
                const float* vsrc01 = k2n_reduce_vec4<SplitVOReduction>(
                    sDKReduce, warp0_v * 2 + 0, 1, lane);
                float4 vs0 = *reinterpret_cast<const float4*>(vsrc00);
                float4 vs1 = *reinterpret_cast<const float4*>(vsrc01);
                float4 os0 = make_float4(0.f, 0.f, 0.f, 0.f);
                float4 os1 = make_float4(0.f, 0.f, 0.f, 0.f);
                if constexpr (!SplitVOReduction) {
                    const float* osrc00 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, warp0_o * 2 + 1, 0, lane);
                    const float* osrc01 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, warp0_o * 2 + 1, 1, lane);
                    os0 = *reinterpret_cast<const float4*>(osrc00);
                    os1 = *reinterpret_cast<const float4*>(osrc01);
                }
                CUTE_UNROLL
                for (int w = 1; w < DKSplitWarps; ++w) {
                    int pvw = warp0_v + w;
                    const float* vsrc0 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, pvw * 2 + 0, 0, lane);
                    const float* vsrc1 = k2n_reduce_vec4<SplitVOReduction>(
                        sDKReduce, pvw * 2 + 0, 1, lane);
                    float4 vt0 = *reinterpret_cast<const float4*>(vsrc0);
                    float4 vt1 = *reinterpret_cast<const float4*>(vsrc1);
                    vs0.x += vt0.x; vs0.y += vt0.y;
                    vs0.z += vt0.z; vs0.w += vt0.w;
                    vs1.x += vt1.x; vs1.y += vt1.y;
                    vs1.z += vt1.z; vs1.w += vt1.w;
                    if constexpr (!SplitVOReduction) {
                        int pow = warp0_o + w;
                        const float* osrc0 = k2n_reduce_vec4<SplitVOReduction>(
                            sDKReduce, pow * 2 + 1, 0, lane);
                        const float* osrc1 = k2n_reduce_vec4<SplitVOReduction>(
                            sDKReduce, pow * 2 + 1, 1, lane);
                        float4 ot0 = *reinterpret_cast<const float4*>(osrc0);
                        float4 ot1 = *reinterpret_cast<const float4*>(osrc1);
                        os0.x += ot0.x; os0.y += ot0.y;
                        os0.z += ot0.z; os0.w += ot0.w;
                        os1.x += ot1.x; os1.y += ot1.y;
                        os1.z += ot1.z; os1.w += ot1.w;
                    }
                }
                rv0(0) = vs0.x; rv0(1) = vs0.y;
                rv0(2) = vs0.z; rv0(3) = vs0.w;
                rv0(4) = vs1.x; rv0(5) = vs1.y;
                rv0(6) = vs1.z; rv0(7) = vs1.w;
                if constexpr (!SplitVOReduction) {
                    ro1_partial(0) = os0.x; ro1_partial(1) = os0.y;
                    ro1_partial(2) = os0.z; ro1_partial(3) = os0.w;
                    ro1_partial(4) = os1.x; ro1_partial(5) = os1.y;
                    ro1_partial(6) = os1.z; ro1_partial(7) = os1.w;
                }
            }
        }
        // Only the group leader owns V/U/output. Other warps rejoin after the
        // U B-fragment broadcast and update their disjoint DK rows.
        if (DKSplitWarps == 1 || is_group_leader) {
        CUTE_UNROLL
        for (int bi = 0; bi < COLB; ++bi) {
            int lcol = dv_local + bi * 16;
            int gcol = dv0 + bi * 16;
            auto V = make_tensor(make_smem_ptr(sV + lcol),
                make_layout(
                    make_shape(_16{}, _16{}),
                    make_stride(Int<CtaDV>{}, _1{})));
            auto v_part = thr_mma.partition_C(V);
            auto& rvsub = (bi == 0) ? rv0 : rv1;

            // Ordinary path forms vbeta=(v-vsub)*beta.  The dedicated-leader
            // path consumes K1's P/O affine matrices, which already include
            // INV@diag(beta), so its recurrent input is x=(v-vsub).
            uint32_t c_half[8];
            CUTE_UNROLL
            for (int e = 0; e < 8; ++e) {
                float vv;
                if constexpr (DedicatedLeader) {
                    BF16 vv_bf16;
                    vv_bf16.storage = fk::get_half_from_pack(
                        leader_v_packed[e / 2], e & 1);
                    vv = fk::bf16_to_f32(vv_bf16);
                } else if (PreScan && chain_m) {
                    // Transition chain: St' = A_t.St is exactly this recurrence
                    // with v := 0, so sV is never even loaded for chain_m.
                    vv = 0.f;
                } else {
                    vv = fk::bf16_to_f32(v_part(e));
                }
                float val = vv - rvsub(e);
                if constexpr (!DedicatedLeader)
                    val *= beta_part(e);
                c_half[e] = BF16(val).storage;
            }
            uint32_t c_regs[4], vbeta_b[4];
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i)
                c_regs[i] = fk::pack_half2_u32(c_half[2 * i], c_half[2 * i + 1]);
            NOWS_CFRAG_TO_BFRAG(c_regs, vbeta_b, lane, warpScr);

            if constexpr (DedicatedLeader) {
                // K1 stored O=Mqk@INV@diag(beta) and
                // P=kr^T@INV@diag(beta), so both consumers use x directly.
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) u_bregs[bi][i] = vbeta_b[i];
            } else {
                // U = INV @ vbeta  (INV @ B^T with B = vbeta^T)
                auto u_acc = fk::make_acc16_bf16_1warp(lane);
                cute::clear(u_acc);
                fk::gemm16_sr_1warp_bf16_acc(Inv, vbeta_b, lane, u_acc);
                // cache U^T as B-fragment for Mqk@U and kr^T@U
                NOWS_ACC_TO_BFRAG(u_acc, u_bregs[bi], lane, warpScr);
            }

            // In the ordinary DK-split path the group leader also finishes the
            // output before broadcasting U.  The dedicated-leader experiment
            // delays this independent GEMM until after the U barrier so it can
            // overlap the four worker warps' state update.  pre_scan has no
            // output at all: its local O would be computed from a zero/identity
            // carry, so both the Mqk@U GEMM and the store are dropped.
            if constexpr (!DedicatedLeader && !PreScan) {
                // out = o1 + Mqk @ U   (accumulate into o1 acc)
                auto& o1 = (bi == 0) ? o1_0 : o1_1;
                fk::gemm16_sr_1warp_bf16_acc(Mqk, u_bregs[bi], lane, o1);
                // Convert the fp32 fragment to bf16 through the compile-time
                // selected output epilogue.  The default keeps the direct
                // crow[]/ccol[] lane mapping; the tc01 A/B uses warpScr.
                {
                    if constexpr (Tc01VectorOutput) {
                        const int64_t row_stride = int64_t(H) * DV;
                        BF16* o_head = o + int64_t(tok0) * row_stride
                                     + int64_t(head) * DV + gcol;
                        tc01::k2n_store_output_vec(
                            o1, lane, actual, warpScr, o_head, row_stride);
                    } else {
                        auto ro1 = recast<float>(o1);
                        const int64_t row_stride = int64_t(H) * DV;
                        BF16* o_head = o + int64_t(tok0) * row_stride
                                     + int64_t(head) * DV + gcol;
#if defined(FLASHKDA_ARCH_150)
                        CUTE_UNROLL
                        for (int i = 0; i < 4; ++i) {
                            const int e = 2 * i;
                            int r = crow[e];
                            int c = ccol[e];
                            if (r < actual) {
                                uint32_t pair = fk::pack_half2_u32(
                                    BF16(ro1(e)).storage,
                                    BF16(ro1(e + 1)).storage);
                                *reinterpret_cast<uint32_t*>(
                                    o_head + int64_t(r) * row_stride + c) = pair;
                            }
                        }
#else
                        CUTE_UNROLL
                        for (int e = 0; e < 8; ++e) {
                            int r = crow[e];
                            int c = ccol[e];
                            if (r < actual)
                                o_head[int64_t(r) * row_stride + c] =
                                    BF16(ro1(e));
                        }
#endif
                    }
                }
            }
        }
        }

        if constexpr (DKSplitWarps > 1) {
            uint32_t* sUB = reinterpret_cast<uint32_t*>(sDKReduce);
            if (is_group_leader) {
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i)
                    sUB[(dv_warp * WARP + lane) * 4 + i] = u_bregs[0][i];
            }
            __syncthreads();
            // The producer already owns the canonical register fragment; only
            // consumers need to reload the shared broadcast image.
            if (!is_group_leader) {
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i)
                    u_bregs[0][i] = sUB[(dv_warp * WARP + lane) * 4 + i];
            }
            if constexpr (DedicatedLeader && DKSplitWarps == 4) {
                if (is_group_leader) {
                    const float* osrc0 = k2n_reduce_vec4<true>(
                        sDKReduce, 0 * 2 + 1, 0, lane);
                    const float* osrc1 = k2n_reduce_vec4<true>(
                        sDKReduce, 0 * 2 + 1, 1, lane);
                    float4 os0 = *reinterpret_cast<const float4*>(osrc0);
                    float4 os1 = *reinterpret_cast<const float4*>(osrc1);
                    auto ro1 = recast<float>(o1_0);
                    ro1(0) = os0.x; ro1(1) = os0.y;
                    ro1(2) = os0.z; ro1(3) = os0.w;
                    ro1(4) = os1.x; ro1(5) = os1.y;
                    ro1(6) = os1.z; ro1(7) = os1.w;
                }
            }
        }

        if constexpr (DedicatedLeader) {
            if (is_group_leader) {
                // x is now visible to all DK workers.  Finish the independent
                // output branch on the fifth warp while the other four execute
                // Phase 6; the loop-close CTA barrier joins the two branches.
                fk::gemm16_rr_1warp_bf16_acc(
                    leader_mqk_a, u_bregs[0], o1_0);
                auto ro1 = recast<float>(o1_0);
                const int64_t row_stride = int64_t(H) * DV;
                BF16* o_head = o + int64_t(tok0) * row_stride
                             + int64_t(head) * DV + dv0;
#if defined(FLASHKDA_ARCH_150)
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const int e = 2 * i;
                    int r = crow[e];
                    int c = ccol[e];
                    if (r < actual) {
                        uint32_t pair = fk::pack_half2_u32(
                            BF16(ro1(e)).storage, BF16(ro1(e + 1)).storage);
                        *reinterpret_cast<uint32_t*>(
                            o_head + int64_t(r) * row_stride + c) = pair;
                    }
                }
#else
                CUTE_UNROLL
                for (int e = 0; e < 8; ++e) {
                    int r = crow[e];
                    int c = ccol[e];
                    if (r < actual)
                        o_head[int64_t(r) * row_stride + c] = BF16(ro1(e));
                }
#endif
            }
        }

        // ===== Phase 6: delta = P @ x ; state update (in registers) =====
        if (is_dk_worker) {
            CUTE_UNROLL
            for (int ml = 0; ml < LOCAL_DKB; ++ml) {
                int m = dk_warp * LOCAL_DKB + ml;
                CUTE_UNROLL
                for (int bi = 0; bi < COLB; ++bi) {
                    auto delta = fk::make_acc16_bf16_1warp(lane);
                    cute::clear(delta);
                    if constexpr (Tc01KrTranspose)
                        fk::gemm16_sr_contig_krT_1warp_bf16_acc(
                            sKr, m, u_bregs[bi], lane, delta);
                    else
                        fk::gemm16_sr_swzlA_trans_1warp_bf16_acc(
                            sKr, m, u_bregs[bi], lane, delta);
                    auto rd = recast<float>(delta);
                    CUTE_UNROLL
                    for (int e = 0; e < 8; ++e) {
                        float g = sGt[m * 16 + crow[e]];
                        float old = k2n_state_to_f32(rstate[ml][bi][e]);
                        float updated = old * g + rd(e);
                        rstate[ml][bi][e] = k2n_state_from_f32<StateT>(updated);
                    }
                }
            }
        }
        // ===== M-chain exact early exit =====
        // The transition chain runs with v := 0, so x = (0 - kd@St)*beta and
        // delta = P@x: if this CTA's whole state slice is zero then vsub, x and
        // delta are all zero and the update degenerates to St' = St*g = 0.  A
        // zero slice is therefore an absorbing state -- every remaining chunk,
        // and the stored M_i image, is exactly zero -- so the chain can stop.
        // This is what makes CP affordable: KDA's decaying gate kills the
        // transition operator within a few dozen chunks for any realistic gate,
        // while the bias chain (which keeps absorbing v) must always run to the
        // end.  Sign bits are masked out: a negative underflow leaves -0.0,
        // which is absorbing too, and the only difference it can make to the
        // stored image is the sign of a zero.
        // Checked every ZCHK chunks so the two extra barriers amortise; both
        // barriers are CTA-wide, so borrowing the fragment scratch between them
        // cannot race with its in-loop uses.
        bool m_dead = false;
        if constexpr (PreScan) {
            constexpr int ZCHK = 8;
            if (chain_m && (t % ZCHK) == ZCHK - 1 && t + 1 < t_tiles) {
                uint32_t* sZ = reinterpret_cast<uint32_t*>(smem + S::O_scr);
                if (tid == 0) *sZ = 0u;
                __syncthreads();
                uint32_t bits = 0;
                if (is_dk_worker) {
                    CUTE_UNROLL
                    for (int ml = 0; ml < LOCAL_DKB; ++ml)
                        CUTE_UNROLL
                        for (int bi = 0; bi < COLB; ++bi)
                            CUTE_UNROLL
                            for (int e = 0; e < 8; ++e)
                                bits |= uint32_t(k2n_state_bits(rstate[ml][bi][e]))
                                        & 0x7fffu;
                }
                if (bits) atomicOr(sZ, 1u);
                __syncthreads();
                m_dead = (*sZ == 0u);
            }
        }
        __syncthreads();   // recycle double-buffered smem for chunk t+2
        if constexpr (PreScan) {
            if (m_dead) {
                // Chunk t+1's prefetch is already in flight and targets the
                // staging buffer the epilogue reuses for the state image.
                if (tid == 0) k2n_aiu_wait_group<0>();
                __syncthreads();
                break;
            }
        }
    }

    // ===== Store final state St^T -> final_state[slot,h,dv,dk] =====
    // state_out_slot is CTA-uniform (segment descriptor / chain id / seq), so the
    // skip branch below never diverges across the barrier in the staging path.
    if constexpr (HasStateOut) {
        if (state_out_slot >= 0) {
            const int state_slot = final_state.slot(state_out_slot);
            const int64_t state_off =
                (int64_t(state_slot) * H + head) * DV * DK;
            if constexpr (Fp16State && PreScan) {
                k2n_store_state<StateT,
                    NThreads, CtaDV, DKSplitWarps, DedicatedLeader>(
                    reinterpret_cast<StateT*>(final_state.bf16_ptr(state_off)),
                    dv_base, dv0, dk_warp, tid, is_dk_worker, crow, ccol,
                    &rstate[0][0][0], smem);
            } else if constexpr (CompactBf16State) {
                k2n_store_state<StateT,
                    NThreads, CtaDV, DKSplitWarps, DedicatedLeader>(
                    reinterpret_cast<StateT*>(final_state.bf16_ptr(state_off)),
                    dv_base, dv0, dk_warp, tid, is_dk_worker, crow, ccol,
                    &rstate[0][0][0], smem);
            } else if (final_state.is_fp32()) {
                float* fsp = final_state.fp32_ptr(state_off);
                if (is_dk_worker) {
                    CUTE_UNROLL
                    for (int ml = 0; ml < LOCAL_DKB; ++ml)
                        CUTE_UNROLL
                        for (int bi = 0; bi < COLB; ++bi)
                            CUTE_UNROLL
                            for (int e = 0; e < 8; ++e) {
                                int m = dk_warp * LOCAL_DKB + ml;
                                int dk = m * 16 + crow[e];
                                int dv = dv0 + bi * 16 + ccol[e];
                                fsp[int64_t(dv) * DK + dk] =
                                    k2n_state_to_f32(rstate[ml][bi][e]);
                            }
                }
            } else {
                k2n_store_bf16_state_converted<StateT,
                    NThreads, CtaDV, DKSplitWarps, DedicatedLeader>(
                    final_state.bf16_ptr(state_off), dv_base, dv0, dk_warp, tid,
                    is_dk_worker, crow, ccol, &rstate[0][0][0], smem);
            }
        }
    }
}

}  // namespace flashkda_cute_nows

void launch_flashkda_recurrence_nows(
    void* o, const void* v, const void* beta,
    const void* ws,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices, const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream) {
    using BF16 = cutlass::bfloat16_t;
    namespace fkn = flashkda_cute_nows;
    const bool is_tc01 = flashkda::host_runtime::current_device_is_tc01();
    const bool use_tc01_contiguous =
        kr_layout == flashkda::KrWorkspaceLayout::kContiguousRawKrT
        || kr_layout == flashkda::KrWorkspaceLayout::kContiguousAffineP;
    const bool force_normal_raw =
        kr_layout == flashkda::KrWorkspaceLayout::kNormalRaw;
    const bool use_affine_p =
        kr_layout == flashkda::KrWorkspaceLayout::kNormalAffineP
        || kr_layout == flashkda::KrWorkspaceLayout::kContiguousAffineP;
    dim3 grid(N, H);
    bool use_tile_prefix = is_varlen && (N >= 64);
    const int* tile_prefix = use_tile_prefix
        ? reinterpret_cast<const int*>(
              reinterpret_cast<const unsigned char*>(ws)
              + int64_t(H) * total_tiles * flashkda::WS_SLOT_BYTES)
        : nullptr;
    const bool compact_bf16_state =
        (!has_state_in || (!init_state_is_fp32 && init_state_indices == nullptr))
        && (!has_state_out
            || (!final_state_is_fp32 && final_state_indices == nullptr));

    // ---- Kernel2 recurrence dispatch (register-resident nows path) ----
    // Four independent knobs, each AUTO by shape with an env override:
    //   (1) dvsplit on/off  -> use_dvsplit   (PLA_FLASHKDA_K2_DVSPLIT=0 forces off)
    //   (2) BV tile 16/32   -> use_bv16      (PLA_FLASHKDA_K2_DVSPLIT=16/32 forces)
    //   (3) DK workers 1/2/4/8 -> requested_dk_split (PLA_FLASHKDA_K2_DKSPLIT)
    //   (4) dedicated leader warp -> requested_dedicated_leader (PLA_FLASHKDA_K2_DKLEADER)
    // The cascade at the bottom just maps (BV, dk_split, leader) -> thread count
    // NT for the constexpr template instantiation; all combos are numerically
    // identical (guarded by the err_ratio<1e-2 consistency gate).
    //
    // Register-resident value-dimension split. Each CTA owns a disjoint
    // state/output slice, so the K2 grid grows from (N,H) to (N,H,DV/BV) and a
    // low-head Batch=1 launch fills more of the 39 CUs. BV=32 splits DV into 4
    // slices (64-thread CTAs, two compute warps); BV=16 into 8 slices
    // (32-thread CTAs, one compute warp). Both preserve the original per-warp
    // 16-column arithmetic and are bitwise-independent of the slice count.
    //
    // A dvsplit sweep (H in {6..96} x S in {256..8192}, no_state + with_state,
    // card3, CV<2%) established the AUTO region and per-H tile choice:
    //   H<=24    -> BV16 (grid *8) fastest, 0.83-0.97x of the plain (N,H) nows
    //   24<H<=64 -> BV32 (grid *4) fastest, 0.85-0.97x; BV16 regresses at H>=64
    //   H>=78    -> gains vanish and no_state at H=96 regresses, so keep the
    //               plain (N,H) launch (fall through below).
    // The split wins for both state modes, so the old has_state_in/out
    // requirement is dropped. Correctness is guarded by the err_ratio<1e-2
    // consistency gate. PLA_FLASHKDA_K2_DVSPLIT=0 is the immediate rollback;
    // =16/32 forces a specific tile for A/B or validation.
    const char* eds = getenv("PLA_FLASHKDA_K2_DVSPLIT");
    // Fixed short-work promotion.  Below 256 tokens the old H24/H12 path was
    // the 512-thread shared-state kernel, while H48 used a full-DV nows CTA.
    // Register-resident DV splitting reduces the complete call by 8--56% over
    // all 20 exact Kimi fixed rows (10 lengths x two state modes).  H96 has a
    // later crossover: BV32 wins for both modes from 160 tokens, and for the
    // state-in path from 48 tokens.  Cap that new H96 window at 4096 so the
    // older long-sequence negative controls remain untouched.
    bool short_fixed_dvsplit = N == 1 && !is_varlen
        && ((T_total < 256 && (H == 12 || H == 24 || H == 48))
            || (H == 96 && T_total <= 4096
                && (T_total >= 160 || (has_state_in && T_total >= 48))));
    bool auto_fixed_dvsplit = N == 1 && !is_varlen
        && ((T_total >= 256 && H <= 64) || short_fixed_dvsplit);
    // Varlen promotion: route the whole low-head varlen fan-in through the BV32
    // dvsplit, which multiplies K2's grid by 4 (e.g. 36/72 -> 144/288 CTAs) to
    // hide the serial recurrence.  HARDWARE A/B (PPU1.5 ZW-M890P dev2,
    // tests/bench_ragged_gate_ab.py): forcing dvsplit is monotonically >= the
    // plain (N,H) launch across N=2..128, avg_seg=64..4096, H in {6,12}
    // (1.00-1.81x, never a regression; output bit-identical, both state modes).
    // So the previous hardcoded B6/T8192 point and the avg_seg>=512 floor are
    // both dropped.  PLA_FLASHKDA_K2_DVSPLIT=0 is the instant rollback; =16/32
    // forces a specific tile.
    //
    // Widened to H<=24 (was H in {6,12}) per a second A/B (ab_dvsplit_gate.py):
    //   * replaying the 25 Kimi-K3 shapes that lost to FLA-Triton fixed 35/37
    //     losing points, 1.36-1.97x faster than the default path;
    //   * a grid over N=2..218 x T=4373..32768, equal and 90%-skewed splits,
    //     showed BV32 dvsplit monotonically >= default at H=24 (1.015-1.97x,
    //     no regression, both state modes).
    // BV32 -- not BV16 -- is the right tile here (see auto_bv below), even though
    // the older fixed/N==1 sweep above preferred BV16 for H<=24: on varlen H=24
    // BV32 beat BV16 on 46/50 replayed points (e.g. 6.21 vs 8.89 ms).
    // The old H48/H96 exclusion remains correct for small, heavily skewed
    // batches: multiplying a single dominant recurrence chain only repeats
    // workspace work.  It does not apply to TP1/TP2's large-batch long-sequence
    // bucket, where N>=16 supplies many short chains and the 248-register
    // full-DV CTA is occupancy-limited to one 4-warp block group.  Production
    // N={17,44,109,218} endpoints with an fp32 initial state show BV32
    // improving the complete call by 2.5--4.4% (both H48/H96, 15 samples,
    // CV<0.4%), while BV16 regresses 18--21%.  The complete 256-case/TP
    // production replay additionally found that no-initial-state calls stop
    // benefiting once N grows past 40: at N=109/110 they regress 0.4--1.1%,
    // because the unsplit grid already has enough independent chains.  State-in
    // calls retain a >=1.5% gain at every tested N.  Keep the structural gate
    // narrow so both that high-N no-state region and the earlier skewed-batch
    // negative controls continue to use full DV.
    bool large_batch_longseq_dvsplit = is_varlen && N >= 16
        && T_total >= 16384 && (H == 48 || H == 96)
        && (has_state_in || N <= 40);
    // Short high-head varlen promotion.  A same-process paired replay of every
    // exact Kimi T<=4096 row (50 samples, both state modes) finds three
    // monotone BV32 regions beyond the established H<=24 policy:
    //   H96 state-in, N>=3: min 1.021x, no regressions;
    //   H48 state-in, N>=2: min 1.004x, no regressions;
    //   H48 no-state, N<=3: min 1.041x, no regressions.
    // H96/N2 and the remaining H48 no-state batches contain repeat-work
    // regressions and deliberately retain full-DV.  The 4096 cap separates
    // this calibration from the independently measured long-sequence gate.
    bool short_high_head_varlen_dvsplit = is_varlen && T_total <= 4096
        && ((H == 96 && has_state_in && N >= 3)
            || (H == 48 && (has_state_in || N <= 3)));
    bool auto_varlen_dvsplit =
        is_varlen && (H <= 24 || large_batch_longseq_dvsplit
                      || short_high_head_varlen_dvsplit);
    if (is_tc01) {
        // ZW810E has 64 CUs and needs its own BV/DK/high-head policy.  Exact
        // Kimi replay (both state modes, alternating order) establishes:
        //   * fixed H12/H24: BV32 only below 256 tokens, then plain;
        //   * fixed H48: BV16 below 256 only without state; state-in keeps
        //     BV32, while no-state is faster on plain full-DV at longer lengths;
        //   * fixed H96: BV32 throughout the measured T<=4096 interval;
        //   * packed H12/H24: BV32/DK1 for every short N>=2 bucket;
        //   * explicit-cu N1: the same low-head <256 rule, H96 throughout, and
        //     H48 for state-in or T<256;
        //   * short H96/N2: BV32 only with an input state;
        //   * short H48/N2: BV32 in both modes, and H48/N11 only with state.
        // H48 N3/N4/N22, H96 N3+, and the large-batch long-sequence tc02 gate
        // are negative controls here.
        // Preserve the previously validated long low-head N2/N3 cases; the
        // wider fan-in promotion is intentionally capped at T<=4096.
        if (!is_varlen && N == 1 && T_total <= 4096) {
            auto_fixed_dvsplit =
                (H <= 24 && T_total < 256)
                || (H == 48 && (has_state_in || T_total < 256))
                || H == 96;
        }
        bool established_low_head_varlen =
            (H == 12 && N >= 2 && N <= 3) || (H == 24 && N == 2);
        bool short_low_head_varlen = T_total <= 4096 && H <= 24
            && ((N == 1 && T_total < 256) || N >= 2);
        bool short_high_head_tc01 = T_total <= 4096
            && ((H == 96 && (N == 1 || (has_state_in && N == 2)))
                || (H == 48
                    && ((N == 1 && (has_state_in || T_total < 256))
                        || N == 2 || (has_state_in && N == 11))));
        // Long packed Kimi chains need extra grid width on tc01.  A production
        // H={12,24,48,96} x N={1,2,3} sweep shows a sharp N<=2 crossover:
        // splitting improves the selected N1 rows by 1.13--1.85x and the
        // selected N2 rows by 1.11--1.17x, while N3 usually regresses because
        // it already exposes enough independent recurrence chains.
        bool long_low_concurrency_tc01 =
            // K1's affine H12/N1 gate starts at 4096.  Include that exact
            // boundary here so K2 consumes the affine P/O representation;
            // the plain recurrence expects raw INV/Mqk/beta instead.
            (T_total == 4096 && N == 1 && H == 12)
            || (T_total > 4096
                && ((N == 1
                     && (H == 12 || H == 24 || H == 48 || H == 96))
                    || (N == 2 && (H == 12 || H == 24 || H == 48))));
        // Medium-fan-in H12 remains recurrence-latency bound on tc01.  A full
        // production-group replay (217 cases, both TP/state modes) makes the
        // crossover structural: every N4 row wins with BV16/DK4 (min 1.035x,
        // aggregate 1.267x) and every N5 row wins with BV32/DK2 (min 1.044x,
        // aggregate 1.117x).  H24 has only two additional monotone short
        // intervals; its longer rows retain the existing path.
        bool medium_low_head_tc01 = H == 12 && (N == 4 || N == 5);
        bool short_h24_dk2_tc01 = H == 24
            && ((N == 4 && T_total <= 4096)
                || (N == 5 && T_total <= 8192));
        auto_varlen_dvsplit = is_varlen
            && (established_low_head_varlen || short_low_head_varlen
                || short_high_head_tc01 || long_low_concurrency_tc01
                || medium_low_head_tc01 || short_h24_dk2_tc01);
    }
    bool auto_dvsplit = auto_fixed_dvsplit || auto_varlen_dvsplit;
    bool has_dvsplit_override = eds && eds[0];
    // Explicit BV16/BV32 also enables the mathematically identical varlen
    // instantiation.  Keep varlen AUTO off until its target-shape performance
    // sweep is complete; this gives a strict, instant rollback for A/B tests.
    bool valid_dvsplit_override = has_dvsplit_override
        && (atoi(eds) == 16 || atoi(eds) == 32);
    bool use_dvsplit = has_dvsplit_override
        ? (valid_dvsplit_override && (is_varlen || N == 1))
        : auto_dvsplit;
    // The host's immutable workspace-layout decision is authoritative. It has
    // already validated the exact BV/DK schedule; force entry to the matching
    // consumer even if another thread mutates a diagnostic env afterwards.
    if (use_tc01_contiguous || use_affine_p) use_dvsplit = true;
    if (use_dvsplit) {
        // H12 with only two or three sequences has at most 36 independent
        // recurrence chains before DV splitting.  BV16 doubles the number of
        // slices and is 4--10% faster on the near-balanced Kimi tails where CP
        // deliberately stays off.  Wider fan-in keeps BV32 to avoid needless
        // over-subscription; dominant N2/N3 tails are intercepted by CP first.
        // The final fixed tiny sweep (five exact production lengths, both
        // state modes, 40 samples, BV16/BV32 x DK1/2/4/8) resolves the small
        // granularity crossovers more finely than the original <128 rule:
        //   H12: BV16 except no-state [32,64), where BV32 avoids tiny slices;
        //   H24: BV16 below 32 and from 128 onward, BV32 in [32,128).
        // These are work-size intervals, not case-id exceptions.  Varlen keeps
        // its independently measured choice.
        int auto_fixed_bv = H == 12
            ? ((has_state_in || T_total < 32 || T_total >= 64) ? 16 : 32)
            : (H == 24
                ? ((T_total < 32 || T_total >= 128) ? 16 : 32)
                : (H <= 24 ? 16 : 32));
        int auto_bv = auto_varlen_dvsplit
            ? ((H == 12 && N <= 3) ? 16 : 32)
            : auto_fixed_bv;
        // On tc02, a single H6/H8 sequence exposes only 24/32 CTAs with
        // BV32. BV16 doubles that grid without changing the DK reduction or
        // state precision. Keep short sequences and wider fan-in on their
        // established schedules; CP dispatch is resolved before this path.
        if (!is_tc01 && is_varlen && N == 1
            && (H == 6 || H == 8) && T_total >= 1024) {
            auto_bv = 16;
        }
        // tc01 generally uses BV32 for promoted varlen work. The measured
        // exceptions are H12/N1 below 4096 (BV16/DK2), H12/N4 (BV16/DK4), and
        // long H12/N1 affine work (BV16/DK4/leader). Fixed H48/no-state also
        // uses BV16 below 256; fixed H48/state retains BV32 because BV16 was
        // 5.5% slower at T={121,169,242}.
        if (is_tc01) {
            if (is_varlen && H == 12 && N == 4) {
                auto_bv = 16;
            } else if (is_varlen && H == 12 && N == 5) {
                auto_bv = 32;
            } else if (is_varlen && N == 1
                       && (T_total > 4096
                           || (T_total == 4096 && H == 12))) {
                auto_bv = H == 24 ? 32 : 16;
            } else if (is_varlen && T_total > 4096 && N == 2) {
                auto_bv = H == 24 ? 16 : 32;
            } else {
                auto_bv = (!is_varlen && H == 48 && !has_state_in
                           && T_total < 256) ? 16 : 32;
            }
        }
        int requested_bv = has_dvsplit_override ? atoi(eds) : auto_bv;
        bool use_bv16 = requested_bv == 16;
        const char* edk = getenv("PLA_FLASHKDA_K2_DKSPLIT");
        bool auto_bv16_dk4lead = N == 1 && use_bv16 && T_total >= 4096
            && ((!is_varlen && (H == 6 || H == 12))
                || (is_tc01 && is_varlen && H == 12));
        bool auto_varlen_bv16_dk4 = is_varlen && H == 12 && N <= 3
            && use_bv16;
        // Short fixed recurrence is latency-bound.  Four DK workers shorten
        // the chain by another 8--20% end-to-end at H12 and 2--6% at H24.  At
        // H24/BV32, DK2 is the measured winner only in [64,128); below that the
        // extra warp is launch/sync overhead.  The dedicated-leader/affine form
        // remains reserved for the established T>=4096 H6/H12 path; these
        // short cases use regular worker groups.
        bool auto_fixed_short_dk4 = !is_varlen && N == 1
            && ((H == 12 && T_total < 4096 && use_bv16)
                || (H == 24 && T_total <= 4096
                    && (T_total < 32 || T_total >= 128)));
        bool auto_fixed_short_dk2 = !is_varlen && N == 1
            && H == 24 && T_total >= 64 && T_total < 128;
        int requested_dk_split = edk && edk[0]
            ? atoi(edk)
            : ((auto_bv16_dk4lead || auto_varlen_bv16_dk4
                || auto_fixed_short_dk4) ? 4
                : (auto_fixed_short_dk2 ? 2 : 1));
        if (is_tc01 && !(edk && edk[0])) {
            const bool long_affine_dk4 = is_varlen && N == 1
                && H == 12 && T_total >= 4096 && use_bv16;
            const bool medium_h12_dk4 = is_varlen && H == 12
                && N == 4 && use_bv16;
            const bool long_dk2 = is_varlen && T_total > 4096
                && ((N == 1 && H == 24)
                    || (N == 2 && (H == 12 || H == 48)));
            const bool medium_dk2 = is_varlen
                && ((H == 12 && N == 5)
                    || (H == 24
                        && ((N == 4 && T_total <= 4096)
                            || (N == 5 && T_total <= 8192))));
            requested_dk_split = (long_affine_dk4 || medium_h12_dk4)
                ? 4 : ((long_dk2 || medium_dk2) ? 2 : 1);
        }
        if (requested_dk_split != 2 && requested_dk_split != 4
            && requested_dk_split != 8)
            requested_dk_split = 1;
        const char* edkl = getenv("PLA_FLASHKDA_K2_DKLEADER");
        bool requested_dedicated_leader = use_bv16 && requested_dk_split == 4
            && (edkl && edkl[0] ? atoi(edkl) != 0 : auto_bv16_dk4lead);
        if (force_normal_raw) requested_dedicated_leader = false;
        if (use_affine_p) requested_dedicated_leader = true;
        auto dispatch_dvsplit_impl = [&](
                auto bv_tag, auto nt_tag, auto vl_tag,
                auto dk_split_tag, auto leader_tag,
                auto kr_transpose_tag, auto vec_output_tag) {
            constexpr int BV = decltype(bv_tag)::value;
            constexpr int NT = decltype(nt_tag)::value;
            constexpr int DKSW = decltype(dk_split_tag)::value;
            constexpr bool DEDICATED = decltype(leader_tag)::value;

            using SDV = fkn::K2NSmem<NT, BV, DKSW, DEDICATED>;
            constexpr int BLOCK_THREADS =
                fkn::K2NLaunchConfig<NT>::NThreads;
            dim3 split_grid(N, H, fkn::DV / BV);

            auto run_dvsplit = [&](auto si, auto so, auto compact_state) {
                constexpr bool COMPACT = decltype(compact_state)::value;
                auto kern = fkn::flashkda_recurrence_nows_kernel<
                    NT, decltype(vl_tag)::value,
                    decltype(si)::value, decltype(so)::value, COMPACT,
                    BV, DKSW, DEDICATED, false, false,
                    decltype(kr_transpose_tag)::value,
                    decltype(vec_output_tag)::value>;
                auto state_in = fkn::make_k2n_state_in_accessor<COMPACT>(
                    init_state, init_state_indices, init_state_is_fp32);
                auto state_out = fkn::make_k2n_state_out_accessor<COMPACT>(
                    final_state, final_state_indices, final_state_is_fp32);
                hggcFuncSetAttribute(
                    kern, hggcFuncAttributeMaxDynamicSharedMemorySize,
                    SDV::SMEM_BYTES);
                kern<<<split_grid, BLOCK_THREADS, SDV::SMEM_BYTES, stream>>>(
                    reinterpret_cast<BF16*>(o),
                    reinterpret_cast<const BF16*>(v),
                    reinterpret_cast<const BF16*>(beta),
                    reinterpret_cast<const BF16*>(ws),
                    state_in, state_out,
                    T_total, H, N, total_tiles, cu_seqlens, tile_prefix,
                    /*seg_tbl=*/nullptr);
            };
            auto pick_state = [&](auto si, auto so) {
                if (compact_bf16_state)
                    run_dvsplit(si, so, std::true_type{});
                else
                    run_dvsplit(si, so, std::false_type{});
            };
            if (has_state_in) {
                if (has_state_out)
                    pick_state(std::true_type{}, std::true_type{});
                else
                    pick_state(std::true_type{}, std::false_type{});
            } else {
                if (has_state_out)
                    pick_state(std::false_type{}, std::true_type{});
                else
                    pick_state(std::false_type{}, std::false_type{});
            }
        };
        auto dispatch_dvsplit = [&](
                auto bv_tag, auto nt_tag, auto vl_tag,
                auto dk_split_tag, auto leader_tag,
                auto kr_transpose_tag) {
            dispatch_dvsplit_impl(
                bv_tag, nt_tag, vl_tag, dk_split_tag, leader_tag,
                kr_transpose_tag, std::false_type{});
        };
        auto pick_dvsplit_varlen = [&](auto bv_tag, auto nt_tag,
                                       auto dk_split_tag, auto leader_tag) {
            if (is_varlen)
                dispatch_dvsplit(
                    bv_tag, nt_tag, std::true_type{}, dk_split_tag, leader_tag,
                    std::false_type{});
            else
                dispatch_dvsplit(
                    bv_tag, nt_tag, std::false_type{}, dk_split_tag, leader_tag,
                    std::false_type{});
        };
        if (use_affine_p) {
            if (is_varlen) {
                if (use_tc01_contiguous)
                    dispatch_dvsplit(
                        std::integral_constant<int, 16>{},
                        std::integral_constant<int, 160>{}, std::true_type{},
                        std::integral_constant<int, 4>{}, std::true_type{},
                        std::true_type{});
                else
                    dispatch_dvsplit(
                        std::integral_constant<int, 16>{},
                        std::integral_constant<int, 160>{}, std::true_type{},
                        std::integral_constant<int, 4>{}, std::true_type{},
                        std::false_type{});
            } else {
                dispatch_dvsplit(
                    std::integral_constant<int, 16>{},
                    std::integral_constant<int, 160>{}, std::false_type{},
                    std::integral_constant<int, 4>{}, std::true_type{},
                    std::false_type{});
            }
            return;
        }
        if (use_tc01_contiguous) {
            const bool full_dv_contiguous =
                (H == 48 && N == 3 && T_total >= 8192)
                || (H == 24 && N == 3 && T_total > 8192)
                || (H == 12 && N == 9 && T_total > 8192);
            if (full_dv_contiguous) {
                dispatch_dvsplit(
                    std::integral_constant<int, 128>{},
                    std::integral_constant<int, 128>{}, std::true_type{},
                    std::integral_constant<int, 1>{}, std::false_type{},
                    std::true_type{});
            } else if (H == 96) {
                if (N == 1)
                    dispatch_dvsplit(
                        std::integral_constant<int, 16>{},
                        std::integral_constant<int, 32>{}, std::true_type{},
                        std::integral_constant<int, 1>{}, std::false_type{},
                        std::true_type{});
                else {
                    // Promoted tc01 epilogue: 302/302 H96 contiguous-raw rows
                    // were bit-exact and improved the paired aggregate 1.03476x.
                    // Unset/nonzero selects vector stores; zero is the instant
                    // rollback. Only this BV128/NT128/DK1 specialization owns
                    // the 512-byte warp scratch required by the true tag.
                    const char* evec =
                        getenv("PLA_FLASHKDA_TC01_VEC_OUTPUT");
                    const bool vec_output = is_tc01
                        && kr_layout
                            == flashkda::KrWorkspaceLayout::kContiguousRawKrT
                        && !(evec && evec[0] && atoi(evec) == 0);
                    if (vec_output)
                        dispatch_dvsplit_impl(
                            std::integral_constant<int, 128>{},
                            std::integral_constant<int, 128>{},
                            std::true_type{},
                            std::integral_constant<int, 1>{},
                            std::false_type{}, std::true_type{},
                            std::true_type{});
                    else
                        dispatch_dvsplit(
                            std::integral_constant<int, 128>{},
                            std::integral_constant<int, 128>{},
                            std::true_type{},
                            std::integral_constant<int, 1>{},
                            std::false_type{}, std::true_type{});
                }
            } else if (N == 1 && H == 12 && T_total < 4096) {
                dispatch_dvsplit(
                    std::integral_constant<int, 16>{},
                    std::integral_constant<int, 64>{}, std::true_type{},
                    std::integral_constant<int, 2>{}, std::false_type{},
                    std::true_type{});
            } else if ((N == 2 && H == 24) || (N == 1 && H == 48)) {
                dispatch_dvsplit(
                    std::integral_constant<int, 16>{},
                    std::integral_constant<int, 32>{}, std::true_type{},
                    std::integral_constant<int, 1>{}, std::false_type{},
                    std::true_type{});
            } else if (N == 3 && H == 12) {
                dispatch_dvsplit(
                    std::integral_constant<int, 32>{},
                    std::integral_constant<int, 64>{}, std::true_type{},
                    std::integral_constant<int, 1>{}, std::false_type{},
                    std::true_type{});
            } else {
                dispatch_dvsplit(
                    std::integral_constant<int, 32>{},
                    std::integral_constant<int, 128>{}, std::true_type{},
                    std::integral_constant<int, 2>{}, std::false_type{},
                    std::true_type{});
            }
            return;
        }
        if (use_bv16 && requested_dk_split == 8)
            pick_dvsplit_varlen(
                std::integral_constant<int, 16>{},
                std::integral_constant<int, 256>{},
                std::integral_constant<int, 8>{}, std::false_type{});
        else if (use_bv16 && requested_dedicated_leader)
            pick_dvsplit_varlen(
                std::integral_constant<int, 16>{},
                std::integral_constant<int, 160>{},
                std::integral_constant<int, 4>{}, std::true_type{});
        else if (use_bv16 && requested_dk_split == 4)
            pick_dvsplit_varlen(
                std::integral_constant<int, 16>{},
                std::integral_constant<int, 128>{},
                std::integral_constant<int, 4>{}, std::false_type{});
        else if (use_bv16 && requested_dk_split == 2)
            pick_dvsplit_varlen(
                std::integral_constant<int, 16>{},
                std::integral_constant<int, 64>{},
                std::integral_constant<int, 2>{}, std::false_type{});
        else if (use_bv16)
            pick_dvsplit_varlen(
                std::integral_constant<int, 16>{},
                std::integral_constant<int, 32>{},
                std::integral_constant<int, 1>{}, std::false_type{});
        else if (requested_dk_split == 4)
            pick_dvsplit_varlen(
                std::integral_constant<int, 32>{},
                std::integral_constant<int, 256>{},
                std::integral_constant<int, 4>{}, std::false_type{});
        else if (requested_dk_split == 2)
            pick_dvsplit_varlen(
                std::integral_constant<int, 32>{},
                std::integral_constant<int, 128>{},
                std::integral_constant<int, 2>{}, std::false_type{});
        else
            pick_dvsplit_varlen(
                std::integral_constant<int, 32>{},
                std::integral_constant<int, 64>{},
                std::integral_constant<int, 1>{}, std::false_type{});
        return;
    }

    // Plain register-resident recurrence fall-through: grid (N,H), one CTA per
    // (seq,head).  Reached by full-DV single-sequence launches (H>64) and all
    // varlen (B>1) launches.  The 256-thread template gives each warp one
    // 16-column block and halves per-warp work when the launch fits one
    // resident wave (N==1, T>=256, H<=78); otherwise the 128-thread template.
    bool use_256 = N == 1 && T_total >= 256 && H <= 78;

    auto run = [&](auto nt, auto vl, auto si, auto so, auto compact_state,
                   auto vec_output_tag) {
        constexpr int NT = decltype(nt)::value;
        constexpr bool COMPACT = decltype(compact_state)::value;
        int smem = fkn::K2NSmem<NT>::SMEM_BYTES;
        constexpr int BLOCK_THREADS = fkn::K2NLaunchConfig<NT>::NThreads;
        auto kern = fkn::flashkda_recurrence_nows_kernel<
            NT, decltype(vl)::value,
            decltype(si)::value, decltype(so)::value, COMPACT,
            fkn::DV, 1, false, false, false, false,
            decltype(vec_output_tag)::value>;
        auto state_in = fkn::make_k2n_state_in_accessor<COMPACT>(
            init_state, init_state_indices, init_state_is_fp32);
        auto state_out = fkn::make_k2n_state_out_accessor<COMPACT>(
            final_state, final_state_indices, final_state_is_fp32);
        hggcFuncSetAttribute(kern, hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
        kern<<<grid, BLOCK_THREADS, smem, stream>>>(
            reinterpret_cast<BF16*>(o), reinterpret_cast<const BF16*>(v),
            reinterpret_cast<const BF16*>(beta), reinterpret_cast<const BF16*>(ws),
            state_in, state_out,
            T_total, H, N, total_tiles, cu_seqlens, tile_prefix,
            /*seg_tbl=*/nullptr);
    };
    auto pick_state = [&](auto nt, auto vl, auto si, auto so,
                          auto vec_output_tag) {
        if (compact_bf16_state)
            run(nt, vl, si, so, std::true_type{}, vec_output_tag);
        else
            run(nt, vl, si, so, std::false_type{}, vec_output_tag);
    };
    auto pick_so = [&](auto nt, auto vl, auto si, auto vec_output_tag) {
        if (has_state_out)
            pick_state(nt, vl, si, std::true_type{}, vec_output_tag);
        else
            pick_state(nt, vl, si, std::false_type{}, vec_output_tag);
    };
    auto pick_si = [&](auto nt, auto vl, auto vec_output_tag) {
        if (has_state_in)
            pick_so(nt, vl, std::true_type{}, vec_output_tag);
        else
            pick_so(nt, vl, std::false_type{}, vec_output_tag);
    };
    auto pick_vl = [&](auto nt, auto vec_output_tag) {
        if (is_varlen)
            pick_si(nt, std::true_type{}, vec_output_tag);
        else
            pick_si(nt, std::false_type{}, vec_output_tag);
    };
    // Promoted tc01 epilogue for normal-raw long-batch paths.  Complete paired
    // sweeps were bit-exact and improved the aggregate by 1.01938x (H96),
    // 1.02487x (H48), and 1.03239x (H24).  Repeat-50 rechecks bounded the
    // remaining low-tail noise below 1%, so all three heads share one rollback
    // knob with the contiguous H96 N2/N3 specialization. H12 stayed effectively
    // flat and retains the scalar-store epilogue.
    const char* evec = getenv("PLA_FLASHKDA_TC01_VEC_OUTPUT");
    const bool broad_vec_output = is_tc01 && is_varlen
        && N >= 4 && N <= 16 && T_total >= 8192
        && (H == 24 || H == 48 || H == 96)
        && !(evec && evec[0] && atoi(evec) == 0);
    if (use_256)
        pick_vl(std::integral_constant<int, 256>{}, std::false_type{});
    else if (broad_vec_output)
        pick_si(std::integral_constant<int, 128>{}, std::true_type{},
                std::true_type{});
    else
        pick_vl(std::integral_constant<int, 128>{}, std::false_type{});
}

// ===================== CP (context-parallel) launchers =====================
// Both passes are template modes of the very same register-resident recurrence
// kernel above, so the DV split, the {ws,v,beta} double-buffer prefetch and the
// register-resident mainloop are inherited unchanged; only the segment indexing,
// the carry initialisation and the epilogue differ.
//
//   pass 1  pre_scan : grid (n_seg, H, 2*ZSPLIT).  grid.z's lower half runs the
//                      bias chain B_i (carry 0, real v), the upper half the
//                      transition chain M_i (carry I, v:=0).  No output.
//   pass 2  merge    : kernel3_cp.cu, serial over a sequence's segments.
//   pass 3  segment  : grid (n_seg, H, ZSPLIT), init_state = merged carries,
//                      emits O for every token and final_state for the segments
//                      the descriptor marks as a sequence's last.
//
// The bv selector picks the value tile (16/32/128 -> 32/64/128 threads) so the
// crossover sweep can A/B the tile without a rebuild.  DK-split and the affine
// dedicated leader stay off: the pre_scan static_asserts forbid the latter and
// CP already has all the parallelism the DK split was invented to buy.
void launch_flashkda_cp_prescan(
    const void* v, const void* beta, const void* ws,
    void* chain_state, bool fp16_state,
    int T_total, int H, int n_seg, int total_tiles,
    const int* seg_tbl, int bv, hggcStream_t stream) {
    using BF16 = cutlass::bfloat16_t;
    namespace fkn = flashkda_cute_nows;
    auto go = [&](auto nt_tag, auto bv_tag, auto fp16_tag) {
        constexpr int NT = decltype(nt_tag)::value;
        constexpr int BV = decltype(bv_tag)::value;
        constexpr bool FP16_STATE = decltype(fp16_tag)::value;
        using SDV = fkn::K2NSmem<NT, BV>;
        auto kern = fkn::flashkda_recurrence_nows_kernel<
            NT, /*IsVarlen=*/false, /*HasStateIn=*/false, /*HasStateOut=*/true,
            /*CompactBf16State=*/true,
            BV, /*DKSplitWarps=*/1, /*DedicatedLeader=*/false,
            /*UseSegTable=*/true, /*PreScan=*/true,
            /*Tc01KrTranspose=*/false, /*Tc01VectorOutput=*/false, FP16_STATE>;
        auto state_in = fkn::make_k2n_state_in_accessor<true>(
            nullptr, nullptr, false);
        auto state_out = fkn::make_k2n_state_out_accessor<true>(
            chain_state, nullptr, false);
        hggcFuncSetAttribute(kern, hggcFuncAttributeMaxDynamicSharedMemorySize,
                             SDV::SMEM_BYTES);
        dim3 grid(n_seg, H, 2 * (fkn::DV / BV));
        kern<<<grid, NT, SDV::SMEM_BYTES, stream>>>(
            /*o=*/nullptr,
            reinterpret_cast<const BF16*>(v),
            reinterpret_cast<const BF16*>(beta),
            reinterpret_cast<const BF16*>(ws),
            state_in, state_out,
            T_total, H, n_seg, total_tiles,
            /*cu_seqlens=*/nullptr, /*tile_prefix=*/nullptr, seg_tbl);
    };
    auto pick = [&](auto nt_tag, auto bv_tag) {
        if (fp16_state) go(nt_tag, bv_tag, std::true_type{});
        else            go(nt_tag, bv_tag, std::false_type{});
    };
    if (bv == 16)
        pick(std::integral_constant<int, 32>{}, std::integral_constant<int, 16>{});
    else if (bv == 32)
        pick(std::integral_constant<int, 64>{}, std::integral_constant<int, 32>{});
    else
        pick(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
}

void launch_flashkda_cp_segment(
    void* o, const void* v, const void* beta, const void* ws,
    const void* carries, bool fp16_state, void* final_state,
    const int32_t* final_state_indices, bool final_state_is_fp32,
    int T_total, int H, int n_seg, int total_tiles,
    const int* seg_tbl, int bv, hggcStream_t stream) {
    using BF16 = cutlass::bfloat16_t;
    namespace fkn = flashkda_cute_nows;
    const bool compact_bf16_state = final_state == nullptr
        || (!final_state_is_fp32 && final_state_indices == nullptr);
    auto go = [&](auto nt_tag, auto bv_tag) {
        constexpr int NT = decltype(nt_tag)::value;
        constexpr int BV = decltype(bv_tag)::value;
        using SDV = fkn::K2NSmem<NT, BV>;
        // The merge always materialises a carry for every segment. Select the
        // no-state epilogue when the caller passed a null final-state pointer;
        // in that instantiation even a sequence's last segment skips the store.
        auto run = [&](auto store_final_tag, auto compact_state, auto fp16_tag) {
            constexpr bool COMPACT = decltype(compact_state)::value;
            constexpr bool FP16_STATE = decltype(fp16_tag)::value;
            auto kern = fkn::flashkda_recurrence_nows_kernel<
                NT, /*IsVarlen=*/false, /*HasStateIn=*/true,
                decltype(store_final_tag)::value,
                COMPACT, BV, /*DKSplitWarps=*/1, /*DedicatedLeader=*/false,
                /*UseSegTable=*/true, /*PreScan=*/false,
                /*Tc01KrTranspose=*/false, /*Tc01VectorOutput=*/false, FP16_STATE>;
            auto state_in = fkn::make_k2n_state_in_accessor<COMPACT>(
                carries, nullptr, false);
            auto state_out = fkn::make_k2n_state_out_accessor<COMPACT>(
                final_state, final_state_indices, final_state_is_fp32);
            hggcFuncSetAttribute(kern, hggcFuncAttributeMaxDynamicSharedMemorySize,
                                 SDV::SMEM_BYTES);
            dim3 grid(n_seg, H, fkn::DV / BV);
            kern<<<grid, NT, SDV::SMEM_BYTES, stream>>>(
                reinterpret_cast<BF16*>(o),
                reinterpret_cast<const BF16*>(v),
                reinterpret_cast<const BF16*>(beta),
                reinterpret_cast<const BF16*>(ws),
                state_in, state_out,
                T_total, H, n_seg, total_tiles,
                /*cu_seqlens=*/nullptr, /*tile_prefix=*/nullptr, seg_tbl);
        };
        auto pick_state = [&](auto store_final_tag) {
            if (fp16_state)
                run(store_final_tag, std::false_type{}, std::true_type{});
            else if (compact_bf16_state)
                run(store_final_tag, std::true_type{}, std::false_type{});
            else
                run(store_final_tag, std::false_type{}, std::false_type{});
        };
        if (final_state != nullptr) pick_state(std::true_type{});
        else                        pick_state(std::false_type{});
    };
    if (bv == 16)
        go(std::integral_constant<int, 32>{}, std::integral_constant<int, 16>{});
    else if (bv == 32)
        go(std::integral_constant<int, 64>{}, std::integral_constant<int, 32>{});
    else
        go(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
}
