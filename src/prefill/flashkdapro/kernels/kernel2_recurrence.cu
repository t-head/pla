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
 * FlashKDA PPU1.5 — Kernel 2 (recurrence, non-WS), correctness-first port.
 *
 * Blueprint: csrc/smxx/fwd_kernel2_nows.cuh + fwd_kernel2_common.cuh + torch_ref.
 * One CTA processes one (sequence, head), sequential over its chunks.
 * grid = (N, H), 256 threads. State St = state_slice^T is kept in smem [dk,dv].
 *
 * Per chunk (matches torch_ref lines 231-243):
 *   vsub = kd @ St                      [16,dv]   (St[dk,dv])
 *   o1   = qd @ St                      [16,dv]
 *   vbeta= (v - vsub) * beta[row]       [16,dv]
 *   U    = INV @ vbeta                  [16,dv]
 *   out  = o1 + Mqk @ U                 [16,dv]   -> global o
 *   delta_s = kr^T @ U                  [dk,dv]
 *   St   = St * ex2(g_total)[dk] + delta_s
 * All GEMMs via gemm_tn_bf16_smem (C=A@B^T). Plain global loads (AIU later, t8).
 */
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <cstdlib>
#include <cutlass/bfloat16.h>
#include "prefill/flashkdapro/common/launch_api.h"
#include "prefill/flashkdapro/common/runtime_arch.h"
#include "flashkda_ppu_util.cuh"

namespace flashkda_cute {

namespace fk = flashkda;
using BF16 = cutlass::bfloat16_t;

static constexpr int CH = 16;
static constexpr int DK = 128;
static constexpr int DV = 128;
static constexpr int K2_THREADS = 256;

// Keep the legacy, compact BF16 state ABI as a distinct kernel instantiation.
// The generic accessor retains indexed/fp32 state support, while the compact
// accessor carries only a typed pointer: no index pointer, dtype flag, or
// runtime branch reaches the generated device code for the legacy path.
template <bool CompactBf16>
struct K2StateInAccessor;

template <>
struct K2StateInAccessor<true> {
    const BF16* ptr;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return logical_slot;
    }
    __device__ __forceinline__ BF16 load(int64_t offset) const {
        return ptr[offset];
    }
    __device__ __forceinline__ float load_fp32(int64_t offset) const {
        return fk::bf16_to_f32(ptr[offset]);
    }
};

template <>
struct K2StateInAccessor<false> {
    const void* ptr;
    const int32_t* indices;
    bool is_fp32;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return indices ? indices[logical_slot] : logical_slot;
    }
    __device__ __forceinline__ BF16 load(int64_t offset) const {
        if (is_fp32)
            return BF16(reinterpret_cast<const float*>(ptr)[offset]);
        return reinterpret_cast<const BF16*>(ptr)[offset];
    }
    __device__ __forceinline__ float load_fp32(int64_t offset) const {
        if (is_fp32)
            return reinterpret_cast<const float*>(ptr)[offset];
        return fk::bf16_to_f32(reinterpret_cast<const BF16*>(ptr)[offset]);
    }
};

template <bool CompactBf16>
struct K2StateOutAccessor;

template <>
struct K2StateOutAccessor<true> {
    BF16* ptr;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return logical_slot;
    }
    __device__ __forceinline__ void store(int64_t offset, BF16 value) const {
        ptr[offset] = value;
    }
    __device__ __forceinline__ void store_fp32(int64_t offset, float value) const {
        ptr[offset] = BF16(value);
    }
};

template <>
struct K2StateOutAccessor<false> {
    void* ptr;
    const int32_t* indices;
    bool is_fp32;

    __device__ __forceinline__ int slot(int logical_slot) const {
        return indices ? indices[logical_slot] : logical_slot;
    }
    __device__ __forceinline__ void store(int64_t offset, BF16 value) const {
        if (is_fp32)
            reinterpret_cast<float*>(ptr)[offset] = fk::bf16_to_f32(value);
        else
            reinterpret_cast<BF16*>(ptr)[offset] = value;
    }
    __device__ __forceinline__ void store_fp32(int64_t offset, float value) const {
        if (is_fp32)
            reinterpret_cast<float*>(ptr)[offset] = value;
        else
            reinterpret_cast<BF16*>(ptr)[offset] = BF16(value);
    }
};

template <bool CompactBf16>
inline K2StateInAccessor<CompactBf16> make_k2_state_in_accessor(
    const void* ptr, const int32_t* indices, bool is_fp32) {
    if constexpr (CompactBf16)
        return {reinterpret_cast<const BF16*>(ptr)};
    else
        return {ptr, indices, is_fp32};
}

template <bool CompactBf16>
inline K2StateOutAccessor<CompactBf16> make_k2_state_out_accessor(
    void* ptr, const int32_t* indices, bool is_fp32) {
    if constexpr (CompactBf16)
        return {reinterpret_cast<BF16*>(ptr)};
    else
        return {ptr, indices, is_fp32};
}

__forceinline__ __device__ void k2_aiu_load_2d_b16(
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
    // See k1_aiu_load_2d_b16 for the axis-remap rationale (tc01 has no
    // explicit stride operand; stride-in-elements is smuggled into dim_c).
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

__forceinline__ __device__ void k2_aiu_commit() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
__forceinline__ __device__ void k2_aiu_wait_group() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// device transpose [R,C]->[C,R] (bf16), cooperative over nthreads.
template <int R, int C>
__device__ __forceinline__ void transpose_bf16(const BF16* src, BF16* dst,
                                                int tid, int nthreads) {
    for (int idx = tid; idx < R * C; idx += nthreads) {
        int r = idx / C, c = idx % C;
        dst[c * R + r] = src[idx];
    }
}

// dynamic smem byte offsets (128-byte aligned)
static constexpr int A128(int x) { return (x + 127) & ~127; }

template <int DVT>
struct K2Smem {
    static constexpr int NBUF = 2;
    // Merged single-slot workspace block per stage (kd|qd|kr_swzl|inv|mqk|gt).
    static constexpr int B_WS    = A128(fk::WS_SLOT_BYTES);   // 14080
    static constexpr int B_V     = A128(CH * DVT * 2);
    static constexpr int B_BETA  = A128(CH * 4);

    // byte offsets of each field within one merged ws stage block
    static constexpr int WOFF_KD  = fk::WS_OFF_KD_U  * 2;   // 0
    static constexpr int WOFF_QD  = fk::WS_OFF_QD_U  * 2;   // 4096
    static constexpr int WOFF_KR  = fk::WS_OFF_KR_U  * 2;   // 8192
    static constexpr int WOFF_INV = fk::WS_OFF_INV_U * 2;   // 12288
    static constexpr int WOFF_MQK = fk::WS_OFF_MQK_U * 2;   // 12800
    static constexpr int WOFF_GT  = fk::WS_OFF_GT_U  * 2;   // 13312 (f32 [DK])
    static constexpr int WOFF_BETA = fk::WS_OFF_BETA_U * 2;

    static constexpr int O_St     = 0;
    static constexpr int O_delta  = O_St   + A128(DK * DVT * 2);
    static constexpr int O_end0   = O_delta+ A128(DK * DVT * 4);  // f32 delta (matches original precision)
    static constexpr int O_ws     = O_end0;                      // merged block
    static constexpr int O_v      = O_ws   + NBUF * B_WS;
    static constexpr int O_beta   = O_v    + NBUF * B_V;         // f32 [CH]
    static constexpr int O_vsub   = O_beta + NBUF * B_BETA;
    static constexpr int O_o1     = O_vsub + A128(CH * DVT * 2);
    static constexpr int O_vbeta  = O_o1   + A128(CH * DVT * 2);
    static constexpr int SMEM_BYTES = O_vbeta + A128(CH * DVT * 2);
};

template <int DVT, int NTHREADS>
__device__ __forceinline__ void k2_preload_stage(
    int stage, int actual, int ws_idx, int64_t tok0,
    int H, int T_total, int head, int dv0, int tid,
    const BF16* __restrict__ v, const BF16* __restrict__ beta,
    const BF16* __restrict__ ws,
    BF16* sWsBase, BF16* sVBase, float* sBetaBase) {
    using S = K2Smem<DVT>;
    BF16* sWs = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sWsBase) + stage * S::B_WS);
    BF16* sV = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sVBase) + stage * S::B_V);
    float* sBeta = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(sBetaBase) + stage * S::B_BETA);

    // One contiguous slot (kd|qd|kr_swzl|inv|mqk|gt) = fk::WS_SLOT_UNITS bf16.
    const BF16* g_slot = ws + int64_t(ws_idx) * fk::WS_SLOT_UNITS;

    if (tid == 0) {
        // Single AIU linear bulk load of the whole 14080B slot (dim_c=128,
        // dim_w=55). kr is pre-swizzled by Kernel1 so the byte image matches
        // exactly what the old k2_aiu_swzl_write produced -> consumers unchanged.
        k2_aiu_load_2d_b16(sWs, g_slot, DK, fk::WS_SLOT_UNITS / DK,
                           int64_t(DK) * sizeof(BF16));
        if (actual == CH) {
            const BF16* v_src = v + tok0 * int64_t(H) * DV + int64_t(head) * DV + dv0;
            k2_aiu_load_2d_b16(sV, v_src, DVT, CH, int64_t(H) * DV * sizeof(BF16));
        }
        k2_aiu_commit();
    }
    if (actual != CH) {
        for (int idx = tid; idx < CH * DVT; idx += NTHREADS) {
            int r = idx / DVT, c = idx % DVT;
            sV[idx] = (r < actual)
                ? v[(tok0 + r) * int64_t(H) * DV + int64_t(head) * DV + (dv0 + c)] : BF16(0);
        }
    }
    (void)beta;
    (void)sBeta;
}

template <int DVT, int NTHREADS, bool IsVarlen, bool HasStateIn,
          bool HasStateOut, bool CompactBf16State>
__global__ void __launch_bounds__(NTHREADS) flashkda_recurrence_kernel(
    BF16* __restrict__ o,             // [T_total, H, DV]
    const BF16* __restrict__ v,       // [T_total, H, DV]
    const BF16* __restrict__ beta,    // [H*T_total]
    const BF16* __restrict__ ws,      // merged workspace [nht, WS_SLOT_UNITS]
    K2StateInAccessor<CompactBf16State> init_state,
    K2StateOutAccessor<CompactBf16State> final_state,
    int T_total, int H, int N, int total_tiles,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix)
{
    using S = K2Smem<DVT>;
    static_assert(DV % DVT == 0, "DVT must evenly divide DV");
    extern __shared__ __align__(128) unsigned char smem[];
    BF16* sSt   = reinterpret_cast<BF16*>(smem + S::O_St);
    float* sDelta= reinterpret_cast<float*>(smem + S::O_delta);
    BF16* sWsBase   = reinterpret_cast<BF16*>(smem + S::O_ws);
    BF16* sVBase    = reinterpret_cast<BF16*>(smem + S::O_v);
    float* sBetaBase= reinterpret_cast<float*>(smem + S::O_beta);
    BF16* sVsub = reinterpret_cast<BF16*>(smem + S::O_vsub);
    BF16* sO1   = reinterpret_cast<BF16*>(smem + S::O_o1);
    BF16* sVbeta= reinterpret_cast<BF16*>(smem + S::O_vbeta);

    const int seq = blockIdx.x, head = blockIdx.y, dv_blk = blockIdx.z, tid = threadIdx.x;
    const int dv0 = dv_blk * DVT;
#if defined(FLASHKDA_ARCH_150)
    static_assert((DK * DVT) % NTHREADS == 0, "State ownership must be uniform");
    constexpr int MASTER_VALUES = DK * DVT / NTHREADS;
    float rmaster[MASTER_VALUES];
#endif

    int64_t bos, eos; int tile_base;
    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq]; eos = cu_seqlens[seq + 1];
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
    }
    int seq_len = int(eos - bos);
    int t_tiles = (seq_len + CH - 1) / CH;

    // --- init state St[dk,dv_local] = init_state[seq,h,dv_global,dk]^T (or zero) ---
    if constexpr (HasStateIn) {
        const int state_slot = init_state.slot(seq);
        const int64_t state_off = (int64_t(state_slot) * H + head) * DV * DK;
#if defined(FLASHKDA_ARCH_150)
        CUTE_UNROLL
        for (int ordinal = 0; ordinal < MASTER_VALUES; ++ordinal) {
            const int idx = tid + ordinal * NTHREADS;
            int i = idx / DVT, c = idx % DVT;
            const float value = init_state.load_fp32(
                state_off + int64_t(dv0 + c) * DK + i);
            rmaster[ordinal] = value;
            sSt[idx] = BF16(value);
        }
#else
        for (int idx = tid; idx < DK * DVT; idx += NTHREADS) {
            int i = idx / DVT, c = idx % DVT;
            sSt[idx] = init_state.load(
                state_off + int64_t(dv0 + c) * DK + i);
        }
#endif
    } else {
#if defined(FLASHKDA_ARCH_150)
        CUTE_UNROLL
        for (int ordinal = 0; ordinal < MASTER_VALUES; ++ordinal) {
            sSt[tid + ordinal * NTHREADS] = BF16(0);
            rmaster[ordinal] = 0.f;
        }
#else
        for (int idx = tid; idx < DK * DVT; idx += NTHREADS) {
            sSt[idx] = BF16(0);
        }
#endif
    }
    __syncthreads();

    // Prologue: seed the depth-2 double-buffered AIU pipeline by issuing loads
    // for the first two chunks, each committed as its own group (2 groups in
    // flight). Mirrors the FlashQLA PPU1.5 prologue structure.
    for (int pf = 0; pf < 2 && pf < t_tiles; ++pf) {
        int pf_actual = min(CH, seq_len - pf * CH);
        int pf_ws_idx = head * total_tiles + tile_base + pf;
        int64_t pf_tok0 = bos + int64_t(pf) * CH;
        k2_preload_stage<DVT, NTHREADS>(pf & 1, pf_actual, pf_ws_idx, pf_tok0,
                              H, T_total, head, dv0, tid,
                              v, beta, ws,
                              sWsBase, sVBase, sBetaBase);
    }

    for (int t = 0; t < t_tiles; ++t) {
        int actual = min(CH, seq_len - t * CH);
        int stage = t & 1;
        int64_t tok0 = bos + int64_t(t) * CH;

        // Depth-2 pipeline head: keep the next chunk's load in flight. Drain to
        // at most 1 pending group (the prefetched chunk t+1) except on the last
        // iteration where nothing is prefetched, so drain fully.
        if (tid == 0) {
            if (t < t_tiles - 1) k2_aiu_wait_group<1>();
            else                 k2_aiu_wait_group<0>();
        }
        __syncthreads();

        BF16* sWs = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sWsBase) + stage * S::B_WS);
        BF16* sKd = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sWs) + S::WOFF_KD);
        BF16* sQd = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sWs) + S::WOFF_QD);
        BF16* sKr = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sWs) + S::WOFF_KR);
        BF16* sInv = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sWs) + S::WOFF_INV);
        BF16* sMqk = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sWs) + S::WOFF_MQK);
        float* sGt = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(sWs) + S::WOFF_GT);
        BF16* sV = reinterpret_cast<BF16*>(reinterpret_cast<unsigned char*>(sVBase) + stage * S::B_V);
        float* sBeta = reinterpret_cast<float*>(
            reinterpret_cast<unsigned char*>(sWs) + S::WOFF_BETA);

        // vsub[16,dv_local] = kd @ St; o1[16,dv_local] = qd @ St
        // St is kept as [dk,dv_local]; the helper uses a logical transposed B view.
        fk::gemm_nn2_bf16_smem<CH, DVT, DK, BF16>(sKd, sQd, sSt, sVsub, sO1, tid, NTHREADS);
        __syncthreads();

        // Producer warps generate vbeta B-fragments through ldmatrix.trans scratch,
        // then cache swizzled U registers in sVbeta scratch for consumer warps.
        fk::k2_fused_u_o_delta_cached_vbeta_movmatrix_bf16_smem<DK, DVT, CH>(
            sInv, sV, sVsub, sBeta, sMqk, sKr, sO1, sDelta, sVbeta, tid, NTHREADS);
        __syncthreads();

        // out[16,dv_local] = sO1  -> global (rows < actual)
        for (int idx = tid; idx < CH * DVT; idx += NTHREADS) {
            int r = idx / DVT, c = idx % DVT;
            if (r < actual) {
                o[(tok0 + r) * int64_t(H) * DV + int64_t(head) * DV + (dv0 + c)] = sO1[idx];
            }
        }

        // St[dk,dv_local] = St * ex2(g_total)[dk] + delta_s
        // (ws_gt already holds ex2(g_total) from Kernel1 -> use directly)
#if defined(FLASHKDA_ARCH_150)
        CUTE_UNROLL
        for (int ordinal = 0; ordinal < MASTER_VALUES; ++ordinal) {
            const int idx = tid + ordinal * NTHREADS;
            float g = sGt[idx / DVT];
            const float updated = rmaster[ordinal] * g + sDelta[idx];
            rmaster[ordinal] = updated;
            sSt[idx] = BF16(updated);
        }
#else
        constexpr int ROWS_PER_STEP = NTHREADS / DVT;
        int row = tid / DVT;
        int col = tid - row * DVT;
        for (int i = row; i < DK; i += ROWS_PER_STEP) {
            int idx = i * DVT + col;
            float g = sGt[i];
            sSt[idx] = BF16(fk::bf16_to_f32(sSt[idx]) * g + sDelta[idx]);
        }
#endif
        // Chunk fence: all reads of chunk t's double-buffered stage buffers are
        // now complete (last reader is sGt[stage] in the St-update loop above).
        // Required before recycling this slot for chunk t+2 (NBUF=2 => (t+2)&1 == t&1).
        __syncthreads();

        // Tail prefetch: issue chunk t+2 into the slot chunk t just vacated,
        // keeping one chunk in flight for the next iteration's wait_group<1>.
        int pf_t = t + 2;
        if (pf_t < t_tiles) {
            int pf_actual = min(CH, seq_len - pf_t * CH);
            int pf_ws_idx = head * total_tiles + tile_base + pf_t;
            int64_t pf_tok0 = bos + int64_t(pf_t) * CH;
            k2_preload_stage<DVT, NTHREADS>(pf_t & 1, pf_actual, pf_ws_idx, pf_tok0,
                                  H, T_total, head, dv0, tid,
                                  v, beta, ws,
                                  sWsBase, sVBase, sBetaBase);
        }
    }

    // --- store final state final[seq,h,dv_global,dk] = St^T ---
    if constexpr (HasStateOut) {
        const int state_slot = final_state.slot(seq);
        const int64_t state_off = (int64_t(state_slot) * H + head) * DV * DK;
#if defined(FLASHKDA_ARCH_150)
        CUTE_UNROLL
        for (int ordinal = 0; ordinal < MASTER_VALUES; ++ordinal) {
            const int idx = tid + ordinal * NTHREADS;
            int i = idx / DVT, c = idx % DVT;
            final_state.store_fp32(
                state_off + int64_t(dv0 + c) * DK + i, rmaster[ordinal]);
        }
#else
        for (int idx = tid; idx < DK * DVT; idx += NTHREADS) {
            int i = idx / DVT, c = idx % DVT;
            final_state.store(
                state_off + int64_t(dv0 + c) * DK + i, sSt[idx]);
        }
#endif
    }
}

}  // namespace flashkda_cute

void launch_flashkda_recurrence(
    void* o, const void* v, const void* beta,
    const void* ws,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices, const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream) {
    // K2 variant selection: register-resident "nows" fast path vs the baseline
    // column-split (shared-memory state) recurrence.
    //   env PLA_FLASHKDA_K2_NOWS = "0"     -> force baseline
    //                            = nonzero -> force nows
    //                            = unset   -> AUTO (below)
    // AUTO promotes a shape to nows when any of these hold:
    //   (a) single sequence, long enough    : N==1 && T_total>=256
    //   (b) high total concurrency           : N*H >= 40
    //   (c) low-head varlen (HW A/B verified): is_varlen && H <= 24
    //   (d) short fixed Kimi low-head shape  : N==1, T<256, H in {12,24}
    // Rationale: (a)/(b) fill a resident wave; (c) is the same concurrency-
    // starved win-region as single-seq, routed to nows (+inner DV-split for
    // occupancy) instead of the shared-state baseline.
    // HARDWARE A/B (PPU1.5 ZW-M890P dev2, tests/bench_ragged_gate_ab.py): forcing
    // nows+dvsplit over the baseline is monotonically >= baseline across
    // N=2..128, avg_seg=64..4096, H in {6,12} (1.00-1.81x, no regression; output
    // bit-identical to baseline, err vs FLA ~4.6e-3). So the whole low-head
    // varlen fan-in is promoted (no hardcoded shape, no avg_seg floor).
    // (c) was widened from H in {6,12} to H<=24 together with the inner dvsplit
    // gate (see kernel2_recurrence_nows.cu for the H=24 A/B).  H=24 already
    // reached nows via (b) for every N>=2 (N*H>=48); widening (c) keeps the two
    // gates coherent so that 12<H<=24 with N*H<40 cannot fall to the baseline and
    // miss the dvsplit it was measured to want.
    // (d) is the outer half of the short-work promotion in the nows launcher:
    // forcing nows+DV split improves every measured fixed length/state pair,
    // while the 512-thread shared-state kernel launches only 48--96 CTAs and
    // pays 58 KiB TSM plus CTA-wide synchronization per block.  Keep the exact
    // production head counts so unrelated low-head configurations retain the
    // established crossover.  PLA_FLASHKDA_K2_NOWS=0 rolls the whole selection
    // back to the baseline.
    {
        const bool is_tc01 = flashkda::host_runtime::current_device_is_tc01();
        const char* e = getenv("PLA_FLASHKDA_K2_NOWS");
        bool auto_nows = (N == 1 && T_total >= 256)              // (a)
                      || (int64_t(N) * int64_t(H) >= 40)         // (b)
                      || (is_varlen && H <= 24)                  // (c)
                      || (!is_varlen && N == 1 && T_total < 256  // (d)
                          && (H == 12 || H == 24))
                      || (!is_tc01 && is_varlen && N == 1 && H == 32
                          && T_total < 256);
        // The tc02 H32 short-varlen path uses the matching BV32 nows gate.
        // Its FP32 shared-state CTA crosses the 4-to-3 residency threshold;
        // value splitting keeps FP32 master state in the existing nows kernel.
        if (is_tc01) {
            // The broad low-head varlen promotion was calibrated on tc02.  On
            // tc01, OperatorsForge's explicit [0,T] ABI made H12/B1/T<256 enter
            // plain nows, which is 1.22--1.40x slower than the established shared-
            // state baseline.  Paired with the tc01 BV32 split, however, all 11
            // such rows are monotonic wins (max 0.733x).  Keep that pair together,
            // plus the complete-sweep monotonic H24/B1 and H12/N2--3 additions.
            // The short-work branch additionally promotes fixed H12/H24 rows;
            // keep that file-local policy on tc01 as well and validate it on
            // ZW810E. Every other shape retains the pre-merge long-sequence /
            // aggregate-concurrency rules above.
            auto_nows = (N == 1 && T_total >= 256)
                     || (int64_t(N) * int64_t(H) >= 40)
                     || (is_varlen && ((H == 24 && N == 1)
                                    || (H == 12
                                        && ((N == 1 && T_total < 256)
                                            || (N >= 2 && N <= 3)))))
                     || (!is_varlen && N == 1 && T_total < 256
                         && (H == 12 || H == 24));
        }
        bool use_nows = (e && e[0]) ? (e[0] != '0') : auto_nows;
        // A frozen affine or private transposed workspace always selects its
        // matching nows consumer, even if an env changes after host selection.
        const bool contiguous_layout =
            kr_layout == flashkda::KrWorkspaceLayout::kContiguousRawKrT
            || kr_layout == flashkda::KrWorkspaceLayout::kContiguousAffineP;
        const bool affine_layout =
            kr_layout == flashkda::KrWorkspaceLayout::kNormalAffineP
            || kr_layout == flashkda::KrWorkspaceLayout::kContiguousAffineP;
        if (use_nows || contiguous_layout || affine_layout) {
            launch_flashkda_recurrence_nows(
                o, v, beta, ws,
                init_state, final_state,
                init_state_indices, final_state_indices,
                init_state_is_fp32, final_state_is_fp32,
                T_total, H, N, total_tiles, cu_seqlens,
                is_varlen, has_state_in, has_state_out,
                kr_layout, stream);
            return;
        }
    }
    using BF16 = cutlass::bfloat16_t;
    namespace fkc = flashkda_cute;
    bool use_tile_prefix = is_varlen && (N >= 64);
    const int* tile_prefix = use_tile_prefix
        ? reinterpret_cast<const int*>(
              reinterpret_cast<const unsigned char*>(ws)
              + int64_t(H) * total_tiles * flashkda::WS_SLOT_BYTES)
        : nullptr;

    constexpr int DVT = 32;
    static_assert(fkc::DV % DVT == 0, "DV must be divisible by DVT");
    dim3 grid(N, H, fkc::DV / DVT);
    int smem = fkc::K2Smem<DVT>::SMEM_BYTES;

    // Wide-CTA: 512 threads for the baseline column-split.  Measured 12-18%
    // faster than 256 across all B=1 H-configs (halved delta iterations and
    // doubled cooperative parallelism in state-update / output-write); the
    // baseline is only selected for small grids (N*H<40) so this is always
    // beneficial.
    constexpr int K2W_THREADS = 512;

    const bool compact_bf16_state =
        (!has_state_in || (!init_state_is_fp32 && init_state_indices == nullptr))
        && (!has_state_out
            || (!final_state_is_fp32 && final_state_indices == nullptr));

    auto run = [&](auto vl, auto si, auto so, auto compact_state,
                   auto nthreads_c) {
        constexpr int NT = decltype(nthreads_c)::value;
        constexpr bool COMPACT = decltype(compact_state)::value;
        auto kern = fkc::flashkda_recurrence_kernel<
            DVT, NT, decltype(vl)::value, decltype(si)::value,
            decltype(so)::value, COMPACT>;
        auto state_in = fkc::make_k2_state_in_accessor<COMPACT>(
            init_state, init_state_indices, init_state_is_fp32);
        auto state_out = fkc::make_k2_state_out_accessor<COMPACT>(
            final_state, final_state_indices, final_state_is_fp32);
        hggcFuncSetAttribute(kern, hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
        kern<<<grid, NT, smem, stream>>>(
            reinterpret_cast<BF16*>(o), reinterpret_cast<const BF16*>(v),
            reinterpret_cast<const BF16*>(beta),
            reinterpret_cast<const BF16*>(ws),
            state_in, state_out,
            T_total, H, N, total_tiles, cu_seqlens, tile_prefix);
    };
    auto pick_state = [&](auto vl, auto si, auto so, auto nt) {
        if (compact_bf16_state)
            run(vl, si, so, std::true_type{}, nt);
        else
            run(vl, si, so, std::false_type{}, nt);
    };
    auto pick_so = [&](auto vl, auto si, auto nt) {
        if (has_state_out) pick_state(vl, si, std::true_type{}, nt);
        else pick_state(vl, si, std::false_type{}, nt);
    };
    auto pick_si = [&](auto vl, auto nt) {
        if (has_state_in) pick_so(vl, std::true_type{}, nt);
        else pick_so(vl, std::false_type{}, nt);
    };
    auto pick_vl = [&](auto nt) {
        if (is_varlen) pick_si(std::true_type{}, nt);
        else pick_si(std::false_type{}, nt);
    };
    pick_vl(std::integral_constant<int, K2W_THREADS>{});
}
