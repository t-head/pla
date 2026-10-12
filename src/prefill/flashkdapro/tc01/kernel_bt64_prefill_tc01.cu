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
 * FlashKDApro tc01 output-v4: FLA-style BT=64 Qg-prefix split candidate.
 *
 * K1 remains the established CH=16 producer.  This TU groups four raw K1
 * slots without crossing ragged sequence boundaries and executes
 *
 *   inter -> recompute -> recurrence -> prepare_qg -> output.
 *
 * Dense products remain TensorCell BF16 MMA (20 tc01 TensorCell CUs).  The
 * Qg prefix is split into a zero-MMA kernel so that its independent scalar
 * FP32 work can use the 64 generic-CU domain.  The intermediate ABI and all
 * carried/checkpoint values are BF16 by design; dot products accumulate in
 * FP32 and round only at the documented stage stores.
 */
#include <hggc_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "prefill/flashkdapro/common/bt64_workspace_abi.h"
#include "prefill/flashkdapro/common/launch_api.h"
#include "prefill/flashkdapro/common/runtime_arch.h"
// Includes flashkda_arch.cuh before CuTe/cutlass and selects the matching PPU
// trait family from the compiler-provided __HGGC_ARCH__.
#include "flashkda_ppu_util.cuh"

// tc01's AIU Copy_Atom needs mix-coordinate tensors.  Keep these includes
// after flashkda_ppu_util.cuh: flashkda_arch.cuh must select PPU0010
// before any architecture-specific copy traits are parsed.
#if defined(FLASHKDA_ARCH_100)
#include <cute/ppu_tensor_mix.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_atom.hpp>
#endif

namespace flashkda_cute_bt64 {

namespace fk = flashkda;
using BF16 = cutlass::bfloat16_t;
using namespace cute;

static constexpr int BT = fk::BT64_CHUNK;
static constexpr int BC = fk::BT64_SUBCHUNK;
static constexpr int NS = fk::BT64_SUBCHUNKS;
static constexpr int DK = fk::BT64_DK;
static constexpr int DV = fk::BT64_DV;
static constexpr int WARP = 32;
static constexpr int THREADS = 128;
static constexpr int BV = 16;
static constexpr int OUTPUT_THREADS = 256;
static constexpr int OUTPUT_BV = 64;
static constexpr int RECOMPUTE_THREADS_TC01 = 256;
static constexpr int RECOMPUTE_BV_TC01 = 64;
static constexpr int RECOMPUTE_SMEM_BYTES_TC01 = 18688;

struct BtMap {
    int seq;
    int local_bt;
    int tok;
    int len;
    int ws_base;
    bool valid;
};

struct SeqMap {
    int tok;
    int len;
    int bt_base;
    int nbt;
    int ws_base;
};

__device__ __forceinline__ int ceil_div_pos(int x, int d) {
    return x > 0 ? (x + d - 1) / d : 0;
}

// Map one packed BT index by independently scanning the BT64 and CH16 prefix.
// N is 2/4 in the promoted shapes, so this replaces a persistent descriptor
// allocation with only a handful of uniform device loads per CTA.
__device__ __forceinline__ BtMap map_packed_bt(
    int packed_bt, const int32_t* cu, int N) {
    BtMap m{-1, 0, 0, 0, 0, false};
    int bt_prefix = 0;
    int ws_prefix = 0;
    for (int s = 0; s < N; ++s) {
        const int bos = cu[s];
        const int eos = cu[s + 1];
        const int slen = eos > bos ? eos - bos : 0;
        const int nbt = ceil_div_pos(slen, BT);
        if (packed_bt < bt_prefix + nbt) {
            const int local = packed_bt - bt_prefix;
            const int remain = slen - local * BT;
            m.seq = s;
            m.local_bt = local;
            m.tok = bos + local * BT;
            m.len = remain < BT ? remain : BT;
            m.ws_base = ws_prefix + local * NS;
            m.valid = true;
            return m;
        }
        bt_prefix += nbt;
        ws_prefix += ceil_div_pos(slen, BC);
    }
    return m;
}

__device__ __forceinline__ SeqMap map_sequence(
    int seq, const int32_t* cu) {
    SeqMap m{0, 0, 0, 0, 0};
    for (int s = 0; s < seq; ++s) {
        const int bos = cu[s];
        const int eos = cu[s + 1];
        const int slen = eos > bos ? eos - bos : 0;
        m.bt_base += ceil_div_pos(slen, BT);
        m.ws_base += ceil_div_pos(slen, BC);
    }
    m.tok = cu[seq];
    const int eos = cu[seq + 1];
    m.len = eos > m.tok ? eos - m.tok : 0;
    m.nbt = ceil_div_pos(m.len, BT);
    return m;
}

__device__ __forceinline__ const BF16* k1_slot(
    const BF16* ws16, int total_tiles, int head, int ws_tile) {
    return ws16
        + int64_t(head * total_tiles + ws_tile) * fk::WS_SLOT_UNITS;
}

template <bool KrContiguous>
__device__ __forceinline__ BF16 load_k1_kr(
    const BF16* slot, int row, int dk) {
    const BF16* kr = slot + fk::WS_OFF_KR_U;
    if constexpr (KrContiguous) return kr[dk * BC + row];
    return kr[row * DK + dk];
}

#if defined(FLASHKDA_ARCH_100)
// Register-fused merge of one strict-lower 16x16 block:
//
//   A[BI,BJ] = -A[BI,BI] * BF16(sum_k Akk[BI,k] * A[k,BJ]).
//
// All operands are read directly from their row-major 64x64 images.  The MMA
// B views use (1, BT) strides, so no blockwide transpose/staging is needed.
// The inner products accumulate in one FP32 C fragment, round once into the
// warp-private BF16 tile, and feed the outer MMA from that tile.  The final
// store deliberately spells the old round-then-negate sequence so this change
// does not move an intentional BF16 boundary.
template <int BI, int BJ>
__device__ __forceinline__ void merge_offdiag_block_reg(
    const BF16* Akk, BF16* A, BF16* inner_bf16, int lane) {
    static_assert(0 <= BJ && BJ < BI && BI < NS,
                  "invalid BT64 strict-lower block");

    auto inner = fk::make_acc16_bf16_1warp(lane);
    cute::clear(inner);
    CUTE_UNROLL
    for (int bk = BJ; bk < BI; ++bk) {
        auto lhs = cute::make_tensor(
            cute::make_smem_ptr(const_cast<BF16*>(
                Akk + BI * BC * BT + bk * BC)),
            cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                              cute::make_stride(cute::Int<BT>{}, cute::_1{})));
        auto rhs = cute::make_tensor(
            cute::make_smem_ptr(A + bk * BC * BT + BJ * BC),
            cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                              cute::make_stride(cute::_1{}, cute::Int<BT>{})));
        fk::gemm16_ss_1warp_bf16_acc(lhs, rhs, lane, inner);
    }
    fk::store_acc16_bf16mma<BF16>(inner, inner_bf16, BC, lane);
    __syncwarp();

    auto diag = cute::make_tensor(
        cute::make_smem_ptr(A + BI * BC * BT + BI * BC),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                          cute::make_stride(cute::Int<BT>{}, cute::_1{})));
    auto inner_rhs = cute::make_tensor(
        cute::make_smem_ptr(inner_bf16),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                          cute::make_stride(cute::_1{}, cute::Int<BC>{})));
    auto outer = fk::make_acc16_bf16_1warp(lane);
    cute::clear(outer);
    fk::gemm16_ss_1warp_bf16_acc(diag, inner_rhs, lane, outer);

    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto out = cute::make_tensor(
        cute::make_smem_ptr(A + BI * BC * BT + BJ * BC),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                          cute::make_stride(cute::Int<BT>{}, cute::_1{})));
    auto out_part = thr_mma.partition_C(out);
    auto outer_f32 = cute::recast<float>(outer);
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(out_part); ++m)
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(out_part); ++n)
            CUTE_UNROLL
            for (int v = 0; v < cute::size<0>(out_part); ++v) {
                BF16 rounded = BF16(outer_f32(v, m, n));
                out_part(v, m, n) = BF16(-fk::bf16_to_f32(rounded));
            }
}

// Publish one pair accumulator directly into its final BF16 16x16 block.
// The frozen tc01 C ownership gives each lane two rows: values [0,4) own
// lane/4 and values [4,8) own lane/4+8.  Akk keeps the original FP32 beta
// multiply before the single BF16 rounding; Aqk only performs that rounding.
template <bool ApplyBeta, typename TC>
__device__ __forceinline__ void publish_pair_acc(
    TC const& acc, BF16* dst, const float* beta, int lane) {
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto out = cute::make_tensor(
        cute::make_smem_ptr(dst),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                          cute::make_stride(cute::Int<BT>{}, cute::_1{})));
    auto out_part = thr_mma.partition_C(out);
    auto acc_f32 = cute::recast<float>(acc);
    float beta_lo = 1.f;
    float beta_hi = 1.f;
    if constexpr (ApplyBeta) {
        beta_lo = beta[lane / 4];
        beta_hi = beta[lane / 4 + 8];
    }
    CUTE_UNROLL
    for (int value = 0; value < 8; ++value) {
        float x = acc_f32(value);
        if constexpr (ApplyBeta)
            x *= value < 4 ? beta_lo : beta_hi;
        out_part(value) = BF16(x);
    }
}
#else
// Preserve the frozen ppu0015 image byte-for-byte.  This translation unit is
// tc01-only at dispatch, but the fat compiler still emits both architecture
// images; the original block-serial helpers remain the explicit fallback.
template <int NThreads>
__device__ __forceinline__ void copy_matrix_block(
    const BF16* src, int ld, int br, int bc, BF16* dst, int tid) {
    for (int i = tid; i < BC * BC; i += NThreads) {
        const int r = i / BC;
        const int c = i - r * BC;
        dst[i] = src[(br * BC + r) * ld + bc * BC + c];
    }
}

template <int NThreads>
__device__ __forceinline__ void block_gemm_nn(
    const BF16* src_a, int lda, int abr, int abc,
    const BF16* src_b, int ldb, int bbr, int bbc,
    BF16* sA, BF16* sB, BF16* sC, int tid) {
    copy_matrix_block<NThreads>(src_a, lda, abr, abc, sA, tid);
    copy_matrix_block<NThreads>(src_b, ldb, bbr, bbc, sB, tid);
    __syncthreads();
    fk::gemm_nn_bf16_smem<BC, BC, BC>(sA, sB, sC, tid, NThreads);
    __syncthreads();
}

template <int NThreads>
__device__ __forceinline__ void block_gemm_nn_f32(
    const BF16* src_a, int lda, int abr, int abc,
    const BF16* src_b, int ldb, int bbr, int bbc,
    BF16* sA, BF16* sB, float* sC, int tid) {
    copy_matrix_block<NThreads>(src_a, lda, abr, abc, sA, tid);
    copy_matrix_block<NThreads>(src_b, ldb, bbr, bbc, sB, tid);
    __syncthreads();
    fk::gemm_nn_bf16_smem<BC, BC, BC>(sA, sB, sC, tid, NThreads);
    __syncthreads();
}

template <int NThreads>
__device__ __forceinline__ void copy_or_add_float_tile(
    float* dst, const float* src, bool add, int tid) {
    for (int i = tid; i < BC * BC; i += NThreads) {
        if (add) dst[i] += src[i];
        else dst[i] = src[i];
    }
    __syncthreads();
}

template <int NThreads>
__device__ __forceinline__ void float_to_bf16_tile(
    BF16* dst, const float* src, int tid) {
    for (int i = tid; i < BC * BC; i += NThreads) dst[i] = BF16(src[i]);
    __syncthreads();
}

template <int NThreads>
__device__ __forceinline__ void store_neg_tile(
    BF16* dst, int ld, int br, int bc, const BF16* src, int tid) {
    for (int i = tid; i < BC * BC; i += NThreads) {
        const int r = i / BC;
        const int c = i - r * BC;
        dst[(br * BC + r) * ld + bc * BC + c] =
            BF16(-fk::bf16_to_f32(src[i]));
    }
    __syncthreads();
}
#endif

struct alignas(128) InterSmem {
    BF16 A[BT * BT];
    BF16 Aqk[BT * BT];
    BF16 Akk[BT * BT];
    BF16 lhs_k[BC * DK];
    BF16 lhs_q[BC * DK];
    BF16 rhs[BC * DK];
#if !defined(FLASHKDA_ARCH_100)
    float pair_out[2 * BC * BC];
#endif
#if defined(FLASHKDA_ARCH_100)
    BF16 merge_inner[3 * BC * BC];
#else
    BF16 mm_a[BC * BC];
    BF16 mm_b[BC * BC];
    BF16 mm_c[BC * BC];
    float mm_f[BC * BC];
    float mm_sum[BC * BC];
#endif
    float beta[BT];
    float gt[NS * DK];
};
#if defined(FLASHKDA_ARCH_100)
static_assert(sizeof(InterSmem) == 40704,
              "tc01 BT64 direct pair epilogue TSM size changed");
#endif
static_assert(sizeof(InterSmem) <= 64 * 1024,
              "tc01 BT64 inter shared memory exceeds 64 KiB");

#if defined(FLASHKDA_ARCH_100)
// Treat two consecutive CH16 matrices as one [64,64] linear block, so AIU does not cut a 128-element row in half.
// The DefaultCopy consumer layout is unchanged; no vector payload is kept and no TSM or global staging is added.
__device__ __forceinline__ void inter_copy_kq_linear_tc01(
    BF16* dst, const BF16* slot) {
    static_assert(fk::WS_OFF_QD_U == fk::WS_OFF_KD_U + BC * DK,
                  "K/Q global image must be contiguous");
    static_assert(offsetof(InterSmem, lhs_q) ==
                      offsetof(InterSmem, lhs_k) + BC * DK * sizeof(BF16),
                  "K/Q TSM image must be contiguous");
    static_assert(2 * BC * DK == 64 * 64, "linear block size must exactly cover K/Q");
    using Copy = cute::PPU0010_AIU_LOAD<
        cute::C<64 * 64 * int(sizeof(BF16)) * 8>,
        cute::bfloat16_t, false, false>;
    const BF16* src = slot + fk::WS_OFF_KD_U;
    cute::AiuDesc desc;
    desc.template init<cute::bfloat16_t, false, 64, 64>(
        reinterpret_cast<const uint8_t*>(src), 64, 64,
        cute::make_stride(cute::_64{}, cute::_1{}));
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    Copy::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(address)), src, desc, 0, 0);
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
}
#endif

template <bool KrContiguous>
__global__ void __launch_bounds__(THREADS, 1) flashkda_bt64_inter_kernel(
    const BF16* __restrict__ ws16,
    BF16* __restrict__ bt_ws,
    int H, int N, int total_tiles, int total_bt,
    const int32_t* __restrict__ cu) {
    const int gbt = int(blockIdx.x);
    const int head = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    InterSmem& sm = *reinterpret_cast<InterSmem*>(smem_raw);
    const int nsub = ceil_div_pos(map.len, BC);

    for (int i = tid; i < BT * BT; i += THREADS) {
        sm.A[i] = BF16(0.f);
        sm.Aqk[i] = BF16(0.f);
        sm.Akk[i] = BF16(0.f);
    }
    for (int i = tid; i < BT; i += THREADS) sm.beta[i] = 0.f;
    for (int i = tid; i < NS * DK; i += THREADS) {
        const int sub = i / DK;
        const int dk = i - sub * DK;
        if (sub < nsub) {
            const BF16* slot = k1_slot(
                ws16, total_tiles, head, map.ws_base + sub);
            sm.gt[i] = reinterpret_cast<const float*>(
                slot + fk::WS_OFF_GT_U)[dk];
        } else {
            sm.gt[i] = 1.f;
        }
    }
    __syncthreads();

    // Diagonal inverse/Aqk blocks come directly from K1.  Missing CH16 blocks
    // are synthetic identity/zero blocks and therefore cannot consume the next
    // sequence's first workspace slot.
    for (int sub = 0; sub < NS; ++sub) {
        const BF16* slot = sub < nsub
            ? k1_slot(ws16, total_tiles, head, map.ws_base + sub) : nullptr;
        for (int i = tid; i < BC * BC; i += THREADS) {
            const int r = i / BC;
            const int c = i - r * BC;
            const int dst = (sub * BC + r) * BT + sub * BC + c;
            sm.A[dst] = slot ? slot[fk::WS_OFF_INV_U + i]
                             : BF16(r == c ? 1.f : 0.f);
            sm.Aqk[dst] = slot ? slot[fk::WS_OFF_MQK_U + i] : BF16(0.f);
        }
        for (int r = tid; r < BC; r += THREADS) {
            sm.beta[sub * BC + r] = slot
                ? reinterpret_cast<const float*>(
                      slot + fk::WS_OFF_BETA_U)[r]
                : 0.f;
        }
    }
    __syncthreads();

    // Six off-diagonal CH16 pairs.  Warp 0 computes Akk, warp 1 Aqk; both
    // contract eight DK blocks.  Gate products are rounded to BF16 at the same
    // operand boundary as FLA. Akk applies beta in FP32 before its BF16 store.
    for (int si = 1; si < NS; ++si) {
        for (int sj = 0; sj < si; ++sj) {
            const BF16* slot_i = si < nsub
                ? k1_slot(ws16, total_tiles, head, map.ws_base + si) : nullptr;
            const BF16* slot_j = sj < nsub
                ? k1_slot(ws16, total_tiles, head, map.ws_base + sj) : nullptr;
#if defined(FLASHKDA_ARCH_100)
            static_assert(THREADS == DK,
                          "tc01 bridge hoist requires one thread per dk");
            float bridge = 1.f;
            for (int p = sj + 1; p < si; ++p)
                bridge *= sm.gt[p * DK + tid];
#endif
#if defined(FLASHKDA_ARCH_100)
            // The previous pair's CTA barrier has already retired the old readers; AIU overlaps with the Kr/bridge preparation.
            if (slot_i) {
                if (tid == 0) inter_copy_kq_linear_tc01(sm.lhs_k, slot_i);
            } else {
                for (int i = tid; i < BC * DK; i += THREADS) {
                    sm.lhs_k[i] = BF16(0.f);
                    sm.lhs_q[i] = BF16(0.f);
                }
            }
            #pragma unroll 1
            for (int r = 0; r < BC; r += 4) {
                BF16 kr[4];
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j)
                    kr[j] = slot_j ? load_k1_kr<KrContiguous>(slot_j, r + j, tid) : BF16(0.f);
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j) {
                    const int i = (r + j) * DK + tid;
                    sm.rhs[i] = slot_j ? BF16(fk::bf16_to_f32(kr[j]) * bridge) : BF16(0.f);
                }
            }
            if (slot_i && tid == 0)
                asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#else
            for (int i = tid; i < BC * DK; i += THREADS) {
                const int r = i / DK;
                const int dk = i - r * DK;
                sm.lhs_k[i] = slot_i
                    ? slot_i[fk::WS_OFF_KD_U + i] : BF16(0.f);
                sm.lhs_q[i] = slot_i
                    ? slot_i[fk::WS_OFF_QD_U + i] : BF16(0.f);
#if !defined(FLASHKDA_ARCH_100)
                float bridge = 1.f;
                for (int p = sj + 1; p < si; ++p)
                    bridge *= sm.gt[p * DK + dk];
#endif
                sm.rhs[i] = slot_j
                    ? BF16(fk::bf16_to_f32(
                          load_k1_kr<KrContiguous>(slot_j, r, dk)) * bridge)
                    : BF16(0.f);
            }
#endif
            __syncthreads();

            const int warp = tid / WARP;
            const int lane = tid % WARP;
            if (warp < 2) {
                auto Kv = make_tensor(make_smem_ptr(sm.lhs_k),
                    make_layout(make_shape(Int<BC>{}, Int<DK>{}),
                                make_stride(Int<DK>{}, _1{})));
                auto Qv = make_tensor(make_smem_ptr(sm.lhs_q),
                    make_layout(make_shape(Int<BC>{}, Int<DK>{}),
                                make_stride(Int<DK>{}, _1{})));
                auto Rv = make_tensor(make_smem_ptr(sm.rhs),
                    make_layout(make_shape(Int<BC>{}, Int<DK>{}),
                                make_stride(Int<DK>{}, _1{})));
                auto acc = fk::make_acc16_bf16_1warp(lane);
                cute::clear(acc);
                CUTE_UNROLL
                for (int kb = 0; kb < DK / BC; ++kb) {
                    auto L = warp == 0
                        ? local_tile(Kv, make_shape(_16{}, _16{}), make_coord(0, kb))
                        : local_tile(Qv, make_shape(_16{}, _16{}), make_coord(0, kb));
                    auto R = local_tile(
                        Rv, make_shape(_16{}, _16{}), make_coord(0, kb));
                    fk::gemm16_ss_1warp_bf16_acc(L, R, lane, acc);
                }
#if defined(FLASHKDA_ARCH_100)
                BF16* pair_dst = warp == 0 ? sm.Akk : sm.Aqk;
                pair_dst += si * BC * BT + sj * BC;
                if (warp == 0)
                    publish_pair_acc<true>(
                        acc, pair_dst, sm.beta + si * BC, lane);
                else
                    publish_pair_acc<false>(acc, pair_dst, nullptr, lane);
#else
                fk::store_acc16_bf16mma<float>(
                    acc, sm.pair_out + warp * BC * BC, BC, lane);
#endif
            }
#if defined(FLASHKDA_ARCH_100)
            __syncthreads();
#else
            __syncthreads();
            for (int i = tid; i < BC * BC; i += THREADS) {
                const int r = i / BC;
                const int c = i - r * BC;
                const int dst = (si * BC + r) * BT + sj * BC + c;
                sm.Akk[dst] = BF16(
                    sm.pair_out[i] * sm.beta[si * BC + r]);
                sm.Aqk[dst] = BF16(sm.pair_out[BC * BC + i]);
            }
            __syncthreads();
#endif
        }
    }

#if defined(FLASHKDA_ARCH_100)
    // Three dependency layers replace six block-serial output schedules.  A
    // warp owns one strict-lower block and keeps its FP32 inner accumulator in
    // registers.  Only the layer boundaries are blockwide because A20 needs
    // A10, and A30/A31 need the preceding layer's A20/A21 respectively.
    const int merge_warp = tid / WARP;
    const int merge_lane = tid % WARP;
    BF16* merge_tile = sm.merge_inner + merge_warp * BC * BC;

    // L1: A10 = -A11 * BF16(Akk10 * A00).
    if (merge_warp == 0)
        merge_offdiag_block_reg<1, 0>(
            sm.Akk, sm.A, merge_tile, merge_lane);
    __syncthreads();

    // L2: A20 and A21 are independent after A10 is visible.
    if (merge_warp == 0)
        merge_offdiag_block_reg<2, 0>(
            sm.Akk, sm.A, merge_tile, merge_lane);
    else if (merge_warp == 1)
        merge_offdiag_block_reg<2, 1>(
            sm.Akk, sm.A, merge_tile, merge_lane);
    __syncthreads();

    // L3: A30, A31 and A32 are independent after A20/A21 are visible.
    if (merge_warp == 0)
        merge_offdiag_block_reg<3, 0>(
            sm.Akk, sm.A, merge_tile, merge_lane);
    else if (merge_warp == 1)
        merge_offdiag_block_reg<3, 1>(
            sm.Akk, sm.A, merge_tile, merge_lane);
    else if (merge_warp == 2)
        merge_offdiag_block_reg<3, 2>(
            sm.Akk, sm.A, merge_tile, merge_lane);
    __syncthreads();
#else
    // Frozen ppu0015 fallback: retain the original block-serial merge.
    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 1, 0, sm.A, BT, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    float_to_bf16_tile<THREADS>(sm.mm_c, sm.mm_f, tid);
    block_gemm_nn<THREADS>(
        sm.A, BT, 1, 1, sm.mm_c, BC, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_c, tid);
    store_neg_tile<THREADS>(sm.A, BT, 1, 0, sm.mm_c, tid);

    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 2, 0, sm.A, BT, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    copy_or_add_float_tile<THREADS>(sm.mm_sum, sm.mm_f, false, tid);
    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 2, 1, sm.A, BT, 1, 0,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    copy_or_add_float_tile<THREADS>(sm.mm_sum, sm.mm_f, true, tid);
    float_to_bf16_tile<THREADS>(sm.mm_c, sm.mm_sum, tid);
    block_gemm_nn<THREADS>(
        sm.A, BT, 2, 2, sm.mm_c, BC, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_c, tid);
    store_neg_tile<THREADS>(sm.A, BT, 2, 0, sm.mm_c, tid);

    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 3, 0, sm.A, BT, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    copy_or_add_float_tile<THREADS>(sm.mm_sum, sm.mm_f, false, tid);
    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 3, 1, sm.A, BT, 1, 0,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    copy_or_add_float_tile<THREADS>(sm.mm_sum, sm.mm_f, true, tid);
    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 3, 2, sm.A, BT, 2, 0,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    copy_or_add_float_tile<THREADS>(sm.mm_sum, sm.mm_f, true, tid);
    float_to_bf16_tile<THREADS>(sm.mm_c, sm.mm_sum, tid);
    block_gemm_nn<THREADS>(
        sm.A, BT, 3, 3, sm.mm_c, BC, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_c, tid);
    store_neg_tile<THREADS>(sm.A, BT, 3, 0, sm.mm_c, tid);

    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 2, 1, sm.A, BT, 1, 1,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    float_to_bf16_tile<THREADS>(sm.mm_c, sm.mm_f, tid);
    block_gemm_nn<THREADS>(
        sm.A, BT, 2, 2, sm.mm_c, BC, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_c, tid);
    store_neg_tile<THREADS>(sm.A, BT, 2, 1, sm.mm_c, tid);

    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 3, 1, sm.A, BT, 1, 1,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    copy_or_add_float_tile<THREADS>(sm.mm_sum, sm.mm_f, false, tid);
    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 3, 2, sm.A, BT, 2, 1,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    copy_or_add_float_tile<THREADS>(sm.mm_sum, sm.mm_f, true, tid);
    float_to_bf16_tile<THREADS>(sm.mm_c, sm.mm_sum, tid);
    block_gemm_nn<THREADS>(
        sm.A, BT, 3, 3, sm.mm_c, BC, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_c, tid);
    store_neg_tile<THREADS>(sm.A, BT, 3, 1, sm.mm_c, tid);

    block_gemm_nn_f32<THREADS>(
        sm.Akk, BT, 3, 2, sm.A, BT, 2, 2,
        sm.mm_a, sm.mm_b, sm.mm_f, tid);
    float_to_bf16_tile<THREADS>(sm.mm_c, sm.mm_f, tid);
    block_gemm_nn<THREADS>(
        sm.A, BT, 3, 3, sm.mm_c, BC, 0, 0,
        sm.mm_a, sm.mm_b, sm.mm_c, tid);
    store_neg_tile<THREADS>(sm.A, BT, 3, 2, sm.mm_c, tid);
#endif

    BF16* bt_slot = bt_ws
        + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;
    for (int i = tid; i < BT * BT; i += THREADS) {
        bt_slot[fk::BT64_WS_OFF_A_U + i] = sm.A[i];
        bt_slot[fk::BT64_WS_OFF_AQK_U + i] = sm.Aqk[i];
    }
}

// An independent pair CTA publishes the six lower-triangular blocks; U only becomes live after the same-stream merge completes.
// Host and device use the same struct budget, so the host sizeof cannot fall into another architecture's branch.
static constexpr int SPLIT_PAIR_COUNT = 6;
static constexpr int SPLIT_TMP_AKK = fk::BT64_WS_OFF_U_U;
struct alignas(128) SplitPairSmem {
    BF16 lhs_k[BC * DK];
    BF16 lhs_q[BC * DK];
    BF16 rhs[BC * DK];
    float beta[BC];
};
struct alignas(128) SplitMergeSmem {
    BF16 A[BT * BT];
    BF16 Akk[BT * BT];
    BF16 merge_inner[3 * BC * BC];
};
static_assert(sizeof(SplitPairSmem) == 12416, "pair TSM budget drifted");
static_assert(sizeof(SplitMergeSmem) == 17920, "merge TSM budget drifted");
static_assert(offsetof(SplitPairSmem, lhs_q) == BC * DK * sizeof(BF16),
              "pair K/Q must be physically contiguous");
static_assert(SPLIT_TMP_AKK + BT * BT <= fk::BT64_WS_OFF_U_U + fk::BT64_U_UNITS,
              "Akk handoff must lie inside the not-yet-live U");

#if defined(FLASHKDA_ARCH_100)
template <bool ApplyBeta, typename TC>
__device__ __forceinline__ void publish_split_pair_global(
    TC const& acc, BF16* dst, const float* beta, int lane) {
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto out = make_tensor(make_gmem_ptr(dst),
        make_layout(make_shape(_16{}, _16{}), make_stride(Int<BT>{}, _1{})));
    auto out_part = thr_mma.partition_C(out);
    auto acc_f32 = recast<float>(acc);
    float beta_lo = 1.f, beta_hi = 1.f;
    if constexpr (ApplyBeta) {
        beta_lo = beta[lane / 4];
        beta_hi = beta[lane / 4 + 8];
    }
    CUTE_UNROLL
    for (int value = 0; value < 8; ++value) {
        float x = acc_f32(value);
        if constexpr (ApplyBeta) x *= value < 4 ? beta_lo : beta_hi;
        out_part(value) = BF16(x);
    }
}
#endif

template <bool KrContiguous>
__global__ void __launch_bounds__(THREADS, 1) flashkda_bt64_split_pair_kernel(
    const BF16* __restrict__ ws16, BF16* __restrict__ bt_ws,
    int H, int N, int total_tiles, int total_bt, const int32_t* __restrict__ cu) {
#if defined(FLASHKDA_ARCH_100)
    const int flat = int(blockIdx.x);
    const int gbt = flat / SPLIT_PAIR_COUNT;
    const int pair = flat % SPLIT_PAIR_COUNT;
    const int si = pair < 1 ? 1 : (pair < 3 ? 2 : 3);
    const int sj = pair - si * (si - 1) / 2;
    const int head = int(blockIdx.y), tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int nsub = ceil_div_pos(map.len, BC);
    const BF16* slot_i = si < nsub
        ? k1_slot(ws16, total_tiles, head, map.ws_base + si) : nullptr;
    const BF16* slot_j = sj < nsub
        ? k1_slot(ws16, total_tiles, head, map.ws_base + sj) : nullptr;
    extern __shared__ __align__(128) unsigned char raw[];
    SplitPairSmem& sm = *reinterpret_cast<SplitPairSmem*>(raw);
    if (slot_i) {
        if (tid == 0) inter_copy_kq_linear_tc01(sm.lhs_k, slot_i);
    } else {
        for (int i = tid; i < BC * DK; i += THREADS) {
            sm.lhs_k[i] = BF16(0.f);
            sm.lhs_q[i] = BF16(0.f);
        }
    }
    if (tid < BC) sm.beta[tid] = slot_i
        ? reinterpret_cast<const float*>(slot_i + fk::WS_OFF_BETA_U)[tid] : 0.f;
    static_assert(THREADS == DK, "one thread per dk");
    float bridge = 1.f;
    for (int p = sj + 1; p < si; ++p) {
        const float gt = p < nsub ? reinterpret_cast<const float*>(
            k1_slot(ws16, total_tiles, head, map.ws_base + p) + fk::WS_OFF_GT_U)[tid] : 1.f;
        bridge *= gt;
    }
    #pragma unroll 1
    for (int r = 0; r < BC; r += 4) {
        BF16 kr[4];
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j)
            kr[j] = slot_j ? load_k1_kr<KrContiguous>(slot_j, r + j, tid) : BF16(0.f);
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j)
            sm.rhs[(r + j) * DK + tid] = slot_j
                ? BF16(fk::bf16_to_f32(kr[j]) * bridge) : BF16(0.f);
    }
    if (slot_i && tid == 0) asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    const int warp = tid / WARP, lane = tid % WARP;
    if (warp < 2) {
        auto Lv = make_tensor(make_smem_ptr(warp == 0 ? sm.lhs_k : sm.lhs_q),
            make_layout(make_shape(Int<BC>{}, Int<DK>{}), make_stride(Int<DK>{}, _1{})));
        auto Rv = make_tensor(make_smem_ptr(sm.rhs),
            make_layout(make_shape(Int<BC>{}, Int<DK>{}), make_stride(Int<DK>{}, _1{})));
        auto acc = fk::make_acc16_bf16_1warp(lane);
        cute::clear(acc);
        CUTE_UNROLL
        for (int kb = 0; kb < DK / BC; ++kb) {
            auto L = local_tile(Lv, make_shape(_16{}, _16{}), make_coord(0, kb));
            auto R = local_tile(Rv, make_shape(_16{}, _16{}), make_coord(0, kb));
            fk::gemm16_ss_1warp_bf16_acc(L, R, lane, acc);
        }
        BF16* slot = bt_ws + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;
        const int offset = si * BC * BT + sj * BC;
        if (warp == 0)
            publish_split_pair_global<true>(acc, slot + SPLIT_TMP_AKK + offset, sm.beta, lane);
        else
            publish_split_pair_global<false>(acc, slot + fk::BT64_WS_OFF_AQK_U + offset, nullptr, lane);
    }
#endif
}

__global__ void __launch_bounds__(THREADS, 1) flashkda_bt64_split_merge_kernel(
    const BF16* __restrict__ ws16, BF16* __restrict__ bt_ws,
    int H, int N, int total_tiles, int total_bt, const int32_t* __restrict__ cu) {
#if defined(FLASHKDA_ARCH_100)
    const int gbt = int(blockIdx.x), head = int(blockIdx.y), tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int nsub = ceil_div_pos(map.len, BC);
    BF16* slot = bt_ws + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;
    extern __shared__ __align__(128) unsigned char raw[];
    SplitMergeSmem& sm = *reinterpret_cast<SplitMergeSmem*>(raw);
    constexpr int VEC = int(sizeof(uint4) / sizeof(BF16));
    static_assert(VEC == 8 && BC % VEC == 0 && BT % VEC == 0,
                  "merge vectors must not cross a CH16 block or a matrix row");
    static_assert(fk::WS_SLOT_UNITS % VEC == 0 && fk::WS_OFF_INV_U % VEC == 0
                  && fk::WS_OFF_MQK_U % VEC == 0 && SPLIT_TMP_AKK % VEC == 0
                  && fk::BT64_WS_SLOT_UNITS % VEC == 0
                  && fk::BT64_WS_OFF_AQK_U % VEC == 0,
                  "merge global vectors must be 16-byte aligned");
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    // Only the input load width changes; the read set of the lower-triangular raw Akk and the Aqk writer stay the same.
    for (int vec = tid; vec < BT * BT / VEC; vec += THREADS) {
        const int i = vec * VEC;
        const int r = i / BT, c = i % BT;
        const int br = r / BC, bc = c / BC;
        if (br == bc) {
            const BF16* k1 = br < nsub
                ? k1_slot(ws16, total_tiles, head, map.ws_base + br) : nullptr;
            const int local = (r % BC) * BC + c % BC;
            if (k1) {
                const uint4 inv = *reinterpret_cast<const uint4*>(k1 + fk::WS_OFF_INV_U + local);
                const uint4 mqk = *reinterpret_cast<const uint4*>(k1 + fk::WS_OFF_MQK_U + local);
                *reinterpret_cast<uint4*>(sm.A + i) = inv;
                *reinterpret_cast<uint4*>(slot + fk::BT64_WS_OFF_AQK_U + i) = mqk;
            } else {
                *reinterpret_cast<uint4*>(sm.A + i) = zero;
                if (r >= c && r < c + VEC) sm.A[i + r - c] = BF16(1.f);
                *reinterpret_cast<uint4*>(slot + fk::BT64_WS_OFF_AQK_U + i) = zero;
            }
        } else {
            *reinterpret_cast<uint4*>(sm.A + i) = zero;
            if (br < bc) *reinterpret_cast<uint4*>(slot + fk::BT64_WS_OFF_AQK_U + i) = zero;
        }
        // An explicit branch keeps a conditional lvalue from selecting an address between the global vector and the local zero.
        if (br > bc) {
            const uint4 akk = *reinterpret_cast<const uint4*>(slot + SPLIT_TMP_AKK + i);
            *reinterpret_cast<uint4*>(sm.Akk + i) = akk;
        } else {
            *reinterpret_cast<uint4*>(sm.Akk + i) = zero;
        }
    }
    __syncthreads();
    const int warp = tid / WARP, lane = tid % WARP;
    BF16* inner = sm.merge_inner + (warp < 3 ? warp : 0) * BC * BC;
    if (warp == 0) merge_offdiag_block_reg<1, 0>(sm.Akk, sm.A, inner, lane);
    __syncthreads();
    if (warp == 0) merge_offdiag_block_reg<2, 0>(sm.Akk, sm.A, inner, lane);
    else if (warp == 1) merge_offdiag_block_reg<2, 1>(sm.Akk, sm.A, inner, lane);
    __syncthreads();
    if (warp == 0) merge_offdiag_block_reg<3, 0>(sm.Akk, sm.A, inner, lane);
    else if (warp == 1) merge_offdiag_block_reg<3, 1>(sm.Akk, sm.A, inner, lane);
    else if (warp == 2) merge_offdiag_block_reg<3, 2>(sm.Akk, sm.A, inner, lane);
    __syncthreads();
    for (int i = tid; i < BT * BT; i += THREADS)
        slot[fk::BT64_WS_OFF_A_U + i] = sm.A[i];
#endif
}

#if defined(FLASHKDA_ARCH_100)
// tc01 recompute-native closes three independently proven physical contracts:
//
//   row-major bt_ws A[64,64]
//     -> one PPU0010 AIU cube
//     -> AIU-native 8x64 TSM layout
//     -> PPU0010_TSM_LD_SWZL A registers;
//
//   row-major B[64,64]
//     -> aligned b32 loads
//     -> the recurrence path's fixed tc01 B-fragment shuffle;
//
//   tc01 C registers
//     -> the output-v5 CLayout transpose
//     -> aligned uint4 stores to the existing U/W image.
//
// A is loaded before any W store aliases its global image.  It remains in the
// native physical layout for the whole kernel and is never elementwise read.
using RecomputeACopyInstTc01 = cute::PPU0010_AIU_LOAD<
    cute::C<BT * RECOMPUTE_BV_TC01 * int(sizeof(cute::bfloat16_t)) * 8>,
    cute::bfloat16_t, /*Trans=*/false, /*Swzl=*/true>;
using RecomputeAGmemCopyTc01 = decltype(cute::make_tiled_copy(
    cute::Copy_Atom<RecomputeACopyInstTc01, cute::bfloat16_t>{},
    cute::Layout<cute::Shape<cute::_1, cute::_1>,
                 cute::Stride<cute::_1, cute::_1>>{},
    cute::Layout<cute::Shape<cute::Int<BT>,
                              cute::Int<RECOMPUTE_BV_TC01>>>{}));
using RecomputeASmemLayoutAtomTc01 = cute::Layout<
    cute::Shape<cute::_8, cute::Int<RECOMPUTE_BV_TC01>>,
    cute::Stride<cute::Int<RECOMPUTE_BV_TC01>, cute::_1>>;
using RecomputeASmemLayoutTc01 = decltype(cute::tile_to_shape(
    RecomputeASmemLayoutAtomTc01{},
    cute::make_shape(cute::Int<BT>{}, cute::Int<BT>{})));
using RecomputeALoadTc01 = cute::PPU0010_TSM_LD_SWZL<
    cute::bfloat16_t, BT, RECOMPUTE_BV_TC01,
    /*Swap=*/false, /*Trans=*/false, /*Inst=*/1>;

using RecomputeTc01MmaTraits =
    cute::MMA_Traits<flashkda::PpuBf16MMA>;
using RecomputeTc01ExpectedCLayout = cute::Layout<
    cute::Shape<cute::Shape<cute::_4, cute::_8>,
                cute::Shape<cute::_4, cute::_2>>,
    cute::Stride<cute::Stride<cute::_16, cute::_1>,
                 cute::Stride<cute::_64, cute::_8>>>;
static_assert(std::is_same<typename RecomputeTc01MmaTraits::ALayout,
                           typename RecomputeTc01MmaTraits::BLayout>::value,
              "tc01 recompute native A and shuffled B ownership must match");
static_assert(
    std::is_same<typename RecomputeTc01MmaTraits::CLayout,
                 RecomputeTc01ExpectedCLayout>::value,
    "tc01 recompute direct store requires the frozen C ownership");
static_assert(fk::BT64_WS_OFF_A_U == 0,
              "tc01 recompute native A must start at the slot base");
static_assert(fk::BT64_WS_OFF_A_U * int(sizeof(BF16)) % 128 == 0,
              "tc01 recompute native A source must be 128-byte aligned");
static_assert(fk::BT64_WS_SLOT_BYTES % 128 == 0,
              "tc01 recompute slot stride must preserve AIU alignment");
static_assert(DV % RECOMPUTE_BV_TC01 == 0 &&
              DK % RECOMPUTE_BV_TC01 == 0,
              "tc01 recompute U/W images must be BV64 tiled");

__device__ __forceinline__ void recompute_aiu_load_a_tc01(
    BF16* dst, const BF16* src) {
    auto gA = cute::make_mix_tensor(
        cute::make_gmem_ptr(
            reinterpret_cast<const cute::bfloat16_t*>(src)),
        cute::make_layout(
            cute::make_shape(cute::Int<BT>{}, cute::Int<BT>{}),
            cute::make_counting_stride<2, 1, 1>()));
    auto sA = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(dst)),
        RecomputeASmemLayoutTc01{});
    RecomputeAGmemCopyTc01 tiled_copy;
    tiled_copy.desc_.template init<cute::bfloat16_t,
                                    /*Trans=*/false, BT, BT>(
        reinterpret_cast<const uint8_t*>(src),
        /*MN=*/BT, /*K=*/BT, cute::make_stride(BT, cute::_1{}));
    auto thr = tiled_copy.get_slice(0);
    auto tAgA = thr.partition_S(gA);
    auto tAsA = thr.partition_D(sA);
    cute::copy(tiled_copy, tAgA, tAsA);
}

__device__ __forceinline__ void recompute_aiu_commit_tc01() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void recompute_aiu_wait_all_tc01() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void* recompute_smem_base_tc01(
    const BF16* ptr) {
#if defined(__HGGC_ARCH__)
    const uint32_t address = static_cast<uint32_t>(
        __cvta_generic_to_shared(ptr));
    return reinterpret_cast<void*>(static_cast<uintptr_t>(address));
#else
    return const_cast<BF16*>(ptr);
#endif
}

__device__ __forceinline__ void recompute_load_a_native_tc01(
    const BF16* src, int coord_row, int coord_col, uint32_t regs[4]) {
    RecomputeALoadTc01::copy(
        regs, recompute_smem_base_tc01(src),
        /*coord_w=*/coord_col, /*coord_h=*/coord_row,
        /*cube_in_stage=*/0, /*stage=*/0);
}

// This is the physically qualified recurrence state-B loader with only its
// row stride widened from 32 to 64.  Lane/register ownership is unchanged:
// reg r owns N=(lane>>2)+8*(r>=2), K=2*(lane&3)+8*(r&1)+{0,1}.
__device__ __forceinline__ void recompute_load_b_vec32_shuffle_tc01(
    const BF16* matrix, int nb, int kb, int lane, uint32_t regs[4]) {
    static_assert(BC == 16 && RECOMPUTE_BV_TC01 == 64,
                  "tc01 recompute B loader is specialized to B[64,64]");
    const int n0 = nb * BC + ((lane & 3) << 1);
    const int k0 = kb * BC + (lane >> 2);
    const BF16* row0 = matrix + k0 * RECOMPUTE_BV_TC01 + n0;
    const BF16* row8 = row0 + 8 * RECOMPUTE_BV_TC01;
    const uint32_t* row0_u32 = reinterpret_cast<const uint32_t*>(row0);
    const uint32_t* row8_u32 = reinterpret_cast<const uint32_t*>(row8);
    uint32_t raw[4];
    raw[0] = row0_u32[0];
    raw[1] = row0_u32[4];
    raw[2] = row8_u32[0];
    raw[3] = row8_u32[4];

    const int src_lane = ((lane & 3) << 3) | (lane >> 3);
    const int half_shift = (lane & 4) ? 16 : 0;
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int src_reg = ((i & 1) << 1) | (i >> 1);
        const uint32_t even_k =
            __shfl_sync(0xffffffff, raw[src_reg], src_lane);
        const uint32_t odd_k =
            __shfl_sync(0xffffffff, raw[src_reg], src_lane + 4);
        regs[i] = ((even_k >> half_shift) & 0xffffu)
            | (((odd_k >> half_shift) & 0xffffu) << 16);
    }
}

// Output-v5's bitwise-qualified tc01 CLayout transpose.  Every 16x16 tile is
// rounded once in registers and emitted as 32 aligned uint4 stores; there is
// no output scratch or second BF16 materialization.
template <typename Acc>
__device__ __forceinline__ void recompute_store_acc_direct_tc01(
    const Acc& acc, BF16* dst, int row_stride, int mb, int nb, int lane) {
    Tensor rC = recast<float>(acc);
    uint32_t frag[4];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const uint32_t lo = BF16(rC(2 * i, 0, 0)).storage;
        const uint32_t hi = BF16(rC(2 * i + 1, 0, 0)).storage;
        frag[i] = fk::pack_half2_u32(lo, hi);
    }

    uint32_t p0 = __shfl_xor_sync(0xffffffffu, frag[1], 1, 4);
    uint32_t p1 = __shfl_xor_sync(0xffffffffu, frag[0], 1, 4);
    uint32_t p2 = __shfl_xor_sync(0xffffffffu, frag[3], 1, 4);
    uint32_t p3 = __shfl_xor_sync(0xffffffffu, frag[2], 1, 4);
    if (lane & 1) {
        frag[0] = p0;
        frag[2] = p2;
    } else {
        frag[1] = p1;
        frag[3] = p3;
    }

    p0 = __shfl_xor_sync(0xffffffffu, frag[2], 2, 4);
    p1 = __shfl_xor_sync(0xffffffffu, frag[3], 2, 4);
    p2 = __shfl_xor_sync(0xffffffffu, frag[0], 2, 4);
    p3 = __shfl_xor_sync(0xffffffffu, frag[1], 2, 4);
    if (lane & 2) {
        frag[0] = p0;
        frag[1] = p1;
    } else {
        frag[2] = p2;
        frag[3] = p3;
    }

    uint4 payload;
    payload.x = fk::pack_half2_u32(
        fk::get_half_from_pack(frag[0], 0),
        fk::get_half_from_pack(frag[1], 0));
    payload.y = fk::pack_half2_u32(
        fk::get_half_from_pack(frag[2], 0),
        fk::get_half_from_pack(frag[3], 0));
    payload.z = fk::pack_half2_u32(
        fk::get_half_from_pack(frag[0], 1),
        fk::get_half_from_pack(frag[1], 1));
    payload.w = fk::pack_half2_u32(
        fk::get_half_from_pack(frag[2], 1),
        fk::get_half_from_pack(frag[3], 1));

    const int row = (lane >> 2) + ((lane & 2) << 2);
    const int half = lane & 1;
    *reinterpret_cast<uint4*>(
        dst + (mb * BC + row) * row_stride + nb * BC + half * 8) = payload;
}

__device__ __forceinline__ void recompute_gemm_native_pass_tc01(
    const BF16* a, const BF16* b, BF16* dst, int row_stride, int tid) {
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    static_assert(RECOMPUTE_THREADS_TC01 / WARP == 8,
                  "tc01 recompute native schedule requires eight warps");
    const int nb = warp & 3;
    const int mb0 = (warp >> 2) * 2;
    const int mb1 = mb0 + 1;
    auto acc0 = fk::make_acc16_bf16_1warp(lane);
    auto acc1 = fk::make_acc16_bf16_1warp(lane);
    cute::clear(acc0);
    cute::clear(acc1);
    CUTE_UNROLL
    for (int kb = 0; kb < BT / BC; ++kb) {
        uint32_t b_regs[4];
        recompute_load_b_vec32_shuffle_tc01(b, nb, kb, lane, b_regs);
        uint32_t a_regs[4];
        recompute_load_a_native_tc01(
            a, mb0 * BC, kb * BC, a_regs);
        fk::gemm16_rr_1warp_bf16_acc(a_regs, b_regs, acc0);
        recompute_load_a_native_tc01(
            a, mb1 * BC, kb * BC, a_regs);
        fk::gemm16_rr_1warp_bf16_acc(a_regs, b_regs, acc1);
    }
    recompute_store_acc_direct_tc01(
        acc0, dst, row_stride, mb0, nb, lane);
    recompute_store_acc_direct_tc01(
        acc1, dst, row_stride, mb1, nb, lane);
}
#endif  // FLASHKDA_ARCH_100

// The checkpoint is not live before the recurrence starts and holds exactly two BT x 128 BF16 inputs.
// Kernel boundaries on the same stream guarantee the exclusive lifetime of prepare -> recompute -> recurrence.
static_assert(DK == DV && 2 * BT * DK == DV * DK,
              "recompute scratch must fit the dead checkpoint image");
struct alignas(128) RecomputeInputSmem {
    float beta[BT];
    float gt[NS * DK];
};

template <bool KrContiguous>
__global__ void __launch_bounds__(RECOMPUTE_THREADS_TC01, 1)
flashkda_bt64_prepare_recompute_kernel(
    const BF16* __restrict__ v, const BF16* __restrict__ ws16,
    BF16* __restrict__ bt_ws, BF16* __restrict__ prepared,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* __restrict__ cu) {
    const int gbt = int(blockIdx.x);
    const int head = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int nsub = ceil_div_pos(map.len, BC);
    __shared__ RecomputeInputSmem sm;
    for (int i = tid; i < BT; i += RECOMPUTE_THREADS_TC01) {
        const int sub = i / BC;
        sm.beta[i] = sub < nsub ? reinterpret_cast<const float*>(
            k1_slot(ws16, total_tiles, head, map.ws_base + sub)
            + fk::WS_OFF_BETA_U)[i % BC] : 0.f;
    }
    for (int i = tid; i < NS * DK; i += RECOMPUTE_THREADS_TC01) {
        const int sub = i / DK;
        sm.gt[i] = sub < nsub ? reinterpret_cast<const float*>(
            k1_slot(ws16, total_tiles, head, map.ws_base + sub)
            + fk::WS_OFF_GT_U)[i % DK] : 1.f;
    }
    __syncthreads();
    BF16* dst = prepared + int64_t(gbt * H + head) * DV * DK;
    BF16* kg_dst = bt_ws + int64_t(gbt * H + head)
        * fk::BT64_WS_SLOT_UNITS + fk::BT64_WS_OFF_KG_U;
    // Only the scalar work is split out; each element keeps the load -> prefix/suffix -> scale -> BF16 order.
    for (int i = tid; i < BT * DK; i += RECOMPUTE_THREADS_TC01) {
        const int r = i / DK;
        const int dk = i % DK;
        const int sub = r / BC;
        float vv = 0.f, kd = 0.f, kg = 0.f;
        if (r < map.len && map.tok + r < T_total) {
            vv = fk::bf16_to_f32(v[int64_t(map.tok + r) * H * DV
                + int64_t(head) * DV + dk]);
        }
        if (sub < nsub && r < map.len) {
            const BF16* slot = k1_slot(
                ws16, total_tiles, head, map.ws_base + sub);
            kd = fk::bf16_to_f32(slot[fk::WS_OFF_KD_U + (r % BC) * DK + dk]);
            kg = fk::bf16_to_f32(load_k1_kr<KrContiguous>(slot, r % BC, dk));
            float prefix = 1.f, suffix = 1.f;
            for (int p = 0; p < sub; ++p) prefix *= sm.gt[p * DK + dk];
            for (int p = sub + 1; p < nsub; ++p) suffix *= sm.gt[p * DK + dk];
            kd *= prefix;
            kg *= suffix;
        }
        dst[i] = BF16(vv * sm.beta[r]);
        dst[BT * DK + i] = BF16(kd * sm.beta[r]);
        kg_dst[i] = BF16(kg);
    }
}

struct alignas(128) RecomputeSmem {
#if defined(FLASHKDA_ARCH_100)
    BF16 A[BT * BT];
    BF16 in[BT * RECOMPUTE_BV_TC01];
#else
    BF16 A[BT * BT];
    BF16 in[BT * BC];
    BF16 out[BT * BC];
#endif
    float beta[BT];
    float gt[NS * DK];
};
#if defined(FLASHKDA_ARCH_100)
static_assert(sizeof(RecomputeSmem) == RECOMPUTE_SMEM_BYTES_TC01,
              "tc01 recompute-native TSM budget drifted");
static_assert(sizeof(RecomputeSmem) <= 32 * 1024,
              "tc01 recompute-native must remain within 32 KiB TSM");
#endif
static_assert(sizeof(RecomputeSmem) <= 64 * 1024,
              "tc01 BT64 recompute shared memory exceeds 64 KiB");

template <bool KrContiguous>
#if defined(FLASHKDA_ARCH_100)
__global__ void __launch_bounds__(RECOMPUTE_THREADS_TC01, 1)
#else
__global__ void __launch_bounds__(THREADS, 1)
#endif
flashkda_bt64_recompute_kernel(
    const BF16* __restrict__ v,
    const BF16* __restrict__ ws16,
    BF16* __restrict__ bt_ws,
    const BF16* __restrict__ prepared,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* __restrict__ cu) {
#if defined(FLASHKDA_ARCH_100)
    const int gbt = int(blockIdx.x);
    const int head = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int nsub = ceil_div_pos(map.len, BC);
    BF16* bt_slot = bt_ws
        + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    RecomputeSmem& sm = *reinterpret_cast<RecomputeSmem*>(smem_raw);

    // The complete A enters TSM before W overwrites it; the inputs were already rounded by an independent zero-MMA stage.
    if (tid == 0) {
        recompute_aiu_load_a_tc01(sm.A, bt_slot + fk::BT64_WS_OFF_A_U);
        recompute_aiu_commit_tc01();
        recompute_aiu_wait_all_tc01();
    }
    __syncthreads();
    const BF16* input = prepared + int64_t(gbt * H + head) * DV * DK;

    // The GEMM for U uses exactly the same contraction order as the original.
    for (int fb = 0; fb < DV / RECOMPUTE_BV_TC01; ++fb) {
        for (int vec = tid; vec < BT * RECOMPUTE_BV_TC01 / 8;
             vec += RECOMPUTE_THREADS_TC01) {
            const int i = vec * 8;
            const int r = i / RECOMPUTE_BV_TC01;
            const int c = i % RECOMPUTE_BV_TC01;
            *reinterpret_cast<uint4*>(sm.in + i) =
                *reinterpret_cast<const uint4*>(
                    input + r * DV + fb * RECOMPUTE_BV_TC01 + c);
        }
        __syncthreads();
        recompute_gemm_native_pass_tc01(
            sm.A, sm.in,
            bt_slot + fk::BT64_WS_OFF_U_U
                + fb * RECOMPUTE_BV_TC01,
            DV, tid);
        __syncthreads();
    }

    // Kg is already published in place; the W input is only copied bitwise, with no extra BF16 rounding.
    for (int fb = 0; fb < DK / RECOMPUTE_BV_TC01; ++fb) {
        for (int vec = tid; vec < BT * RECOMPUTE_BV_TC01 / 8;
             vec += RECOMPUTE_THREADS_TC01) {
            const int i = vec * 8;
            const int r = i / RECOMPUTE_BV_TC01;
            const int c = i % RECOMPUTE_BV_TC01;
            *reinterpret_cast<uint4*>(sm.in + i) =
                *reinterpret_cast<const uint4*>(input + BT * DK
                    + r * DK + fb * RECOMPUTE_BV_TC01 + c);
        }
        __syncthreads();
        recompute_gemm_native_pass_tc01(
            sm.A, sm.in,
            bt_slot + fk::BT64_WS_OFF_W_U
                + fb * RECOMPUTE_BV_TC01,
            DK, tid);
        __syncthreads();
    }
#else
    const int gbt = int(blockIdx.x);
    const int head = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int nsub = ceil_div_pos(map.len, BC);
    BF16* bt_slot = bt_ws
        + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    RecomputeSmem& sm = *reinterpret_cast<RecomputeSmem*>(smem_raw);

    // P0 alias invariant: this is the sole (gbt,head) CTA. Copy all of A and
    // join before the first W store can overwrite its global byte image.
    for (int i = tid; i < BT * BT; i += THREADS)
        sm.A[i] = bt_slot[fk::BT64_WS_OFF_A_U + i];
    for (int i = tid; i < BT; i += THREADS) {
        const int sub = i / BC;
        const int row = i - sub * BC;
        if (sub < nsub) {
            const BF16* slot = k1_slot(
                ws16, total_tiles, head, map.ws_base + sub);
            sm.beta[i] = reinterpret_cast<const float*>(
                slot + fk::WS_OFF_BETA_U)[row];
        } else {
            sm.beta[i] = 0.f;
        }
    }
    for (int i = tid; i < NS * DK; i += THREADS) {
        const int sub = i / DK;
        const int dk = i - sub * DK;
        if (sub < nsub) {
            const BF16* slot = k1_slot(
                ws16, total_tiles, head, map.ws_base + sub);
            sm.gt[i] = reinterpret_cast<const float*>(
                slot + fk::WS_OFF_GT_U)[dk];
        } else {
            sm.gt[i] = 1.f;
        }
    }
    __syncthreads();

    // U = A @ BF16(V * beta), processed in 16 value columns.
    for (int fb = 0; fb < DV / BC; ++fb) {
        for (int i = tid; i < BT * BC; i += THREADS) {
            const int r = i / BC;
            const int c = i - r * BC;
            float vv = 0.f;
            if (r < map.len && map.tok + r < T_total) {
                const int64_t off = int64_t(map.tok + r) * H * DV
                    + int64_t(head) * DV + fb * BC + c;
                vv = fk::bf16_to_f32(v[off]);
            }
            sm.in[i] = BF16(vv * sm.beta[r]);
        }
        __syncthreads();
        fk::gemm_nn_bf16_smem<BT, BC, BT>(
            sm.A, sm.in, sm.out, tid, THREADS);
        __syncthreads();
        for (int i = tid; i < BT * BC; i += THREADS) {
            const int r = i / BC;
            const int c = i - r * BC;
            bt_slot[fk::BT64_WS_OFF_U_U + r * DV + fb * BC + c] = sm.out[i];
        }
        __syncthreads();
    }

    // W = A @ BF16(Kd * prefix_gate * beta).  Kg is K restored to the
    // end of this BT64: Kr_i * product(gt_{i+1..last}).
    for (int fb = 0; fb < DK / BC; ++fb) {
        for (int i = tid; i < BT * BC; i += THREADS) {
            const int r = i / BC;
            const int c = i - r * BC;
            const int sub = r / BC;
            const int row = r - sub * BC;
            const int dk = fb * BC + c;
            float kd = 0.f;
            float kg = 0.f;
            if (sub < nsub && r < map.len) {
                const BF16* slot = k1_slot(
                    ws16, total_tiles, head, map.ws_base + sub);
                kd = fk::bf16_to_f32(slot[fk::WS_OFF_KD_U + row * DK + dk]);
                kg = fk::bf16_to_f32(
                    load_k1_kr<KrContiguous>(slot, row, dk));
                float prefix = 1.f;
                float suffix = 1.f;
                for (int p = 0; p < sub; ++p)
                    prefix *= sm.gt[p * DK + dk];
                for (int p = sub + 1; p < nsub; ++p)
                    suffix *= sm.gt[p * DK + dk];
                kd *= prefix;
                kg *= suffix;
            }
            sm.in[i] = BF16(kd * sm.beta[r]);
            bt_slot[fk::BT64_WS_OFF_KG_U + r * DK + dk] = BF16(kg);
        }
        __syncthreads();
        fk::gemm_nn_bf16_smem<BT, BC, BT>(
            sm.A, sm.in, sm.out, tid, THREADS);
        __syncthreads();
        for (int i = tid; i < BT * BC; i += THREADS) {
            const int r = i / BC;
            const int c = i - r * BC;
            bt_slot[fk::BT64_WS_OFF_W_U + r * DK + fb * BC + c] = sm.out[i];
        }
        __syncthreads();
    }
#endif
}

#if defined(FLASHKDA_ARCH_100)
// tc01 profile-only experiment: W keeps the proven native path and Kg becomes
// the one new variable.  State/x remain on their established row-major
// DefaultCopy paths, and recurrence arithmetic/rounding stays unchanged.
using RecurrenceWCopyInstTc01 = cute::PPU0010_AIU_LOAD<
    cute::C<BT * 64 * int(sizeof(cute::bfloat16_t)) * 8>,
    cute::bfloat16_t, /*Trans=*/false, /*Swzl=*/true>;
using RecurrenceWGmemCopyTc01 = decltype(cute::make_tiled_copy(
    cute::Copy_Atom<RecurrenceWCopyInstTc01, cute::bfloat16_t>{},
    cute::Layout<cute::Shape<cute::_1, cute::_1>,
                 cute::Stride<cute::_1, cute::_1>>{},
    cute::Layout<cute::Shape<cute::Int<BT>, cute::Int<64>>>{}));
using RecurrenceWSmemLayoutAtomTc01 = cute::Layout<
    cute::Shape<cute::_8, cute::Int<64>>,
    cute::Stride<cute::Int<64>, cute::_1>>;
using RecurrenceWSmemLayoutTc01 = decltype(cute::tile_to_shape(
    RecurrenceWSmemLayoutAtomTc01{},
    cute::make_shape(cute::Int<BT>{}, cute::Int<DK>{})));
using RecurrenceWLoadTc01 = cute::PPU0010_TSM_LD_SWZL<
    cute::bfloat16_t, BT, 64, /*Swap=*/false, /*Trans=*/false, /*Inst=*/2>;
using RecurrenceKgLoadTc01 = cute::PPU0010_TSM_LD_SWZL<
    cute::bfloat16_t, BT, 64, /*Swap=*/false, /*Trans=*/true, /*Inst=*/2>;

using RecurrenceTc01MmaTraits =
    cute::MMA_Traits<flashkda::PpuBf16MMA>;
static_assert(std::is_same<typename RecurrenceTc01MmaTraits::ALayout,
                           typename RecurrenceTc01MmaTraits::BLayout>::value,
              "tc01 recurrence native-W needs identical A/B ownership");
static_assert(DK == 2 * 64,
              "tc01 recurrence native-W expects two 128-byte K slices");
static_assert(fk::BT64_WS_OFF_W_U == 0,
              "tc01 recurrence native-W source alignment contract drifted");
static_assert(fk::BT64_WS_OFF_KG_U * int(sizeof(BF16)) % 128 == 0,
              "tc01 recurrence native-Kg source must be 128-byte aligned");
static_assert(fk::BT64_WS_SLOT_BYTES % 128 == 0,
              "tc01 recurrence native-W slot must remain 128-byte aligned");

__device__ __forceinline__ void recurrence_aiu_load_w_tc01(
    BF16* dst, const BF16* src) {
    auto gW = cute::make_mix_tensor(
        cute::make_gmem_ptr(
            reinterpret_cast<const cute::bfloat16_t*>(src)),
        cute::make_layout(
            cute::make_shape(cute::Int<BT>{}, cute::Int<DK>{}),
            cute::make_counting_stride<2, 1, 1>()));
    auto sW = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(dst)),
        RecurrenceWSmemLayoutTc01{});
    RecurrenceWGmemCopyTc01 tiled_copy;
    tiled_copy.desc_.template init<cute::bfloat16_t,
                                    /*Trans=*/false, BT, DK>(
        reinterpret_cast<const uint8_t*>(src),
        /*MN=*/BT, /*K=*/DK, cute::make_stride(DK, cute::_1{}));
    auto thr = tiled_copy.get_slice(0);
    auto tWgW = thr.partition_S(gW);
    auto tWsW = thr.partition_D(sW);
    cute::copy(tiled_copy, tWgW, tWsW);
}

__device__ __forceinline__ void recurrence_aiu_commit_tc01() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void recurrence_aiu_wait_all_tc01() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void* recurrence_smem_base_tc01(
    const BF16* ptr) {
#if defined(__HGGC_ARCH__)
    const uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(ptr));
    return reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
#else
    return const_cast<BF16*>(ptr);
#endif
}

__device__ __forceinline__ void recurrence_load_w_nontrans_tc01(
    const BF16* src, int coord_row, int coord_col, uint32_t regs[4]) {
    const int cube = coord_col / 64;
    RecurrenceWLoadTc01::copy(
        regs, recurrence_smem_base_tc01(src),
        /*coord_w=*/coord_col - cube * 64, /*coord_h=*/coord_row,
        /*cube_in_stage=*/cube, /*stage=*/0);
}

// Kg is physically row-major [BT=64,DK=128], while the recurrence consumes
// logical A=Kg^T [DK=128,BT=64].  The AIU producer remains Trans=false so its
// descriptor sees the real 128-element row stride.  Only the paired TSM
// consumer is Trans=true.  coord_m selects one of two physical 64-column
// cubes; coord_k is the physical row / logical contraction coordinate.
__device__ __forceinline__ void recurrence_load_kg_trans_tc01(
    const BF16* src, int coord_m, int coord_k, uint32_t regs[4]) {
    const int cube = coord_m / 64;
    RecurrenceKgLoadTc01::copy(
        regs, recurrence_smem_base_tc01(src),
        /*coord_h=*/coord_k, /*coord_w=*/coord_m - cube * 64,
        /*cube_in_stage=*/cube, /*stage=*/0);
}

// Vectorized row-major state-B loader plus the physically qualified fixed
// tc01 register transpose.  Four aligned words first load adjacent N values
// from state[K,N].  DefaultCopy's real B ABI packs adjacent K values for one
// N, so two direct shuffles per output register close the layout contract.
__device__ __forceinline__ void recurrence_load_b_vec32_shuffle_tc01(
    const BF16* state, int nb, int kb, int lane, uint32_t regs[4]) {
    static_assert(BC == 16 && BV == 16,
                  "tc01 state-B mapping is specialized to B[16,16]");
    static_assert((BV * int(sizeof(BF16))) % int(sizeof(uint32_t)) == 0,
                  "tc01 state rows must preserve b32 alignment");

    const int n0 = nb * BC + ((lane & 3) << 1);
    const int k0 = kb * BC + (lane >> 2);
    const BF16* row0 = state + k0 * BV + n0;
    const BF16* row8 = row0 + 8 * BV;
    const uint32_t* row0_u32 = reinterpret_cast<const uint32_t*>(row0);
    const uint32_t* row8_u32 = reinterpret_cast<const uint32_t*>(row8);
    uint32_t raw[4];
    raw[0] = row0_u32[0];
    raw[1] = row0_u32[4];
    raw[2] = row8_u32[0];
    raw[3] = row8_u32[4];

    const int src_lane = ((lane & 3) << 3) | (lane >> 3);
    const int half_shift = (lane & 4) ? 16 : 0;
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int src_reg = ((i & 1) << 1) | (i >> 1);  // 0, 2, 1, 3
        const uint32_t even_k =
            __shfl_sync(0xffffffff, raw[src_reg], src_lane);
        const uint32_t odd_k =
            __shfl_sync(0xffffffff, raw[src_reg], src_lane + 4);
        regs[i] = ((even_k >> half_shift) & 0xffffu)
            | (((odd_k >> half_shift) & 0xffffu) << 16);
    }
}

// U is physically [BT,DV] in bt_ws, while this recurrence CTA owns one
// aligned [BT,BV] value slice.  Preserve sm.x's row-major representation for
// its C-fragment, transposed-B and x_global consumers, but replace 2,048
// scalar b16 transfers with 256 aligned 16-byte transfers.  This stays
// tc01-only so the ppu0015 fallback remains byte-for-byte source-identical.
static constexpr int kRecurrenceUVecBytesTc01 = int(sizeof(uint4));
static constexpr int kRecurrenceUVecElemsTc01 =
    kRecurrenceUVecBytesTc01 / int(sizeof(BF16));
static_assert(kRecurrenceUVecBytesTc01 == 16,
              "tc01 recurrence U vector must remain 16 bytes");
static_assert(BV % kRecurrenceUVecElemsTc01 == 0,
              "tc01 recurrence U row must be vector divisible");
static_assert((DV * int(sizeof(BF16))) % alignof(uint4) == 0,
              "tc01 recurrence U source row stride must be aligned");
static_assert((BV * int(sizeof(BF16))) % alignof(uint4) == 0,
              "tc01 recurrence U dv-tile offset must be aligned");
static_assert((fk::BT64_WS_OFF_U_U * int(sizeof(BF16))) %
                  alignof(uint4) == 0,
              "tc01 recurrence U workspace base must be aligned");
static_assert(fk::BT64_WS_SLOT_BYTES % alignof(uint4) == 0,
              "tc01 recurrence workspace slot must be aligned");

__device__ __forceinline__ void recurrence_copy_u_vec16_tc01(
    BF16* dst, const BF16* bt_slot, int dv_base, int tid) {
    constexpr int kVecsPerRow = BV / kRecurrenceUVecElemsTc01;
    constexpr int kVectorCount = BT * kVecsPerRow;
    CUTE_UNROLL
    for (int vec = tid; vec < kVectorCount; vec += THREADS) {
        const int row = vec / kVecsPerRow;
        const int vec_in_row = vec - row * kVecsPerRow;
        const int col = vec_in_row * kRecurrenceUVecElemsTc01;
        *reinterpret_cast<uint4*>(dst + row * BV + col) =
            *reinterpret_cast<const uint4*>(
                bt_slot + fk::BT64_WS_OFF_U_U
                + row * DV + dv_base + col);
    }
}

// Exact split form of gemm16_ss_1warp_bf16_acc's generic B half.  X remains
// on DefaultCopy, while state-B is supplied by the loader above and W/Kg are
// independently supplied as native A.
template <typename SmemB>
__device__ __forceinline__ void recurrence_load_b_default_tc01(
    SmemB const& sB, int lane, uint32_t regs[4]) {
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    Tensor rB_full = thr_mma.partition_fragment_B(sB);
    auto smem_copy_B = cute::make_tiled_copy_B(
        cute::Copy_Atom<cute::DefaultCopy, BF16>{}, mma);
    auto thr_copy_B = smem_copy_B.get_thread_slice(lane);
    auto sB_part = thr_copy_B.partition_S(sB);
    auto rB_retile = thr_copy_B.retile_D(rB_full);
    CUTE_UNROLL
    for (int k = 0; k < cute::size<2>(rB_full); ++k)
        cute::copy(smem_copy_B, sB_part(_, _, k), rB_retile(_, _, k));
    Tensor packed = recast<uint32_t>(rB_full);
    CUTE_STATIC_ASSERT_V(cute::size(packed) == cute::Int<4>{});
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) regs[i] = packed(i);
}

// One warp owns a fixed N tile in both recurrence GEMMs.  Keeping multiple M
// accumulators live lets every contraction step load/transpose the shared B
// fragment once, then pair it with each independent A fragment.  These
// epilogues retain the original partition_C ownership, FP32 expression order,
// and single BF16 materialization for each logical output element.
template <typename Acc>
__device__ __forceinline__ void recurrence_store_x_acc_tc01(
    Acc& acc, BF16* x, int mb, int nb, int lane) {
    auto Xtile = make_tensor(
        make_smem_ptr(x + mb * BC * BV + nb * BC),
        make_layout(make_shape(_16{}, _16{}),
                    make_stride(Int<BV>{}, _1{})));
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    Tensor tCsX = thr_mma.partition_C(Xtile);
    Tensor rC = recast<float>(acc);
    CUTE_STATIC_ASSERT_V(cute::size<0>(tCsX) == cute::Int<8>{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCsX) == cute::Int<1>{});
    CUTE_STATIC_ASSERT_V(cute::size<2>(tCsX) == cute::Int<1>{});
    BF16 prior[8];
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) prior[v] = tCsX(v, 0, 0);
    // Only constrains compiler scheduling: issue all independent TSM reads first, without adding device synchronization.
    asm volatile("" ::: "memory");
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
        const float u = fk::bf16_to_f32(prior[v]);
        tCsX(v, 0, 0) = BF16(u - rC(v, 0, 0));
    }
}

template <typename Acc>
__device__ __forceinline__ void recurrence_store_state_acc_tc01(
    Acc& acc, BF16* state, float* gate, int mb, int nb, int lane) {
    auto StateTile = make_tensor(
        make_smem_ptr(state + mb * BC * BV + nb * BC),
        make_layout(make_shape(_16{}, _16{}),
                    make_stride(Int<BV>{}, _1{})));
    auto GateTile = make_tensor(
        make_smem_ptr(gate + mb * BC),
        make_layout(make_shape(_16{}, _16{}),
                    make_stride(_1{}, _0{})));
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    Tensor tCsState = thr_mma.partition_C(StateTile);
    Tensor tCsGate = thr_mma.partition_C(GateTile);
    Tensor rC = recast<float>(acc);
    CUTE_STATIC_ASSERT_V(cute::size<0>(tCsState) == cute::Int<8>{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(tCsState) == cute::Int<1>{});
    CUTE_STATIC_ASSERT_V(cute::size<2>(tCsState) == cute::Int<1>{});
    BF16 prior[8];
    float gates[8];
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
        prior[v] = tCsState(v, 0, 0);
        gates[v] = tCsGate(v, 0, 0);
    }
    // The prior/gate registers live only within this epilogue; state is still rounded at the original position.
    asm volatile("" ::: "memory");
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
        const float old = fk::bf16_to_f32(prior[v]);
        tCsState(v, 0, 0) = BF16(old * gates[v] + rC(v, 0, 0));
    }
}
#endif  // FLASHKDA_ARCH_100

struct alignas(128) RecurrenceSmem {
    // Internal state is logical [DK,BV]. External/checkpoint images are [DV,DK].
    BF16 state[DK * BV];
    BF16 matrix[BT * DK];       // W, then Kg
    BF16 x[BT * BV];
    float gate[DK];
};
static_assert(sizeof(RecurrenceSmem) <= 64 * 1024,
              "tc01 BT64 recurrence shared memory exceeds 64 KiB");
static_assert(sizeof(RecurrenceSmem) == 23040,
              "tc01 BT64 register-fused recurrence TSM budget drifted");
#if defined(FLASHKDA_ARCH_100)
static_assert(offsetof(RecurrenceSmem, x) % alignof(uint4) == 0,
              "tc01 recurrence sm.x must remain 16-byte aligned");
#endif

// ---- shared recurrence pieces -------------------------------------------
// The plain whole-sequence kernel and the CP-composed segment kernel run the
// same per-BT64-block mainloop; keep one body so the two cannot drift.

__device__ __forceinline__ void bt64_recurrence_init_state(
    RecurrenceSmem& sm, const void* init_state, bool init_state_is_fp32,
    bool has_state_in, int64_t slot_head_off, int dv_base, int tid) {
    for (int i = tid; i < DK * BV; i += THREADS) {
        const int dk = i / BV;
        const int lc = i - dk * BV;
        const int dv = dv_base + lc;
        BF16 value = BF16(0.f);
        if (has_state_in && init_state != nullptr) {
            const int64_t off = slot_head_off + int64_t(dv) * DK + dk;
            value = init_state_is_fp32
                ? BF16(reinterpret_cast<const float*>(init_state)[off])
                : reinterpret_cast<const BF16*>(init_state)[off];
        }
        sm.state[i] = value;
    }
}

struct Bt64RecurrenceCtx {
    const BF16* ws16;
    const BF16* bt_ws;
    BF16* x_global;
    BF16* h_checkpoint;
    int T_total;
    int H;
    int total_tiles;
    int total_bt;
    int head;
    int dv_base;
};

// Runs blocks [bt_base, bt_base+nbt): tok0/seq_len/ws_base0 describe the
// first block's absolute token, the owned range's token length, and its CH16
// workspace base.  The plain kernel's range is a whole sequence
// (map_sequence); the CP-composed kernel's range is one BT64-aligned segment.
__device__ __forceinline__ void bt64_recurrence_block_range(
    const Bt64RecurrenceCtx& ctx, RecurrenceSmem& sm,
    int tok0, int seq_len, int bt_base, int ws_base0, int nbt,
    int tid, int warp, int lane) {
    const BF16* ws16 = ctx.ws16;
    const BF16* bt_ws = ctx.bt_ws;
    BF16* x_global = ctx.x_global;
    BF16* h_checkpoint = ctx.h_checkpoint;
    const int T_total = ctx.T_total;
    const int H = ctx.H;
    const int total_tiles = ctx.total_tiles;
    const int total_bt = ctx.total_bt;
    const int head = ctx.head;
    const int dv_base = ctx.dv_base;
    constexpr int NWARPS = THREADS / WARP;
    for (int local_bt = 0; local_bt < nbt; ++local_bt) {
        const int gbt = bt_base + local_bt;
        if (gbt >= total_bt) break;
        const int tok = tok0 + local_bt * BT;
        const int remain = seq_len - local_bt * BT;
        const int bt_len = remain < BT ? remain : BT;
        const int nsub = ceil_div_pos(bt_len, BC);
        const int ws_base = ws_base0 + local_bt * NS;
        const BF16* bt_slot = bt_ws
            + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;

        // Output consumes the state at the start of this BT. Store exactly the
        // BF16 image used by this recurrence, transposed to [DV,DK].
        for (int i = tid; i < DK * BV; i += THREADS) {
            const int dk = i / BV;
            const int lc = i - dk * BV;
            const int dv = dv_base + lc;
            const int64_t off =
                (int64_t(gbt * H + head) * DV + dv) * DK + dk;
            h_checkpoint[off] = sm.state[i];
        }

        // x = BF16(FP32(U) - FP32(W @ state)).  Stage the already-BF16 U
        // image directly in x, then let each MMA C-fragment owner apply the
        // FP32 subtraction and the single BF16 materialization in place.  The
        // partition_C view is the selected tc01 atom's ownership map; no
        // architecture-specific lane coordinates are assumed here.
#if defined(FLASHKDA_ARCH_100)
        if (tid == 0) {
            recurrence_aiu_load_w_tc01(
                sm.matrix, bt_slot + fk::BT64_WS_OFF_W_U);
            recurrence_aiu_commit_tc01();
        }
#else
        for (int i = tid; i < BT * DK; i += THREADS)
            sm.matrix[i] = bt_slot[fk::BT64_WS_OFF_W_U + i];
#endif
#if defined(FLASHKDA_ARCH_100)
        recurrence_copy_u_vec16_tc01(sm.x, bt_slot, dv_base, tid);
#else
        for (int i = tid; i < BT * BV; i += THREADS) {
            const int r = i / BV;
            const int lc = i - r * BV;
            sm.x[i] = bt_slot[
                fk::BT64_WS_OFF_U_U + r * DV + dv_base + lc];
        }
#endif
#if defined(FLASHKDA_ARCH_100)
        // Overlap W's disjoint AIU transfer with the vec16 U copy above,
        // then publish both arrays with the original block barrier.
        if (tid == 0) recurrence_aiu_wait_all_tc01();
#endif
        __syncthreads();

        {
#if !defined(FLASHKDA_ARCH_100)
            auto Wv = make_tensor(make_smem_ptr(sm.matrix),
                make_layout(make_shape(Int<BT>{}, Int<DK>{}),
                            make_stride(Int<DK>{}, _1{})));
#endif
            auto StateBv = make_tensor(make_smem_ptr(sm.state),
                make_layout(make_shape(Int<BV>{}, Int<DK>{}),
                            make_stride(_1{}, Int<BV>{})));
            constexpr int X_MB = BT / BC;
            constexpr int X_NB = BV / BC;
            constexpr int X_KB = DK / BC;
#if defined(FLASHKDA_ARCH_100)
            static_assert(NWARPS == 4 && X_MB == 4 && X_NB == 1,
                          "tc01 BV16 W@state needs four warps and a 4x1 tile");
            // Each warp owns one M tile; the K accumulation order of every element is preserved.
            const int nb = 0;
            const int mb = warp;
            auto acc = fk::make_acc16_bf16_1warp(lane);
            cute::clear(acc);
            CUTE_UNROLL
            for (int kb = 0; kb < X_KB; ++kb) {
                uint32_t b_regs[4];
                recurrence_load_b_vec32_shuffle_tc01(
                    sm.state, nb, kb, lane, b_regs);
                uint32_t a_regs[4];
                recurrence_load_w_nontrans_tc01(
                    sm.matrix, mb * BC, kb * BC, a_regs);
                fk::gemm16_rr_1warp_bf16_acc(a_regs, b_regs, acc);
            }
            recurrence_store_x_acc_tc01(acc, sm.x, mb, nb, lane);
#else
            for (int blk = warp; blk < X_MB * X_NB; blk += NWARPS) {
                const int mb = blk / X_NB;
                const int nb = blk - mb * X_NB;
                auto acc = fk::make_acc16_bf16_1warp(lane);
                cute::clear(acc);
                CUTE_UNROLL
                for (int kb = 0; kb < X_KB; ++kb) {
                    auto Ablk = local_tile(
                        Wv, make_shape(_16{}, _16{}), make_coord(mb, kb));
                    auto Bblk = local_tile(
                        StateBv, make_shape(_16{}, _16{}), make_coord(nb, kb));
                    fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
                }

                auto Xtile = make_tensor(
                    make_smem_ptr(sm.x + mb * BC * BV + nb * BC),
                    make_layout(make_shape(_16{}, _16{}),
                                make_stride(Int<BV>{}, _1{})));
                fk::TiledMma1WarpBf16 mma;
                auto thr_mma = mma.get_thread_slice(lane);
                Tensor tCsX = thr_mma.partition_C(Xtile);
                Tensor rC = recast<float>(acc);
                CUTE_UNROLL
                for (int m = 0; m < cute::size<1>(tCsX); ++m)
                    CUTE_UNROLL
                    for (int n = 0; n < cute::size<2>(tCsX); ++n)
                        CUTE_UNROLL
                        for (int v = 0; v < cute::size<0>(tCsX); ++v) {
                            const float u = fk::bf16_to_f32(tCsX(v, m, n));
                            tCsX(v, m, n) = BF16(u - rC(v, m, n));
                        }
            }
#endif
        }
        __syncthreads();
        for (int i = tid; i < BT * BV; i += THREADS) {
            const int r = i / BV;
            const int lc = i - r * BV;
            if (r < bt_len && tok + r < T_total) {
                const int64_t off = int64_t(tok + r) * H * DV
                    + int64_t(head) * DV + dv_base + lc;
                x_global[off] = sm.x[i];
            }
        }
        __syncthreads();

        // state = diag(product gt) * state + Kg^T @ x.  On tc01, issue the
        // one new native Kg transfer before the unchanged gate loop and wait
        // after it, so the asynchronous copy can overlap only existing scalar
        // work.  W/state/U/K-loop/MMA order and BF16 materialization are kept.
#if defined(FLASHKDA_ARCH_100)
        if (tid == 0) {
            recurrence_aiu_load_w_tc01(
                sm.matrix, bt_slot + fk::BT64_WS_OFF_KG_U);
            recurrence_aiu_commit_tc01();
        }
#endif
        for (int dk = tid; dk < DK; dk += THREADS) {
#if defined(FLASHKDA_ARCH_100)
            // Four independent reads precede the multiply that consumes them; the tail block only touches CH16 slots that actually exist.
            const float g0 = reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base) + fk::WS_OFF_GT_U)[dk];
            const float g1 = nsub > 1 ? reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base + 1) + fk::WS_OFF_GT_U)[dk] : 1.f;
            const float g2 = nsub > 2 ? reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base + 2) + fk::WS_OFF_GT_U)[dk] : 1.f;
            const float g3 = nsub > 3 ? reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base + 3) + fk::WS_OFF_GT_U)[dk] : 1.f;
            float g = 1.f;
            g *= g0;
            g *= g1;
            g *= g2;
            g *= g3;
#else
            float g = 1.f;
            for (int sub = 0; sub < nsub; ++sub) {
                const BF16* slot = k1_slot(
                    ws16, total_tiles, head, ws_base + sub);
                g *= reinterpret_cast<const float*>(
                    slot + fk::WS_OFF_GT_U)[dk];
            }
#endif
            sm.gate[dk] = g;
        }
#if defined(FLASHKDA_ARCH_100)
        if (tid == 0) recurrence_aiu_wait_all_tc01();
#else
        for (int i = tid; i < BT * DK; i += THREADS)
            sm.matrix[i] = bt_slot[fk::BT64_WS_OFF_KG_U + i];
#endif
        __syncthreads();
        {
#if !defined(FLASHKDA_ARCH_100)
            auto KgTv = make_tensor(make_smem_ptr(sm.matrix),
                make_layout(make_shape(Int<DK>{}, Int<BT>{}),
                            make_stride(_1{}, Int<DK>{})));
#endif
            auto Xbv = make_tensor(make_smem_ptr(sm.x),
                make_layout(make_shape(Int<BV>{}, Int<BT>{}),
                            make_stride(_1{}, Int<BV>{})));
            constexpr int STATE_MB = DK / BC;
            constexpr int STATE_NB = BV / BC;
            constexpr int STATE_KB = BT / BC;
#if defined(FLASHKDA_ARCH_100)
            static_assert(NWARPS == 4 && STATE_MB == 8 && STATE_NB == 1,
                          "tc01 BV16 state needs four warps and an 8x1 tile");
            const int nb = 0;
            constexpr int STATE_GROUPS = 1;
            // Keep the two groups sequential in the generated program so the
            // second pair cannot extend the first pair's accumulator liveness.
            // This is the occupancy-control variable of partial2.
#pragma unroll 1
            for (int group = 0; group < STATE_GROUPS; ++group) {
                const int mb0 = warp + group * 8;
                const int mb1 = mb0 + 4;
                auto acc0 = fk::make_acc16_bf16_1warp(lane);
                auto acc1 = fk::make_acc16_bf16_1warp(lane);
                cute::clear(acc0);
                cute::clear(acc1);
                CUTE_UNROLL
                for (int kb = 0; kb < STATE_KB; ++kb) {
                    uint32_t b_regs[4];
                    recurrence_load_b_vec32_shuffle_tc01(
                        sm.x, nb, kb, lane, b_regs);
                    uint32_t a_regs[4];
                    recurrence_load_kg_trans_tc01(
                        sm.matrix, mb0 * BC, kb * BC, a_regs);
                    fk::gemm16_rr_1warp_bf16_acc(
                        a_regs, b_regs, acc0);
                    recurrence_load_kg_trans_tc01(
                        sm.matrix, mb1 * BC, kb * BC, a_regs);
                    fk::gemm16_rr_1warp_bf16_acc(
                        a_regs, b_regs, acc1);
                }
                recurrence_store_state_acc_tc01(
                    acc0, sm.state, sm.gate, mb0, nb, lane);
                recurrence_store_state_acc_tc01(
                    acc1, sm.state, sm.gate, mb1, nb, lane);
            }
#else
            for (int blk = warp;
                 blk < STATE_MB * STATE_NB; blk += NWARPS) {
                const int mb = blk / STATE_NB;
                const int nb = blk - mb * STATE_NB;
                auto acc = fk::make_acc16_bf16_1warp(lane);
                cute::clear(acc);
                CUTE_UNROLL
                for (int kb = 0; kb < STATE_KB; ++kb) {
                    auto Ablk = local_tile(
                        KgTv, make_shape(_16{}, _16{}), make_coord(mb, kb));
                    auto Bblk = local_tile(
                        Xbv, make_shape(_16{}, _16{}), make_coord(nb, kb));
                    fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
                }

                auto StateTile = make_tensor(
                    make_smem_ptr(sm.state + mb * BC * BV + nb * BC),
                    make_layout(make_shape(_16{}, _16{}),
                                make_stride(Int<BV>{}, _1{})));
                auto GateTile = make_tensor(
                    make_smem_ptr(sm.gate + mb * BC),
                    make_layout(make_shape(_16{}, _16{}),
                                make_stride(_1{}, _0{})));
                fk::TiledMma1WarpBf16 mma;
                auto thr_mma = mma.get_thread_slice(lane);
                Tensor tCsState = thr_mma.partition_C(StateTile);
                Tensor tCsGate = thr_mma.partition_C(GateTile);
                Tensor rC = recast<float>(acc);
                CUTE_UNROLL
                for (int m = 0; m < cute::size<1>(tCsState); ++m)
                    CUTE_UNROLL
                    for (int n = 0; n < cute::size<2>(tCsState); ++n)
                        CUTE_UNROLL
                        for (int v = 0; v < cute::size<0>(tCsState); ++v) {
                            const float old =
                                fk::bf16_to_f32(tCsState(v, m, n));
                            tCsState(v, m, n) = BF16(
                                old * tCsGate(v, m, n) + rC(v, m, n));
                        }
            }
#endif
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void bt64_recurrence_store_state(
    const RecurrenceSmem& sm, void* final_state, bool final_state_is_fp32,
    int64_t slot_head_off, int dv_base, int tid) {
    for (int i = tid; i < DK * BV; i += THREADS) {
        const int dk = i / BV;
        const int lc = i - dk * BV;
        const int dv = dv_base + lc;
        const int64_t off = slot_head_off + int64_t(dv) * DK + dk;
        if (final_state_is_fp32) {
            reinterpret_cast<float*>(final_state)[off] =
                fk::bf16_to_f32(sm.state[i]);
        } else {
            reinterpret_cast<BF16*>(final_state)[off] = sm.state[i];
        }
    }
}

__global__ void __launch_bounds__(THREADS, 1) flashkda_bt64_recurrence_kernel(
    const BF16* __restrict__ ws16,
    const BF16* __restrict__ bt_ws,
    BF16* __restrict__ x_global,
    BF16* __restrict__ h_checkpoint,
    const void* __restrict__ init_state,
    void* __restrict__ final_state,
    const int32_t* __restrict__ init_state_indices,
    const int32_t* __restrict__ final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* __restrict__ cu,
    bool has_state_in, bool has_state_out) {
    const int dv_tile = int(blockIdx.x);
    const int seq = int(blockIdx.y);
    const int head = int(blockIdx.z);
    const int tid = int(threadIdx.x);
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    if (seq >= N) return;
    const int dv_base = dv_tile * BV;
    const SeqMap smap = map_sequence(seq, cu);

    extern __shared__ __align__(128) unsigned char smem_raw[];
    RecurrenceSmem& sm = *reinterpret_cast<RecurrenceSmem*>(smem_raw);

    const int init_slot = init_state_indices ? init_state_indices[seq] : seq;
    bt64_recurrence_init_state(
        sm, init_state, init_state_is_fp32, has_state_in,
        (int64_t(init_slot) * H + head) * DV * DK, dv_base, tid);
    __syncthreads();

    const Bt64RecurrenceCtx ctx{ws16, bt_ws, x_global, h_checkpoint,
        T_total, H, total_tiles, total_bt, head, dv_base};
    bt64_recurrence_block_range(
        ctx, sm, smap.tok, smap.len, smap.bt_base, smap.ws_base, smap.nbt,
        tid, warp, lane);

    if (has_state_out && final_state != nullptr) {
        const int final_slot = final_state_indices
            ? final_state_indices[seq] : seq;
        bt64_recurrence_store_state(
            sm, final_state, final_state_is_fp32,
            (int64_t(final_slot) * H + head) * DV * DK, dv_base, tid);
    }
}

// CP-composed recurrence: blockIdx.y enumerates BT64-aligned segments instead
// of whole sequences.  Each descriptor in seg_bt carries the segment's
// absolute first token, token length, absolute CH16 workspace base, absolute
// BT64 block base (an independent ceil(seq_len/64) scan that cannot be derived
// from the CH16 prefix), and the owning sequence's state-out slot (-1 when the
// segment is not a sequence tail).  The init state is the bf16 carry image
// produced by the CP merge stage, stored in the same [slot, head, dv, dk]
// layout the plain kernel's init path uses, and h_checkpoint / x_global keep
// their absolute indexing so segments stay disjoint for the downstream
// prepare_qg/output stages.
__global__ void __launch_bounds__(THREADS, 1)
flashkda_bt64_recurrence_segment_kernel(
    const BF16* __restrict__ ws16,
    const BF16* __restrict__ bt_ws,
    BF16* __restrict__ x_global,
    BF16* __restrict__ h_checkpoint,
    const BF16* __restrict__ carries,
    void* __restrict__ final_state,
    const int32_t* __restrict__ final_state_indices,
    bool final_state_is_fp32,
    int T_total, int H, int total_tiles, int total_bt,
    const int32_t* __restrict__ seg_bt, int n_seg,
    bool has_state_out) {
    const int dv_tile = int(blockIdx.x);
    const int seg = int(blockIdx.y);
    const int head = int(blockIdx.z);
    const int tid = int(threadIdx.x);
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    if (seg >= n_seg) return;
    const int dv_base = dv_tile * BV;
    const int32_t* desc = seg_bt + int64_t(seg) * fk::KDA_SEG_BT_STRIDE;
    const int tok0 = desc[fk::KDA_SEG_BT_TOK];
    const int seq_len = desc[fk::KDA_SEG_BT_LEN];
    const int ws_base0 = desc[fk::KDA_SEG_BT_WSBASE];
    const int bt_base = desc[fk::KDA_SEG_BT_BTBASE];
    const int outslot = desc[fk::KDA_SEG_BT_OUTSLOT];
    const int nbt = ceil_div_pos(seq_len, BT);

    extern __shared__ __align__(128) unsigned char smem_raw[];
    RecurrenceSmem& sm = *reinterpret_cast<RecurrenceSmem*>(smem_raw);

    bt64_recurrence_init_state(
        sm, carries, /*init_state_is_fp32=*/false, /*has_state_in=*/true,
        (int64_t(seg) * H + head) * DV * DK, dv_base, tid);
    __syncthreads();

    const Bt64RecurrenceCtx ctx{ws16, bt_ws, x_global, h_checkpoint,
        T_total, H, total_tiles, total_bt, head, dv_base};
    bt64_recurrence_block_range(
        ctx, sm, tok0, seq_len, bt_base, ws_base0, nbt, tid, warp, lane);

    if (has_state_out && final_state != nullptr && outslot >= 0) {
        const int final_slot = final_state_indices
            ? final_state_indices[outslot] : outslot;
        bt64_recurrence_store_state(
            sm, final_state, final_state_is_fp32,
            (int64_t(final_slot) * H + head) * DV * DK, dv_base, tid);
    }
}

#include "bt64_glm_n1_recurrence.cuh"

// output-v4 execution-domain split.  The recurrence launch immediately before
// this kernel has consumed W for every (sequence, head, dv-tile).  CUDA stream
// ordering therefore makes the complete W_OR_A region dead before any CTA here
// overwrites it with Qg.  The region is exactly one [BT,DK] BF16 matrix.
//
// This source body deliberately duplicates the v3 output kernel's scalar Qg
// expression rather than reassociating or precomputing gate products:
//   BF16 Qd -> FP32, sequential FP32 prefix, FP32 multiply, one BF16 round.
// Keeping the zero-MMA work in a separate function is what allows the exact
// binary to qualify for tc01's generic 64-CU execution domain (mma_en=0).
static_assert(fk::BT64_WS_OFF_W_OR_A_U == fk::BT64_WS_OFF_W_U,
              "Qg handoff must alias the dead W image");
static_assert(fk::BT64_W_UNITS == BT * DK,
              "dead W image must exactly hold Qg[BT,DK]");

__global__ void __launch_bounds__(OUTPUT_THREADS, 1)
flashkda_bt64_prepare_qg_kernel(
    const BF16* __restrict__ ws16,
    BF16* __restrict__ bt_ws,
    int H, int N, int total_tiles, int total_bt,
    const int32_t* __restrict__ cu) {
    const int gbt = int(blockIdx.x);
    const int head = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int nsub = ceil_div_pos(map.len, BC);
    BF16* qg_out = bt_ws
        + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS
        + fk::BT64_WS_OFF_W_OR_A_U;

    // Qg is relative to this BT start.  K1 Qd is already BF16-rounded after
    // the local CH16 gate/scale conversion.  Multiply only complete preceding
    // CH16 gt products, in the exact v3 left-to-right FP32 order.  The explicit
    // sub/nsub and row/map.len guards make a partial BT synthesize zeros and
    // never read the next packed sequence's first K1 workspace slot.
    for (int i = tid; i < BT * DK; i += OUTPUT_THREADS) {
        const int r = i / DK;
        const int dk = i - r * DK;
        const int sub = r / BC;
        const int row = r - sub * BC;
        float qg = 0.f;
        if (sub < nsub && r < map.len) {
            const BF16* slot = k1_slot(
                ws16, total_tiles, head, map.ws_base + sub);
            qg = fk::bf16_to_f32(slot[fk::WS_OFF_QD_U + row * DK + dk]);
            float prefix = 1.f;
            for (int p = 0; p < sub; ++p) {
                const BF16* prev = k1_slot(
                    ws16, total_tiles, head, map.ws_base + p);
                prefix *= reinterpret_cast<const float*>(
                    prev + fk::WS_OFF_GT_U)[dk];
            }
            qg *= prefix;
        }
        qg_out[i] = BF16(qg);
    }
}

#if defined(FLASHKDA_ARCH_100)
// -------------------------------------------------------------------------
// output-v3 base: tc01-native operand staging/load.
//
// The producer and consumer form one closed contract:
//
//   PPU0010_AIU_LOAD<Swzl=true>
//     -> AIU-native 8xCONT cube layout
//     -> PPU0010_TSM_LD_SWZL
//     -> PPU0010 BF16 MMA A/B registers.
//
// Output-v4 adds Qg to the same closed native producer/consumer contract after
// the separate prepare kernel has materialized its row-major global image.
// Qg/state/Aqk are non-transpose operands.  X is written in its natural
// [token, dv] order and read with the tc01 Trans=true fragment instruction,
// yielding the logical B=[dv, token] tile without a software transpose.

template <int Rows, int Cols>
struct OutputOperandTileTc01 {
    using CuteElem = cute::bfloat16_t;
    static constexpr int kRows = Rows;
    static constexpr int kCols = Cols;
    static constexpr int kContBytes =
        Cols * int(sizeof(CuteElem)) > 128
            ? 128 : Cols * int(sizeof(CuteElem));
    static constexpr int kCont = kContBytes / int(sizeof(CuteElem));
    static constexpr int kInst = Cols / kCont;
    static constexpr int kBitsPerAiu =
        Rows * kCont * int(sizeof(CuteElem)) * 8;

    static_assert(sizeof(CuteElem) == sizeof(BF16),
                  "tc01 native BF16 storage width mismatch");
    static_assert(Rows % 16 == 0 && Cols % 16 == 0,
                  "output operands must be 16x16 MMA tiled");
    static_assert(Cols % kCont == 0,
                  "tc01 AIU 128-byte slices must exactly tile K");

    using CopyInst = cute::PPU0010_AIU_LOAD<
        cute::C<kBitsPerAiu>, CuteElem,
        /*Trans=*/false, /*Swzl=*/true>;
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<CopyInst, CuteElem>{},
        cute::Layout<cute::Shape<cute::_1, cute::_1>,
                     cute::Stride<cute::_1, cute::_1>>{},
        cute::Layout<cute::Shape<cute::Int<Rows>,
                                  cute::Int<kCont>>>{}));

    using SmemLayoutAtom = cute::Layout<
        cute::Shape<cute::_8, cute::Int<kCont>>,
        cute::Stride<cute::Int<kCont>, cute::_1>>;
    using SmemLayout = decltype(cute::tile_to_shape(
        SmemLayoutAtom{},
        cute::make_shape(cute::Int<Rows>{}, cute::Int<Cols>{})));

    using NonTransLoad = cute::PPU0010_TSM_LD_SWZL<
        CuteElem, Rows, kCont, /*Swap=*/false, /*Trans=*/false, kInst>;
    using TransLoad = cute::PPU0010_TSM_LD_SWZL<
        CuteElem, Rows, kCont, /*Swap=*/false, /*Trans=*/true, kInst>;
};

using OutputStateTileTc01 = OutputOperandTileTc01<OUTPUT_BV, DK>;
using OutputSquareTileTc01 = OutputOperandTileTc01<BT, BT>;

// The shipped tc01 atom declares byte-identical A/B ownership.  This is the
// reason a direct four-register TSM fragment can be fed to either MMA operand;
// the accompanying basis verifier enumerates all 256 logical coordinates.
using OutputTc01MmaTraits =
    cute::MMA_Traits<flashkda::PpuBf16MMA>;
static_assert(std::is_same<typename OutputTc01MmaTraits::ALayout,
                           typename OutputTc01MmaTraits::BLayout>::value,
              "tc01 output-v3 requires identical A/B fragment ownership");

template <typename Traits>
__device__ __forceinline__ void output_aiu_load_tc01(
    BF16* dst, const BF16* src, int row_stride_elems, int valid_rows) {
    using CuteElem = typename Traits::CuteElem;
    auto gA = cute::make_mix_tensor(
        cute::make_gmem_ptr(reinterpret_cast<const CuteElem*>(src)),
        cute::make_layout(
            cute::make_shape(cute::Int<Traits::kRows>{},
                             cute::Int<Traits::kCols>{}),
            cute::make_counting_stride<2, 1, 1>()));
    auto sA = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<CuteElem*>(dst)),
        typename Traits::SmemLayout{});
    typename Traits::GmemTiledCopy tiled_copy;
    tiled_copy.desc_.template init<CuteElem, /*Trans=*/false,
                                    Traits::kRows, Traits::kCols>(
        reinterpret_cast<const uint8_t*>(src),
        /*MN=*/valid_rows, /*K=*/Traits::kCols,
        cute::make_stride(row_stride_elems, cute::_1{}));
    auto thr = tiled_copy.get_slice(0);
    auto tAgA = thr.partition_S(gA);
    auto tAsA = thr.partition_D(sA);
    cute::copy(tiled_copy, tAgA, tAsA);
}

__device__ __forceinline__ void output_aiu_commit_tc01() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void output_aiu_wait_all_tc01() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void* output_smem_base_tc01(const BF16* ptr) {
#if defined(__HGGC_ARCH__)
    const uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(ptr));
    return reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
#else
    return const_cast<BF16*>(ptr);
#endif
}

// Load one logical non-transpose 16x16 fragment.  coord_col is reduced within
// the 128-byte channel cube; cube_in_stage selects the K cube.  Copy_Atom's
// DstLayout is four packed b32 values per lane, exactly the tc01 A/B fragment
// order checked above and by verify_basis_layout.py.
template <typename Traits>
__device__ __forceinline__ void output_load_nontrans_tc01(
    const BF16* src, int coord_row, int coord_col, uint32_t regs[4]) {
    using Load = typename Traits::NonTransLoad;
    const int cube = coord_col / Traits::kCont;
    const int col_in_cube = coord_col - cube * Traits::kCont;
    Load::copy(
        regs, output_smem_base_tc01(src),
        /*coord_w=*/col_in_cube, /*coord_h=*/coord_row,
        /*cube_in_stage=*/cube, /*stage=*/0);
}

// X remains physically [token, dv].  The Trans=true reader returns the
// logical [dv, token] B fragment.  X is 64x64, so no extra channel cube exists.
template <typename Traits>
__device__ __forceinline__ void output_load_trans_tc01(
    const BF16* src, int physical_row, int physical_col, uint32_t regs[4]) {
    using Load = typename Traits::TransLoad;
    static_assert(Traits::kInst == 1,
                  "trans output helper currently supports one channel cube");
    Load::copy(
        regs, output_smem_base_tc01(src),
        /*coord_h=*/physical_row, /*coord_w=*/physical_col,
        /*cube_in_stage=*/0, /*stage=*/0);
}
#endif  // FLASHKDA_ARCH_100

template <typename Acc>
__device__ __forceinline__ void store_output_vec(
    const Acc& acc, int lane, int actual,
    BF16* warp_scratch, BF16* o_head, int64_t row_stride) {
    // tc01 exposes the MMA C fragment in registers but the existing helper ABI
    // only materializes it to TSM.  Keep that boundary to one BF16 16x16 tile
    // per warp, then issue two aligned 16-byte global stores for every row.
    fk::store_acc16_bf16mma<BF16>(acc, warp_scratch, BC, lane);
    __syncwarp();
    if (lane < 2 * actual) {
        const int row = lane >> 1;
        const int half = lane & 1;
        *reinterpret_cast<uint4*>(
            o_head + int64_t(row) * row_stride + half * 8) =
            *reinterpret_cast<const uint4*>(
                warp_scratch + row * BC + half * 8);
    }
    __syncwarp();
}

__device__ __forceinline__ void output_gemm_chain_and_store(
    const BF16* qg, const BF16* state, const BF16* Aqk, const BF16* x,
    BF16* output_scratch, BF16* o, int map_tok, int map_len,
    int head, int dv_base, int H, int tid) {
    constexpr int MB = BT / BC;
    constexpr int NB = OUTPUT_BV / BC;
    constexpr int STATE_KB = DK / BC;
    constexpr int INTRA_KB = BT / BC;
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    constexpr int NWARPS = OUTPUT_THREADS / WARP;

#if !defined(FLASHKDA_ARCH_100)
    auto Qg = cute::make_tensor(
        cute::make_smem_ptr(const_cast<BF16*>(qg)),
        cute::make_layout(
            cute::make_shape(cute::Int<BT>{}, cute::Int<DK>{}),
            cute::make_stride(cute::Int<DK>{}, cute::_1{})));
    auto State = cute::make_tensor(
        cute::make_smem_ptr(const_cast<BF16*>(state)),
        cute::make_layout(
            cute::make_shape(cute::Int<OUTPUT_BV>{}, cute::Int<DK>{}),
            cute::make_stride(cute::_1{}, cute::Int<OUTPUT_BV>{})));
    auto AqkView = cute::make_tensor(
        cute::make_smem_ptr(const_cast<BF16*>(Aqk)),
        cute::make_layout(
            cute::make_shape(cute::Int<BT>{}, cute::Int<BT>{}),
            cute::make_stride(cute::Int<BT>{}, cute::_1{})));
    auto X = cute::make_tensor(
        cute::make_smem_ptr(const_cast<BF16*>(x)),
        cute::make_layout(
            cute::make_shape(cute::Int<OUTPUT_BV>{}, cute::Int<BT>{}),
            cute::make_stride(cute::_1{}, cute::Int<OUTPUT_BV>{})));
#endif

    for (int blk = warp; blk < MB * NB; blk += NWARPS) {
        const int mb = blk / NB;
        const int nb = blk - mb * NB;

        // Match FLA's chained-dot dataflow: qg @ state initializes the FP32
        // register fragment and Aqk @ x accumulates directly into it.  This
        // removes both full FP32 result arrays and the second C fragment.
        auto acc = fk::make_acc16_bf16_1warp(lane);
        cute::clear(acc);
        CUTE_UNROLL
        for (int kb = 0; kb < STATE_KB; ++kb) {
#if defined(FLASHKDA_ARCH_100)
            uint32_t a_regs[4];
            uint32_t b_regs[4];
            output_load_nontrans_tc01<OutputStateTileTc01>(
                qg, mb * BC, kb * BC, a_regs);
            output_load_nontrans_tc01<OutputStateTileTc01>(
                state, nb * BC, kb * BC, b_regs);
            fk::gemm16_rr_1warp_bf16_acc(a_regs, b_regs, acc);
#else
            auto Ablk = cute::local_tile(
                Qg, cute::make_shape(cute::_16{}, cute::_16{}),
                cute::make_coord(mb, kb));
            auto Bblk = cute::local_tile(
                State, cute::make_shape(cute::_16{}, cute::_16{}),
                cute::make_coord(nb, kb));
            fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
#endif
        }
        CUTE_UNROLL
        for (int kb = 0; kb < INTRA_KB; ++kb) {
#if defined(FLASHKDA_ARCH_100)
            uint32_t a_regs[4];
            uint32_t b_regs[4];
            output_load_nontrans_tc01<OutputSquareTileTc01>(
                Aqk, mb * BC, kb * BC, a_regs);
            // X is stored [token, dv], so physical row=K and physical col=N.
            output_load_trans_tc01<OutputSquareTileTc01>(
                x, kb * BC, nb * BC, b_regs);
            fk::gemm16_rr_1warp_bf16_acc(a_regs, b_regs, acc);
#else
            auto Ablk = cute::local_tile(
                AqkView, cute::make_shape(cute::_16{}, cute::_16{}),
                cute::make_coord(mb, kb));
            auto Bblk = cute::local_tile(
                X, cute::make_shape(cute::_16{}, cute::_16{}),
                cute::make_coord(nb, kb));
            fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
#endif
        }
        const int remain = map_len - mb * BC;
        const int actual = remain <= 0 ? 0 : (remain < BC ? remain : BC);
        if (actual > 0) {
            const int64_t row_stride = int64_t(H) * DV;
            BF16* o_head = o + int64_t(map_tok + mb * BC) * row_stride
                         + int64_t(head) * DV + dv_base + nb * BC;
            store_output_vec(
                acc, lane, actual, output_scratch + warp * BC * BC,
                o_head, row_stride);
        }
    }
}

struct alignas(128) OutputSmem {
    BF16 qg[BT * DK];
    BF16 state[DK * OUTPUT_BV];
    BF16 Aqk[BT * BT];
    BF16 x[BT * OUTPUT_BV];
    BF16 output_scratch[(OUTPUT_THREADS / WARP) * BC * BC];
};
static_assert(sizeof(OutputSmem) <= 64 * 1024,
              "tc01 BT64 output shared memory exceeds 64 KiB");

__global__ void __launch_bounds__(OUTPUT_THREADS, 1) flashkda_bt64_output_kernel(
    BF16* __restrict__ o,
    const BF16* __restrict__ ws16,
    const BF16* __restrict__ bt_ws,
    const BF16* __restrict__ x_global,
    const BF16* __restrict__ h_checkpoint,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* __restrict__ cu) {
    const int dv_tile = int(blockIdx.x);
    const int gbt = int(blockIdx.y);
    const int head = int(blockIdx.z);
    const int tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int dv_base = dv_tile * OUTPUT_BV;
    const BF16* bt_slot = bt_ws
        + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    OutputSmem& sm = *reinterpret_cast<OutputSmem*>(smem_raw);

#if defined(FLASHKDA_ARCH_100)
    // One issuer launches all four native-layout transfers as one AIU group.
    // Qg comes from the prepare kernel's row-major image in the dead W_OR_A
    // region. X is loaded in natural [token,dv] order; only its TSM consumer
    // transposes the fragment.
    if (tid == 0) {
        output_aiu_load_tc01<OutputStateTileTc01>(
            sm.qg, bt_slot + fk::BT64_WS_OFF_W_OR_A_U,
            /*row_stride_elems=*/DK, /*valid_rows=*/BT);
        const BF16* state_src = h_checkpoint
            + (int64_t(gbt * H + head) * DV + dv_base) * DK;
        output_aiu_load_tc01<OutputStateTileTc01>(
            sm.state, state_src, /*row_stride_elems=*/DK,
            /*valid_rows=*/OUTPUT_BV);
        output_aiu_load_tc01<OutputSquareTileTc01>(
            sm.Aqk, bt_slot + fk::BT64_WS_OFF_AQK_U,
            /*row_stride_elems=*/BT, /*valid_rows=*/BT);
        const BF16* x_src = x_global
            + int64_t(map.tok) * H * DV + int64_t(head) * DV + dv_base;
        output_aiu_load_tc01<OutputSquareTileTc01>(
            sm.x, x_src, /*row_stride_elems=*/H * DV,
            /*valid_rows=*/map.len);
        output_aiu_commit_tc01();
    }
#endif

#if !defined(FLASHKDA_ARCH_100)
    // The ppu1.5 image is compile coverage only (the public launcher is tc01
    // guarded). Preserve a row-major fallback for the fat-binary build.
    for (int i = tid; i < BT * DK; i += OUTPUT_THREADS)
        sm.qg[i] = bt_slot[fk::BT64_WS_OFF_W_OR_A_U + i];
    for (int i = tid; i < DK * OUTPUT_BV; i += OUTPUT_THREADS) {
        const int dk = i / OUTPUT_BV;
        const int lc = i - dk * OUTPUT_BV;
        const int dv = dv_base + lc;
        const int64_t off =
            (int64_t(gbt * H + head) * DV + dv) * DK + dk;
        sm.state[i] = h_checkpoint[off];
    }
    for (int i = tid; i < BT * BT; i += OUTPUT_THREADS)
        sm.Aqk[i] = bt_slot[fk::BT64_WS_OFF_AQK_U + i];
    for (int i = tid; i < BT * OUTPUT_BV; i += OUTPUT_THREADS) {
        const int r = i / OUTPUT_BV;
        const int lc = i - r * OUTPUT_BV;
        if (r < map.len && map.tok + r < T_total) {
            const int64_t off = int64_t(map.tok + r) * H * DV
                + int64_t(head) * DV + dv_base + lc;
            sm.x[i] = x_global[off];
        } else {
            sm.x[i] = BF16(0.f);
        }
    }
#else
    if (tid == 0) output_aiu_wait_all_tc01();
#endif
    __syncthreads();
    output_gemm_chain_and_store(
        sm.qg, sm.state, sm.Aqk, sm.x, sm.output_scratch,
        o, map.tok, map.len, head, dv_base, H, tid);
}

inline void throw_cuda_failure(
    const char* operation, hggcError_t error,
    int T_total, int H, int N, int total_bt) {
    const char* detail = hggcGetErrorString(error);
    throw std::runtime_error(
        std::string("FlashKDApro tc01 BT64 ") + operation + " failed: "
        + (detail ? detail : "unknown CUDA error")
        + " (T=" + std::to_string(T_total)
        + " H=" + std::to_string(H)
        + " N=" + std::to_string(N)
        + " total_bt=" + std::to_string(total_bt) + ")");
}

template <typename Kernel>
inline void set_dynamic_smem_or_throw(
    Kernel kernel, int bytes, const char* stage,
    int T_total, int H, int N, int total_bt) {
    const hggcError_t error = hggcFuncSetAttribute(
        kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (error != hggcSuccess)
        throw_cuda_failure(stage, error, T_total, H, N, total_bt);
}

inline void check_launch_or_throw(
    const char* stage, int T_total, int H, int N, int total_bt) {
    const hggcError_t error = hggcPeekAtLastError();
    if (error != hggcSuccess)
        throw_cuda_failure(stage, error, T_total, H, N, total_bt);
}

}  // namespace flashkda_cute_bt64
#include "fused_native_a_tc01.cuh"
#include "fused_merge_recompute_tc01.cuh"
namespace flashkda_cute_bt64 {

// Parallel front stages shared by the plain and CP-composed pipelines:
// split_pair -> split_merge (or the experimental fused merge_recompute) ->
// prepare_recompute -> recompute.  Every stage is block-parallel over
// (bt, head), so the front composes freely with either serial stage.
template <bool KrContiguous>
void launch_bt64_parallel_front(
    const BF16* v, const BF16* ws16, BF16* bt_ws, BF16* h_checkpoint_scratch,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu, hggcStream_t stream) {
    // The entry has already confirmed tc01 and a successful BT64 allocation; other shapes keep the original order.
    bool reorder = false, fused = false;
    if (!KrContiguous && H == 8 && N == 1 && T_total >= 4096) {
        const char* mode = std::getenv("PLA_FLASHKDA_TC01_BT64_FUSE");
        const std::string selected = mode && mode[0] ? mode : "off";
        if (selected != "off" && selected != "reorder" && selected != "merge_recompute")
            throw std::runtime_error("the current B experiment only supports off/reorder/merge_recompute");
        fused = selected == "merge_recompute";
        reorder = selected == "reorder" || fused;
    }
    dim3 bt_grid(total_bt, H);
    dim3 pair_grid(total_bt * SPLIT_PAIR_COUNT, H);
    set_dynamic_smem_or_throw(
        flashkda_bt64_split_pair_kernel<KrContiguous>, sizeof(SplitPairSmem),
        "split pair smem attribute", T_total, H, N, total_bt);
    flashkda_bt64_split_pair_kernel<KrContiguous>
        <<<pair_grid, THREADS, sizeof(SplitPairSmem), stream>>>(
            ws16, bt_ws, H, N, total_tiles, total_bt, cu);
    check_launch_or_throw("split pair launch", T_total, H, N, total_bt);
    if (reorder) {
        flashkda_bt64_prepare_recompute_kernel<KrContiguous>
            <<<bt_grid, RECOMPUTE_THREADS_TC01, 0, stream>>>(
                v, ws16, bt_ws, h_checkpoint_scratch,
                T_total, H, N, total_tiles, total_bt, cu);
        check_launch_or_throw("early prepare recompute launch", T_total, H, N, total_bt);
    }
    if (fused) {
        set_dynamic_smem_or_throw(
            flashkda_bt64_merge_recompute_kernel, sizeof(FusedMergeRecomputeSmem),
            "merge recompute smem attribute", T_total, H, N, total_bt);
        flashkda_bt64_merge_recompute_kernel
            <<<bt_grid, RECOMPUTE_THREADS_TC01, sizeof(FusedMergeRecomputeSmem), stream>>>(
                ws16, bt_ws, h_checkpoint_scratch, H, N, total_tiles, total_bt, cu);
        check_launch_or_throw("merge recompute launch", T_total, H, N, total_bt);
    } else {
        set_dynamic_smem_or_throw(
            flashkda_bt64_split_merge_kernel, sizeof(SplitMergeSmem),
            "split merge smem attribute", T_total, H, N, total_bt);
        flashkda_bt64_split_merge_kernel
            <<<bt_grid, THREADS, sizeof(SplitMergeSmem), stream>>>(
                ws16, bt_ws, H, N, total_tiles, total_bt, cu);
        check_launch_or_throw("split merge launch", T_total, H, N, total_bt);

        // The checkpoint is not live yet; the recurrence may overwrite this staging area only after recompute finishes.
        if (!reorder) {
            flashkda_bt64_prepare_recompute_kernel<KrContiguous>
                <<<bt_grid, RECOMPUTE_THREADS_TC01, 0, stream>>>(
                    v, ws16, bt_ws, h_checkpoint_scratch,
                    T_total, H, N, total_tiles, total_bt, cu);
            check_launch_or_throw("prepare recompute launch", T_total, H, N, total_bt);
        }

        set_dynamic_smem_or_throw(
            flashkda_bt64_recompute_kernel<KrContiguous>,
            RECOMPUTE_SMEM_BYTES_TC01,
            "recompute smem attribute", T_total, H, N, total_bt);
        flashkda_bt64_recompute_kernel<KrContiguous>
            <<<bt_grid, RECOMPUTE_THREADS_TC01,
               RECOMPUTE_SMEM_BYTES_TC01, stream>>>(
                v, ws16, bt_ws, h_checkpoint_scratch,
                T_total, H, N, total_tiles, total_bt, cu);
        check_launch_or_throw("recompute launch", T_total, H, N, total_bt);
    }
}

// Serial stage of the CP-composed pipeline: one CTA per (dv-tile, segment,
// head) replays the shared BT64 mainloop starting from the merged carry.
// n_seg fills the diagnostic N slot; the stage strings disambiguate it.
void launch_bt64_segment_recurrence(
    const BF16* ws16, const BF16* bt_ws, BF16* x, BF16* h_checkpoint,
    const BF16* carries, void* final_state,
    const int32_t* final_state_indices, bool final_state_is_fp32,
    int T_total, int H, int total_tiles, int total_bt,
    const int32_t* seg_bt, int n_seg, bool has_state_out,
    hggcStream_t stream) {
    dim3 segment_grid(DV / BV, n_seg, H);
    set_dynamic_smem_or_throw(
        flashkda_bt64_recurrence_segment_kernel, sizeof(RecurrenceSmem),
        "segment recurrence smem attribute", T_total, H, n_seg, total_bt);
    flashkda_bt64_recurrence_segment_kernel
        <<<segment_grid, THREADS, sizeof(RecurrenceSmem), stream>>>(
            ws16, bt_ws, x, h_checkpoint, carries, final_state,
            final_state_indices, final_state_is_fp32,
            T_total, H, total_tiles, total_bt, seg_bt, n_seg, has_state_out);
    check_launch_or_throw(
        "segment recurrence launch", T_total, H, n_seg, total_bt);
}

// Parallel back stages shared by both pipelines: prepare_qg -> output.
void launch_bt64_parallel_back(
    BF16* o, const BF16* ws16, BF16* bt_ws, const BF16* x,
    const BF16* h_checkpoint,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu, hggcStream_t stream) {
    dim3 bt_grid(total_bt, H);
    // Same-stream kernel completion is the global handoff: every recurrence
    // CTA has finished reading W before prepare_qg reuses W_OR_A.  Keeping this
    // scalar stage in its own exact function is required for mma_en=0/64-CU
    // qualification on tc01.
    flashkda_bt64_prepare_qg_kernel
        <<<bt_grid, OUTPUT_THREADS, 0, stream>>>(
            ws16, bt_ws, H, N, total_tiles, total_bt, cu);
    check_launch_or_throw("prepare_qg launch", T_total, H, N, total_bt);

    dim3 output_grid(DV / OUTPUT_BV, total_bt, H);
    set_dynamic_smem_or_throw(
        flashkda_bt64_output_kernel, sizeof(OutputSmem),
        "output smem attribute", T_total, H, N, total_bt);
    flashkda_bt64_output_kernel
        <<<output_grid, OUTPUT_THREADS, sizeof(OutputSmem), stream>>>(
            o, ws16, bt_ws, x, h_checkpoint,
            T_total, H, N, total_tiles, total_bt, cu);
    check_launch_or_throw("output launch", T_total, H, N, total_bt);
}

template <bool KrContiguous>
void launch_bt64_impl(
    BF16* o, const BF16* v, const BF16* ws16,
    BF16* bt_ws, BF16* x, BF16* h_checkpoint,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices,
    const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu, bool has_state_in, bool has_state_out,
    hggcStream_t stream) {
    launch_bt64_parallel_front<KrContiguous>(
        v, ws16, bt_ws, h_checkpoint,
        T_total, H, N, total_tiles, total_bt, cu, stream);

    dim3 recurrence_grid(DV / BV, N, H);
    // P1 is used only for the TP8 single-sequence long path; H4, packed and CP segments keep the original recurrence.
    if (H == 8 && N == 1 && T_total >= 256) {
        set_dynamic_smem_or_throw(
            flashkda_bt64_recurrence_n1_kernel, sizeof(RecurrenceSmem),
            "N1 recurrence smem attribute", T_total, H, N, total_bt);
        flashkda_bt64_recurrence_n1_kernel
            <<<recurrence_grid, THREADS, sizeof(RecurrenceSmem), stream>>>(
                ws16, bt_ws, x, h_checkpoint,
                init_state, final_state,
                init_state_indices, final_state_indices,
                init_state_is_fp32, final_state_is_fp32,
                T_total, H, N, total_tiles, total_bt, cu,
                has_state_in, has_state_out);
    } else {
        set_dynamic_smem_or_throw(
            flashkda_bt64_recurrence_kernel, sizeof(RecurrenceSmem),
            "recurrence smem attribute", T_total, H, N, total_bt);
        flashkda_bt64_recurrence_kernel
            <<<recurrence_grid, THREADS, sizeof(RecurrenceSmem), stream>>>(
                ws16, bt_ws, x, h_checkpoint,
                init_state, final_state,
                init_state_indices, final_state_indices,
                init_state_is_fp32, final_state_is_fp32,
                T_total, H, N, total_tiles, total_bt, cu,
                has_state_in, has_state_out);
    }
    check_launch_or_throw("recurrence launch", T_total, H, N, total_bt);

    launch_bt64_parallel_back(
        o, ws16, bt_ws, x, h_checkpoint,
        T_total, H, N, total_tiles, total_bt, cu, stream);
}

}  // namespace flashkda_cute_bt64

void launch_flashkda_bt64_prefill_tc01(
    void* o, const void* v, const void* ws16,
    void* bt_ws, void* x, void* h_checkpoint,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices,
    const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu_seqlens, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream) {
    namespace bt64 = flashkda_cute_bt64;
    using BF16 = cutlass::bfloat16_t;

    if (!flashkda::host_runtime::current_device_is_tc01()) {
        throw std::runtime_error(
            "FlashKDApro BT64 candidate is tc01-only; refusing a tc02 launch");
    }
    if (!o || !v || !ws16 || !bt_ws || !x || !h_checkpoint || !cu_seqlens) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 received a null required pointer");
    }
    if (T_total <= 0 || H <= 0 || N <= 0 || total_tiles <= 0 || total_bt <= 0) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 received invalid dimensions");
    }
    if (has_state_in && !init_state) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 has_state_in requires initial_state");
    }
    if (has_state_out && !final_state) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 has_state_out requires final_state");
    }

    if (kr_layout == flashkda::KrWorkspaceLayout::kNormalRaw) {
        bt64::launch_bt64_impl<false>(
            reinterpret_cast<BF16*>(o), reinterpret_cast<const BF16*>(v),
            reinterpret_cast<const BF16*>(ws16), reinterpret_cast<BF16*>(bt_ws),
            reinterpret_cast<BF16*>(x), reinterpret_cast<BF16*>(h_checkpoint),
            init_state, final_state, init_state_indices, final_state_indices,
            init_state_is_fp32, final_state_is_fp32,
            T_total, H, N, total_tiles, total_bt, cu_seqlens,
            has_state_in, has_state_out, stream);
        return;
    }
    if (kr_layout == flashkda::KrWorkspaceLayout::kContiguousRawKrT) {
        bt64::launch_bt64_impl<true>(
            reinterpret_cast<BF16*>(o), reinterpret_cast<const BF16*>(v),
            reinterpret_cast<const BF16*>(ws16), reinterpret_cast<BF16*>(bt_ws),
            reinterpret_cast<BF16*>(x), reinterpret_cast<BF16*>(h_checkpoint),
            init_state, final_state, init_state_indices, final_state_indices,
            init_state_is_fp32, final_state_is_fp32,
            T_total, H, N, total_tiles, total_bt, cu_seqlens,
            has_state_in, has_state_out, stream);
        return;
    }
    throw std::runtime_error(
        "FlashKDApro tc01 BT64 requires a raw K1 workspace layout");
}

void launch_flashkda_bt64_cp_front_tc01(
    const void* v, const void* ws16, void* bt_ws, void* h_checkpoint_scratch,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu_seqlens, flashkda::KrWorkspaceLayout kr_layout,
    hggcStream_t stream) {
    namespace bt64 = flashkda_cute_bt64;
    using BF16 = cutlass::bfloat16_t;

    if (!flashkda::host_runtime::current_device_is_tc01()) {
        throw std::runtime_error(
            "FlashKDApro BT64 CP candidate is tc01-only; refusing a tc02 launch");
    }
    if (!v || !ws16 || !bt_ws || !h_checkpoint_scratch || !cu_seqlens) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 CP front received a null required pointer");
    }
    if (T_total <= 0 || H <= 0 || N <= 0 || total_tiles <= 0 || total_bt <= 0) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 CP front received invalid dimensions");
    }

    if (kr_layout == flashkda::KrWorkspaceLayout::kNormalRaw) {
        bt64::launch_bt64_parallel_front<false>(
            reinterpret_cast<const BF16*>(v),
            reinterpret_cast<const BF16*>(ws16),
            reinterpret_cast<BF16*>(bt_ws),
            reinterpret_cast<BF16*>(h_checkpoint_scratch),
            T_total, H, N, total_tiles, total_bt, cu_seqlens, stream);
        return;
    }
    if (kr_layout == flashkda::KrWorkspaceLayout::kContiguousRawKrT) {
        bt64::launch_bt64_parallel_front<true>(
            reinterpret_cast<const BF16*>(v),
            reinterpret_cast<const BF16*>(ws16),
            reinterpret_cast<BF16*>(bt_ws),
            reinterpret_cast<BF16*>(h_checkpoint_scratch),
            T_total, H, N, total_tiles, total_bt, cu_seqlens, stream);
        return;
    }
    throw std::runtime_error(
        "FlashKDApro tc01 BT64 CP front requires a raw K1 workspace layout");
}

void launch_flashkda_bt64_cp_segment_tc01(
    const void* ws16, const void* bt_ws, void* x, void* h_checkpoint,
    const void* carries, void* final_state,
    const int32_t* final_state_indices, bool final_state_is_fp32,
    int T_total, int H, int total_tiles, int total_bt,
    const int32_t* seg_bt, int n_seg, bool has_state_out,
    hggcStream_t stream) {
    namespace bt64 = flashkda_cute_bt64;
    using BF16 = cutlass::bfloat16_t;

    if (!flashkda::host_runtime::current_device_is_tc01()) {
        throw std::runtime_error(
            "FlashKDApro BT64 CP candidate is tc01-only; refusing a tc02 launch");
    }
    if (!ws16 || !bt_ws || !x || !h_checkpoint || !carries || !seg_bt) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 CP segment received a null required pointer");
    }
    if (T_total <= 0 || H <= 0 || total_tiles <= 0 || total_bt <= 0
        || n_seg <= 0) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 CP segment received invalid dimensions");
    }
    if (has_state_out && !final_state) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 CP segment has_state_out requires final_state");
    }

    bt64::launch_bt64_segment_recurrence(
        reinterpret_cast<const BF16*>(ws16),
        reinterpret_cast<const BF16*>(bt_ws),
        reinterpret_cast<BF16*>(x), reinterpret_cast<BF16*>(h_checkpoint),
        reinterpret_cast<const BF16*>(carries), final_state,
        final_state_indices, final_state_is_fp32,
        T_total, H, total_tiles, total_bt, seg_bt, n_seg, has_state_out,
        stream);
}

void launch_flashkda_bt64_cp_back_tc01(
    void* o, const void* ws16, void* bt_ws, const void* x,
    const void* h_checkpoint,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu_seqlens, hggcStream_t stream) {
    namespace bt64 = flashkda_cute_bt64;
    using BF16 = cutlass::bfloat16_t;

    if (!flashkda::host_runtime::current_device_is_tc01()) {
        throw std::runtime_error(
            "FlashKDApro BT64 CP candidate is tc01-only; refusing a tc02 launch");
    }
    if (!o || !ws16 || !bt_ws || !x || !h_checkpoint || !cu_seqlens) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 CP back received a null required pointer");
    }
    if (T_total <= 0 || H <= 0 || N <= 0 || total_tiles <= 0 || total_bt <= 0) {
        throw std::runtime_error(
            "FlashKDApro tc01 BT64 CP back received invalid dimensions");
    }

    bt64::launch_bt64_parallel_back(
        reinterpret_cast<BF16*>(o), reinterpret_cast<const BF16*>(ws16),
        reinterpret_cast<BF16*>(bt_ws), reinterpret_cast<const BF16*>(x),
        reinterpret_cast<const BF16*>(h_checkpoint),
        T_total, H, N, total_tiles, total_bt, cu_seqlens, stream);
}
