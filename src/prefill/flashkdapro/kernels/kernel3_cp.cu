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
 * FlashKDApro PPU1.5 — Kernel 3: CP (context-parallel) carry merge.
 *
 * This is the only genuinely serial stage of the CP pipeline, and it is serial
 * over SEGMENTS instead of over chunks, i.e. over T/(CH*seg_tiles) steps instead
 * of T/CH.  That reduction is the whole point of CP.
 *
 * Why a plain matrix compose is exact.  The KDA chunk recurrence is affine in
 * the state:
 *     St' = A_t . St + P_t . x_t ,   A_t = diag(g_t) - P_t . kd_t
 * so composing a segment of chunks yields an affine operator (M_i, B_i) with
 *     carry_out = M_i . carry_in + B_i .
 * pre_scan (kernel2_recurrence_nows.cu, PreScan=true) produces both parts with
 * the same mainloop: B_i is the recurrence from a zero carry, M_i the recurrence
 * from the identity with v := 0 (cuLA's M/B duality).  Since A_t multiplies from
 * the LEFT, every column of the carried matrix evolves independently, which is
 * exactly why the existing DV split can be reused verbatim for both chains.
 *
 * This kernel then walks a sequence's segments in order:
 *     carries[first]     = initial_state[seq]  (or zero)
 *     carries[first+i+1] = carries[first+i] @ M_i + B_i
 * and the segment pass consumes carries[] as its per-segment init_state.
 *
 * Transposed storage.  K2 stores state as St^T: image[c*DK + dk] = S[dk][c].
 * With Mst[c][dk] = M[dk][c] the composition becomes, in stored coordinates,
 *     Cnew^T[dv][dk] = sum_j Cold^T[dv][j] * Mst[j][dk]  +  B^T[dv][dk]
 * i.e. a plain NN GEMM whose B operand is the raw Mst image, no transpose
 * needed anywhere.
 *
 * grid = (DV/BVM, N_seq, H), 256 threads.  Each CTA owns BVM=16 rows (value
 * columns) of the carry and streams the full 128x128 Mst per step.  M/B tiles
 * are double-buffered because the loads for step i+1 do not depend on step i's
 * result, so the only unavoidable serialisation is the GEMM chain itself.
 */
#include <hggc_fp16.h>
#include <cstdlib>
#include <cutlass/bfloat16.h>
#include <type_traits>
#include "prefill/flashkdapro/common/launch_api.h"
#include "prefill/flashkdapro/common/runtime_arch.h"
#include "flashkda_ppu_util.cuh"
#include "flashkda_cute_traits.cuh"

namespace flashkda_cute_cp {

namespace fk = flashkda;
using BF16 = cutlass::bfloat16_t;
using FP16 = cute::half_t;
using namespace cute;

static constexpr int DK = 128;
static constexpr int DV = 128;
static constexpr int WARP = 32;
static constexpr int BVM = 16;          // carry rows per CTA
static constexpr int MERGE_THREADS = 256;

// Cooperative 16B-vector global->shared copy of a contiguous 16-bit block.
template <int NBytes, int NThreads>
__device__ __forceinline__ void cp_copy_g2s(void* dst, const void* src, int tid) {
    static_assert(NBytes % 16 == 0, "vector copy needs 16B multiples");
    constexpr int NVEC = NBytes / 16;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    CUTE_UNROLL
    for (int i = tid; i < NVEC; i += NThreads) d4[i] = s4[i];
}

// Exact tc02 linear AIU operands and single-leader completion contract from
// the qualified K2 loader. Only M/B global-to-shared stage transfers use AIU.
__forceinline__ __device__ void cp_stage_aiu_load_2d_b16(
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
#else
    (void)smem_ptr; (void)gmem_ptr; (void)dim_c; (void)dim_w; (void)stride_w_bytes;
#endif
}
__forceinline__ __device__ void cp_stage_aiu_commit() {
#if defined(FLASHKDA_ARCH_150)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N>
__forceinline__ __device__ void cp_stage_aiu_wait_group() {
#if defined(FLASHKDA_ARCH_150)
    asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// C[m][n] += acc, accumulating the GEMM result in fp32 before one destination
// rounding.  The carry is re-used by every later segment, so both bf16 and fp16
// paths use this tighter epilogue.
template <typename TC>
__device__ __forceinline__ void cp_store_add_f32(
    TC const& tC, BF16* dst, int ldc, int lane) {
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto sOut = make_tensor(make_smem_ptr(dst),
        make_layout(make_shape(_16{}, _16{}), make_stride(ldc, 1)));
    Tensor tCsC = thr_mma.partition_C(sOut);
    Tensor rC = recast<float>(tC);
    CUTE_UNROLL
    for (int m = 0; m < size<1>(tCsC); ++m)
        CUTE_UNROLL
        for (int n = 0; n < size<2>(tCsC); ++n)
            CUTE_UNROLL
            for (int v = 0; v < size<0>(tCsC); ++v)
                tCsC(v, m, n) =
                    BF16(fk::bf16_to_f32(tCsC(v, m, n)) + rC(v, m, n));
}

template <typename TC>
__device__ __forceinline__ void cp_store_add_f32(
    TC const& tC, FP16* dst, int ldc, int lane) {
    flashqla::TiledMma1Warp mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto sOut = make_tensor(make_smem_ptr(dst),
        make_layout(make_shape(_16{}, _16{}), make_stride(ldc, 1)));
    Tensor tCsC = thr_mma.partition_C(sOut);
    Tensor rC = recast<float>(tC);
    CUTE_UNROLL
    for (int m = 0; m < size<1>(tCsC); ++m)
        CUTE_UNROLL
        for (int n = 0; n < size<2>(tCsC); ++n)
            CUTE_UNROLL
            for (int v = 0; v < size<0>(tCsC); ++v)
                tCsC(v, m, n) = FP16(float(tCsC(v, m, n)) + rC(v, m, n));
}

// sD[BVM,DK] += sC[BVM,DK] @ sMst[DK,DK]   (sMst physically row-major [K,N])
template <typename ElemT>
__device__ __forceinline__ void cp_merge_gemm(
    const ElemT* sC, const ElemT* sMst, ElemT* sD, int tid) {
    constexpr int MB = BVM / 16, NB = DK / 16, KB = DK / 16;
    const int warp = tid / WARP, lane = tid % WARP;
    constexpr int NWARPS = MERGE_THREADS / WARP;
    auto Av = make_tensor(make_smem_ptr(const_cast<ElemT*>(sC)),
        make_layout(make_shape(Int<BVM>{}, Int<DK>{}),
                    make_stride(Int<DK>{}, _1{})));
    // Logical [N,K] view of the row-major [K,N] Mst image: no transpose in smem.
    auto Bv = make_tensor(make_smem_ptr(const_cast<ElemT*>(sMst)),
        make_layout(make_shape(Int<DK>{}, Int<DK>{}),
                    make_stride(_1{}, Int<DK>{})));
    for (int blk = warp; blk < MB * NB; blk += NWARPS) {
        int mb = blk / NB, nb = blk % NB;
        if constexpr (std::is_same<ElemT, BF16>::value) {
            auto acc = fk::make_acc16_bf16_1warp(lane);
            cute::clear(acc);
            CUTE_UNROLL
            for (int kb = 0; kb < KB; ++kb) {
                auto Ablk = local_tile(Av, make_shape(_16{}, _16{}), make_coord(mb, kb));
                auto Bblk = local_tile(Bv, make_shape(_16{}, _16{}), make_coord(nb, kb));
                fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
            }
            cp_store_add_f32(acc, sD + mb * 16 * DK + nb * 16, DK, lane);
        } else {
            auto acc = flashqla::make_acc16_1warp(lane);
            cute::clear(acc);
            CUTE_UNROLL
            for (int kb = 0; kb < KB; ++kb) {
                auto Ablk = local_tile(Av, make_shape(_16{}, _16{}), make_coord(mb, kb));
                auto Bblk = local_tile(Bv, make_shape(_16{}, _16{}), make_coord(nb, kb));
                flashqla::gemm16_ss_1warp_acc(Ablk, Bblk, lane, acc);
            }
            cp_store_add_f32(acc, sD + mb * 16 * DK + nb * 16, DK, lane);
        }
    }
}

struct MergeSmem {
    static constexpr int M_BYTES = DK * DK * 2;        // 32768
    static constexpr int C_BYTES = BVM * DK * 2;       // 8192
    static constexpr int STAGE   = M_BYTES + C_BYTES;  // one {Mst,B} stage
    static constexpr int O_STAGE = 0;
    static constexpr int O_CARRY = 2 * STAGE;
    static constexpr int BYTES   = O_CARRY + C_BYTES;  // 76 KiB
};

// seq_seg: per sequence, 2 int32 = {first segment index, segment count}.
template <typename ElemT, bool AsyncStage>
__device__ __forceinline__ void merge_serial(
    const ElemT* __restrict__ chain,       // [2, n_seg, H, DV, DK] from pre_scan
    const void* __restrict__ init_state,   // [slots,H,DV,DK], bf16/fp32 or nullptr
    const int32_t* __restrict__ init_state_indices,
    bool init_state_is_fp32,
    ElemT* __restrict__ carries,           // [n_seg, H, DV, DK]
    const int* __restrict__ seq_seg,
    int H, int n_seg) {
    extern __shared__ __align__(128) unsigned char smem[];
    const int dv_tile = int(blockIdx.x);
    const int seq = int(blockIdx.y);
    const int head = int(blockIdx.z);
    const int tid = int(threadIdx.x);
    const int dv_base = dv_tile * BVM;

    const int first = seq_seg[seq * 2 + 0];
    const int cnt = seq_seg[seq * 2 + 1];

    ElemT* sCarry = reinterpret_cast<ElemT*>(smem + MergeSmem::O_CARRY);
    const int64_t head_off = int64_t(head) * DV * DK + int64_t(dv_base) * DK;
    const int64_t chain_stride = int64_t(n_seg) * H * DV * DK;

    // carry of a sequence's first segment: the caller's initial_state, or zero.
    if (init_state) {
        const int state_slot = init_state_indices ? init_state_indices[seq] : seq;
        const int64_t state_off = int64_t(state_slot) * H * DV * DK + head_off;
        if (init_state_is_fp32) {
            const float* src = reinterpret_cast<const float*>(init_state)
                             + state_off;
            constexpr int NELEM = BVM * DK;
            for (int i = tid; i < NELEM; i += MERGE_THREADS)
                sCarry[i] = ElemT(src[i]);
        } else {
            const BF16* src = reinterpret_cast<const BF16*>(init_state)
                            + state_off;
            if constexpr (std::is_same<ElemT, BF16>::value) {
                cp_copy_g2s<MergeSmem::C_BYTES, MERGE_THREADS>(sCarry, src, tid);
            } else {
                constexpr int NELEM = BVM * DK;
                for (int i = tid; i < NELEM; i += MERGE_THREADS)
                    sCarry[i] = ElemT(fk::bf16_to_f32(src[i]));
            }
        }
    } else {
        uint4* z = reinterpret_cast<uint4*>(sCarry);
        constexpr int NVEC = MergeSmem::C_BYTES / 16;
        for (int i = tid; i < NVEC; i += MERGE_THREADS)
            z[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    cp_copy_g2s<MergeSmem::C_BYTES, MERGE_THREADS>(
        carries + int64_t(first) * H * DV * DK + head_off, sCarry, tid);

    auto stage_ptr = [&](int s) {
        return smem + MergeSmem::O_STAGE + s * MergeSmem::STAGE;
    };
    auto issue_stage = [&](int i, int s) {
        // One group owns both byte-identical linear M/B stage images.
        if constexpr (AsyncStage) {
            if (tid == 0) {
                cp_stage_aiu_load_2d_b16(
                    stage_ptr(s) + MergeSmem::M_BYTES,
                    chain + int64_t(first + i) * H * DV * DK + head_off,
                    DK, BVM, int64_t(DK) * sizeof(ElemT));
                cp_stage_aiu_load_2d_b16(
                    stage_ptr(s),
                    chain + chain_stride + int64_t(first + i) * H * DV * DK
                          + int64_t(head) * DV * DK,
                    DK, DK, int64_t(DK) * sizeof(ElemT));
                cp_stage_aiu_commit();
            }
        } else {
            // B_i tile (the segment's zero-carry result) and the full Mst image.
            cp_copy_g2s<MergeSmem::C_BYTES, MERGE_THREADS>(
                stage_ptr(s) + MergeSmem::M_BYTES,
                chain + (int64_t(first + i) * H * DV * DK) + head_off, tid);
            cp_copy_g2s<MergeSmem::M_BYTES, MERGE_THREADS>(
                stage_ptr(s),
                chain + chain_stride
                      + (int64_t(first + i) * H * DV * DK)
                      + int64_t(head) * DV * DK, tid);
        }
    };

    if (cnt > 1) issue_stage(0, 0);
    for (int i = 0; i < cnt - 1; ++i) {
        const int s = i & 1;
        ElemT* sMst = reinterpret_cast<ElemT*>(stage_ptr(s));
        ElemT* sD = reinterpret_cast<ElemT*>(stage_ptr(s) + MergeSmem::M_BYTES);
        if constexpr (AsyncStage) {
            if (tid == 0) cp_stage_aiu_wait_group<0>();
        }
        // Publish completed AIU writes to every GEMM warp. The previous
        // iteration's end barriers protect the other stage before reuse.
        __syncthreads();
        // Prefetch step i+1 into the other stage: M/B never depend on the carry,
        // so only the GEMM chain below is truly serial.
        if (i + 1 < cnt - 1) issue_stage(i + 1, s ^ 1);
        // sD holds B_i; accumulate carry @ Mst on top of it, in place.
        cp_merge_gemm(sCarry, sMst, sD, tid);
        __syncthreads();
        cp_copy_g2s<MergeSmem::C_BYTES, MERGE_THREADS>(sCarry, sD, tid);
        __syncthreads();
        cp_copy_g2s<MergeSmem::C_BYTES, MERGE_THREADS>(
            carries + int64_t(first + i + 1) * H * DV * DK + head_off,
            sCarry, tid);
    }
}

template <typename ElemT, bool AsyncStage = false>
__global__ void __launch_bounds__(MERGE_THREADS, 1) flashkda_cp_merge_kernel(
    const ElemT* chain, const void* initial, const int32_t* indices, bool state_fp32,
    ElemT* carries, const int* mapping, int heads, int segments) {
    merge_serial<ElemT, AsyncStage>(chain, initial, indices, state_fp32,
                                    carries, mapping, heads, segments);
}

__device__ __forceinline__ unsigned pack_initial_pair(float a, float b) {
    return unsigned(BF16(a).storage) | (unsigned(BF16(b).storage) << 16);
}

__device__ __forceinline__ uint4 initial_vector(
    const void* initial, const int32_t* indices, bool state_fp32, int seq,
    int64_t stride, int64_t lane_offset) {
    if (!initial) return make_uint4(0, 0, 0, 0);
    const int slot = indices ? indices[seq] : seq;
    const int64_t offset = int64_t(slot) * stride + lane_offset;
    if (!state_fp32)
        return *reinterpret_cast<const uint4*>(static_cast<const BF16*>(initial) + offset);
    const float4* src = reinterpret_cast<const float4*>(static_cast<const float*>(initial) + offset);
    const float4 lo = src[0], hi = src[1];
    return make_uint4(pack_initial_pair(lo.x, lo.y), pack_initial_pair(lo.z, lo.w),
                      pack_initial_pair(hi.x, hi.y), pack_initial_pair(hi.z, hi.w));
}

__device__ __forceinline__ bool finite_pair(unsigned pair) {
    return (pair & 0x7f80u) != 0x7f80u && (pair & 0x7f800000u) != 0x7f800000u;
}

// BF16(bias + 0.f) is bit-identical to bias for normal finite values and +0.
// The serial GEMM handles signed zero, subnormals, infinity and NaN.
__device__ __forceinline__ bool copyable_bias(unsigned short bits) {
    const unsigned exponent = bits & 0x7f80u;
    return bits == 0 || (exponent != 0 && exponent != 0x7f80u);
}

__device__ __forceinline__ bool copyable_pair(unsigned pair) {
    return copyable_bias(static_cast<unsigned short>(pair)) && copyable_bias(pair >> 16);
}

template <bool PreparedBiasProof, bool BiasCarriesPrepared = false>
__global__ void __launch_bounds__(MERGE_THREADS, 1) flashkda_cp_merge_proof_kernel(
    const BF16* chain, const void* initial, const int32_t* indices, bool state_fp32,
    BF16* carries, const int* mapping, const int32_t* zero_slices, int heads, int segments, const int32_t* skip_heads) {
    const int tid = threadIdx.x, seq = blockIdx.y, head = blockIdx.z;
    if (skip_heads && skip_heads[seq * heads + head]) return;
    const int first = mapping[2 * seq], count = mapping[2 * seq + 1];
    const int64_t stride = int64_t(heads) * DV * DK;
    const int64_t lane_offset = int64_t(head) * DV * DK + int(blockIdx.x) * BVM * DK + tid * 8;
    static_assert(MERGE_THREADS * sizeof(uint4) == MergeSmem::C_BYTES);
    const uint4 init = initial_vector(initial, indices, state_fp32, seq, stride, lane_offset);
    bool copy = finite_pair(init.x) && finite_pair(init.y) && finite_pair(init.z) && finite_pair(init.w);
    if constexpr (PreparedBiasProof) {
        // The producer proves complete row slices. Partition those checks
        // across the CTA and include them in the existing all-thread vote.
        // Uncertain bias uses the original full merge, retaining its handling
        // of signed zero, subnormals and nonfinite values.
        for (int i = tid; i < count - 1; i += MERGE_THREADS) {
            const int index = ((first + i) * heads + head) * 2;
            const int lo = zero_slices[index], hi = zero_slices[index + 1];
            copy &= (lo & 5) == 5 && (hi & 5) == 5;
        }
    } else {
        for (int i = 0; i < count - 1; ++i) {
            const int index = ((first + i) * heads + head) * 2;
            const uint4 bias = *reinterpret_cast<const uint4*>(chain + int64_t(first + i) * stride + lane_offset);
            copy &= zero_slices[index] && zero_slices[index + 1]
                 && copyable_pair(bias.x) && copyable_pair(bias.y)
                 && copyable_pair(bias.z) && copyable_pair(bias.w);
        }
    }
    __shared__ int warp_copy[MERGE_THREADS / WARP];
    const int agree = __all_sync(0xffffffff, copy);
    if ((tid % WARP) == 0) warp_copy[tid / WARP] = agree;
    __syncthreads();
    bool all_copy = true;
    CUTE_UNROLL
    for (int w = 0; w < MERGE_THREADS / WARP; ++w) all_copy &= warp_copy[w];
    if (!all_copy) {
        merge_serial<BF16, true>(chain, initial, indices, state_fp32, carries, mapping, heads, segments);
        return;
    }
    *reinterpret_cast<uint4*>(carries + int64_t(first) * stride + lane_offset) = init;
    if constexpr (!BiasCarriesPrepared)
        for (int i = 0; i < count - 1; ++i) {
            const int64_t offset = int64_t(first + i) * stride + lane_offset;
            *reinterpret_cast<uint4*>(carries + offset + stride) = *reinterpret_cast<const uint4*>(chain + offset);
        }
}

}  // namespace flashkda_cute_cp

static void launch_flashkda_cp_merge_impl(
    const void* chain, const void* init_state, void* carries, bool fp16_state,
    const int32_t* init_state_indices, bool init_state_is_fp32,
    const int* seq_seg, int H, int N, int n_seg, hggcStream_t stream,
    bool short_auto) {
    namespace cp = flashkda_cute_cp;
    dim3 grid(cp::DV / cp::BVM, N, H);
    auto launch = [&](auto type_tag, auto async_tag) {
        using ElemT = typename decltype(type_tag)::type;
        auto kern = cp::flashkda_cp_merge_kernel<ElemT, decltype(async_tag)::value>;
        hggcFuncSetAttribute(kern, hggcFuncAttributeMaxDynamicSharedMemorySize,
                             cp::MergeSmem::BYTES);
        kern<<<grid, cp::MERGE_THREADS, cp::MergeSmem::BYTES, stream>>>(
            reinterpret_cast<const ElemT*>(chain),
            init_state, init_state_indices, init_state_is_fp32,
            reinterpret_cast<ElemT*>(carries),
            seq_seg, H, n_seg);
    };
    const char* async_env = getenv("PLA_FLASHKDA_TC02_CP_ASYNC_MERGE");
    const bool async_stage = !flashkda::host_runtime::current_device_is_tc01()
        && ((async_env && async_env[0])
            ? async_env[0] == '1' && async_env[1] == '\0'
            : short_auto);
    auto select = [&](auto type_tag) {
        if (async_stage) launch(type_tag, std::true_type{});
        else             launch(type_tag, std::false_type{});
    };
    if (fp16_state) select(flashkda::TypeTag<cp::FP16>{});
    else            select(flashkda::TypeTag<cp::BF16>{});
}

void launch_flashkda_cp_merge(
    const void* chain, const void* init_state, void* carries, bool fp16_state,
    const int32_t* init_state_indices, bool init_state_is_fp32,
    const int* seq_seg, int H, int N, int n_seg, hggcStream_t stream) {
    launch_flashkda_cp_merge_impl(chain, init_state, carries, fp16_state,
        init_state_indices, init_state_is_fp32, seq_seg, H, N, n_seg, stream,
        false);
}

void launch_flashkda_cp_merge_short_auto(
    const void* chain, const void* init_state, void* carries, bool fp16_state,
    const int32_t* init_state_indices, bool init_state_is_fp32,
    const int* seq_seg, int H, int N, int n_seg, hggcStream_t stream) {
    launch_flashkda_cp_merge_impl(chain, init_state, carries, fp16_state,
        init_state_indices, init_state_is_fp32, seq_seg, H, N, n_seg, stream,
        true);
}

void launch_flashkda_cp_merge_zero_flags(
    const void* chain, const void* initial, void* carries, const int32_t* indices,
    bool state_fp32, const int* mapping, const int32_t* zero_slices,
    int heads, int sequences, int segments, hggcStream_t stream, bool prepared_bias, const int32_t* skip_heads) {
    namespace cp = flashkda_cute_cp;
    auto launch=[&](auto prepared) {
        auto kernel=cp::flashkda_cp_merge_proof_kernel<decltype(prepared)::value, decltype(prepared)::value>;
        hggcFuncSetAttribute(kernel,hggcFuncAttributeMaxDynamicSharedMemorySize,cp::MergeSmem::BYTES);
        kernel<<<dim3(cp::DV/cp::BVM,sequences,heads),cp::MERGE_THREADS,cp::MergeSmem::BYTES,stream>>>(
            static_cast<const cp::BF16*>(chain),initial,indices,state_fp32,
            static_cast<cp::BF16*>(carries),mapping,zero_slices,heads,segments,skip_heads);
    };
    if (prepared_bias) launch(std::true_type{});
    else launch(std::false_type{});
}
