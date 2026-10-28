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
 * FlashKDA PPU1.5 — Kernel 1 (prepare), correctness-first port.
 *
 * Blueprint: csrc/smxx/fwd_kernel1.cuh (_flash_kda_fwd_prepare) + torch_ref.py.
 * One CTA normally processes one (chunk tile, head).  The promoted long-sequence
 * path processes two adjacent chunks per CTA; block size is dispatch-selected.
 *
 * Per (tile,head) it reproduces the blueprint math EXACTLY (op order matches
 * torch_ref so numerics align):
 *   1. load q,k,g[16,128] bf16, beta[16] bf16, dt_bias[128] f32
 *   2. L2-normalize q,k rows (warp-shuffle tree reduction, kernel-match)
 *   3. gate: g = lower*sigmoid(exp(A_log)*(g+dt_bias)); cumsum -> g_cumsum,
 *      g_total = last row cumsum
 *   4. decay: k_decayed=k*ex2(gc); q_decayed=q*ex2(gc)*scale;
 *      k_inv=k*ex2(-gc); k_restored=k_inv*ex2(g_total)
 *   5. L = tril(k_decayed@k_inv^T,-1)*beta (f16); Mqk = tril(q_decayed@k_inv^T)
 *   6. INV = (I-L); Neumann: INV=INV(I+L2)(I+L4)(I+L8), fp16 accumulate
 *   7. store k_decayed,q_decayed,k_restored,g_total,INV,Mqk to workspace
 *
 * NOTE: input full tiles use PPU AIU 2D async loads; tail tiles keep manual
 * zero-fill fallback. Workspace stores remain cooperative global stores until a
 * validated PPU AIU store primitive is available in this codebase.
 */
#include <hggc_fp16.h>
#include <cutlass/bfloat16.h>
#include <cstdlib>
#include <map>

#include "prefill/flashkdapro/common/launch_api.h"
#include "prefill/flashkdapro/common/runtime_arch.h"
#include "flashkda_ppu_util.cuh"   // ex2/tanh/sigmoid/cvt + bf16 MMA helpers
#include "flashkda_f16acc.cuh"     // f16-accumulate MMA helpers (Neumann)

namespace flashkda_cute {

namespace fk = flashkda;
using cutlass::bfloat16_t;
using BF16 = cutlass::bfloat16_t;
using cute::half_t;

static constexpr int CHUNK = 16;
static constexpr int D = 128;
static constexpr float LOG2E = 1.4426950408889634f;

template <typename T, int N>
struct alignas(128) K1Array {
    T data[N];
};

struct alignas(128) K1Smem {
    union {
        struct {
            K1Array<BF16, CHUNK * D> q;
            K1Array<BF16, CHUNK * D> k;
            K1Array<float, CHUNK * D> g;
        } phase_a;
        struct {
            K1Array<BF16, CHUNK * D> kd;
            K1Array<BF16, CHUNK * D> qd;
            K1Array<half_t, CHUNK * CHUNK> L;
            K1Array<half_t, CHUNK * CHUNK> INV;
            K1Array<BF16, CHUNK * CHUNK> Mqk;
        } phase_b;
    } main;
    K1Array<BF16, 32> beta;
    union {
        K1Array<BF16, CHUNK * D> g_bf16;
        K1Array<BF16, CHUNK * D> kr;
    } aux_bf16;
    union {
        K1Array<float, D> dt;
        K1Array<float, D> gt;
    } aux_f32;
    K1Array<half_t, CHUNK * CHUNK> pow;
};

__forceinline__ __device__ void k1_aiu_commit() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
__forceinline__ __device__ void k1_aiu_wait_group() {
#if defined(__HGGC_ARCH__)
    asm volatile("ppu.cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// A row-major (16,16) fp16 smem tensor view.
__device__ __forceinline__ auto view16(half_t* p) {
    return cute::make_tensor(cute::make_smem_ptr(p),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                          cute::make_stride(cute::_16{}, cute::_1{})));
}
// Transposed (16,16) view (swap strides) — turns A@B^T (TN) into A@B (NN).
__device__ __forceinline__ auto view16_T(half_t* p) {
    return cute::make_tensor(cute::make_smem_ptr(p),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                          cute::make_stride(cute::_1{}, cute::_16{})));
}
__device__ __forceinline__ auto view16b(BF16* p) {
    return cute::make_tensor(cute::make_smem_ptr(p),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                          cute::make_stride(cute::_16{}, cute::_1{})));
}
// (16,128) row-major bf16 view.
__device__ __forceinline__ auto view16x128b(BF16* p) {
    return cute::make_tensor(cute::make_smem_ptr(p),
        cute::make_layout(cute::make_shape(cute::_16{}, cute::Int<D>{}),
                          cute::make_stride(cute::Int<D>{}, cute::_1{})));
}

}  // namespace flashkda_cute

#include "prefill/flashkdapro/tc01/kernel1_prepare_tc01.cuh"
#include "prefill/flashkdapro/tc02/kernel1_prepare_tc02.cuh"

namespace flashkda_cute {

// Keep the common prepare body architecture-neutral.  The compiler-selected
// device pass calls the instruction spelling owned by that architecture.
__forceinline__ __device__ void k1_aiu_load_2d_b16(
    void* smem_ptr, const void* gmem_ptr, int dim_c, int dim_w,
    int64_t stride_w_bytes) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
    tc02::k1_aiu_load_2d_b16(
        smem_ptr, gmem_ptr, dim_c, dim_w, stride_w_bytes);
#elif defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 100
    tc01::k1_aiu_load_2d_b16(
        smem_ptr, gmem_ptr, dim_c, dim_w, stride_w_bytes);
#else
    (void)smem_ptr; (void)gmem_ptr; (void)dim_c; (void)dim_w;
    (void)stride_w_bytes;
#endif
}

// Build the exact number of 16-token tiles before each sequence. This mirrors
// upstream FlashKDA commit 1ce47ea3: the one-time O(N) pass replaces the old
// O(N) scan in every varlen K1 CTA. One lane is intentional; N is normally
// small enough that launch + scan is cheaper than a second parallel scan
// protocol, and it keeps the prefix construction bit-exact and deterministic.
__global__ void flashkda_build_tile_prefix(
    const int32_t* __restrict__ cu_seqlens,
    int* __restrict__ tile_prefix,
    int N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int acc = 0;
        tile_prefix[0] = 0;
        for (int i = 0; i < N; ++i) {
            int slen = int(cu_seqlens[i + 1] - cu_seqlens[i]);
            acc += (slen + CHUNK - 1) / CHUNK;
            tile_prefix[i + 1] = acc;
        }
    }
}

// Keep padded-output handling out of the large prepare specialization.  Even
// when cu_seqlens[N] == T_total, compiling the old int64 flatten/div/mod loop
// into K1 raised the 256-thread varlen kernel from 64/128 to 80/160 vregs/sregs
// on tc02 and reduced occupancy.  This small kernel uses a single CTA and
// writes the whole suffix as one contiguous range, so it needs no div/mod at
// all.  A negative valid_tokens means the Host did not have trusted offsets;
// in that compatibility path one lane reads the endpoint from cu_seqlens.  A
// single CTA is important here: launching one empty checker CTA per head made
// the exact-length legacy API pay roughly 7 us * H on tc02.
__global__ void flashkda_zero_padded_output_tail(
    BF16* __restrict__ out,
    int T_total, int H, int N,
    const int32_t* __restrict__ cu_seqlens,
    int valid_tokens) {
    __shared__ int device_valid_tokens;
    if (valid_tokens < 0) {
        if (threadIdx.x == 0) device_valid_tokens = cu_seqlens[N];
        __syncthreads();
        valid_tokens = device_valid_tokens;
    }
    if (valid_tokens < 0 || valid_tokens >= T_total) return;

    const int64_t suffix_begin = int64_t(valid_tokens) * H * D;
    const int64_t suffix_elements = int64_t(T_total - valid_tokens) * H * D;
    for (int64_t i = int(threadIdx.x); i < suffix_elements;
         i += int(blockDim.x)) {
        out[suffix_begin + i] = BF16(0.0f);
    }
}

// ===================== Kernel 1: prepare =====================
template <int NThreads, bool IsVarlen, bool UseTilePrefix,
          bool PrecomputeAffine = false, int ChunksPerCTA = 1,
          bool FmaFront = false, bool Tc01KrContiguous = false>
__global__ void __launch_bounds__(NThreads) flashkda_prepare_kernel(
    const BF16* __restrict__ q,       // [H, T_total, D] (stride D, D*H, 1)
    const BF16* __restrict__ k,
    const BF16* __restrict__ g_in,
    const BF16* __restrict__ beta,    // [T_total,H]
    const float* __restrict__ A_log,  // [H]
    const float* __restrict__ dt_bias,// [H, D]
    float scale, float gate_scale,
    int T_total, int H, int beta_stride,
    int N, int total_tiles, int tile_count,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    int tile_offset, int head_offset, int* __restrict__ ready,
    bool fp16_kd_qd,
    // workspace output: ONE contiguous slot of fk::WS_SLOT_UNITS 16-bit units
    // per (head,tile), indexed by ws_idx = head*total_tiles + gtile. The Kd and
    // Qd fields are FP16 when fp16_kd_qd is true; every offset and slot size
    // stays unchanged.
    BF16* __restrict__ ws)
{
    static_assert(!PrecomputeAffine || NThreads == 128,
                  "fused affine prepare is specialized for four warps");
    static_assert(ChunksPerCTA >= 1, "ChunksPerCTA must be positive");
    static_assert(!FmaFront || (NThreads == 128 && !PrecomputeAffine
                                && ChunksPerCTA == 1),
                  "the tc01 FMA front is a plain four-warp CPC1 specialization");
    static_assert(!Tc01KrContiguous
                      || (NThreads == 128 && ChunksPerCTA == 1),
                  "contiguous tc01 A layout is a four-warp CPC1 specialization");
    const int head = int(blockIdx.y) + head_offset;
    const int tid = threadIdx.x;

    // FlashInfer PR #4605's most portable scheduling idea is K1_CPC: keep a
    // prepare CTA resident while it walks adjacent chunks of the same head.
    // Unlike its SM100 engine, this needs no TMEM/TMA/CLC primitive and keeps
    // every chunk workspace/ready-flag ABI unchanged.  CPC=1 is bit-for-bit
    // the original launch and remains available as the rollback baseline.
    // Keep one copy of the sizeable prepare body.  Fully unrolling this loop
    // raises the PPU1.5 register allocation from 64 to 88 registers/thread,
    // cutting theoretical occupancy from 75% to 62.5% and erasing any launch
    // amortisation benefit.
    #pragma unroll 1
    for (int cpc_iter = 0; cpc_iter < ChunksPerCTA; ++cpc_iter) {
    const int gtile = int(blockIdx.x) * ChunksPerCTA
        + cpc_iter + tile_offset;
    if (gtile >= tile_offset + tile_count) return;

    // --- locate (seq, local tile) ---
    int seq_idx, local_t; int64_t bos, eos;
    if constexpr (IsVarlen) {
        int tiles_before = 0;
        if constexpr (UseTilePrefix) {
            // Unlike upstream, PPU currently launches a synchronization-free
            // upper-bound grid. Reject its spare CTAs before the binary search.
            if (gtile >= tile_prefix[N]) return;
            int lo = 0, hi = N;
            while (lo + 1 < hi) {
                int mid = (lo + hi) >> 1;
                if (tile_prefix[mid] <= gtile) lo = mid;
                else hi = mid;
            }
            seq_idx = lo;
            tiles_before = tile_prefix[lo];
        } else {
            seq_idx = 0;
            for (; seq_idx < N; ++seq_idx) {
                int slen = int(
                    cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]);
                int nt = (slen + CHUNK - 1) / CHUNK;
                if (tiles_before + nt > gtile) break;
                tiles_before += nt;
            }
            // The host deliberately launches a synchronization-free upper
            // bound. Padding and per-sequence chunk rounding can leave spare
            // CTAs; using seq_idx itself as the sentinel avoids keeping a
            // separate `found` predicate live through this large kernel.
            if (seq_idx == N) return;
        }
        local_t = gtile - tiles_before;
        bos = cu_seqlens[seq_idx]; eos = cu_seqlens[seq_idx + 1];
    } else {
        int T_seq = T_total / N;
        int tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
        seq_idx = gtile / tiles_per_seq;
        local_t = gtile - seq_idx * tiles_per_seq;
        bos = int64_t(seq_idx) * T_seq; eos = bos + T_seq;
    }
    int seq_len = int(eos - bos);
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    if (local_t >= t_tiles) return;
    int actual_len = min(CHUNK, seq_len - local_t * CHUNK);
    int64_t tok0 = bos + int64_t(local_t) * CHUNK;

    // --- shared memory ---
    extern __shared__ __align__(128) unsigned char smem_raw[];
    K1Smem& smem = *reinterpret_cast<K1Smem*>(smem_raw);

    BF16* sq = smem.main.phase_a.q.data;
    BF16* sk = smem.main.phase_a.k.data;
    float* sg = smem.main.phase_a.g.data;
    BF16* g_bf16 = smem.aux_bf16.g_bf16.data;
    float* s_dt = smem.aux_f32.dt.data;
    float* sgt = smem.aux_f32.gt.data;
    BF16* s_beta_raw = smem.beta.data;
    float* s_beta = reinterpret_cast<float*>(smem.beta.data);
    BF16* s_kd = smem.main.phase_b.kd.data;
    BF16* s_qd = smem.main.phase_b.qd.data;
    BF16* s_ki = smem.aux_bf16.kr.data;
    half_t* sL = smem.main.phase_b.L.data;
    half_t* sINV = smem.main.phase_b.INV.data;
    half_t* sPow = smem.pow.data;
#if defined(FLASHKDA_ARCH_150)
    // The Neumann scratch is unused until after the diagonal is consumed.
    // Reuse its first 16 BF16 entries without growing shared memory or the ABI.
    BF16* sMqkDiag = reinterpret_cast<BF16*>(sPow);
#endif
    half_t* sTmp = sL;
    BF16* sMqk = smem.main.phase_b.Mqk.data;

    const int64_t rstride = int64_t(D) * H;   // per-token row stride in q/k/g

    // --- 1. load q,k,g tiles (rows>=actual zeroed), beta, dt_bias ---
    if (actual_len == CHUNK) {
        if (tid == 0) {
            const BF16* q_src = q + int64_t(head) * D + tok0 * rstride;
            const BF16* k_src = k + int64_t(head) * D + tok0 * rstride;
            const BF16* g_src = g_in + int64_t(head) * D + tok0 * rstride;
            k1_aiu_load_2d_b16(sq, q_src, D, CHUNK, rstride * sizeof(BF16));
            k1_aiu_load_2d_b16(sk, k_src, D, CHUNK, rstride * sizeof(BF16));
            k1_aiu_load_2d_b16(g_bf16, g_src, D, CHUNK, rstride * sizeof(BF16));
            const BF16* beta_src = beta + tok0 * beta_stride + head;
            k1_aiu_load_2d_b16(
                s_beta_raw, beta_src, 1, CHUNK,
                beta_stride * sizeof(BF16));
            k1_aiu_commit();
            k1_aiu_wait_group<0>();
        }
        __syncthreads();
        for (int idx = tid; idx < CHUNK * D; idx += NThreads)
            sg[idx] = fk::bf16_to_f32(g_bf16[idx]);
    } else {
        for (int idx = tid; idx < CHUNK * D; idx += NThreads) {
            int r = idx / D, c = idx % D;
            if (r < actual_len) {
                int64_t off = int64_t(head) * D + (tok0 + r) * rstride + c;
                sq[idx] = q[off];
                sk[idx] = k[off];
                sg[idx] = fk::bf16_to_f32(g_in[off]);   // raw log-gate (pre-activation)
            } else {
                sq[idx] = BF16(0); sk[idx] = BF16(0); sg[idx] = 0.0f;
            }
        }
    }
    for (int c = tid; c < D; c += NThreads)
        s_dt[c] = dt_bias[int64_t(head) * D + c];
    // K1 and K2 both use sigmoid(beta).  Compute it once here and preserve the
    // exact fp32 result in the workspace instead of repeating the intrinsic in
    // every value-split K2 CTA.  The full-tile raw beta vector arrived through
    // the same AIU group as q/k/g; tails keep the scalar fallback.
    float beta_raw_reg = 0.0f;
    if (tid < 32) {
        if (tid < CHUNK) {
            if (actual_len == CHUNK)
                beta_raw_reg = fk::bf16_to_f32(s_beta_raw[tid]);
            else if (tid < actual_len)
                beta_raw_reg = fk::bf16_to_f32(
                    beta[(tok0 + tid) * beta_stride + head]);
        }
        __syncwarp();
        if (tid < CHUNK)
            s_beta[tid] = fk::sigmoid_tanh_approx_f32(beta_raw_reg);
    }
    __syncthreads();

    // --- 2. L2 normalize q,k rows (kernel-match: 8 elems/thread, 16 thr/row) ---
    {
        constexpr int EPT = 8, TPR = D / EPT;   // 16 threads per row
        int col = (tid % TPR) * EPT;
        for (int row = tid / TPR; row < CHUNK; row += NThreads / TPR) {
            float qv[EPT], kv[EPT], qs = 0.f, ks = 0.f;
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                qv[i] = fk::bf16_to_f32(sq[row * D + col + i]);
                kv[i] = fk::bf16_to_f32(sk[row * D + col + i]);
                qs += qv[i] * qv[i]; ks += kv[i] * kv[i];
            }
            #pragma unroll
            for (int d = 8; d >= 1; d >>= 1) {
                qs += __shfl_xor_sync(0xFFFFFFFF, qs, d);
                ks += __shfl_xor_sync(0xFFFFFFFF, ks, d);
            }
            float qn = rsqrtf(qs + 1e-6f), kn = rsqrtf(ks + 1e-6f);
#if defined(FLASHKDA_ARCH_150)
            float qk_dot = 0.f;
#endif
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                const BF16 nq = BF16(qv[i] * qn);
                const BF16 nk = BF16(kv[i] * kn);
                sq[row * D + col + i] = nq;
                sk[row * D + col + i] = nk;
#if defined(FLASHKDA_ARCH_150)
                qk_dot += fk::bf16_to_f32(nq) * fk::bf16_to_f32(nk);
#endif
            }
#if defined(FLASHKDA_ARCH_150)
            // On the diagonal the gate factors cancel exactly. Accumulate the
            // normalized BF16 operands in FP32 before applying scale, rather
            // than multiplying separately rounded q*exp(g) and k*exp(-g).
            #pragma unroll
            for (int d = 8; d >= 1; d >>= 1)
                qk_dot += __shfl_xor_sync(0xFFFFFFFF, qk_dot, d);
            if (col == 0) sMqkDiag[row] = BF16(qk_dot * scale);
#endif
        }
    }
    // --- 3. gate activation + cumsum (threads 0..127 = one column each) ---
    float a_log_exp = expf(A_log[head]);
    for (int col = tid; col < D; col += NThreads) {
        float dt = s_dt[col];
        float sum = 0.f;
        #pragma unroll
        for (int r = 0; r < CHUNK; ++r) {
            float gv = 0.f;
            if (r < actual_len) {
                gv = sg[r * D + col] + dt;
                gv = a_log_exp * gv;
                gv = gate_scale * fk::sigmoid_tanh_approx_f32(gv);
            }
            sum += gv;
            sg[r * D + col] = sum;   // cumulative sum
        }
        sgt[col] = fk::ex2_approx_ftz_f32(sum);  // ex2(g_total), same column owner
    }
    __syncthreads();

    // --- 4. decay: elementwise over 16x128 ---
    for (int idx = tid; idx < CHUNK * D; idx += NThreads) {
        int c = idx % D;
        float gc = sg[idx];
        float e = fk::ex2_approx_ftz_f32(gc);
        float ei = fk::ex2_approx_ftz_f32(-gc);
        float qf = fk::bf16_to_f32(sq[idx]);
        float kf = fk::bf16_to_f32(sk[idx]);
        BF16 ki = BF16(kf * ei);
        s_qd[idx] = BF16(qf * e * scale);
        s_kd[idx] = BF16(kf * e);
        s_ki[idx] = ki;
    }
    __syncthreads();

    // --- 5. L = k_decayed @ k_inv^T (bf16->f32->f16); Mqk = q_decayed @ k_inv^T ---
    if constexpr (FmaFront) {
        tc01::k1_fma_triangular_front<NThreads>(
            tid, s_kd, s_qd, s_ki, sL, sMqk);
    } else {
        // The dense fallback computes both complete 16x16 tiles with TensorCell:
        // warp 0 -> L, warp 1 -> Mqk, eight K=16 blocks per product.
        int warp = tid / 32, lane = tid % 32;
        if (warp == 0 || warp == 1) {
            auto acc = fk::make_acc16_bf16_1warp(lane);
            cute::clear(acc);
            const BF16* Aptr = (warp == 0) ? s_kd : s_qd;
            #pragma unroll
            for (int kb = 0; kb < D / 16; ++kb) {
                auto A = view16x128b(const_cast<BF16*>(Aptr));
                auto Bk = view16x128b(s_ki);
                auto Ablk = cute::local_tile(A, cute::make_shape(cute::_16{}, cute::_16{}),
                                             cute::make_coord(0, kb));
                auto Bblk = cute::local_tile(Bk, cute::make_shape(cute::_16{}, cute::_16{}),
                                             cute::make_coord(0, kb));
                fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
            }
            if (warp == 0)
                fk::store_acc16_bf16mma<half_t>(acc, reinterpret_cast<half_t*>(sL), 16, lane);
            else
                fk::store_acc16_bf16mma<BF16>(acc, sMqk, 16, lane);
        }
    }
    __syncthreads();

    // --- 6a. tril + beta + INV = I - L (elementwise 16x16) ---
    for (int idx = tid; idx < CHUNK * CHUNK; idx += NThreads) {
        int i = idx / CHUNK, j = idx % CHUNK;
        half_t x = sL[i * CHUNK + j];
        if (i <= j) {
            x = half_t(0.f);
        } else {
            float b = s_beta[i];
            x = half_t(float(x) * b);
        }
        sL[i * CHUNK + j] = x;
        // Mqk: keep lower triangle (i>=j), zero strictly-upper
        if (i < j) sMqk[i * CHUNK + j] = BF16(0.f);
#if defined(FLASHKDA_ARCH_150)
        if (i == j) sMqk[i * CHUNK + j] = sMqkDiag[i];
#endif
        // The split front leaves INV construction to its one-warp MMA tail.
        if constexpr (!FmaFront)
            sINV[i * CHUNK + j] = (i == j) ? half_t(1.f) : half_t(-float(x));
    }
    __syncthreads();

    if constexpr (PrecomputeAffine) {
        // Materialize the exact BF16 kr image once.  The ordinary path fuses
        // this multiply into the global-store loop; the affine path consumes
        // kr in eight parallel K1 MMAs below.
        for (int idx = tid; idx < CHUNK * D; idx += NThreads) {
            int c = idx & (D - 1);
            s_ki[idx] = BF16(fk::bf16_to_f32(s_ki[idx]) * sgt[c]);
        }
        __syncthreads();
    }

    // Resolve the workspace slot before the serial Neumann tail.  On the
    // 128-thread kernel, warps 1-3 write every already-finalized field while
    // warp 0 evaluates the strict INV recurrence.  The existing CTA barrier
    // below is the join; this adds no synchronization and leaves only INV for
    // the cooperative epilogue.
    const int ws_idx = head * total_tiles + gtile;
    BF16* slot  = ws + int64_t(ws_idx) * fk::WS_SLOT_UNITS;
    BF16* o_kd  = slot + fk::WS_OFF_KD_U;
    BF16* o_qd  = slot + fk::WS_OFF_QD_U;
    BF16* o_kr  = slot + fk::WS_OFF_KR_U;   // private layout selected by host
    BF16* o_inv = slot + fk::WS_OFF_INV_U;
    BF16* o_mqk = slot + fk::WS_OFF_MQK_U;
    float* o_gt = reinterpret_cast<float*>(slot + fk::WS_OFF_GT_U);
    float* o_beta = reinterpret_cast<float*>(slot + fk::WS_OFF_BETA_U);
    if constexpr (NThreads == 128) {
        if (tid >= 32) {
            constexpr int StoreThreads = NThreads - 32;
            int stid = tid - 32;
            if constexpr (PrecomputeAffine) {
                constexpr int BF16_PER_VEC = int(sizeof(uint4) / sizeof(BF16));
                for (int vec = stid;
                     vec < CHUNK * D / BF16_PER_VEC; vec += StoreThreads) {
                    int idx = vec * BF16_PER_VEC;
                    *reinterpret_cast<uint4*>(o_kd + idx) =
                        *reinterpret_cast<const uint4*>(s_kd + idx);
                    *reinterpret_cast<uint4*>(o_qd + idx) =
                        *reinterpret_cast<const uint4*>(s_qd + idx);
                }
            } else {
                for (int idx = stid; idx < CHUNK * D; idx += StoreThreads) {
                    if (fp16_kd_qd) {
                        reinterpret_cast<half_t*>(o_kd)[idx] =
                            half_t(fk::bf16_to_f32(s_kd[idx]));
                        reinterpret_cast<half_t*>(o_qd)[idx] =
                            half_t(fk::bf16_to_f32(s_qd[idx]));
                    } else {
                        o_kd[idx] = s_kd[idx];
                        o_qd[idx] = s_qd[idx];
                    }
#if defined(FLASHKDA_ARCH_150)
                    int src = fk::ws_kr_swzl_src_index(idx);
                    int cc = src & (D - 1);
                    o_kr[idx] = BF16(fk::bf16_to_f32(s_ki[src]) * sgt[cc]);
#else
                    if constexpr (Tc01KrContiguous) {
                        // Store kr^T as a contiguous [DK,CH] tile.  K2's state
                        // update then reads each 16x16 A tile without the
                        // 256-byte column stride that creates tc01 TSM bank
                        // conflicts on the ordinary [CH,DK] image.
                        int cc = idx / CHUNK;
                        int row = idx % CHUNK;
                        int src = row * D + cc;
                        o_kr[idx] = BF16(
                            fk::bf16_to_f32(s_ki[src]) * sgt[cc]);
                    } else {
                        int cc = idx & (D - 1);
                        o_kr[idx] = BF16(fk::bf16_to_f32(s_ki[idx]) * sgt[cc]);
                    }
#endif
                }
            }
            for (int c = stid; c < D; c += StoreThreads) o_gt[c] = sgt[c];
            if constexpr (!PrecomputeAffine) {
                if (stid < CHUNK) o_beta[stid] = s_beta[stid];
            }
            if constexpr (!PrecomputeAffine) {
                for (int idx = stid; idx < CHUNK * CHUNK; idx += StoreThreads)
                    o_mqk[idx] = sMqk[idx];
            }
        }
    }

    // --- 6b. Neumann series (warp 0, fp16 accumulate), matches torch_ref ---
    // INV = INV*(I+L2)*(I+L4)*(I+L8), pow = L^2, L^4, L^8
    if constexpr (!FmaFront) if (tid < 32) {
        int lane = tid;
        // pow = L @ L
        {
            auto acc = fk::make_acc16_f16acc_1warp(lane);
            fk::gemm16_ss_1warp_f16acc(view16(sL), view16_T(sL), lane, acc);
            fk::store_acc16_f16acc_1warp(acc, reinterpret_cast<__half*>(sPow), 16, lane);
        }
        __syncwarp();
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            // tmp = INV @ pow ; INV += tmp
            {
                auto acc = fk::make_acc16_f16acc_1warp(lane);
                fk::gemm16_ss_1warp_f16acc(view16(sINV), view16_T(sPow), lane, acc);
                fk::store_acc16_f16acc_1warp(acc, reinterpret_cast<__half*>(sTmp), 16, lane);
            }
            __syncwarp();
            for (int idx = lane; idx < CHUNK * CHUNK; idx += 32)
                sINV[idx] = half_t(float(sINV[idx]) + float(sTmp[idx]));
            __syncwarp();
            if (it < 2) {
                // pow = pow @ pow
                auto acc = fk::make_acc16_f16acc_1warp(lane);
                fk::gemm16_ss_1warp_f16acc(view16(sPow), view16_T(sPow), lane, acc);
                fk::store_acc16_f16acc_1warp(acc, reinterpret_cast<__half*>(sTmp), 16, lane);
                __syncwarp();
                for (int idx = lane; idx < CHUNK * CHUNK; idx += 32) sPow[idx] = sTmp[idx];
                __syncwarp();
            }
        }
    }
    __syncthreads();

    // For the promoted BV16/DK4/leader recurrence, associate the two
    // chunk-constant products in K1:
    //   P  = kr^T @ INV @ diag(beta)       [128,16]
    //   O  = Mqk  @ INV @ diag(beta)       [16,16]
    // K2 then consumes x=(v-kd@state) directly.  This preserves the strict
    // per-chunk recurrent dependency while removing one MMA and one fragment
    // transpose from every value-split CTA/chunk.  P^T reuses the kr slot and
    // O reuses the Mqk slot, so the workspace size does not grow.
    if constexpr (PrecomputeAffine) {
        BF16* sBmatT = reinterpret_cast<BF16*>(sPow);
        BF16* sOmat = reinterpret_cast<BF16*>(sL);
        // kd/qd were already written by warps 1-3 while warp 0 evaluated the
        // Neumann tail, and the join barrier above has completed.  Reuse the
        // now-dead kd tile for P so it does not alias sL/sMqk through phase_a.g.
        BF16* sPTrans = s_kd;
        for (int idx = tid; idx < CHUNK * CHUNK; idx += NThreads) {
            int j = idx / CHUNK, i = idx % CHUNK;
            sBmatT[idx] = BF16(float(sINV[i * CHUNK + j]) * s_beta[j]);
        }
        __syncthreads();

        int warp = tid / 32, lane = tid & 31;
        if (warp == 0) {
            auto acc = fk::make_acc16_bf16_1warp(lane);
            cute::clear(acc);
            fk::gemm16_ss_1warp_bf16_acc(
                view16b(sMqk), view16b(sBmatT), lane, acc);
            fk::store_acc16_bf16mma<BF16>(acc, sOmat, CHUNK, lane);
        }
        // Schedule the eight independent P row-blocks around warp 0's O MMA:
        //   round 0: w0=O, w1=P0, w2=P1, w3=P2
        //   round 1: w0=P3, w1=P4, w2=P5, w3=P6
        //   round 2: w0=P7
        // The old code used the same three critical-path MMA rounds but put two
        // CTA barriers and a global O store between O and P.
        int p_kb0 = warp == 0 ? 3 : warp - 1;
        int p_kb1 = warp == 0 ? 7 : warp + 3;
        CUTE_UNROLL
        for (int pass = 0; pass < 2; ++pass) {
            int kb = pass == 0 ? p_kb0 : p_kb1;
            auto KrT = cute::make_tensor(
                cute::make_smem_ptr(s_ki + kb * 16),
                cute::make_layout(
                    cute::make_shape(cute::_16{}, cute::_16{}),
                    cute::make_stride(cute::_1{}, cute::Int<D>{})));
            auto acc = fk::make_acc16_bf16_1warp(lane);
            cute::clear(acc);
            fk::gemm16_ss_1warp_bf16_acc(
                view16b(sBmatT), KrT, lane, acc);
            if constexpr (Tc01KrContiguous) {
                // acc is logical P^T[r,c]. Store it directly as P[c,r] in a
                // contiguous [DK,CH] block, preserving the same BF16 rounding.
                fk::store_acc16_bf16mma_transposed<BF16>(
                    acc, sPTrans + kb * CHUNK * CHUNK, lane);
            } else {
                fk::store_acc16_bf16mma<BF16>(
                    acc, sPTrans + kb * CHUNK, D, lane);
            }
        }
        __syncthreads();
        constexpr int BF16_PER_VEC = int(sizeof(uint4) / sizeof(BF16));
        for (int vec = tid;
             vec < CHUNK * CHUNK / BF16_PER_VEC; vec += NThreads) {
            *reinterpret_cast<uint4*>(o_mqk + vec * BF16_PER_VEC) =
                *reinterpret_cast<const uint4*>(sOmat + vec * BF16_PER_VEC);
        }
        // ws_kr_swzl_src_index permutes aligned 8-element groups but remains
        // contiguous within each group, so the physical workspace image can be
        // copied as aligned 16-byte vectors with no bit or layout change.
        for (int vec = tid;
             vec < CHUNK * D / BF16_PER_VEC; vec += NThreads) {
            int dst = vec * BF16_PER_VEC;
            int src;
            if constexpr (Tc01KrContiguous) {
                src = dst;
            } else {
#if defined(FLASHKDA_ARCH_150)
                src = fk::ws_kr_swzl_src_index(dst);
#else
                src = dst;
#endif
            }
            *reinterpret_cast<uint4*>(o_kr + dst) =
                *reinterpret_cast<const uint4*>(sPTrans + src);
        }
    }

    // --- 7. write back workspace (single contiguous byte-image slot) ---
    if constexpr (NThreads == 128) {
        // The affine consumer uses O=Mqk@INV@diag(beta) and
        // P=kr^T@INV@diag(beta), so neither raw INV nor beta is read by K2.
        if constexpr (!PrecomputeAffine) {
            if constexpr (FmaFront) {
                // Temporary private ABI between the split kernels.  FP16 and
                // BF16 are both two bytes; the tail consumes these bits as FP16
                // and overwrites the field with the public BF16 INV image before
                // recurrence starts.
                half_t* o_l = reinterpret_cast<half_t*>(o_inv);
                for (int idx = tid; idx < CHUNK * CHUNK; idx += NThreads)
                    o_l[idx] = sL[idx];
            } else {
                for (int idx = tid; idx < CHUNK * CHUNK; idx += NThreads)
                    o_inv[idx] = BF16(float(sINV[idx]));
            }
        }
    } else {
        for (int idx = tid; idx < CHUNK * D; idx += NThreads) {
            if (fp16_kd_qd) {
                reinterpret_cast<half_t*>(o_kd)[idx] =
                    half_t(fk::bf16_to_f32(s_kd[idx]));
                reinterpret_cast<half_t*>(o_qd)[idx] =
                    half_t(fk::bf16_to_f32(s_qd[idx]));
            } else {
                o_kd[idx] = s_kd[idx];
                o_qd[idx] = s_qd[idx];
            }
            // kr swizzle: slot slot-index idx gathers pre-swizzle row-major src.
            if constexpr (!PrecomputeAffine) {
#if defined(FLASHKDA_ARCH_150)
                int src = fk::ws_kr_swzl_src_index(idx);
                int cc = src & (D - 1);             // src % 128 (source column)
                o_kr[idx] = BF16(fk::bf16_to_f32(s_ki[src]) * sgt[cc]);
#else
                int cc = idx & (D - 1);
                o_kr[idx] = BF16(fk::bf16_to_f32(s_ki[idx]) * sgt[cc]);
#endif
            }
        }
        for (int c = tid; c < D; c += NThreads) o_gt[c] = sgt[c];
        for (int i = tid; i < CHUNK; i += NThreads) o_beta[i] = s_beta[i];
        for (int idx = tid; idx < CHUNK * CHUNK; idx += NThreads) {
            o_inv[idx] = BF16(float(sINV[idx]));
            if constexpr (!PrecomputeAffine)
                o_mqk[idx] = sMqk[idx];
        }
    }
    if constexpr (!FmaFront) if (ready != nullptr) {
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            atomicExch(ready + ws_idx, 1);
        }
    }
    // The normal serial/CP host path does not use ready flags.  Before a
    // multi-chunk CTA overwrites shared storage for its next chunk, wait for
    // every thread to finish the current workspace stores.  Skip the final
    // iteration, and avoid a second barrier when the ready path already issued
    // one above.
    if constexpr (ChunksPerCTA > 1) {
        if (ready == nullptr && cpc_iter + 1 < ChunksPerCTA)
            __syncthreads();
    }
    }
}

}  // namespace flashkda_cute

namespace flashkda_cute {
// Query the storage convention from the device compilation pass, where the
// target architecture macro is defined. The debug prepare binding uses this
// to return a logical row-major field on both generations.
__global__ void flashkda_probe_kr_preswizzled(int* out) {
    if (threadIdx.x == 0) {
#if defined(FLASHKDA_ARCH_150)
        *out = 1;
#else
        *out = 0;
#endif
    }
}
}  // namespace flashkda_cute

bool flashkda_kr_is_preswizzled(hggcStream_t stream) {
    const int device = flashkda::host_runtime::current_device();
    // The device compilation pass owns this ABI. Publish only a successfully
    // observed value, and never reuse another logical device's embedded image.
    static std::mutex mutex;
    static std::map<int, int> cached;
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = cached.find(device);
    if (found != cached.end()) return found->second != 0;

    auto check_cuda = [device](hggcError_t status, const char* operation) {
        if (status != hggcSuccess)
            flashkda::host_runtime::throw_ppu_query_error(operation, device, status);
    };
    hggcStreamCaptureStatus capture = hggcStreamCaptureStatusNone;
    check_cuda(hggcStreamIsCapturing(stream, &capture),
               "KR layout probe hggcStreamIsCapturing");
    if (capture != hggcStreamCaptureStatusNone) {
        throw std::runtime_error(
            "FlashKDApro KR layout probe is cold during CUDA graph capture; "
            "run flashkda_prepare once in eager mode on this device before capture");
    }

    int* device_result = nullptr;
    int host_result = -1;
    check_cuda(hggcMalloc(&device_result, sizeof(int)), "KR layout probe hggcMalloc");
    try {
        flashkda_cute::flashkda_probe_kr_preswizzled<<<1, 1, 0, stream>>>(device_result);
        check_cuda(hggcGetLastError(), "KR layout probe launch");
        check_cuda(hggcMemcpyAsync(&host_result, device_result, sizeof(int),
                                   hggcMemcpyDeviceToHost, stream),
                   "KR layout probe hggcMemcpyAsync");
        check_cuda(hggcStreamSynchronize(stream), "KR layout probe hggcStreamSynchronize");
        const hggcError_t free_status = hggcFree(device_result);
        device_result = nullptr;
        check_cuda(free_status, "KR layout probe hggcFree");
    } catch (...) {
        if (device_result != nullptr) hggcFree(device_result);
        throw;
    }
    if (host_result != 0 && host_result != 1)
        throw std::runtime_error("FlashKDApro KR layout probe returned an invalid storage convention");
    cached.emplace(device, host_result);
    return host_result != 0;
}

void launch_flashkda_zero_padded_output_tail(
    void* out, int T_total, int H, int N,
    const int32_t* cu_seqlens, int valid_tokens,
    hggcStream_t stream) {
    // A known exact/full endpoint avoids even the lightweight launch. Unknown
    // endpoints use -1 and are checked by the device kernel for compatibility
    // with standalone callers that only provide device cu_seqlens.
    if (valid_tokens >= T_total || T_total <= 0 || H <= 0) return;
    constexpr int THREADS = 128;
    flashkda_cute::flashkda_zero_padded_output_tail<<<1, THREADS, 0, stream>>>(
        reinterpret_cast<cutlass::bfloat16_t*>(out),
        T_total, H, N, cu_seqlens, valid_tokens);
}

// launch wrapper
void launch_flashkda_prepare(
    const void* q, const void* k, const void* g_in, const void* beta,
    const float* A_log, const float* dt_bias, float scale, float gate_scale,
    int T_total, int H, int beta_stride,
    int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen,
    int tile_offset, int tile_count, int head_offset, int head_count,
    int* ready, void* ws, bool force_plain, bool cp_fma_front,
    bool fp16_kd_qd, flashkda::KrWorkspaceLayout kr_layout,
    hggcStream_t stream) {
    using BF16 = cutlass::bfloat16_t;
    int smem = sizeof(flashkda_cute::K1Smem);
    const bool is_tc01 =
        flashkda::host_runtime::current_device_is_tc01();
    const bool use_tc01_raw_krt =
        kr_layout == flashkda::KrWorkspaceLayout::kContiguousRawKrT;
    const bool use_tc01_affine_p =
        kr_layout == flashkda::KrWorkspaceLayout::kContiguousAffineP;
    const bool force_normal_raw =
        kr_layout == flashkda::KrWorkspaceLayout::kNormalRaw;
    const bool force_normal_affine =
        kr_layout == flashkda::KrWorkspaceLayout::kNormalAffineP;
    const bool use_tc01_contiguous =
        use_tc01_raw_krt || use_tc01_affine_p;
    const bool force_affine = force_normal_affine || use_tc01_affine_p;
    // A separate builder launch costs 2-5% for tiny batches on PPU1.5. Paired
    // measurements cross over by N=64 even at H=1, so the upstream tile-prefix
    // path is used automatically for large varlen batches.
    bool use_prefix = is_varlen && (N >= 64);
    bool k2_uses_prefix = is_varlen && (N >= 64);
    int* tile_prefix = reinterpret_cast<int*>(
        reinterpret_cast<unsigned char*>(ws)
        + int64_t(H) * total_tiles * flashkda::WS_SLOT_BYTES);
    if (use_prefix || k2_uses_prefix) {
        flashkda_cute::flashkda_build_tile_prefix<<<1, 32, 0, stream>>>(
            cu_seqlens, tile_prefix, N);
    }
    const char* ethreads = getenv("PLA_FLASHKDA_K1_THREADS");
    // Paired E2E measurements and ACU profiling show that 128 threads reduce
    // prepare time on three disjoint regions:
    //   * the promoted fixed low-head target shapes; and
    //   * TP1/TP2 large-batch, long-sequence varlen shapes.  On the latter,
    //     exact production N=17/N=218 endpoints improve 5.8--12.5% in full
    //     FlashKDApro calls (both state modes, 20 samples).  ACU shows that the
    //     256-thread kernel is already at 97--98% achieved occupancy and is
    //     dominated by CTA-wide sync stalls, so its extra four warps add
    //     barrier work instead of latency hiding.  N=17 covers the linear
    //     cu_seqlens lookup and N=218 the tile-prefix/binary-search path.
    //   * short/small Kimi-K3 work.  The complete T<=4096 varlen bucket has 55
    //     exact production shapes per TP and two state modes.  K1/128 improves
    //     all 110 rows at every TP: the worst gains are 5.59/3.78/1.37/1.72%
    //     for H=96/48/24/12.  Fixed batches cross over later as H falls, so use
    //     the measured monotone side of each crossover rather than forcing the
    //     low-head tiny calls (where 256 threads still win).
    // Preserve 256 outside those measured regions and retain the explicit
    // environment override for validation/rollback.
    // The int32/strided-input varlen specialization occupies 80/160
    // VREG/SREG with 256 threads on tc02, limiting it to 75% occupancy.  The
    // 128-thread specialization wins across the complete Kimi case-list range:
    // H=96/48/24/12, N=1/56 and T=8K/17K were all 6--10% faster end-to-end.
    // Keep tc01 on its separately calibrated gates below; this promotion is a
    // PPU1.5 result and must not be extrapolated across architectures.
    // ZW810E complete-case-list validation extends the earlier short/large-N
    // samples to every production varlen region: forcing 128 threads improves
    // aggregate KDA time by 8.8% (5,376/5,376 cases pass), with gains in every
    // H/N/T aggregate bucket.  Keep the H12/N2--3 short 64-thread exception
    // below, which the global 128-thread experiment intentionally overrode.
    bool varlen_128 = is_varlen;
    bool short_fixed_128 = !is_varlen && N == 1 && T_total <= 4096
        && ((H == 96 && T_total >= 128)
            || (H == 48 && T_total >= 192)
            || (H == 24 && T_total >= 448)
            || (H == 12 && T_total >= 512));
    bool auto_128 = (!is_varlen && N == 1 && T_total >= 4096
                     && (H == 6 || H == 12))
        || (is_varlen && N >= 16 && T_total >= 16384
            && (H == 48 || H == 96))
        || varlen_128 || short_fixed_128;
    int auto_threads = auto_128 ? 128 : 256;
    // ZW810E has 64 CUs.  On the exact Kimi short bucket, H12/N2--3 gains
    // 2.5%/2.2% geometrically from four-warp -> two-warp K1 while H12/N1 and
    // N>=11 prefer 128 threads.  Keep this tc01-only interval independent of
    // the 39-CU tc02 crossovers above.
    if (is_tc01 && is_varlen
        && T_total <= 4096 && H == 12 && N >= 2 && N <= 3) {
        auto_threads = 64;
    }
    int requested_threads = ethreads && ethreads[0]
        ? atoi(ethreads) : auto_threads;
    if (requested_threads != 64 && requested_threads != 96
        && requested_threads != 128)
        requested_threads = 256;
    // The host snapshots the private workspace layout once. Later diagnostic
    // env reads must not steer this producer away from the specialization that
    // implements that ABI.
    if (use_tc01_contiguous || force_normal_affine) requested_threads = 128;
    // PR #4605's K1_CPC=2 has independent crossovers on the two PPU
    // generations.  ZW810E fresh-process E2E measurements promote fixed N1
    // H6 from 6K (+2.55%) and H12 from 8K (+1.91%); both 4K shapes are flat.
    // The CP force-plain path is also flat, so keep CPC1 there.  On the 39-CU
    // PPU1.5, retain the established narrow H12 >=128K gate.  CPC4/8 are
    // correct but consistently slower than CPC2 on tc01.  CPC1 remains the
    // environment rollback and the only packed implementation.
    bool auto_cpc2_tc01 = is_tc01 && !force_plain && !is_varlen && N == 1
        && ((H == 6 && T_total >= 6144)
            || (H == 12 && T_total >= 8192));
    bool auto_cpc2_tc02 = !is_tc01 && !is_varlen && N == 1
        && H == 12 && T_total >= 131072;
    bool auto_cpc2 = auto_cpc2_tc01 || auto_cpc2_tc02;
    const char* ecpc = getenv("PLA_FLASHKDA_K1_CPC");
    int requested_cpc = ecpc && ecpc[0] ? atoi(ecpc) : (auto_cpc2 ? 2 : 1);
    if (requested_cpc != 2 || requested_threads != 128 || is_varlen)
        requested_cpc = 1;
    if (use_tc01_contiguous) requested_cpc = 1;
    // Affine K1 (P=kr^T@INV@diag(beta), O=Mqk@INV@diag(beta)) is the promoted
    // precompute for the low-head Batch=1 target and needs the 128-thread shape.
    // It must match exactly what the DK4/leader BV16 recurrence consumes, so any
    // explicit K2 env override that steers off that path disables it (the plain
    // recurrence would then read raw INV/Mqk/beta, which affine does not emit).
    const char* edk = getenv("PLA_FLASHKDA_K2_DKSPLIT");
    const char* edkl = getenv("PLA_FLASHKDA_K2_DKLEADER");
    const char* eds = getenv("PLA_FLASHKDA_K2_DVSPLIT");
    const char* enows = getenv("PLA_FLASHKDA_K2_NOWS");
    // OperatorsForge supplies an explicit [0,T] cu_seqlens tensor even for a
    // single sequence.  That ABI is mathematically identical to fixed N1 and
    // should use the same affine precompute for tc01's long H12 critical chain.
    bool affine_shape = N == 1 && T_total >= 4096
        && ((!is_varlen && (H == 6 || H == 12))
            || (is_tc01 && is_varlen && H == 12));
    bool affine_dk4 = !(edk && edk[0]) || atoi(edk) == 4;
    bool affine_leader = !(edkl && edkl[0]) || atoi(edkl) != 0;
    bool affine_bv16 = !(eds && eds[0]) || atoi(eds) == 16;
    bool affine_nows = !(enows && enows[0]) || atoi(enows) != 0;
    // force_plain is the host's veto: the CP pipeline runs the plain
    // (non-dedicated-leader) recurrence template, so it needs raw INV/Mqk/beta.
    bool precompute_affine = !force_plain && affine_shape && affine_dk4
        && affine_leader && affine_bv16 && affine_nows;
    if (force_normal_raw || use_tc01_raw_krt) precompute_affine = false;
    if (force_affine) precompute_affine = true;
    // tc01 split inspired by FLA's instruction-level policy:
    // triangular K=128 dots use FFMA on 64 generic CUs, while the dense Neumann
    // tail remains MMA on the 20 TensorCell CUs.  The automatic gate is the
    // conservative all-win region from the paired case-list sweep.  H12/N4
    // was validated separately over every T>=8192 inventory row and therefore
    // joins the automatic gate without widening H24/H48/H96 to N4.  Keep an
    // explicit environment rollback for production diagnosis.
    const char* efma_split = getenv("PLA_FLASHKDA_K1_FMA_SPLIT");
    // The launch-overhead crossover is head dependent.  These are the first
    // monotone-safe totals from the complete 5,376-case paired sweep; the
    // additional grid-work floor protects ragged boundary shapes.
    bool fma_split_crossover = (H == 12 && T_total >= 8192)
        || (H == 24 && T_total >= 2048)
        || (H == 48 && T_total >= 512)
        || H == 96;
    bool auto_fma_split = (N <= 3 || (H == 12 && N == 4))
        && fma_split_crossover
        && int64_t(tile_count) * int64_t(head_count) >= 384;
    // Physical kr layout is independent of the dot-product implementation.
    // Existing long low-head gates already require the FMA split in the host;
    // H96 keeps its separately measured N<=3 crossover and uses the ordinary
    // monolithic prepare for wider fan-in.
    const bool tc01_h8_cp_fma = cp_fma_front && force_plain && is_tc01
        && is_varlen && N == 1 && H == 8 && T_total >= 4096;
    if (tc01_h8_cp_fma) {
        requested_threads = 128;
        requested_cpc = 1;
    }
    bool request_fma_split = tc01_h8_cp_fma
        || (efma_split && efma_split[0]
            ? atoi(efma_split) != 0 : auto_fma_split);
    bool fma_split = is_tc01 && request_fma_split
        && (!force_plain || tc01_h8_cp_fma)
        && ready == nullptr && requested_threads == 128
        && requested_cpc == 1 && !precompute_affine;
    if (fma_split) {
        bool transpose_kr = use_tc01_raw_krt;
        auto launch_split = [&](auto varlen, auto prefix, auto kr_transpose) {
            auto front = flashkda_cute::flashkda_prepare_kernel<
                128, decltype(varlen)::value, decltype(prefix)::value,
                false, 1, true, decltype(kr_transpose)::value>;
            hggcFuncSetAttribute(
                front, hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
            dim3 grid(tile_count, head_count);
            front<<<grid, 128, smem, stream>>>(
                reinterpret_cast<const BF16*>(q), reinterpret_cast<const BF16*>(k),
                reinterpret_cast<const BF16*>(g_in), reinterpret_cast<const BF16*>(beta),
                A_log, dt_bias, scale, gate_scale, T_total, H, beta_stride,
                N, total_tiles, tile_count, cu_seqlens,
                tile_prefix, tile_offset, head_offset, ready, fp16_kd_qd,
                reinterpret_cast<BF16*>(ws));
            auto tail = flashkda_cute::flashkda_neumann_tail_kernel<
                decltype(varlen)::value, decltype(prefix)::value>;
            tail<<<grid, 32, 0, stream>>>(
                N, total_tiles, tile_count, cu_seqlens, tile_prefix,
                tile_offset, head_offset, reinterpret_cast<BF16*>(ws));
        };
        if (is_varlen) {
            if (use_prefix) {
                if (transpose_kr)
                    launch_split(std::true_type{}, std::true_type{}, std::true_type{});
                else
                    launch_split(std::true_type{}, std::true_type{}, std::false_type{});
            } else {
                if (transpose_kr)
                    launch_split(std::true_type{}, std::false_type{}, std::true_type{});
                else
                    launch_split(std::true_type{}, std::false_type{}, std::false_type{});
            }
        } else {
            launch_split(std::false_type{}, std::false_type{}, std::false_type{});
        }
        return;
    }
    auto K = [&](auto nt, auto varlen, auto prefix, auto affine, auto cpc,
                 auto kr_contiguous) {
        constexpr int NT = decltype(nt)::value;
        constexpr int CPC = decltype(cpc)::value;
        auto kern = flashkda_cute::flashkda_prepare_kernel<
            NT, decltype(varlen)::value, decltype(prefix)::value,
            decltype(affine)::value, CPC, false,
            decltype(kr_contiguous)::value>;
        hggcFuncSetAttribute(kern, hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
        dim3 grid((tile_count + CPC - 1) / CPC, head_count);
        kern<<<grid, NT, smem, stream>>>(
            reinterpret_cast<const BF16*>(q), reinterpret_cast<const BF16*>(k),
            reinterpret_cast<const BF16*>(g_in), reinterpret_cast<const BF16*>(beta),
            A_log, dt_bias, scale, gate_scale, T_total, H, beta_stride,
            N, total_tiles, tile_count, cu_seqlens,
            tile_prefix, tile_offset, head_offset, ready, fp16_kd_qd,
            reinterpret_cast<BF16*>(ws));
    };
    // affine is only ever enabled on the 128-thread shape; every other thread
    // count uses the plain (non-affine) prepare.
    auto pick_shape = [&](auto nt, auto affine, auto cpc, auto kr_contiguous) {
        if (is_varlen) {
            if (use_prefix)
                K(nt, std::true_type{}, std::true_type{}, affine, cpc,
                  kr_contiguous);
            else
                K(nt, std::true_type{}, std::false_type{}, affine, cpc,
                  kr_contiguous);
        } else {
            K(nt, std::false_type{}, std::false_type{}, affine, cpc,
              kr_contiguous);
        }
    };
    if (requested_threads == 64)
        pick_shape(std::integral_constant<int, 64>{}, std::false_type{},
                   std::integral_constant<int, 1>{}, std::false_type{});
    else if (requested_threads == 96)
        pick_shape(std::integral_constant<int, 96>{}, std::false_type{},
                   std::integral_constant<int, 1>{}, std::false_type{});
    else if (requested_threads == 128) {
        // Instantiate CPC=2 only for fixed prepare.  The rejected CPC=4/8 and
        // packed variants would add device code without a promoted dispatch.
        if (!is_varlen && requested_cpc == 2) {
            if (precompute_affine)
                K(std::integral_constant<int, 128>{}, std::false_type{},
                  std::false_type{}, std::true_type{},
                  std::integral_constant<int, 2>{}, std::false_type{});
            else
                K(std::integral_constant<int, 128>{}, std::false_type{},
                  std::false_type{}, std::false_type{},
                  std::integral_constant<int, 2>{}, std::false_type{});
        } else if (precompute_affine) {
            if (use_tc01_affine_p)
                pick_shape(std::integral_constant<int, 128>{}, std::true_type{},
                           std::integral_constant<int, 1>{}, std::true_type{});
            else
                pick_shape(std::integral_constant<int, 128>{}, std::true_type{},
                           std::integral_constant<int, 1>{}, std::false_type{});
        } else {
            if (use_tc01_raw_krt)
                pick_shape(std::integral_constant<int, 128>{}, std::false_type{},
                           std::integral_constant<int, 1>{}, std::true_type{});
            else
                pick_shape(std::integral_constant<int, 128>{}, std::false_type{},
                           std::integral_constant<int, 1>{}, std::false_type{});
        }
    }
    else
        pick_shape(std::integral_constant<int, 256>{}, std::false_type{},
                   std::integral_constant<int, 1>{}, std::false_type{});
}
