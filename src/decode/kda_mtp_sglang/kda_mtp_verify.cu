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

// PPU1.5 port of fused_kda_decode_mtp_dspark (SGLang KDA MTP-verify).
//
// Ported verbatim in structure from the H20 (SM90) CUDA C++ version
// (csrc/_h20_kda_verify.cu.orig). This kernel uses NO tensor cells -- it is a
// CUDA-core delta-rule recurrence with warp-shuffle reductions and a
// register-resident state -- so the port only swaps device intrinsics and
// retargets the launch config; the algorithm is unchanged. Device intrinsics
// used are pla's standard __expf / rsqrtf / __bfloat16* conversions plus a
// local ppu_sigmoid / ppu_silu (mirroring
// src/decode/k-last/fused_kda_decode_mega_kernel.cuh); there is no
// per-namespace compat shim.
//
// Scope (fixed K3/dspark contract): head_dim 128 for q/k/v,
// num_q_heads == num_k_heads == num_v_heads, conv width 4, exactly
// 1 + num_spec dense tokens per request, CACHE_RING, no fused output norm.
//
// Arithmetic follows the DSL/H20, including the two places it differs from the
// commit-side fold:
//   * q/k normalise with rsqrt-multiply, where the fold uses the division form.
//   * the ring records PRE-norm k and the PRE-exp gate; the fold re-derives both.
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <cstdint>

#include "kda_mtp_abi.h"

namespace {

using BF16 = __ppu_bfloat16;

// PPU sigmoid uses the hardware __ppu_sgmdf intrinsic; the host-side fallback
// matches the H20 reference. Same pattern as
// src/decode/k-last/fused_kda_decode_mega_kernel.cuh::ppu_sigmoid.
__device__ __forceinline__ float ppu_sigmoid(float x) {
#if defined(__HGGC_ARCH__)
  return __ppu_sgmdf(x);
#else
  return 1.0f / (1.0f + __expf(-x));
#endif
}
__device__ __forceinline__ float ppu_silu(float x) { return x * ppu_sigmoid(x); }

constexpr int DIM = 128;
constexpr int CONV_W = 4;
constexpr int CONV_WIN = CONV_W - 1;
constexpr float L2_EPS = 1e-6f;

#ifndef KDA_V_THREADS
#define KDA_V_THREADS 256
#endif
#ifndef KDA_V_KGRPS
// PPU1.5 sweep (card 6, THREADS x KGRPS over 128/256/512 x 4/8/16/32): KGRPS=4
// with THREADS=256 is the consistent best (bs=1/8/16/64: 28.6/41.0/70.1/212.0us
// vs 29.5/42.7/73.9/214.6 at KGRPS=8). KGRPS=4 -> VROWS=64, VPASSES=2, a 2-step
// crosslane butterfly instead of 3; the extra state registers cost less than the
// reduction they save at this occupancy. (H20's best was 8; PPU differs.)
#define KDA_V_KGRPS 4
#endif
constexpr int THREADS = KDA_V_THREADS;
constexpr int KGRPS = KDA_V_KGRPS;
constexpr int VROWS = THREADS / KGRPS;
constexpr int K_PER_THREAD = DIM / KGRPS;
constexpr int F4_PER_THREAD = K_PER_THREAD / 4;
constexpr int VPASSES = DIM / VROWS;

// PPU1.5 structural knob: at low batch the verify grid (H x N) is far under one
// wave (bs=8: 96 blocks / 39 CU = 0.62 waves), so memory/sync latency cannot be
// hidden. VSPLIT partitions the VPASSES value-row passes across VSPLIT blocks on
// grid.z, multiplying the grid by VSPLIT. Each split recomputes phase-1
// (conv/norm/gate) locally but only split 0 writes the shared rings / conv
// windows, preserving single-write semantics.
//
// KDA_V_VSPLIT = 0 -> AUTO: the host picks VSPLIT=2 when H*N < KDA_V_VSPLIT_GATE
// (measured crossover ~256: bs<=16 wins by 3-15%, bs>=32 loses to duplicated
// phase-1), else 1. Set 1 or 2 to force (used by the sweep).
#ifndef KDA_V_VSPLIT
#define KDA_V_VSPLIT 0
#endif
constexpr int VSPLIT_FORCE = KDA_V_VSPLIT;
constexpr bool CAN_SPLIT2 = (VPASSES % 2 == 0);
#ifndef KDA_V_VSPLIT_GATE
#define KDA_V_VSPLIT_GATE 256
#endif

static_assert(THREADS % 32 == 0, "whole warps");
static_assert(THREADS % KGRPS == 0, "KGRPS must divide THREADS");
static_assert(KGRPS >= 1 && KGRPS <= 32 && 32 % KGRPS == 0,
              "the KGRPS threads of a v row must share a warp and tile the row");
static_assert(DIM % VROWS == 0, "v rows must tile DIM");

__device__ __forceinline__ float bfly_add(float x, int width) {
  for (int off = width >> 1; off >= 1; off >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, off, 32);
  return x;
}

struct Smem {
  float q[DIM];
  float k[DIM];
  float decay[DIM];
  float v[DIM];
  float beta;
  float red[THREADS / 32 + 1];
  float wq[CONV_WIN][DIM];
  float wk[CONV_WIN][DIM];
  float wv[CONV_WIN][DIM];
  float cw_q[CONV_W][DIM];
  float cw_k[CONV_W][DIM];
  float cw_v[CONV_W][DIM];
};

// Layout strides come from the torch-free ABI so the host wrapper and this
// device TU agree on the exact struct passed by value into the kernel.
using pla::kda_mtp::ConvStrides;
using pla::kda_mtp::WindowStrides;
using pla::kda_mtp::VerifyStrides;

template <int VS>
__global__ __launch_bounds__(THREADS) void verify_kernel(
    const BF16* __restrict__ x_q, const BF16* __restrict__ x_k,
    const BF16* __restrict__ x_v, const BF16* __restrict__ g_in,
    const BF16* __restrict__ beta_in, const float* __restrict__ w_q,
    const float* __restrict__ w_k, const float* __restrict__ w_v,
    const BF16* __restrict__ cs_q, const BF16* __restrict__ cs_k,
    const BF16* __restrict__ cs_v, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, const float* __restrict__ h0,
    BF16* __restrict__ out, BF16* __restrict__ ic_q,
    BF16* __restrict__ ic_k, BF16* __restrict__ ic_v,
    BF16* __restrict__ ring_rawv, BF16* __restrict__ ring_rawk,
    float* __restrict__ ring_g, float* __restrict__ ring_beta,
    const int32_t* __restrict__ slot_idx, const int32_t* __restrict__ scratch_idx,
    const int32_t* __restrict__ cu_seqlens, int H, int T_LOOP, int L, float scale,
    float lower_bound, VerifyStrides strides) {
  const int i_hv = blockIdx.x;
  const int i_n = blockIdx.y;
  const int i_vs = blockIdx.z;              // v-split index (0..VS-1)
  const bool ring_writer = (i_vs == 0);     // only split 0 writes shared rings
  constexpr int PASSES_PB = VPASSES / VS;   // value-row passes per block
  static_assert(VPASSES % VS == 0, "VS must divide VPASSES");
  const int tid = threadIdx.x;

  const int bos = cu_seqlens[i_n];
  const int n_tok = cu_seqlens[i_n + 1] - bos;
  const int slot = slot_idx[i_n];
  const long head_off = (long)i_hv * DIM;

  // CUDA-graph padding rows: zero the output, touch no cache.
  if (slot < 0) {
    if (ring_writer)
      for (int t = 0; t < T_LOOP; ++t)
        for (int j = tid; j < DIM; j += THREADS)
          out[(long)(bos + t) * H * DIM + head_off + j] = __float2bfloat16(0.f);
    return;
  }

  const int srow = scratch_idx[i_n];
  extern __shared__ char smem_raw[];
  Smem& sh = *reinterpret_cast<Smem*>(smem_raw);

  const int v_local = tid / KGRPS;
  const int k_grp = tid % KGRPS;

  for (int j = tid; j < DIM; j += THREADS) {
#pragma unroll
    for (int w = 0; w < CONV_W; ++w) {
      sh.cw_q[w][j] = w_q[(head_off + j) * CONV_W + w];
      sh.cw_k[w][j] = w_k[(head_off + j) * CONV_W + w];
      sh.cw_v[w][j] = w_v[(head_off + j) * CONV_W + w];
    }
#pragma unroll
    for (int w = 0; w < CONV_WIN; ++w) {
      sh.wq[w][j] = __bfloat162float(cs_q[
          slot * strides.cs_q.slot + (head_off + j) * strides.cs_q.channel +
          w * strides.cs_q.window]);
      sh.wk[w][j] = __bfloat162float(cs_k[
          slot * strides.cs_k.slot + (head_off + j) * strides.cs_k.channel +
          w * strides.cs_k.window]);
      sh.wv[w][j] = __bfloat162float(cs_v[
          slot * strides.cs_v.slot + (head_off + j) * strides.cs_v.channel +
          w * strides.cs_v.window]);
    }
  }

  float4 st[PASSES_PB][F4_PER_THREAD];
#pragma unroll
  for (int p = 0; p < PASSES_PB; ++p) {
    const int v = v_local + (i_vs * PASSES_PB + p) * VROWS;
    const float4* g4 = reinterpret_cast<const float4*>(
        h0 + ((long)slot * H + i_hv) * DIM * DIM + (long)v * DIM);
#pragma unroll
    for (int i = 0; i < F4_PER_THREAD; ++i) st[p][i] = g4[k_grp + KGRPS * i];
  }

  const float exp_A = __expf(A_log[i_hv]);
  __syncthreads();

  for (int t = 0; t < T_LOOP; ++t) {
    // the DSL clamps the token index so a short request re-reads its last token
    const int tok = bos + (t < n_tok ? t : n_tok - 1);
    const long ro_base = ((long)slot * H + i_hv) * (long)L * DIM + (long)t * DIM;

    for (int j = tid; j < DIM; j += THREADS) {
      const float xq = __bfloat162float(x_q[tok * strides.x_q + head_off + j]);
      const float xk = __bfloat162float(x_k[tok * strides.x_k + head_off + j]);
      const float xv = __bfloat162float(x_v[tok * strides.x_v + head_off + j]);

      float cq = xq * sh.cw_q[CONV_W - 1][j];
      float ck = xk * sh.cw_k[CONV_W - 1][j];
      float cv = xv * sh.cw_v[CONV_W - 1][j];
#pragma unroll
      for (int w = 0; w < CONV_WIN; ++w) {
        cq += sh.wq[w][j] * sh.cw_q[w][j];
        ck += sh.wk[w][j] * sh.cw_k[w][j];
        cv += sh.wv[w][j] * sh.cw_v[w][j];
      }
      cq = ppu_silu(cq);
      ck = ppu_silu(ck);
      cv = ppu_silu(cv);

      const float gk = lower_bound *
                       ppu_sigmoid(exp_A * (__bfloat162float(g_in[
                                              tok * strides.g + head_off + j]) +
                                              dt_bias[head_off + j]));
      if (ring_writer) {
        ring_rawk[ro_base + j] = __float2bfloat16(ck);
        ring_rawv[ro_base + j] = __float2bfloat16(cv);
        ring_g[ro_base + j] = gk;
      }

      sh.q[j] = cq;
      sh.k[j] = ck;
      sh.v[j] = cv;
      sh.decay[j] = __expf(gk);

      // advance, then snapshot: the window AFTER consuming x_t is what the conv
      // rollback restores
      sh.wq[0][j] = sh.wq[1][j]; sh.wq[1][j] = sh.wq[2][j]; sh.wq[2][j] = xq;
      sh.wk[0][j] = sh.wk[1][j]; sh.wk[1][j] = sh.wk[2][j]; sh.wk[2][j] = xk;
      sh.wv[0][j] = sh.wv[1][j]; sh.wv[1][j] = sh.wv[2][j]; sh.wv[2][j] = xv;
      if (ring_writer) {
        const int64_t iq = srow * strides.ic_q.row + t * strides.ic_q.step +
                           (head_off + j) * strides.ic_q.channel;
        const int64_t ik = srow * strides.ic_k.row + t * strides.ic_k.step +
                           (head_off + j) * strides.ic_k.channel;
        const int64_t iv = srow * strides.ic_v.row + t * strides.ic_v.step +
                           (head_off + j) * strides.ic_v.channel;
#pragma unroll
        for (int w = 0; w < CONV_WIN; ++w) {
          ic_q[iq + w * strides.ic_q.window] = __float2bfloat16(sh.wq[w][j]);
          ic_k[ik + w * strides.ic_k.window] = __float2bfloat16(sh.wk[w][j]);
          ic_v[iv + w * strides.ic_v.window] = __float2bfloat16(sh.wv[w][j]);
        }
      }
    }
    if (tid == 0) {
      sh.beta = ppu_sigmoid(__bfloat162float(beta_in[tok * strides.beta + i_hv]));
      if (ring_writer)
        ring_beta[((long)slot * H + i_hv) * (long)L + t] = sh.beta;
    }
    __syncthreads();  // phase 1 done writing sh.q/k/v/decay

    // L2 norm.  On the low-batch VS>=2 path, every warp redundantly computes
    // the same 128-channel reduction.  The extra TSM reads/FALU work are cheap
    // at this under-filled grid and avoid the dominant cross-warp publication
    // barrier.  VS=1 keeps the single-warp + smem broadcast path because its
    // many resident blocks already hide that barrier.
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float nq, nk;
    if constexpr (VS >= 2) {
      float lq = 0.f, lk = 0.f;
#pragma unroll
      for (int m = 0; m < DIM / 32; ++m) {
        const float qq = sh.q[lane + m * 32];
        const float kk = sh.k[lane + m * 32];
        lq += qq * qq;
        lk += kk * kk;
      }
      lq = bfly_add(lq, 32);
      lk = bfly_add(lk, 32);
      nq = 0.f;
      nk = 0.f;
      if (lane == 0) {
        nq = rsqrtf(lq + L2_EPS) * scale;
        nk = rsqrtf(lk + L2_EPS);
      }
      nq = __shfl_sync(0xffffffffu, nq, 0, 32);
      nk = __shfl_sync(0xffffffffu, nk, 0, 32);
    } else {
      if (wid == 0) {
        float lq = 0.f, lk = 0.f;
#pragma unroll
        for (int m = 0; m < DIM / 32; ++m) {
          const float qq = sh.q[lane + m * 32];
          const float kk = sh.k[lane + m * 32];
          lq += qq * qq;
          lk += kk * kk;
        }
        lq = bfly_add(lq, 32);
        lk = bfly_add(lk, 32);
        if (lane == 0) {
          sh.red[0] = rsqrtf(lq + L2_EPS) * scale;
          sh.red[1] = rsqrtf(lk + L2_EPS);
        }
      }
      __syncthreads();
      nq = sh.red[0];
      nk = sh.red[1];
    }
    // VS>=2 (low batch, sync-bound): fold the L2 norm into phase-2 as post-
    // reduction scalars (norm is linear), dropping the normalize smem pass and
    // one __syncthreads per token. VS<2 (high batch, occupancy-bound): keep the
    // original in-place normalize pass -- measured faster there, since the extra
    // live scalars cost occupancy where syncs are already hidden by many warps.
    if constexpr (VS < 2) {
      for (int j = tid; j < DIM; j += THREADS) {
        sh.q[j] *= nq;
        sh.k[j] *= nk;
      }
      __syncthreads();
    }
    const float kfold = (VS >= 2) ? nk : 1.0f;   // compile-time select
    const float qfold = (VS >= 2) ? nq : 1.0f;

    const float4* sq4 = reinterpret_cast<const float4*>(sh.q);
    const float4* sk4 = reinterpret_cast<const float4*>(sh.k);
    const float4* sd4 = reinterpret_cast<const float4*>(sh.decay);
    const float beta = sh.beta;

#pragma unroll
    for (int p = 0; p < PASSES_PB; ++p) {
      const int v = v_local + (i_vs * PASSES_PB + p) * VROWS;
      float shk = 0.f;
#pragma unroll
      for (int i = 0; i < F4_PER_THREAD; ++i) {
        const int f4 = k_grp + KGRPS * i;
        const float4 d4 = sd4[f4], kk = sk4[f4];
        st[p][i].x *= d4.x; st[p][i].y *= d4.y;
        st[p][i].z *= d4.z; st[p][i].w *= d4.w;
        // At VS=1 the grid is large enough that the fixed FMA chain wins
        // (~1.3-1.8% at bs=32/64).  On the latency-bound VS=2 path it adds
        // ~1% at bs=1, so keep the compiler's original expression there.
        if constexpr (VS < 2) {
          shk = fmaf(st[p][i].w, kk.w,
                     fmaf(st[p][i].z, kk.z,
                          fmaf(st[p][i].y, kk.y,
                               fmaf(st[p][i].x, kk.x, shk))));
        } else {
          shk += st[p][i].x * kk.x + st[p][i].y * kk.y + st[p][i].z * kk.z +
                 st[p][i].w * kk.w;
        }
      }
      shk = bfly_add(shk, KGRPS) * kfold;   // kfold = nk (VS>=2) or 1 (folded)
      const float vn = (sh.v[v] - shk) * beta;
      const float vnk = vn * kfold;         // rank-1: st += vn * (kfold*k)

      float shq = 0.f;
#pragma unroll
      for (int i = 0; i < F4_PER_THREAD; ++i) {
        const int f4 = k_grp + KGRPS * i;
        const float4 kk = sk4[f4], qq = sq4[f4];
        st[p][i].x = fmaf(vnk, kk.x, st[p][i].x);
        st[p][i].y = fmaf(vnk, kk.y, st[p][i].y);
        st[p][i].z = fmaf(vnk, kk.z, st[p][i].z);
        st[p][i].w = fmaf(vnk, kk.w, st[p][i].w);
        if constexpr (VS < 2) {
          shq = fmaf(st[p][i].w, qq.w,
                     fmaf(st[p][i].z, qq.z,
                          fmaf(st[p][i].y, qq.y,
                               fmaf(st[p][i].x, qq.x, shq))));
        } else {
          shq += st[p][i].x * qq.x + st[p][i].y * qq.y + st[p][i].z * qq.z +
                 st[p][i].w * qq.w;
        }
      }
      shq = bfly_add(shq, KGRPS) * qfold;   // qfold = nq (VS>=2) or 1
      if (k_grp == 0 && t < n_tok)
        out[(long)(bos + t) * H * DIM + head_off + v] = __float2bfloat16(shq);
    }
    __syncthreads();
  }
  // recurrent_state is read-only for verify: nothing is written back
}

}  // namespace

namespace pla {
namespace kda_mtp {
namespace {
template <int VS>
void launch_verify_vsplit(
    const void* x_q, const void* x_k, const void* x_v,
    const void* g_in, const void* beta_in,
    const float* w_q, const float* w_k, const float* w_v,
    const void* cs_q, const void* cs_k, const void* cs_v,
    const float* a_log, const float* dt_bias, const float* h0,
    void* out, void* ic_q, void* ic_k, void* ic_v,
    void* ring_rawv, void* ring_rawk, float* ring_g, float* ring_beta,
    const int32_t* slot_idx, const int32_t* scratch_idx,
    const int32_t* cu_seqlens,
    int H, int N, int t_loop, int ring_len,
    float scale, float lower_bound, VerifyStrides strides, hggcStream_t stream) {
  const int smem = static_cast<int>(sizeof(Smem));
  if (smem > 48 * 1024) {
    hggcFuncSetAttribute(verify_kernel<VS>,
                         hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  dim3 grid(H, N, VS);
  verify_kernel<VS><<<grid, THREADS, smem, stream>>>(
      static_cast<const BF16*>(x_q), static_cast<const BF16*>(x_k),
      static_cast<const BF16*>(x_v), static_cast<const BF16*>(g_in),
      static_cast<const BF16*>(beta_in), w_q, w_k, w_v,
      static_cast<const BF16*>(cs_q), static_cast<const BF16*>(cs_k),
      static_cast<const BF16*>(cs_v), a_log, dt_bias, h0,
      static_cast<BF16*>(out), static_cast<BF16*>(ic_q),
      static_cast<BF16*>(ic_k), static_cast<BF16*>(ic_v),
      static_cast<BF16*>(ring_rawv), static_cast<BF16*>(ring_rawk),
      ring_g, ring_beta, slot_idx, scratch_idx, cu_seqlens,
      H, t_loop, ring_len, scale, lower_bound, strides);
}
}  // namespace

void launch_verify(
    const void* x_q, const void* x_k, const void* x_v,
    const void* g_in, const void* beta_in,
    const float* w_q, const float* w_k, const float* w_v,
    const void* cs_q, const void* cs_k, const void* cs_v,
    const float* a_log, const float* dt_bias, const float* h0,
    void* out, void* ic_q, void* ic_k, void* ic_v,
    void* ring_rawv, void* ring_rawk, float* ring_g, float* ring_beta,
    const int32_t* slot_idx, const int32_t* scratch_idx,
    const int32_t* cu_seqlens,
    int H, int N, int t_loop, int ring_len,
    float scale, float lower_bound, VerifyStrides strides, hggcStream_t stream) {
#define PLA_VERIFY_ARGS                                                        \
  x_q, x_k, x_v, g_in, beta_in, w_q, w_k, w_v, cs_q, cs_k, cs_v, a_log,      \
      dt_bias, h0, out, ic_q, ic_k, ic_v, ring_rawv, ring_rawk, ring_g,        \
      ring_beta, slot_idx, scratch_idx, cu_seqlens, H, N, t_loop, ring_len,    \
      scale, lower_bound, strides, stream
  int vsplit = VSPLIT_FORCE;
  if (vsplit == 0) {
    vsplit = (CAN_SPLIT2 && static_cast<long>(H) * N < KDA_V_VSPLIT_GATE) ? 2 : 1;
  }
  if (vsplit == 2) {
    if constexpr (CAN_SPLIT2) launch_verify_vsplit<2>(PLA_VERIFY_ARGS);
    else launch_verify_vsplit<1>(PLA_VERIFY_ARGS);
  } else {
    launch_verify_vsplit<1>(PLA_VERIFY_ARGS);
  }
#undef PLA_VERIFY_ARGS
}

}  // namespace kda_mtp
}  // namespace pla
