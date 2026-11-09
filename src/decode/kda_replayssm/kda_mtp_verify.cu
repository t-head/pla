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
// 1 + num_spec dense tokens per request, CACHE_RING, optional fused gated
// output RMSNorm (a compile-time output mode, so the plain build is untouched).
//
// Arithmetic follows the DSL/H20, including the two places it differs from the
// commit-side fold:
//   * q/k normalise with rsqrt-multiply, where the fold uses the division form.
//   * the ring records PRE-norm k and the PRE-exp gate; the fold re-derives both.
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <cstdint>
#include <cmath>
#include <utility>

#include "kda_mtp_abi.h"

namespace {

using BF16 = __ppu_bfloat16;
using pla::kda_mtp::IndexArg;

// Branch-free: an int64 index's low word (== static_cast<int>, little endian)
// is int32 element 2*i. A wide/narrow branch made the compiler drain every
// outstanding scalar load before the next index load, serializing the cold
// slot / cu_seqlens misses at CTA start.
__device__ __forceinline__ int index_at(IndexArg idx, int i) {
  return __ldg(static_cast<const int*>(idx.data) + (i << (idx.wide ? 1 : 0)));
}

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

// A finite A_log may overflow expf. The mathematical product is still zero
// when the gate sum is zero; avoid forming inf * 0 before the sigmoid.
__device__ __forceinline__ float gate_product(float exp_a, float gate_sum) {
  return gate_sum == 0.f ? 0.f : exp_a * gate_sum;
}

constexpr int DIM = 128;
constexpr int CONV_W = 4;
constexpr int CONV_WIN = CONV_W - 1;
constexpr float L2_EPS = 1e-6f;

// Empirical FP32 conditioning guard, qualified by directed cancellation and
// checkpoint-derived replay. A large beta alone is not evidence of loss of
// precision: require the computed residual to be small relative to the terms
// that formed it. This is not a proof for all unrestricted finite inputs.
__device__ __forceinline__ bool precision_risk(float residual, float terms) {
  constexpr float max_finite = 0x1.fffffep127f;
  return !(residual <= max_finite && terms <= max_finite) ||
         (terms > 0.f && residual <= 1e-2f * terms);
}

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

__device__ __forceinline__ float bfly_max(float x, int width) {
  for (int off = width >> 1; off >= 1; off >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off, 32));
  return x;
}

using pla::kda_mtp::NativeVerify;
using pla::kda_mtp::OnormArgs;

// Output epilogue of the verify kernels. kPlain stores bf16 o. kNorm applies
// the gated output RMSNorm in the CTA; it needs all 128 value rows of a head,
// so only the unsplit (VS == 1) stepwise kernel uses it. kStage stores fp32 o
// to onorm.staging for onorm_kernel: split stepwise launches and the generic
// kernel, where an in-CTA norm on the serial token loop cost 13% over the
// plain kernel against 4% for the epilogue (H=12, N=32, 10 steps, ZW-M890P).
//
// The trailing pack carries only the active mode argument. SGLang plain
// takes no extra argument, SGLang norm modes take OnormArgs, and native
// replay takes NativeVerify with plain output.
enum class Out { kPlain, kNorm, kStage };

template <typename T, bool ENABLED, typename... Args>
__device__ __forceinline__ T verify_arg(const Args&... args) {
  if constexpr (ENABLED) return (args, ...);
  else return T{};
}

__device__ __forceinline__ float onorm_value(const OnormArgs& onorm, long tok,
                                             long channel, int v, float o,
                                             float rms) {
  const float gate = __bfloat162float(
      static_cast<const BF16*>(onorm.gate)[tok * onorm.gate_stride + channel]);
  return o * rms * onorm.weight[v] * ppu_sigmoid(gate);
}

// One warp per (token, head) row of the fp32 output staged by a split launch.
constexpr int ONORM_ROWS_PER_CTA = 8;
__global__ __launch_bounds__(32 * ONORM_ROWS_PER_CTA) void onorm_kernel(
    BF16* __restrict__ out, OnormArgs onorm, int rows, int H) {
  const int row = blockIdx.x * ONORM_ROWS_PER_CTA + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const float4 o = reinterpret_cast<const float4*>(onorm.staging)[
      (long)row * (DIM / 4) + lane];
  const float rms = rsqrtf(
      bfly_add(o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w, 32) *
          (1.0f / DIM) + onorm.eps);
  const long tok = row / H;
  const long channel = (long)(row % H) * DIM + lane * 4;
  BF16* dst = out + (long)row * DIM + lane * 4;
  dst[0] = __float2bfloat16(onorm_value(onorm, tok, channel + 0, lane * 4 + 0, o.x, rms));
  dst[1] = __float2bfloat16(onorm_value(onorm, tok, channel + 1, lane * 4 + 1, o.y, rms));
  dst[2] = __float2bfloat16(onorm_value(onorm, tok, channel + 2, lane * 4 + 2, o.z, rms));
  dst[3] = __float2bfloat16(onorm_value(onorm, tok, channel + 3, lane * 4 + 3, o.w, rms));
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

template <int VS, Out MODE, bool NATIVE = false, typename... Args>
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
    IndexArg slot_idx, IndexArg scratch_idx,
    IndexArg cu_seqlens, int H, int T_LOOP, int L, float scale,
    float lower_bound, VerifyStrides strides, Args... args) {
  static_assert(MODE != Out::kNorm, "the generic kernel stages a normed output");
  static_assert(!NATIVE || MODE == Out::kPlain, "native replay emits plain output");
  static_assert(sizeof...(Args) == (NATIVE || MODE != Out::kPlain ? 1 : 0),
                "verify takes only the argument for its active layout or epilogue");
  const auto native = verify_arg<NativeVerify, NATIVE>(args...);
  [[maybe_unused]] const auto onorm = verify_arg<OnormArgs, MODE != Out::kPlain>(args...);
  constexpr bool PROTECT = NATIVE && pla::kda_mtp::kNativeNumericalProtection;
  const int i_hv = blockIdx.x;
  const int i_n = blockIdx.y;
  const int i_vs = blockIdx.z;              // v-split index (0..VS-1)
  const bool ring_writer = (i_vs == 0);     // only split 0 writes shared rings
  constexpr int PASSES_PB = VPASSES / VS;   // value-row passes per block
  static_assert(VPASSES % VS == 0, "VS must divide VPASSES");
  const int tid = threadIdx.x;
  if constexpr (PROTECT) {
    static_assert(VS == 1, "native conditioning owns all four value tiles");
    if (native.precision_flags && tid < 4)
      native.precision_flags[((long)i_n * H + i_hv) * 4 + tid] = 0;
  }

  auto read_index = [](IndexArg arg, int n) -> int64_t {
    if constexpr (NATIVE)
      return arg.wide ? static_cast<const int64_t*>(arg.data)[n * arg.stride] :
                        static_cast<const int*>(arg.data)[n * arg.stride];
    return index_at(arg, n);
  };
  const int64_t begin = read_index(cu_seqlens, i_n);
  const int64_t end = read_index(cu_seqlens, i_n + 1);
  const int64_t state_slot = read_index(slot_idx, i_n);
  if constexpr (NATIVE) {
    if (read_index(cu_seqlens, 0) != 0 || begin < 0 || end < begin ||
        end > native.tokens || end - begin > T_LOOP ||
        state_slot >= native.slots || begin == end) return;
  }
  const int bos = begin, n_tok = end - begin, slot = state_slot;
  const int srow = NATIVE ? i_n : index_at(scratch_idx, i_n);
  const int ring_row = NATIVE ? i_n : slot;
  const long head_off = (long)i_hv * DIM;

  // CUDA-graph padding rows: zero the output, touch no cache.
  if (NATIVE ? state_slot <= native.null_block_id : slot < 0) {
    if (ring_writer)
      for (int t = 0; t < (NATIVE ? n_tok : T_LOOP); ++t)
        for (int j = tid; j < DIM; j += THREADS) {
          if constexpr (MODE == Out::kStage)
            onorm.staging[(long)(bos + t) * H * DIM + head_off + j] = 0.f;
          else
            out[(long)(bos + t) * H * DIM + head_off + j] = __float2bfloat16(0.f);
        }
    return;
  }

  extern __shared__ char smem_raw[];
  Smem& sh = *reinterpret_cast<Smem*>(smem_raw);
  // Used only by the native instantiation. Each warp covers 32/KGRPS rows.
  __shared__ float condition[PROTECT ? 4 : 1][PROTECT ? 4 : 1][PROTECT ? KGRPS : 1];

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
        h0 + (NATIVE ? (long)slot * native.state_slot_stride + (long)i_hv * DIM * DIM :
                       ((long)slot * H + i_hv) * DIM * DIM) + (long)v * DIM);
#pragma unroll
    for (int i = 0; i < F4_PER_THREAD; ++i) st[p][i] = g4[k_grp + KGRPS * i];
  }

  const float exp_A = __expf(A_log[i_hv]);
  __syncthreads();

  for (int t = 0; t < (NATIVE ? n_tok : T_LOOP); ++t) {
    // the DSL clamps the token index so a short request re-reads its last token
    const int tok = bos + (t < n_tok ? t : n_tok - 1);
    const long ro_base = ((long)ring_row * H + i_hv) * (long)L * DIM + (long)t * DIM;

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
                       ppu_sigmoid(gate_product(exp_A, (__bfloat162float(g_in[
                                              tok * strides.g + head_off + j]) +
                                              dt_bias[head_off + j])));
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
        ring_beta[((long)ring_row * H + i_hv) * (long)L + t] = sh.beta;
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
      float state_residual = 0.f, state_terms = 0.f, projection_terms = 0.f;
#pragma unroll
      for (int i = 0; i < F4_PER_THREAD; ++i) {
        const int f4 = k_grp + KGRPS * i;
        const float4 kk = sk4[f4], qq = sq4[f4];
        if constexpr (PROTECT) {
          if (native.precision_flags)
            state_terms += fabsf(st[p][i].x) + fabsf(st[p][i].y) +
                           fabsf(st[p][i].z) + fabsf(st[p][i].w) +
                           fabsf(vnk * kk.x) + fabsf(vnk * kk.y) +
                           fabsf(vnk * kk.z) + fabsf(vnk * kk.w);
        }
        st[p][i].x = fmaf(vnk, kk.x, st[p][i].x);
        st[p][i].y = fmaf(vnk, kk.y, st[p][i].y);
        st[p][i].z = fmaf(vnk, kk.z, st[p][i].z);
        st[p][i].w = fmaf(vnk, kk.w, st[p][i].w);
        if constexpr (PROTECT) {
          if (native.precision_flags) {
            state_residual += fabsf(st[p][i].x) + fabsf(st[p][i].y) +
                              fabsf(st[p][i].z) + fabsf(st[p][i].w);
            projection_terms += fabsf(st[p][i].x * qq.x) + fabsf(st[p][i].y * qq.y) +
                                fabsf(st[p][i].z * qq.z) + fabsf(st[p][i].w * qq.w);
          }
        }
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
      if constexpr (PROTECT) {
        if (native.precision_flags) {
          const float sr = bfly_add(state_residual, 32);
          const float sb = bfly_add(state_terms, 32);
          const float qr = bfly_add(k_grp == 0 ? fabsf(shq) : 0.f, 32);
          const float qb = bfly_add(projection_terms, 32) * fabsf(qfold);
          if (lane == 0) {
            const int first_value = p * VROWS + wid * (32 / KGRPS);
            const int tile = first_value / 32;
            const int part = (first_value % 32) / (32 / KGRPS);
            condition[0][tile][part] = sr;
            condition[1][tile][part] = sb;
            condition[2][tile][part] = qr;
            condition[3][tile][part] = qb;
          }
        }
      }
      if (k_grp == 0 && t < n_tok) {
        if constexpr (MODE == Out::kStage)
          onorm.staging[(long)(bos + t) * H * DIM + head_off + v] = shq;
        else
          out[(long)(bos + t) * H * DIM + head_off + v] = __float2bfloat16(shq);
      }
    }
    __syncthreads();
    if constexpr (PROTECT) {
      if (native.precision_flags && tid < 4) {
        float sr = 0.f, sb = 0.f, qr = 0.f, qb = 0.f;
#pragma unroll
        for (int part = 0; part < KGRPS; ++part) {
          sr += condition[0][tid][part];
          sb += condition[1][tid][part];
          qr += condition[2][tid][part];
          qb += condition[3][tid][part];
        }
        if (precision_risk(sr, sb) || precision_risk(qr, qb))
          native.precision_flags[((long)i_n * H + i_hv) * 4 + tid] = 1;
      }
    }
  }
  // recurrent_state is read-only for verify: nothing is written back
}

// Butterfly transpose-reduce. On entry each lane holds N partials; every level
// keeps one half of the live values, adds in the partner lane's copy of that
// half and hands it the other half, so after log2(W) levels value g of a lane
// holds the W-lane group sum of index g + (N/W) * (lane % W). N values cost
// N - N/W shuffles instead of the N * log2(W) of one butterfly per value.
template <int N, int O>
__device__ __forceinline__ void transpose_step(float* v, int lane) {
  if constexpr (O >= 1) {
    const bool upper = (lane & O) != 0;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      const float send = upper ? v[i] : v[i + N / 2];
      const float keep = upper ? v[i + N / 2] : v[i];
      v[i] = keep + __shfl_xor_sync(0xffffffffu, send, O, 32);
    }
    transpose_step<N / 2, O / 2>(v, lane);
  }
}

// Sums each of the N values over aligned W-lane groups and passes every sum to
// emit(index, sum) on exactly one lane of its group, using whichever of the
// transpose-reduce and the per-value butterfly needs fewer shuffles.
template <int N, int W, typename Emit>
__device__ __forceinline__ void group_sums(const float (&in)[N], int lane, Emit emit) {
  constexpr int LOG = W >= 32 ? 5 : W >= 16 ? 4 : W >= 8 ? 3 : W >= 4 ? 2 : W >= 2 ? 1 : 0;
  constexpr int PAD = (N + W - 1) / W * W;
  if constexpr (PAD - PAD / W < N * LOG) {
    float v[PAD];
#pragma unroll
    for (int i = 0; i < PAD; ++i) v[i] = i < N ? in[i < N ? i : 0] : 0.f;
    transpose_step<PAD, W / 2>(v, lane);
#pragma unroll
    for (int g = 0; g < PAD / W; ++g) {
      const int idx = g + (PAD / W) * (lane % W);
      if (idx < N) emit(idx, v[g]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      const float s = bfly_add(in[i], W);
      if (lane % W == 0) emit(i, s);
    }
  }
}

// verify_steps_kernel<STEPS, VS>:
// D(t,j) = product(d_r, r=j+1..t), Pk_t = D(t,-1)*k_t, Pq_t = D(t,-1)*q_t.
// L[t,j] = dot(k_j*D(t,j), k_t), R[t,j] = dot(k_j*D(t,j), q_t).
// W_t = beta_t * (Pk_t - sum(j<t, L[t,j]*W_j)).
// U_t = beta_t * (v_t  - sum(j<t, L[t,j]*U_j)).
// Qeff_t = Pq_t - sum(j<=t, R[t,j]*W_j).
// bias_t = sum(j<=t, R[t,j]*U_j), out_t = H0*Qeff_t + bias_t.
//
// One 128-thread CTA per (head, request, value split). Everything before the
// state projection is channel-parallel: thread c keeps q/k/decay/v of channel
// c for every step in registers, forms its terms of the L2 norms and of the
// L/R triangles on the pre-norm q/k (the norms scale the reduced sums), and
// runs the elimination for its channel. Each warp reduces all of those terms
// with one transpose-reduce (3 x 31 shuffles at STEPS=8) instead of a
// warp-per-step schedule with two 32-lane butterflies per (t, j) pair; ACU on
// H=4/N=1/STEPS=8 showed that schedule plus the four dependent state rounds
// behind it running as a single-warp latency chain at 0.64% occupancy.
//
// VS > 1 splits the 128 value rows of the state projection over grid.z so a
// small H*N grid reaches more CUs. Every split re-derives the channel phases
// (cheap) but only split 0 writes the rings and conv snapshots. A split CTA
// loads its first state rows right after the slot check, so their DRAM latency
// overlaps the channel phases. EAGER gives VS = 1 the same early rows, at the
// register cost the host weighs in verify_steps_eager.
template <int STEPS>
struct VerifySteps {
  static constexpr int KK_OFF = 2 * STEPS;                         // [0,S) |q|^2, [S,2S) |k|^2
  static constexpr int KQ_OFF = KK_OFF + STEPS * (STEPS - 1) / 2;  // L[t][p], p < t
  static constexpr int SUMS = KQ_OFF + STEPS * (STEPS + 1) / 2;    // R[t][p], p <= t
  static constexpr int BETA_OFF = SUMS;
  static constexpr int TOTAL = BETA_OFF + STEPS;
  __host__ __device__ static constexpr int kk(int t, int p) {
    return KK_OFF + t * (t - 1) / 2 + p;
  }
  __host__ __device__ static constexpr int kq(int t, int p) {
    return KQ_OFF + t * (t + 1) / 2 + p;
  }
};

template <int STEPS, int VS, bool EAGER, Out MODE, bool NATIVE = false, typename... Args>
__global__ __launch_bounds__(128) void verify_steps_kernel(
    const BF16* __restrict__ x_q, const BF16* __restrict__ x_k,
    const BF16* __restrict__ x_v, const BF16* __restrict__ g_in,
    const BF16* __restrict__ beta_in, const float* __restrict__ w_q,
    const float* __restrict__ w_k, const float* __restrict__ w_v,
    const BF16* __restrict__ cs_q, const BF16* __restrict__ cs_k,
    const BF16* __restrict__ cs_v, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, BF16* __restrict__ out,
    BF16* __restrict__ ic_q, BF16* __restrict__ ic_k,
    BF16* __restrict__ ic_v, BF16* __restrict__ rawv, BF16* __restrict__ rawk,
    float* __restrict__ ring_g, float* __restrict__ ring_beta,
    IndexArg slot_idx, IndexArg scratch_idx, IndexArg cu, int H, int L,
    float scale, float lower_bound, VerifyStrides strides,
    const float* __restrict__ h0, Args... args) {
  static_assert(!NATIVE || MODE == Out::kPlain, "native replay emits plain output");
  static_assert(sizeof...(Args) == (NATIVE || MODE != Out::kPlain ? 1 : 0),
                "verify takes only the argument for its active layout or epilogue");
  const auto native = verify_arg<NativeVerify, NATIVE>(args...);
  [[maybe_unused]] const auto onorm = verify_arg<OnormArgs, MODE != Out::kPlain>(args...);
  constexpr bool PROTECT = NATIVE && pla::kda_mtp::kNativeNumericalProtection;
  using Layout = VerifySteps<STEPS>;
  static_assert(!PROTECT || Layout::SUMS >= 4 * STEPS + 1,
                "native final conditioning must fit the shared reduction table");
  constexpr int WARPS = 128 / 32;
  constexpr int ROWS = DIM / VS;  // value rows projected by this CTA
  constexpr int GROUPS = 8;       // threads per state row
  constexpr int VECTORS = DIM / 4 / GROUPS;
  constexpr int PASS_ROWS = 128 / GROUPS;
  constexpr int PASSES = ROWS / PASS_ROWS;
  // The split grids are latency-bound and keep the first two passes of state
  // rows in flight across the channel phases (all rows for VS>=4; holding all
  // four passes at VS=2 cost 144-160 registers). VS=1 is occupancy-bound at
  // large H*N (ACU at H=96/N=512/STEPS=4: 120 registers -> 8 blocks/CU vs 14
  // for a 72-register build), so by default it keeps no state in flight and
  // lets other CTAs hide the per-pass loads; EAGER keeps two passes anyway.
  constexpr int EARLY = VS == 1 && !EAGER ? 0 : PASSES < 2 ? PASSES : 2;
  constexpr int PREFETCH = PASSES < 2 ? PASSES : 2;
  static_assert(VS == 1 || VS == 2 || VS == 4 || VS == 8, "value split");
  static_assert(VS == 1 || !EAGER, "split CTAs always load early");
  static_assert(PASSES >= 1 && PASSES % PREFETCH == 0, "state passes");
  static_assert(MODE != Out::kNorm || VS == 1,
                "the in-CTA norm needs every value row of the head");
  const int n = blockIdx.y, h = blockIdx.x;
  const int row0 = VS == 1 ? 0 : static_cast<int>(blockIdx.z) * ROWS;
  const bool writer = VS == 1 || blockIdx.z == 0;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if constexpr (PROTECT) {
    static_assert(VS == 1, "native conditioning owns all four value tiles");
    static_assert(Layout::SUMS >= 4 * STEPS, "native conditioning scratch capacity");
  }
  int slot, bos, n_tok, srow;
  if constexpr (NATIVE) {
    auto read_index = [](IndexArg arg, int i) -> int64_t {
      return arg.wide ? static_cast<const int64_t*>(arg.data)[i * arg.stride] :
                        static_cast<const int*>(arg.data)[i * arg.stride];
    };
    const int64_t begin = read_index(cu, n), end = read_index(cu, n + 1);
    const int64_t source = read_index(slot_idx, n);
    if (read_index(cu, 0) != 0 || begin < 0 || end < begin ||
        end > native.tokens || end - begin > STEPS || begin == end ||
        source >= native.slots) {
      if constexpr (PROTECT) {
        if (native.precision_flags && tid < 4)
          native.precision_flags[((long)n * H + h) * 4 + tid] = 0;
      }
      return;
    }
    bos = static_cast<int>(begin);
    n_tok = static_cast<int>(end - begin);
    slot = source <= native.null_block_id ? -1 : static_cast<int>(source);
    srow = n;
  } else {
    slot = index_at(slot_idx, n);
    bos = index_at(cu, n);
    n_tok = index_at(cu, n + 1) - bos;
    srow = index_at(scratch_idx, n);
  }
  // The per-channel weights do not depend on the indices: issue them now so
  // they overlap the index misses instead of joining the slot-dependent loads.
  const int channel = tid;
  const long ch = h * DIM + channel;
  const float4 wq = reinterpret_cast<const float4*>(w_q)[ch];
  const float4 wk = reinterpret_cast<const float4*>(w_k)[ch];
  const float4 wv = reinterpret_cast<const float4*>(w_v)[ch];
  const float a_h = a_log[h];
  const float bias = dt_bias[ch];
  if (slot < 0) {
    if constexpr (PROTECT) {
      if (native.precision_flags && tid < 4)
        native.precision_flags[((long)n * H + h) * 4 + tid] = 0;
    }
    for (int i = tid; i < (NATIVE ? n_tok : STEPS) * ROWS; i += 128) {
      if constexpr (MODE == Out::kStage)
        onorm.staging[(long)(bos + i / ROWS) * H * DIM + h * DIM + row0 + i % ROWS] =
            0.f;
      else
        out[(long)(bos + i / ROWS) * H * DIM + h * DIM + row0 + i % ROWS] =
            __float2bfloat16(0.f);
    }
    return;
  }
  const int kg = tid % GROUPS, rlocal = tid / GROUPS;
  const long state_stride = NATIVE ? native.state_slot_stride : (long)H * DIM * DIM;
  const float4* state = reinterpret_cast<const float4*>(
      h0 + (long)slot * state_stride + ((long)h * DIM + row0) * DIM);
  float4 early[EARLY > 0 ? EARLY : 1][VECTORS];
#pragma unroll
  for (int p = 0; p < EARLY; ++p)
#pragma unroll
    for (int i = 0; i < VECTORS; ++i)
      early[p][i] = state[(rlocal + p * PASS_ROWS) * (DIM / 4) + kg + GROUPS * i];

  __shared__ float partial[WARPS][Layout::SUMS];
  __shared__ float total[Layout::TOTAL];
  __shared__ __align__(16) float result[2 * STEPS][DIM];
  __shared__ float stage[STEPS][ROWS];
  __shared__ float query_terms[PROTECT ? STEPS : 1][PROTECT ? DIM : 1];
  __shared__ float state_max[PROTECT ? ROWS : 1];
  float q[STEPS], k[STEPS], d[STEPS], v[STEPS];
  {
    float cq[3], ck[3], cv[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      cq[j] = __bfloat162float(cs_q[slot * strides.cs_q.slot +
          ch * strides.cs_q.channel + j * strides.cs_q.window]);
      ck[j] = __bfloat162float(cs_k[slot * strides.cs_k.slot +
          ch * strides.cs_k.channel + j * strides.cs_k.window]);
      cv[j] = __bfloat162float(cs_v[slot * strides.cs_v.slot +
          ch * strides.cs_v.channel + j * strides.cs_v.window]);
    }
    const float exp_a = __expf(a_h);
#pragma unroll
    for (int step = 0; step < STEPS; ++step) {
      if constexpr (NATIVE) {
        if (step >= n_tok) {
          q[step] = k[step] = v[step] = 0.f;
          d[step] = 1.f;
          if (channel == 0) total[Layout::BETA_OFF + step] = 0.f;
          continue;
        }
      }
      const int tok = bos + (step < n_tok ? step : n_tok - 1);
      const float xq = __bfloat162float(x_q[tok * strides.x_q + ch]);
      const float xk = __bfloat162float(x_k[tok * strides.x_k + ch]);
      const float xv = __bfloat162float(x_v[tok * strides.x_v + ch]);
      const float gate_input = __bfloat162float(g_in[tok * strides.g + ch]);
      const float beta_input = __bfloat162float(beta_in[tok * strides.beta + h]);
      float cq_t = xq * wq.w, ck_t = xk * wk.w, cv_t = xv * wv.w;
      cq_t += cq[0] * wq.x; cq_t += cq[1] * wq.y; cq_t += cq[2] * wq.z;
      ck_t += ck[0] * wk.x; ck_t += ck[1] * wk.y; ck_t += ck[2] * wk.z;
      cv_t += cv[0] * wv.x; cv_t += cv[1] * wv.y; cv_t += cv[2] * wv.z;
      q[step] = ppu_silu(cq_t);
      k[step] = ppu_silu(ck_t);
      v[step] = ppu_silu(cv_t);
      const float g = lower_bound * ppu_sigmoid(gate_product(exp_a, gate_input + bias));
      d[step] = __expf(g);
      cq[0] = cq[1]; cq[1] = cq[2]; cq[2] = xq;
      ck[0] = ck[1]; ck[1] = ck[2]; ck[2] = xk;
      cv[0] = cv[1]; cv[1] = cv[2]; cv[2] = xv;
      const float b = ppu_sigmoid(beta_input);
      if (channel == 0) total[Layout::BETA_OFF + step] = b;
      if (writer) {
        const long ri = ((long)(NATIVE ? n : slot) * H + h) * L * DIM + step * DIM + channel;
        rawk[ri] = __float2bfloat16(k[step]);
        rawv[ri] = __float2bfloat16(v[step]);
        ring_g[ri] = g;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          ic_q[srow * strides.ic_q.row + step * strides.ic_q.step +
              ch * strides.ic_q.channel + j * strides.ic_q.window] =
              __float2bfloat16(cq[j]);
          ic_k[srow * strides.ic_k.row + step * strides.ic_k.step +
              ch * strides.ic_k.channel + j * strides.ic_k.window] =
              __float2bfloat16(ck[j]);
          ic_v[srow * strides.ic_v.row + step * strides.ic_v.step +
              ch * strides.ic_v.channel + j * strides.ic_v.window] =
              __float2bfloat16(cv[j]);
        }
        if (channel == 0) ring_beta[((long)(NATIVE ? n : slot) * H + h) * L + step] = b;
      }
    }
  }
  {
    float terms[Layout::SUMS];
#pragma unroll
    for (int t = 0; t < STEPS; ++t) {
      terms[t] = q[t] * q[t];
      terms[STEPS + t] = k[t] * k[t];
    }
#pragma unroll
    for (int t = 0; t < STEPS; ++t) {
      float decay = 1.f;  // D(t, p)
#pragma unroll
      for (int p = t; p >= 0; --p) {
        if (p < t) decay *= d[p + 1];
        const float kd = k[p] * decay;
        if (p < t) terms[Layout::kk(t, p)] = kd * k[t];
        terms[Layout::kq(t, p)] = kd * q[t];
      }
    }
    group_sums<Layout::SUMS, 32>(terms, lane,
                                 [&](int i, float s) { partial[warp][i] = s; });
  }
  __syncthreads();
  if (tid < Layout::SUMS) {
    auto reduced = [&](int i) {
      float s = partial[0][i];
#pragma unroll
      for (int w = 1; w < WARPS; ++w) s += partial[w][i];
      return s;
    };
    auto inv_q = [&](int t) { return rsqrtf(reduced(t) + L2_EPS) * scale; };
    auto inv_k = [&](int t) { return rsqrtf(reduced(STEPS + t) + L2_EPS); };
    float val;
    if (tid < STEPS) {
      val = inv_q(tid);
    } else if (tid < 2 * STEPS) {
      val = inv_k(tid - STEPS);
    } else {
      // row t of the L triangle holds t entries, row t of R holds t + 1
      const bool is_q = tid >= Layout::KQ_OFF;
      int p = tid - (is_q ? Layout::KQ_OFF : Layout::KK_OFF);
      int t = 0, len = is_q ? 1 : 0;
      while (p >= len) {
        p -= len;
        ++t;
        ++len;
      }
      val = reduced(tid) * inv_k(p) * (is_q ? inv_q(t) : inv_k(t));
    }
    total[tid] = val;
  }
  __syncthreads();
  {
    float update_k[STEPS], update_v[STEPS];
    float cum = 1.f;  // D(step, -1)
#pragma unroll
    for (int step = 0; step < STEPS; ++step) {
      cum *= d[step];
      float uk = cum * k[step] * total[STEPS + step];
      float uv = v[step];
#pragma unroll
      for (int prev = 0; prev < step; ++prev) {
        const float c = total[Layout::kk(step, prev)];
        uk = fmaf(-c, update_k[prev], uk);
        uv = fmaf(-c, update_v[prev], uv);
      }
      const float b = total[Layout::BETA_OFF + step];
      update_k[step] = b * uk;
      update_v[step] = b * uv;
      float pq = cum * q[step] * total[step];
      float pq_terms = fabsf(pq);
      float pv = 0.f, pv_terms = 0.f;
#pragma unroll
      for (int prev = 0; prev <= step; ++prev) {
        const float c = total[Layout::kq(step, prev)];
        if constexpr (PROTECT) {
          if (native.precision_flags) {
            pq_terms += fabsf(c * update_k[prev]);
            pv_terms += fabsf(c * update_v[prev]);
          }
        }
        pq = fmaf(-c, update_k[prev], pq);
        pv = fmaf(c, update_v[prev], pv);
      }
      result[step][channel] = pq;
      result[STEPS + step][channel] = pv;
      if constexpr (PROTECT) {
        // Stage is not needed for output until projection. Reuse it to carry
        // the absolute value-update terms without another shared allocation.
        if (native.precision_flags) {
          stage[step][channel] = pv_terms;
          query_terms[step][channel] = pq_terms;
        }
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < PASSES; pass += PREFETCH) {
    float4 loaded[PREFETCH][VECTORS];
#pragma unroll
    for (int pair = 0; pair < PREFETCH; ++pair) {
      const int p = pass + pair;
      float magnitude = 0.f;
#pragma unroll
      for (int i = 0; i < VECTORS; ++i) {
        loaded[pair][i] = p < EARLY
            ? early[p < EARLY ? p : 0][i]
            : state[(rlocal + p * PASS_ROWS) * (DIM / 4) + kg + GROUPS * i];
        if constexpr (PROTECT) {
          if (native.precision_flags) {
            const float4 s = loaded[pair][i];
            magnitude = fmaxf(magnitude,
                fmaxf(fmaxf(fabsf(s.x), fabsf(s.y)),
                      fmaxf(fabsf(s.z), fabsf(s.w))));
          }
        }
      }
      if constexpr (PROTECT) {
        if (native.precision_flags) {
          magnitude = bfly_max(magnitude, GROUPS);
          if (kg == 0) state_max[rlocal + p * PASS_ROWS] = magnitude;
        }
      }
    }
    float dot[PREFETCH][STEPS] = {};
#pragma unroll
    for (int i = 0; i < VECTORS; ++i) {
      const int f4 = kg + GROUPS * i;
#pragma unroll
      for (int p = 0; p < STEPS; ++p) {
        const float4 a = reinterpret_cast<const float4*>(result[p])[f4];
#pragma unroll
        for (int pair = 0; pair < PREFETCH; ++pair) {
          const float4 s = loaded[pair][i];
          dot[pair][p] = fmaf(s.w, a.w, fmaf(s.z, a.z,
              fmaf(s.y, a.y, fmaf(s.x, a.x, dot[pair][p]))));
        }
      }
    }
#pragma unroll
    for (int pair = 0; pair < PREFETCH; ++pair) {
      const int row = rlocal + (pass + pair) * PASS_ROWS;
      group_sums<STEPS, GROUPS>(dot[pair], lane, [&](int step, float s) {
        const float bias = result[STEPS + step][row0 + row];
        if constexpr (PROTECT) {
          if (native.precision_flags) {
            // Preserve the value-update term bound in the consumed bias slot.
            // The final guard adds ||H0_row||inf * ||qeff_step||1, which also
            // bounds cancellation inside the state projection itself.
            result[STEPS + step][row0 + row] = stage[step][row];
          }
        }
        stage[step][row] = s + bias;
      });
    }
  }
  if constexpr (MODE == Out::kNorm) {
    // Thread tid owns value row tid of every step; issue its gate and weight
    // loads before the barrier so they overlap it.
    float gate[STEPS];
#pragma unroll
    for (int step = 0; step < STEPS; ++step) {
      const int tok = bos + (step < n_tok ? step : n_tok - 1);
      gate[step] = __bfloat162float(static_cast<const BF16*>(onorm.gate)[
          tok * onorm.gate_stride + ch]);
    }
    const float weight = onorm.weight[channel];
    __syncthreads();
#pragma unroll
    for (int step = 0; step < STEPS; ++step) {
      // Every warp reduces the whole row itself rather than publishing one sum.
      float sum = 0.f;
#pragma unroll
      for (int m = 0; m < DIM / 32; ++m) {
        const float o = stage[step][lane + m * 32];
        sum += o * o;
      }
      const float rms = rsqrtf(bfly_add(sum, 32) * (1.0f / DIM) + onorm.eps);
      if (step < n_tok)
        out[(long)(bos + step) * H * DIM + ch] = __float2bfloat16(
            stage[step][channel] * rms * weight * ppu_sigmoid(gate[step]));
    }
    return;
  }
  __syncthreads();
  if constexpr (PROTECT) {
    if (native.precision_flags) {
      // The state projection is finished, so these reduction temporaries do
      // not overlap its live registers. One transpose reduction replaces the
      // separate query reduction and four pairs of output butterflies.
      float conditioning[4 * STEPS + 1];
#pragma unroll
      for (int step = 0; step < STEPS; ++step) {
        const float value = stage[step][tid];
        conditioning[step] = fabsf(result[step][tid]);
        conditioning[STEPS + step] = query_terms[step][tid];
        conditioning[2 * STEPS + step] = fabsf(value);
        conditioning[3 * STEPS + step] = result[STEPS + step][tid];
        if (step < n_tok)
          out[(long)(bos + step) * H * DIM + h * DIM + tid] =
              __float2bfloat16(value);
      }
      conditioning[4 * STEPS] = state_max[tid];
      group_sums<4 * STEPS + 1, 32>(conditioning, lane,
          [&](int i, float s) { partial[warp][i] = s; });
      __syncthreads();
      // Four adjacent lanes publish one contiguous head flag vector. Every
      // valid head writes once; early-return paths above write explicit zeroes.
      if (tid < 4) {
        bool risky = false;
#pragma unroll
        for (int step = 0; step < STEPS; ++step) {
          float residual = 0.f, terms = 0.f;
#pragma unroll
          for (int w = 0; w < WARPS; ++w) {
            residual += partial[w][step];
            terms += partial[w][STEPS + step];
          }
          if (step < n_tok)
            risky |= precision_risk(residual, terms) ||
                     precision_risk(partial[tid][2 * STEPS + step],
                                    residual * partial[tid][4 * STEPS] +
                                    partial[tid][3 * STEPS + step]);
        }
        native.precision_flags[((long)n * H + h) * 4 + tid] = risky;
      }
      return;
    }
  }
  for (int i = tid; i < STEPS * ROWS; i += 128) {
    const int step = i / ROWS, r = i % ROWS;
    if (step < n_tok) {
      if constexpr (MODE == Out::kStage)
        onorm.staging[(long)(bos + step) * H * DIM + h * DIM + row0 + r] =
            stage[step][r];
      else
        out[(long)(bos + step) * H * DIM + h * DIM + row0 + r] =
            __float2bfloat16(stage[step][r]);
    }
  }
}

__device__ __forceinline__ double accurate_sigmoid(double x) {
  if (x >= 0.0) return 1.0 / (1.0 + ::exp(-x));
  const double e = ::exp(x);
  return e / (1.0 + e);
}

__device__ __forceinline__ double accurate_gate(double a, double x,
                                                double lower_bound) {
  if (x == 0.0) return lower_bound * 0.5;
  // x is a sum of finite BF16/FP32 inputs. Outside this range even the
  // smallest nonzero such sum is saturated, or the product is negligible.
  if (a > 700.0) return x > 0.0 ? lower_bound : 0.0;
  if (a < -700.0) return lower_bound * 0.5;
  return lower_bound * accurate_sigmoid(::exp(a) * x);
}

__device__ __forceinline__ double accurate_warp_sum(double x) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, offset, 32);
  return x;
}

// Conditional native precision repair. A separate kernel keeps FP64 state and math
// out of the common kernels' register allocation. Each CTA first scans 128
// head/request pairs and cooperatively repairs only selected pairs. Persistent
// state is read-only, so the common output can safely be overwritten here.
//
// The common kernels publish observed cancellation flags. Very negative gate
// bounds retain a direct guard; A_log and beta magnitudes alone do not select
// this slow path. This is not a universal finite-input conditioning proof.
__global__ __launch_bounds__(128) void verify_precise_native(
    const BF16* x_q, const BF16* x_k, const BF16* x_v,
    const BF16* g_in, const BF16* beta_in,
    const float* w_q, const float* w_k, const float* w_v,
    const BF16* cs_q, const BF16* cs_k, const BF16* cs_v,
    const float* a_log, const float* dt_bias, const float* h0,
    BF16* out, BF16* ic_q, BF16* ic_k, BF16* ic_v,
    BF16* rawv, BF16* rawk, float* ring_g, float* ring_beta,
    IndexArg slots, IndexArg cu, int H, int N, int Q,
    float scale, float lower_bound, VerifyStrides strides,
    pla::kda_mtp::NativeVerify native) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  auto read_index = [](IndexArg arg, int i) -> int64_t {
    return arg.wide ? static_cast<const int64_t*>(arg.data)[i * arg.stride] :
                      static_cast<const int*>(arg.data)[i * arg.stride];
  };
  const int64_t first = static_cast<int64_t>(blockIdx.x) * 128;
  const int64_t candidate = first + tid;
  bool repair = false;
  if (candidate < static_cast<int64_t>(H) * N) {
    const int n = candidate / H, h = candidate % H;
    const int64_t begin = read_index(cu, n), end = read_index(cu, n + 1);
    const int64_t source = read_index(slots, n);
    if (read_index(cu, 0) == 0 && begin >= 0 && end > begin &&
        end <= native.tokens && end - begin <= Q &&
        source > native.null_block_id && source < native.slots) {
      repair = lower_bound < -32.f;
      if (native.precision_flags) {
        const int* flags = native.precision_flags + candidate * 4;
        repair |= flags[0] || flags[1] || flags[2] || flags[3];
      }
    }
  }
  __shared__ unsigned masks[4];
  const unsigned mask = __ballot_sync(0xffffffffu, repair);
  if (lane == 0) masks[warp] = mask;
  __syncthreads();
  if (!(masks[0] | masks[1] | masks[2] | masks[3])) return;

  __shared__ double sq[DIM], sk[DIM], sv[DIM], sd[DIM];
  __shared__ double norm_q[4], norm_k[4], inv_q, inv_k;
  // Four warps own 32 value rows per pass. Each lane retains only four key
  // components for each of eight rows, rather than an entire state row.
  constexpr int VALUE_ROWS = 8;
  for (int group = 0; group < 4; ++group) {
    unsigned remaining = masks[group];
    while (remaining) {
      const int selected = __ffs(remaining) - 1;
      remaining &= remaining - 1;
      const int64_t identity = first + group * 32 + selected;
      const int n = identity / H, h = identity % H;
      const int64_t begin = read_index(cu, n), end = read_index(cu, n + 1);
      const int source = static_cast<int>(read_index(slots, n));
      const long channel = static_cast<long>(h) * DIM + tid;
      double weights_q[4], weights_k[4], weights_v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        weights_q[j] = w_q[channel * 4 + j];
        weights_k[j] = w_k[channel * 4 + j];
        weights_v[j] = w_v[channel * 4 + j];
      }
      const double a = a_log[h], bias = dt_bias[channel];
      for (int pass = 0; pass < DIM / (4 * VALUE_ROWS); ++pass) {
        double state[VALUE_ROWS][4];
#pragma unroll
        for (int row = 0; row < VALUE_ROWS; ++row) {
          const int value_row = pass * 4 * VALUE_ROWS + warp * VALUE_ROWS + row;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            state[row][j] = h0[static_cast<long>(source) * native.state_slot_stride +
                (static_cast<long>(h) * DIM + value_row) * DIM + lane + j * 32];
        }
        double history_q[3], history_k[3], history_v[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          history_q[j] = __bfloat162float(cs_q[static_cast<long>(source) * strides.cs_q.slot +
              channel * strides.cs_q.channel + j * strides.cs_q.window]);
          history_k[j] = __bfloat162float(cs_k[static_cast<long>(source) * strides.cs_k.slot +
              channel * strides.cs_k.channel + j * strides.cs_k.window]);
          history_v[j] = __bfloat162float(cs_v[static_cast<long>(source) * strides.cs_v.slot +
              channel * strides.cs_v.channel + j * strides.cs_v.window]);
        }
        for (int t = 0; t < end - begin; ++t) {
          const long token = begin + t;
          const double xq = __bfloat162float(x_q[token * strides.x_q + channel]);
          const double xk = __bfloat162float(x_k[token * strides.x_k + channel]);
          const double xv = __bfloat162float(x_v[token * strides.x_v + channel]);
          double q = xq * weights_q[3], k = xk * weights_k[3], v = xv * weights_v[3];
#pragma unroll
          for (int j = 0; j < 3; ++j) {
            q += history_q[j] * weights_q[j];
            k += history_k[j] * weights_k[j];
            v += history_v[j] * weights_v[j];
          }
          q *= accurate_sigmoid(q);
          k *= accurate_sigmoid(k);
          v *= accurate_sigmoid(v);
          const double gate = accurate_gate(a,
              static_cast<double>(__bfloat162float(g_in[token * strides.g + channel])) + bias,
              static_cast<double>(lower_bound));
          const double beta = accurate_sigmoid(
              static_cast<double>(__bfloat162float(beta_in[token * strides.beta + h])));
          history_q[0] = history_q[1]; history_q[1] = history_q[2]; history_q[2] = xq;
          history_k[0] = history_k[1]; history_k[1] = history_k[2]; history_k[2] = xk;
          history_v[0] = history_v[1]; history_v[1] = history_v[2]; history_v[2] = xv;
          sq[tid] = q; sk[tid] = k; sv[tid] = v; sd[tid] = ::exp(gate);
          const double qsum = accurate_warp_sum(q * q);
          const double ksum = accurate_warp_sum(k * k);
          if (lane == 0) { norm_q[warp] = qsum; norm_k[warp] = ksum; }
          if (pass == 0) {
            const long record = ((static_cast<long>(n) * H + h) * Q + t) * DIM + tid;
            rawk[record] = __float2bfloat16(static_cast<float>(k));
            rawv[record] = __float2bfloat16(static_cast<float>(v));
            ring_g[record] = static_cast<float>(gate);
            if (tid == 0) ring_beta[(static_cast<long>(n) * H + h) * Q + t] = static_cast<float>(beta);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
              ic_q[static_cast<long>(n) * strides.ic_q.row + t * strides.ic_q.step +
                  channel * strides.ic_q.channel + j * strides.ic_q.window] =
                  __float2bfloat16(static_cast<float>(history_q[j]));
              ic_k[static_cast<long>(n) * strides.ic_k.row + t * strides.ic_k.step +
                  channel * strides.ic_k.channel + j * strides.ic_k.window] =
                  __float2bfloat16(static_cast<float>(history_k[j]));
              ic_v[static_cast<long>(n) * strides.ic_v.row + t * strides.ic_v.step +
                  channel * strides.ic_v.channel + j * strides.ic_v.window] =
                  __float2bfloat16(static_cast<float>(history_v[j]));
            }
          }
          __syncthreads();
          if (tid == 0) {
            inv_q = static_cast<double>(scale) / ::sqrt(norm_q[0] + norm_q[1] + norm_q[2] + norm_q[3] + 1.0e-6);
            inv_k = 1.0 / ::sqrt(norm_k[0] + norm_k[1] + norm_k[2] + norm_k[3] + 1.0e-6);
          }
          __syncthreads();
          sq[tid] *= inv_q; sk[tid] *= inv_k;
          __syncthreads();
#pragma unroll
          for (int row = 0; row < VALUE_ROWS; ++row) {
            const int value_row = pass * 4 * VALUE_ROWS + warp * VALUE_ROWS + row;
            double dot_k = 0.0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int key = lane + j * 32;
              state[row][j] *= sd[key];
              dot_k = ::fma(state[row][j], sk[key], dot_k);
            }
            dot_k = accurate_warp_sum(dot_k);
            const double delta = (sv[value_row] - dot_k) * beta;
            double dot_q = 0.0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int key = lane + j * 32;
              state[row][j] = ::fma(delta, sk[key], state[row][j]);
              dot_q = ::fma(state[row][j], sq[key], dot_q);
            }
            dot_q = accurate_warp_sum(dot_q);
            if (lane == 0)
              out[token * H * DIM + static_cast<long>(h) * DIM + value_row] =
                  __float2bfloat16(static_cast<float>(dot_q));
          }
          __syncthreads();
        }
      }
    }
  }
}

}  // namespace

namespace pla {
namespace kda_mtp {
namespace {
template <int VS, Out MODE>
void launch_verify_vsplit(
    const void* x_q, const void* x_k, const void* x_v,
    const void* g_in, const void* beta_in,
    const float* w_q, const float* w_k, const float* w_v,
    const void* cs_q, const void* cs_k, const void* cs_v,
    const float* a_log, const float* dt_bias, const float* h0,
    void* out, void* ic_q, void* ic_k, void* ic_v,
    void* ring_rawv, void* ring_rawk, float* ring_g, float* ring_beta,
    IndexArg slot_idx, IndexArg scratch_idx, IndexArg cu_seqlens,
    int H, int N, int t_loop, int ring_len,
    float scale, float lower_bound, VerifyStrides strides,
    const OnormArgs& onorm, hggcStream_t stream) {
  const int smem = static_cast<int>(sizeof(Smem));
  dim3 grid(H, N, VS);
#define PLA_VERIFY_GENERIC_ARGS                                                \
      static_cast<const BF16*>(x_q), static_cast<const BF16*>(x_k),            \
      static_cast<const BF16*>(x_v), static_cast<const BF16*>(g_in),           \
      static_cast<const BF16*>(beta_in), w_q, w_k, w_v,                        \
      static_cast<const BF16*>(cs_q), static_cast<const BF16*>(cs_k),          \
      static_cast<const BF16*>(cs_v), a_log, dt_bias, h0,                      \
      static_cast<BF16*>(out), static_cast<BF16*>(ic_q),                       \
      static_cast<BF16*>(ic_k), static_cast<BF16*>(ic_v),                      \
      static_cast<BF16*>(ring_rawv), static_cast<BF16*>(ring_rawk),            \
      ring_g, ring_beta, slot_idx, scratch_idx, cu_seqlens,                    \
      H, t_loop, ring_len, scale, lower_bound, strides
  if constexpr (MODE == Out::kPlain) {
    if (smem > 48 * 1024) {
      hggcFuncSetAttribute(verify_kernel<VS, MODE>,
                           hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    verify_kernel<VS, MODE><<<grid, THREADS, smem, stream>>>(PLA_VERIFY_GENERIC_ARGS);
  } else {
    if (smem > 48 * 1024) {
      hggcFuncSetAttribute(verify_kernel<VS, MODE, false, OnormArgs>,
                           hggcFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    verify_kernel<VS, MODE, false, OnormArgs><<<grid, THREADS, smem, stream>>>(
        PLA_VERIFY_GENERIC_ARGS, onorm);
  }
#undef PLA_VERIFY_GENERIC_ARGS
}

// The plain output keeps its original instantiation; a normed output is staged
// in fp32 for onorm_kernel.
template <int VS, typename... Args>
void launch_verify_generic(const OnormArgs& onorm, hggcStream_t stream,
                           Args... args) {
  if (onorm.gate == nullptr)
    launch_verify_vsplit<VS, Out::kPlain>(args..., onorm, stream);
  else
    launch_verify_vsplit<VS, Out::kStage>(args..., onorm, stream);
}

void launch_onorm(void* out, const OnormArgs& onorm, int rows, int H,
                  hggcStream_t stream) {
  onorm_kernel<<<(rows + ONORM_ROWS_PER_CTA - 1) / ONORM_ROWS_PER_CTA,
                 32 * ONORM_ROWS_PER_CTA, 0, stream>>>(
      static_cast<BF16*>(out), onorm, rows, H);
}

// Cache lazy/eager resident CTA counts per thread and window for the current
// device. Refresh on device changes; failed queries use lazy and may retry.
template <int STEPS>
bool verify_steps_eager_at(int64_t ctas, int device) {
  const int64_t forced = verify_eager_override();
  if (forced != 0) return forced > 0;
  static thread_local int cached_device = -1;
  static thread_local std::pair<int64_t, int64_t> caps(0, 0);
  if (cached_device != device) {
    int cus = 0, lazy = 0, eager = 0;
    if (hggcDeviceGetAttribute(&cus, hggcDevAttrMultiProcessorCount, device) !=
            hggcSuccess ||
        hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &lazy, verify_steps_kernel<STEPS, 1, false, Out::kPlain>, 128, 0) !=
            hggcSuccess ||
        hggcOccupancyMaxActiveBlocksPerMultiprocessor(
            &eager, verify_steps_kernel<STEPS, 1, true, Out::kPlain>, 128, 0) !=
            hggcSuccess)
      return false;
    caps = std::pair<int64_t, int64_t>(static_cast<int64_t>(cus) * lazy,
                                       static_cast<int64_t>(cus) * eager);
    cached_device = device;
  }
  return verify_steps_eager(ctas, caps.first, caps.second);
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
    IndexArg slot_idx, IndexArg scratch_idx, IndexArg cu_seqlens,
    int H, int N, int t_loop, int ring_len,
    float scale, float lower_bound, VerifyStrides strides,
    const OnormArgs& onorm, hggcStream_t stream) {
  const bool norm = onorm.gate != nullptr;
  const int rows = N * t_loop * H;
  int device = -1;
  if (use_verify_fast(H, N, t_loop) && hggcGetDevice(&device) == hggcSuccess &&
      reinterpret_cast<uintptr_t>(w_q) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(w_k) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(w_v) % 16 == 0) {
    const int split = norm ? verify_steps_split_norm(H, N, t_loop)
                           : verify_steps_split(H, N, t_loop);
#define PLA_VERIFY_STEPS_ARGS                                                  \
      static_cast<const BF16*>(x_q), static_cast<const BF16*>(x_k),            \
      static_cast<const BF16*>(x_v), static_cast<const BF16*>(g_in),           \
      static_cast<const BF16*>(beta_in), w_q, w_k, w_v,                        \
      static_cast<const BF16*>(cs_q), static_cast<const BF16*>(cs_k),          \
      static_cast<const BF16*>(cs_v), a_log, dt_bias,                          \
      static_cast<BF16*>(out), static_cast<BF16*>(ic_q),                       \
      static_cast<BF16*>(ic_k), static_cast<BF16*>(ic_v),                      \
      static_cast<BF16*>(ring_rawv), static_cast<BF16*>(ring_rawk),            \
      ring_g, ring_beta, slot_idx, scratch_idx, cu_seqlens,                    \
      H, ring_len, scale, lower_bound, strides, h0
#define PLA_VERIFY_STEPS_MODES(STEPS, VS, EAGER, NORM_MODE)                    \
  {                                                                            \
    if (norm)                                                                  \
      verify_steps_kernel<STEPS, VS, EAGER, NORM_MODE, false, OnormArgs>       \
          <<<dim3(H, N, VS), 128, 0, stream>>>(PLA_VERIFY_STEPS_ARGS, onorm);  \
    else                                                                       \
      verify_steps_kernel<STEPS, VS, EAGER, Out::kPlain>                       \
          <<<dim3(H, N, VS), 128, 0, stream>>>(PLA_VERIFY_STEPS_ARGS);         \
  }
#define PLA_VERIFY_STEPS_LAUNCH(STEPS)                                         \
  switch (split) {                                                             \
    case 8: PLA_VERIFY_STEPS_MODES(STEPS, 8, false, Out::kStage) break;        \
    case 4: PLA_VERIFY_STEPS_MODES(STEPS, 4, false, Out::kStage) break;        \
    case 2: PLA_VERIFY_STEPS_MODES(STEPS, 2, false, Out::kStage) break;        \
    default:                                                                   \
      if (verify_steps_eager_at<STEPS>(static_cast<int64_t>(H) * N, device))   \
        PLA_VERIFY_STEPS_MODES(STEPS, 1, true, Out::kNorm)                     \
      else                                                                     \
        PLA_VERIFY_STEPS_MODES(STEPS, 1, false, Out::kNorm)                    \
      break;                                                                   \
  }
    bool launched = true;
    switch (t_loop) {
      case 1: PLA_VERIFY_STEPS_LAUNCH(1); break;
      case 2: PLA_VERIFY_STEPS_LAUNCH(2); break;
      case 3: PLA_VERIFY_STEPS_LAUNCH(3); break;
      case 4: PLA_VERIFY_STEPS_LAUNCH(4); break;
      case 5: PLA_VERIFY_STEPS_LAUNCH(5); break;
      case 6: PLA_VERIFY_STEPS_LAUNCH(6); break;
      case 7: PLA_VERIFY_STEPS_LAUNCH(7); break;
      case 8: PLA_VERIFY_STEPS_LAUNCH(8); break;
      default: launched = false; break;  // wider windows use the generic path
    }
#undef PLA_VERIFY_STEPS_LAUNCH
#undef PLA_VERIFY_STEPS_MODES
#undef PLA_VERIFY_STEPS_ARGS
    if (launched) {
      if (norm && split > 1) launch_onorm(out, onorm, rows, H, stream);
      return;
    }
  }
#define PLA_VERIFY_ARGS                                                        \
  onorm, stream, x_q, x_k, x_v, g_in, beta_in, w_q, w_k, w_v, cs_q, cs_k,    \
      cs_v, a_log, dt_bias, h0, out, ic_q, ic_k, ic_v, ring_rawv, ring_rawk,   \
      ring_g, ring_beta, slot_idx, scratch_idx, cu_seqlens, H, N, t_loop,      \
      ring_len, scale, lower_bound, strides
  int vsplit = VSPLIT_FORCE;
  if (vsplit == 0) {
    vsplit = (CAN_SPLIT2 && static_cast<long>(H) * N < KDA_V_VSPLIT_GATE) ? 2 : 1;
  }
  const bool split2 = CAN_SPLIT2 && vsplit == 2;
  if (split2) {
    if constexpr (CAN_SPLIT2) launch_verify_generic<2>(PLA_VERIFY_ARGS);
  } else {
    launch_verify_generic<1>(PLA_VERIFY_ARGS);
  }
#undef PLA_VERIFY_ARGS
  if (norm) launch_onorm(out, onorm, rows, H, stream);
}

void launch_verify_native(
    const void* x_q, const void* x_k, const void* x_v,
    const void* g_in, const void* beta_in,
    const float* w_q, const float* w_k, const float* w_v,
    const void* cs_q, const void* cs_k, const void* cs_v,
    const float* a_log, const float* dt_bias, const float* h0,
    void* out, void* ic_q, void* ic_k, void* ic_v,
    void* rawv, void* rawk, float* gate, float* beta,
    IndexArg slots, IndexArg cu, int H, int N, int Q,
    float scale, float lower_bound, VerifyStrides strides,
    NativeVerify native, hggcStream_t stream) {
  if (N == 0) return;
  const bool ppu15 = current_device_is_ppu15();
  if (Q == 4 && static_cast<int64_t>(H) * N >= 768 &&
      use_verify_fast(H, N, Q) && ppu15 &&
      reinterpret_cast<uintptr_t>(w_q) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(w_k) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(w_v) % 16 == 0) {
    verify_steps_kernel<4, 1, false, Out::kPlain, true><<<dim3(H, N, 1), 128, 0, stream>>>(
        static_cast<const BF16*>(x_q), static_cast<const BF16*>(x_k),
        static_cast<const BF16*>(x_v), static_cast<const BF16*>(g_in),
        static_cast<const BF16*>(beta_in), w_q, w_k, w_v,
        static_cast<const BF16*>(cs_q), static_cast<const BF16*>(cs_k),
        static_cast<const BF16*>(cs_v), a_log, dt_bias,
        static_cast<BF16*>(out), static_cast<BF16*>(ic_q),
        static_cast<BF16*>(ic_k), static_cast<BF16*>(ic_v),
        static_cast<BF16*>(rawv), static_cast<BF16*>(rawk), gate, beta,
        slots, slots, cu, H, Q, scale, lower_bound, strides, h0, native);
  } else {
    verify_kernel<1, Out::kPlain, true><<<dim3(H, N, 1), THREADS, sizeof(Smem), stream>>>(
      static_cast<const BF16*>(x_q), static_cast<const BF16*>(x_k),
      static_cast<const BF16*>(x_v), static_cast<const BF16*>(g_in),
      static_cast<const BF16*>(beta_in), w_q, w_k, w_v,
      static_cast<const BF16*>(cs_q), static_cast<const BF16*>(cs_k),
      static_cast<const BF16*>(cs_v), a_log, dt_bias, h0,
      static_cast<BF16*>(out), static_cast<BF16*>(ic_q),
      static_cast<BF16*>(ic_k), static_cast<BF16*>(ic_v),
      static_cast<BF16*>(rawv), static_cast<BF16*>(rawk), gate, beta,
      slots, slots, cu, H, Q, Q, scale, lower_bound, strides, native);
  }
  if constexpr (kNativeNumericalProtection) {
    if (ppu15) verify_precise_native<<<(static_cast<int64_t>(H) * N + 127) / 128, 128, 0, stream>>>(
      static_cast<const BF16*>(x_q), static_cast<const BF16*>(x_k),
      static_cast<const BF16*>(x_v), static_cast<const BF16*>(g_in),
      static_cast<const BF16*>(beta_in), w_q, w_k, w_v,
      static_cast<const BF16*>(cs_q), static_cast<const BF16*>(cs_k),
      static_cast<const BF16*>(cs_v), a_log, dt_bias, h0,
      static_cast<BF16*>(out), static_cast<BF16*>(ic_q),
      static_cast<BF16*>(ic_k), static_cast<BF16*>(ic_v),
      static_cast<BF16*>(rawv), static_cast<BF16*>(rawk), gate, beta,
      slots, cu, H, N, Q, scale, lower_bound, strides, native);
  }
}

}  // namespace kda_mtp
}  // namespace pla
