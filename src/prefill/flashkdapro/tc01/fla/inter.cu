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

#include "heads.h"
#include "fla_activation.h"
// Frozen FLA long inter-solve: tc01, N=1, H=8, K=128, BT=64, BC=16.
// Arithmetic/write domains: docs/fla_long_inter_contract.md.
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <cstdint>
#include <type_traits>

// Correlated keys and slow decay make the TF32 polynomial inverse unstable:
// large powers cancel to a small inverse, and their TF32 error can turn the
// chunk recurrence into an expanding map. Use the FP32 triangular solve for
// full chunks as well as tails. Zero remains a diagnostic negative control.
#ifndef FLA_INTER_FORCE_FP32_DIAGONAL
#define FLA_INTER_FORCE_FP32_DIAGONAL 1
#endif

#if !defined(__HGGCCC__)
#error "FLA long inter-solve requires the fixed PPU SDK compiler"
#endif

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/ppu_tensor_mix.hpp>
#include <cute/arch/mma_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/algorithm/functional.hpp>

namespace fla_inter_solve_cute_legacy {
using namespace cute;
#if FLA_HEADS > 0
constexpr int kH = FLA_HEADS;
#else
#define kH FLA_DEVICE_HEADS(y)
#endif
constexpr int kK = 128;
constexpr int kBT = 64;
constexpr int kBC = 16;
constexpr int kThreads = 32;
constexpr int kSharedBytes = 6144;

using Bf16Mma = decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>{},
    Layout<Shape<_1, _1, _1>>{}));
using Tf32Mma = decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x8_F32TF32TF32F32_TN>{},
    Layout<Shape<_1, _1, _1>>{}));
static_assert(std::is_same_v<
    MMA_Traits<PPU0010_16x16x16_F32BF16BF16F32_TN>::CLayout,
    MMA_Traits<PPU0010_16x16x8_F32TF32TF32F32_TN>::CLayout>);

template <int K>
using LogicalLayout = decltype(tile_to_shape(
    Layout<Shape<_8, Int<K>>, Stride<Int<K>, _1>>{},
    make_shape(_16{}, Int<K>{})));

// Exact producer/reader pairing from operand-probe build04/device02.
// These physical layouts are 32-byte-slice-major, not 64-byte-row-major.
using Bf16WriteLayout = decltype(composition(
    Swizzle<1, 5, 3>{}, _0{}, composition(Swizzle<1, 3, 3>{},
    Layout<Shape<_16, Shape<_16, _2>>,
           Stride<_16, Stride<_1, _256>>>{})));
using Tf32WriteLayout = decltype(composition(
    Swizzle<1, 4, 3>{}, _0{}, composition(Swizzle<1, 2, 3>{},
    Layout<Shape<_16, Shape<_8, _2>>,
           Stride<_8, Stride<_1, _128>>>{})));
using Bf16Copy = Copy_Atom<PPU0010_TSM_LD_SWZL<
    bfloat16_t, 16, 32, false, false, 1>, bfloat16_t>;
using Tf32Copy = Copy_Atom<PPU0010_TSM_LD_SWZL<
    float, 16, 16, false, false, 1>, float>;

// Byte offsets 0/1024/2048 are Q/K/B during BF16. During the solve image0
// is a temporary row-major ownership exchange. During the merge all six
// 1024-byte images use Tf32WriteLayout. No values survive in shared memory
// across those phase boundaries; the live matrices are FP32 C fragments.
struct alignas(128) Bf16Images {
  bfloat16_t q[512];
  bfloat16_t k[512];
  bfloat16_t b[512];
};
union alignas(128) SharedImages {
  Bf16Images bf16;
  uint32_t fp32[6][256];
};
static_assert(sizeof(SharedImages) == kSharedBytes);
static_assert(cosize(Bf16WriteLayout{}) == 512);
static_assert(cosize(Tf32WriteLayout{}) == 256);

template <class Element, class Word>
CUTE_DEVICE Element* tsm_shared_base(Word* pointer) {
  return reinterpret_cast<Element*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(pointer)));
}

CUTE_DEVICE auto make_accumulator(int lane) {
  auto dummy = make_tensor(make_smem_ptr(reinterpret_cast<float*>(0x10)),
      make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{})));
  auto result = Tf32Mma{}.get_thread_slice(lane).partition_fragment_C(dummy);
  clear(result);
  return result;
}

CUTE_DEVICE auto c_coordinates(int lane) {
  return Tf32Mma{}.get_thread_slice(lane).partition_C(
      make_identity_tensor(make_shape(_16{}, _16{})));
}

template <class Mma, class Copy, class SmemA, class SmemB, class Accum>
CUTE_DEVICE void mma_two_k(SmemA const& a, SmemB const& b,
                           Accum& accum, int lane) {
  Mma mma;
  auto thr = mma.get_thread_slice(lane);
  auto rA = thr.partition_fragment_A(a);
  auto rB = thr.partition_fragment_B(b);
  auto copy_a = make_tiled_copy_A(Copy{}, mma);
  auto copy_b = make_tiled_copy_B(Copy{}, mma);
  auto lane_a = copy_a.get_thread_slice(lane);
  auto lane_b = copy_b.get_thread_slice(lane);
  auto sA = lane_a.partition_S(make_mix_tensor_like(a));
  auto sB = lane_b.partition_S(make_mix_tensor_like(b));
  auto tA = lane_a.retile_D(rA);
  auto tB = lane_b.retile_D(rB);
  static_assert(decltype(size<2>(rA))::value == 2);
  static_assert(decltype(size<2>(rB))::value == 2);
  CUTE_UNROLL
  for (int kt = 0; kt < 2; ++kt) {
    copy(copy_a, sA(_, _, kt), tA(_, _, kt));
    copy(copy_b, sB(_, _, kt), tB(_, _, kt));
    // The caller supplies C. Never clear here: both K tiles, all BF16 slabs,
    // and the TF32 sum-of-dots continue this same FP32 accumulator.
    gemm(mma, rA(_, _, kt), rB(_, _, kt), accum);
  }
}

template <int Width, class Pointer>
CUTE_DEVICE auto global_tensor(Pointer p, int64_t tokens) {
  return make_tensor(make_gmem_ptr(p),
      make_layout(make_shape(tokens, FLA_CUTE_INT(kH), Int<Width>{}),
                  make_stride(FLA_CUTE_INT(kH * Width), Int<Width>{}, _1{})));
}

CUTE_DEVICE float natural_exp(float value) {
  // Frozen tl.exp -> mul log2(e) (0x3fb8aa3b), exp2.f32.
  return __exp2f(__fmul_rn(value, __uint_as_float(0x3fb8aa3bu)));
}

// Fixed-size byte copies preserve raw payloads without type-punned references.
// The binding and producer coordinates establish these vector alignments.
static_assert(sizeof(float4) == 16 && alignof(float4) == 16);
static_assert(sizeof(uint2) == 8 && alignof(uint2) == 8);
template <class Vector, class Element>
CUTE_DEVICE Vector load_aligned_vector(Element const* address) {
  Vector result;
  __builtin_memcpy(&result,
      __builtin_assume_aligned(address, alignof(Vector)), sizeof(Vector));
  return result;
}

CUTE_DEVICE float bf16_vector_component(uint2 const& packed, int component) {
  const uint32_t word = component < 2 ? packed.x : packed.y;
  __ppu_bfloat16_raw raw;
  raw.x = static_cast<unsigned short>(word >> ((component & 1) * 16));
  return __bfloat162float(__ppu_bfloat16(raw));
}

// Each producer owns four adjacent BF16 halfwords at an 8-byte-aligned
// address in Bf16WriteLayout. Pack only already-rounded payloads.
CUTE_DEVICE void store_bf16x4(bfloat16_t* address,
                             bfloat16_t const (&values)[4]) {
  const uint2 bits = make_uint2(
      uint32_t(values[0].raw()) | (uint32_t(values[1].raw()) << 16),
      uint32_t(values[2].raw()) | (uint32_t(values[3].raw()) << 16));
  __builtin_memcpy(__builtin_assume_aligned(address, 8), &bits, sizeof(bits));
}

// Triton keeps block-0 K/G live while forming (1,0), (2,0), and (3,0).
// Mirror that reuse explicitly: one lane owns the same four vector groups used
// by bf16_pair, so no cross-lane exchange or shared-memory round trip is added.
struct Source0Cache {
  float4 gate[4];
  uint2 k[4];
};

template <class TensorK, class TensorG>
CUTE_DEVICE Source0Cache load_source0_cache(TensorK const& k, TensorG const& g,
    int64_t start, int head, int slab, int lane) {
  Source0Cache result;
  CUTE_UNROLL
  for (int group = 0; group < 4; ++group) {
    const int linear = lane * 4 + group * 128;
    const int row = linear / 32;
    const int col = linear % 32;
    const int d = slab * 32 + col;
    result.gate[group] = load_aligned_vector<float4>(&g(start + row, head, d));
    result.k[group] = load_aligned_vector<uint2>(&k(start + row, head, d));
  }
  return result;
}

template <int R, int S, class TensorQ, class TensorK, class TensorG, class Accum>
CUTE_DEVICE void bf16_pair(TensorQ const& q, TensorK const& k,
    TensorG const& g, int64_t start, int64_t end, int head, int slab,
    Source0Cache& source0, SharedImages& shared,
    Accum& cq, Accum& ck, int lane) {
  const int64_t target = start + R * kBC;
  const int64_t source = start + S * kBC;
  if (target >= end) return;  // uniform: protects the anchor and all barriers
  auto sq = make_tensor(make_smem_ptr(shared.bf16.q), Bf16WriteLayout{});
  auto sk = make_tensor(make_smem_ptr(shared.bf16.k), Bf16WriteLayout{});
  auto sb = make_tensor(make_smem_ptr(shared.bf16.b), Bf16WriteLayout{});
  __syncthreads();  // previous pair/slab has completed all TSM reads
  // All groups share these four features. Load the valid target anchor once.
  const int anchor_d = slab * 32 + (lane * 4) % 32;
  const float4 anchor_vector = load_aligned_vector<float4>(&g(target, head, anchor_d));
  const float anchor[4] = {
      anchor_vector.x, anchor_vector.y, anchor_vector.z, anchor_vector.w};
  // Load gates and raw target Q/K across all four groups; retain the original
  // compute masks below. Invalid target slots are zero, never a math mask.
  float4 target_gate_cache[4] = {};
  uint2 target_q_cache[4] = {};
  uint2 target_k_cache[4] = {};
  float4 source_gate_cache[4];
  uint2 source_k_cache[4];
  CUTE_UNROLL
  for (int group = 0; group < 4; ++group) {
    const int linear = lane * 4 + group * 128;
    const int row = linear / 32;
    const int col = linear % 32;
    const int d = slab * 32 + col;
    if constexpr (S == 0) {
      if (target + row < end) {
        target_gate_cache[group] =
            load_aligned_vector<float4>(&g(target + row, head, d));
        target_q_cache[group] =
            load_aligned_vector<uint2>(&q(target + row, head, d));
        target_k_cache[group] =
            load_aligned_vector<uint2>(&k(target + row, head, d));
      }
    }
  }
  // The first target batch is the dominant M12 memory-dependency hotspot.
  // Issue all target Q/K/G reads first, then fill the sole source-0 cache so
  // the two independent global batches overlap before either is consumed.
  if constexpr (R == 1 && S == 0) {
    source0 = load_source0_cache(k, g, start, head, slab, lane);
  }
  CUTE_UNROLL
  for (int group = 0; group < 4; ++group) {
    const int linear = lane * 4 + group * 128;
    const int row = linear / 32;
    const int col = linear % 32;
    const int d = slab * 32 + col;
    if constexpr (S == 0) {
      source_gate_cache[group] = source0.gate[group];
      source_k_cache[group] = source0.k[group];
    } else {
      source_gate_cache[group] =
          load_aligned_vector<float4>(&g(source + row, head, d));
    }
  }
  CUTE_UNROLL
  for (int group = 0; group < 4; ++group) {
    const int linear = lane * 4 + group * 128;
    const int row = linear / 32;
    const int col = linear % 32;
    const int d = slab * 32 + col;
    float qv[4] = {}, kv[4] = {};
    // Caller order (R,0) before (R,S>0) retains complete Q/K, including
    // tail zeros, until the next R/slab. Every call still refreshes B.
    if constexpr (S == 0) {
      // Mask the whole row before any wide target access. No feature tail exists.
      if (target + row < end) {
        const float4 gate_vector = target_gate_cache[group];
        const uint2 q_vector = target_q_cache[group];
        const uint2 k_vector = target_k_cache[group];
        const float gate[4] = {
            gate_vector.x, gate_vector.y, gate_vector.z, gate_vector.w};
        CUTE_UNROLL
        for (int component = 0; component < 4; ++component) {
          const float e = natural_exp(__fsub_rn(gate[component], anchor[component]));
          qv[component] = __fmul_rn(bf16_vector_component(q_vector, component), e);
          kv[component] = __fmul_rn(bf16_vector_component(k_vector, component), e);
        }
      }
    }
    // Any source block preceding an active target is a complete 16 rows.
    const float4 source_gate_vector = source_gate_cache[group];
    uint2 source_k_vector;
    if constexpr (S == 0) {
      source_k_vector = source_k_cache[group];
    } else {
      source_k_vector =
          load_aligned_vector<uint2>(&k(source + row, head, d));
    }
    const float source_gate[4] = {
        source_gate_vector.x, source_gate_vector.y,
        source_gate_vector.z, source_gate_vector.w};
    bfloat16_t rounded_q[4], rounded_k[4], rounded_b[4];
    CUTE_UNROLL
    for (int component = 0; component < 4; ++component) {
      const float be = natural_exp(__fsub_rn(anchor[component], source_gate[component]));
      const float bv = __fmul_rn(bf16_vector_component(source_k_vector, component), be);
      // Full producers retain the same Q/K roundings; reuse consumers keep
      // those already-rounded TSM bytes. B is newly rounded for every pair.
      if constexpr (S == 0) {
        rounded_q[component] = bfloat16_t(qv[component]);
        rounded_k[component] = bfloat16_t(kv[component]);
      }
      rounded_b[component] = bfloat16_t(bv);
    }
    if constexpr (S == 0) {
      store_bf16x4(&sq(row, col), rounded_q);
      store_bf16x4(&sk(row, col), rounded_k);
    }
    store_bf16x4(&sb(row, col), rounded_b);
  }
  __syncthreads();  // computed BF16 producer -> physical TSM reader
  auto rq = make_tensor(make_smem_ptr(tsm_shared_base<bfloat16_t>(shared.bf16.q)),
                        LogicalLayout<32>{});
  auto rk = make_tensor(make_smem_ptr(tsm_shared_base<bfloat16_t>(shared.bf16.k)),
                        LogicalLayout<32>{});
  auto rb = make_tensor(make_smem_ptr(tsm_shared_base<bfloat16_t>(shared.bf16.b)),
                        LogicalLayout<32>{});
  mma_two_k<Bf16Mma, Bf16Copy>(rq, rb, cq, lane);
  mma_two_k<Bf16Mma, Bf16Copy>(rk, rb, ck, lane);
}

template <int R, int S, class Beta, class Output, class Accum>
CUTE_DEVICE void finish_pair(Beta const& beta, Output& aqk,
    int64_t start, int64_t end, int head, float scale,
    Accum const& cq, Accum& ck, int lane) {
  auto coord = c_coordinates(lane);
  // Each half owns one row across four C values; retain both original masks.
  float beta_cache[2] = {};
  CUTE_UNROLL
  for (int half = 0; half < 2; ++half) {
    auto rc = coord(4 * half);
    const int64_t token = start + R * kBC + get<0>(rc);
    if (token < end) beta_cache[half] = fla_sigmoid_bf16(beta(token, head));
  }
  CUTE_UNROLL
  for (int v = 0; v < size(cq); ++v) {
    auto rc = coord(v);
    const int64_t token = start + R * kBC + get<0>(rc);
    if (token < end) {
      aqk(token, head, S * kBC + get<1>(rc)) =
          __float2bfloat16_rn(__fmul_rn(cq(v), scale));
      ck(v) = __fmul_rn(ck(v), beta_cache[v / 4]);
    } else {
      ck(v) = 0.0f;
    }
  }
}

template <int R>
CUTE_DEVICE int diagonal_active(int64_t start, int64_t end) {
  const int64_t remaining = end - (start + R * kBC);
  return remaining <= 0 ? 0 :
      (remaining < kBC ? static_cast<int>(remaining) : kBC);
}

template <int R, class Input, class Accum>
CUTE_DEVICE void initialize_diagonal(Input const& akkd, int64_t start,
    int64_t end, int head, Accum& result, int lane) {
  const int row = lane >> 1;
  const int col0 = 8 * (lane & 1);
  const int64_t base = start + R * kBC;
  const int active = diagonal_active<R>(start, end);
  float raw[8] = {};
  // A row has 16 contiguous FP32 values and both half-row bases are 16-byte
  // aligned.  Fetch only four-word groups containing at least one live
  // lower-triangular element; masking below preserves the original -0.0 bits.
  if (row < active && row > col0) {
    const float4 words =
        load_aligned_vector<float4>(&akkd(base + row, head, col0));
    raw[0] = words.x; raw[1] = words.y;
    raw[2] = words.z; raw[3] = words.w;
  }
  if (row < active && row > col0 + 4) {
    const float4 words =
        load_aligned_vector<float4>(&akkd(base + row, head, col0 + 4));
    raw[4] = words.x; raw[5] = words.y;
    raw[6] = words.z; raw[7] = words.w;
  }
  CUTE_UNROLL
  for (int v = 0; v < 8; ++v) {
    const int col = col0 + v;
    result(v) = row > col ? -raw[v] : -0.0f;
  }
}

template <class Accum>
CUTE_DEVICE void exchange_diagonal(SharedImages& shared, Accum& result,
                                    int lane) {
  const int row = lane >> 1;
  const int col0 = 8 * (lane & 1);
  CUTE_UNROLL
  for (int v = 0; v < 8; ++v) {
    const int col = col0 + v;
    result(v) = __fadd_rn(result(v), row == col ? 1.0f : 0.0f);
    shared.fp32[0][row * 16 + col] = __float_as_uint(result(v));
  }
  __syncthreads();
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(result); ++v) {
    auto rc = coord(v);
    result(v) = __uint_as_float(shared.fp32[0][get<0>(rc) * 16 + get<1>(rc)]);
  }
  __syncthreads();
}

template <class Input, class Accum>
CUTE_DEVICE void diagonal_solve_four(Input const& akkd, int64_t start,
    int64_t end, int head, SharedImages& shared, Accum& d0, Accum& d1,
    Accum& d2, Accum& d3, int lane) {
  const int row = lane >> 1;
  const int col0 = 8 * (lane & 1);
  const int active0 = diagonal_active<0>(start, end);
  const int active1 = diagonal_active<1>(start, end);
  const int active2 = diagonal_active<2>(start, end);
  const int active3 = diagonal_active<3>(start, end);
  initialize_diagonal<0>(akkd, start, end, head, d0, lane);
  initialize_diagonal<1>(akkd, start, end, head, d1, lane);
  initialize_diagonal<2>(akkd, start, end, head, d2, lane);
  initialize_diagonal<3>(akkd, start, end, head, d3, lane);
  CUTE_NO_UNROLL
  for (int i = 2; i < kBC; ++i) {
    const bool live0 = i < active0;
    const bool live1 = i < active1;
    const bool live2 = i < active2;
    const bool live3 = i < active3;
    const float a0 = live0 && row < i ? -akkd(start + i, head, row) : 0.0f;
    const float a1 = live1 && row < i ? -akkd(start + kBC + i, head, row) : 0.0f;
    const float a2 = live2 && row < i ? -akkd(start + 2 * kBC + i, head, row) : 0.0f;
    const float a3 = live3 && row < i ? -akkd(start + 3 * kBC + i, head, row) : 0.0f;
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
      const int col = col0 + v;
      float product_d0 = __fmul_rn(a0, d0(v));
      float product_d1 = __fmul_rn(a1, d1(v));
      float product_d2 = __fmul_rn(a2, d2(v));
      float product_d3 = __fmul_rn(a3, d3(v));
      CUTE_UNROLL
      for (int mask = 16; mask >= 2; mask >>= 1) {
        product_d0 = __fadd_rn(product_d0,
            __shfl_xor_sync(0xffffffffu, product_d0, mask));
        product_d1 = __fadd_rn(product_d1,
            __shfl_xor_sync(0xffffffffu, product_d1, mask));
        product_d2 = __fadd_rn(product_d2,
            __shfl_xor_sync(0xffffffffu, product_d2, mask));
        product_d3 = __fadd_rn(product_d3,
            __shfl_xor_sync(0xffffffffu, product_d3, mask));
      }
      // Lane 2*col holds original masked a, not updated diagonal/product.
      const float original0 = __shfl_sync(0xffffffffu, a0, 2 * col, 32);
      const float original1 = __shfl_sync(0xffffffffu, a1, 2 * col, 32);
      const float original2 = __shfl_sync(0xffffffffu, a2, 2 * col, 32);
      const float original3 = __shfl_sync(0xffffffffu, a3, 2 * col, 32);
      if (live0 && row == i) d0(v) = __fadd_rn(original0, product_d0);
      if (live1 && row == i) d1(v) = __fadd_rn(original1, product_d1);
      if (live2 && row == i) d2(v) = __fadd_rn(original2, product_d2);
      if (live3 && row == i) d3(v) = __fadd_rn(original3, product_d3);
    }
  }
  // Image0 changes ownership, not precision. One initial barrier plus the
  // post-load barrier from each exchange safely separates all four images.
  __syncthreads();
  exchange_diagonal(shared, d0, lane);
  exchange_diagonal(shared, d1, lane);
  exchange_diagonal(shared, d2, lane);
  exchange_diagonal(shared, d3, lane);
}

template <bool Right, class Accum>
CUTE_DEVICE void stage_fp32(uint32_t* image, Accum const& value, int lane) {
  auto write = make_tensor(make_smem_ptr(image), Tf32WriteLayout{});
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    // TN computes left @ stored_B^T: right X must be written at (n,k)=(col,row).
    const uint32_t bits = cutlass::tfloat32_t::bitcast(__float_as_uint(value(v))).raw();
    if constexpr (Right) write(col, row) = bits;
    else write(row, col) = bits;
  }
}

template <class Accum>
CUTE_DEVICE void tf32_dot(uint32_t* left, uint32_t* right, Accum& c, int lane) {
  auto a = make_tensor(make_smem_ptr(tsm_shared_base<float>(left)), LogicalLayout<16>{});
  auto b = make_tensor(make_smem_ptr(tsm_shared_base<float>(right)), LogicalLayout<16>{});
  mma_two_k<Tf32Mma, Tf32Copy>(a, b, c, lane);
}

template <class Accum>
CUTE_DEVICE void copy_accumulator(Accum& destination, Accum const& source) {
  CUTE_UNROLL
  for (int v = 0; v < size(destination); ++v) destination(v) = source(v);
}

template <int R, class Input, class Accum>
CUTE_DEVICE void load_full_diagonal_x0(Input const& akkd, int64_t start,
    int head, Accum& result, int lane) {
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(result); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    result(v) = row == col ? 1.0f :
        (row > col ? -akkd(start + R * kBC + row, head, col) : 0.0f);
  }
}

template <class Accum>
CUTE_DEVICE void stage_identity_plus(uint32_t* image, Accum const& value,
                                     int lane) {
  auto write = make_tensor(make_smem_ptr(image), Tf32WriteLayout{});
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    const float transformed = __fadd_rn(value(v), row == col ? 1.0f : 0.0f);
    const uint32_t bits =
        cutlass::tfloat32_t::bitcast(__float_as_uint(transformed)).raw();
    write(col, row) = bits;
  }
}

template <class Accum>
CUTE_DEVICE void polynomial_square(SharedImages& shared, Accum const& power,
                                   Accum& temporary, int lane) {
  __syncthreads();
  stage_fp32<false>(shared.fp32[4], power, lane);
  stage_fp32<true>(shared.fp32[5], power, lane);
  clear(temporary);
  __syncthreads();
  tf32_dot(shared.fp32[4], shared.fp32[5], temporary, lane);
}

template <class Accum>
CUTE_DEVICE void polynomial_factor(SharedImages& shared, Accum const& result,
                                   Accum const& power, Accum& temporary,
                                   int lane) {
  __syncthreads();
  stage_fp32<false>(shared.fp32[4], result, lane);
  stage_identity_plus(shared.fp32[5], power, lane);
  clear(temporary);
  __syncthreads();
  tf32_dot(shared.fp32[4], shared.fp32[5], temporary, lane);
}

template <int R, class Input, class Accum>
CUTE_DEVICE void diagonal_poly_full(Input const& akkd, int64_t start, int head,
    SharedImages& shared, Accum& result, int lane) {
  // Strictly-lower A is nilpotent (A^16=0):
  // (I+A)^-1 = (I-A)(I+A^2)(I+A^4)(I+A^8).
  load_full_diagonal_x0<R>(akkd, start, head, result, lane);
  auto power = make_accumulator(lane);
  auto temporary = make_accumulator(lane);
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(power); ++v) {
    auto rc = coord(v);
    power(v) = (get<0>(rc) == get<1>(rc) ? 1.0f : 0.0f) - result(v);
  }
  CUTE_UNROLL
  for (int exponent = 2; exponent <= 8; exponent <<= 1) {
    polynomial_square(shared, power, temporary, lane);
    copy_accumulator(power, temporary);
    polynomial_factor(shared, result, power, temporary, lane);
    copy_accumulator(result, temporary);
  }
  // Restore the exact unit-lower structure after TF32 MMA.  In particular,
  // inverse[row,row-1] is exactly -A[row,row-1], so use the FP32 input and
  // avoid a one-BF16-ULP boundary error from TF32 operand rounding.
  CUTE_UNROLL
  for (int v = 0; v < size(result); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    // The factorization already preserves the structural 1/0 entries exactly;
    // only the first subdiagonal needs an FP32-input rounding repair.
    if (row == col + 1)
      result(v) = -akkd(start + R * kBC + row, head, col);
  }
}

template <class Accum>
CUTE_DEVICE void negate_accumulator(Accum& value) {
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) value(v) = -value(v);
}

template <int R, int S, class Output, class Accum>
CUTE_DEVICE void store_inverse(Output& output, Accum const& value,
    int64_t start, int64_t end, int head, int lane) {
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) {
    auto rc = coord(v);
    const int64_t token = start + R * kBC + get<0>(rc);
    if (token < end) {
      output(token, head, S * kBC + get<1>(rc)) = __float2bfloat16_rn(value(v));
    }
  }
}

}  // namespace fla_inter_solve_cute_legacy

namespace fla_inter_solve_cute_wide {
using namespace cute;
#if FLA_HEADS > 0
constexpr int kH = FLA_HEADS;
#else
#define kH FLA_DEVICE_HEADS(y)
#endif
constexpr int kK = 128;
constexpr int kBT = 64;
constexpr int kBC = 16;
constexpr int kThreads = 192;
constexpr int kSharedBytes = 22528;

using Bf16Mma = decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>{},
    Layout<Shape<_1, _1, _1>>{}));
using Tf32Mma = decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x8_F32TF32TF32F32_TN>{},
    Layout<Shape<_1, _1, _1>>{}));
static_assert(std::is_same_v<
    MMA_Traits<PPU0010_16x16x16_F32BF16BF16F32_TN>::CLayout,
    MMA_Traits<PPU0010_16x16x8_F32TF32TF32F32_TN>::CLayout>);

template <int K>
using LogicalLayout = decltype(tile_to_shape(
    Layout<Shape<_8, Int<K>>, Stride<Int<K>, _1>>{},
    make_shape(_16{}, Int<K>{})));

// Exact producer/reader pairing from operand-probe build04/device02.
// These physical layouts are 32-byte-slice-major, not 64-byte-row-major.
using Bf16WriteLayout = decltype(composition(
    Swizzle<1, 5, 3>{}, _0{}, composition(Swizzle<1, 3, 3>{},
    Layout<Shape<_16, Shape<_16, _2>>,
           Stride<_16, Stride<_1, _256>>>{})));
using Tf32WriteLayout = decltype(composition(
    Swizzle<1, 4, 3>{}, _0{}, composition(Swizzle<1, 2, 3>{},
    Layout<Shape<_16, Shape<_8, _2>>,
           Stride<_8, Stride<_1, _128>>>{})));
using Bf16Copy = Copy_Atom<PPU0010_TSM_LD_SWZL<
    bfloat16_t, 16, 32, false, false, 1>, bfloat16_t>;
using Tf32Copy = Copy_Atom<PPU0010_TSM_LD_SWZL<
    float, 16, 16, false, false, 1>, float>;

// The CTA is six warps. Warps 0-5 each own one off-diagonal pair with a
// private Bf16Images buffer (blocks 3w..3w+2); warp w<4 additionally owns one
// diagonal inverse, polynomial-staged through its own blocks 3w+1/3w+2. Each
// pair warp publishes its beta-weighted CK into block 3*w (inside its own
// buffer, so no cross-warp race); diagonal results publish into the separate
// diagpub blocks 18-21. After one CTA-wide barrier the block
// forward-substitution runs on warps 0-2 with two private staging blocks per
// warp (phase1/2 on three warps, phase3||phase4 on two, phase5 on warp 0);
// every accumulator's dot sequence stays on a single warp with bit-identical
// operands, so results are bit-exact against the serial chain.
struct alignas(128) Bf16Images {
  bfloat16_t q[512];
  bfloat16_t k[512];
  bfloat16_t b[512];
};
union alignas(128) SharedArena {
  struct alignas(128) {
    Bf16Images pairs[6];
    uint32_t diagpub[4][256];
  } a;
  uint32_t img[22][256];
};
static_assert(sizeof(SharedArena) == kSharedBytes);
static_assert(sizeof(Bf16Images) == 3 * 1024);
static_assert(cosize(Bf16WriteLayout{}) == 512);
static_assert(cosize(Tf32WriteLayout{}) == 256);
// Image-block map. Pair warp w publishes at block 3*w; polynomial staging for
// diagonal warp w<4 uses blocks 3*w+1/3*w+2; d00..d33 publish at blocks
// 18..21; chain staging uses two blocks per chain warp (w0: 1,2; w1: 4,5;
// w2: 7,8) -- all free after the single CTA barrier.
constexpr int kPairPublish[6] = {0, 3, 6, 9, 12, 15};
constexpr int kDiagPublish = 18;  // d00, d11, d22, d33 at blocks 18..21.
constexpr int kChainA[2] = {1, 2};
constexpr int kChainB[2] = {4, 5};
constexpr int kChainC[2] = {7, 8};

template <class Element, class Word>
CUTE_DEVICE Element* tsm_shared_base(Word* pointer) {
  return reinterpret_cast<Element*>(
      static_cast<uintptr_t>(__cvta_generic_to_shared(pointer)));
}

CUTE_DEVICE auto make_accumulator(int lane) {
  auto dummy = make_tensor(make_smem_ptr(reinterpret_cast<float*>(0x10)),
      make_layout(make_shape(_16{}, _16{}), make_stride(_16{}, _1{})));
  auto result = Tf32Mma{}.get_thread_slice(lane).partition_fragment_C(dummy);
  clear(result);
  return result;
}

CUTE_DEVICE auto c_coordinates(int lane) {
  return Tf32Mma{}.get_thread_slice(lane).partition_C(
      make_identity_tensor(make_shape(_16{}, _16{})));
}

template <class Mma, class Copy, class SmemA, class SmemB, class Accum>
CUTE_DEVICE void mma_two_k(SmemA const& a, SmemB const& b,
                           Accum& accum, int lane) {
  Mma mma;
  auto thr = mma.get_thread_slice(lane);
  auto rA = thr.partition_fragment_A(a);
  auto rB = thr.partition_fragment_B(b);
  auto copy_a = make_tiled_copy_A(Copy{}, mma);
  auto copy_b = make_tiled_copy_B(Copy{}, mma);
  auto lane_a = copy_a.get_thread_slice(lane);
  auto lane_b = copy_b.get_thread_slice(lane);
  auto sA = lane_a.partition_S(make_mix_tensor_like(a));
  auto sB = lane_b.partition_S(make_mix_tensor_like(b));
  auto tA = lane_a.retile_D(rA);
  auto tB = lane_b.retile_D(rB);
  static_assert(decltype(size<2>(rA))::value == 2);
  static_assert(decltype(size<2>(rB))::value == 2);
  CUTE_UNROLL
  for (int kt = 0; kt < 2; ++kt) {
    copy(copy_a, sA(_, _, kt), tA(_, _, kt));
    copy(copy_b, sB(_, _, kt), tB(_, _, kt));
    // The caller supplies C. Never clear here: both K tiles, all BF16 slabs,
    // and the TF32 sum-of-dots continue this same FP32 accumulator.
    gemm(mma, rA(_, _, kt), rB(_, _, kt), accum);
  }
}

template <int Width, class Pointer>
CUTE_DEVICE auto global_tensor(Pointer p, int64_t tokens) {
  return make_tensor(make_gmem_ptr(p),
      make_layout(make_shape(tokens, FLA_CUTE_INT(kH), Int<Width>{}),
                  make_stride(FLA_CUTE_INT(kH * Width), Int<Width>{}, _1{})));
}

CUTE_DEVICE float natural_exp(float value) {
  // Frozen tl.exp -> mul log2(e) (0x3fb8aa3b), exp2.f32.
  return __exp2f(__fmul_rn(value, __uint_as_float(0x3fb8aa3bu)));
}

// Fixed-size byte copies preserve raw payloads without type-punned references.
// The binding and producer coordinates establish these vector alignments.
static_assert(sizeof(float4) == 16 && alignof(float4) == 16);
static_assert(sizeof(uint2) == 8 && alignof(uint2) == 8);
template <class Vector, class Element>
CUTE_DEVICE Vector load_aligned_vector(Element const* address) {
  Vector result;
  __builtin_memcpy(&result,
      __builtin_assume_aligned(address, alignof(Vector)), sizeof(Vector));
  return result;
}

CUTE_DEVICE float bf16_vector_component(uint2 const& packed, int component) {
  const uint32_t word = component < 2 ? packed.x : packed.y;
  __ppu_bfloat16_raw raw;
  raw.x = static_cast<unsigned short>(word >> ((component & 1) * 16));
  return __bfloat162float(__ppu_bfloat16(raw));
}

// Each producer owns four adjacent BF16 halfwords at an 8-byte-aligned
// address in Bf16WriteLayout. Pack only already-rounded payloads.
CUTE_DEVICE void store_bf16x4(bfloat16_t* address,
                             bfloat16_t const (&values)[4]) {
  const uint2 bits = make_uint2(
      uint32_t(values[0].raw()) | (uint32_t(values[1].raw()) << 16),
      uint32_t(values[2].raw()) | (uint32_t(values[3].raw()) << 16));
  __builtin_memcpy(__builtin_assume_aligned(address, 8), &bits, sizeof(bits));
}

// One pair per warp with a private image buffer: every pair produces its own
// complete target Q/K images (the widened form of the old S==0 production
// path). The loads, exp/mul sequences, and roundings are value-identical to
// the narrow kernel; only the buffer address and barrier width changed.
template <int R, int S, class TensorQ, class TensorK, class TensorG, class Accum>
CUTE_DEVICE void bf16_pair(TensorQ const& q, TensorK const& k,
    TensorG const& g, int64_t start, int64_t end, int head, int slab,
    Bf16Images& images, Accum& cq, Accum& ck, int lane) {
  const int64_t target = start + R * kBC;
  const int64_t source = start + S * kBC;
  if (target >= end) return;  // warp-uniform: protects the anchor and barriers
  auto sq = make_tensor(make_smem_ptr(images.q), Bf16WriteLayout{});
  auto sk = make_tensor(make_smem_ptr(images.k), Bf16WriteLayout{});
  auto sb = make_tensor(make_smem_ptr(images.b), Bf16WriteLayout{});
  __syncwarp();  // this warp's previous slab has completed all TSM reads
  // All groups share these four features. Load the valid target anchor once.
  const int anchor_d = slab * 32 + (lane * 4) % 32;
  const float4 anchor_vector = load_aligned_vector<float4>(&g(target, head, anchor_d));
  const float anchor[4] = {
      anchor_vector.x, anchor_vector.y, anchor_vector.z, anchor_vector.w};
  // Load gates and raw target Q/K across all four groups; retain the original
  // compute masks below. Invalid target slots are zero, never a math mask.
  float4 target_gate_cache[4] = {};
  uint2 target_q_cache[4] = {};
  uint2 target_k_cache[4] = {};
  CUTE_UNROLL
  for (int group = 0; group < 4; ++group) {
    const int linear = lane * 4 + group * 128;
    const int row = linear / 32;
    const int col = linear % 32;
    const int d = slab * 32 + col;
    if (target + row < end) {
      target_gate_cache[group] =
          load_aligned_vector<float4>(&g(target + row, head, d));
      target_q_cache[group] =
          load_aligned_vector<uint2>(&q(target + row, head, d));
      target_k_cache[group] =
          load_aligned_vector<uint2>(&k(target + row, head, d));
    }
  }
  // The target batch is the dominant M12 memory-dependency hotspot. Issue all
  // target Q/K/G reads first, then the independent source K/G reads, so the
  // two global batches overlap before either is consumed.
  float4 source_gate_cache[4];
  uint2 source_k_cache[4];
  CUTE_UNROLL
  for (int group = 0; group < 4; ++group) {
    const int linear = lane * 4 + group * 128;
    const int row = linear / 32;
    const int col = linear % 32;
    const int d = slab * 32 + col;
    source_gate_cache[group] =
        load_aligned_vector<float4>(&g(source + row, head, d));
    source_k_cache[group] =
        load_aligned_vector<uint2>(&k(source + row, head, d));
  }
  CUTE_UNROLL
  for (int group = 0; group < 4; ++group) {
    const int linear = lane * 4 + group * 128;
    const int row = linear / 32;
    const int col = linear % 32;
    float qv[4] = {}, kv[4] = {};
    // Mask the whole row before any wide target access. No feature tail exists.
    if (target + row < end) {
      const float4 gate_vector = target_gate_cache[group];
      const uint2 q_vector = target_q_cache[group];
      const uint2 k_vector = target_k_cache[group];
      const float gate[4] = {
          gate_vector.x, gate_vector.y, gate_vector.z, gate_vector.w};
      CUTE_UNROLL
      for (int component = 0; component < 4; ++component) {
        const float e = natural_exp(__fsub_rn(gate[component], anchor[component]));
        qv[component] = __fmul_rn(bf16_vector_component(q_vector, component), e);
        kv[component] = __fmul_rn(bf16_vector_component(k_vector, component), e);
      }
    }
    // Any source block preceding an active target is a complete 16 rows.
    const float4 source_gate_vector = source_gate_cache[group];
    const uint2 source_k_vector = source_k_cache[group];
    const float source_gate[4] = {
        source_gate_vector.x, source_gate_vector.y,
        source_gate_vector.z, source_gate_vector.w};
    bfloat16_t rounded_q[4], rounded_k[4], rounded_b[4];
    CUTE_UNROLL
    for (int component = 0; component < 4; ++component) {
      const float be = natural_exp(__fsub_rn(anchor[component], source_gate[component]));
      const float bv = __fmul_rn(bf16_vector_component(source_k_vector, component), be);
      rounded_q[component] = bfloat16_t(qv[component]);
      rounded_k[component] = bfloat16_t(kv[component]);
      rounded_b[component] = bfloat16_t(bv);
    }
    store_bf16x4(&sq(row, col), rounded_q);
    store_bf16x4(&sk(row, col), rounded_k);
    store_bf16x4(&sb(row, col), rounded_b);
  }
  __syncwarp();  // computed BF16 producer -> physical TSM reader (same warp)
  auto rq = make_tensor(make_smem_ptr(tsm_shared_base<bfloat16_t>(images.q)),
                        LogicalLayout<32>{});
  auto rk = make_tensor(make_smem_ptr(tsm_shared_base<bfloat16_t>(images.k)),
                        LogicalLayout<32>{});
  auto rb = make_tensor(make_smem_ptr(tsm_shared_base<bfloat16_t>(images.b)),
                        LogicalLayout<32>{});
  mma_two_k<Bf16Mma, Bf16Copy>(rq, rb, cq, lane);
  mma_two_k<Bf16Mma, Bf16Copy>(rk, rb, ck, lane);
}

template <int R, int S, class Beta, class Output, class Accum>
CUTE_DEVICE void finish_pair(Beta const& beta, Output& aqk,
    int64_t start, int64_t end, int head, float scale,
    Accum const& cq, Accum& ck, int lane) {
  auto coord = c_coordinates(lane);
  // Each half owns one row across four C values; retain both original masks.
  float beta_cache[2] = {};
  CUTE_UNROLL
  for (int half = 0; half < 2; ++half) {
    auto rc = coord(4 * half);
    const int64_t token = start + R * kBC + get<0>(rc);
    if (token < end) beta_cache[half] = fla_sigmoid_bf16(beta(token, head));
  }
  CUTE_UNROLL
  for (int v = 0; v < size(cq); ++v) {
    auto rc = coord(v);
    const int64_t token = start + R * kBC + get<0>(rc);
    if (token < end) {
      aqk(token, head, S * kBC + get<1>(rc)) =
          __float2bfloat16_rn(__fmul_rn(cq(v), scale));
      ck(v) = __fmul_rn(ck(v), beta_cache[v / 4]);
    } else {
      ck(v) = 0.0f;
    }
  }
}

template <int R>
CUTE_DEVICE int diagonal_active(int64_t start, int64_t end) {
  const int64_t remaining = end - (start + R * kBC);
  return remaining <= 0 ? 0 :
      (remaining < kBC ? static_cast<int>(remaining) : kBC);
}

template <int R, class Input, class Accum>
CUTE_DEVICE void initialize_diagonal(Input const& akkd, int64_t start,
    int64_t end, int head, Accum& result, int lane) {
  const int row = lane >> 1;
  const int col0 = 8 * (lane & 1);
  const int64_t base = start + R * kBC;
  const int active = diagonal_active<R>(start, end);
  float raw[8] = {};
  // A row has 16 contiguous FP32 values and both half-row bases are 16-byte
  // aligned.  Fetch only four-word groups containing at least one live
  // lower-triangular element; masking below preserves the original -0.0 bits.
  if (row < active && row > col0) {
    const float4 words =
        load_aligned_vector<float4>(&akkd(base + row, head, col0));
    raw[0] = words.x; raw[1] = words.y;
    raw[2] = words.z; raw[3] = words.w;
  }
  if (row < active && row > col0 + 4) {
    const float4 words =
        load_aligned_vector<float4>(&akkd(base + row, head, col0 + 4));
    raw[4] = words.x; raw[5] = words.y;
    raw[6] = words.z; raw[7] = words.w;
  }
  CUTE_UNROLL
  for (int v = 0; v < 8; ++v) {
    const int col = col0 + v;
    result(v) = row > col ? -raw[v] : -0.0f;
  }
}

template <class Accum>
CUTE_DEVICE void exchange_diagonal(uint32_t* image, Accum& result, int lane) {
  const int row = lane >> 1;
  const int col0 = 8 * (lane & 1);
  CUTE_UNROLL
  for (int v = 0; v < 8; ++v) {
    const int col = col0 + v;
    result(v) = __fadd_rn(result(v), row == col ? 1.0f : 0.0f);
    image[row * 16 + col] = __float_as_uint(result(v));
  }
  __syncwarp();
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(result); ++v) {
    auto rc = coord(v);
    result(v) = __uint_as_float(image[get<0>(rc) * 16 + get<1>(rc)]);
  }
  __syncwarp();
}

template <class Input, class Accum>
CUTE_DEVICE void diagonal_solve_four(Input const& akkd, int64_t start,
    int64_t end, int head, uint32_t* image, Accum& d0, Accum& d1,
    Accum& d2, Accum& d3, int lane) {
  const int row = lane >> 1;
  const int col0 = 8 * (lane & 1);
  const int active0 = diagonal_active<0>(start, end);
  const int active1 = diagonal_active<1>(start, end);
  const int active2 = diagonal_active<2>(start, end);
  const int active3 = diagonal_active<3>(start, end);
  initialize_diagonal<0>(akkd, start, end, head, d0, lane);
  initialize_diagonal<1>(akkd, start, end, head, d1, lane);
  initialize_diagonal<2>(akkd, start, end, head, d2, lane);
  initialize_diagonal<3>(akkd, start, end, head, d3, lane);
  CUTE_NO_UNROLL
  for (int i = 2; i < kBC; ++i) {
    const bool live0 = i < active0;
    const bool live1 = i < active1;
    const bool live2 = i < active2;
    const bool live3 = i < active3;
    const float a0 = live0 && row < i ? -akkd(start + i, head, row) : 0.0f;
    const float a1 = live1 && row < i ? -akkd(start + kBC + i, head, row) : 0.0f;
    const float a2 = live2 && row < i ? -akkd(start + 2 * kBC + i, head, row) : 0.0f;
    const float a3 = live3 && row < i ? -akkd(start + 3 * kBC + i, head, row) : 0.0f;
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
      const int col = col0 + v;
      float product_d0 = __fmul_rn(a0, d0(v));
      float product_d1 = __fmul_rn(a1, d1(v));
      float product_d2 = __fmul_rn(a2, d2(v));
      float product_d3 = __fmul_rn(a3, d3(v));
      CUTE_UNROLL
      for (int mask = 16; mask >= 2; mask >>= 1) {
        product_d0 = __fadd_rn(product_d0,
            __shfl_xor_sync(0xffffffffu, product_d0, mask));
        product_d1 = __fadd_rn(product_d1,
            __shfl_xor_sync(0xffffffffu, product_d1, mask));
        product_d2 = __fadd_rn(product_d2,
            __shfl_xor_sync(0xffffffffu, product_d2, mask));
        product_d3 = __fadd_rn(product_d3,
            __shfl_xor_sync(0xffffffffu, product_d3, mask));
      }
      // Lane 2*col holds original masked a, not updated diagonal/product.
      const float original0 = __shfl_sync(0xffffffffu, a0, 2 * col, 32);
      const float original1 = __shfl_sync(0xffffffffu, a1, 2 * col, 32);
      const float original2 = __shfl_sync(0xffffffffu, a2, 2 * col, 32);
      const float original3 = __shfl_sync(0xffffffffu, a3, 2 * col, 32);
      if (live0 && row == i) d0(v) = __fadd_rn(original0, product_d0);
      if (live1 && row == i) d1(v) = __fadd_rn(original1, product_d1);
      if (live2 && row == i) d2(v) = __fadd_rn(original2, product_d2);
      if (live3 && row == i) d3(v) = __fadd_rn(original3, product_d3);
    }
  }
  // The exchange image changes ownership, not precision. One initial barrier
  // plus the post-load barrier from each exchange safely separates the uses.
  __syncwarp();
  exchange_diagonal(image, d0, lane);
  exchange_diagonal(image, d1, lane);
  exchange_diagonal(image, d2, lane);
  exchange_diagonal(image, d3, lane);
}

template <bool Right, class Accum>
CUTE_DEVICE void stage_fp32(uint32_t* image, Accum const& value, int lane) {
  auto write = make_tensor(make_smem_ptr(image), Tf32WriteLayout{});
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    // TN computes left @ stored_B^T: right X must be written at (n,k)=(col,row).
    const uint32_t bits = cutlass::tfloat32_t::bitcast(__float_as_uint(value(v))).raw();
    if constexpr (Right) write(col, row) = bits;
    else write(row, col) = bits;
  }
}

// Publish an accumulator's exact FP32 bits at neutral (row, col) coordinates.
// The chain phase re-stages from these words, so the TF32 bitcast happens on
// bit-identical inputs.
template <class Accum>
CUTE_DEVICE void publish_accumulator(uint32_t* image, Accum const& value,
                                     int lane) {
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) {
    auto rc = coord(v);
    image[get<0>(rc) * 16 + get<1>(rc)] = __float_as_uint(value(v));
  }
}

template <bool Right>
CUTE_DEVICE void stage_fp32_from(uint32_t* image, uint32_t const* published,
                                 int lane) {
  auto write = make_tensor(make_smem_ptr(image), Tf32WriteLayout{});
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < 8; ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    const uint32_t bits =
        cutlass::tfloat32_t::bitcast(published[row * 16 + col]).raw();
    if constexpr (Right) write(col, row) = bits;
    else write(row, col) = bits;
  }
}

template <class Accum>
CUTE_DEVICE void tf32_dot(uint32_t* left, uint32_t* right, Accum& c, int lane) {
  auto a = make_tensor(make_smem_ptr(tsm_shared_base<float>(left)), LogicalLayout<16>{});
  auto b = make_tensor(make_smem_ptr(tsm_shared_base<float>(right)), LogicalLayout<16>{});
  mma_two_k<Tf32Mma, Tf32Copy>(a, b, c, lane);
}

// Chain-phase dots on one warp: stage both operands into the warp's two
// private slots with warp-level barriers bracketing the TSM reads. Operand
// bits and per-accumulator dot order match the serial chain exactly.
template <class Accum>
CUTE_DEVICE void dot_from_publish(uint32_t* slot_left, uint32_t* slot_right,
    uint32_t const* left_published, uint32_t const* right_published,
    Accum& c, int lane) {
  __syncwarp();
  stage_fp32_from<false>(slot_left, left_published, lane);
  stage_fp32_from<true>(slot_right, right_published, lane);
  __syncwarp();
  tf32_dot(slot_left, slot_right, c, lane);
}

template <class Accum>
CUTE_DEVICE void dot_publish_accum(uint32_t* slot_left, uint32_t* slot_right,
    uint32_t const* left_published, Accum const& right, Accum& c, int lane) {
  __syncwarp();
  stage_fp32_from<false>(slot_left, left_published, lane);
  stage_fp32<true>(slot_right, right, lane);
  __syncwarp();
  tf32_dot(slot_left, slot_right, c, lane);
}

template <class Accum>
CUTE_DEVICE void dot_accum_publish(uint32_t* slot_left, uint32_t* slot_right,
    Accum const& left, uint32_t const* right_published, Accum& c, int lane) {
  __syncwarp();
  stage_fp32<false>(slot_left, left, lane);
  stage_fp32_from<true>(slot_right, right_published, lane);
  __syncwarp();
  tf32_dot(slot_left, slot_right, c, lane);
}

template <class Accum>
CUTE_DEVICE void copy_accumulator(Accum& destination, Accum const& source) {
  CUTE_UNROLL
  for (int v = 0; v < size(destination); ++v) destination(v) = source(v);
}

template <int R, class Input, class Accum>
CUTE_DEVICE void load_full_diagonal_x0(Input const& akkd, int64_t start,
    int head, Accum& result, int lane) {
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(result); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    result(v) = row == col ? 1.0f :
        (row > col ? -akkd(start + R * kBC + row, head, col) : 0.0f);
  }
}

template <class Accum>
CUTE_DEVICE void stage_identity_plus(uint32_t* image, Accum const& value,
                                     int lane) {
  auto write = make_tensor(make_smem_ptr(image), Tf32WriteLayout{});
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    const float transformed = __fadd_rn(value(v), row == col ? 1.0f : 0.0f);
    const uint32_t bits =
        cutlass::tfloat32_t::bitcast(__float_as_uint(transformed)).raw();
    write(col, row) = bits;
  }
}

template <class Accum>
CUTE_DEVICE void polynomial_square(uint32_t* left_image,
                                   uint32_t* right_image, Accum const& power,
                                   Accum& temporary, int lane) {
  __syncwarp();
  stage_fp32<false>(left_image, power, lane);
  stage_fp32<true>(right_image, power, lane);
  clear(temporary);
  __syncwarp();
  tf32_dot(left_image, right_image, temporary, lane);
}

template <class Accum>
CUTE_DEVICE void polynomial_factor(uint32_t* left_image,
                                   uint32_t* right_image, Accum const& result,
                                   Accum const& power, Accum& temporary,
                                   int lane) {
  __syncwarp();
  stage_fp32<false>(left_image, result, lane);
  stage_identity_plus(right_image, power, lane);
  clear(temporary);
  __syncwarp();
  tf32_dot(left_image, right_image, temporary, lane);
}

template <int R, class Input, class Accum>
CUTE_DEVICE void diagonal_poly_full(Input const& akkd, int64_t start, int head,
    uint32_t* left_image, uint32_t* right_image, Accum& result, int lane) {
  // Strictly-lower A is nilpotent (A^16=0):
  // (I+A)^-1 = (I-A)(I+A^2)(I+A^4)(I+A^8).
  load_full_diagonal_x0<R>(akkd, start, head, result, lane);
  auto power = make_accumulator(lane);
  auto temporary = make_accumulator(lane);
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(power); ++v) {
    auto rc = coord(v);
    power(v) = (get<0>(rc) == get<1>(rc) ? 1.0f : 0.0f) - result(v);
  }
  CUTE_UNROLL
  for (int exponent = 2; exponent <= 8; exponent <<= 1) {
    polynomial_square(left_image, right_image, power, temporary, lane);
    copy_accumulator(power, temporary);
    polynomial_factor(left_image, right_image, result, power, temporary, lane);
    copy_accumulator(result, temporary);
  }
  // Restore the exact unit-lower structure after TF32 MMA.  In particular,
  // inverse[row,row-1] is exactly -A[row,row-1], so use the FP32 input and
  // avoid a one-BF16-ULP boundary error from TF32 operand rounding.
  CUTE_UNROLL
  for (int v = 0; v < size(result); ++v) {
    auto rc = coord(v);
    const int row = get<0>(rc), col = get<1>(rc);
    // The factorization already preserves the structural 1/0 entries exactly;
    // only the first subdiagonal needs an FP32-input rounding repair.
    if (row == col + 1)
      result(v) = -akkd(start + R * kBC + row, head, col);
  }
}

template <class Accum>
CUTE_DEVICE void negate_accumulator(Accum& value) {
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) value(v) = -value(v);
}

template <int R, int S, class Output, class Accum>
CUTE_DEVICE void store_inverse(Output& output, Accum const& value,
    int64_t start, int64_t end, int head, int lane) {
  auto coord = c_coordinates(lane);
  CUTE_UNROLL
  for (int v = 0; v < size(value); ++v) {
    auto rc = coord(v);
    const int64_t token = start + R * kBC + get<0>(rc);
    if (token < end) {
      output(token, head, S * kBC + get<1>(rc)) = __float2bfloat16_rn(value(v));
    }
  }
}

// Four K32 slabs of the owning pair, then the global AQK write and the
// beta-weighted CK publish for the chain phase.
template <int R, int S, class TensorQ, class TensorK, class TensorG,
          class Beta, class Output>
CUTE_DEVICE void run_pair(TensorQ const& q, TensorK const& k,
    TensorG const& g, Beta const& beta, Output& aqk,
    int64_t start, int64_t end, int head, float scale,
    Bf16Images& images, uint32_t* publish, int lane) {
  auto cq = make_accumulator(lane), ck = make_accumulator(lane);
  CUTE_NO_UNROLL
  for (int slab = 0; slab < 4; ++slab) {
    bf16_pair<R, S>(q, k, g, start, end, head, slab, images, cq, ck, lane);
  }
  finish_pair<R, S>(beta, aqk, start, end, head, scale, cq, ck, lane);
  publish_accumulator(publish, ck, lane);
}

}  // namespace fla_inter_solve_cute_wide

extern "C" __global__ __launch_bounds__(32, 1)
__block_size__((32, 1, 1))
void fla_inter_solve_bf16_d128_bt64_bc16_h8_kernel(
    const __ppu_bfloat16* __restrict__ q, const __ppu_bfloat16* __restrict__ k,
    const float* __restrict__ g, const __ppu_bfloat16* __restrict__ beta_raw,
    int64_t beta_stride,
    const float* __restrict__ akkd, __ppu_bfloat16* __restrict__ aqk,
    __ppu_bfloat16* __restrict__ akk, float scale, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ chunk_indices, int chunks, int total_tokens, int n) {
  using namespace fla_inter_solve_cute_legacy;
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& shared = *reinterpret_cast<SharedImages*>(shared_bytes);
  const int lane = static_cast<int>(threadIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int64_t chunk = static_cast<int64_t>(blockIdx.x);
  if (chunk >= chunks || head >= kH) return;
  auto indices = make_tensor(make_gmem_ptr(chunk_indices),
      make_layout(make_shape(static_cast<int64_t>(chunks), _2{}), make_stride(_2{}, _1{})));
  const int sequence = indices(chunk, 0);
  const int local_chunk = indices(chunk, 1);
  // The adapter is N1-only; reject an invalid sequence before reading cu.
  if (sequence < 0 || sequence >= n || local_chunk < 0) return;
  auto gCu = make_tensor(make_gmem_ptr(cu), make_layout(make_shape(int64_t(n) + 1)));
  const int64_t bos = gCu(static_cast<int64_t>(sequence));
  const int64_t end = gCu(static_cast<int64_t>(sequence) + 1);
  const int64_t start = bos + static_cast<int64_t>(local_chunk) * kBT;
  if (bos < 0 || end > total_tokens || start >= end) return;
  const int64_t extent = total_tokens;
  auto gQ = global_tensor<kK>(q, extent);
  auto gK = global_tensor<kK>(k, extent);
  auto gG = global_tensor<kK>(g, extent);
  auto gBeta = make_tensor(make_gmem_ptr(beta_raw),
      make_layout(make_shape(extent, FLA_CUTE_INT(kH)), make_stride(beta_stride, _1{})));
  auto gAkkd = global_tensor<kBC>(akkd, extent);
  auto gAqk = global_tensor<kBT>(aqk, extent);
  auto gAkk = global_tensor<kBT>(akk, extent);

  auto a10 = make_accumulator(lane), a20 = make_accumulator(lane);
  auto a21 = make_accumulator(lane), a30 = make_accumulator(lane);
  auto a31 = make_accumulator(lane), a32 = make_accumulator(lane);
  {
    auto q10 = make_accumulator(lane), q20 = make_accumulator(lane);
    auto q21 = make_accumulator(lane), q30 = make_accumulator(lane);
    auto q31 = make_accumulator(lane), q32 = make_accumulator(lane);
    // Four K32 slabs; six expanded pairs yield 24 static BF16 atom calls.
    CUTE_NO_UNROLL
    for (int slab = 0; slab < 4; ++slab) {
      Source0Cache source0{};
      bf16_pair<1, 0>(gQ, gK, gG, start, end, head, slab, source0, shared, q10, a10, lane);
      bf16_pair<2, 0>(gQ, gK, gG, start, end, head, slab, source0, shared, q20, a20, lane);
      bf16_pair<2, 1>(gQ, gK, gG, start, end, head, slab, source0, shared, q21, a21, lane);
      bf16_pair<3, 0>(gQ, gK, gG, start, end, head, slab, source0, shared, q30, a30, lane);
      bf16_pair<3, 1>(gQ, gK, gG, start, end, head, slab, source0, shared, q31, a31, lane);
      bf16_pair<3, 2>(gQ, gK, gG, start, end, head, slab, source0, shared, q32, a32, lane);
    }
    finish_pair<1, 0>(gBeta, gAqk, start, end, head, scale, q10, a10, lane);
    finish_pair<2, 0>(gBeta, gAqk, start, end, head, scale, q20, a20, lane);
    finish_pair<2, 1>(gBeta, gAqk, start, end, head, scale, q21, a21, lane);
    finish_pair<3, 0>(gBeta, gAqk, start, end, head, scale, q30, a30, lane);
    finish_pair<3, 1>(gBeta, gAqk, start, end, head, scale, q31, a31, lane);
    finish_pair<3, 2>(gBeta, gAqk, start, end, head, scale, q32, a32, lane);
  }

  auto d00 = make_accumulator(lane), d11 = make_accumulator(lane);
  auto d22 = make_accumulator(lane), d33 = make_accumulator(lane);
  if (!FLA_INTER_FORCE_FP32_DIAGONAL && end - start >= kBT) {
    diagonal_poly_full<0>(gAkkd, start, head, shared, d00, lane);
    diagonal_poly_full<1>(gAkkd, start, head, shared, d11, lane);
    diagonal_poly_full<2>(gAkkd, start, head, shared, d22, lane);
    diagonal_poly_full<3>(gAkkd, start, head, shared, d33, lane);
  } else {
    diagonal_solve_four(gAkkd, start, end, head, shared,
                        d00, d11, d22, d33, lane);
  }
  auto d10 = make_accumulator(lane), d21 = make_accumulator(lane);
  auto d32 = make_accumulator(lane), d20 = make_accumulator(lane);
  auto d31 = make_accumulator(lane), d30 = make_accumulator(lane);
  auto& s = shared.fp32;
  {
    auto t10 = make_accumulator(lane), t21 = make_accumulator(lane);
    auto t32 = make_accumulator(lane);
    // Phase1: six distinct image lives, three independent first dots.
    __syncthreads();
    stage_fp32<false>(s[0], d11, lane); stage_fp32<true>(s[1], a10, lane);
    stage_fp32<false>(s[2], d22, lane); stage_fp32<true>(s[3], a21, lane);
    stage_fp32<false>(s[4], d33, lane); stage_fp32<true>(s[5], a32, lane);
    __syncthreads();
    tf32_dot(s[0], s[1], t10, lane);
    tf32_dot(s[2], s[3], t21, lane);
    tf32_dot(s[4], s[5], t32, lane);
    // Phase2: old six operands dead; publish second-dot operands.
    __syncthreads();
    stage_fp32<false>(s[0], t10, lane); stage_fp32<true>(s[1], d00, lane);
    stage_fp32<false>(s[2], t21, lane); stage_fp32<true>(s[3], d11, lane);
    stage_fp32<false>(s[4], t32, lane); stage_fp32<true>(s[5], d22, lane);
    __syncthreads();
    tf32_dot(s[0], s[1], d10, lane);
    tf32_dot(s[2], s[3], d21, lane);
    tf32_dot(s[4], s[5], d32, lane);
    negate_accumulator(d10); negate_accumulator(d21); negate_accumulator(d32);
  }
  {
    auto t20 = make_accumulator(lane);
    // Phase3: s4 holds D22 until the final dot; s5 is initially dead.
    __syncthreads();
    stage_fp32<false>(s[0], a20, lane); stage_fp32<true>(s[1], d00, lane);
    stage_fp32<false>(s[2], a21, lane); stage_fp32<true>(s[3], d10, lane);
    stage_fp32<false>(s[4], d22, lane);
    __syncthreads();
    tf32_dot(s[0], s[1], t20, lane);
    tf32_dot(s[2], s[3], t20, lane);  // C is the preceding logical dot
    __syncthreads();
    stage_fp32<true>(s[5], t20, lane);
    __syncthreads();
    tf32_dot(s[4], s[5], d20, lane);
    negate_accumulator(d20);
  }
  {
    auto t31 = make_accumulator(lane);
    // Phase4: same two-dot chain, with D33 retained in s4.
    __syncthreads();
    stage_fp32<false>(s[0], a31, lane); stage_fp32<true>(s[1], d11, lane);
    stage_fp32<false>(s[2], a32, lane); stage_fp32<true>(s[3], d21, lane);
    stage_fp32<false>(s[4], d33, lane);
    __syncthreads();
    tf32_dot(s[0], s[1], t31, lane);
    tf32_dot(s[2], s[3], t31, lane);
    __syncthreads();
    stage_fp32<true>(s[5], t31, lane);
    __syncthreads();
    tf32_dot(s[4], s[5], d31, lane);
    negate_accumulator(d31);
  }
  {
    auto t30 = make_accumulator(lane);
    // Phase5: all six images feed the three-dot chain before s0/s1 reuse.
    __syncthreads();
    stage_fp32<false>(s[0], a30, lane); stage_fp32<true>(s[1], d00, lane);
    stage_fp32<false>(s[2], a31, lane); stage_fp32<true>(s[3], d10, lane);
    stage_fp32<false>(s[4], a32, lane); stage_fp32<true>(s[5], d20, lane);
    __syncthreads();
    tf32_dot(s[0], s[1], t30, lane);
    tf32_dot(s[2], s[3], t30, lane);
    tf32_dot(s[4], s[5], t30, lane);
    __syncthreads();
    stage_fp32<false>(s[0], d33, lane); stage_fp32<true>(s[1], t30, lane);
    __syncthreads();
    tf32_dot(s[0], s[1], d30, lane);
    negate_accumulator(d30);
  }
  // Exactly ten block-lower tiles. Caller owns upper zeros; Aqk is not cleared.
  store_inverse<0, 0>(gAkk, d00, start, end, head, lane);
  store_inverse<1, 0>(gAkk, d10, start, end, head, lane);
  store_inverse<1, 1>(gAkk, d11, start, end, head, lane);
  store_inverse<2, 0>(gAkk, d20, start, end, head, lane);
  store_inverse<2, 1>(gAkk, d21, start, end, head, lane);
  store_inverse<2, 2>(gAkk, d22, start, end, head, lane);
  store_inverse<3, 0>(gAkk, d30, start, end, head, lane);
  store_inverse<3, 1>(gAkk, d31, start, end, head, lane);
  store_inverse<3, 2>(gAkk, d32, start, end, head, lane);
  store_inverse<3, 3>(gAkk, d33, start, end, head, lane);
}

extern "C" __global__ __launch_bounds__(192, 1)
__block_size__((192, 1, 1))
void fla_inter_solve_bf16_d128_bt64_bc16_h8_wide_kernel(
    const __ppu_bfloat16* __restrict__ q, const __ppu_bfloat16* __restrict__ k,
    const float* __restrict__ g, const __ppu_bfloat16* __restrict__ beta_raw,
    int64_t beta_stride,
    const float* __restrict__ akkd, __ppu_bfloat16* __restrict__ aqk,
    __ppu_bfloat16* __restrict__ akk, float scale, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ chunk_indices, int chunks, int total_tokens, int n) {
  using namespace fla_inter_solve_cute_wide;
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& arena = *reinterpret_cast<SharedArena*>(shared_bytes);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int head = static_cast<int>(blockIdx.y);
  const int64_t chunk = static_cast<int64_t>(blockIdx.x);
  if (chunk >= chunks || head >= kH) return;
  auto indices = make_tensor(make_gmem_ptr(chunk_indices),
      make_layout(make_shape(static_cast<int64_t>(chunks), _2{}), make_stride(_2{}, _1{})));
  const int sequence = indices(chunk, 0);
  const int local_chunk = indices(chunk, 1);
  // The adapter is N1-only; reject an invalid sequence before reading cu.
  if (sequence < 0 || sequence >= n || local_chunk < 0) return;
  auto gCu = make_tensor(make_gmem_ptr(cu), make_layout(make_shape(int64_t(n) + 1)));
  const int64_t bos = gCu(static_cast<int64_t>(sequence));
  const int64_t end = gCu(static_cast<int64_t>(sequence) + 1);
  const int64_t start = bos + static_cast<int64_t>(local_chunk) * kBT;
  if (bos < 0 || end > total_tokens || start >= end) return;
  const int64_t extent = total_tokens;
  auto gQ = global_tensor<kK>(q, extent);
  auto gK = global_tensor<kK>(k, extent);
  auto gG = global_tensor<kK>(g, extent);
  auto gBeta = make_tensor(make_gmem_ptr(beta_raw),
      make_layout(make_shape(extent, FLA_CUTE_INT(kH)), make_stride(beta_stride, _1{})));
  auto gAkkd = global_tensor<kBC>(akkd, extent);
  auto gAqk = global_tensor<kBT>(aqk, extent);
  auto gAkk = global_tensor<kBT>(akk, extent);

  // Warps 0-5: one off-diagonal pair each, pair order (1,0),(2,0),(2,1),
  // (3,0),(3,1),(3,2). Private buffers make all six fully parallel.
  if (warp == 0) {
    run_pair<1, 0>(gQ, gK, gG, gBeta, gAqk, start, end, head, scale,
                   arena.a.pairs[0], arena.img[kPairPublish[0]], lane);
  } else if (warp == 1) {
    run_pair<2, 0>(gQ, gK, gG, gBeta, gAqk, start, end, head, scale,
                   arena.a.pairs[1], arena.img[kPairPublish[1]], lane);
  } else if (warp == 2) {
    run_pair<2, 1>(gQ, gK, gG, gBeta, gAqk, start, end, head, scale,
                   arena.a.pairs[2], arena.img[kPairPublish[2]], lane);
  } else if (warp == 3) {
    run_pair<3, 0>(gQ, gK, gG, gBeta, gAqk, start, end, head, scale,
                   arena.a.pairs[3], arena.img[kPairPublish[3]], lane);
  } else if (warp == 4) {
    run_pair<3, 1>(gQ, gK, gG, gBeta, gAqk, start, end, head, scale,
                   arena.a.pairs[4], arena.img[kPairPublish[4]], lane);
  } else if (warp == 5) {
    run_pair<3, 2>(gQ, gK, gG, gBeta, gAqk, start, end, head, scale,
                   arena.a.pairs[5], arena.img[kPairPublish[5]], lane);
  }
  // Warps 0-3 then each own one diagonal inverse, polynomial-staged through
  // their own pair buffer's dead K/B image blocks. A tail chunk keeps the
  // serial four-block shfl solve on warp 0 (its block 1 is free scratch).
  // The diagnostic FP32 switch uses that same solve for full chunks as well.
  if (!FLA_INTER_FORCE_FP32_DIAGONAL && end - start >= kBT) {
    if (warp == 0) {
      auto d00 = make_accumulator(lane);
      diagonal_poly_full<0>(gAkkd, start, head,
          arena.img[kPairPublish[0] + 1], arena.img[kPairPublish[0] + 2],
          d00, lane);
      publish_accumulator(arena.img[kDiagPublish + 0], d00, lane);
      store_inverse<0, 0>(gAkk, d00, start, end, head, lane);
    } else if (warp == 1) {
      auto d11 = make_accumulator(lane);
      diagonal_poly_full<1>(gAkkd, start, head,
          arena.img[kPairPublish[1] + 1], arena.img[kPairPublish[1] + 2],
          d11, lane);
      publish_accumulator(arena.img[kDiagPublish + 1], d11, lane);
      store_inverse<1, 1>(gAkk, d11, start, end, head, lane);
    } else if (warp == 2) {
      auto d22 = make_accumulator(lane);
      diagonal_poly_full<2>(gAkkd, start, head,
          arena.img[kPairPublish[2] + 1], arena.img[kPairPublish[2] + 2],
          d22, lane);
      publish_accumulator(arena.img[kDiagPublish + 2], d22, lane);
      store_inverse<2, 2>(gAkk, d22, start, end, head, lane);
    } else if (warp == 3) {
      auto d33 = make_accumulator(lane);
      diagonal_poly_full<3>(gAkkd, start, head,
          arena.img[kPairPublish[3] + 1], arena.img[kPairPublish[3] + 2],
          d33, lane);
      publish_accumulator(arena.img[kDiagPublish + 3], d33, lane);
      store_inverse<3, 3>(gAkk, d33, start, end, head, lane);
    }
  } else if (warp == 0) {
    auto d00 = make_accumulator(lane), d11 = make_accumulator(lane);
    auto d22 = make_accumulator(lane), d33 = make_accumulator(lane);
    diagonal_solve_four(gAkkd, start, end, head,
        arena.img[kPairPublish[0] + 1], d00, d11, d22, d33, lane);
    publish_accumulator(arena.img[kDiagPublish + 0], d00, lane);
    publish_accumulator(arena.img[kDiagPublish + 1], d11, lane);
    publish_accumulator(arena.img[kDiagPublish + 2], d22, lane);
    publish_accumulator(arena.img[kDiagPublish + 3], d33, lane);
    store_inverse<0, 0>(gAkk, d00, start, end, head, lane);
    store_inverse<1, 1>(gAkk, d11, start, end, head, lane);
    store_inverse<2, 2>(gAkk, d22, start, end, head, lane);
    store_inverse<3, 3>(gAkk, d33, start, end, head, lane);
  }
  // Single CTA-wide rendezvous: every published image is fully written. The
  // forward-substitution chain then splits across warps 0-2 (phase1/2 on
  // three warps, phase3||phase4 on warps 0/1, phase5 on warp 0); every
  // accumulator's dot sequence keeps the serial chain's order and operands,
  // so the split replay is bit-exact against the narrow kernel.
  __syncthreads();
  if (warp == 0) {
    auto& s0 = arena.img[kChainA[0]];
    auto& s1 = arena.img[kChainA[1]];
    auto d10 = make_accumulator(lane), d20 = make_accumulator(lane);
    auto d30 = make_accumulator(lane);
    {
      auto t10 = make_accumulator(lane);
      // Phase1/2: t10 = D11*A10; D10 = -(t10*D00).
      dot_from_publish(s0, s1, arena.img[kDiagPublish + 1],
                       arena.img[kPairPublish[0]], t10, lane);
      dot_accum_publish(s0, s1, t10, arena.img[kDiagPublish + 0], d10, lane);
      negate_accumulator(d10);
    }
    {
      auto t20 = make_accumulator(lane);
      // Phase3: t20 = A20*D00 + A21*D10; D20 = -(D22*t20).
      dot_from_publish(s0, s1, arena.img[kPairPublish[1]],
                       arena.img[kDiagPublish + 0], t20, lane);
      dot_publish_accum(s0, s1, arena.img[kPairPublish[2]], d10, t20, lane);
      dot_publish_accum(s0, s1, arena.img[kDiagPublish + 2], t20, d20, lane);
      negate_accumulator(d20);
    }
    {
      auto t30 = make_accumulator(lane);
      // Phase5: t30 = A30*D00 + A31*D10 + A32*D20; D30 = -(D33*t30).
      dot_from_publish(s0, s1, arena.img[kPairPublish[3]],
                       arena.img[kDiagPublish + 0], t30, lane);
      dot_publish_accum(s0, s1, arena.img[kPairPublish[4]], d10, t30, lane);
      dot_publish_accum(s0, s1, arena.img[kPairPublish[5]], d20, t30, lane);
      dot_publish_accum(s0, s1, arena.img[kDiagPublish + 3], t30, d30, lane);
      negate_accumulator(d30);
    }
    store_inverse<1, 0>(gAkk, d10, start, end, head, lane);
    store_inverse<2, 0>(gAkk, d20, start, end, head, lane);
    store_inverse<3, 0>(gAkk, d30, start, end, head, lane);
  } else if (warp == 1) {
    auto& s0 = arena.img[kChainB[0]];
    auto& s1 = arena.img[kChainB[1]];
    auto d21 = make_accumulator(lane), d31 = make_accumulator(lane);
    {
      auto t21 = make_accumulator(lane);
      // Phase1/2: t21 = D22*A21; D21 = -(t21*D11).
      dot_from_publish(s0, s1, arena.img[kDiagPublish + 2],
                       arena.img[kPairPublish[2]], t21, lane);
      dot_accum_publish(s0, s1, t21, arena.img[kDiagPublish + 1], d21, lane);
      negate_accumulator(d21);
    }
    {
      auto t31 = make_accumulator(lane);
      // Phase4 (overlaps warp 0's phase3): t31 = A31*D11 + A32*D21;
      // D31 = -(D33*t31).
      dot_from_publish(s0, s1, arena.img[kPairPublish[4]],
                       arena.img[kDiagPublish + 1], t31, lane);
      dot_publish_accum(s0, s1, arena.img[kPairPublish[5]], d21, t31, lane);
      dot_publish_accum(s0, s1, arena.img[kDiagPublish + 3], t31, d31, lane);
      negate_accumulator(d31);
    }
    store_inverse<2, 1>(gAkk, d21, start, end, head, lane);
    store_inverse<3, 1>(gAkk, d31, start, end, head, lane);
  } else if (warp == 2) {
    auto& s0 = arena.img[kChainC[0]];
    auto& s1 = arena.img[kChainC[1]];
    auto d32 = make_accumulator(lane);
    {
      auto t32 = make_accumulator(lane);
      // Phase1/2: t32 = D33*A32; D32 = -(t32*D22).
      dot_from_publish(s0, s1, arena.img[kDiagPublish + 3],
                       arena.img[kPairPublish[5]], t32, lane);
      dot_accum_publish(s0, s1, t32, arena.img[kDiagPublish + 2], d32, lane);
      negate_accumulator(d32);
    }
    store_inverse<3, 2>(gAkk, d32, start, end, head, lane);
  }
  // Warps 3-5 fall through; the ten block-lower AKK tiles are all stored.
}

// Wide-path dispatch. The 192-thread six-warp wide kernel wins on short
// sequences (latency-dominated: more resident warps per SM hide the
// bf16/TSM/mma latency; ABBA short geomean 0.9434 vs the legacy 32-thread
// kernel) but loses once the device is throughput-saturated (the legacy
// kernel's eight fully independent CTAs per SM fill each other's latency
// bubbles, while the wide kernel's intra-CTA serial chain leaves warps 3-5
// idle: ABBA long geomean 1.0514, inter stage +16.6% at h8 t16384).
// Both kernels are bit-exact against each other (640/640 align gate), so the
// dispatch choice never changes results. Crossover evidence (exclusive card,
// reps=50, staged build): stage-timing sweep at h8 gives inter ratios 0.70 at
// t508 and 1.05/1.14/1.07/1.07 at t1017/2048/4096/8192, forward ratios
// 0.98 at t1017 vs 1.03 at t2048; the ABBA set shows graph_ms -5.7% over
// short cases (tokens <= 1017) and +5.1% over long ones (tokens >= 4071).
// Recalibrated on the monolithic build (810E exclusive card, reps=200,
// ABBA x2, 2026-09-22): heads, not packed blocks, sets the crossover —
// h12 ragged is already flat at 84 blocks while h8 single-seq still wins
// at 96. Per-head-tier caps:
//   heads <= 4: wins at 8/12/16 chunks (ratios 0.84/0.77/0.89) -> cap 16
//   heads <= 8: wins at 6-12 chunks (0.80-0.91), flat at 16 (1.007) -> cap 12
//   heads >= 12: no win evidence (ragged flat at 7 chunks, +6.4% regression
//                at 13) -> wide disabled
constexpr int kWideMaxChunksH4 = 16;  // heads <= 4: <= ~1024 tokens per head
constexpr int kWideMaxChunksH8 = 12;  // heads <= 8: <= ~768 tokens per head

extern "C" hggcError_t launch_fla_inter_solve_bf16_d128_bt64_bc16_h8(
    const void* q, const void* k, const float* g, const uint16_t* beta_raw,
    int64_t beta_stride,
    const float* akkd, void* aqk, void* akk, float scale,
    const int32_t* cu, const int32_t* chunk_indices,
    int chunks, int total_tokens, int n, int heads, hggcStream_t stream) {
  if (chunks < 0 || total_tokens < 0) return hggcErrorInvalidValue;
  if (chunks == 0) return hggcSuccess;
  if (!q || !k || !g || !beta_raw || !akkd || !aqk || !akk || !cu || !chunk_indices)
    return hggcErrorInvalidValue;
  const int wide_max_chunks =
      heads <= 4 ? kWideMaxChunksH4 : (heads <= 8 ? kWideMaxChunksH8 : 0);
  if (chunks <= wide_max_chunks) {
    fla_inter_solve_bf16_d128_bt64_bc16_h8_wide_kernel<<<
        dim3(chunks, heads, 1), dim3(192, 1, 1),
        fla_inter_solve_cute_wide::kSharedBytes, stream>>>(
        static_cast<const __ppu_bfloat16*>(q), static_cast<const __ppu_bfloat16*>(k),
        g, reinterpret_cast<const __ppu_bfloat16*>(beta_raw), beta_stride,
        akkd, static_cast<__ppu_bfloat16*>(aqk), static_cast<__ppu_bfloat16*>(akk),
        scale, cu, chunk_indices, chunks, total_tokens, n);
  } else {
    fla_inter_solve_bf16_d128_bt64_bc16_h8_kernel<<<
        dim3(chunks, heads, 1), dim3(32, 1, 1),
        fla_inter_solve_cute_legacy::kSharedBytes, stream>>>(
        static_cast<const __ppu_bfloat16*>(q), static_cast<const __ppu_bfloat16*>(k),
        g, reinterpret_cast<const __ppu_bfloat16*>(beta_raw), beta_stride,
        akkd, static_cast<__ppu_bfloat16*>(aqk), static_cast<__ppu_bfloat16*>(akk),
        scale, cu, chunk_indices, chunks, total_tokens, n);
  }
  return hggcGetLastError();
}
