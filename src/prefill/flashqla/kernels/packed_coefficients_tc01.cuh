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

// Independent packed coefficients and the measured short KKT fusion.
#pragma once
#include "packed_coefficients_api.h"

namespace flashqla_packed_tc01 {
namespace b = flashqla_kkt_tc01;
namespace fq = flashqla;
constexpr float LOG2E = 1.4426950408889634f;

#if defined(FLASHQLA_ARCH_100)
template <typename E>
__device__ __forceinline__ void compute_qk(E* q, E* k, float* p, int tx) {
    using Traits = b::KktKTraitsTc01<E>;
    using CE = typename Traits::CuteElem;
    using cute::_;
    auto sQ = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<CE*>(q)),
                               typename Traits::SmemLayout{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<CE*>(k)),
                               typename Traits::SmemLayout{});
    fq::TiledMmaM32T<E> mma;
    auto thread = mma.get_thread_slice(tx);
    auto rA = thread.partition_fragment_A(sQ);
    auto rB = thread.partition_fragment_B(sK);
    auto copyA = cute::make_tiled_copy_A(typename Traits::SmemCopyAtom{}, mma);
    auto copyB = cute::make_tiled_copy_B(typename Traits::SmemCopyAtom{}, mma);
    auto sliceA = copyA.get_thread_slice(tx);
    auto sliceB = copyB.get_thread_slice(tx);
    auto sourceA = sliceA.partition_S(cute::make_mix_tensor_like(sQ));
    auto sourceB = sliceB.partition_S(cute::make_mix_tensor_like(sK));
    auto targetA = sliceA.retile_D(rA);
    auto targetB = sliceB.retile_D(rB);
    auto acc = fq::make_acc<32,32,fq::TiledMmaM32T<E>,E>(tx);
    cute::clear(acc);
    CUTE_UNROLL
    for (int part = 0; part < cute::size<2>(rA); ++part) {
        cute::copy(copyA, sourceA(_,_,part), targetA(_,_,part));
        cute::copy(copyB, sourceB(_,_,part), targetB(_,_,part));
        cute::gemm(mma, rA(_,_,part), rB(_,_,part), acc);
    }
    // p aliases k. Every warp must retire all K reads before any P store.
    __syncthreads();
    fq::store_acc_to_smem_fp32<fq::TiledMmaM32T<E>>(acc,p,32,32,32,tx);
    __syncthreads();
}
#endif

template <typename E>
__global__ void __launch_bounds__(128,6) kkt_coefficients_kernel(
    E* a, E* pg, E* ag, float* ex, float* rev,
    const E* q, const E* k, const float* beta, const float* g,
    const int* cu, const int* chunks, int num_chunks, int tokens,
    int H, int Hg, float scale) {
#if defined(FLASHQLA_ARCH_100)
    const int tx = threadIdx.x, warp = tx >> 5, lane = tx & 31;
    const int task = blockIdx.x;
    if (task >= num_chunks * H) return;
    int start, end, local, head;
    b::kkt_task_coords<true>(cu,chunks,H,Hg,task,start,end,local,head);
    const int begin = start + local * 32;
    const int actual = begin < end ? b::kkt_min(32,end-begin) : 0;
    if (actual <= 0) return;
    const int kh = head / (H / Hg);
    extern __shared__ __align__(16) char memory[];
    E* K = reinterpret_cast<E*>(memory+b::KKT_OFF_K0);
    E* Q = reinterpret_cast<E*>(memory+b::KKT_OFF_K1);
    float* gram = reinterpret_cast<float*>(memory+b::KKT_OFF_KKT);
    E* inverse = reinterpret_cast<E*>(memory+b::KKT_OFF_MH);
    float* br = reinterpret_cast<float*>(memory+b::KKT_OFF_B0);
    b::kkt_load_k_tile_tc01<E>(K,k+(int64_t(begin)*Hg+kh)*128,Hg*128,actual,tx);
    b::kkt_load_k_tile_tc01<E>(Q,q+(int64_t(begin)*Hg+kh)*128,Hg*128,actual,tx);
    b::kkt_aiu_commit();
    if (tx < 32) br[tx] = tx < actual ? beta[int64_t(begin+tx)*H+head] : 0.f;
    b::kkt_aiu_wait_group<0>();
    __syncthreads();
    b::kkt_compute_head<E>(K,br,gram,inverse,inverse,actual,tx,warp,lane);
    float* P = reinterpret_cast<float*>(K);
    compute_qk(Q,K,P,tx);
    b::kkt_compute_tri_tc01<E>(a,gram,br,reinterpret_cast<float*>(inverse),
        begin,actual,head,tokens,H,tx,warp,lane);
    __syncthreads();
    // The inverse no longer needs Q. Reuse its first row for scalar gates.
    float* G = reinterpret_cast<float*>(Q);
    if (tx < 32) G[tx] = g[int64_t(begin+b::kkt_min(tx,actual-1))*H+head];
    __syncthreads();
    const float last = G[actual-1];
    if (tx < actual) {
        ex[int64_t(begin+tx)*H+head] = exp2f(G[tx]*LOG2E);
        rev[int64_t(begin+tx)*H+head] = exp2f((last-G[tx])*LOG2E);
    }
    const float* inv = reinterpret_cast<float*>(inverse);
    for (int item = tx; item < actual*32; item += 128) {
        const int row = item/32, col = item%32;
        float pv = 0.f, av = 0.f;
        if (col <= row && col < actual) {
            const float difference = G[row]-G[col];
            pv = (scale*exp2f(difference*LOG2E))*P[row*32+col];
            // This explicit Element boundary is part of the existing contract.
            const E rounded = fq::float_to_element<E>(inv[row*b::KKT_FP32_LD+col]);
            av = (fq::element_to_float<E>(rounded)*
                  exp2f(fminf(difference,0.f)*LOG2E))*br[col];
        }
        const int packed = col ^ (((row>>1)&3)<<3);
        const int64_t address = (int64_t(begin+row)*H+head)*32+packed;
        pg[address] = fq::float_to_element<E>(pv);
        ag[address] = fq::float_to_element<E>(av);
    }
#endif
}
// One QK MMA per QK head; reuse its exact accumulator for all GVA V heads.
template <typename Element>
__global__ void __launch_bounds__(128)
packed_coefficients_kernel(
    Element* __restrict__ pg, Element* __restrict__ ag,
    float* __restrict__ ex, float* __restrict__ rev,
    const Element* __restrict__ q, const Element* __restrict__ k,
    const Element* __restrict__ a, const float* __restrict__ beta,
    const float* __restrict__ g, const int* __restrict__ cu,
    const int* __restrict__ chunks, int H, int Hg, float scale, int heads_per_group) {
#if defined(FLASHQLA_ARCH_100)
  const int chunk = int(blockIdx.x), head_begin = int(blockIdx.y) * heads_per_group;
  const int kh = head_begin / (H / Hg);
  const int seq = chunks[2 * chunk], local = chunks[2 * chunk + 1];
  const int begin = cu[seq] + local * 32;
  const int actual = min(32, cu[seq + 1] - begin);
  if (actual <= 0) return;
  const int l = int(threadIdx.x);
  extern __shared__ char storage[];
  Element* Q = reinterpret_cast<Element*>(storage);
  Element* K = Q + 4096;
  float* G = reinterpret_cast<float*>(K + 4096);
  float* B = G + 32;
  if (l == 0) {
    b::kkt_load_k_tile_tc01<Element>(Q, q + ((int64_t)begin * Hg + kh) * 128, Hg * 128, actual, l);
    b::kkt_load_k_tile_tc01<Element>(K, k + ((int64_t)begin * Hg + kh) * 128, Hg * 128, actual, l);
    b::kkt_aiu_commit(); b::kkt_aiu_wait_group<0>();
  }
  __syncthreads();
  using MmaP = fq::TiledMmaM32T<Element>;
  MmaP mma;
  auto thr = mma.get_thread_slice(l);
  auto acc = fq::make_acc<32, 32, MmaP, Element>(l);
  if ((l >> 5) == 2) cute::clear(acc);
  else fq::flashqla_gemm_ss_swzlAB_m32<32, 32, 128,
      decltype(acc), Element>(Q, K, l, acc);
  auto coords = cute::make_identity_tensor(
      cute::make_shape(cute::Int<32>{}, cute::Int<32>{}));
  auto tile = thr.partition_C(coords);
  auto values = cute::recast<float>(acc);
  for (int head = head_begin; head < head_begin + heads_per_group; ++head) {
    if (l < 32) {
      G[l] = g[(int64_t)(begin + min(l, actual - 1)) * H + head];
      B[l] = beta[(int64_t)(begin + min(l, actual - 1)) * H + head];
    }
    __syncthreads();
    if (l < actual) {
      ex[(int64_t)(begin+l)*H+head] = exp2f(G[l]*LOG2E);
      rev[(int64_t)(begin+l)*H+head] = exp2f((G[actual-1]-G[l])*LOG2E);
    }
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tile); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tile); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tile); ++vv) {
          auto pos = tile(vv, m, n);
          const int row = cute::get<0>(pos), col = cute::get<1>(pos);
          if (row < actual) {
            float value = 0.0f, ag_value = 0.0f;
            if (row >= col && col < actual) {
              const float difference = G[row] - G[col];
              const float ratio = exp2f(difference * LOG2E);
              value = scale * ratio * values(vv, m, n);
              const int64_t offset = ((int64_t)(begin + row) * H + head) * 32 + col;
              const float av = fq::element_to_float<Element>(a[offset]);
              const float ag_ratio = difference <= 0.0f ? ratio : 1.0f;
              ag_value = av * ag_ratio * B[col];
            }
            pg[((int64_t)(begin + row) * H + head) * 32 + (col ^ (((row >> 1) & 3) << 3))] =
                fq::float_to_element<Element>(value);
            ag[((int64_t)(begin + row) * H + head) * 32 + (col ^ (((row >> 1) & 3) << 3))] =
                fq::float_to_element<Element>(ag_value);
          }
        }
      }
    }
    __syncthreads();
  }
#endif
}
}  // namespace flashqla_packed_tc01


void launch_gdn_packed_coefficients_tc01(
    void* a, void* pg, void* ag, float* ex, float* rev,
    const void* q, const void* k, const float* beta, const float* g,
    const int* cu, const int* chunks, int num_chunks, int tokens,
    int H, int Hg, float scale, int dtype, bool fuse_kkt,
    int heads_per_group, hggcStream_t stream) {
  if (!gdn_packed_coefficients_tc01_enabled())
    throw std::runtime_error("packed GDN requires the register-chain reordered schedule");
  if (num_chunks <= 0) return;
  if (heads_per_group <= 0 || (H / Hg) % heads_per_group)
    throw std::runtime_error("invalid packed coefficient head group");
#define PACKED_COEFF(E) do { \
    if (fuse_kkt) { \
      constexpr int shared = flashqla_kkt_tc01::KKT_SMEM_TC01; \
      const auto status = hggcFuncSetAttribute( \
          flashqla_packed_tc01::kkt_coefficients_kernel<E>, \
          hggcFuncAttributeMaxDynamicSharedMemorySize, shared); \
      if (status != hggcSuccess) throw std::runtime_error(hggcGetErrorString(status)); \
      flashqla_packed_tc01::kkt_coefficients_kernel<E> \
          <<<num_chunks*H,128,shared,stream>>>( \
          static_cast<E*>(a),static_cast<E*>(pg),static_cast<E*>(ag),ex,rev, \
          static_cast<const E*>(q),static_cast<const E*>(k),beta,g,cu,chunks, \
          num_chunks,tokens,H,Hg,scale); \
    } else { \
      constexpr int shared = 2*32*128*2 + 2*32*4; \
      flashqla_packed_tc01::packed_coefficients_kernel<E> \
          <<<dim3(num_chunks,H/heads_per_group),128,shared,stream>>>( \
          static_cast<E*>(pg),static_cast<E*>(ag),ex,rev, \
          static_cast<const E*>(q),static_cast<const E*>(k), \
          static_cast<const E*>(a),beta,g,cu,chunks,H,Hg,scale,heads_per_group); \
    } \
  } while (0)
  if (dtype == 0) PACKED_COEFF(__half);
  else PACKED_COEFF(__ppu_bfloat16);
#undef PACKED_COEFF
}
