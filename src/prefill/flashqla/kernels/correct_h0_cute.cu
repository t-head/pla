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
 * FlashQLA PPU1.5 CuTe rewrite — Kernel (4) correct_h0
 *
 * plan.md Task 6. Corrects per-CP-segment initial states via the
 * affine-semigroup recurrence:
 *
 *   h[0]   = raw_h0                  (has_raw) or 0
 *   h[i+1] = ht[i] + mt[i] @ h[i]    if fallback_mask[cp_idx, head]
 *          = ht[i]                    otherwise (state fully decayed)
 *
 * The GEMM mt[DK,DK] @ h[DK,DV_TILE] -> [DK,DV_TILE] is NN. The CuTe FIXED
 * atom is TN (C[m,n] = A[m,k] x B[n,k]). To realize NN we stage h TRANSPOSED
 * into hT_buf[DV_TILE, DK] so B[n,k] = h[k,n], giving C = mt @ h.
 *
 * Dense launches pair two independent warp groups so adjacent DV tiles share
 * one 32-KiB mt load.  Low-occupancy launches keep one group per CTA and expose
 * all four DV tiles as independent blocks; the host selects that form only
 * when its complete grid fits in one device wave.
 *
 * Algorithm authority: csrc_ppu15/correct_h0_ppu15.cu.
 * Grid: raw_batch * (H * DV/(NGRP*DV_TILE)),  Block: 128*NGRP
 */
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <cstdint>
#include <mutex>

#include "flashqla_cute_traits.cuh"

namespace flashqla_cute {

namespace fq = flashqla;

constexpr int CH0_DK  = 128;
constexpr int CH0_DV  = 128;
constexpr int CH0_DVT = 32;
constexpr int CH0_WG  = 128;
constexpr int CH0_ELEMS = CH0_DK * CH0_DVT / 128;  // 32 per thread @128 thr

// PPU1.5 can move the dense [DK, DK] transition tile with one AIU 2D copy.
// The scalar int4 loop otherwise assigns consecutive 16-byte vectors to
// consecutive lanes, which ACU reports as 80% excessive cacheline traffic on
// the short CP cases where correct_h0 is latency-critical.  PPU1.0/1.1 keeps
// the established cooperative copy because its AIU descriptor is different.
__device__ __forceinline__ void ch0_load_mt_tc02(
    void* smem_ptr, const void* gmem_ptr) {
#if defined(FLASHQLA_ARCH_150)
    // Use the same proven 32-row tile shape as the fused/prepare K/V loads;
    // four instructions cover the full matrix without relying on a 128-row
    // AIU tile size that those kernels never exercise.
    CUTE_UNROLL
    for (int row = 0; row < CH0_DK; row += 32) {
        auto* smem_row = reinterpret_cast<uint16_t*>(smem_ptr) + row * CH0_DK;
        const auto* gmem_row =
            reinterpret_cast<const uint16_t*>(gmem_ptr) + row * CH0_DK;
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(smem_row));
        asm volatile(
            "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
            " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
            :: "r"(smem_addr), "l"(gmem_row),
               "r"(CH0_DK), "r"(32), "r"(1),
               "r"(CH0_DK), "r"(32), "r"(1),
               "l"(int64_t(CH0_DK * sizeof(uint16_t))), "l"(int64_t(0)),
               "r"(0), "r"(0), "r"(0),
               "r"(0)
            : "memory");
    }
#else
    (void)smem_ptr;
    (void)gmem_ptr;
#endif
}

__device__ __forceinline__ void ch0_aiu_commit() {
#if defined(FLASHQLA_ARCH_150)
    asm volatile("ppu.cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void ch0_aiu_wait_all() {
#if defined(FLASHQLA_ARCH_150)
    asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
#endif
}

// Flat offset into a [cp/raw, H, DK, DV] tensor for (idx0, head, dk, dv-in-tile).
__device__ __forceinline__ int64_t ch0_off(
    int idx0, int head, int dk, int dv_tile, int dv, int H) {
    return (int64_t)idx0 * H * CH0_DK * CH0_DV +
           (int64_t)head * CH0_DK * CH0_DV +
           (int64_t)dk * CH0_DV + dv_tile * CH0_DVT + dv;
}

template <typename Element, int NGRP>
__global__ void __launch_bounds__(CH0_WG * NGRP, 1)
correct_h0_cute_kernel(
    void* __restrict__ cp_h0,
    const void* __restrict__ raw_h0,         // null when has_raw == 0
    const int* __restrict__ raw_h0_indices,  // optional K-last pool indices
    const Element* __restrict__ ht_buffer,
    const Element* __restrict__ mt_buffer,
    const signed char* __restrict__ fallback_mask,
    const int* __restrict__ seq_map_r2c,
    int cp_batch_size, int raw_batch_size, int H, int has_raw,
    int raw_h0_k_last, bool raw_h0_is_bf16, bool cp_h0_is_bf16)
{
    const int tiles_per_head = CH0_DV / CH0_DVT;      // 4
    const int blocks_per_head = tiles_per_head / NGRP;
    const int blocks_per_batch = H * blocks_per_head;

    const int bid = blockIdx.x;
    const int raw_batch_idx = bid / blocks_per_batch;
    const int sub_block = bid % blocks_per_batch;
    const int head_idx = sub_block / blocks_per_head;
    const int block_dv = sub_block % blocks_per_head;
    const int group = threadIdx.x / CH0_WG;
    const int tx = threadIdx.x % CH0_WG;
    const int dv_tile = block_dv * NGRP + group;

    if (raw_batch_idx >= raw_batch_size) return;
    const int seq_start = seq_map_r2c[raw_batch_idx];
    const int seq_end   = seq_map_r2c[raw_batch_idx + 1];
    const int num_iters = seq_end - seq_start - 1;

    // Keep the running state in the same accumulator layout used by the MMA.
    // The previous implementation kept a second flat register array, wrote the
    // GEMM accumulator to a 16-KiB fp32 smem tile, synchronized, and read it
    // back into that array. Reusing this accumulator preserves the fp32 state
    // and the original arithmetic order while removing that round trip.
    using Mma = fq::TiledMma4WarpT<Element>;
    Mma mma;
    auto thr = mma.get_thread_slice(tx);
    auto c_id = cute::make_identity_tensor(
        cute::make_shape(cute::Int<CH0_DK>{}, cute::Int<CH0_DVT>{}));
    auto tc = thr.partition_C(c_id);
    auto h_acc = fq::make_acc<CH0_DK, CH0_DVT, Mma, Element>(tx);
    cute::clear(h_acc);
    auto rH = cute::recast<float>(h_acc);

    if (has_raw) {
        const int raw_state_idx = raw_h0_indices != nullptr
            ? raw_h0_indices[raw_batch_idx] : raw_batch_idx;
        CUTE_UNROLL
        for (int m = 0; m < cute::size<1>(tc); ++m) {
          CUTE_UNROLL
          for (int n = 0; n < cute::size<2>(tc); ++n) {
            CUTE_UNROLL
            for (int vv = 0; vv < cute::size<0>(tc); ++vv) {
              auto crd = tc(vv, m, n);
              int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
            if (raw_h0_k_last) {
                const int dv_idx = dv_tile * CH0_DVT + dv;
                const int64_t off =
                    ((int64_t)raw_state_idx * H + head_idx) * CH0_DV * CH0_DK +
                    (int64_t)dv_idx * CH0_DK + dk;
                rH(vv, m, n) = fq::load_state_f32(raw_h0, off, raw_h0_is_bf16);
            } else {
                rH(vv, m, n) = fq::load_state_f32(
                    raw_h0, ch0_off(raw_state_idx, head_idx, dk, dv_tile, dv, H),
                    raw_h0_is_bf16);
            }
            }
          }
        }
    }
    if (seq_start >= 0 && seq_start < cp_batch_size) {
        CUTE_UNROLL
        for (int m = 0; m < cute::size<1>(tc); ++m) {
          CUTE_UNROLL
          for (int n = 0; n < cute::size<2>(tc); ++n) {
            CUTE_UNROLL
            for (int vv = 0; vv < cute::size<0>(tc); ++vv) {
              auto crd = tc(vv, m, n);
              int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
              fq::store_state_f32(
                  cp_h0, ch0_off(seq_start, head_idx, dk, dv_tile, dv, H),
                  rH(vv, m, n), cp_h0_is_bf16);
            }
          }
        }
    }
    if (num_iters <= 0) return;

    // Single-buffered smem: one shared mt[DK,DK] (32KB) plus one disjoint
    // hT[DVT,DK] (8KB) per warp group. The MMA result stays in h_acc instead
    // of using a 16KB fp32 tile.
    extern __shared__ char smem[];
    Element* mt_buf = reinterpret_cast<Element*>(smem);
    Element* hT_buf = mt_buf + CH0_DK * CH0_DK +
        group * CH0_DVT * CH0_DK;

    for (int i_s = 0; i_s < num_iters; ++i_s) {
        int cp_idx = seq_start + i_s;
        bool needs_mt = (cp_idx >= 0 && cp_idx < cp_batch_size)
            ? (bool)fallback_mask[cp_idx * H + head_idx] : false;

        if (needs_mt) {
            const Element* mt_src = mt_buffer + (int64_t)cp_idx * H * CH0_DK * CH0_DK +
                                   (int64_t)head_idx * CH0_DK * CH0_DK;
#if defined(FLASHQLA_ARCH_150)
            if (threadIdx.x == 0) {
                ch0_load_mt_tc02(mt_buf, mt_src);
                ch0_aiu_commit();
            }
#else
            constexpr int MT_VEC_COUNT = CH0_DK * CH0_DK * sizeof(Element) / sizeof(int4);
            auto* mt_buf_vec = reinterpret_cast<int4*>(mt_buf);
            const auto* mt_src_vec = reinterpret_cast<const int4*>(mt_src);
            // All 256 threads publish one shared mt tile. Using the block-wide
            // index avoids duplicating the copy in the second warp group.
            for (int idx = threadIdx.x; idx < MT_VEC_COUNT; idx += blockDim.x) {
                mt_buf_vec[idx] = mt_src_vec[idx];
            }
#endif
            CUTE_UNROLL
            for (int m = 0; m < cute::size<1>(tc); ++m) {
              CUTE_UNROLL
              for (int n = 0; n < cute::size<2>(tc); ++n) {
                CUTE_UNROLL
                for (int vv = 0; vv < cute::size<0>(tc); ++vv) {
                  auto crd = tc(vv, m, n);
                  int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
                  hT_buf[dv * CH0_DK + dk] =
                      fq::float_to_element<Element>(rH(vv, m, n));
                }
              }
            }
#if defined(FLASHQLA_ARCH_150)
            if (threadIdx.x == 0) ch0_aiu_wait_all();
#endif
            __syncthreads();

            {
                auto sMt = fq::make_smem_tensor_h<CH0_DK, CH0_DK>(mt_buf);
                auto sHt = fq::make_smem_tensor_h<CH0_DVT, CH0_DK>(hT_buf);
                fq::flashqla_gemm_ss<decltype(sMt), decltype(sHt),
                    decltype(h_acc), Mma, Element>(sMt, sHt, tx, h_acc);
            }
            CUTE_UNROLL
            for (int m = 0; m < cute::size<1>(tc); ++m) {
              CUTE_UNROLL
              for (int n = 0; n < cute::size<2>(tc); ++n) {
                CUTE_UNROLL
                for (int vv = 0; vv < cute::size<0>(tc); ++vv) {
                  auto crd = tc(vv, m, n);
                  int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
                  rH(vv, m, n) += fq::element_to_float<Element>(
                      ht_buffer[ch0_off(
                          cp_idx, head_idx, dk, dv_tile, dv, H)]);
                }
              }
            }
            // Every thread has completed its final mt/hT read before the next
            // iteration may overwrite the single shared-memory buffer.
            __syncthreads();
        } else {
            CUTE_UNROLL
            for (int m = 0; m < cute::size<1>(tc); ++m) {
              CUTE_UNROLL
              for (int n = 0; n < cute::size<2>(tc); ++n) {
                CUTE_UNROLL
                for (int vv = 0; vv < cute::size<0>(tc); ++vv) {
                  auto crd = tc(vv, m, n);
                  int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
                  rH(vv, m, n) = fq::element_to_float<Element>(
                      ht_buffer[ch0_off(
                          cp_idx, head_idx, dk, dv_tile, dv, H)]);
                }
              }
            }
        }

        int out_cp_idx = cp_idx + 1;
        if (out_cp_idx < cp_batch_size) {
            CUTE_UNROLL
            for (int m = 0; m < cute::size<1>(tc); ++m) {
              CUTE_UNROLL
              for (int n = 0; n < cute::size<2>(tc); ++n) {
                CUTE_UNROLL
                for (int vv = 0; vv < cute::size<0>(tc); ++vv) {
                  auto crd = tc(vv, m, n);
                  int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
                  fq::store_state_f32(
                      cp_h0,
                      ch0_off(out_cp_idx, head_idx, dk, dv_tile, dv, H),
                      rH(vv, m, n), cp_h0_is_bf16);
                }
              }
            }
        }
    }
}

}  // namespace flashqla_cute

// ============================================================
// Host launch
// ============================================================
static int ch0_smem_bytes(int ngrp) {
    using namespace flashqla_cute;
    return (CH0_DK * CH0_DK + ngrp * CH0_DVT * CH0_DK) * 2;
}

namespace {

constexpr int CH0_MAX_CACHED_DEVICES = 64;

template <typename Element, int Ngrp>
void ch0_configure_dynamic_smem(int device_index, int smem_bytes) {
    auto configure = [smem_bytes]() {
        if (hggcFuncSetAttribute(
                flashqla_cute::correct_h0_cute_kernel<Element, Ngrp>,
                hggcFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes) != hggcSuccess) {
            hggcGetLastError();
        }
    };
    if (device_index < 0 || device_index >= CH0_MAX_CACHED_DEVICES) {
        configure();
        return;
    }
    // Each template instantiation owns a separate cache; indexing it by
    // logical device avoids both cross-device omissions and first-call races.
    static std::once_flag configured[CH0_MAX_CACHED_DEVICES];
    std::call_once(configured[device_index], configure);
}

}  // namespace

static void ch0_launch(
    void* cp_h0, const void* raw_h0, const void* ht_buffer,
    const void* mt_buffer, const signed char* fallback_mask,
    const int* seq_map_r2c, const int* raw_h0_indices,
    int cp_batch_size, int raw_batch_size, int H,
    int has_raw, int raw_h0_k_last, bool raw_h0_is_bf16,
    bool cp_h0_is_bf16, int dtype_tag,
    int device_index, int num_cu, hggcStream_t stream) {
    const int effective_num_cu = num_cu > 0 ? num_cu : 39;
    // NGRP=1 duplicates the read-only mt tile but exposes four blocks per
    // head. Use it only when every DV tile fits in one physical-CU wave; past
    // that point NGRP=2 has the same wave count and its shared mt load wins.
    const bool use_one_group = raw_batch_size * H * 4 <= effective_num_cu;
    const int ngrp = use_one_group ? 1 : 2;
    int smem_bytes = ch0_smem_bytes(ngrp);
    int blocks_per_batch = H * (
        flashqla_cute::CH0_DV /
        (ngrp * flashqla_cute::CH0_DVT));
    dim3 grid(raw_batch_size * blocks_per_batch);
    dim3 block(flashqla_cute::CH0_WG * ngrp);

#define CH0_LAUNCH_N(ElementType, NG) \
    do { \
      ch0_configure_dynamic_smem<ElementType, NG>(device_index, smem_bytes); \
      flashqla_cute::correct_h0_cute_kernel<ElementType, NG><<<grid, block, smem_bytes, stream>>>( \
          cp_h0, raw_h0, raw_h0_indices, reinterpret_cast<const ElementType*>(ht_buffer), \
          reinterpret_cast<const ElementType*>(mt_buffer), fallback_mask, seq_map_r2c, \
          cp_batch_size, raw_batch_size, H, has_raw, raw_h0_k_last, \
          raw_h0_is_bf16, cp_h0_is_bf16); \
    } while (0)

    if (dtype_tag == 0) {
        if (use_one_group) { CH0_LAUNCH_N(__half, 1); }
        else { CH0_LAUNCH_N(__half, 2); }
    } else {
        if (use_one_group) { CH0_LAUNCH_N(__ppu_bfloat16, 1); }
        else { CH0_LAUNCH_N(__ppu_bfloat16, 2); }
    }

#undef CH0_LAUNCH_N
}

void launch_correct_h0_no_raw_cute(
    void* cp_h0, const void* ht_buffer, const void* mt_buffer,
    const signed char* fallback_mask, const int* seq_map_r2c,
    int cp_batch_size, int raw_batch_size, int H, bool cp_h0_is_bf16,
    int dtype_tag, int device_index, int num_cu, hggcStream_t stream) {
    ch0_launch(cp_h0, nullptr, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
               nullptr, cp_batch_size, raw_batch_size, H, 0, 0, false,
               cp_h0_is_bf16, dtype_tag, device_index, num_cu, stream);
}

void launch_correct_h0_raw_cute(
    void* cp_h0, const void* raw_h0, const void* ht_buffer,
    const void* mt_buffer, const signed char* fallback_mask,
    const int* seq_map_r2c, const int* raw_h0_indices,
    int cp_batch_size, int raw_batch_size, int H, int raw_h0_k_last,
    bool raw_h0_is_bf16, bool cp_h0_is_bf16,
    int dtype_tag, int device_index, int num_cu, hggcStream_t stream) {
    ch0_launch(cp_h0, raw_h0, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
               raw_h0_indices, cp_batch_size, raw_batch_size, H, 1,
               raw_h0_k_last, raw_h0_is_bf16, cp_h0_is_bf16, dtype_tag,
               device_index, num_cu, stream);
}
