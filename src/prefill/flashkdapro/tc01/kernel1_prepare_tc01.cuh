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

#pragma once

namespace flashkda_cute {
namespace tc01 {

#if defined(FLASHKDA_ARCH_100)
__forceinline__ __device__ void k1_aiu_load_2d_b16(
    void* smem_ptr, const void* gmem_ptr, int dim_c, int dim_w,
    int64_t stride_w_bytes) {
    // tc01's ppu001 linear 2D load carries the source row stride in dim_c.
    // Fetch only the leading cube_c elements of each row while retaining the
    // caller's stride between rows.
    uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int32_t dim_c_stride = static_cast<int32_t>(stride_w_bytes >> 1);
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.2d.b16"
        " [%0], [%1], {%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_addr), "l"(gmem_ptr),
           "r"(1), "r"(dim_w), "r"(dim_c_stride),
           "r"(0), "r"(0), "r"(0),
           "r"(1), "r"(dim_w), "r"(dim_c), "r"(1)
        : "memory");
}
#endif

template <int NThreads>
__forceinline__ __device__ void k1_fma_triangular_front(
    int tid, const BF16* s_kd, const BF16* s_qd, const BF16* s_ki,
    half_t* sL, BF16* sMqk) {
    // ZW810E has 64 generic CUs but only 20 TensorCell-capable CUs.  FLA's
    // token-parallel triangular dot products deliberately lower to FFMA and
    // run on all 64.  Mirror that division here: this front kernel computes
    // only the 120 strict-lower L dots and 136 lower-inclusive Mqk dots with
    // FP32 FMA.  A small TensorCell tail below retains the six dense fp16
    // Neumann MMAs, so we do not scalarize work for which MMA is still best.
    for (int idx = tid; idx < CHUNK * CHUNK; idx += NThreads) {
        sL[idx] = half_t(0.f);
        sMqk[idx] = BF16(0.f);
    }
    __syncthreads();

    int row = tid >> 3;
    int lane8 = tid & 7;
    for (int j = lane8; j < row; j += 8) {
        float l_acc = 0.f;
        float mqk_acc = 0.f;
        #pragma unroll 8
        for (int c = 0; c < D; ++c) {
            float ki = fk::bf16_to_f32(s_ki[j * D + c]);
            l_acc = fmaf(fk::bf16_to_f32(s_kd[row * D + c]), ki, l_acc);
            mqk_acc = fmaf(fk::bf16_to_f32(s_qd[row * D + c]), ki, mqk_acc);
        }
        sL[row * CHUNK + j] = half_t(l_acc);
        sMqk[row * CHUNK + j] = BF16(mqk_acc);
    }
    // Diagonal Mqk is required, while diagonal L is structurally zero.
    if (lane8 == 0) {
        float mqk_acc = 0.f;
        #pragma unroll 8
        for (int c = 0; c < D; ++c) {
            mqk_acc = fmaf(
                fk::bf16_to_f32(s_qd[row * D + c]),
                fk::bf16_to_f32(s_ki[row * D + c]), mqk_acc);
        }
        sMqk[row * CHUNK + row] = BF16(mqk_acc);
    }
}

}  // namespace tc01

// TensorCell-only tail for the tc01 FMA-front experiment.  The front has
// already rounded the raw L dot to fp16 and multiplied beta with the same
// second fp16 rounding as the monolithic kernel.  This kernel reconstructs INV
// and preserves the original six fp16-accumulate MMA operations bit-for-bit.
struct alignas(128) K1NeumannSmem {
    K1Array<half_t, CHUNK * CHUNK> L;
    K1Array<half_t, CHUNK * CHUNK> INV;
    K1Array<half_t, CHUNK * CHUNK> pow;
};

template <bool IsVarlen, bool UseTilePrefix>
__global__ void __launch_bounds__(32) flashkda_neumann_tail_kernel(
    int N, int total_tiles, int tile_count,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    int tile_offset, int head_offset,
    BF16* __restrict__ ws) {
    const int gtile = int(blockIdx.x) + tile_offset;
    if (gtile >= tile_offset + tile_count) return;
    if constexpr (IsVarlen) {
        if constexpr (UseTilePrefix) {
            if (gtile >= tile_prefix[N]) return;
        } else {
            int tiles_before = 0;
            int seq_idx = 0;
            for (; seq_idx < N; ++seq_idx) {
                int slen = int(cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]);
                int nt = (slen + CHUNK - 1) / CHUNK;
                if (tiles_before + nt > gtile) break;
                tiles_before += nt;
            }
            if (seq_idx == N) return;
        }
    }

    __shared__ K1NeumannSmem smem;
    const int lane = int(threadIdx.x);
    const int head = int(blockIdx.y) + head_offset;
    BF16* slot = ws + int64_t(head * total_tiles + gtile) * fk::WS_SLOT_UNITS;
    BF16* o_inv = slot + fk::WS_OFF_INV_U;
    const half_t* o_l = reinterpret_cast<const half_t*>(o_inv);
    for (int idx = lane; idx < CHUNK * CHUNK; idx += 32) {
        half_t x = o_l[idx];
        smem.L.data[idx] = x;
        int i = idx / CHUNK, j = idx % CHUNK;
        smem.INV.data[idx] = (i == j) ? half_t(1.f) : half_t(-float(x));
    }
    __syncwarp();

    {
        auto acc = fk::make_acc16_f16acc_1warp(lane);
        fk::gemm16_ss_1warp_f16acc(
            view16(smem.L.data), view16_T(smem.L.data), lane, acc);
        fk::store_acc16_f16acc_1warp(
            acc, reinterpret_cast<__half*>(smem.pow.data), CHUNK, lane);
    }
    __syncwarp();
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        {
            auto acc = fk::make_acc16_f16acc_1warp(lane);
            fk::gemm16_ss_1warp_f16acc(
                view16(smem.INV.data), view16_T(smem.pow.data), lane, acc);
            fk::store_acc16_f16acc_1warp(
                acc, reinterpret_cast<__half*>(smem.L.data), CHUNK, lane);
        }
        __syncwarp();
        for (int idx = lane; idx < CHUNK * CHUNK; idx += 32)
            smem.INV.data[idx] = half_t(
                float(smem.INV.data[idx]) + float(smem.L.data[idx]));
        __syncwarp();
        if (it < 2) {
            auto acc = fk::make_acc16_f16acc_1warp(lane);
            fk::gemm16_ss_1warp_f16acc(
                view16(smem.pow.data), view16_T(smem.pow.data), lane, acc);
            fk::store_acc16_f16acc_1warp(
                acc, reinterpret_cast<__half*>(smem.L.data), CHUNK, lane);
            __syncwarp();
            for (int idx = lane; idx < CHUNK * CHUNK; idx += 32)
                smem.pow.data[idx] = smem.L.data[idx];
            __syncwarp();
        }
    }
    for (int idx = lane; idx < CHUNK * CHUNK; idx += 32)
        o_inv[idx] = BF16(float(smem.INV.data[idx]));
}

}  // namespace flashkda_cute
