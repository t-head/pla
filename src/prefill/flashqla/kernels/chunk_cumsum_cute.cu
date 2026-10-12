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
 * FlashQLA PPU1.5 CuTe rewrite — Kernel: chunk_local_cumsum
 *
 * Forward chunk-local cumulative sum of the log-gate `g` along the token axis,
 * independently per (batch, chunk, head). Replaces the host-side pure-Python
 * per-chunk loop in host/flashqla_cute_ops.py.
 *
 * Algorithm authority: flash_qla/ops/utils/cumsum.py (TileLang dump). Semantics:
 *     g_cum[b, c_start + t, h] = sum_{s=0}^{t} g[b, c_start + s, h]
 * where chunks are CS=32 tokens and the accumulator RESETS at every chunk
 * boundary and at sequence boundaries given by cu_seqlens. The partial last
 * chunk of a sequence (len < CS) is handled naturally by the [c_start,c_end)
 * bound. reverse=false (our forward path never uses the reverse variant).
 *
 * Packed cu_seqlens cover [0,T), and every token in each sequence belongs to
 * exactly one chunk, so the kernel fully initializes the output buffer.
 *
 * Grid: num_chunks,  Block: 128,  Shared: 0.
 * Each thread owns one head (strided by blockDim.x) and performs a serial
 * CS-step scan. The scan carries a token-axis dependency, so parallelism comes
 * from (chunk, head); this is a memory-light kernel, launch parallelism (one
 * block per chunk) is what fills the CUs.
 */
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <cstdint>

namespace flashqla_cute {

__device__ __forceinline__ float qk_to_float(__half x) {
    return __half2float(x);
}

__device__ __forceinline__ float qk_to_float(__ppu_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename Element>
__device__ __forceinline__ Element qk_from_float(float x);

template <>
__device__ __forceinline__ __half qk_from_float(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __ppu_bfloat16 qk_from_float(float x) {
    return __float2bfloat16_rn(x);
}

template <typename Element>
struct QkPairOps;

template <>
struct QkPairOps<__half> {
    using Pair = __half2;

    __device__ __forceinline__ static float2 to_float2(Pair value) {
        return __half22float2(value);
    }

    __device__ __forceinline__ static Pair from_float2(float2 value) {
        return __floats2half2_rn(value.x, value.y);
    }
};

template <>
struct QkPairOps<__ppu_bfloat16> {
    using Pair = __ppu_bfloat162;

    __device__ __forceinline__ static float2 to_float2(Pair value) {
        return __bfloat1622float2(value);
    }

    __device__ __forceinline__ static Pair from_float2(float2 value) {
        return __float22bfloat162_rn(value);
    }
};

__device__ __forceinline__ float qk_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return __shfl_sync(0xffffffff, value, 0);
}

template <int CS>
__global__ void chunk_local_cumsum_cute_kernel(
    const float* __restrict__ g,         // [1, T, H] row-major view (T, H)
    const int* __restrict__ cu_seqlens,  // [batch+1]
    const int* __restrict__ chunk_indices, // [num_chunks, 2]
    float* __restrict__ g_cum,           // [1, T, H] output
    int H
) {
    const int global_chunk = blockIdx.x;
    const int b         = chunk_indices[global_chunk * 2];
    const int chunk     = chunk_indices[global_chunk * 2 + 1];
    const int seq_start = cu_seqlens[b];
    const int seq_end   = cu_seqlens[b + 1];
    const int c_start   = seq_start + chunk * CS;
    if (c_start >= seq_end) return;
    int c_end = c_start + CS;
    if (c_end > seq_end) c_end = seq_end;

    // Each thread scans one head over the token range serially.
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.0f;
        for (int t = c_start; t < c_end; ++t) {
            acc += g[(long)t * H + h];
            g_cum[(long)t * H + h] = acc;
        }
    }
}

template <int CS, typename Element>
__global__ void chunk_local_cumsum_qk_norm_cute_kernel(
    const float* __restrict__ g,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_indices,
    float* __restrict__ g_cum,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    Element* __restrict__ q_norm,
    Element* __restrict__ k_norm,
    int num_chunks,
    int T,
    int H,
    int Hg,
    float eps
) {
    constexpr int D = 128;
    constexpr int QK_PER_BLOCK = 16;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // Eight warps normalize two token/head rows each. Reductions remain fp32,
    // matching the framework L2Norm contract before the result is rounded back
    // to the input dtype.
    int qk_idx = blockIdx.x * QK_PER_BLOCK + warp * 2;
    for (int item = 0; item < 2; ++item, ++qk_idx) {
        if (qk_idx < T * Hg) {
            const int64_t base = (int64_t)qk_idx * D;
            using PairOps = QkPairOps<Element>;
            using Pair = typename PairOps::Pair;
            const Pair* q_pairs = reinterpret_cast<const Pair*>(q + base);
            const Pair* k_pairs = reinterpret_cast<const Pair*>(k + base);
            Pair* q_norm_pairs = reinterpret_cast<Pair*>(q_norm + base);
            Pair* k_norm_pairs = reinterpret_cast<Pair*>(k_norm + base);
            float q_sq = 0.0f;
            float k_sq = 0.0f;
            float q_values[D / 32];
            float k_values[D / 32];
#pragma unroll
            for (int d_pair = lane; d_pair < D / 2; d_pair += 32) {
                const float2 qv = PairOps::to_float2(q_pairs[d_pair]);
                const float2 kv = PairOps::to_float2(k_pairs[d_pair]);
                const int value_idx = (d_pair / 32) * 2;
                q_values[value_idx] = qv.x;
                q_values[value_idx + 1] = qv.y;
                k_values[value_idx] = kv.x;
                k_values[value_idx + 1] = kv.y;
                q_sq += qv.x * qv.x + qv.y * qv.y;
                k_sq += kv.x * kv.x + kv.y * kv.y;
            }
            float q_inv = rsqrtf(qk_warp_sum(q_sq) + eps);
            float k_inv = rsqrtf(qk_warp_sum(k_sq) + eps);
#pragma unroll
            for (int d_pair = lane; d_pair < D / 2; d_pair += 32) {
                const int value_idx = (d_pair / 32) * 2;
                q_norm_pairs[d_pair] = PairOps::from_float2(make_float2(
                    q_values[value_idx] * q_inv,
                    q_values[value_idx + 1] * q_inv));
                k_norm_pairs[d_pair] = PairOps::from_float2(make_float2(
                    k_values[value_idx] * k_inv,
                    k_values[value_idx + 1] * k_inv));
            }
        }
    }

    // Cumsum is independent of Q/K normalization. The following KKT launch on
    // the same stream is the global publication boundary for all three outputs.
    if ((int)blockIdx.x < num_chunks) {
        const int b = chunk_indices[blockIdx.x * 2];
        const int chunk = chunk_indices[blockIdx.x * 2 + 1];
        const int seq_start = cu_seqlens[b];
        const int seq_end = cu_seqlens[b + 1];
        const int c_start = seq_start + chunk * CS;
        const int c_end = min(c_start + CS, seq_end);
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            float acc = 0.0f;
            for (int t = c_start; t < c_end; ++t) {
                acc += g[(int64_t)t * H + h];
                g_cum[(int64_t)t * H + h] = acc;
            }
        }
    }
}

template <int CS, typename Element>
__global__ void chunk_local_cumsum_qk_norm_gated_cute_kernel(
    const Element* __restrict__ a,
    const Element* __restrict__ b,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_indices,
    float* __restrict__ g_cum,
    float* __restrict__ beta,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    Element* __restrict__ q_norm,
    Element* __restrict__ k_norm,
    int num_chunks,
    int T,
    int H,
    int Hg,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_q,
    int64_t stride_k,
    float eps
) {
    constexpr int D = 128;
    constexpr int QK_PER_BLOCK = 16;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // Q/K normalization keeps the same one-warp-per-row ownership and exact
    // bf16/fp16 rounding point as the materialized-gate path.
    int qk_idx = blockIdx.x * QK_PER_BLOCK + warp * 2;
    for (int item = 0; item < 2; ++item, ++qk_idx) {
        if (qk_idx < T * Hg) {
            const int token = qk_idx / Hg;
            const int head = qk_idx - token * Hg;
            const int64_t q_base = (int64_t)token * stride_q + head * D;
            const int64_t k_base = (int64_t)token * stride_k + head * D;
            const int64_t out_base = (int64_t)qk_idx * D;
            float q_sq = 0.0f;
            float k_sq = 0.0f;
#pragma unroll
            for (int d = lane; d < D; d += 32) {
                float qv = qk_to_float(q[q_base + d]);
                float kv = qk_to_float(k[k_base + d]);
                q_sq += qv * qv;
                k_sq += kv * kv;
            }
            float q_inv = rsqrtf(qk_warp_sum(q_sq) + eps);
            float k_inv = rsqrtf(qk_warp_sum(k_sq) + eps);
#pragma unroll
            for (int d = lane; d < D; d += 32) {
                q_norm[out_base + d] = qk_from_float<Element>(
                    qk_to_float(q[q_base + d]) * q_inv);
                k_norm[out_base + d] = qk_from_float<Element>(
                    qk_to_float(k[k_base + d]) * k_inv);
            }
        }
    }

    // One warp owns one (chunk, value-head): its 32 lanes evaluate the 32 Gate
    // elements in parallel. Every lane then accumulates the lane values in
    // token order. This deliberately preserves the serial cumsum association
    // while avoiding 32 serial softplus/sigmoid evaluations on one thread.
    // beta remains materialized because KKT and fused recurrence both consume
    // it; raw g and the separate framework gating launch disappear.
    const int gate_group = (int)blockIdx.x * (blockDim.x / 32) + warp;
    if (gate_group < num_chunks * H) {
        const int global_chunk = gate_group / H;
        const int h = gate_group - global_chunk * H;
        const int seq = chunk_indices[global_chunk * 2];
        const int chunk = chunk_indices[global_chunk * 2 + 1];
        const int seq_start = cu_seqlens[seq];
        const int seq_end = cu_seqlens[seq + 1];
        const int c_start = seq_start + chunk * CS;
        const int c_end = min(c_start + CS, seq_end);
        const int valid_tokens = c_end - c_start;
        const int t = c_start + lane;
        float gate = 0.0f;
        float beta_value = 0.0f;
        if (lane < valid_tokens) {
            const float x =
                qk_to_float(a[(int64_t)t * stride_a + h]) + dt_bias[h];
            const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
            gate = -expf(A_log[h]) * softplus;
            const float beta_raw = qk_to_float(b[(int64_t)t * stride_b + h]);
            beta_value = 1.0f / (1.0f + expf(-beta_raw));
        }
        float prefix = 0.0f;
#pragma unroll
        for (int source_lane = 0; source_lane < CS; ++source_lane) {
            prefix += __shfl_sync(0xffffffff, gate, source_lane);
            if (lane == source_lane && lane < valid_tokens) {
                const int64_t output_offset = (int64_t)t * H + h;
                g_cum[output_offset] = prefix;
                beta[output_offset] = beta_value;
            }
        }
    }
}

}  // namespace flashqla_cute

// ============================================================
// Host launch
// ============================================================
void launch_chunk_local_cumsum_cute(
    const float* g,
    const int* cu_seqlens,
    const int* chunk_indices,
    float* g_cum,
    int num_chunks,
    int H,
    hggcStream_t stream
) {
    constexpr int CS = 32;
    if (num_chunks < 1) return;
    dim3 grid(num_chunks);
    dim3 block(128);
    flashqla_cute::chunk_local_cumsum_cute_kernel<CS><<<grid, block, 0, stream>>>(
        g, cu_seqlens, chunk_indices, g_cum, H);
}

void launch_chunk_local_cumsum_qk_norm_cute(
    const float* g,
    const int* cu_seqlens,
    const int* chunk_indices,
    float* g_cum,
    const void* q,
    const void* k,
    void* q_norm,
    void* k_norm,
    int num_chunks,
    int T,
    int H,
    int Hg,
    int dtype_tag,
    hggcStream_t stream
) {
    constexpr int CS = 32;
    constexpr int QK_PER_BLOCK = 16;
    int grid_size = (T * Hg + QK_PER_BLOCK - 1) / QK_PER_BLOCK;
    if (grid_size < num_chunks) grid_size = num_chunks;
    if (grid_size < 1) return;
    dim3 grid(grid_size);
    dim3 block(256);
    if (dtype_tag == 0) {
        flashqla_cute::chunk_local_cumsum_qk_norm_cute_kernel<CS, __half>
            <<<grid, block, 0, stream>>>(
                g, cu_seqlens, chunk_indices, g_cum,
                reinterpret_cast<const __half*>(q),
                reinterpret_cast<const __half*>(k),
                reinterpret_cast<__half*>(q_norm),
                reinterpret_cast<__half*>(k_norm),
                num_chunks, T, H, Hg, 1e-6f);
    } else {
        flashqla_cute::chunk_local_cumsum_qk_norm_cute_kernel<CS, __ppu_bfloat16>
            <<<grid, block, 0, stream>>>(
                g, cu_seqlens, chunk_indices, g_cum,
                reinterpret_cast<const __ppu_bfloat16*>(q),
                reinterpret_cast<const __ppu_bfloat16*>(k),
                reinterpret_cast<__ppu_bfloat16*>(q_norm),
                reinterpret_cast<__ppu_bfloat16*>(k_norm),
                num_chunks, T, H, Hg, 1e-6f);
    }
}

void launch_chunk_local_cumsum_qk_norm_gated_cute(
    const void* a,
    const void* b,
    const float* A_log,
    const float* dt_bias,
    const int* cu_seqlens,
    const int* chunk_indices,
    float* g_cum,
    float* beta,
    const void* q,
    const void* k,
    void* q_norm,
    void* k_norm,
    int num_chunks,
    int T,
    int H,
    int Hg,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_q,
    int64_t stride_k,
    int dtype_tag,
    hggcStream_t stream
) {
    constexpr int CS = 32;
    constexpr int QK_PER_BLOCK = 16;
    int grid_size = (T * Hg + QK_PER_BLOCK - 1) / QK_PER_BLOCK;
    constexpr int WARPS_PER_BLOCK = 8;
    const int gate_blocks =
        (num_chunks * H + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (grid_size < gate_blocks) grid_size = gate_blocks;
    if (grid_size < 1) return;
    dim3 grid(grid_size);
    dim3 block(256);
    if (dtype_tag == 0) {
        flashqla_cute::chunk_local_cumsum_qk_norm_gated_cute_kernel<CS, __half>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(a),
                reinterpret_cast<const __half*>(b), A_log, dt_bias,
                cu_seqlens, chunk_indices, g_cum, beta,
                reinterpret_cast<const __half*>(q),
                reinterpret_cast<const __half*>(k),
                reinterpret_cast<__half*>(q_norm),
                reinterpret_cast<__half*>(k_norm),
                num_chunks, T, H, Hg, stride_a, stride_b,
                stride_q, stride_k, 1e-6f);
    } else {
        flashqla_cute::chunk_local_cumsum_qk_norm_gated_cute_kernel<
            CS, __ppu_bfloat16><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __ppu_bfloat16*>(a),
                reinterpret_cast<const __ppu_bfloat16*>(b), A_log, dt_bias,
                cu_seqlens, chunk_indices, g_cum, beta,
                reinterpret_cast<const __ppu_bfloat16*>(q),
                reinterpret_cast<const __ppu_bfloat16*>(k),
                reinterpret_cast<__ppu_bfloat16*>(q_norm),
                reinterpret_cast<__ppu_bfloat16*>(k_norm),
                num_chunks, T, H, Hg, stride_a, stride_b,
                stride_q, stride_k, 1e-6f);
    }
}
