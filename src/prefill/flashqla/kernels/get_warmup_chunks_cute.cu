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
 * FlashQLA PPU1.5 CuTe rewrite — Kernel (2) get_warmup_chunks
 *
 * plan.md Task 2. Determines, per (batch, head), how many warmup chunks are
 * needed by accumulating chunk-boundary gate values from the END of the
 * sequence backwards; once the cumulative gate < threshold (-10) the earlier
 * chunks can be skipped (fallback_mask = 0).
 *
 * Algorithm authority: csrc_sm90/get_warmup_chunks.cu (tilelang dump) and the
 * wmma port csrc_ppu15/get_warmup_chunks_ppu15.cu. This is a scalar reduction
 * kernel (no MMA / no AIU pipeline); the CuTe layer here is a make_tensor view
 * over the fp32 cumsum gate g[1, T, H], keeping semantics bit-identical.
 *
 * Parametric on H (num v-heads); plan target H=16 (SM90/wmma hardcoded H=8).
 *
 * Grid: batch_size,  Block: 32 (one warp),  Shared memory: 0
 */
#include <hggc_runtime.h>
#include <cstdint>

#include "cute/tensor.hpp"

namespace flashqla_cute {

// One warp per sequence. First H lanes each own one head.
// g layout: [1, T, H] row-major -> logical (T, H) with stride (H, 1).
__global__ void get_warmup_chunks_cute_kernel(
    const int* __restrict__ cu_seqlens,        // [batch+1]
    signed char* __restrict__ fallback_mask,   // [batch * H] output
    const float* __restrict__ g,               // [1, T, H] chunk-local cumsum gate
    const signed char* __restrict__ ht_mask,   // [batch] int8
    int* __restrict__ num_warmup_chunks,       // [batch * H] output
    int batch_size,
    int num_tokens,
    int H
) {
    using namespace cute;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (bid >= batch_size) return;

    // CuTe view over the gate tensor: shape (T, H), stride (H, 1).
    Tensor gG = make_tensor(make_gmem_ptr(g),
                            make_layout(make_shape(num_tokens, H),
                                        make_stride(H, 1)));

    // External state provided for this sequence -> no warmup.
    if ((bool)ht_mask[bid]) {
        for (int h = tid; h < H; h += blockDim.x) {
            num_warmup_chunks[bid * H + h] = 0;
            fallback_mask[bid * H + h] = 0;
        }
        return;
    }

    const int seq_start = cu_seqlens[bid];
    const int seq_end   = cu_seqlens[bid + 1];
    const int num_iters = (seq_end - seq_start) >> 5;  // chunks = seqlen / 32

    // Loop over heads — supports H up to 128 (Qwen3.7-Max tp1).
    for (int head = tid; head < H; head += blockDim.x) {
        float g_cumsum   = 0.0f;
        int   n_fragment = num_iters;   // default: need all chunks
        signed char f_fragment = 1;     // default: fallback (use all)

        for (int i_s = 0; i_s < num_iters; ++i_s) {
            float g_val = 0.0f;
            int pos = seq_end - (i_s * 32);   // 1-based token position at chunk boundary
            if (pos >= 1 && pos <= num_tokens) {
                g_val = gG(pos - 1, head);
            }
            g_cumsum += g_val;
            if (g_cumsum < -10.0f && n_fragment == num_iters) {
                n_fragment = i_s + 1;
                f_fragment = 0;  // can skip earlier chunks
            }
        }

        num_warmup_chunks[bid * H + head] = n_fragment;
        fallback_mask[bid * H + head] = (signed char)((bool)f_fragment);
    }
}

}  // namespace flashqla_cute

// ============================================================
// Host launch
// ============================================================
void launch_get_warmup_chunks_cute(
    const float* g,
    const signed char* ht_mask,
    const int* cu_seqlens,
    int* num_warmup_chunks,
    signed char* fallback_mask,
    int batch_size,
    int num_tokens,
    int H,
    hggcStream_t stream
) {
    dim3 grid(batch_size);
    // Block must cover H heads; max H=128 (Qwen3.7-Max tp1).
    dim3 block(128);
    flashqla_cute::get_warmup_chunks_cute_kernel<<<grid, block, 0, stream>>>(
        cu_seqlens, fallback_mask, g, ht_mask, num_warmup_chunks,
        batch_size, num_tokens, H);
}
