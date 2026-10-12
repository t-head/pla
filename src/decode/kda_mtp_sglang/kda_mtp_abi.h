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
// torch-free ABI between the KDA MTP SGLang host wrappers and device launchers.
//
// The .cu files contain the kernels and <<<>>> launch sites and are compiled by
// hgcc against the PPU-native headers. The .cpp host wrapper may keep using the
// CUDA-compatible torch API; only raw pointers, dimensions, strides and a native
// stream handle cross this boundary.

#include <cstdint>

#include <hggc_runtime.h>

namespace pla {
namespace kda_mtp {

enum class ConvDType : int {
  kBFloat16 = 0,
  kFloat32,
};

// Strides (in elements) describing the q/k/v conv-state and scratch-window
// layouts the verify kernel must address. Mirrors master's device-side struct
// but lives here so it can cross the torch-free ABI boundary by value.
struct ConvStrides {
  int64_t slot, channel, window;
};

struct WindowStrides {
  int64_t row, step, channel, window;
};

struct VerifyStrides {
  int64_t x_q, x_k, x_v, g, beta;
  ConvStrides cs_q, cs_k, cs_v;
  WindowStrides ic_q, ic_k, ic_v;
};

void launch_fold(
    float* h0, const void* rawv, const void* rawk,
    const float* gk, const float* beta,
    const int32_t* slot_idx, const int32_t* accept_lens,
    const int32_t* track_idx, const int32_t* track_step,
    int64_t h0_s0, int64_t h0_s1,
    int64_t rawv_s0, int64_t rawv_s1,
    int64_t rawk_s0, int64_t rawk_s1,
    int64_t gk_s0, int64_t gk_s1,
    int64_t beta_s0, int64_t beta_s1,
    int layers, int B, int num_k_heads, int HV, int V, int L,
    int null_block_id, bool has_track, hggcStream_t stream);

void launch_conv_scatter(
    ConvDType dtype, int km1,
    const void* src, void* dst,
    const int32_t* dst_idx, const int32_t* step_idx,
    int layers, int dim, int reqs,
    int64_t src_s0, int64_t src_s1, int64_t src_s2,
    int64_t src_s3, int64_t src_s4,
    int64_t dst_s0, int64_t dst_s1, int64_t dst_s2, int64_t dst_s3,
    int src_reqs, int src_steps, int dst_reqs,
    hggcStream_t stream);

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
    float scale, float lower_bound, VerifyStrides strides, hggcStream_t stream);

}  // namespace kda_mtp
}  // namespace pla
