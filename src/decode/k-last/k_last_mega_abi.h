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
// torch-free ABI between the fused KDA mega decode kernel and its host wrapper.
//
// fused_kda_decode_mega.cu holds the <<<>>> launches, so hgcc compiles it against
// the PPU-native headers and it cannot see torch -- mixing the native header tree
// with the CUDA compatibility tree that torch drags in breaks the execution-space
// qualifiers. fused_kda_decode_mega_host.cpp is the torch side and goes through
// g++. The dtype decision therefore has to cross a boundary where neither side
// can name the other's scalar types: the host speaks at::Half / at::BFloat16, the
// device speaks __half / __ppu_bfloat16, so they meet over this enum.

#include <cstdint>

#include <hggc_runtime.h>

namespace pla {
namespace k_last {

// Activation dtype of the mega decode kernel. Mirrors the two at::ScalarType
// values the host wrapper accepts.
enum class MegaDType : int {
    kBFloat16 = 0,
    kFloat16,
};

}  // namespace k_last
}  // namespace pla

// Defined in fused_kda_decode_mega.cu (device TU); called from
// fused_kda_decode_mega_host.cpp (host TU).
void launch_kda_decode_mega(
    pla::k_last::MegaDType act_dtype,
    bool apply_onorm, bool use_lower_bound,
    const float* a_log, const void* g, const float* dt_bias,
    const void* x_q, const void* x_k, const void* x_v, const void* beta,
    const float* w_q_t, const float* w_k_t, const float* w_v_t,
    const float* bias_q, const float* bias_k, const float* bias_v,
    void* cs_q, void* cs_k, void* cs_v,
    const void* g_out, const float* o_norm_weight, float norm_eps,
    float* state, const int* state_indices, void* out,
    float scale, float lower_bound, int B, int H, int HV,
    int64_t x_row, int64_t beta_row, int64_t onorm_row,
    int64_t conv_slot, int64_t state_slot, hggcStream_t stream);
