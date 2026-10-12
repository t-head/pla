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
// Host-side plumbing for the v-last (V-major) decode kernels.
//
// The two v-last kernels (bigbatch / smallbatch) are compiled into their own
// extension, pla.decode.gdn.v_last._C, separately from the k-last kernels in
// pla.decode. Each kernel's template cascade is instantiated one translation
// unit per H value.
//
// This header is for the host entry only -- it speaks torch. The device TUs
// include v_last_args.h instead; see the comment there for why they cannot see
// the torch headers.
#include <torch/all.h>

#include "v_last_args.h"

namespace pla {
namespace v_last {

inline DType dtype_of(const at::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kFloat:              return DType::kFloat32;
        case at::kHalf:               return DType::kFloat16;
        case at::kBFloat16:           return DType::kBFloat16;
        default:
            TORCH_CHECK(false, "pla.decode.gdn.v_last: unsupported dtype ",
                        t.scalar_type());
    }
}

// Optional tensors keep their "absent -> nullptr" contract from the previous
// Tensor::defined() checks inside the kernels.
template <typename T>
inline T* opt_ptr(at::Tensor& t) {
    return t.defined() ? t.data_ptr<T>() : nullptr;
}

template <typename T>
inline const T* opt_ptr(const at::Tensor& t) {
    return t.defined() ? t.data_ptr<T>() : nullptr;
}

}  // namespace v_last
}  // namespace pla

// Host-side parameter list of the variant entry points in the .cpp. H is passed
// separately because it selects which per-H entry point runs.
#define PLA_V_LAST_HOST_PARAMS                                  \
    const at::Tensor& A_log, const at::Tensor& a,               \
    const at::Tensor& dt_bias,                                  \
    const float softplus_beta, const float softplus_threshold,  \
    const at::Tensor& q, const at::Tensor& k,                   \
    const at::Tensor& v, const at::Tensor& b,                   \
    at::Tensor& o, at::Tensor& h0_source,                       \
    const at::Tensor& h0_indices, const at::Tensor& cu_seqlens, \
    const float scale,                                          \
    const int T, const int B, const int HV,                      \
    const int Kdim, const int Vdim,                              \
    const bool use_initial_state,                                \
    const bool use_qk_l2norm_in_kernel,                          \
    const bool is_varlen

#define PLA_V_LAST_HOST_ARGS                                    \
    A_log, a, dt_bias, softplus_beta, softplus_threshold,       \
    q, k, v, b, o, h0_source, h0_indices, cu_seqlens,           \
    scale, T, B, HV, Kdim, Vdim,                                \
    use_initial_state, use_qk_l2norm_in_kernel, is_varlen
