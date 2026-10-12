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
// torch-free ABI between the v-last host entry and the device translation units.
//
// The device TUs (fused_sigmoid_gating_{bigbatch,smallbatch}_h*.cu) are compiled
// by hgcc against the PPU-native headers. They must not include torch, ATen or
// c10: those pull in the CUDA compatibility header tree, and a translation unit
// that mixes the compat tree with the native one loses its execution-space
// qualifiers -- native __device__ helpers start being diagnosed as __host__, and
// then so do __syncthreads/__shfl_xor_sync/__stcg in our own kernels.
//
// So the torch boundary stops in fused_sigmoid_gating_v_last.cpp (host, g++):
// it unwraps the tensors into the POD Args below and hands them over. The
// device side only ever sees raw pointers, scalars and a dtype tag.

// Included from both the device TUs (hgcc) and the host entry (g++), so it pulls
// in only <cstdint>, <stdexcept> (the dispatch macros throw) and hggc_runtime.h
// for hggcStream_t. Deliberately not hggc_fp16.h/hggc_bf16.h: the scalar types
// they declare are needed only where the dispatch macros expand, which is inside
// the device TUs, and there is no reason to make g++ parse headers full of
// device code.
#include <cstdint>
#include <stdexcept>

#include <hggc_runtime.h>

namespace pla {
namespace v_last {

// Mirrors the subset of torch dtypes the v-last kernels dispatch over. The host
// entry translates at::ScalarType into this tag so the kernels need no c10.
enum class DType : int {
    kFloat32 = 0,
    kFloat16,
    kBFloat16,
};

// One decode call. Pointers are untyped; the device side casts them according to
// scalar_dtype / alog_dtype. Null means "argument absent" for the three optional
// tensors (h0_source, h0_indices, cu_seqlens), matching the previous
// Tensor::defined() checks.
struct Args {
    const void* A_log;
    const void* a;
    const void* dt_bias;
    float softplus_beta;
    float softplus_threshold;
    const void* q;
    const void* k;
    const void* v;
    const void* b;
    void* o;
    float* h0_source;
    const int32_t* h0_indices;
    const int32_t* cu_seqlens;
    float scale;
    int T;
    int B;
    int HV;
    int Kdim;
    int Vdim;
    bool use_initial_state;
    bool use_qk_l2norm_in_kernel;
    bool is_varlen;
    DType alog_dtype;
    DType scalar_dtype;
    // Native handle obtained once on the host side (see src/ppu_torch_stream.h).
    hggcStream_t stream;
};

}  // namespace v_last
}  // namespace pla

// Signature of a per-H entry point. H is encoded in the function name.
#define PLA_V_LAST_ENTRY_PARAMS const pla::v_last::Args& args
#define PLA_V_LAST_ENTRY_ARGS args

// dtype dispatch, torch-free. The native scalar types are bit-compatible with
// the torch ones they replace (at::Half -> __half, at::BFloat16 ->
// __ppu_bfloat16) and expose operator float() on host and device.
#define PLA_V_LAST_DISPATCH_ALOG(TAG, ...)                              \
    do {                                                                \
        if ((TAG) == pla::v_last::DType::kFloat32) {                     \
            using alog_t = float;                                       \
            __VA_ARGS__();                                              \
        } else if ((TAG) == pla::v_last::DType::kFloat16) {              \
            using alog_t = __half;                                      \
            __VA_ARGS__();                                              \
        } else if ((TAG) == pla::v_last::DType::kBFloat16) {             \
            using alog_t = __ppu_bfloat16;                              \
            __VA_ARGS__();                                              \
        } else {                                                        \
            throw std::runtime_error("Unsupported type");               \
        }                                                               \
    } while (0)

#define PLA_V_LAST_DISPATCH_SCALAR(TAG, ...)                            \
    do {                                                                \
        if ((TAG) == pla::v_last::DType::kFloat16) {                     \
            using scalar_t = __half;                                    \
            __VA_ARGS__();                                              \
        } else if ((TAG) == pla::v_last::DType::kBFloat16) {             \
            using scalar_t = __ppu_bfloat16;                            \
            __VA_ARGS__();                                              \
        } else {                                                        \
            throw std::runtime_error("Unsupported type");               \
        }                                                               \
    } while (0)

// Multi-processor (CU) count of the current device, cached once by the host entry
// in fused_sigmoid_gating_v_last.cpp and read by the bigbatch kernel's
// choose_hv_per_cta(). Deliberately external linkage: writer and readers live in
// different translation units, and a per-TU copy would leave the readers at the
// initial value of 1 -- which silently disables the hv_per_cta optimisation and
// can select an hv_per_cta no KERNEL_LAUNCH() arm matches (i.e. no launch).
extern int g_v_last_cu_count;
