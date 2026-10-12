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

// pla.decode.gdn.v_last -- v-last (V-major) GDN decode extension.
//
// Hosts the entry point that used to live in
// src/decode/k-last/fused_sigmoid_gating.cpp. It moved here because the kernels
// it launches (bigbatch / smallbatch) are now compiled into this separate
// extension, so the k-last extension can no longer reach them.
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hggc_runtime.h>
#include <mutex>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "common.cuh"
#include "../../ppu_torch_stream.h"
#include "v_last_common.cuh"
#include "v_last_dispatch.cuh"
#include "../../sanity_check.h"
#include "../../bypass_exec.h"
#include "../../op_forge_case.h"
#include "sanity_checks.h"

namespace py = pybind11;

// Environment-variable cache declared extern by common.cuh. Each extension is a
// separate shared object, so this one needs its own definitions (the k-last
// extension defines its own copy in fused_sigmoid_gating.cpp).
int g_sail_pla_cuda = 0;
int g_ppu_lib_perf_instrument = 0;
int g_ppu_lib_show_params = 0;
std::atomic<bool> g_env_initialized{false};

// Declared extern by v_last_common.cuh; see the comment there for why the
// bigbatch kernel must not hold a per-translation-unit copy.
int g_v_last_cu_count = 1;

#define CHECK_DTYPE(x, dataType) TORCH_CHECK(x.dtype() == dataType, #x "'s data type should be " #dataType)

// Cache the device CU count once; choose_hv_per_cta() in the bigbatch kernel
// keys its hv_per_cta heuristic off it.
static void init_cu_count(const at::Tensor& q) {
    static std::once_flag init_flag;
    std::call_once(init_flag, [&q]() {
        hggcDeviceProp deviceProp;
        if (hggcGetDeviceProperties(&deviceProp, q.device().index()) != hggcSuccess) {
            return;
        }
        g_v_last_cu_count = deviceProp.multiProcessorCount;
    });
}

static pla::v_last::Args make_args(PLA_V_LAST_HOST_PARAMS) {
    pla::v_last::Args args{};
    args.A_log = A_log.data_ptr();
    args.a = a.data_ptr();
    args.dt_bias = dt_bias.data_ptr();
    args.softplus_beta = softplus_beta;
    args.softplus_threshold = softplus_threshold;
    args.q = q.data_ptr();
    args.k = k.data_ptr();
    args.v = v.data_ptr();
    args.b = b.data_ptr();
    args.o = o.data_ptr();
    args.h0_source = pla::v_last::opt_ptr<float>(h0_source);
    args.h0_indices = pla::v_last::opt_ptr<int32_t>(h0_indices);
    args.cu_seqlens = pla::v_last::opt_ptr<int32_t>(cu_seqlens);
    args.scale = scale;
    args.T = T;
    args.B = B;
    args.HV = HV;
    args.Kdim = Kdim;
    args.Vdim = Vdim;
    args.use_initial_state = use_initial_state;
    args.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel;
    args.is_varlen = is_varlen;
    args.alog_dtype = pla::v_last::dtype_of(A_log);
    args.scalar_dtype = pla::v_last::dtype_of(q);
    args.stream = pla::current_stream();
    return args;
}

static void fused_sigmoid_gating_delta_rule_bigbatch(
    PLA_V_LAST_HOST_PARAMS, const int H
) {
    init_cu_count(q);
    const pla::v_last::Args args = make_args(PLA_V_LAST_HOST_ARGS);
    PLA_V_LAST_DISPATCH_H(bigbatch);
}

static void fused_sigmoid_gating_delta_rule_smallbatch(
    PLA_V_LAST_HOST_PARAMS, const int H
) {
    const pla::v_last::Args args = make_args(PLA_V_LAST_HOST_ARGS);
    PLA_V_LAST_DISPATCH_H(smallbatch);
}

at::Tensor fused_sigmoid_gating_delta_rule_forward(
    const at::Tensor& A_log,
    const at::Tensor& a,
    const at::Tensor& dt_bias,
    const float softplus_beta,
    const float softplus_threshold,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& b,
    at::Tensor& h0_source,
    at::Tensor& o,
    const at::Tensor& h0_indices,
    const at::Tensor& cu_seqlens,
    const float scale,
    const int32_t T,
    const int32_t B,
    const int32_t H,
    const int32_t HV,
    const int32_t K,
    const int32_t V,
    const bool use_initial_state,
    const bool use_qk_l2norm_in_kernel,
    const bool is_varlen
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_case(B, T, K, V, H, HV, get_dtype_string(q), false,
                              is_varlen ? &cu_seqlens : nullptr, -5.0f, is_varlen);
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("gdn_v_last", {
            {"A_log", &A_log}, {"a", &a}, {"dt_bias", &dt_bias},
            {"q", &q}, {"k", &k}, {"v", &v}, {"b", &b},
            {"h0_source", &h0_source}, {"o", &o},
            {"h0_indices", &h0_indices}, {"cu_seqlens", &cu_seqlens}},
            "scale=" + std::to_string(scale) +
            " softplus_beta=" + std::to_string(softplus_beta) +
            " softplus_threshold=" + std::to_string(softplus_threshold) +
            " use_initial_state=" + std::to_string(use_initial_state) +
            " use_qk_l2norm_in_kernel=" + std::to_string(use_qk_l2norm_in_kernel) +
            " is_varlen=" + std::to_string(is_varlen));
        return o;
    }
    sanity_v_last_inputs(q, k, v, a, b, A_log, dt_bias, o, h0_source, h0_indices,
                         cu_seqlens, B, T, H, HV, K, V, scale, softplus_beta,
                         softplus_threshold, use_initial_state, is_varlen);
    const c10::cuda::CUDAGuard device_guard(q.device());
    // Generate and push NVTX tag / print only when enabled
    if (g_ppu_lib_perf_instrument || g_ppu_lib_show_params) {
        const at::Tensor* dump_cu_seqlens =
            is_varlen && cu_seqlens.defined() ? &cu_seqlens : nullptr;
        std::string nvtx_tag = generate_nvtx_tag(
            B, T, K, V, H, HV, get_dtype_string(q), false,
            dump_cu_seqlens, -5.0f, is_varlen);
        push_nvtx_and_print(nvtx_tag, true);
    }

    if (h0_source.defined()) {
        CHECK_DTYPE(h0_source, torch::kFloat32);
    }

    if (h0_indices.defined()) {
        CHECK_DTYPE(h0_indices, torch::kInt32);
    }

    if (cu_seqlens.defined()) {
        CHECK_DTYPE(cu_seqlens, torch::kInt32);
    }

    TORCH_CHECK(K == 128 && V == 128,
                "pla.decode.gdn.v_last requires head_dim K == V == 128, got K=",
                K, ", V=", V);
    TORCH_CHECK(H > 0 && HV >= H && HV % H == 0,
                "pla.decode.gdn.v_last requires HV to be a positive multiple of H, got H=",
                H, ", HV=", HV);

    constexpr int BATCH_THRESHOLD = 17 * 4; // 17 is the batch threshold for TP4
    int option = g_sail_pla_cuda;

#define PLA_V_LAST_FORWARD_ARGS                                     \
    A_log, a, dt_bias, softplus_beta, softplus_threshold,           \
    q, k, v, b, o, h0_source, h0_indices, cu_seqlens,               \
    scale, T, B, HV, K, V,                                          \
    use_initial_state, use_qk_l2norm_in_kernel, is_varlen, H

    // enable this path by default w or w/o setting SAIL_PLA_CUDA
    if (option == 0 || option == 1) {
        if ((B * H) <= BATCH_THRESHOLD || HV == 1) {
            fused_sigmoid_gating_delta_rule_smallbatch(PLA_V_LAST_FORWARD_ARGS);
        } else {
            fused_sigmoid_gating_delta_rule_bigbatch(PLA_V_LAST_FORWARD_ARGS);
        }
    } else if (option == 2) {
        fused_sigmoid_gating_delta_rule_smallbatch(PLA_V_LAST_FORWARD_ARGS);
    } else if (option == 3) {
        fused_sigmoid_gating_delta_rule_bigbatch(PLA_V_LAST_FORWARD_ARGS);
    } else {
        TORCH_CHECK(false, "SAIL_PLA_CUDA must be one of 0, 1, 2, or 3, got ", option);
    }

#undef PLA_V_LAST_FORWARD_ARGS

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    (void)pla::bypass::enabled();
    (void)pla::instrument::show_params();
    init_env_variables();



    m.def("fused_sigmoid_gating_delta_rule_forward", &fused_sigmoid_gating_delta_rule_forward,
          "Fused Sigmoid Gating Delta Rule Forward, v-last layout (CUDA)");
}
