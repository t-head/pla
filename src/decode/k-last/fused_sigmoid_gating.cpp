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

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <variant>
#include "common.cuh"
#include "bypass_exec.h"
#include "op_forge_case.h"

namespace py = pybind11;

// Define global variables (shared across .cpp and .cu)
int g_sail_pla_cuda = 0;
int g_ppu_lib_perf_instrument = 0;
int g_ppu_lib_show_params = 0;
std::atomic<bool> g_env_initialized{false};

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DTYPE(x, dataType) TORCH_CHECK(x.dtype() == dataType, #x "'s data type should be " #dataType)

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last(
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    float softplus_beta,
    float softplus_threshold,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor b,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    std::optional<float> scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    bool disable_state_update,
    std::optional<torch::Tensor> intermediate_states_buffer,
    std::optional<torch::Tensor> intermediate_state_indices,
    std::optional<std::variant<torch::Tensor, int64_t>> cache_steps_or_num_accept,
    std::optional<torch::Tensor> retrieve_parent_token,
    std::optional<float> lower_bound,
    std::optional<bool> is_sglang
);

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last_packed(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    float softplus_beta, float softplus_threshold,
    std::optional<float> scale,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    torch::Tensor& out,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    std::optional<bool> is_sglang
);

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias,
    float softplus_beta, float softplus_threshold,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor b,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    std::optional<float> scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    bool disable_state_update,
    std::optional<torch::Tensor> intermediate_states_buffer,
    std::optional<torch::Tensor> intermediate_state_indices,
    std::optional<std::variant<torch::Tensor, int64_t>> cache_steps_or_num_accept,
    std::optional<torch::Tensor> retrieve_parent_token,
    int bv, int wn, int sn, int hv_per_cta,
    std::optional<bool> is_sglang,
    std::optional<float> lower_bound
);

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable(
    torch::Tensor mixed_qkv, torch::Tensor a, torch::Tensor b,
    torch::Tensor A_log, torch::Tensor dt_bias,
    float softplus_beta, float softplus_threshold,
    std::optional<float> scale,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    torch::Tensor& out,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    int bv, int wn, int sn, int hv_per_cta,
    std::optional<bool> is_sglang
);

// Kimi K3 KDA decode fused kernel exposing the exact vllm
// torch.ops._C.fused_kda_decode I/O contract. Operates in place on
// conv_state / state / out. Defaults to the PPU-native fusion kernel; setting
// the environment variable KDA_DECODE_IMPL=ref routes to the ported NVIDIA
// reference internally (the reference is not exposed to Python separately).
void fused_kda_decode_mega_forward(
    torch::Tensor const& x, torch::Tensor const& weight,
    std::optional<torch::Tensor> bias, torch::Tensor& conv_state,
    torch::Tensor const& raw_g, torch::Tensor const& raw_beta,
    torch::Tensor const& a_log, torch::Tensor const& dt_bias,
    torch::Tensor const& state_indices, torch::Tensor& state,
    torch::Tensor& out, std::optional<double> lower_bound,
    std::optional<torch::Tensor> output_gate,
    std::optional<torch::Tensor> norm_weight, double norm_eps);

// Recurrent KDA from raw gate / beta inputs, signature-compatible with vLLM's
// Triton fused_recurrent_kda plus a trailing is_sglang pad-slot switch.
// Implemented in fused_recurrent_kda.cpp on top of the k_last IS_KDA path.
std::tuple<torch::Tensor, torch::Tensor> fused_recurrent_kda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor raw_g, torch::Tensor raw_beta, torch::Tensor A_log,
    std::optional<torch::Tensor> dt_bias, std::optional<double> lower_bound,
    torch::Tensor initial_state, torch::Tensor cu_seqlens,
    torch::Tensor ssm_state_indices,
    std::optional<torch::Tensor> num_accepted_tokens,
    std::optional<torch::Tensor> out,
    std::optional<bool> fuse_gate, bool is_sglang);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    (void)pla::bypass::enabled();
    (void)pla::instrument::show_params();
    init_env_variables();



    m.def("fused_sigmoid_gating_delta_rule_forward_k_last", &fused_sigmoid_gating_delta_rule_forward_k_last,
          "k-last format forward",
          py::arg("A_log"),
          py::arg("a"),
          py::arg("dt_bias"),
          py::arg("softplus_beta"),
          py::arg("softplus_threshold"),
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("b"),
          py::arg("initial_state_source"),
          py::arg("initial_state_indices"),
          py::arg("scale"),
          py::arg("use_qk_l2norm_in_kernel"),
          py::arg("cu_seqlens"),
          py::arg("is_kda"),
          py::arg("disable_state_update"),
          py::arg("intermediate_states_buffer"),
          py::arg("intermediate_state_indices"),
          py::arg("cache_steps_or_num_accept"),
          py::arg("retrieve_parent_token"),
          py::arg("lower_bound") = py::none(),
          py::arg("is_sglang") = py::none());
    m.def("fused_sigmoid_gating_delta_rule_forward_k_last_packed", &fused_sigmoid_gating_delta_rule_forward_k_last_packed,
          "k-last packed format forward",
          py::arg("mixed_qkv"), py::arg("a"), py::arg("b"),
          py::arg("A_log"), py::arg("dt_bias"),
          py::arg("softplus_beta"), py::arg("softplus_threshold"),
          py::arg("scale"),
          py::arg("initial_state_source"), py::arg("initial_state_indices"),
          py::arg("out"),
          py::arg("use_qk_l2norm_in_kernel"),
          py::arg("cu_seqlens"),
          py::arg("is_kda"),
          py::arg("is_sglang") = py::none());
    m.def("fused_sigmoid_gating_delta_rule_forward_k_last_tunable", &fused_sigmoid_gating_delta_rule_forward_k_last_tunable,
          "k-last format forward with explicit template params for auto-tuning",
          py::arg("A_log"), py::arg("a"), py::arg("dt_bias"),
          py::arg("softplus_beta"), py::arg("softplus_threshold"),
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("b"),
          py::arg("initial_state_source"), py::arg("initial_state_indices"),
          py::arg("scale"),
          py::arg("use_qk_l2norm_in_kernel"),
          py::arg("cu_seqlens"),
          py::arg("is_kda"),
          py::arg("disable_state_update"),
          py::arg("intermediate_states_buffer"),
          py::arg("intermediate_state_indices"),
          py::arg("cache_steps_or_num_accept"),
          py::arg("retrieve_parent_token"),
          py::arg("bv"), py::arg("wn"), py::arg("sn"), py::arg("hv_per_cta"),
          py::arg("is_sglang") = py::none(),
          py::arg("lower_bound") = py::none());
    m.def("fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable", &fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable,
          "k-last packed format forward with explicit template params for auto-tuning",
          py::arg("mixed_qkv"), py::arg("a"), py::arg("b"),
          py::arg("A_log"), py::arg("dt_bias"),
          py::arg("softplus_beta"), py::arg("softplus_threshold"),
          py::arg("scale"),
          py::arg("initial_state_source"), py::arg("initial_state_indices"),
          py::arg("out"),
          py::arg("use_qk_l2norm_in_kernel"),
          py::arg("cu_seqlens"),
          py::arg("is_kda"),
          py::arg("bv"), py::arg("wn"), py::arg("sn"), py::arg("hv_per_cta"),
          py::arg("is_sglang") = py::none());
    m.def("fused_kda_decode_mega_forward", &fused_kda_decode_mega_forward,
          "Kimi K3 KDA decode fused kernel (vllm contract; PPU-native BF16/FP16 by "
          "default, KDA_DECODE_IMPL=ref switches to the reference internally)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = std::nullopt,
          py::arg("conv_state"), py::arg("raw_g"), py::arg("raw_beta"),
          py::arg("a_log"), py::arg("dt_bias"), py::arg("state_indices"),
          py::arg("state"), py::arg("out"),
          py::arg("lower_bound") = std::nullopt,
          py::arg("output_gate") = std::nullopt,
          py::arg("norm_weight") = std::nullopt, py::arg("norm_eps") = 1e-5);
    m.def("fused_recurrent_kda", &fused_recurrent_kda,
          "Recurrent KDA decode from raw gate/beta (vLLM fused_recurrent_kda "
          "contract; is_sglang selects the sglang pad-slot convention). "
          "Returns (out, initial_state); the state pool is updated in place.",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("raw_g"), py::arg("raw_beta"), py::arg("A_log"),
          py::arg("dt_bias"), py::arg("lower_bound"),
          py::arg("initial_state"), py::arg("cu_seqlens"),
          py::arg("ssm_state_indices"),
          py::arg("num_accepted_tokens") = std::nullopt,
          py::arg("out") = std::nullopt,
          py::arg("fuse_gate") = std::nullopt,
          py::arg("is_sglang") = false);
}
