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

// Torch-facing wrapper for the fused KDA mega decode kernel.
//
// Split out of fused_kda_decode_mega.cu, which is a device translation unit
// (it holds the <<<>>> launches) compiled by hgcc against the PPU-native
// headers and therefore cannot see torch. This half is host-only, goes through
// g++, and reaches the kernel through launch_kda_decode_mega().

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../ppu_torch_stream.h"
#include "common.cuh"
#include "k_last_mega_abi.h"
#include "kda_decode_sanity.h"
#include "../../bypass_exec.h"
#include "../../op_forge_case.h"


// Reference implementation (vllm contract, bf16-only) defined in
// fused_kda_decode_kernel_ref.cu. It is no longer exposed to Python directly;
// it is reachable only through the unified entry below when the environment
// variable KDA_DECODE_IMPL selects the "ref" path.
void fused_kda_decode_ref_forward(
    at::Tensor const& x, at::Tensor const& weight,
    std::optional<at::Tensor> bias, at::Tensor& conv_state,
    at::Tensor const& raw_g, at::Tensor const& raw_beta, at::Tensor const& a_log,
    at::Tensor const& dt_bias, at::Tensor const& state_indices,
    at::Tensor& state, at::Tensor& out, std::optional<double> lower_bound,
    std::optional<at::Tensor> output_gate, std::optional<at::Tensor> norm_weight,
    double norm_eps);

// ----------------------------------------------------------------------------
// Unified KDA decode entry: SAME I/O contract as torch.ops._C.fused_kda_decode.
// Operates in place on conv_state / state / out. Defaults to the PPU fusion
// kernel; set KDA_DECODE_IMPL=ref (or =1) to route to the reference kernel for
// A/B comparison within a single process. Activations may be bfloat16 or
// float16 (chosen from x.scalar_type()); the ref path is bf16-only.
// ----------------------------------------------------------------------------
void fused_kda_decode_mega_forward(
    at::Tensor const& x, at::Tensor const& weight,
    std::optional<at::Tensor> bias, at::Tensor& conv_state,
    at::Tensor const& raw_g, at::Tensor const& raw_beta, at::Tensor const& a_log,
    at::Tensor const& dt_bias, at::Tensor const& state_indices,
    at::Tensor& state, at::Tensor& out, std::optional<double> lower_bound,
    std::optional<at::Tensor> output_gate, std::optional<at::Tensor> norm_weight,
    double norm_eps) {
  constexpr int kHeadDim = 128;
  constexpr int kConvWidth = 4;

  if (pla::bypass::enabled()) {
    const int batch = x.dim() >= 1 ? x.size(0) : 0;
    const int heads = x.dim() >= 2 ? x.size(1) / (3 * 128) : 0;
    if (pla::instrument::show_params()) pla::instrument::dump_case(batch, 1, 128, 128, heads, heads,
                          get_dtype_string(x), true, nullptr,
                          static_cast<float>(lower_bound.value_or(-5.0)));
    if (pla::instrument::show_params()) pla::instrument::dump_metadata("fused_kda_decode_mega", {
        {"x", &x}, {"weight", &weight}, {"bias", bias ? &*bias : nullptr},
        {"conv_state", &conv_state}, {"raw_g", &raw_g}, {"raw_beta", &raw_beta},
        {"a_log", &a_log}, {"dt_bias", &dt_bias}, {"state_indices", &state_indices},
        {"state", &state}, {"out", &out},
        {"output_gate", output_gate ? &*output_gate : nullptr},
        {"norm_weight", norm_weight ? &*norm_weight : nullptr}},
        "norm_eps=" + std::to_string(norm_eps));
    return;
  }

  // Impl selector, read fresh every call (NOT cached) so one process can A/B
  // compare ppu vs ref by toggling the environment variable between calls.
  if (const char* impl = std::getenv("KDA_DECODE_IMPL");
      impl != nullptr &&
      (std::strcmp(impl, "ref") == 0 || std::strcmp(impl, "1") == 0)) {
    fused_kda_decode_ref_forward(x, weight, bias, conv_state, raw_g, raw_beta,
                                 a_log, dt_bias, state_indices, state, out,
                                 lower_bound, output_gate, norm_weight,
                                 norm_eps);
    return;
  }

  // Activation dtype: bf16 or fp16. All activation tensors must share it.
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto const act_dtype = x.scalar_type();
  TORCH_CHECK(act_dtype == at::kBFloat16 || act_dtype == at::kHalf,
              "x must be a bfloat16 or float16 tensor");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat,
              "weight must be a CUDA float32 tensor");
  TORCH_CHECK(conv_state.is_cuda() && conv_state.scalar_type() == act_dtype,
              "conv_state must be a CUDA tensor with the same dtype as x");
  TORCH_CHECK(raw_g.is_cuda() && raw_g.scalar_type() == act_dtype,
              "raw_g must be a CUDA tensor with the same dtype as x");
  TORCH_CHECK(raw_beta.is_cuda() && raw_beta.scalar_type() == act_dtype,
              "raw_beta must be a CUDA tensor with the same dtype as x");
  TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == at::kFloat,
              "A_log must be a CUDA float32 tensor");
  TORCH_CHECK(dt_bias.is_cuda() && dt_bias.scalar_type() == at::kFloat,
              "dt_bias must be a CUDA float32 tensor");
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kFloat,
              "state must be a CUDA float32 tensor");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == act_dtype,
              "out must be a CUDA tensor with the same dtype as x");
  TORCH_CHECK(state_indices.is_cuda() && state_indices.scalar_type() == at::kInt,
              "state_indices must be a CUDA int32 tensor");

  TORCH_CHECK(x.dim() == 2, "x must have shape [B, 3 * H * 128]");
  int const batch_size = static_cast<int>(x.size(0));
  int64_t const qkv_width = x.size(1);
  TORCH_CHECK(qkv_width % (3 * kHeadDim) == 0,
              "x must have shape [B, 3 * H * 128]");
  int const num_heads = static_cast<int>(qkv_width / (3 * kHeadDim));
  TORCH_CHECK(
      num_heads == 4 || num_heads == 6 || num_heads == 8 || num_heads == 12 ||
          num_heads == 16 || num_heads == 24 || num_heads == 48 ||
          num_heads == 64 || num_heads == 96,
      "H must be 4, 6, 8, 12, 16, 24, 48, 64, or 96, got ", num_heads);
  TORCH_CHECK(batch_size > 0, "KDA decode fusion requires at least one row");
  int const dim = num_heads * kHeadDim;

  TORCH_CHECK(weight.dim() == 3 && weight.is_contiguous() &&
                  weight.size(0) == 3 && weight.size(1) == kConvWidth &&
                  weight.size(2) == dim,
              "weight must have shape [3, 4, H * 128]");
  TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == 3 * dim &&
                  conv_state.size(2) == kConvWidth - 1,
              "conv_state must have shape [slots, 3 * H * 128, 3]");
  TORCH_CHECK(raw_g.dim() == 4 && raw_g.size(0) == 1 &&
                  raw_g.size(1) == batch_size && raw_g.size(2) == num_heads &&
                  raw_g.size(3) == kHeadDim,
              "raw_g must have shape [1, B, H, 128]");
  TORCH_CHECK(raw_beta.dim() == 3 && raw_beta.size(0) == 1 &&
                  raw_beta.size(1) == batch_size &&
                  raw_beta.size(2) == num_heads,
              "raw_beta must have shape [1, B, H]");
  TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == num_heads,
              "A_log must be contiguous with H elements");
  TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == dim,
              "dt_bias must be contiguous with H * 128 elements");
  TORCH_CHECK(state_indices.is_contiguous() &&
                  state_indices.numel() == batch_size,
              "state_indices must be contiguous with B elements");
  TORCH_CHECK(state.dim() == 4 && state.size(1) == num_heads &&
                  state.size(2) == kHeadDim && state.size(3) == kHeadDim,
              "state must have shape [slots, H, 128, 128]");
  TORCH_CHECK(out.dim() == 4 && out.size(0) == 1 && out.size(1) == batch_size &&
                  out.size(2) == num_heads && out.size(3) == kHeadDim,
              "out must have shape [1, B, H, 128]");
  TORCH_CHECK(x.stride(1) == 1, "x must be contiguous in its channel dimension");
  TORCH_CHECK(conv_state.stride(0) >= 3 * dim * (kConvWidth - 1) &&
                  conv_state.stride(1) == 1 &&
                  conv_state.stride(2) == 3 * dim,
              "conv_state must use the SD cache layout");
  TORCH_CHECK(state.stride(0) >= (int64_t)num_heads * kHeadDim * kHeadDim &&
                  state.stride(1) == kHeadDim * kHeadDim &&
                  state.stride(2) == kHeadDim && state.stride(3) == 1,
              "state must have contiguous [H, 128, 128] slot contents");
  TORCH_CHECK(raw_g.is_contiguous(), "raw_g must be contiguous");
  TORCH_CHECK(raw_beta.stride(2) == 1,
              "raw_beta must be contiguous in its head dimension");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  bool const apply_onorm = output_gate.has_value();
  TORCH_CHECK(apply_onorm == norm_weight.has_value(),
              "output_gate and norm_weight must be provided together");
  void const* output_gate_ptr = nullptr;
  float const* norm_weight_ptr = nullptr;
  int64_t output_gate_row_stride = 0;
  if (apply_onorm) {
    TORCH_CHECK(output_gate->is_cuda() &&
                    output_gate->scalar_type() == act_dtype,
                "output_gate must be a CUDA tensor with the same dtype as x");
    bool const gate_is_3d =
        output_gate->dim() == 3 && output_gate->size(0) == batch_size &&
        output_gate->size(1) == num_heads && output_gate->size(2) == kHeadDim;
    bool const gate_is_4d =
        output_gate->dim() == 4 && output_gate->size(0) == 1 &&
        output_gate->size(1) == batch_size &&
        output_gate->size(2) == num_heads && output_gate->size(3) == kHeadDim;
    TORCH_CHECK(gate_is_3d || gate_is_4d,
                "output_gate must have shape [B, H, 128] or [1, B, H, 128]");
    int const row_dim = gate_is_3d ? 0 : 1;
    TORCH_CHECK(output_gate->stride(output_gate->dim() - 1) == 1,
                "output_gate must be contiguous in its last dimension");
    TORCH_CHECK(output_gate->stride(row_dim + 1) == kHeadDim,
                "output_gate must have contiguous head rows");
    TORCH_CHECK(norm_weight->is_cuda() &&
                    norm_weight->scalar_type() == at::kFloat,
                "norm_weight must be a CUDA float32 tensor");
    TORCH_CHECK(norm_weight->is_contiguous() && norm_weight->numel() == kHeadDim,
                "norm_weight must be contiguous with 128 elements");
    TORCH_CHECK(norm_eps >= 0.0, "norm_eps must be non-negative");
    output_gate_ptr = output_gate->data_ptr();
    norm_weight_ptr = static_cast<float const*>(norm_weight->data_ptr());
    output_gate_row_stride = output_gate->stride(row_dim);
  }

  float const* bias_q_ptr = nullptr;
  float const* bias_k_ptr = nullptr;
  float const* bias_v_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kFloat,
                "bias must be a CUDA float32 tensor");
    TORCH_CHECK(bias->is_contiguous() && bias->numel() == 3 * dim,
                "bias must be contiguous with 3 * H * 128 elements");
    auto const* bp = static_cast<float const*>(bias->data_ptr());
    bias_q_ptr = bp;
    bias_k_ptr = bp + dim;
    bias_v_ptr = bp + 2 * dim;
  }

  check_kda_decode_inputs(x, weight, bias, conv_state, raw_g, raw_beta, a_log,
                          dt_bias, state_indices, state, out, lower_bound,
                          output_gate, norm_weight, norm_eps);

  // Activation element size (bf16 and fp16 are both 2 bytes, but derive it from
  // the tensor so the byte offsets stay correct for either dtype).
  int64_t const elem_bytes = x.element_size();
  auto const* x_ptr = static_cast<char const*>(x.data_ptr());
  auto const* weight_ptr = static_cast<char const*>(weight.data_ptr());
  auto* conv_ptr = static_cast<char*>(conv_state.data_ptr());
  int64_t const segment_bytes = dim * elem_bytes;
  int64_t const weight_segment_bytes =
      dim * kConvWidth * static_cast<int64_t>(sizeof(float));
  int64_t const conv_segment_bytes =
      dim * conv_state.stride(1) * elem_bytes;

  auto const* x_q = x_ptr;
  auto const* x_k = x_ptr + segment_bytes;
  auto const* x_v = x_ptr + 2 * segment_bytes;
  auto const* w_q_t = reinterpret_cast<float const*>(weight_ptr);
  auto const* w_k_t = reinterpret_cast<float const*>(weight_ptr + weight_segment_bytes);
  auto const* w_v_t = reinterpret_cast<float const*>(weight_ptr + 2 * weight_segment_bytes);
  auto* cs_q = conv_ptr;
  auto* cs_k = conv_ptr + conv_segment_bytes;
  auto* cs_v = conv_ptr + 2 * conv_segment_bytes;

  float const lower_bound_value =
      lower_bound.has_value() ? static_cast<float>(*lower_bound) : 0.0f;
  bool const use_lower_bound = lower_bound.has_value();
  float const scale = 0.08838834764831845f;   // 1 / sqrt(128)

  hggcStream_t const stream = pla::current_stream();

  auto run = [&](pla::k_last::MegaDType dtype) {
    launch_kda_decode_mega(
        dtype, apply_onorm, use_lower_bound,
        static_cast<float const*>(a_log.data_ptr()), raw_g.data_ptr(),
        static_cast<float const*>(dt_bias.data_ptr()),
        x_q, x_k, x_v, raw_beta.data_ptr(),
        w_q_t, w_k_t, w_v_t, bias_q_ptr, bias_k_ptr, bias_v_ptr,
        cs_q, cs_k, cs_v,
        output_gate_ptr, norm_weight_ptr, static_cast<float>(norm_eps),
        static_cast<float*>(state.data_ptr()),
        static_cast<int const*>(state_indices.data_ptr()), out.data_ptr(),
        scale, lower_bound_value, batch_size, num_heads, num_heads,
        x.stride(0), raw_beta.stride(1), output_gate_row_stride,
        conv_state.stride(0), state.stride(0), stream);
  };

  if (act_dtype == at::kBFloat16) {
    run(pla::k_last::MegaDType::kBFloat16);
  } else {
    run(pla::k_last::MegaDType::kFloat16);
  }

  hggcError_t const error = hggcGetLastError();
  TORCH_CHECK(error == hggcSuccess,
              "fused_kda_decode_mega kernel launch failed: ",
              hggcGetErrorString(error));
}
