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

// Torch-facing wrappers for the KDA MTP fold and conv-scatter kernels.
// The .cu sibling is device-only and is reached through kda_mtp_abi.h.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "sanity_checks.h"
#include "../../bypass_exec.h"
#include "../../op_forge_case.h"
#include "../../ppu_torch_stream.h"
#include "kda_mtp_abi.h"

namespace {
void check_fold(const torch::Tensor& t, const char* nm, int dim) {
  TORCH_CHECK(t.is_cuda(), nm, " must be CUDA");
  TORCH_CHECK(t.dim() == dim, nm, " expected ", dim, " dims, got ", t.dim());
}

}  // namespace

void kda_fold_all_layers(torch::Tensor h0, torch::Tensor rawv,
                         torch::Tensor rawk, torch::Tensor gk,
                         torch::Tensor beta, torch::Tensor slot_idx,
                         torch::Tensor accept_lens,
                         std::optional<torch::Tensor> track_idx,
                         std::optional<torch::Tensor> track_step,
                         int64_t num_k_heads, int64_t null_block_id) {
  if (pla::bypass::enabled()) {
    if (pla::instrument::show_params()) pla::instrument::dump_metadata("kda_fold_all_layers", {
        {"h0", &h0}, {"rawv", &rawv}, {"rawk", &rawk}, {"gk", &gk},
        {"beta", &beta}, {"slot_idx", &slot_idx}, {"accept_lens", &accept_lens},
        {"track_idx", track_idx ? &*track_idx : nullptr},
        {"track_step", track_step ? &*track_step : nullptr}},
        "num_k_heads=" + std::to_string(num_k_heads) +
        " null_block_id=" + std::to_string(null_block_id));
    return;
  }
  sanity_fold_all_layers_inputs(h0, rawv, rawk, gk, beta, slot_idx, accept_lens,
                                track_idx, track_step, num_k_heads, null_block_id);
  check_fold(h0, "h0", 5);
  const c10::cuda::CUDAGuard device_guard(h0.device());
  check_fold(rawv, "rawv", 5);
  check_fold(rawk, "rawk", 5);
  check_fold(gk, "gk", 5);
  check_fold(beta, "beta", 4);
  TORCH_CHECK(h0.scalar_type() == torch::kFloat32, "h0 must be fp32");
  TORCH_CHECK(rawv.scalar_type() == torch::kBFloat16 &&
                  rawk.scalar_type() == torch::kBFloat16,
              "rawv/rawk must be bf16");
  TORCH_CHECK(gk.scalar_type() == torch::kFloat32 &&
                  beta.scalar_type() == torch::kFloat32,
              "gk/beta must be fp32");

  const int layers = h0.size(0), HV = h0.size(2), V = h0.size(3),
            K = h0.size(4);
  TORCH_CHECK(K == 128, "this kernel is specialised for K=128, got ", K);
  const int B = slot_idx.size(0), L = rawv.size(3);
  TORCH_CHECK(num_k_heads > 0 && HV % num_k_heads == 0 && V > 0 &&
              layers > 0 && HV > 0 && L >= 1 && L <= 32,
              "fold requires grouped heads, positive dimensions, and ring length 1..32");
  TORCH_CHECK(h0.storage_offset() % 4 == 0,
              "recurrent state must be 16-byte aligned for float4 loads/stores");
  TORCH_CHECK(track_idx.has_value() == track_step.has_value(),
              "tracking indices and steps must be supplied together");
  const int slots = h0.size(1);
  TORCH_CHECK(rawv.sizes() == torch::IntArrayRef({layers, slots, HV, L, V}) &&
              rawk.sizes() == torch::IntArrayRef({layers, slots, num_k_heads, L, K}) &&
              gk.sizes() == torch::IntArrayRef({layers, slots, HV, L, K}) &&
              beta.sizes() == torch::IntArrayRef({layers, slots, HV, L}),
              "fold state/ring shape mismatch");
  for (const auto& t : {h0, rawv, rawk, gk, beta})
    TORCH_CHECK(t.device() == h0.device() && t.is_contiguous(),
                "fold state/rings must be contiguous on the same device");
  auto check_index = [&](const torch::Tensor& t) {
    TORCH_CHECK(t.device() == h0.device() && t.dim() == 1 && t.numel() == B &&
                (t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64),
                "fold indices must be int32/int64 [B] on the state device");
  };
  check_index(slot_idx);
  check_index(accept_lens);
  if (track_idx.has_value()) {
    check_index(*track_idx);
    check_index(*track_step);
  }
  if (B == 0) return;
  const bool has_track = track_idx.has_value() && track_step.has_value();

  sanity_fold_all_layers_values(h0, rawv, rawk, beta, slot_idx, accept_lens,
                                track_idx, track_step, B, L, slots, null_block_id);

  auto si = slot_idx.to(torch::kInt32).contiguous();
  auto al = accept_lens.to(torch::kInt32).contiguous();
  auto ti = has_track ? track_idx->to(torch::kInt32).contiguous() : si;
  auto ts = has_track ? track_step->to(torch::kInt32).contiguous() : al;

  pla::kda_mtp::launch_fold(
      h0.data_ptr<float>(), rawv.data_ptr(), rawk.data_ptr(),
      gk.data_ptr<float>(), beta.data_ptr<float>(), si.data_ptr<int32_t>(),
      al.data_ptr<int32_t>(), ti.data_ptr<int32_t>(), ts.data_ptr<int32_t>(),
      h0.stride(0), h0.stride(1), rawv.stride(0), rawv.stride(1),
      rawk.stride(0), rawk.stride(1), gk.stride(0), gk.stride(1),
      beta.stride(0), beta.stride(1), layers, B, (int)num_k_heads, HV, V, L,
      (int)null_block_id, has_track, pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kda_conv_window_scatter(torch::Tensor dst, torch::Tensor src,
                             torch::Tensor dst_idx, torch::Tensor step_idx) {
  if (pla::bypass::enabled()) {
    if (pla::instrument::show_params()) pla::instrument::dump_metadata("kda_conv_window_scatter", {
        {"dst", &dst}, {"src", &src}, {"dst_idx", &dst_idx}, {"step_idx", &step_idx}});
    return;
  }
  const pla::sanity::Context sanity("kda_conv_window_scatter", dst);
  if (sanity) {
    for (const auto& item : std::initializer_list<std::pair<const char*, torch::Tensor>>{
             {"dst", dst}, {"src", src}, {"dst_idx", dst_idx}, {"step_idx", step_idx}})
      kda_sglang_tensor(sanity, item.second, item.first);
  }
  TORCH_CHECK(dst.dim() == 4 && src.dim() == 5, "ranks: dst 4, src 5");
  TORCH_CHECK(dst.is_cuda() && src.device() == dst.device() &&
              dst.scalar_type() == src.scalar_type(),
              "conv source and destination must have the same CUDA device and dtype");
  const c10::cuda::CUDAGuard device_guard(dst.device());
  TORCH_CHECK(src.size(0) == dst.size(0) && src.size(3) == dst.size(2) &&
              src.size(4) == dst.size(3), "conv source/destination shape mismatch");
  TORCH_CHECK(dst.is_contiguous() || dst.transpose(-1, -2).is_contiguous(),
              "conv destination must be contiguous in dim_width or width_dim layout");
  TORCH_CHECK(dst_idx.device() == dst.device() && step_idx.device() == dst.device() &&
              dst_idx.dim() == 1 && step_idx.dim() == 1 &&
              dst_idx.numel() == step_idx.numel(), "invalid conv scatter indices");
  for (const auto& t : {dst_idx, step_idx})
    TORCH_CHECK(t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64,
                "conv scatter indices must be int32/int64");
  const int layers = dst.size(0), dim = dst.size(2), km1 = dst.size(3);
  if (sanity) {
    sanity.check(dst_idx.numel() == 0 || (layers > 0 && dim > 0),
                 "nonempty scatter requires positive layer/channel dimensions");
    if (sanity.values_enabled()) {
      const auto destinations = kda_sglang_int32_values(sanity, dst_idx);
      const auto steps = kda_sglang_int32_values(sanity, step_idx);
      std::unordered_set<int64_t> writers;
      for (size_t n = 0; n < steps.size(); ++n) {
        // Negative steps and destinations are documented no-op masks.
        if (steps[n] < 0 || destinations[n] < 0) continue;
        sanity.check(destinations[n] < dst.size(1) &&
                         static_cast<int64_t>(n) < src.size(1) && steps[n] < src.size(2),
                     "active conv scatter destination/source step is out of bounds");
        sanity.check(writers.insert(destinations[n]).second,
                     "active requests must not scatter into the same conv slot");
        // The copy is byte-exact, so NaN/Inf in the source is permitted here.
      }
    }
  }
  auto di = dst_idx.to(torch::kInt32).contiguous();
  auto sti = step_idx.to(torch::kInt32).contiguous();
  const int reqs = sti.size(0);
  if (reqs == 0) return;
  TORCH_CHECK(km1 >= 1 && km1 <= 4,
              "conv scatter: unsupported conv window ", km1);
  pla::kda_mtp::ConvDType dtype;
  if (dst.scalar_type() == torch::kBFloat16) {
    dtype = pla::kda_mtp::ConvDType::kBFloat16;
  } else {
    TORCH_CHECK(dst.scalar_type() == torch::kFloat32,
                "conv scatter: unsupported dtype ", dst.scalar_type());
    dtype = pla::kda_mtp::ConvDType::kFloat32;
  }
  pla::kda_mtp::launch_conv_scatter(
      dtype, km1, src.data_ptr(), dst.data_ptr(),
      di.data_ptr<int32_t>(), sti.data_ptr<int32_t>(),
      layers, dim, reqs,
      src.stride(0), src.stride(1), src.stride(2), src.stride(3), src.stride(4),
      dst.stride(0), dst.stride(1), dst.stride(2), dst.stride(3),
      (int)src.size(1), (int)src.size(2), (int)dst.size(1),
      pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
