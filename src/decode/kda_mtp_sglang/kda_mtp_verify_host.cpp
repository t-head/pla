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

// Torch-facing wrapper for the KDA MTP verify kernel.
// The .cu sibling is device-only and is reached through kda_mtp_abi.h.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <string>

#include "sanity_checks.h"
#include "../../bypass_exec.h"
#include "../../op_forge_case.h"
#include "../../ppu_torch_stream.h"
#include "kda_mtp_abi.h"

void kda_verify_dspark(torch::Tensor x_q, torch::Tensor x_k, torch::Tensor x_v,
                       torch::Tensor g_in, torch::Tensor beta_in,
                       torch::Tensor w_q, torch::Tensor w_k, torch::Tensor w_v,
                       torch::Tensor cs_q, torch::Tensor cs_k, torch::Tensor cs_v,
                       torch::Tensor A_log, torch::Tensor dt_bias,
                       torch::Tensor h0, torch::Tensor out, torch::Tensor ic_q,
                       torch::Tensor ic_k, torch::Tensor ic_v,
                       torch::Tensor ring_rawv, torch::Tensor ring_rawk,
                       torch::Tensor ring_g, torch::Tensor ring_beta,
                       torch::Tensor slot_idx, torch::Tensor scratch_idx,
                       torch::Tensor cu_seqlens, double scale,
                       double lower_bound) {
  if (pla::bypass::enabled()) {
    const int batch = cu_seqlens.numel() > 0 ? cu_seqlens.numel() - 1 : 0;
    const int tokens = x_q.dim() >= 2 ? x_q.size(1) : 0;
    const int heads = x_q.dim() >= 3 ? x_q.size(2) : 0;
    const int value_heads = x_v.dim() >= 3 ? x_v.size(2) : 0;
    const int key_dim = x_q.dim() >= 4 ? x_q.size(3) : 0;
    const int value_dim = x_v.dim() >= 4 ? x_v.size(3) : 0;
    if (pla::instrument::show_params()) pla::instrument::dump_case(batch, batch > 0 ? tokens / batch : tokens,
                          key_dim, value_dim, heads, value_heads,
                          get_dtype_string(x_q), true, &cu_seqlens,
                          static_cast<float>(lower_bound));
    if (pla::instrument::show_params()) pla::instrument::dump_metadata("kda_verify_dspark", {
        {"x_q", &x_q}, {"x_k", &x_k}, {"x_v", &x_v},
        {"g", &g_in}, {"beta", &beta_in},
        {"w_q", &w_q}, {"w_k", &w_k}, {"w_v", &w_v},
        {"cs_q", &cs_q}, {"cs_k", &cs_k}, {"cs_v", &cs_v},
        {"A_log", &A_log}, {"dt_bias", &dt_bias}, {"h0", &h0}, {"out", &out},
        {"ic_q", &ic_q}, {"ic_k", &ic_k}, {"ic_v", &ic_v},
        {"ring_rawv", &ring_rawv}, {"ring_rawk", &ring_rawk},
        {"ring_g", &ring_g}, {"ring_beta", &ring_beta},
        {"slot_idx", &slot_idx}, {"scratch_idx", &scratch_idx},
        {"cu_seqlens", &cu_seqlens}}, "scale=" + std::to_string(scale));
    return;
  }
  sanity_verify_dspark_inputs(x_q, x_k, x_v, g_in, beta_in, w_q, w_k, w_v,
                              cs_q, cs_k, cs_v, A_log, dt_bias, h0, out,
                              ic_q, ic_k, ic_v, ring_rawv, ring_rawk, ring_g,
                              ring_beta, slot_idx, scratch_idx, cu_seqlens,
                              scale, lower_bound);
  // Layout constants shared with the device TU.
  constexpr int DIM = 128;
  constexpr int CONV_W = 4;
  constexpr int CONV_WIN = CONV_W - 1;
  TORCH_CHECK(x_q.dim() == 4 && x_q.size(0) == 1, "x_q must be [1, T, H, 128]");
  TORCH_CHECK(x_q.is_cuda(), "verify requires CUDA tensors");
  const c10::cuda::CUDAGuard device_guard(x_q.device());
  const int T = x_q.size(1), H = x_q.size(2);
  TORCH_CHECK(x_q.size(3) == DIM, "specialised for head_dim 128");
  const int N = cu_seqlens.numel() - 1;
  TORCH_CHECK(N > 0 && T > 0 && H > 0 && T % N == 0,
              "dense 1 + num_spec tokens per request");
  const int T_LOOP = T / N;
  TORCH_CHECK(ring_rawv.dim() == 4, "rawv ring must be [slot, H, L, 128]");
  const int L = ring_rawv.size(2), slots = ring_rawv.size(0);
  TORCH_CHECK(L >= T_LOOP, "ring shorter than the draft window");
  TORCH_CHECK(h0.scalar_type() == torch::kFloat32, "recurrent state must be fp32");
  TORCH_CHECK(x_q.scalar_type() == torch::kBFloat16, "inputs must be bf16");
  TORCH_CHECK(h0.sizes() == torch::IntArrayRef({slots, H, DIM, DIM}),
              "state must be [slot, H, 128, 128]");
  for (const auto& t : {x_k, x_v, g_in, beta_in, w_q, w_k, w_v,
                        cs_q, cs_k, cs_v, A_log, dt_bias, h0, out, ic_q, ic_k,
                        ic_v, ring_rawv, ring_rawk, ring_g, ring_beta,
                        slot_idx, scratch_idx, cu_seqlens})
    TORCH_CHECK(t.device() == x_q.device(), "verify tensors must share one device");
  for (const auto& t : {ring_rawv, ring_rawk, ring_g})
    TORCH_CHECK(t.sizes() == torch::IntArrayRef({slots, H, L, DIM}),
                "invalid verify ring shape");
  TORCH_CHECK(ring_beta.sizes() == torch::IntArrayRef({slots, H, L}) &&
              ring_rawv.scalar_type() == torch::kBFloat16 &&
              ring_rawk.scalar_type() == torch::kBFloat16 &&
              ring_g.scalar_type() == torch::kFloat32 &&
              ring_beta.scalar_type() == torch::kFloat32, "invalid verify ring dtype/shape");
  for (const auto& t : {w_q, w_k, w_v})
    TORCH_CHECK(t.sizes() == torch::IntArrayRef({H * DIM, CONV_W}) &&
                t.is_contiguous() && t.scalar_type() == torch::kFloat32,
                "conv weights must be contiguous fp32 [H*128, 4]");
  TORCH_CHECK(A_log.numel() == H && dt_bias.numel() == H * DIM &&
              A_log.is_contiguous() && dt_bias.is_contiguous() &&
              A_log.scalar_type() == torch::kFloat32 &&
              dt_bias.scalar_type() == torch::kFloat32, "invalid gate weights");
  TORCH_CHECK(h0.storage_offset() % 4 == 0,
              "recurrent state must be 16-byte aligned for float4 loads");
  for (const auto& t : {slot_idx, scratch_idx, cu_seqlens})
    TORCH_CHECK(t.dim() == 1 &&
                (t.scalar_type() == torch::kInt32 || t.scalar_type() == torch::kInt64),
                "verify indices must be int32/int64 vectors");
  TORCH_CHECK(slot_idx.numel() == N && scratch_idx.numel() >= N,
              "verify index count mismatch");

  // Layout strides (in elements). The kernel addresses q/k/v, conv states and
  // scratch windows purely through these, so native sglang split/transpose
  // views are consumed without a contiguity copy.
  auto input_stride = [&](const torch::Tensor& x) -> int64_t {
    TORCH_CHECK(x.sizes() == x_q.sizes() && x.scalar_type() == torch::kBFloat16,
                "Q/K/V and g must be bf16 [1, T, H, 128]");
    TORCH_CHECK(x.stride(3) == 1 && x.stride(2) == DIM && x.stride(1) >= H * DIM,
                "Q/K/V and g must have contiguous head/channel dimensions");
    return x.stride(1);
  };
  auto conv_strides = [&](const torch::Tensor& x) -> pla::kda_mtp::ConvStrides {
    TORCH_CHECK(x.dim() == 3 && x.size(0) == slots && x.size(1) == H * DIM &&
                x.size(2) == CONV_WIN && x.scalar_type() == torch::kBFloat16,
                "conv states must be bf16 [slot, H*128, 3]");
    TORCH_CHECK(x.stride(0) > 0 && x.stride(1) > 0 && x.stride(2) > 0,
                "conv state strides must be positive");
    return {x.stride(0), x.stride(1), x.stride(2)};
  };
  auto window_strides = [&](const torch::Tensor& x) -> pla::kda_mtp::WindowStrides {
    TORCH_CHECK(x.dim() == 4 && x.size(1) >= T_LOOP && x.size(2) == H * DIM &&
                x.size(3) == CONV_WIN && x.scalar_type() == torch::kBFloat16,
                "conv windows must be bf16 [scratch, step, H*128, 3]");
    TORCH_CHECK(x.stride(0) > 0 && x.stride(1) > 0 && x.stride(2) > 0 &&
                x.stride(3) > 0,
                "conv window strides must be positive");
    return {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
  };
  TORCH_CHECK(beta_in.dim() == 3 && beta_in.size(0) == 1 &&
              beta_in.size(1) == T && beta_in.size(2) == H &&
              beta_in.scalar_type() == torch::kBFloat16 &&
              beta_in.stride(2) == 1 && beta_in.stride(1) >= H,
              "beta must be bf16 [1, T, H] with contiguous heads");
  TORCH_CHECK(out.sizes() == x_q.sizes() && out.is_contiguous() &&
              out.scalar_type() == torch::kBFloat16 && h0.is_contiguous() &&
              ring_rawv.is_contiguous() && ring_rawk.is_contiguous() &&
              ring_g.is_contiguous() && ring_beta.is_contiguous(),
              "output, recurrent state and rings must be contiguous");
  const pla::kda_mtp::VerifyStrides strides{
      input_stride(x_q), input_stride(x_k), input_stride(x_v),
      input_stride(g_in), beta_in.stride(1),
      conv_strides(cs_q), conv_strides(cs_k), conv_strides(cs_v),
      window_strides(ic_q), window_strides(ic_k), window_strides(ic_v)};

  sanity_verify_dspark_values(x_q, x_k, x_v, w_q, w_k, w_v, cs_q, cs_k, cs_v,
                              h0, ic_q, ic_k, ic_v, slot_idx, scratch_idx,
                              cu_seqlens, N, T, T_LOOP, slots);

  auto si = slot_idx.to(torch::kInt32).contiguous();
  auto sc = scratch_idx.to(torch::kInt32).contiguous();
  auto cu = cu_seqlens.to(torch::kInt32).contiguous();

  pla::kda_mtp::launch_verify(
      x_q.data_ptr(), x_k.data_ptr(), x_v.data_ptr(),
      g_in.data_ptr(), beta_in.data_ptr(),
      w_q.data_ptr<float>(), w_k.data_ptr<float>(), w_v.data_ptr<float>(),
      cs_q.data_ptr(), cs_k.data_ptr(), cs_v.data_ptr(),
      A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), h0.data_ptr<float>(),
      out.data_ptr(), ic_q.data_ptr(), ic_k.data_ptr(), ic_v.data_ptr(),
      ring_rawv.data_ptr(), ring_rawk.data_ptr(),
      ring_g.data_ptr<float>(), ring_beta.data_ptr<float>(),
      si.data_ptr<int32_t>(), sc.data_ptr<int32_t>(), cu.data_ptr<int32_t>(),
      H, N, T_LOOP, L, (float)scale, (float)lower_bound, strides,
      pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
