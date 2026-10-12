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

#include "../../sanity_check.h"

#include <cmath>
#include <limits>
#include <optional>
#include <unordered_set>

// Extra diagnostic checks shared by both implementations. The entry points
// retain their existing dtype/shape/layout checks before calling this helper.
inline void check_kda_decode_inputs(
    const at::Tensor& x, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias, const at::Tensor& conv_state,
    const at::Tensor& raw_g, const at::Tensor& raw_beta,
    const at::Tensor& a_log, const at::Tensor& dt_bias,
    const at::Tensor& state_indices, const at::Tensor& state,
    const at::Tensor& out, const std::optional<double>& lower_bound,
    const std::optional<at::Tensor>& output_gate,
    const std::optional<at::Tensor>& norm_weight, double norm_eps) {
  const pla::sanity::Context check("fused_kda_decode", x);
  if (!check) return;
  for (const auto& item : std::initializer_list<std::pair<const char*, at::Tensor>>{
           {"x", x}, {"weight", weight}, {"conv_state", conv_state},
           {"raw_g", raw_g}, {"raw_beta", raw_beta}, {"A_log", a_log},
           {"dt_bias", dt_bias}, {"state_indices", state_indices},
           {"state", state}, {"out", out}})
    check.tensor(item.second, item.first);
  if (bias) check.tensor(*bias, "bias");
  if (output_gate) check.tensor(*output_gate, "output_gate");
  if (norm_weight) check.tensor(*norm_weight, "norm_weight");
  check.check(x.size(0) <= std::numeric_limits<int>::max() &&
                  x.size(1) <= std::numeric_limits<int>::max(),
              "batch/channel dimensions must fit the kernel int ABI");
  check.check(state_indices.dim() == 1, "state_indices must be a vector");
  check.check(state.storage_offset() % 4 == 0 && state.stride(0) % 4 == 0,
              "state slots must be 16-byte aligned for float4 access");
  if (lower_bound)
    check.check(std::isfinite(*lower_bound) && *lower_bound < 0.0 &&
                    std::isfinite(static_cast<float>(*lower_bound)),
                "lower_bound must be a finite negative float32 value");
  if (output_gate)
    check.check(std::isfinite(norm_eps) && norm_eps >= 0.0 &&
                    std::isfinite(static_cast<float>(norm_eps)),
                "norm_eps must be a finite non-negative float32 value");
  if (!check.values_enabled()) return;
  std::unordered_set<int64_t> written_slots;
  const auto slots = check.integers(state_indices);
  bool has_active_input = false;
  for (size_t n = 0; n < slots.size(); ++n) {
    const auto slot = slots[n];
    // Negative slots are kernel no-ops; zero still needs backing storage.
    if (slot < 0) continue;
    check.check(slot < state.size(0) && slot < conv_state.size(0),
                "state index ", slot, " exceeds recurrent/conv pool capacity");
    // This ABI has no framework flag. Preserve vLLM's repeated null-block
    // zero convention; SGLang callers use -1 for padding and may use slot zero.
    check.check(slot == 0 || written_slots.insert(slot).second,
                "requests must not concurrently update the same state slot: ", slot);
    if (slot == 0) continue;
    has_active_input = true;
    check.finite(x.select(0, n), "selected x");
    check.finite(state.select(0, slot), "selected state");
    check.finite(conv_state.select(0, slot), "selected conv_state");
    if (output_gate)
      check.finite(output_gate->select(output_gate->dim() == 3 ? 0 : 1, n),
                   "selected output_gate");
  }
  if (has_active_input) {
    check.finite(weight, "weight");
    if (bias) check.finite(*bias, "bias");
    if (norm_weight) check.finite(*norm_weight, "norm_weight");
  }
  // Raw sigmoid/softplus gate inputs permit saturating infinities. Padding
  // activations and untouched pool rows are not required to be initialized.
}
