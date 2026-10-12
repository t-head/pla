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

#include <torch/extension.h>

#include "../../sanity_check.h"

#include <cmath>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

inline std::vector<int64_t> kda_sglang_int32_values(
    const pla::sanity::Context& check, const at::Tensor& tensor) {
  auto values = check.integers(tensor);
  for (const auto value : values)
    check.check(value >= std::numeric_limits<int32_t>::min() &&
                    value <= std::numeric_limits<int32_t>::max(),
                "index value cannot be represented by the int32 kernel ABI: ", value);
  return values;
}

inline void kda_sglang_tensor(
    const pla::sanity::Context& check, const at::Tensor& tensor, const char* name) {
  check.tensor(tensor, name);
  for (const auto dimension : tensor.sizes())
    check.check(dimension <= std::numeric_limits<int32_t>::max(),
                name, " dimension exceeds the int32 kernel ABI");
}

inline void sanity_verify_dspark_inputs(
    const torch::Tensor& x_q, const torch::Tensor& x_k, const torch::Tensor& x_v,
    const torch::Tensor& g_in, const torch::Tensor& beta_in,
    const torch::Tensor& w_q, const torch::Tensor& w_k, const torch::Tensor& w_v,
    const torch::Tensor& cs_q, const torch::Tensor& cs_k, const torch::Tensor& cs_v,
    const torch::Tensor& A_log, const torch::Tensor& dt_bias,
    const torch::Tensor& h0, const torch::Tensor& out,
    const torch::Tensor& ic_q, const torch::Tensor& ic_k, const torch::Tensor& ic_v,
    const torch::Tensor& ring_rawv, const torch::Tensor& ring_rawk,
    const torch::Tensor& ring_g, const torch::Tensor& ring_beta,
    const torch::Tensor& slot_idx, const torch::Tensor& scratch_idx,
    const torch::Tensor& cu_seqlens, double scale, double lower_bound) {
  const pla::sanity::Context sanity("kda_verify_dspark", x_q);
  if (sanity) {
    for (const auto& item : std::initializer_list<std::pair<const char*, torch::Tensor>>{
             {"x_q", x_q}, {"x_k", x_k}, {"x_v", x_v}, {"g", g_in},
             {"beta", beta_in}, {"w_q", w_q}, {"w_k", w_k}, {"w_v", w_v},
             {"cs_q", cs_q}, {"cs_k", cs_k}, {"cs_v", cs_v}, {"A_log", A_log},
             {"dt_bias", dt_bias}, {"h0", h0}, {"out", out},
             {"ic_q", ic_q}, {"ic_k", ic_k}, {"ic_v", ic_v},
             {"ring_rawv", ring_rawv}, {"ring_rawk", ring_rawk},
             {"ring_g", ring_g}, {"ring_beta", ring_beta},
             {"slot_idx", slot_idx}, {"scratch_idx", scratch_idx},
             {"cu_seqlens", cu_seqlens}})
      kda_sglang_tensor(sanity, item.second, item.first);
    sanity.check(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)),
                 "scale must be representable as a finite float32 value");
    sanity.check(std::isfinite(lower_bound) && lower_bound < 0.0 &&
                     std::isfinite(static_cast<float>(lower_bound)),
                 "lower_bound must be a finite negative float32 value");
  }
}

inline void sanity_verify_dspark_values(
    const torch::Tensor& x_q, const torch::Tensor& x_k, const torch::Tensor& x_v,
    const torch::Tensor& w_q, const torch::Tensor& w_k, const torch::Tensor& w_v,
    const torch::Tensor& cs_q, const torch::Tensor& cs_k, const torch::Tensor& cs_v,
    const torch::Tensor& h0,
    const torch::Tensor& ic_q, const torch::Tensor& ic_k, const torch::Tensor& ic_v,
    const torch::Tensor& slot_idx, const torch::Tensor& scratch_idx,
    const torch::Tensor& cu_seqlens,
    int64_t N, int64_t T, int64_t T_LOOP, int64_t slots) {
  const pla::sanity::Context sanity("kda_verify_dspark", x_q);
  if (sanity.values_enabled()) {
    const auto slot_values = kda_sglang_int32_values(sanity, slot_idx);
    const auto scratch_values = kda_sglang_int32_values(sanity, scratch_idx);
    const auto offsets = kda_sglang_int32_values(sanity, cu_seqlens);
    sanity.check(offsets.front() == 0 && offsets.back() == T,
                 "cu_seqlens must exactly cover the packed token storage");
    std::unordered_set<int64_t> ring_writers, window_writers;
    bool has_active_input = false;
    for (int n = 0; n < N; ++n) {
      // Output writes always cover T_LOOP positions, including padding rows;
      // ragged cu intervals would overlap the next request's output writes.
      sanity.check(offsets[n + 1] - offsets[n] == T_LOOP,
                   "verify requires a dense fixed token window for every request");
      const auto slot = slot_values[n];
      if (slot < 0) continue;
      has_active_input = true;
      sanity.check(slot < slots, "state/ring slot is out of bounds at request ", n);
      const auto scratch = scratch_values[n];
      sanity.check(scratch >= 0 && scratch < ic_q.size(0) &&
                       scratch < ic_k.size(0) && scratch < ic_v.size(0),
                   "scratch row is out of bounds at request ", n);
      sanity.check(ring_writers.insert(slot).second,
                   "active requests must not write the same ring slot");
      sanity.check(window_writers.insert(scratch).second,
                   "active requests must not write the same scratch row");
      sanity.finite(h0.select(0, slot), "selected h0");
      sanity.finite(cs_q.select(0, slot), "selected cs_q");
      sanity.finite(cs_k.select(0, slot), "selected cs_k");
      sanity.finite(cs_v.select(0, slot), "selected cs_v");
      sanity.finite(x_q.narrow(1, offsets[n], T_LOOP), "selected x_q");
      sanity.finite(x_k.narrow(1, offsets[n], T_LOOP), "selected x_k");
      sanity.finite(x_v.narrow(1, offsets[n], T_LOOP), "selected x_v");
    }
    if (has_active_input) {
      sanity.finite(w_q, "w_q");
      sanity.finite(w_k, "w_k");
      sanity.finite(w_v, "w_v");
    }
    // Raw gate/beta logits can contain valid saturating infinities.
  }
}

inline void sanity_fold_all_layers_inputs(
    const torch::Tensor& h0, const torch::Tensor& rawv, const torch::Tensor& rawk,
    const torch::Tensor& gk, const torch::Tensor& beta,
    const torch::Tensor& slot_idx, const torch::Tensor& accept_lens,
    const std::optional<torch::Tensor>& track_idx,
    const std::optional<torch::Tensor>& track_step,
    int64_t num_k_heads, int64_t null_block_id) {
  const pla::sanity::Context sanity("kda_fold_all_layers", h0);
  if (sanity) {
    for (const auto& item : std::initializer_list<std::pair<const char*, torch::Tensor>>{
             {"h0", h0}, {"rawv", rawv}, {"rawk", rawk}, {"gk", gk},
             {"beta", beta}, {"slot_idx", slot_idx}, {"accept_lens", accept_lens}})
      kda_sglang_tensor(sanity, item.second, item.first);
    if (track_idx) kda_sglang_tensor(sanity, *track_idx, "track_idx");
    if (track_step) kda_sglang_tensor(sanity, *track_step, "track_step");
    sanity.check(num_k_heads > 0 && num_k_heads <= std::numeric_limits<int32_t>::max(),
                 "num_k_heads must fit the positive int32 kernel ABI");
    sanity.check(null_block_id >= std::numeric_limits<int32_t>::min() &&
                     null_block_id <= std::numeric_limits<int32_t>::max(),
                 "null_block_id must fit the int32 kernel ABI");
  }
}

inline void sanity_fold_all_layers_values(
    const torch::Tensor& h0, const torch::Tensor& rawv, const torch::Tensor& rawk,
    const torch::Tensor& beta,
    const torch::Tensor& slot_idx, const torch::Tensor& accept_lens,
    const std::optional<torch::Tensor>& track_idx,
    const std::optional<torch::Tensor>& track_step,
    int64_t B, int64_t L, int64_t slots, int64_t null_block_id) {
  const pla::sanity::Context sanity("kda_fold_all_layers", h0);
  if (sanity.values_enabled()) {
    const bool has_track = track_idx.has_value() && track_step.has_value();
    const auto slots_in = kda_sglang_int32_values(sanity, slot_idx);
    const auto accepts = kda_sglang_int32_values(sanity, accept_lens);
    const auto tracks = has_track ? kda_sglang_int32_values(sanity, *track_idx)
                                 : std::vector<int64_t>{};
    const auto steps = has_track ? kda_sglang_int32_values(sanity, *track_step)
                                : std::vector<int64_t>{};
    std::unordered_set<int64_t> writers;
    for (int64_t n = 0; n < B; ++n) {
      if (slots_in[n] <= null_block_id) continue;
      sanity.check(accepts[n] >= 0 && accepts[n] <= L,
                   "accepted prefix length must be in [0, ring length], request ", n);
      if (accepts[n] == 0) continue;
      sanity.check(slots_in[n] >= 0 && slots_in[n] < slots,
                   "state/ring slot out of bounds at request ", n);
      sanity.check(writers.insert(slots_in[n]).second,
                   "active requests must not update the same state slot");
    }
    for (int64_t n = 0; n < B; ++n) {
      const auto slot = slots_in[n];
      if (slot <= null_block_id || accepts[n] == 0) continue;
      if (has_track && tracks[n] > null_block_id && steps[n] >= 0) {
        sanity.check(tracks[n] >= 0 && tracks[n] < slots && steps[n] < accepts[n],
                     "tracking slot/step is outside the state pool or accepted prefix");
        const bool same_final_state = tracks[n] == slot && steps[n] == accepts[n] - 1;
        sanity.check(same_final_state || writers.insert(tracks[n]).second,
                     "tracking writes overlap another state update");
      }
      sanity.finite(h0.select(1, slot), "selected h0");
      sanity.finite(rawv.select(1, slot).narrow(2, 0, accepts[n]), "accepted rawv");
      sanity.finite(rawk.select(1, slot).narrow(2, 0, accepts[n]), "accepted rawk");
      // gk is a pre-exp gate: -Inf validly means zero decay.
      sanity.finite(beta.select(1, slot).narrow(2, 0, accepts[n]), "accepted beta");
    }
  }
}
