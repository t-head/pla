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

// flashkdapro input diagnostic helpers: shape/dtype/metadata validation kept apart from the kernel launch body,
// shared by the prepare/fwd/cp entries. Only validation and offset derivation happen here; no kernel is launched.
#include <torch/extension.h>

#include "../../sanity_check.h"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace sp = pla::sanity::prefill;

inline std::vector<int64_t> sanity_prepare_inputs(
    const pla::sanity::Context& check,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& g,
    const torch::Tensor& beta, const torch::Tensor& A_log, const torch::Tensor& dt_bias,
    double scale, double lower_bound,
    const c10::optional<torch::Tensor>& cu_opt,
    const c10::optional<std::vector<int64_t>>& raw_opt = c10::nullopt) {
    check.tensor(q, "q");
    check.check(q.dim() == 3 && q.size(2) == 128,
                "q must be [T,H,128]");
    const int64_t T = q.size(0), H = q.size(1);
    sp::int_extent(check, T, "T");
    sp::int_extent(check, H, "H");
    for (const auto& named : std::initializer_list<std::pair<const char*, torch::Tensor>>{
             {"q", q}, {"k", k}, {"g", g}}) {
        check.tensor(named.second, named.first);
        check.check(named.second.sizes() == q.sizes() &&
                        named.second.scalar_type() == torch::kBFloat16 &&
                        named.second.is_contiguous(),
                    named.first, " must be contiguous bf16 [T,H,128] matching q");
    }
    check.tensor(beta, "beta");
    check.check(beta.dim() == 2 && beta.size(0) == T && beta.size(1) == H &&
                    beta.scalar_type() == torch::kBFloat16 &&
                    beta.stride(1) == 1 && beta.stride(0) >= H &&
                    beta.stride(0) <= std::numeric_limits<int>::max(),
                "beta must be bf16 [T,H] contiguous in H with an int32 token stride");
    check.tensor(A_log, "A_log");
    check.tensor(dt_bias, "dt_bias");
    check.check(A_log.scalar_type() == torch::kFloat32 && A_log.dim() == 1 &&
                    A_log.size(0) == H && A_log.is_contiguous(),
                "A_log must be contiguous fp32 [H]");
    check.check(dt_bias.scalar_type() == torch::kFloat32 && dt_bias.dim() == 2 &&
                    dt_bias.size(0) == H && dt_bias.size(1) == 128 && dt_bias.is_contiguous(),
                "dt_bias must be contiguous fp32 [H,128]");
    check.check(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)),
                "scale must be finite and representable as fp32");
    check.check(std::isfinite(lower_bound) && lower_bound >= -5.0 && lower_bound <= 0.0,
                "lower_bound must be in the supported [-5,0] interval");
    auto cu = cu_opt.has_value()
        ? sp::offsets(check, cu_opt.value(), T, "cu_seqlens", true)
        : std::vector<int64_t>{0, T};
    const int64_t N = cu_opt.has_value() ? cu_opt->numel() - 1 : 1;
    check.check((T + 15) / 16 + N <= std::numeric_limits<int>::max() &&
                    ((T + 15) / 16 + N) * H <= std::numeric_limits<int>::max(),
                "head/tile task count exceeds kernel int32 ABI");
    if (raw_opt.has_value()) {
        check.check(raw_opt->size() == size_t(N + 1), "raw_cu_seqlens must have N+1 entries");
        sp::offset_values(check, raw_opt.value(), T, "raw_cu_seqlens");
        if (check.values_enabled())
            check.check(cu == raw_opt.value(), "raw_cu_seqlens must equal device cu_seqlens");
    }
    return cu;
}

inline void sanity_v(const pla::sanity::Context& check, const torch::Tensor& q,
              const torch::Tensor& v) {
    check.tensor(v, "v");
    check.check(v.scalar_type() == torch::kBFloat16 && v.sizes() == q.sizes() &&
                    v.is_contiguous(), "v must be contiguous bf16 [T,H,128] matching q");
}
