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

// v-last input diagnostics: kept apart from the kernel body; the entry point calls them exactly once.
#include <torch/extension.h>

#include "../../sanity_check.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

inline void sanity_v_last_inputs(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
    const at::Tensor& dt_bias, const at::Tensor& o,
    const at::Tensor& h0_source, const at::Tensor& h0_indices,
    const at::Tensor& cu_seqlens,
    int B, int T, int H, int HV, int K, int V,
    float scale, float softplus_beta, float softplus_threshold,
    bool use_initial_state, bool is_varlen) {
    const pla::sanity::Context sanity("gdn_v_last", q);
    if (sanity) {
        sanity.check(B > 0 && T > 0 && H > 0 && HV >= H && HV % H == 0 &&
                         K == 128 && V == 128,
                     "positive dimensions, grouped heads and K=V=128 are required");
        sanity.check(use_initial_state,
                     "v-last kernels require use_initial_state=true");
        for (const auto& item : std::initializer_list<std::pair<const char*, at::Tensor>>{
                 {"q", q}, {"k", k}, {"v", v}, {"a", a}, {"b", b},
                 {"A_log", A_log}, {"dt_bias", dt_bias}, {"o", o},
                 {"h0_source", h0_source}, {"h0_indices", h0_indices}}) {
            sanity.tensor(item.second, item.first);
            sanity.check(item.second.is_contiguous(), item.first, " must be contiguous");
        }
        sanity.check(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                     "q must be fp16 or bf16");
        for (const auto& t : {k, v, a, b, dt_bias, o})
            sanity.check(t.scalar_type() == q.scalar_type(),
                         "q/k/v/a/b/dt_bias/o must share one activation dtype");
        sanity.check(A_log.scalar_type() == at::kFloat ||
                         A_log.scalar_type() == at::kHalf ||
                         A_log.scalar_type() == at::kBFloat16,
                     "A_log must be fp32/fp16/bf16");
        sanity.check(q.dim() == 4 && q.size(2) == H && q.size(3) == K &&
                         k.sizes() == q.sizes(),
                     "q/k must match [batch, tokens, H, K]");
        const int64_t tokens = q.numel() / (static_cast<int64_t>(H) * K);
        sanity.check(v.dim() == 4 && v.size(0) == q.size(0) &&
                         v.size(1) == q.size(1) && v.size(2) == HV && v.size(3) == V,
                     "v must match the q/k token axes and [HV, V]");
        sanity.check(a.numel() == tokens * HV && b.numel() == tokens * HV &&
                         dt_bias.numel() == HV && A_log.numel() == HV &&
                         o.numel() == v.numel(),
                     "gate/output element counts must match the token/head dimensions");
        sanity.check(h0_source.dim() == 4 && h0_source.size(1) == HV &&
                         h0_source.size(2) == K && h0_source.size(3) == V &&
                         h0_source.scalar_type() == at::kFloat &&
                         h0_source.storage_offset() % 4 == 0,
                     "h0_source must be 16-byte aligned fp32 [slots, HV, K, V]");
        sanity.check(h0_indices.dim() == 1 && h0_indices.numel() >= B &&
                         h0_indices.scalar_type() == at::kInt,
                     "h0_indices must contain B int32 indices");
        sanity.check(tokens * std::max(static_cast<int64_t>(H) * K,
                                      static_cast<int64_t>(HV) * V) <= std::numeric_limits<int>::max(),
                     "token/head offsets must fit the kernel int ABI");
        sanity.check(std::isfinite(scale) && std::isfinite(softplus_beta) &&
                         softplus_beta > 0.f && std::isfinite(softplus_threshold),
                     "scale/softplus parameters must be finite and beta positive");
        if (is_varlen) {
            sanity.tensor(cu_seqlens, "cu_seqlens");
            sanity.check(cu_seqlens.dim() == 1 && cu_seqlens.numel() == B + 1 &&
                             cu_seqlens.is_contiguous() && cu_seqlens.scalar_type() == at::kInt,
                         "cu_seqlens must be contiguous int32 [B+1]");
        } else {
            sanity.check(T == 1 && tokens == B,
                         "v-last fixed-length decode requires exactly one token per request");
        }
        if (sanity.values_enabled()) {
            std::vector<int64_t> cu;
            if (is_varlen) {
                cu = sanity.integers(cu_seqlens);
                sanity.check(cu.front() == 0 && cu.back() <= tokens,
                             "cu_seqlens must start at zero and stay inside token storage");
                for (int n = 0; n < B; ++n)
                    sanity.check(cu[n + 1] >= cu[n] && cu[n + 1] - cu[n] <= 1,
                                 "v-last supports zero/one token per request, row ", n);
            }
            const auto slots = sanity.integers(h0_indices);
            std::unordered_set<int64_t> writers;
            for (int n = 0; n < B; ++n) {
                if ((is_varlen && cu[n] == cu[n + 1]) || slots[n] < 0) continue;
                sanity.check(slots[n] < h0_source.size(0), "state slot out of range at row ", n);
                sanity.check(writers.insert(slots[n]).second,
                             "active requests must not update the same state slot");
                sanity.finite(h0_source.select(0, slots[n]), "selected h0_source");
                const auto token = is_varlen ? cu[n] : n;
                sanity.finite(q.flatten(0, 1).select(0, token), "selected q");
                sanity.finite(k.flatten(0, 1).select(0, token), "selected k");
                sanity.finite(v.flatten(0, 1).select(0, token), "selected v");
            }
            // Gates permit saturating infinities; inactive token/state rows
            // may be uninitialized and must not participate in finite scans.
        }
    }
}
