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

// k-last input diagnostics and metadata derivation: kept apart from the kernel launch body; the plain and packed
// translation units share one contract. Only validation and shape derivation happen here; no kernel is launched.
#include <torch/extension.h>
#include <hggc_runtime_api.h>

#include "../../sanity_check.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

// sglang passes an int draft-token count here; vLLM instead passes its
// per-request num_accepted_tokens tensor. Both share the parameter slot so the
// Python signature stays stable across the two frameworks.
//
// Tensor MUST come first: a single-element tensor satisfies __index__, so with
// int64_t first pybind11 would silently convert a 1-request num_accepted_tokens
// into an int and the kernel would lose the per-request accepted counts.
using CacheStepsOrNumAccept = std::variant<torch::Tensor, int64_t>;

// Validate the packed width before deriving H, so integer truncation cannot hide an illegal input.
inline int check_packed_heads(
    const torch::Tensor& mixed_qkv, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& A_log, const torch::Tensor& dt_bias,
    const torch::Tensor& out, int HV, int K, int V, bool is_kda) {
    TORCH_CHECK(mixed_qkv.dim() == 2, "mixed_qkv must have shape [tokens, qkv_width]");
    const int64_t qk_width = mixed_qkv.size(1) - static_cast<int64_t>(HV) * V;
    TORCH_CHECK(qk_width > 0 && qk_width % (2 * K) == 0,
                "packed QKV width must equal 2*H*K + HV*V with H > 0");
    const int H = qk_width / (2 * K);
    TORCH_CHECK(HV >= H && HV % H == 0,
                "HV must be a positive multiple of H, got H=", H, ", HV=", HV);
    const auto gate_width = static_cast<int64_t>(HV) * (is_kda ? K : 1);
    TORCH_CHECK(a.dim() >= 2 && a.size(-1) == gate_width,
                "a head dimensions must match HV and the gate type");
    TORCH_CHECK(b.dim() >= 2 && b.size(-1) == HV,
                "b head dimension must match HV");
    TORCH_CHECK(A_log.numel() == HV && dt_bias.numel() == gate_width,
                "A_log/dt_bias head dimensions must match HV and the gate type");
    TORCH_CHECK(out.dim() >= 3 && out.size(-2) == HV && out.size(-1) == V &&
                out.numel() == mixed_qkv.size(0) * HV * V,
                "out must match the packed token count and [HV, V]");
    return H;
}

inline int get_packed_t_max(
    const torch::Tensor& k,
    int B,
    bool target_verify,
    int cache_steps,
    int logical_slot_columns
) {
    TORCH_CHECK(B > 0, "k-last requires at least one sequence");
    // A view's physical row pitch may contain padding or sibling metadata. Only
    // its logical columns describe vLLM's per-request token capacity.
    if (target_verify && logical_slot_columns > 0) return logical_slot_columns;
    if (k.size(0) == B) return k.size(1);
    if (!target_verify && k.size(0) == 1 && k.size(1) == B) return 1;

    // Ragged target verify packs requests into [1, sum(T_i), ...].
    // cache_steps is the runtime per-request token bound, which may be smaller
    // than the allocated intermediate-state stride under adaptive decoding.
    if (target_verify) {
        if (cache_steps > 0) {
            return cache_steps;
        }
        TORCH_CHECK(k.size(1) % B == 0,
                    "packed target-verify tokens must be divisible by batch size");
        return k.size(1) / B;
    }

    // Preserve the existing total-token upper bound for general packed varlen inputs.
    return k.size(1);
}

// Validate the address contract at the native boundary (also covers tunable).
// Token rows may have padding, but the kernel flattens every leading token axis
// into one affine token stride and assumes contiguous head/feature elements.
inline void sanity_token_rows(const pla::sanity::Context& c,
    const torch::Tensor& t, const char* name, int token_axis,
    int64_t width, int64_t tokens) {
    c.check(t.numel() == tokens * width, name, " token count must match q/k/v");
    int64_t expected = 1;
    for (int d = t.dim() - 1; d > token_axis; --d) {
        c.check(t.size(d) <= 1 || t.stride(d) == expected,
                name, " head/feature axes must be contiguous");
        expected *= t.size(d);
    }
    c.check(expected == width && t.stride(token_axis) >= width,
            name, " token stride must cover its complete head/feature row");
    expected = t.stride(token_axis) * t.size(token_axis);
    for (int d = token_axis - 1; d >= 0; --d) {
        c.check(t.size(d) <= 1 || t.stride(d) == expected,
                name, " leading token axes must flatten with a single stride");
        expected *= t.size(d);
    }
    c.check(t.stride(token_axis) <= std::numeric_limits<int>::max() &&
            (tokens == 0 || (tokens - 1) * t.stride(token_axis) + width <= std::numeric_limits<int>::max()),
            name, " addressing exceeds the kernel's int32 offset domain");
}

inline void sanity_k_last_inputs(
    const torch::Tensor& A_log, const torch::Tensor& a, const torch::Tensor& dt_bias,
    float softplus_beta, float softplus_threshold,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& state,
    const std::optional<torch::Tensor>& indices, std::optional<float> scale,
    const std::optional<torch::Tensor>& cu, bool is_kda, bool disable_state_update,
    const std::optional<torch::Tensor>& cache,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<CacheStepsOrNumAccept>& cache_steps_or_accept,
    const std::optional<torch::Tensor>& parent,
    std::optional<float> lower_bound, bool is_sglang) {
    const pla::sanity::Context c("k_last", q);
    if (!c) return;
    int current_device = -1;
    c.check(hggcGetDevice(&current_device) == hggcSuccess && current_device == q.get_device(),
            "input device must be the current CUDA/PPU device for this native launcher");
    for (const auto& item : {std::make_pair(&q, "q"), {&k, "k"}, {&v, "v"},
                            {&a, "a"}, {&b, "b"}, {&A_log, "A_log"}, {&dt_bias, "dt_bias"}})
        c.tensor(*item.first, item.second);
    c.check(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
            "q/k/v must have shape [B,T,H,D]");
    c.check(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
            "q must have float16 or bfloat16 dtype");
    for (const auto* t : {&k, &v, &a, &b})
        c.check(t->scalar_type() == q.scalar_type(), "q/k/v/a/b dtypes must match");
    c.check(A_log.scalar_type() == at::kFloat || A_log.scalar_type() == at::kHalf ||
            A_log.scalar_type() == at::kBFloat16, "A_log dtype is unsupported");
    c.check(q.sizes() == k.sizes() && v.size(0) == q.size(0) && v.size(1) == q.size(1),
            "q/k/v token shapes must match");
    const int64_t tokens = q.size(0) * q.size(1);
    const int64_t H = q.size(2), HV = v.size(2), K = q.size(3), V = v.size(3);
    c.check(H > 0 && HV >= H && HV % H == 0 && K == 128 && V == 128,
            "requires K=V=128 and HV a positive multiple of H");
    c.check(a.dim() >= 2 && b.dim() >= 2, "a/b require a token and gate dimension");
    sanity_token_rows(c, q, "q", 1, H*K, tokens);
    sanity_token_rows(c, k, "k", 1, H*K, tokens);
    sanity_token_rows(c, v, "v", 1, HV*V, tokens);
    for (const auto* t : {&q, &k, &v})
        c.check(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0 &&
                (t->stride(1)*t->element_size()) % 16 == 0,
                "q/k/v bases and token strides require 16-byte alignment");
    const int a_axis = a.dim() == 4 ? 1 :
        (is_kda && a.dim() == 3 && a.size(1) == HV && a.size(2) == K ? 0 : a.dim()-2);
    sanity_token_rows(c, a, "a", a_axis, HV*(is_kda ? K : 1), tokens);
    sanity_token_rows(c, b, "b", b.dim()-2, HV, tokens);
    if (is_kda)
        c.check(reinterpret_cast<uintptr_t>(a.data_ptr()) % 8 == 0 &&
                (a.stride(a_axis)*a.element_size()) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(dt_bias.data_ptr()) % 8 == 0,
                "KDA a/dt_bias vector loads require 8-byte alignment");
    c.check(A_log.numel() == HV && A_log.is_contiguous() &&
            dt_bias.numel() == HV*(is_kda ? K : 1) && dt_bias.is_contiguous(),
            "A_log/dt_bias must be contiguous and match gate heads/features");
    c.check(std::isfinite(softplus_beta) && softplus_beta > 0 &&
            std::isfinite(softplus_threshold), "softplus beta must be positive and parameters finite");
    c.check(!scale || std::isfinite(*scale), "scale must be finite");
    c.check(!lower_bound || (is_kda && std::isfinite(*lower_bound) && *lower_bound < 0),
            "lower_bound must be finite and negative, and requires KDA");
    int64_t B = k.size(0);
    if (cu) {
        c.tensor(*cu, "cu_seqlens");
        c.check(cu->dim() == 1 && cu->numel() >= 2 && cu->is_contiguous() &&
                cu->scalar_type() == at::kInt, "cu_seqlens must be contiguous int32 [B+1]");
        B = cu->numel()-1;
    }
    c.check(B > 0 && B*H <= std::numeric_limits<int>::max(), "invalid batch/grid size");
    c.check(state.has_value() && indices.has_value(),
            "this launcher requires initial_state_source and initial_state_indices");
    c.tensor(*state, "initial_state_source");
    c.tensor(*indices, "initial_state_indices");
    c.check(state->scalar_type() == at::kFloat || state->scalar_type() == at::kBFloat16,
            "state dtype must be float32 or bfloat16");
    const int64_t state_width = HV*V*K;
    c.check(state->dim() >= 2 && state->size(0) > 0 &&
            state->numel()/state->size(0) >= state_width && state->stride(0) >= state_width,
            "state rows must contain a complete contiguous [HV,V,K] block");
    int64_t dense_stride = 1;
    for (int d = state->dim()-1; d > 0; --d) {
        c.check(state->size(d) <= 1 || state->stride(d) == dense_stride,
                "state inner slot layout must be contiguous");
        dense_stride *= state->size(d);
    }
    // The regular kernel transfers four state elements per lane: float4 for
    // FP32, uint2/cp.async 8B for BF16. The packed kernel has its own 16B ABI.
    const auto state_vector_alignment = 4 * state->element_size();
    c.check(reinterpret_cast<uintptr_t>(state->data_ptr()) % state_vector_alignment == 0 &&
            (state->stride(0) * state->element_size()) % state_vector_alignment == 0,
            "state base and slot pitch require ", state_vector_alignment, "-byte vector alignment");
    const bool table = indices->dim() == 2;
    c.check(indices->scalar_type() == at::kInt && (indices->dim() == 1 || table) &&
            indices->size(0) == B && indices->stride(-1) == 1 &&
            indices->stride(0) <= std::numeric_limits<int>::max(),
            "state indices must be int32 [B] or [B,S] with contiguous columns");
    c.check(!table || (!is_sglang && indices->size(1) > 0 && indices->stride(0) >= indices->size(1)),
            "2D slot tables require is_sglang=False and nonempty logical columns");
    int cache_steps = 0;
    const torch::Tensor* accept = nullptr;
    if (cache_steps_or_accept) {
        if (std::holds_alternative<int64_t>(*cache_steps_or_accept)) {
            const int64_t value = std::get<int64_t>(*cache_steps_or_accept);
            c.check(!table && value > 0 && value <= std::numeric_limits<int>::max(),
                    "cache_steps must be positive int32 and cannot replace vLLM accepted counts");
            cache_steps = static_cast<int>(value);
        } else {
            accept = &std::get<torch::Tensor>(*cache_steps_or_accept);
            c.tensor(*accept, "num_accepted_tokens");
            c.check(table && accept->scalar_type() == at::kInt && accept->dim() == 1 &&
                    accept->numel() == B && accept->is_contiguous(),
                    "num_accepted_tokens requires a 2D slot table and contiguous int32 [B]");
        }
    }
    const bool target_verify = cache || cache_indices || parent || cache_steps || table;
    const int capacity = cu ? get_packed_t_max(k, B, target_verify, cache_steps,
        table ? indices->size(1) : 0) : k.size(1);
    c.check(capacity >= 0 && (!table || capacity <= indices->size(1)),
            "token capacity exceeds logical slot columns");
    const bool mtp = capacity > 1 || disable_state_update || target_verify;
    c.check(dt_bias.scalar_type() == (is_kda && mtp ? at::kFloat : q.scalar_type()),
            "dt_bias dtype does not match the selected MTP/decode path");
    if (is_kda && mtp)
        c.check(reinterpret_cast<uintptr_t>(dt_bias.data_ptr()) % 16 == 0, "MTP KDA dt_bias requires 16-byte alignment");
    c.check(!parent || (cache && cache_indices && is_sglang), "parent tree requires the SGLang intermediate cache");
    // MTP can load the cache index even on vLLM paths, so supplied metadata
    // must have the native int32 [B] ABI. Cache payload is only consumed by
    // SGLang with indices; absent indices leave cache_idx=-1 in the kernel.
    if (cache_indices) {
        c.tensor(*cache_indices, "intermediate_state_indices");
        c.check(cache_indices->scalar_type() == at::kInt && cache_indices->dim() == 1 &&
                cache_indices->numel() == B && cache_indices->is_contiguous(),
                "intermediate_state_indices must be contiguous int32 [B]");
    }
    const bool cache_payload_used = cache && cache_indices && is_sglang;
    int64_t cache_requests = 0;
    int cache_pitch = cache_steps;
    if (cache) {
        c.tensor(*cache, "intermediate_states_buffer");
        c.check(cache->scalar_type() == state->scalar_type(),
                "initial_state_source and intermediate_states_buffer must have the same dtype");
    }
    if (cache_payload_used) {
        c.check(cache->dim() >= 4 && cache->is_contiguous(),
                "intermediate cache must have a contiguous state layout");
        c.check(reinterpret_cast<uintptr_t>(cache->data_ptr()) % state_vector_alignment == 0,
                "intermediate cache base requires ", state_vector_alignment, "-byte vector alignment");
        if (cache->dim() >= 5) cache_pitch = cache->size(1);
        c.check(cache_pitch >= capacity && cache_pitch > 0 &&
                cache->numel() % (cache_pitch*state_width) == 0,
                "intermediate cache allocated step capacity must cover all tokens");
        cache_requests = cache->numel() / (cache_pitch*state_width);
    }
    if (parent) {
        c.tensor(*parent, "retrieve_parent_token");
        c.check((parent->scalar_type() == at::kInt || parent->scalar_type() == at::kLong) &&
                parent->dim() == 2 && parent->size(0) == B && parent->size(1) >= capacity &&
                parent->stride(0) <= std::numeric_limits<int>::max() && parent->stride(1) <= std::numeric_limits<int>::max(),
                "parent table must be int32/int64 [B, >=token_capacity] with int32 strides");
    }
    if (!c.values_enabled()) return;  // No D2H, scalar reads or reductions during capture.
    std::vector<int64_t> lengths(B, k.size(1));
    std::vector<int64_t> starts(B);
    for (int64_t n=0; n<B; ++n) starts[n] = n*k.size(1);
    if (cu) {
        const auto offsets = c.integers(*cu);
        c.check(offsets[0] == 0, "cu_seqlens must start at zero");
        for (int64_t n=0; n<B; ++n) {
            starts[n] = offsets[n];
            lengths[n] = offsets[n+1]-offsets[n];
            c.check(offsets[n] >= 0 && offsets[n+1] <= tokens && lengths[n] >= 0 && lengths[n] <= capacity,
                    "cu_seqlens must be monotone within token storage; sequence ", n,
                    " length ", lengths[n], " exceeds capacity ", capacity);
        }
    }
    const auto slots = c.integers(*indices);
    const auto accepted = accept ? c.integers(*accept) : std::vector<int64_t>();
    const auto cache_slots = cache_payload_used ? c.integers(*cache_indices) : std::vector<int64_t>();
    const auto parents = parent ? c.integers(*parent) : std::vector<int64_t>();
    std::unordered_map<int64_t, int64_t> state_owner;
    std::unordered_set<int64_t> cache_owner;
    const int64_t columns = table ? indices->size(1) : 1;
    for (int64_t n=0; n<B; ++n) {
        if (lengths[n] == 0) continue;  // No accepted/slot/parent value is consumed.
        // The kernel clamps nonpositive accepted counts to the first column.
        // Evaluate subtraction in int64 so INT32_MIN cannot overflow here.
        const int64_t start = accept ? std::max<int64_t>(accepted[n]-1, 0) : 0;
        c.check(start < columns, "accepted count must select a logical slot column");
        const int64_t slot = slots[n*columns+start];
        if (is_sglang ? slot < 0 : slot <= 0) continue;
        c.check(slot < state->size(0), "selected state slot exceeds pool capacity");
        const auto claim = [&](int64_t id) {
            const auto found = state_owner.emplace(id, n);
            c.check(found.second || found.first->second == n,
                    "different active requests alias a state read/write slot");
        };
        for (const auto& item : {std::make_pair(&q, "consumed q"), {&k, "consumed k"}, {&v, "consumed v"}})
            c.finite(item.first->flatten(0, 1).narrow(0, starts[n], lengths[n]), item.second);
        // The generated launcher always consumes initial state; check only the
        // selected rows, since unused pool rows may intentionally be uninitialized.
        c.finite(state->select(0, slot).flatten().narrow(0, 0, state_width), "selected initial_state_source");
        if (!disable_state_update || table) claim(slot);
        if (table) for (int64_t t=0; t<lengths[n]; ++t) {
            const int64_t write_slot = slots[n*columns+t];
            if (write_slot > 0) {
                c.check(write_slot < state->size(0), "written state slot exceeds pool capacity");
                claim(write_slot);
            }
        }
        if (cache_payload_used) {
            const int64_t id = cache_slots[n];
            c.check(id < cache_requests, "intermediate cache index outside allocated requests");
            if (id >= 0) c.check(cache_owner.insert(id).second, "active requests alias an intermediate cache row");
            if (parent && id >= 0) for (int64_t t=1; t<lengths[n]; ++t) {
                const int64_t p = parents[n*parent->size(1)+t];
                c.check(p >= std::numeric_limits<int32_t>::min() && p < t,
                        "parent must be a negative int32 mask or an already computed token");
            }
        }
    }
    // Gates permit saturating +/-Inf inputs (for example sigmoid(-Inf)=0).
    // Unused token/pool padding and output-only buffers are never scanned.
}

inline void sanity_packed_inputs(
    const torch::Tensor& mixed_qkv, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& A_log, const torch::Tensor& dt_bias,
    float softplus_beta, float softplus_threshold, std::optional<float> scale,
    const std::optional<torch::Tensor>& state,
    const std::optional<torch::Tensor>& indices, const torch::Tensor& out,
    const std::optional<torch::Tensor>& cu, bool is_kda, bool is_sglang) {
    const pla::sanity::Context c("k_last_packed", mixed_qkv);
    if (!c) return;
    int current_device = -1;
    c.check(hggcGetDevice(&current_device) == hggcSuccess && current_device == mixed_qkv.get_device(),
            "input device must be the current CUDA/PPU device for this native launcher");
    for (const auto& item : {std::make_pair(&mixed_qkv, "mixed_qkv"), {&a, "a"},
                            {&b, "b"}, {&A_log, "A_log"}, {&dt_bias, "dt_bias"}, {&out, "out"}})
        c.tensor(*item.first, item.second);
    c.check(state.has_value() && indices.has_value(), "packed decode requires initial state and indices");
    c.tensor(*state, "initial_state_source");
    c.tensor(*indices, "initial_state_indices");
    c.check(state->dim() == 4 && state->size(0) > 0 && state->size(1) > 0 &&
            state->size(2) == 128 && state->size(3) == 128,
            "state must have shape [slots,HV,128,128]");
    c.check(state->scalar_type() == at::kFloat || state->scalar_type() == at::kBFloat16,
            "state dtype must be float32 or bfloat16");
    const int64_t HV = state->size(1), width = HV*128*128;
    c.check(state->stride(3) == 1 && state->stride(2) == 128 &&
            state->stride(1) == 128*128 && state->stride(0) >= width,
            "state inner slot layout must be contiguous [HV,V,K]");
    c.check(reinterpret_cast<uintptr_t>(state->data_ptr()) % 16 == 0 &&
            (state->stride(0)*state->element_size()) % 16 == 0,
            "state base and slot pitch require 16-byte alignment");
    c.check(mixed_qkv.dim() == 2 && mixed_qkv.stride(1) == 1 &&
            mixed_qkv.stride(0) >= mixed_qkv.size(1), "mixed_qkv must have contiguous QKV rows");
    const int64_t tokens = mixed_qkv.size(0);
    const int H = check_packed_heads(mixed_qkv, a, b, A_log, dt_bias, out, HV, 128, 128, is_kda);
    c.check(mixed_qkv.scalar_type() == at::kHalf || mixed_qkv.scalar_type() == at::kBFloat16,
            "mixed_qkv dtype must be float16 or bfloat16");
    for (const auto* t : {&a, &b, &dt_bias, &out})
        c.check(t->scalar_type() == mixed_qkv.scalar_type(), "QKV/a/b/dt_bias/out dtypes must match");
    c.check(A_log.scalar_type() == at::kFloat || A_log.scalar_type() == at::kHalf ||
            A_log.scalar_type() == at::kBFloat16, "A_log dtype is unsupported");
    c.check(A_log.is_contiguous() && dt_bias.is_contiguous(), "A_log/dt_bias must be contiguous");
    c.check(out.is_contiguous(), "out must be contiguous");
    c.check(std::isfinite(softplus_beta) && softplus_beta > 0 &&
            std::isfinite(softplus_threshold), "softplus beta must be positive and parameters finite");
    c.check(!scale || std::isfinite(*scale), "scale must be finite");
    for (const auto& item : {std::make_pair(&mixed_qkv, "mixed_qkv"), {&a, "a"}, {&b, "b"}}) {
        const auto& t = *item.first;
        const int axis = t.dim()-2;
        const int64_t row = t.size(-1);
        c.check(t.stride(-1) == 1 && t.stride(axis) >= row && t.numel() == tokens*row,
                item.second, " token rows must be contiguous and match packed token count");
        int64_t expected = t.stride(axis)*t.size(axis);
        for (int d=axis-1; d>=0; --d) {
            c.check(t.size(d) <= 1 || t.stride(d) == expected,
                    item.second, " leading token axes must flatten with one stride");
            expected *= t.size(d);
        }
        c.check(t.stride(axis) <= std::numeric_limits<int>::max() &&
                (tokens == 0 || (tokens-1)*t.stride(axis)+row <= std::numeric_limits<int>::max()),
                item.second, " addressing exceeds int32 kernel offsets");
    }
    c.check(reinterpret_cast<uintptr_t>(mixed_qkv.data_ptr()) % 16 == 0 &&
            (mixed_qkv.stride(0)*mixed_qkv.element_size()) % 16 == 0,
            "packed QKV base/token stride require 16-byte alignment");
    if (is_kda) {
        c.check(reinterpret_cast<uintptr_t>(a.data_ptr()) % 8 == 0 &&
                (a.stride(a.dim()-2)*a.element_size()) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(dt_bias.data_ptr()) % 8 == 0,
                "KDA a/dt_bias vector loads require 8-byte alignment");
    }
    int64_t B = tokens;
    if (cu) {
        c.tensor(*cu, "cu_seqlens");
        c.check(cu->dim() == 1 && cu->numel() >= 2 && cu->is_contiguous() &&
                cu->scalar_type() == at::kInt, "cu_seqlens must be contiguous int32 [B+1]");
        B = cu->numel()-1;
    }
    c.check(B > 0 && B*H <= std::numeric_limits<int>::max(), "invalid batch/grid size");
    c.check(indices->dim() == 1 && indices->numel() == B && indices->is_contiguous() &&
            indices->scalar_type() == at::kInt, "packed state indices must be contiguous int32 [B]");
    if (!c.values_enabled()) return;  // No D2H or value reductions during capture.
    std::vector<int64_t> lengths(B, 1);
    std::vector<int64_t> starts(B);
    for (int64_t n=0; n<B; ++n) starts[n] = n;
    if (cu) {
        const auto offsets = c.integers(*cu);
        c.check(offsets[0] == 0, "cu_seqlens must start at zero");
        for (int64_t n=0; n<B; ++n) {
            starts[n] = offsets[n];
            lengths[n] = offsets[n+1]-offsets[n];
            c.check(offsets[n] >= 0 && offsets[n+1] <= tokens && lengths[n] >= 0 && lengths[n] <= 1,
                    "packed decode requires each cu_seqlens interval to be 0 or 1 within token storage");
        }
    }
    const auto slots = c.integers(*indices);
    std::unordered_set<int64_t> active_slots;
    for (int64_t n=0; n<B; ++n) {
        if (lengths[n] == 0) continue;
        const int64_t id = slots[n];
        if (is_sglang ? id < 0 : id <= 0) continue;
        c.check(id < state->size(0), "selected state slot exceeds pool capacity");
        c.check(active_slots.insert(id).second, "active packed requests alias a mutable state slot");
        c.finite(state->select(0, id), "selected initial_state_source");
        c.finite(mixed_qkv.select(0, starts[n]), "consumed mixed_qkv");
    }
    // Gates permit saturating +/-Inf inputs (for example sigmoid(-Inf)=0).
    // Unused token/pool padding and output-only buffers are never scanned.
}


inline void sanity_recurrent_kda_inputs(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& raw_g, const torch::Tensor& raw_beta,
    const torch::Tensor& A_log, const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens, const torch::Tensor& ssm_state_indices,
    const std::optional<torch::Tensor>& dt_bias,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& out,
    const std::optional<double>& lower_bound) {
    const pla::sanity::Context sanity("fused_recurrent_kda", q);
    if (sanity) {
        for (const auto& item : std::initializer_list<std::pair<const char*, torch::Tensor>>{
                 {"q", q}, {"k", k}, {"v", v}, {"raw_g", raw_g},
                 {"raw_beta", raw_beta}, {"A_log", A_log},
                 {"initial_state", initial_state}, {"cu_seqlens", cu_seqlens},
                 {"ssm_state_indices", ssm_state_indices}})
            sanity.tensor(item.second, item.first);
        sanity.check(q.dim() == 4 && q.size(0) == 1 && q.size(1) > 0 &&
                         q.size(2) > 0 && q.size(3) == 128,
                     "q must have positive [1, T, H, 128] shape");
        sanity.check(q.scalar_type() == torch::kHalf ||
                         q.scalar_type() == torch::kBFloat16,
                     "q must be fp16 or bf16");
        sanity.check(k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
                         k.scalar_type() == q.scalar_type() &&
                         v.scalar_type() == q.scalar_type(),
                     "q/k/v must have matching shapes and dtypes");
        sanity.check(raw_g.sizes() == q.sizes() && raw_g.is_floating_point(),
                     "raw_g must be floating [1, T, H, 128]");
        sanity.check(raw_beta.sizes() == torch::IntArrayRef({1, q.size(1), q.size(2)}) &&
                         raw_beta.is_floating_point(),
                     "raw_beta must be floating [1, T, H]");
        sanity.check(A_log.numel() == q.size(2) &&
                         (A_log.scalar_type() == torch::kFloat ||
                          A_log.scalar_type() == torch::kHalf ||
                          A_log.scalar_type() == torch::kBFloat16),
                     "A_log must contain H fp32/fp16/bf16 values");
        sanity.check(initial_state.dim() == 4 && initial_state.size(1) == q.size(2) &&
                         initial_state.size(2) == 128 && initial_state.size(3) == 128,
                     "initial_state must be [slots, H, 128, 128]");
        sanity.check(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                     "cu_seqlens must be a vector with at least two elements");
        sanity.check(ssm_state_indices.dim() == 1 || ssm_state_indices.dim() == 2,
                     "ssm_state_indices must be a vector or table");
        sanity.check(ssm_state_indices.size(0) == cu_seqlens.numel() - 1,
                     "cu_seqlens and state indices must describe the same requests");
        if (dt_bias) {
            sanity.tensor(*dt_bias, "dt_bias");
            sanity.check(dt_bias->numel() == q.size(2) * 128,
                         "dt_bias must contain H * 128 values");
        }
        if (num_accepted_tokens) {
            sanity.tensor(*num_accepted_tokens, "num_accepted_tokens");
            sanity.check(num_accepted_tokens->dim() == 1 &&
                             num_accepted_tokens->numel() == cu_seqlens.numel() - 1,
                         "num_accepted_tokens must be [N]");
        }
        if (out) {
            sanity.tensor(*out, "out");
            sanity.check(out->sizes() == v.sizes() && out->scalar_type() == v.scalar_type(),
                         "out must match v shape and dtype");
        }
        if (lower_bound)
            sanity.check(std::isfinite(*lower_bound) && *lower_bound < 0.0 &&
                             std::isfinite(static_cast<float>(*lower_bound)),
                         "lower_bound must be a finite negative float32 value");
        // Validate before narrowing: checking the converted int32 metadata in
        // the downstream entry cannot detect an int64 value that wrapped.
        const auto check_metadata = [&](const torch::Tensor& metadata) {
            sanity.check(metadata.scalar_type() == torch::kInt32 ||
                             metadata.scalar_type() == torch::kInt64,
                         "metadata must use int32 or int64");
            if (!sanity.values_enabled()) return;
            for (const auto value : sanity.integers(metadata))
                sanity.check(value >= std::numeric_limits<int32_t>::min() &&
                                 value <= std::numeric_limits<int32_t>::max(),
                             "metadata value cannot be represented by the int32 kernel ABI");
        };
        check_metadata(cu_seqlens);
        check_metadata(ssm_state_indices);
        if (num_accepted_tokens) check_metadata(*num_accepted_tokens);
        // The downstream entry scans only consumed Q/K/V and state rows.
        // Raw sigmoid/softplus gates may validly contain saturating infinities.
    }
}
