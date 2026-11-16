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

/*
 * Layout staging shared by the FlashKDA prefill entries (legacy and tc01 FLA).
 *
 * The kernels address g, A_log, dt_bias and the recurrent states as dense
 * row-major buffers, and q/k/v and beta through a token stride over packed
 * rows. Serving frameworks pass exactly that layout (dense tensors, or q/k/v
 * split out of one fused projection), so those tensors go through untouched
 * after a few metadata reads. Any other strided view is materialized once on
 * the current stream before the first kernel; a strided final state is
 * computed into a dense scratch and published to the caller's view after the
 * last kernel.
 * Malformed tensors (wrong rank/device/dtype) are never rejected here: the
 * entry's own validation keeps reporting them.
 */
#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include "../../sanity_check.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace flashkda::layout {

inline bool dense(const at::Tensor& x, uintptr_t alignment) {
    return x.is_contiguous() &&
           reinterpret_cast<uintptr_t>(x.data_ptr()) % alignment == 0;
}

inline void stage_dense(at::Tensor& x, uintptr_t alignment) {
    if (x.defined() && !dense(x, alignment))
        x = x.clone(at::MemoryFormat::Contiguous);
}

// beta is consumed natively with any int32 token stride over a unit head stride.
inline bool beta_native(const at::Tensor& beta) {
    return beta.stride(1) == 1 && beta.stride(0) >= beta.size(1) &&
           beta.stride(0) <= std::numeric_limits<int32_t>::max() &&
           reinterpret_cast<uintptr_t>(beta.data_ptr()) % beta.element_size() == 0;
}

// Token stride of a [T,H,D] q/k/v view in elements. A single-token view has
// no meaningful stride, so it reports the packed row size.
inline int64_t token_stride(const at::Tensor& x) {
    return x.size(0) > 1 ? x.stride(0) : x.size(1) * x.size(2);
}

// q/k/v are consumed natively with a token stride over packed [H,D] rows
// (strides (s, D, 1)). Every row must start 16B aligned for the AIU tile
// loads, and s must fit the tc01 AIU int32 element-stride field.
inline bool qkv_native(const at::Tensor& x) {
    if (x.dim() != 3) return false;
    const int64_t row = x.size(1) * x.size(2);
    const int64_t s = token_stride(x);
    return (x.size(2) <= 1 || x.stride(2) == 1) &&
           (x.size(1) <= 1 || x.stride(1) == x.size(2)) &&
           s >= row && s <= std::numeric_limits<int32_t>::max() &&
           (s * x.element_size()) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
}

inline void stage_inputs(at::Tensor& q, at::Tensor& k, at::Tensor& v, at::Tensor& g,
                         at::Tensor& beta, at::Tensor& a_log, at::Tensor& dt_bias) {
    for (auto* x : {&q, &k, &v})
        if (x->defined() && x->dim() == 3 && !qkv_native(*x))
            *x = x->clone(at::MemoryFormat::Contiguous);
    stage_dense(g, 16);
    if (beta.defined() && beta.dim() == 2 && !beta_native(beta))
        beta = beta.clone(at::MemoryFormat::Contiguous);
    stage_dense(a_log, 4);
    stage_dense(dt_bias, 16);
}

inline bool same_storage(const at::Tensor& a, const at::Tensor& b) {
    return a.defined() && b.defined() && a.numel() && b.numel() &&
           a.has_storage() && b.has_storage() &&
           a.storage().data_ptr().get() == b.storage().data_ptr().get();
}

// Check the caller's views before clones/gathers hide aliases from validation.
// Activation storage is read-only; only exact in-place state views may overlap.
inline void check_aliases(std::initializer_list<const at::Tensor*> inputs,
                          const at::Tensor& out, const at::Tensor& initial,
                          const at::Tensor& final) {
    for (const auto* x : inputs) {
        TORCH_CHECK(!same_storage(out, *x), "out must not share storage with an input");
        TORCH_CHECK(!same_storage(initial, *x),
                    "initial_state must not share storage with an activation");
        TORCH_CHECK(!same_storage(final, *x),
                    "final_state must not share storage with an activation");
    }
    TORCH_CHECK(!same_storage(initial, out), "initial_state must not share storage with out");
    TORCH_CHECK(!same_storage(final, out), "final_state must not share storage with out");
    if (same_storage(initial, final))
        TORCH_CHECK(!pla::sanity::prefill::overlaps(initial, final) ||
                        pla::sanity::prefill::same_view(initial, final),
                    pla::sanity::enabled() ? "PLA_SANITY_CHECK: " : "",
                    "initial/final state may alias only as the exact same tensor view");
}

// Staging and writeback also consume the original allocations asynchronously.
// Record each caller storage once, including state pools, before replacing views.
inline void record_tensors(std::initializer_list<const at::Tensor*> tensors,
                           const c10::cuda::CUDAStream& stream) {
    std::array<const void*, 13> seen{};
    size_t count = 0;
    for (const auto* x : tensors) {
        if (!x->defined() || !x->is_cuda() || !x->numel() || !x->has_storage()) continue;
        auto address = x->storage().data_ptr().get();
        if (std::find(seen.begin(), seen.begin() + count, address) != seen.begin() + count) continue;
        TORCH_INTERNAL_ASSERT(count < seen.size());
        seen[count++] = address;
        c10::cuda::CUDACachingAllocator::recordStream(x->storage().data_ptr(), stream);
    }
}

// A strided final state computed into dense scratch, published by the entry
// after its last kernel on the same stream.
struct StateStage {
    at::Tensor target, scratch, index;
    bool active = false;

    void publish() const {
        if (!target.defined()) return;
        if (index.defined()) target.index_copy_(0, index, scratch);
        else target.copy_(scratch);
    }
};

// Rewrites initial/final/indices so the kernels see dense states. With indices,
// strided pools are gathered into a dense [N,...] batch, the kernels run without
// indices, and the final rows are scattered back into the caller's slots.
// `initial` and `final` may be the same pool view. The caller's slot map is
// sanity-checked before the gather because the kernels no longer see it.
inline StateStage stage_states(const char* operation, at::Tensor& initial, at::Tensor& final,
                               at::Tensor& indices, const at::Tensor& out) {
    const bool initial_dense = !initial.defined() || dense(initial, 16);
    const bool final_dense = !final.defined() || dense(final, 16);
    if (initial_dense && final_dense) return {};
    for (const auto* state : {&initial, &final})
        if (state->defined() && (!state->is_cuda() || state->dim() != 4)) return {};
    if (initial.defined() && final.defined() && initial.scalar_type() != final.scalar_type())
        return {};
    StateStage stage;
    if (indices.defined()) {
        const auto& pool = initial.defined() ? initial : final;
        if (!indices.is_cuda() || indices.device() != pool.device() ||
            indices.scalar_type() != at::kInt || indices.dim() != 1 || !indices.is_contiguous())
            return {};
        const pla::sanity::Context sanity(operation, pool);
        if (sanity)
            for (const auto* state : {&initial, &final})
                if (state->defined())
                    pla::sanity::prefill::slots(sanity, indices, indices.numel(), state->size(0),
                                                "state_indices", final.defined());
    }
    if (final.defined() && (!final_dense || indices.defined())) {
        TORCH_CHECK(!same_storage(final, out), "final_state must not share storage with out");
        stage.target = final;
        auto sizes = final.sizes().vec();
        if (indices.defined()) {
            sizes[0] = indices.size(0);
            stage.index = indices.to(at::kLong);
        }
        stage.scratch = at::empty(sizes, final.options());
    }
    if (initial.defined() && (!initial_dense || indices.defined()))
        initial = indices.defined() ? initial.index_select(0, indices)
                                    : initial.clone(at::MemoryFormat::Contiguous);
    if (stage.target.defined()) final = stage.scratch;
    indices = at::Tensor();
    stage.active = true;
    return stage;
}

}  // namespace flashkda::layout
