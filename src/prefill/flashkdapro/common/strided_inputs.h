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
 * The kernels address q/k/v/g, A_log, dt_bias and the recurrent states as dense
 * row-major buffers, and beta through a token stride. Serving frameworks pass
 * exactly that layout, so those tensors go through untouched after a few
 * metadata reads. Any other strided view is materialized once on the current
 * stream before the first kernel; a strided final state is computed into a
 * dense scratch and published to the caller's view after the last kernel.
 * Malformed tensors (wrong rank/device/dtype) are never rejected here: the
 * entry's own validation keeps reporting them.
 */
#pragma once

#include <ATen/ATen.h>

#include "../../sanity_check.h"

#include <cstdint>
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

inline void stage_inputs(at::Tensor& q, at::Tensor& k, at::Tensor& v, at::Tensor& g,
                         at::Tensor& beta, at::Tensor& a_log, at::Tensor& dt_bias) {
    for (auto* x : {&q, &k, &v, &g}) stage_dense(*x, 16);
    if (beta.defined() && beta.dim() == 2 && !beta_native(beta))
        beta = beta.clone(at::MemoryFormat::Contiguous);
    stage_dense(a_log, 4);
    stage_dense(dt_bias, 16);
}

inline bool same_storage(const at::Tensor& a, const at::Tensor& b) {
    return a.defined() && b.defined() && a.numel() && b.numel() &&
           a.storage().data_ptr().get() == b.storage().data_ptr().get();
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
    const bool initial_dense = !initial.defined() || initial.is_contiguous();
    const bool final_dense = !final.defined() || final.is_contiguous();
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
