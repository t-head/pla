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

#include "../sanity_check.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

namespace pla {
namespace sanity {
namespace prefill {

inline void int_extent(const Context& check, int64_t extent, const char* name) {
    check.check(extent > 0 && extent <= std::numeric_limits<int>::max(),
                name, " must be positive and fit the kernel int32 ABI");
}

inline void offset_values(const Context& check, const std::vector<int64_t>& cu,
                          int64_t capacity, const char* name) {
    check.check(cu.size() >= 2, name, " must contain at least two offsets");
    check.check(cu.front() == 0, name, " must start at zero");
    for (size_t i = 1; i < cu.size(); ++i) {
        check.check(cu[i] >= cu[i - 1] && cu[i] <= capacity &&
                        cu[i] <= std::numeric_limits<int32_t>::max(),
                    name, " must be monotonic within the physical token capacity; entry ", i);
    }
}

inline std::vector<int64_t> offsets(
    const Context& check, const at::Tensor& cu, int64_t capacity,
    const char* name = "cu_seqlens", bool allow_i64 = false) {
    check.tensor(cu, name);
    check.check(cu.dim() == 1 && cu.is_contiguous() && cu.numel() >= 2 &&
                    cu.numel() < std::numeric_limits<int>::max(),
                name, " must be contiguous [batch+1]");
    check.check(cu.scalar_type() == at::kInt ||
                    (allow_i64 && cu.scalar_type() == at::kLong),
                name, allow_i64 ? " must be int32 or int64" : " must be int32");
    if (!check.values_enabled()) return {};
    auto values = check.integers(cu);
    offset_values(check, values, capacity, name);
    return values;
}

inline void chunks(const Context& check, const at::Tensor& indices,
                   const std::vector<int64_t>& cu, int chunk_size = 32) {
    check.tensor(indices, "chunk_indices");
    check.check(indices.scalar_type() == at::kInt && indices.dim() == 2 &&
                    indices.size(1) == 2 && indices.is_contiguous() &&
                    indices.size(0) <= std::numeric_limits<int>::max(),
                "chunk_indices must be contiguous int32 [num_chunks,2]");
    if (!check.values_enabled()) return;
    const auto values = check.integers(indices);
    int64_t expected = 0;
    for (size_t i = 1; i < cu.size(); ++i)
        expected += (cu[i] - cu[i - 1] + chunk_size - 1) / chunk_size;
    check.check(indices.size(0) == expected,
                "chunk_indices must cover every real chunk exactly once");
    std::set<std::pair<int64_t, int64_t>> seen;
    for (size_t i = 0; i < values.size(); i += 2) {
        const int64_t seq = values[i], chunk = values[i + 1];
        check.check(seq >= 0 && seq < int64_t(cu.size()) - 1,
                    "chunk_indices sequence out of range at row ", i / 2);
        check.check(chunk >= 0 && chunk < (cu[seq + 1] - cu[seq] + chunk_size - 1) / chunk_size,
                    "chunk_indices chunk out of range at row ", i / 2);
        check.check(seen.emplace(seq, chunk).second,
                    "duplicate chunk_indices entry at row ", i / 2);
    }
}

inline void slots(const Context& check, const at::Tensor& indices,
                  int64_t count, int64_t capacity, const char* name,
                  bool unique, const std::vector<int64_t>& cu = {}) {
    check.tensor(indices, name);
    check.check(indices.scalar_type() == at::kInt && indices.dim() == 1 &&
                    indices.numel() == count && indices.is_contiguous(),
                name, " must be contiguous int32 [batch]");
    if (!check.values_enabled()) return;
    const auto values = check.integers(indices);
    std::set<int64_t> seen;
    for (int64_t i = 0; i < count; ++i) {
        // Only callers whose kernel skips empty entries may supply cu here.
        if (!cu.empty() && cu[i] == cu[i + 1]) continue;
        check.check(values[i] >= 0 && values[i] < capacity,
                    name, " slot out of range at sequence ", i);
        if (unique)
            check.check(seen.insert(values[i]).second,
                        name, " duplicate writable state slot at sequence ", i);
    }
}

inline std::vector<int64_t> binary_mask(const Context& check,
                                      const at::Tensor& mask, int64_t count,
                                      const char* name) {
    check.tensor(mask, name);
    check.check(mask.scalar_type() == at::kChar && mask.is_contiguous() &&
                    mask.numel() == count,
                name, " must contain the expected contiguous int8 entries");
    if (!check.values_enabled()) return {};
    const auto host = mask.to(at::TensorOptions().device(at::kCPU).dtype(at::kLong))
                          .contiguous();
    const auto* data = host.data_ptr<int64_t>();
    std::vector<int64_t> values(data, data + count);
    for (int64_t i = 0; i < count; ++i)
        check.check(values[i] == 0 || values[i] == 1,
                    name, " must contain only 0/1, entry ", i);
    return values;
}

// Use only for contiguous tensors. Exact in-place state aliases are supported;
// partial aliases can race between independently scheduled sequences/heads.
inline bool overlaps(const at::Tensor& a, const at::Tensor& b) {
    if (a.numel() == 0 || b.numel() == 0) return false;
    auto span = [](const at::Tensor& t) {
        int64_t last = 0;
        for (int64_t d = 0; d < t.dim(); ++d)
            last += (t.size(d) - 1) * t.stride(d);
        return (last + 1) * t.element_size();
    };
    const auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
    return x < y + span(b) && y < x + span(a);
}

inline bool same_view(const at::Tensor& a, const at::Tensor& b) {
    return a.data_ptr() == b.data_ptr() && a.scalar_type() == b.scalar_type() &&
        a.sizes() == b.sizes() && a.strides() == b.strides();
}

}  // namespace prefill
}  // namespace sanity
}  // namespace pla
