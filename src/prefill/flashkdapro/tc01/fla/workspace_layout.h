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
#include "cp_plan.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fla_runtime {
inline void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}
inline int64_t checked_add(int64_t a, int64_t b) {
  require(a >= 0 && b >= 0 && a <= std::numeric_limits<int64_t>::max() - b, "workspace addition overflow");
  return a + b;
}
inline int64_t checked_mul(int64_t a, int64_t b) {
  require(a >= 0 && b >= 0 && (!b || a <= std::numeric_limits<int64_t>::max() / b), "workspace multiplication overflow");
  return a * b;
}
inline int64_t align_bytes(int64_t n) { return checked_add(n, 255) / 256 * 256; }
inline void dimensions(int64_t t, int64_t h, int64_t n, int64_t chunks) {
  require(t >= 0 && t <= int64_t(INT32_MAX) - 64, "physical token count exceeds the int32 range");
  require(h > 0 && h <= 65535 && n > 0 && n <= 65535, "head/sequence exceeds the device grid range");
  require(chunks >= 0 && chunks <= 65535, "chunk capacity exceeds the device grid range");
  checked_mul(checked_mul(t, h), 128 * 4);
  checked_mul(checked_mul(chunks, h), 128 * 128 * 2);
  checked_mul(checked_mul(n, h), 128 * 128 * 4);
  require((checked_mul(checked_mul(t, h), 2) + 15) / 16 <= INT32_MAX, "L2 grid exceeds the device range");
}
inline int64_t chunk_capacity(int64_t t, int64_t n) {
  require(t >= 0 && n > 0, "illegal metadata dimensions");
  auto nonempty = std::min(t, n);
  return nonempty ? nonempty + (t - nonempty) / 64 : 0;
}
inline std::vector<int64_t> chunk_offsets(const std::vector<int64_t>& raw, int64_t t) {
  require(raw.size() >= 2 && raw.size() <= 65536, "raw offsets must contain N+1 elements");
  require(raw.front() == 0 && raw.back() <= t, "illegal offset start/end");
  std::vector<int64_t> result(raw.size(), 0);
  for (size_t i = 1; i < raw.size(); ++i) {
    require(raw[i] >= raw[i - 1] && raw[i] <= INT32_MAX, "offsets must be non-decreasing and within the int32 range");
    result[i] = checked_add(result[i - 1], (raw[i] - raw[i - 1] + 63) / 64);
  }
  return result;
}
struct Region { int64_t offset = 0, bytes = 0; };
struct Layout {
  int64_t bytes = 0;
  Region add(int64_t size) {
    require(size >= 0, "illegal workspace region size");
    Region r{align_bytes(bytes), size};
    bytes = checked_add(r.offset, size);
    return r;
  }
};
struct MetadataLayout : Layout {
  Region cu, offsets, indices, slots;
  Region cp_bounds, cp_offsets;
  MetadataLayout(int64_t n, int64_t chunks) {
    cu = add(checked_mul(checked_add(n, 1), 4));
    offsets = add(checked_mul(checked_add(n, 1), 8));
    indices = add(checked_mul(chunks, 8));
    slots = add(checked_mul(n, 4));
    cp_bounds = add(checked_mul(
        checked_mul(kCpSplits.size(), kCpDescriptorWidth), 4));
    cp_offsets = add(checked_mul(
        checked_mul(kCpSplits.size(), kCpDescriptorWidth), 8));
    bytes = align_bytes(bytes);
  }
};
struct WorkspaceLayout : Layout {
  Region metadata;
  Region qn, kn, qg, gate, aqk, akkd, inverse, w, u, kg;
  Region v_new, history, cp_bias, cp_state_passing, cp_executed, zeros;
  MetadataLayout meta;
  WorkspaceLayout(int64_t t, int64_t h, int64_t n, int64_t chunks,
                  bool device_metadata, int cp_segments = 0)
      : meta(n, chunks) {
    dimensions(t, h, n, chunks);
    require(cp_segments >= 0 && cp_segments <= kCpMaxSegments,
            "FLA CP segment count out of range");
    if (device_metadata) metadata = add(meta.bytes);
    if (t && chunks) {
      auto th = checked_mul(t, h);
      auto activation = checked_mul(th, 128 * 2);
      qn = add(activation); kn = add(activation); qg = add(activation);
      gate = add(checked_mul(activation, 2));
      aqk = add(checked_mul(th, 64 * 2));
      akkd = add(checked_mul(th, 16 * 4));
      inverse = add(checked_mul(th, 64 * 2));
      zeros = {aqk.offset, checked_add(inverse.offset, inverse.bytes) - aqk.offset};
      w = add(activation); u = add(activation); kg = add(activation);
      if (cp_segments) {
        const auto state_bytes = checked_mul(
            checked_mul(checked_mul(cp_segments, h), 128 * 128), 4);
        cp_bias = add(state_bytes);
        cp_state_passing = add(checked_mul(2, state_bytes));
        // Reserved to preserve the existing arena size while the old repair
        // instrumentation is removed from downstream tooling.
        cp_executed = add(checked_mul(checked_mul(cp_segments, h), 8 * 4));
      } else {
        v_new = add(activation);
        history = add(checked_mul(checked_mul(chunks, h), 128 * 128 * 2));
      }
    }
    bytes = align_bytes(bytes);
  }
};
}  // namespace fla_runtime
