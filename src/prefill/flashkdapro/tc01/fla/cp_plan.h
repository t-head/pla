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
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace fla_runtime {

inline constexpr std::array<int, 6> kCpSplits = {1, 2, 3, 4, 5, 8};
inline constexpr int kCpMaxSegments = 8;
inline constexpr int kCpDescriptorWidth = kCpMaxSegments + 1;

struct CpGeometry {
  int split = 0;
  int count = 0;
  std::array<int32_t, kCpDescriptorWidth> bounds{};
  std::array<int64_t, kCpDescriptorWidth> offsets{};

  bool enabled() const { return count > 0; }
};

inline int cp_split_row(int split) {
  for (int row = 0; row < static_cast<int>(kCpSplits.size()); ++row)
    if (kCpSplits[row] == split) return row;
  throw std::invalid_argument("FLA CP split must be 1, 2, 3, 4, 5, or 8");
}

inline CpGeometry make_cp_geometry(int64_t tokens, int split) {
  cp_split_row(split);
  if (tokens <= 0 || tokens > int64_t(INT32_MAX) - 64)
    throw std::invalid_argument("FLA CP requires a positive signed-int32 token count");
  const int64_t chunks = (tokens + 63) / 64;
  const int count = static_cast<int>(std::min<int64_t>(chunks, split));
  const int64_t quotient = chunks / count;
  const int64_t remainder = chunks % count;
  CpGeometry result;
  result.split = split;
  result.count = count;
  int64_t chunk_offset = 0;
  result.bounds[0] = 0;
  result.offsets[0] = 0;
  for (int segment = 0; segment < count; ++segment) {
    chunk_offset += quotient + (segment < remainder ? 1 : 0);
    result.offsets[segment + 1] = chunk_offset;
    result.bounds[segment + 1] = static_cast<int32_t>(
        std::min<int64_t>(tokens, 64 * chunk_offset));
  }
  for (int segment = count + 1; segment < kCpDescriptorWidth; ++segment) {
    result.offsets[segment] = result.offsets[count];
    result.bounds[segment] = result.bounds[count];
  }
  return result;
}

// Shape-qualified AUTO policy. Every promoted bucket is backed by paired
// API/Graph measurements; gaps remain serial. H4 uses state-only CP1 for the
// short bucket and CP5 for the long bucket. H6/H8 use state CP1 through 64K,
// then retain legacy CP1. H12 retains its established legacy CP1 threshold.
// Explicit split and algorithm overrides remain authoritative.
// H4/H6/H8 short sequences also use the qualified single-segment recurrence/output fusion.
inline int select_fla_cp_auto_split(int heads, int64_t tokens) {
  if (tokens <= 0 || tokens > 102400) return 0;
  switch (heads) {
    case 4:
      if (tokens <= 2048) return 1;
      return tokens >= 12288 && tokens <= 65536 ? 5 : 0;
    case 6:
    case 8:
      return 1;
    case 12:
      return tokens >= 8192 ? 1 : 0;
    default:
      return 0;
  }
}

}  // namespace fla_runtime
