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
#include <ATen/ATen.h>
#include <optional>
#include <vector>

namespace fla_runtime {
inline constexpr const char* kRuntimeAbi = "tc01_fla_monolithic_runtime_v2";
inline constexpr const char* kKernelAbi = "tc01_fla_packed_v3_cpstate_v1";
inline constexpr const char* kCpAbi = "tc01_fla_cp_state_passing_v2";
struct Arguments {
  at::Tensor q, k, v, g, beta, a_log, dt_bias, out;
  double scale = 0.08838834764831845, lower_bound = -5.;
  at::Tensor initial, final, cu, pool, indices;
  std::optional<std::vector<int64_t>> raw;
};
void forward(Arguments args);
std::vector<int> supported_heads();
std::vector<int> supported_cp_heads();
// Standalone negative acceptance entry: only enqueues metadata and adds no synchronization to the production entry.
std::vector<at::Tensor> metadata_only(const at::Tensor& input, int64_t tokens, int64_t heads);
}  // namespace fla_runtime
