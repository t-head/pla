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

#include "heads.h"
#include "launchers.h"

hggcError_t launch_fla_gate_cumsum_bf16_h8_aligned_m1(const void*, const float*, const float*,
    float*, int, const int32_t*, const int32_t*, int, int, int, float, hggcStream_t);
hggcError_t launch_fla_glm53_token_intra_bf16_d128_bt64_bc16_h8(const void*, const void*,
    const float*, const uint16_t*, int64_t, void*, float*, void*, float, const int32_t*, int, int, int, hggcStream_t);
extern "C" {
hggcError_t launch_fla_inter_solve_bf16_d128_bt64_bc16_h8(const void*, const void*, const float*,
    const uint16_t*, int64_t, const float*, void*, void*, float, const int32_t*, const int32_t*, int, int, int, int, hggcStream_t);
hggcError_t configure_fla_recompute_h8_bf16_d128_bt64();
hggcError_t launch_fla_recompute_h8_bf16_d128_bt64(const void*, const void*, const uint16_t*, int64_t,
    const void*, const float*, void*, void*, void*, const int32_t*, const int32_t*, int, int, int, int, hggcStream_t);
hggcError_t configure_fla_glm53_recurrence_h8_bf16_d128_bt64();
hggcError_t launch_fla_glm53_recurrence_h8_bf16_d128_bt64(const void*, const void*, const void*,
    void*, const float*, float*, void*, const void*, void*, int, const int32_t*, const int32_t*, const int64_t*,
    int, int, int, int, hggcStream_t);
hggcError_t configure_fla_glm53_output_h8_bf16_d128_bt64();
hggcError_t launch_fla_glm53_output_h8_bf16_d128_bt64(const void*, const void*, const float*,
    void*, const void*, void*, const void*, const int32_t*, const int32_t*, float, int, int, int, int, hggcStream_t);
#if FLA_HEADS == 4 || FLA_HEADS == 6 || FLA_HEADS == 8 || FLA_HEADS == 12
hggcError_t configure_fla_cp_pretriangular_output();
hggcError_t launch_fla_cp_pretriangular_output(
    const void*, const void*, void*, const void*, const void*, const void*, const float*,
    const void*, void*, int, const int32_t*, float*, float*, int*, const int32_t*,
    const int64_t*, int, int, hggcStream_t);
#endif
#if FLA_HEADS == 4 || FLA_HEADS == 8
hggcError_t FLA_JOIN_HEAD(launch_fla_token_l2, FLA_HEADS)(
    const void*, const void*, void*, void*, const float*, const uint16_t*, int64_t,
    void*, float*, void*, void*, void*, void*, float, const int32_t*, int, int, hggcStream_t);
hggcError_t FLA_JOIN_HEAD(launch_fla_inter_solve_prepared, FLA_HEADS)(
    const void*, const void*, const float*, const uint16_t*, int64_t,
    float*, void*, void*, void*, void*, void*, float, const int32_t*,
    const int32_t*, int, int, int, int, hggcStream_t);
#endif
}

namespace fla_runtime {
const Launchers& FLA_JOIN_HEAD(head_adapter, FLA_HEADS)() {
#if FLA_HEADS == 4 || FLA_HEADS == 6 || FLA_HEADS == 8 || FLA_HEADS == 12
  constexpr Configure cp_configure = configure_fla_cp_pretriangular_output;
  constexpr CpRecurrenceOutput cp_launch = launch_fla_cp_pretriangular_output;
#else
  constexpr Configure cp_configure = nullptr;
  constexpr CpRecurrenceOutput cp_launch = nullptr;
#endif
#if FLA_HEADS == 4 || FLA_HEADS == 8
  constexpr TokenL2 token_l2 = FLA_JOIN_HEAD(launch_fla_token_l2, FLA_HEADS);
  constexpr InterPrepared inter_prepared = FLA_JOIN_HEAD(launch_fla_inter_solve_prepared, FLA_HEADS);
#else
  constexpr TokenL2 token_l2 = nullptr;
  constexpr InterPrepared inter_prepared = nullptr;
#endif
  static const Launchers value = {
      FLA_HEADS,
      configure_fla_recompute_h8_bf16_d128_bt64,
      configure_fla_glm53_recurrence_h8_bf16_d128_bt64,
      configure_fla_glm53_output_h8_bf16_d128_bt64,
      cp_configure,
      launch_fla_gate_cumsum_bf16_h8_aligned_m1,
      launch_fla_glm53_token_intra_bf16_d128_bt64_bc16_h8,
      launch_fla_inter_solve_bf16_d128_bt64_bc16_h8,
      launch_fla_recompute_h8_bf16_d128_bt64,
      launch_fla_glm53_recurrence_h8_bf16_d128_bt64,
      launch_fla_glm53_output_h8_bf16_d128_bt64,
      cp_launch,
      token_l2,
      inter_prepared};
  return value;
}
}  // namespace fla_runtime
