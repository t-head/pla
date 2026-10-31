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
#include <hggc_runtime.h>
#include <cstdint>

// Strongly typed internal ABI; contains no torch, Python or cross-module capsule.
namespace fla_runtime {
using Configure = hggcError_t (*)();
using Gate = hggcError_t (*)(const void*, const float*, const float*, float*, int,
                            const int32_t*, const int32_t*, int, int, int, float, hggcStream_t);
using Token = hggcError_t (*)(const void*, const void*, const float*, const uint16_t*, int64_t,
                             void*, float*, void*, float, const int32_t*, int, int, int, hggcStream_t);
using Inter = hggcError_t (*)(const void*, const void*, const float*, const uint16_t*, int64_t, const float*,
                             void*, void*, float, const int32_t*, const int32_t*, int, int, int, int, hggcStream_t);
using Recompute = hggcError_t (*)(const void*, const void*, const uint16_t*, int64_t, const void*, const float*,
                                 void*, void*, void*, const int32_t*, const int32_t*, int, int, int, int, hggcStream_t);
using Recurrence = hggcError_t (*)(const void*, const void*, const void*, void*, const float*, float*,
                                  void*, const void*, void*, int, const int32_t*, const int32_t*, const int64_t*,
                                  int, int, int, int, hggcStream_t);
using Output = hggcError_t (*)(const void*, const void*, const float*, void*, const void*, void*,
                              const void*, const int32_t*, const int32_t*, float, int, int, int, int, hggcStream_t);
using CpRecurrenceOutput = hggcError_t (*)(
    const void*, const void*, void*, const void*, const void*, const void*, const float*,
    const void*, void*, int, const int32_t*, float*, float*, int*, const int32_t*,
    const int64_t*, int, int, hggcStream_t);
struct Launchers {
  int heads;
  Configure configure_recompute, configure_recurrence, configure_output, configure_cp;
  Gate gate;
  Token token;
  Inter inter;
  Recompute recompute;
  Recurrence recurrence;
  Output output;
  CpRecurrenceOutput cp_recurrence_output;
};
const Launchers& launchers_for(int heads);
}  // namespace fla_runtime

extern "C" hggcError_t launch_fla_inter_prepared__fla_h4(
    const void*, const void*, const float*, const uint16_t*, int64_t,
    float*, void*, void*, void*, void*, void*, float, const int32_t*,
    const int32_t*, int, int, int, int, hggcStream_t);
extern "C" hggcError_t launch_fla_inter_prepared__fla_h8(
    const void*, const void*, const float*, const uint16_t*, int64_t,
    float*, void*, void*, void*, void*, void*, float, const int32_t*,
    const int32_t*, int, int, int, int, hggcStream_t);

hggcError_t launch_fla_l2norm_bf16_d128_bt16(const void*, void*, int64_t, hggcStream_t);
hggcError_t launch_fla_l2norm2_bf16_d128_bt16(const void*, void*, const void*, void*, int64_t, hggcStream_t);
hggcError_t launch_fla_metadata(const void*, bool, int32_t*, int64_t*, int32_t*, int32_t*,
                                int, int, int, hggcStream_t);
