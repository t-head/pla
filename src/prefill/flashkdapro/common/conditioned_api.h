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

namespace flashkda {
// Call-local, by-value descriptors. Device offsets/indices remain live inputs
// on every Graph replay; no mutable metadata is cached in this descriptor.
struct ConditionedArguments {
    const void *q, *k, *v, *g, *beta;
    const float *a_log, *dt_bias;
    void *out, *final_state;
    const void *initial_state;
    const int32_t *cu, *indices;
    const void* fast_state = nullptr;
    const float* value_scales = nullptr;
    int32_t* value_flags = nullptr;
    int tokens, heads, sequences, beta_stride;
    int initial_slots, final_slots;
    bool state_fp32, fla_normalization;
    bool value_guard = false, preserve_initial = false, publish_fast_state = false;
    bool force_direct = false;
    double scale, lower_bound;
};
void launch_conditioned_workspace_snapshot(ConditionedArguments args, const void* ws,
                                             int total_tiles, int32_t* flags,
                                             float* snapshot, hggcStream_t stream);
void launch_conditioned_repair(ConditionedArguments args, const int32_t* flags,
                               const float* snapshot, hggcStream_t stream);
int64_t conditioned_cache_bytes(int capacity);
void launch_conditioned_repair_cached(ConditionedArguments args, const int32_t* flags,
                                      const float* snapshot, void* workspace,
                                      int capacity, hggcStream_t stream);
void launch_conditioned_direct(ConditionedArguments args, bool float_state, hggcStream_t stream,
                               int32_t* output_flags = nullptr, const float* initial_snapshot = nullptr);
void launch_conditioned_preserve(ConditionedArguments args, float* snapshot, hggcStream_t stream);
}  // namespace flashkda
