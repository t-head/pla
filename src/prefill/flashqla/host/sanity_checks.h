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

// flashqla input diagnostic helpers: shape/layout/dtype validation kept apart from the kernel launch body,
// shared by the extension's entry points. Only validation happens here; no kernel is launched.
#include <torch/extension.h>

#include "../../sanity_check.h"

inline void sanity_gate(const pla::sanity::Context& check, const torch::Tensor& value,
                 const char* name, int64_t T, int64_t H) {
    check.tensor(value, name);
    check.check(value.scalar_type() == torch::kFloat32 && value.dim() == 3 &&
                    value.size(0) == 1 && value.size(1) == T && value.size(2) == H &&
                    value.is_contiguous(),
                name, " must be contiguous fp32 [1,T,H]");
}

inline void sanity_feature(const pla::sanity::Context& check, const torch::Tensor& value,
                    const char* name, int64_t T, int64_t H, int64_t D,
                    bool token_strided = false) {
    check.tensor(value, name);
    check.check(value.dim() == 4 && value.size(0) == 1 &&
                    value.size(1) == T && value.size(2) == H && value.size(3) == D,
                name, " must have shape [1,T,H,D]");
    check.check(token_strided
                    ? value.stride(3) == 1 && value.stride(2) == D &&
                        value.stride(1) >= H * D
                    : value.is_contiguous(),
                name, " has an unsupported layout");
    check.check(value.scalar_type() == torch::kFloat16 ||
                    value.scalar_type() == torch::kBFloat16,
                name, " must be fp16 or bf16");
}
