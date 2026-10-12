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

#include <string>
#include <iostream>
#include <atomic>

#include "op_forge_case.h"

#include <hgtx3/hgtx3.hpp>
// RAII range wrapper over the PPU HGTX C++ API
#define PROFILE_SCOPE(name) hgtx3::scoped_range profile_range{name}

// Thread-safe initialization helper
#ifdef __cpp_lib_atomic_guard_probeless
#include <mutex>
#endif

// Module-level cached environment variables (defined in fused_sigmoid_gating.cpp)
extern int g_sail_pla_cuda;
extern int g_ppu_lib_perf_instrument;
extern int g_ppu_lib_show_params;
extern std::atomic<bool> g_env_initialized;

// Initialize environment variables - called only once
static inline void init_env_variables() {
    if (g_env_initialized.load(std::memory_order_acquire)) {
        return;
    }

    char* env_sail_pla = std::getenv("SAIL_PLA_CUDA");
    g_sail_pla_cuda = env_sail_pla != nullptr ? std::atoi(env_sail_pla) : 0;

    char* env_perf = std::getenv("PPU_LIB_PERF_INSTRUMENT");
    g_ppu_lib_perf_instrument = env_perf != nullptr ? std::atoi(env_perf) : 0;

    g_ppu_lib_show_params = pla::instrument::show_params();

    g_env_initialized.store(true, std::memory_order_release);
}

// Helper to push NVTX and/or print tag
// Only call this when g_ppu_lib_perf_instrument or g_ppu_lib_show_params is set
static inline void push_nvtx_and_print(
    const std::string& tag,
    bool use_nvtx
) {
    if (g_ppu_lib_show_params) {
        std::cout << tag << std::endl;
    }
    if (use_nvtx && g_ppu_lib_perf_instrument) {
        PROFILE_SCOPE(tag.c_str());
    }
}

#define DISPATCH_TYPES(TYPE, ...)                              \
    do {                                                       \
        if (false) {                                           \
        } else if ((TYPE) == torch::kFloat16) {                \
            using scalar_t = at::Half;                         \
            __VA_ARGS__();                                     \
        } else if ((TYPE) == torch::kBFloat16) {               \
            using scalar_t = at::BFloat16;                     \
            __VA_ARGS__();                                     \
        } else {                                               \
            throw std::runtime_error("Unsupported type");      \
        }                                                      \
    } while (0)

#define DISPATCH_ALOG_TYPES(TYPE, ...)                         \
    do {                                                       \
        if ((TYPE) == torch::kFloat32) {                       \
            using alog_t = float;                              \
            __VA_ARGS__();                                     \
        } else if ((TYPE) == torch::kFloat16) {                \
            using alog_t = at::Half;                           \
            __VA_ARGS__();                                     \
        } else if ((TYPE) == torch::kBFloat16) {               \
            using alog_t = at::BFloat16;                       \
            __VA_ARGS__();                                     \
        } else {                                               \
            throw std::runtime_error("Unsupported type");      \
        }                                                      \
    } while (0)

// The device-side helpers used to live here, guarded by
// __HGGCCC__ && !PLA_CUSTOM_DEVICE_HELPERS. They moved to
// pla_device_helpers.cuh so device translation units can include them without
// dragging in torch; see the comment at the top of that file.
