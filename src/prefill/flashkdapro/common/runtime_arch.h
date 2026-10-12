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
 * FlashKDApro host-side PPU architecture detection.
 *
 * Device code selects its embedded image with __HGGC_ARCH__.  Host launch
 * policy must instead use the physical device's runtime capability.  The PPU
 * CUDA compatibility runtime reports tc01/ppu001 as 8.0 and tc02/ppu0015 as
 * 8.9.  Keep this mapping exact: CU count is an occupancy property, not an
 * architecture ABI.
 */
#pragma once

#include <hggc_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace flashkda {
namespace host_runtime {

enum class PpuArchitecture {
    kTc01,
    kTc02,
};

inline constexpr int kMaxCachedDevices = 64;

[[noreturn]] inline void throw_ppu_query_error(
    const char* operation, int device, hggcError_t error) {
    const char* detail = hggcGetErrorString(error);
    throw std::runtime_error(
        std::string("FlashKDApro host architecture detection failed: ")
        + operation + " for logical PPU device " + std::to_string(device)
        + ": " + (detail != nullptr ? detail : "unknown HGGC error"));
}

inline int current_device() {
    int device = -1;
    const hggcError_t error = hggcGetDevice(&device);
    if (error != hggcSuccess) {
        throw_ppu_query_error("hggcGetDevice", device, error);
    }
    return device;
}

inline int device_attribute(
    int device, hggcDeviceAttr attribute, const char* attribute_name) {
    int value = 0;
    const hggcError_t error = hggcDeviceGetAttribute(&value, attribute, device);
    if (error != hggcSuccess) {
        throw_ppu_query_error(attribute_name, device, error);
    }
    return value;
}

inline PpuArchitecture query_architecture(int device) {
    const int major = device_attribute(
        device, hggcDevAttrComputeCapabilityMajor,
        "hggcDeviceGetAttribute(hggcDevAttrComputeCapabilityMajor)");
    const int minor = device_attribute(
        device, hggcDevAttrComputeCapabilityMinor,
        "hggcDeviceGetAttribute(hggcDevAttrComputeCapabilityMinor)");

    if (major == 8 && minor == 0) {
        return PpuArchitecture::kTc01;
    }
    if (major == 8 && minor == 9) {
        return PpuArchitecture::kTc02;
    }
    throw std::runtime_error(
        std::string("FlashKDApro host architecture detection failed: ")
        + "unsupported CUDA compute capability "
        + std::to_string(major) + "." + std::to_string(minor)
        + " on logical PPU device " + std::to_string(device)
        + "; expected 8.0 (tc01/ppu001) or 8.9 (tc02/ppu0015)");
}

inline PpuArchitecture current_architecture() {
    const int device = current_device();
    if (device < 0 || device >= kMaxCachedDevices) {
        return query_architecture(device);
    }

    // Runtime properties are immutable for a logical device.  Cache by its
    // CUDA-visible ordinal so device switches and concurrent first calls do
    // not reuse another device's host policy or race the initial query.
    static std::once_flag queried[kMaxCachedDevices];
    static PpuArchitecture cached[kMaxCachedDevices] = {};
    std::call_once(
        queried[device],
        [&]() { cached[device] = query_architecture(device); });
    return cached[device];
}

inline bool current_device_is_tc01() {
    return current_architecture() == PpuArchitecture::kTc01;
}

}  // namespace host_runtime
}  // namespace flashkda
