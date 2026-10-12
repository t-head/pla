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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hggc_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace pla {
namespace sanity {

// Read at the call boundary so a diagnostic session may enable checks after
// importing the extension. The disabled path never queries a CUDA stream.
inline bool enabled() {
    const char* value = std::getenv("PLA_SANITY_CHECK");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

class Context {
public:
    Context(const char* operation, const at::Tensor& reference)
        : operation_(operation), enabled_(enabled()) {
        if (!enabled_) return;
        check(reference.defined(), "reference tensor must be defined");
        check(reference.is_cuda(), "reference tensor must be on CUDA/PPU");
        device_ = reference.get_device();
        const c10::cuda::CUDAGuard guard(reference.device());
        hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
        const auto stream = reinterpret_cast<hggcStream_t>(
            at::cuda::getCurrentCUDAStream(device_).stream());
        const auto result = hggcStreamIsCapturing(stream, &status);
        // An unsuccessful probe must never fall through to a D2H transfer.
        check(result == hggcSuccess, "cannot determine CUDA graph capture status: ",
              hggcGetErrorString(result));
        values_enabled_ = status == hggcStreamCaptureStatusNone;
    }

    explicit operator bool() const { return enabled_; }
    bool values_enabled() const { return values_enabled_; }

    template <typename... Args>
    void check(bool condition, Args&&... message) const {
        if (enabled_) {
            TORCH_CHECK(condition, "PLA_SANITY_CHECK [", operation_, "]: ",
                        std::forward<Args>(message)...);
        }
    }

    void tensor(const at::Tensor& value, const char* name, bool allow_cpu = false) const {
        if (!enabled_) return;
        check(value.defined(), name, " must be defined");
        check(value.layout() == at::kStrided, name, " must have strided layout");
        check((value.is_cuda() && value.get_device() == device_) ||
                  (allow_cpu && value.device().is_cpu()),
              name, " must be on the same CUDA/PPU device as the input",
              allow_cpu ? " (or CPU metadata)" : "");
        for (const auto stride : value.strides()) {
            check(stride >= 0, name, " must not have negative strides");
        }
    }

    std::vector<int64_t> integers(const at::Tensor& value) const {
        check(values_enabled_, "internal error: tensor value read during capture");
        TORCH_CHECK(values_enabled_, "PLA_SANITY_CHECK: value reads require eager execution");
        check(value.scalar_type() == at::kInt || value.scalar_type() == at::kLong,
              "index tensor must have int32 or int64 dtype");
        const auto host = value.to(at::TensorOptions().device(at::kCPU).dtype(at::kLong))
                              .contiguous();
        if (host.numel() == 0) return {};
        const auto* data = host.data_ptr<int64_t>();
        return {data, data + host.numel()};
    }

    void finite(const at::Tensor& value, const char* name) const {
        if (!values_enabled_) return;
        // Only callers whose contract excludes NaN/Inf should request this.
        const auto host = value.to(at::kCPU);
        check(at::isfinite(host).all().item<bool>(), name, " must contain finite values");
    }

private:
    const char* operation_;
    bool enabled_;
    bool values_enabled_ = false;
    int device_ = -1;
};

}  // namespace sanity
}  // namespace pla
