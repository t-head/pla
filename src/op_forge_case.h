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
#include <hggc_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <utility>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Runtime case dumps intentionally use the same serialization contract as
// OperatorsForge/FlashLinearAttn/GeneralizedUTLists.  Keeping the serializer in
// one lightweight header lets the independently-built decode, FlashQLA and
// FlashKDA extensions emit byte-for-byte compatible --format strings.

static inline std::string get_dtype_string(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::ScalarType::BFloat16:
            return "bf16";
        case at::ScalarType::Half:
            return "fp16";
        case at::ScalarType::Float:
            return "fp32";
        default:
            return "unknown";
    }
}

static inline std::vector<int64_t> op_forge_cu_seqlens_to_host(
    const at::Tensor& cu_seqlens
) {
    TORCH_CHECK(cu_seqlens.defined(), "cu_seqlens must be defined");
    TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be one-dimensional");
    TORCH_CHECK(
        cu_seqlens.scalar_type() == at::ScalarType::Int ||
            cu_seqlens.scalar_type() == at::ScalarType::Long,
        "cu_seqlens must be int32 or int64"
    );

    auto host = cu_seqlens.to(
        at::TensorOptions().device(at::kCPU).dtype(at::kLong)
    ).contiguous();
    const auto* values = host.data_ptr<int64_t>();
    return std::vector<int64_t>(values, values + host.numel());
}

// Whether capture is active or cannot be ruled out for the launch stream.
// A dump has to know this: reading cu_seqlens back is a D2H copy, and
// torch rejects those outright mid-capture ("Cannot copy between CPU and CUDA
// tensors during CUDA graph capture unless the CPU tensor is pinned").  Serving
// stacks capture the decode step, so an unguarded read turns
// PPU_LIB_PERF_INSTRUMENT=1 into a hard model failure.  Instrumentation is
// observational and must never do that.
//
// Torch exposes its stream through the CUDA-compatible ATen API, while the
// cuda_free runtime surface uses HGGC names. The handles are ABI-compatible but
// distinct opaque C++ types, so cross that boundary exactly once here.
// A context always exists by the time a kernel entry point serializes its
// arguments, so nothing is initialised here that would not be anyway.
static inline bool op_forge_stream_is_capturing() {
    hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
    const auto stream = reinterpret_cast<hggcStream_t>(
        at::cuda::getCurrentCUDAStream().stream());
    if (hggcStreamIsCapturing(stream, &status) != hggcSuccess) {
        // Clear the sticky error: a failed probe must not resurface as an
        // unrelated launch failure later on. Failure never grants permission
        // for D2H: emit a shapes-only record when capture status is unknown.
        (void)hggcGetLastError();
        return true;
    }
    return status != hggcStreamCaptureStatusNone;
}

static inline std::string op_forge_float_string(float value) {
    std::ostringstream stream;
    if (std::isfinite(value) && std::floor(value) == value) {
        stream << std::fixed << std::setprecision(1) << value;
    } else {
        stream << std::setprecision(9) << value;
    }
    return stream.str();
}

static inline std::string op_forge_offsets_string(
    const std::vector<int64_t>& offsets
) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (i != 0) stream << ",";
        stream << offsets[i];
    }
    stream << "]";
    return stream.str();
}

// Generate a complete OperatorsForge case string.
//
// A cu_seqlens vector containing anything other than one token per request is
// serialized as a packed prefill case.  This keeps the common decode path as
// [D] when frameworks pass [0,1,...,B], while preserving the exact offsets for
// varlen input.  Dedicated prefill APIs pass force_prefill=true; a fixed-length
// single-sequence call with no cu_seqlens is then represented as [0,T].
//
// Under graph capture the offsets are unreadable (see
// op_forge_stream_is_capturing) and are left out: the record then carries only
// the shapes, which is exactly right for the captured decode step and merely
// incomplete for anything else.
static inline std::string generate_nvtx_tag(
    int B, int T, int K, int V, int H, int HV,
    const std::string& dtype_str,
    bool is_kda = false,
    const at::Tensor* cu_seqlens = nullptr,
    float gate_lower_bound = -5.0f,
    bool force_prefill = false
) {
    std::vector<int64_t> offsets;
    const bool have_cu_seqlens = cu_seqlens != nullptr && cu_seqlens->defined();
    if (have_cu_seqlens && op_forge_stream_is_capturing()) {
        TORCH_WARN_ONCE(
            "PLA instrumentation: CUDA graph capture is active or its status "
            "cannot be determined, so case records "
            "omit cu_seqlens_q and describe the call by its shapes only. Run "
            "with graph capture disabled to get replayable varlen dumps."
        );
    } else if (have_cu_seqlens) {
        offsets = op_forge_cu_seqlens_to_host(*cu_seqlens);
        TORCH_CHECK(offsets.size() >= 2, "cu_seqlens must contain at least two offsets");
        TORCH_CHECK(offsets.front() == 0, "cu_seqlens must start at zero");
        TORCH_CHECK(
            static_cast<int>(offsets.size()) == B + 1,
            "cu_seqlens batch size does not match B"
        );
        for (size_t i = 1; i < offsets.size(); ++i) {
            TORCH_CHECK(offsets[i] >= offsets[i - 1],
                        "cu_seqlens must be nondecreasing");
            if (offsets[i] - offsets[i - 1] != 1) force_prefill = true;
        }
    } else if (force_prefill) {
        TORCH_CHECK(B == 1,
                    "fixed prefill without cu_seqlens only supports batch size 1");
        offsets = {0, T};
    }

    const std::string tag = is_kda ? "KDA" : "GDN";
    const std::string op = is_kda ? "kda_attn" : "gdn_attn";

    std::string result = "[" + tag + "]:";
    if (force_prefill) {
        // The requested GeneralizedUTLists are generated with cur_seqlen=200;
        // this metadata is descriptive and is ignored by the --format parser.
        result += "[P][" + std::to_string(B) + ",200]" + op + ":";
    } else {
        result += "[D][" + std::to_string(B) + "]" + op + ":seqlen_q:" +
                  std::to_string(T) + ",";
    }

    result += "head_dim_qk:" + std::to_string(K) +
              ",head_dim_v:" + std::to_string(V) +
              ",num_heads_qk:" + std::to_string(H) +
              ",num_heads_v:" + std::to_string(HV) +
              ",hidden_act:silu,conv_kdim:4,";
    if (is_kda) {
        result += "gate_lower_bound:" + op_forge_float_string(gate_lower_bound) + ",";
    }
    result += "batch_size:" + std::to_string(B) + ",dtype:" + dtype_str;
    if (force_prefill && !offsets.empty()) {
        result += ",cu_seqlens_q:" + op_forge_offsets_string(offsets);
    }
    result += ".";
    return result;
}

namespace pla {
namespace instrument {

// Translation units of one extension share the snapshot; symbols are not merged with other independent extensions.
inline __attribute__((visibility("hidden"))) int show_params() {
    static const int value = [] {
        const char* env = std::getenv("PPU_LIB_SHOW_PARAMS");
        return env ? std::atoi(env) : 0;
    }();
    return value;
}

inline void check_cuda(hggcError_t result, const char* operation) {
    TORCH_CHECK(result == hggcSuccess, "PLA_BYPASS_EXEC: ", operation,
                ": ", hggcGetErrorString(result));
}

inline void dump_metadata(
    const char* operation,
    std::initializer_list<std::pair<const char*, const at::Tensor*>> tensors,
    const std::string& scalars = "") {
    if (!show_params()) return;
    std::ostringstream line;
    line << "[PLA_BYPASS] op=" << operation;
    for (const auto& item : tensors) {
        line << " " << item.first << "=";
        if (!item.second || !item.second->defined()) {
            line << "None";
        } else {
            line << "{shape:" << item.second->sizes() << ",stride:" << item.second->strides()
                 << ",dtype:" << item.second->scalar_type() << ",device:"
                 << item.second->device() << "}";
        }
    }
    if (!scalars.empty()) line << " " << scalars;
    std::cout << line.str() << std::endl;
}

// Do not use Tensor::to(CPU, int64) or CUDA contiguous(): either can launch a
// conversion/packing kernel. Explicit copies also handle strided cu metadata.
inline at::Tensor host_offsets(const at::Tensor& cu) {
    TORCH_CHECK(cu.dim() == 1 && (cu.scalar_type() == at::kInt || cu.scalar_type() == at::kLong),
                "cu must be a one-dimensional integer tensor");
    if (!cu.is_cuda()) return cu.to(at::kLong).contiguous();
    // An invalid cross-device cu must not enable D2H while the caller's
    // current stream is capturing on a different device.
    TORCH_CHECK(!op_forge_stream_is_capturing(), "cu unavailable during capture");
    const c10::cuda::CUDAGuard guard(cu.device());
    const auto stream = reinterpret_cast<hggcStream_t>(
        at::cuda::getCurrentCUDAStream(cu.get_device()).stream());
    hggcStreamCaptureStatus status = hggcStreamCaptureStatusNone;
    check_cuda(hggcStreamIsCapturing(stream, &status), "query dump capture status");
    TORCH_CHECK(status == hggcStreamCaptureStatusNone, "cu unavailable during capture");
    auto host = at::empty({cu.numel()}, cu.options().device(at::kCPU).pinned_memory(true));
    if (cu.numel() != 0) {
        const auto bytes = cu.element_size();
        if (cu.stride(0) == 1) {
            check_cuda(hggcMemcpyAsync(host.data_ptr(), cu.data_ptr(), cu.numel() * bytes,
                                       hggcMemcpyDeviceToHost, stream), "copy cu");
        } else {
            TORCH_CHECK(cu.stride(0) >= 0, "negative cu stride");
            for (int64_t i = 0; i < cu.numel(); ++i)
                check_cuda(hggcMemcpyAsync(static_cast<char*>(host.data_ptr()) + i * bytes,
                    static_cast<const char*>(cu.data_ptr()) + i * cu.stride(0) * bytes,
                    bytes, hggcMemcpyDeviceToHost, stream), "copy strided cu");
        }
        check_cuda(hggcStreamSynchronize(stream), "finish cu copy");
    }
    return host.to(at::kLong);
}

inline void dump_case(int B, int T, int K, int V, int H, int HV,
    const std::string& dtype, bool is_kda, const at::Tensor* cu = nullptr,
    float lower_bound = -5.0f, bool force_prefill = false) {
    if (!show_params()) return;
    at::Tensor host;
    bool unavailable = false;
    if (cu && cu->defined()) {
        if (cu->is_cuda() && op_forge_stream_is_capturing()) {
            // Capture is an expected dump limitation, not an exception. Avoid
            // constructing/printing a C++ backtrace for every captured layer.
            unavailable = true;
            std::cout << "[PLA_BYPASS] cu_unavailable=during capture" << std::endl;
        } else {
            try {
                host = host_offsets(*cu);
            } catch (const c10::Error& error) {
                unavailable = true;
                std::cout << "[PLA_BYPASS] cu_unavailable="
                          << error.what_without_backtrace() << std::endl;
            } catch (const std::exception& error) {
                unavailable = true;
                std::cout << "[PLA_BYPASS] cu_unavailable=" << error.what() << std::endl;
            }
        }
    }
    try {
        // Passing CPU metadata avoids additional GPU copies. Capture emits a
        // shapes-only record, explicitly marked above as incomplete.
        auto tag = generate_nvtx_tag(B, T, K, V, H, HV, dtype, is_kda,
            host.defined() ? &host : nullptr, lower_bound,
            force_prefill && !unavailable);
        if (force_prefill && unavailable) {
            // Keep the API's prefill classification without inventing offsets
            // from a padded token capacity or requesting a capture-time copy.
            tag = std::string(is_kda ? "[KDA]:[P][" : "[GDN]:[P][") +
                std::to_string(B) + ",200]" + (is_kda ? "kda_attn:" : "gdn_attn:") +
                tag.substr(tag.find("head_dim_qk:"));
        }
        std::cout << tag << std::endl;
    } catch (const std::exception&) {
        std::cout << "[PLA_BYPASS] invalid_case B=" << B << " T=" << T
                  << " K=" << K << " V=" << V << " H=" << H << " HV=" << HV
                  << " dtype=" << dtype << " is_kda=" << is_kda;
        if (host.defined()) {
            const auto* values = host.data_ptr<int64_t>();
            std::vector<int64_t> offsets;
            if (host.numel()) offsets.assign(values, values + host.numel());
            std::cout << " raw_cu=" << op_forge_offsets_string(offsets);
        }
        std::cout << " (not a valid replayable UT)" << std::endl;
    }
}

}  // namespace instrument
}  // namespace pla
