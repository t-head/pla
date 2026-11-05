// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../op_forge_case.h"
#include <hgtx3/hgtx3.hpp>
#include <optional>

namespace pla {
namespace instrument {

// Keep independent extension snapshots, just like show_params().
inline __attribute__((visibility("hidden"))) int perf_instrument() {
    static const int value = [] {
        const char* env = std::getenv("PPU_LIB_PERF_INSTRUMENT");
        return env ? std::atoi(env) : 0;
    }();
    return value;
}

// Instantiate in the caller's launch scope, not in a short-lived push helper.
// The owned message outlives the range, including exception unwinding.
class PrefillRange {
public:
    PrefillRange(int B, int T, int K, int V, int H, int HV,
                 const std::string& dtype, bool is_kda,
                 const at::Tensor* cu = nullptr, float lower_bound = -5.0f) {
        if (!perf_instrument() && !show_params()) return;
        // generate_nvtx_tag skips device offsets if the current stream is
        // capturing or its capture status is unknown. Never use a side-stream
        // D2H workaround or invent offsets from a padded physical token count.
        tag_ = generate_nvtx_tag(B, T, K, V, H, HV, dtype, is_kda,
                                 cu, lower_bound, true);
        if (show_params()) std::cout << tag_ << std::endl;
        if (perf_instrument()) range_.emplace(tag_.c_str());
    }

    PrefillRange(const PrefillRange&) = delete;
    PrefillRange& operator=(const PrefillRange&) = delete;

private:
    std::string tag_;
    std::optional<hgtx3::scoped_range> range_;
};

}  // namespace instrument
}  // namespace pla
