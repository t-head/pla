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

// Bridge between the PyTorch stream API and the PPU-native runtime handle.
//
// PPU ships the same runtime under two parallel spellings:
//
//   libhggcrt.13.0.so  ->  hggcStream_t  (struct HGstream_st*)  -- native
//   libcudart.so       ->  cudaStream_t  (struct CUstream_st*)  -- compat
//
// libcudart is a thin forwarder that dlopens libhggcrt/libhggc, and the handles
// are interchangeable at runtime: a stream created through cudaStreamCreate is
// accepted by hggcStreamQuery and vice versa (both return success on
// ZW-M890P). The two C types are nevertheless distinct opaque structs, so C++
// will not convert between them implicitly.
//
// PyTorch only speaks the compat spelling -- its ATen tree has ATen/cuda/ and no
// ATen/hggc/ -- so every place that takes a stream from torch has to cross that
// boundary exactly once. Keeping the cast in this single helper is what lets the
// rest of the sources stay free of CUDA type names.

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Device.h>
#include <hggc_runtime.h>

namespace pla {

// Current stream of the ambient device, as a native PPU handle.
inline hggcStream_t current_stream() {
    return reinterpret_cast<hggcStream_t>(at::cuda::getCurrentCUDAStream().stream());
}

// Current stream of an explicit device.
inline hggcStream_t current_stream(c10::DeviceIndex device) {
    return reinterpret_cast<hggcStream_t>(
        at::cuda::getCurrentCUDAStream(device).stream());
}

}  // namespace pla
