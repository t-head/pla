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

#include <hggc_runtime.h>
#include <stdexcept>

#define DECLARE_KKT_ARCH_LAUNCHER(name) \
void name( \
    const void* K, const float* beta, void* A, \
    const int* cu_seqlens, const int* chunk_indices, \
    int H, int Hg, int num_chunks, int num_tokens, int dtype_tag, \
    int device_index, int num_cu, \
    hggcStream_t stream)

DECLARE_KKT_ARCH_LAUNCHER(launch_kkt_solve_cute_tc01);
DECLARE_KKT_ARCH_LAUNCHER(launch_kkt_solve_cute_tc02);

void launch_kkt_solve_cute(
    const void* K, const float* beta, void* A,
    const int* cu_seqlens, const int* chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens, int dtype_tag,
    int device_index, int num_cu, int compute_capability,
    hggcStream_t stream) {
    decltype(&launch_kkt_solve_cute_tc01) launcher = nullptr;
    if (compute_capability == 80) {
        launcher = launch_kkt_solve_cute_tc01;
    } else if (compute_capability == 89) {
        launcher = launch_kkt_solve_cute_tc02;
    } else {
        throw std::runtime_error(
            "launch_kkt_solve_cute: unsupported compute capability");
    }
    launcher(
        K, beta, A, cu_seqlens, chunk_indices,
        H, Hg, num_chunks, num_tokens, dtype_tag,
        device_index, num_cu, stream);
}

#undef DECLARE_KKT_ARCH_LAUNCHER
