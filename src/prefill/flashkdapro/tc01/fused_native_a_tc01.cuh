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

namespace flashkda_cute_bt64 {

// Derived backwards from the current CUTLASS ppu_tsm_ld_swzl_sim; only valid for BF16 64x64.
// Whether it matches the physical AIU output must be measured with an independent tagged/one-hot basis.
__host__ __device__ constexpr int fused_native_a_offset_tc01(int row, int col) {
    const int slice = col / 16;
    const int line = row / 4;
    const int rotate = (((slice & 1) << 1) + ((slice & 2) >> 1)) * 2;
    const int vec = ((((row % 4) * 2 + (col % 16) / 8) ^ (line & 1)) + rotate) & 7;
    return slice * 1024 + line * 64 + vec * 8 + col % 8;
}

__device__ __forceinline__ void fused_write_native_a_tc01(
    BF16* native, const BF16* row_major, int tid) {
    for (int vec = tid; vec < 512; vec += RECOMPUTE_THREADS_TC01) {
        const int i = vec * 8;
        const int offset = fused_native_a_offset_tc01(i / 64, i % 64);
        *reinterpret_cast<uint4*>(native + offset) =
            *reinterpret_cast<const uint4*>(row_major + i);
    }
}

}  // namespace flashkda_cute_bt64
