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

namespace flashkda_cute_nows {
namespace tc01 {

// tc01 A/B output epilogue for the full-DV DK1 recurrence.  DK1 reserves
// exactly 512 bytes of per-warp movmatrix scratch, which is also one row-major
// 16x16 BF16 tile.  Materialize the accumulator there, then let two lanes per
// valid row issue aligned 16-byte stores to global memory.
template <typename Acc>
__forceinline__ __device__ void k2n_store_output_vec(
    Acc const& acc, int lane, int actual, uint32_t* warp_scr,
    cutlass::bfloat16_t* o_head, int64_t row_stride) {
    using BF16 = cutlass::bfloat16_t;
    BF16* s_out = reinterpret_cast<BF16*>(warp_scr);
    flashkda::store_acc16_bf16mma<BF16>(acc, s_out, 16, lane);
    __syncwarp();
    if (lane < 2 * actual) {
        const int row = lane >> 1;
        const int half = lane & 1;
        *reinterpret_cast<uint4*>(
            o_head + int64_t(row) * row_stride + half * 8) =
            *reinterpret_cast<const uint4*>(s_out + row * 16 + half * 8);
    }
    __syncwarp();
}

}  // namespace tc01
}  // namespace flashkda_cute_nows
