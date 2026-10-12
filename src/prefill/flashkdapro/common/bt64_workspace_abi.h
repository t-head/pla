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
 * FlashKDApro tc01 BT=64 private workspace contract.
 *
 * Candidate 1 groups four established CH=16 K1 slots into the same four-stage
 * decomposition used by FLA-Triton:
 *
 *   inter(A, Aqk) -> recompute(W, U, Kg) -> recurrence(x, h) -> output
 *
 * All offsets below are in BF16 allocation units.  The inter-stage A image is
 * dead once recompute has loaded it, so its 64x64 prefix aliases the 64x128 W
 * image.  Recompute must copy all of A to shared memory before writing W.
 * Aqk remains live until output; U and Kg remain live through recurrence.
 *
 * Slot order is [packed_bt, head, BT64_WS_SLOT_UNITS].  A packed BT index is an
 * exclusive scan of ceil(sequence_length / 64).  The corresponding K1 slot is
 * the independent exclusive scan of ceil(sequence_length / 16), plus
 * 4 * local_bt.  They must never be derived from one another across sequence
 * boundaries: the last BT of a ragged sequence can own only 1, 2, or 3 K1
 * slots.  The tc01 kernels deliberately derive both scans from cu_seqlens and
 * synthesize missing sub-chunks instead of reading the next sequence.
 */
#pragma once

#include <cstdint>

namespace flashkda {

inline constexpr int BT64_CHUNK = 64;
inline constexpr int BT64_SUBCHUNK = 16;
inline constexpr int BT64_SUBCHUNKS = BT64_CHUNK / BT64_SUBCHUNK;
inline constexpr int BT64_DK = 128;
inline constexpr int BT64_DV = 128;

inline constexpr int BT64_A_UNITS = BT64_CHUNK * BT64_CHUNK;       // 4096
inline constexpr int BT64_W_UNITS = BT64_CHUNK * BT64_DK;          // 8192
inline constexpr int BT64_AQK_UNITS = BT64_CHUNK * BT64_CHUNK;     // 4096
inline constexpr int BT64_U_UNITS = BT64_CHUNK * BT64_DV;          // 8192
inline constexpr int BT64_KG_UNITS = BT64_CHUNK * BT64_DK;         // 8192

// Phase alias: inter writes A into the first 4096 units; recompute consumes A
// into shared memory and overwrites the complete 8192-unit region with W.
inline constexpr int BT64_WS_OFF_W_OR_A_U = 0;
inline constexpr int BT64_WS_OFF_A_U = BT64_WS_OFF_W_OR_A_U;
inline constexpr int BT64_WS_OFF_W_U = BT64_WS_OFF_W_OR_A_U;
inline constexpr int BT64_WS_OFF_AQK_U =
    BT64_WS_OFF_W_OR_A_U + BT64_W_UNITS;                            // 8192
inline constexpr int BT64_WS_OFF_U_U =
    BT64_WS_OFF_AQK_U + BT64_AQK_UNITS;                             // 12288
inline constexpr int BT64_WS_OFF_KG_U =
    BT64_WS_OFF_U_U + BT64_U_UNITS;                                 // 20480
inline constexpr int BT64_WS_SLOT_UNITS =
    BT64_WS_OFF_KG_U + BT64_KG_UNITS;                               // 28672
inline constexpr int BT64_WS_SLOT_BYTES = BT64_WS_SLOT_UNITS * 2;   // 56 KiB

static_assert(BT64_CHUNK == BT64_SUBCHUNK * BT64_SUBCHUNKS);
static_assert(BT64_WS_OFF_A_U == 0);
static_assert(BT64_A_UNITS <= BT64_W_UNITS);
static_assert(BT64_WS_OFF_AQK_U == BT64_WS_OFF_W_U + BT64_W_UNITS);
static_assert(BT64_WS_OFF_U_U == BT64_WS_OFF_AQK_U + BT64_AQK_UNITS);
static_assert(BT64_WS_OFF_KG_U == BT64_WS_OFF_U_U + BT64_U_UNITS);
static_assert(BT64_WS_SLOT_UNITS == 28672);
static_assert(BT64_WS_SLOT_BYTES == 56 * 1024);

}  // namespace flashkda
