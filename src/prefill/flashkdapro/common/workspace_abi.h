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
 * FlashKDApro private K1 -> K2 workspace contract.
 *
 * The host selects one layout after all schedule/override decisions are known
 * and passes the same immutable value to both launchers.  Keep this header
 * independent of CUDA runtime details so host policy and device launch code
 * share one ABI definition.
 */
#pragma once

#include <cstdint>

namespace flashkda {

// One private workspace slot per (head, CH=16 tile), in BF16 allocation
// units.  Float fields occupy the same byte image and therefore count as two
// BF16 units per element.  Host allocation, K1 producers, K2 consumers, and CP
// descriptors all include this header so the layout cannot drift by TU.
inline constexpr int WS_CHUNK = 16;
inline constexpr int WS_DK = 128;
inline constexpr int WS_KD_U = WS_CHUNK * WS_DK;       // 2048
inline constexpr int WS_INV_U = WS_CHUNK * WS_CHUNK;   // 256
inline constexpr int WS_GT_U = 256;                    // fp32[128]
inline constexpr int WS_OFF_KD_U = 0;
inline constexpr int WS_OFF_QD_U = 2048;
inline constexpr int WS_OFF_KR_U = 4096;
inline constexpr int WS_OFF_INV_U = 6144;
inline constexpr int WS_OFF_MQK_U = 6400;
inline constexpr int WS_OFF_GT_U = 6656;
inline constexpr int WS_OFF_BETA_U = 6912;             // fp32[16]
inline constexpr int WS_SLOT_UNITS = 7040;
inline constexpr int WS_SLOT_BYTES = WS_SLOT_UNITS * 2;
static_assert(WS_OFF_QD_U == WS_OFF_KD_U + WS_KD_U);
static_assert(WS_OFF_KR_U == WS_OFF_QD_U + WS_KD_U);
static_assert(WS_OFF_INV_U == WS_OFF_KR_U + WS_KD_U);
static_assert(WS_OFF_MQK_U == WS_OFF_INV_U + WS_INV_U);
static_assert(WS_OFF_GT_U == WS_OFF_MQK_U + WS_INV_U);
static_assert(WS_OFF_BETA_U == WS_OFF_GT_U + WS_GT_U);
static_assert(WS_SLOT_UNITS >= WS_OFF_BETA_U + 2 * WS_CHUNK);

// CP segment-table fields.  WSBASE indexes the same workspace slot ABI above.
inline constexpr int KDA_SEG_STRIDE = 4;
inline constexpr int KDA_SEG_TOK = 0;
inline constexpr int KDA_SEG_LEN = 1;
inline constexpr int KDA_SEG_WSBASE = 2;
inline constexpr int KDA_SEG_OUTSLOT = 3;

// BT64-aligned CP segment descriptor.  The BT64 slot base is an independent
// exclusive scan of ceil(seq_len/64) and cannot be derived from the CH16
// WSBASE across ragged sequence tails, so the composed BT64+CP path carries
// it explicitly.  TOK/LEN/WSBASE duplicate the matching KDA_SEG_* entry;
// both tables are built from the same segmentation and never disagree.
inline constexpr int KDA_SEG_BT_STRIDE = 5;
inline constexpr int KDA_SEG_BT_TOK = 0;
inline constexpr int KDA_SEG_BT_LEN = 1;
inline constexpr int KDA_SEG_BT_WSBASE = 2;
inline constexpr int KDA_SEG_BT_BTBASE = 3;
inline constexpr int KDA_SEG_BT_OUTSLOT = 4;

// The two normal layouts keep the established tc02 pre-swizzle or tc01
// [CH,DK] image; the two tc01 contiguous layouts expose the state-update A
// matrix as row-major [DK,CH].  In both physical representations the enum also
// freezes the value ABI: raw kr/INV/Mqk/beta versus affine
// P=kr^T*INV*diag(beta) and O=Mqk*INV*diag(beta).
enum class KrWorkspaceLayout : int {
    kNormalRaw = 0,
    kNormalAffineP = 1,
    kContiguousRawKrT = 2,
    kContiguousAffineP = 3,
};

}  // namespace flashkda
