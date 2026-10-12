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

struct alignas(128) FusedMergeRecomputeSmem {
    BF16 A[BT * BT];
    union {
        struct { BF16 Akk[BT * BT]; BF16 inner[3 * BC * BC]; } merge;
        struct { BF16 native[BT * BT]; BF16 in[BT * RECOMPUTE_BV_TC01]; } recompute;
    } phase;
};
static_assert(sizeof(FusedMergeRecomputeSmem) == 24576, "TSM budget of B drifted");
static_assert(offsetof(FusedMergeRecomputeSmem, phase) == 8192, "native A of B must be aligned");

__global__ void __launch_bounds__(RECOMPUTE_THREADS_TC01, 1)
flashkda_bt64_merge_recompute_kernel(
    const BF16* __restrict__ ws16, BF16* __restrict__ bt_ws,
    const BF16* __restrict__ prepared,
    int H, int N, int total_tiles, int total_bt, const int32_t* __restrict__ cu
#ifdef FLASHKDA_FUSION_STAGE_DUMP
    , BF16* __restrict__ dump_a = nullptr
#endif
    ) {
#if defined(FLASHKDA_ARCH_100)
    const int gbt = int(blockIdx.x), head = int(blockIdx.y), tid = int(threadIdx.x);
    if (gbt >= total_bt) return;
    const BtMap map = map_packed_bt(gbt, cu, N);
    if (!map.valid) return;
    const int nsub = ceil_div_pos(map.len, BC);
    BF16* slot = bt_ws + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;
    extern __shared__ __align__(128) unsigned char raw[];
    auto& sm = *reinterpret_cast<FusedMergeRecomputeSmem*>(raw);
    constexpr int VEC = int(sizeof(uint4) / sizeof(BF16));
    static_assert(VEC == 8 && BC % VEC == 0 && BT % VEC == 0);
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (int vec = tid; vec < BT * BT / VEC; vec += RECOMPUTE_THREADS_TC01) {
        const int i = vec * VEC;
        const int r = i / BT, c = i % BT;
        const int br = r / BC, bc = c / BC;
        if (br == bc) {
            const BF16* k1 = br < nsub
                ? k1_slot(ws16, total_tiles, head, map.ws_base + br) : nullptr;
            const int local = (r % BC) * BC + c % BC;
            if (k1) {
                const uint4 inv = *reinterpret_cast<const uint4*>(k1 + fk::WS_OFF_INV_U + local);
                const uint4 mqk = *reinterpret_cast<const uint4*>(k1 + fk::WS_OFF_MQK_U + local);
                *reinterpret_cast<uint4*>(sm.A + i) = inv;
                *reinterpret_cast<uint4*>(slot + fk::BT64_WS_OFF_AQK_U + i) = mqk;
            } else {
                *reinterpret_cast<uint4*>(sm.A + i) = zero;
                if (r >= c && r < c + VEC) sm.A[i + r - c] = BF16(1.f);
                *reinterpret_cast<uint4*>(slot + fk::BT64_WS_OFF_AQK_U + i) = zero;
            }
        } else {
            *reinterpret_cast<uint4*>(sm.A + i) = zero;
            if (br < bc) *reinterpret_cast<uint4*>(slot + fk::BT64_WS_OFF_AQK_U + i) = zero;
        }
        if (br > bc) {
            const uint4 akk = *reinterpret_cast<const uint4*>(slot + SPLIT_TMP_AKK + i);
            *reinterpret_cast<uint4*>(sm.phase.merge.Akk + i) = akk;
        } else {
            *reinterpret_cast<uint4*>(sm.phase.merge.Akk + i) = zero;
        }
    }
    __syncthreads();
    const int warp = tid / WARP, lane = tid % WARP;
    BF16* inner = sm.phase.merge.inner + (warp < 3 ? warp : 0) * BC * BC;
    BF16* akk = sm.phase.merge.Akk;
    if (warp == 0) merge_offdiag_block_reg<1, 0>(akk, sm.A, inner, lane);
    __syncthreads();
    if (warp == 0) merge_offdiag_block_reg<2, 0>(akk, sm.A, inner, lane);
    else if (warp == 1) merge_offdiag_block_reg<2, 1>(akk, sm.A, inner, lane);
    __syncthreads();
    if (warp == 0) merge_offdiag_block_reg<3, 0>(akk, sm.A, inner, lane);
    else if (warp == 1) merge_offdiag_block_reg<3, 1>(akk, sm.A, inner, lane);
    else if (warp == 2) merge_offdiag_block_reg<3, 2>(akk, sm.A, inner, lane);
    __syncthreads();
    // Switch the union's lifetime only after the complete A is published and all Akk consumers are done.
#ifdef FLASHKDA_FUSION_STAGE_DUMP
    for (int i = tid; i < BT * BT; i += RECOMPUTE_THREADS_TC01)
        dump_a[int64_t(gbt * H + head) * BT * BT + i] = sm.A[i];
#endif
    fused_write_native_a_tc01(sm.phase.recompute.native, sm.A, tid);
    __syncthreads();
    const BF16* input = prepared + int64_t(gbt * H + head) * DV * DK;
    // Keep the old recompute's input loads, four GEMMs, K-loop and U/W store order.
    for (int fb = 0; fb < DV / RECOMPUTE_BV_TC01; ++fb) {
        for (int vec = tid; vec < BT * RECOMPUTE_BV_TC01 / 8;
             vec += RECOMPUTE_THREADS_TC01) {
            const int i = vec * 8;
            const int r = i / RECOMPUTE_BV_TC01;
            const int c = i % RECOMPUTE_BV_TC01;
            *reinterpret_cast<uint4*>(sm.phase.recompute.in + i) =
                *reinterpret_cast<const uint4*>(input + r * DV + fb * RECOMPUTE_BV_TC01 + c);
        }
        __syncthreads();
        recompute_gemm_native_pass_tc01(sm.phase.recompute.native, sm.phase.recompute.in,
            slot + fk::BT64_WS_OFF_U_U + fb * RECOMPUTE_BV_TC01, DV, tid);
        __syncthreads();
    }
    for (int fb = 0; fb < DK / RECOMPUTE_BV_TC01; ++fb) {
        for (int vec = tid; vec < BT * RECOMPUTE_BV_TC01 / 8;
             vec += RECOMPUTE_THREADS_TC01) {
            const int i = vec * 8;
            const int r = i / RECOMPUTE_BV_TC01;
            const int c = i % RECOMPUTE_BV_TC01;
            *reinterpret_cast<uint4*>(sm.phase.recompute.in + i) =
                *reinterpret_cast<const uint4*>(input + BT * DK + r * DK + fb * RECOMPUTE_BV_TC01 + c);
        }
        __syncthreads();
        recompute_gemm_native_pass_tc01(sm.phase.recompute.native, sm.phase.recompute.in,
            slot + fk::BT64_WS_OFF_W_U + fb * RECOMPUTE_BV_TC01, DK, tid);
        __syncthreads();
    }
#endif
}

}  // namespace flashkda_cute_bt64
