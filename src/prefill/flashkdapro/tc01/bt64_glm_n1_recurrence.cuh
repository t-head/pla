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

// Default entry for GLM5.3 TP8 N1; the original recurrence is kept for every non-target path.
#pragma once
#if defined(FLASHKDA_ARCH_100)
// The persistent state uses [BV,DK]; each row is XOR-permuted in units of 8-element vectors.
// No TSM is added; B-fragment reads along contiguous K need no per-read shuffle.
__device__ __forceinline__ int recurrence_state_offset_tc01(int n, int k) {
    return n * DK + (k ^ ((n & 7) << 3));
}

__device__ __forceinline__ void recurrence_load_state_t_tc01(
    const BF16* state, int kb, int lane, uint32_t regs[4]) {
    static_assert(BV == 16 && DK == 128, "stateT only supports BV16/DK128");
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int n = (lane >> 2) + 8 * (i >> 1);
        const int k = kb * BC + 2 * (lane & 3) + 8 * (i & 1);
        regs[i] = *reinterpret_cast<const uint32_t*>(
            state + recurrence_state_offset_tc01(n, k));
    }
}

__device__ __forceinline__ void recurrence_checkpoint_state_t_tc01(
    const BF16* state, BF16* checkpoint, int gbt, int H,
    int head, int dv_base, int tid) {
    CUTE_UNROLL
    for (int vec = tid; vec < BV * DK / 8; vec += THREADS) {
        const int n = vec / (DK / 8);
        const int k = (vec % (DK / 8)) * 8;
        const int64_t off =
            (int64_t(gbt * H + head) * DV + dv_base + n) * DK + k;
        *reinterpret_cast<uint4*>(checkpoint + off) =
            *reinterpret_cast<const uint4*>(state + recurrence_state_offset_tc01(n, k));
    }
}

template <typename Acc>
__device__ __forceinline__ void recurrence_store_state_t_acc_tc01(
    Acc& acc, BF16* state, float* gate, int mb, int nb, int lane) {
    fk::TiledMma1WarpBf16 mma;
    auto thr_mma = mma.get_thread_slice(lane);
    auto identity = make_identity_tensor(make_shape(_16{}, _16{}));
    auto coords = thr_mma.partition_C(identity);
    Tensor rC = recast<float>(acc);
    CUTE_STATIC_ASSERT_V(cute::size<0>(coords) == cute::Int<8>{});
    CUTE_STATIC_ASSERT_V(cute::size<1>(coords) == cute::Int<1>{});
    CUTE_STATIC_ASSERT_V(cute::size<2>(coords) == cute::Int<1>{});
    BF16 prior[8];
    float gates[8];
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
        const auto coord = coords(v, 0, 0);
        const int k = mb * BC + int(cute::get<0>(coord));
        const int n = nb * BC + int(cute::get<1>(coord));
        prior[v] = state[recurrence_state_offset_tc01(n, k)];
        gates[v] = gate[k];
    }
    asm volatile("" ::: "memory");
    uint32_t c_regs[4] = {};
    CUTE_UNROLL
    for (int v = 0; v < 8; ++v) {
        const float old = fk::bf16_to_f32(prior[v]);
        const BF16 rounded = BF16(old * gates[v] + rC(v, 0, 0));
        c_regs[v / 2] |= uint32_t(rounded.storage) << (16 * (v & 1));
    }
    // The writer switches from C ownership to B ownership; first make sure all old-value reads of this warp are done.
    __syncwarp();
    uint32_t b_regs[4];
    fk::bf16_cfrag_to_bfrag_layout_u32(c_regs, b_regs, lane);
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int n = nb * BC + (lane >> 2) + 8 * (i >> 1);
        const int k = mb * BC + 2 * (lane & 3) + 8 * (i & 1);
        *reinterpret_cast<uint32_t*>(state + recurrence_state_offset_tc01(n, k)) = b_regs[i];
    }
}
#endif

__global__ void __launch_bounds__(THREADS, 1) flashkda_bt64_recurrence_n1_kernel(
    const BF16* __restrict__ ws16,
    const BF16* __restrict__ bt_ws,
    BF16* __restrict__ x_global,
    BF16* __restrict__ h_checkpoint,
    const void* __restrict__ init_state,
    void* __restrict__ final_state,
    const int32_t* __restrict__ init_state_indices,
    const int32_t* __restrict__ final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* __restrict__ cu,
    bool has_state_in, bool has_state_out) {
    const int dv_tile = int(blockIdx.x);
    const int seq = int(blockIdx.y);
    const int head = int(blockIdx.z);
    const int tid = int(threadIdx.x);
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    constexpr int NWARPS = THREADS / WARP;
    if (seq >= N) return;
    const int dv_base = dv_tile * BV;
    const SeqMap smap = map_sequence(seq, cu);

    extern __shared__ __align__(128) unsigned char smem_raw[];
    RecurrenceSmem& sm = *reinterpret_cast<RecurrenceSmem*>(smem_raw);

    const int init_slot = init_state_indices ? init_state_indices[seq] : seq;
    for (int i = tid; i < DK * BV; i += THREADS) {
#if defined(FLASHKDA_ARCH_100)
        const int lc = i / DK;
        const int dk = i % DK;
#else
        const int dk = i / BV;
        const int lc = i - dk * BV;
#endif
        const int dv = dv_base + lc;
        BF16 value = BF16(0.f);
        if (has_state_in && init_state != nullptr) {
            const int64_t off =
                (int64_t(init_slot) * H + head) * DV * DK
                + int64_t(dv) * DK + dk;
            value = init_state_is_fp32
                ? BF16(reinterpret_cast<const float*>(init_state)[off])
                : reinterpret_cast<const BF16*>(init_state)[off];
        }
#if defined(FLASHKDA_ARCH_100)
        sm.state[recurrence_state_offset_tc01(lc, dk)] = value;
#else
        sm.state[i] = value;
#endif
    }
    __syncthreads();

    for (int local_bt = 0; local_bt < smap.nbt; ++local_bt) {
        const int gbt = smap.bt_base + local_bt;
        if (gbt >= total_bt) break;
        const int tok = smap.tok + local_bt * BT;
        const int remain = smap.len - local_bt * BT;
        const int bt_len = remain < BT ? remain : BT;
        const int nsub = ceil_div_pos(bt_len, BC);
        const int ws_base = smap.ws_base + local_bt * NS;
        const BF16* bt_slot = bt_ws
            + int64_t(gbt * H + head) * fk::BT64_WS_SLOT_UNITS;

        // Output consumes the state at the start of this BT. Store exactly the
        // BF16 image used by this recurrence, transposed to [DV,DK].
#if defined(FLASHKDA_ARCH_100)
        recurrence_checkpoint_state_t_tc01(
            sm.state, h_checkpoint, gbt, H, head, dv_base, tid);
#else
        for (int i = tid; i < DK * BV; i += THREADS) {
            const int dk = i / BV;
            const int lc = i - dk * BV;
            const int dv = dv_base + lc;
            const int64_t off =
                (int64_t(gbt * H + head) * DV + dv) * DK + dk;
            h_checkpoint[off] = sm.state[i];
        }
#endif

        // x = BF16(FP32(U) - FP32(W @ state)).  Stage the already-BF16 U
        // image directly in x, then let each MMA C-fragment owner apply the
        // FP32 subtraction and the single BF16 materialization in place.  The
        // partition_C view is the selected tc01 atom's ownership map; no
        // architecture-specific lane coordinates are assumed here.
#if defined(FLASHKDA_ARCH_100)
        if (tid == 0) {
            recurrence_aiu_load_w_tc01(
                sm.matrix, bt_slot + fk::BT64_WS_OFF_W_U);
            recurrence_aiu_commit_tc01();
        }
#else
        for (int i = tid; i < BT * DK; i += THREADS)
            sm.matrix[i] = bt_slot[fk::BT64_WS_OFF_W_U + i];
#endif
#if defined(FLASHKDA_ARCH_100)
        recurrence_copy_u_vec16_tc01(sm.x, bt_slot, dv_base, tid);
#else
        for (int i = tid; i < BT * BV; i += THREADS) {
            const int r = i / BV;
            const int lc = i - r * BV;
            sm.x[i] = bt_slot[
                fk::BT64_WS_OFF_U_U + r * DV + dv_base + lc];
        }
#endif
#if defined(FLASHKDA_ARCH_100)
        // Overlap W's disjoint AIU transfer with the vec16 U copy above,
        // then publish both arrays with the original block barrier.
        if (tid == 0) recurrence_aiu_wait_all_tc01();
#endif
        __syncthreads();

        {
#if !defined(FLASHKDA_ARCH_100)
            auto Wv = make_tensor(make_smem_ptr(sm.matrix),
                make_layout(make_shape(Int<BT>{}, Int<DK>{}),
                            make_stride(Int<DK>{}, _1{})));
#endif
            auto StateBv = make_tensor(make_smem_ptr(sm.state),
                make_layout(make_shape(Int<BV>{}, Int<DK>{}),
                            make_stride(_1{}, Int<BV>{})));
            constexpr int X_MB = BT / BC;
            constexpr int X_NB = BV / BC;
            constexpr int X_KB = DK / BC;
#if defined(FLASHKDA_ARCH_100)
            static_assert(NWARPS == 4 && X_MB == 4 && X_NB == 1,
                          "tc01 BV16 W@state needs four warps and a 4x1 tile");
            // Each warp owns one M tile; the K accumulation order of every element is preserved.
            const int nb = 0;
            const int mb = warp;
            auto acc = fk::make_acc16_bf16_1warp(lane);
            cute::clear(acc);
            CUTE_UNROLL
            for (int kb = 0; kb < X_KB; ++kb) {
                uint32_t b_regs[4];
                recurrence_load_state_t_tc01(
                    sm.state, kb, lane, b_regs);
                uint32_t a_regs[4];
                recurrence_load_w_nontrans_tc01(
                    sm.matrix, mb * BC, kb * BC, a_regs);
                fk::gemm16_rr_1warp_bf16_acc(a_regs, b_regs, acc);
                // Keep the compiler scheduling boundary of the validated layout, without adding device synchronization.
                asm volatile("" ::: "memory");
            }
            recurrence_store_x_acc_tc01(acc, sm.x, mb, nb, lane);
#else
            for (int blk = warp; blk < X_MB * X_NB; blk += NWARPS) {
                const int mb = blk / X_NB;
                const int nb = blk - mb * X_NB;
                auto acc = fk::make_acc16_bf16_1warp(lane);
                cute::clear(acc);
                CUTE_UNROLL
                for (int kb = 0; kb < X_KB; ++kb) {
                    auto Ablk = local_tile(
                        Wv, make_shape(_16{}, _16{}), make_coord(mb, kb));
                    auto Bblk = local_tile(
                        StateBv, make_shape(_16{}, _16{}), make_coord(nb, kb));
                    fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
                }

                auto Xtile = make_tensor(
                    make_smem_ptr(sm.x + mb * BC * BV + nb * BC),
                    make_layout(make_shape(_16{}, _16{}),
                                make_stride(Int<BV>{}, _1{})));
                fk::TiledMma1WarpBf16 mma;
                auto thr_mma = mma.get_thread_slice(lane);
                Tensor tCsX = thr_mma.partition_C(Xtile);
                Tensor rC = recast<float>(acc);
                CUTE_UNROLL
                for (int m = 0; m < cute::size<1>(tCsX); ++m)
                    CUTE_UNROLL
                    for (int n = 0; n < cute::size<2>(tCsX); ++n)
                        CUTE_UNROLL
                        for (int v = 0; v < cute::size<0>(tCsX); ++v) {
                            const float u = fk::bf16_to_f32(tCsX(v, m, n));
                            tCsX(v, m, n) = BF16(u - rC(v, m, n));
                        }
            }
#endif
        }
        __syncthreads();
        for (int i = tid; i < BT * BV; i += THREADS) {
            const int r = i / BV;
            const int lc = i - r * BV;
            if (r < bt_len && tok + r < T_total) {
                const int64_t off = int64_t(tok + r) * H * DV
                    + int64_t(head) * DV + dv_base + lc;
                x_global[off] = sm.x[i];
            }
        }
        __syncthreads();

        // state = diag(product gt) * state + Kg^T @ x.  On tc01, issue the
        // one new native Kg transfer before the unchanged gate loop and wait
        // after it, so the asynchronous copy can overlap only existing scalar
        // work.  W/state/U/K-loop/MMA order and BF16 materialization are kept.
#if defined(FLASHKDA_ARCH_100)
        if (tid == 0) {
            recurrence_aiu_load_w_tc01(
                sm.matrix, bt_slot + fk::BT64_WS_OFF_KG_U);
            recurrence_aiu_commit_tc01();
        }
#endif
        for (int dk = tid; dk < DK; dk += THREADS) {
#if defined(FLASHKDA_ARCH_100)
            // Four independent reads precede the multiply that consumes them; the tail block only touches CH16 slots that actually exist.
            const float g0 = reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base) + fk::WS_OFF_GT_U)[dk];
            const float g1 = nsub > 1 ? reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base + 1) + fk::WS_OFF_GT_U)[dk] : 1.f;
            const float g2 = nsub > 2 ? reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base + 2) + fk::WS_OFF_GT_U)[dk] : 1.f;
            const float g3 = nsub > 3 ? reinterpret_cast<const float*>(k1_slot(
                ws16, total_tiles, head, ws_base + 3) + fk::WS_OFF_GT_U)[dk] : 1.f;
            float g = 1.f;
            g *= g0;
            g *= g1;
            g *= g2;
            g *= g3;
#else
            float g = 1.f;
            for (int sub = 0; sub < nsub; ++sub) {
                const BF16* slot = k1_slot(
                    ws16, total_tiles, head, ws_base + sub);
                g *= reinterpret_cast<const float*>(
                    slot + fk::WS_OFF_GT_U)[dk];
            }
#endif
            sm.gate[dk] = g;
        }
#if defined(FLASHKDA_ARCH_100)
        if (tid == 0) recurrence_aiu_wait_all_tc01();
#else
        for (int i = tid; i < BT * DK; i += THREADS)
            sm.matrix[i] = bt_slot[fk::BT64_WS_OFF_KG_U + i];
#endif
        __syncthreads();
        {
#if !defined(FLASHKDA_ARCH_100)
            auto KgTv = make_tensor(make_smem_ptr(sm.matrix),
                make_layout(make_shape(Int<DK>{}, Int<BT>{}),
                            make_stride(_1{}, Int<DK>{})));
#endif
            auto Xbv = make_tensor(make_smem_ptr(sm.x),
                make_layout(make_shape(Int<BV>{}, Int<BT>{}),
                            make_stride(_1{}, Int<BV>{})));
            constexpr int STATE_MB = DK / BC;
            constexpr int STATE_NB = BV / BC;
            constexpr int STATE_KB = BT / BC;
#if defined(FLASHKDA_ARCH_100)
            static_assert(NWARPS == 4 && STATE_MB == 8 && STATE_NB == 1,
                          "tc01 BV16 state needs four warps and an 8x1 tile");
            const int nb = 0;
            constexpr int STATE_GROUPS = 1;
            // Keep the two groups sequential in the generated program so the
            // second pair cannot extend the first pair's accumulator liveness.
            // This is the occupancy-control variable of partial2.
#pragma unroll 1
            for (int group = 0; group < STATE_GROUPS; ++group) {
                const int mb0 = warp + group * 8;
                const int mb1 = mb0 + 4;
                auto acc0 = fk::make_acc16_bf16_1warp(lane);
                auto acc1 = fk::make_acc16_bf16_1warp(lane);
                cute::clear(acc0);
                cute::clear(acc1);
                CUTE_UNROLL
                for (int kb = 0; kb < STATE_KB; ++kb) {
                    uint32_t b_regs[4];
                    recurrence_load_b_vec32_shuffle_tc01(
                        sm.x, nb, kb, lane, b_regs);
                    uint32_t a_regs[4];
                    recurrence_load_kg_trans_tc01(
                        sm.matrix, mb0 * BC, kb * BC, a_regs);
                    fk::gemm16_rr_1warp_bf16_acc(
                        a_regs, b_regs, acc0);
                    recurrence_load_kg_trans_tc01(
                        sm.matrix, mb1 * BC, kb * BC, a_regs);
                    fk::gemm16_rr_1warp_bf16_acc(
                        a_regs, b_regs, acc1);
                }
                recurrence_store_state_t_acc_tc01(
                    acc0, sm.state, sm.gate, mb0, nb, lane);
                recurrence_store_state_t_acc_tc01(
                    acc1, sm.state, sm.gate, mb1, nb, lane);
            }
#else
            for (int blk = warp;
                 blk < STATE_MB * STATE_NB; blk += NWARPS) {
                const int mb = blk / STATE_NB;
                const int nb = blk - mb * STATE_NB;
                auto acc = fk::make_acc16_bf16_1warp(lane);
                cute::clear(acc);
                CUTE_UNROLL
                for (int kb = 0; kb < STATE_KB; ++kb) {
                    auto Ablk = local_tile(
                        KgTv, make_shape(_16{}, _16{}), make_coord(mb, kb));
                    auto Bblk = local_tile(
                        Xbv, make_shape(_16{}, _16{}), make_coord(nb, kb));
                    fk::gemm16_ss_1warp_bf16_acc(Ablk, Bblk, lane, acc);
                }

                auto StateTile = make_tensor(
                    make_smem_ptr(sm.state + mb * BC * BV + nb * BC),
                    make_layout(make_shape(_16{}, _16{}),
                                make_stride(Int<BV>{}, _1{})));
                auto GateTile = make_tensor(make_smem_ptr(sm.gate + mb * BC),
                    make_layout(make_shape(_16{}, _16{}),
                                make_stride(_1{}, _0{})));
                fk::TiledMma1WarpBf16 mma;
                auto thr_mma = mma.get_thread_slice(lane);
                Tensor tCsState = thr_mma.partition_C(StateTile);
                Tensor tCsGate = thr_mma.partition_C(GateTile);
                Tensor rC = recast<float>(acc);
                CUTE_UNROLL
                for (int m = 0; m < cute::size<1>(tCsState); ++m)
                    CUTE_UNROLL
                    for (int n = 0; n < cute::size<2>(tCsState); ++n)
                        CUTE_UNROLL
                        for (int v = 0; v < cute::size<0>(tCsState); ++v) {
                            const float old =
                                fk::bf16_to_f32(tCsState(v, m, n));
                            tCsState(v, m, n) = BF16(
                                old * tCsGate(v, m, n) + rC(v, m, n));
                        }
            }
#endif
        }
        __syncthreads();
    }

    if (has_state_out && final_state != nullptr) {
        const int final_slot = final_state_indices
            ? final_state_indices[seq] : seq;
        for (int i = tid; i < DK * BV; i += THREADS) {
#if defined(FLASHKDA_ARCH_100)
            const int lc = i / DK;
            const int dk = i % DK;
            const BF16 value = sm.state[recurrence_state_offset_tc01(lc, dk)];
#else
            const int dk = i / BV;
            const int lc = i - dk * BV;
            const BF16 value = sm.state[i];
#endif
            const int dv = dv_base + lc;
            const int64_t off =
                (int64_t(final_slot) * H + head) * DV * DK
                + int64_t(dv) * DK + dk;
            if (final_state_is_fp32) {
                reinterpret_cast<float*>(final_state)[off] = fk::bf16_to_f32(value);
            } else {
                reinterpret_cast<BF16*>(final_state)[off] = value;
            }
        }
    }
}
