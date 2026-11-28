// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2026 MoonshotAI
// SPDX-License-Identifier: BSD-3-Clause AND MIT
// Adapted from FlashKDA.

#include "prefill/flashkdapro/common/launch_api.h"
#include <algorithm>

#define ACOMPUTE_VERSION 10500
#include "prefill/flashkdapro/common/tc02/memory.cuh"

namespace flashkda_paired {
using namespace flashkda;

// Inverse of K1's normal Kr workspace swizzle, for a logical [16,128] index.
CUTE_HOST_DEVICE int kr_offset(int logical) {
    int const row = logical / 128, col = logical % 128;
    return (col / 32) * 512 + row * 32 + ((col % 32) ^ (((row >> 1) & 3) << 3));
}

// Prefixes keep adjacent transforms inside each sequence, including empty rows.
__global__ void flashkda_paired_pair_prefix(int32_t const *cu, int T, int N, int *raw_prefix,
                                          int *pair_prefix) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    if (threadIdx.x == 0) {
        int raw = 0, paired = 0;
        raw_prefix[0] = pair_prefix[0] = 0;
        for (int seq = 0; seq < N; ++seq) {
            int const len = cu ? int(cu[seq + 1] - cu[seq]) : T / N;
            raw += (len + 15) / 16;
            paired += (len + 31) / 32;
            raw_prefix[seq + 1] = raw;
            pair_prefix[seq + 1] = paired;
        }
    }
#endif
}

// True FP32 GEMM: the PPU TN atom needs a transposed B view for A @ B.
CUTE_DEVICE void pair_gemm_fp32(float const *a, float const *b, float *output, int lane) {
    using namespace cute;
    auto mma =
        make_tiled_mma(MMA_Atom<MmaF32>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _8>{});
    auto thr = mma.get_slice(lane);
    Tensor ta = make_tensor(make_smem_ptr(a), Layout<Shape<_16, _16>, Stride<Int<20>, _1>>{});
    Tensor tb = make_tensor(make_smem_ptr(b), Layout<Shape<_16, _16>, Stride<_1, Int<20>>>{});
    Tensor tc = make_tensor(make_smem_ptr(output), Layout<Shape<_16, _16>, Stride<Int<20>, _1>>{});
    auto acc = thr.make_fragment_C(thr.partition_C(tc));
    clear(acc);
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i) {
        Tensor aa = local_tile(ta, Shape<_16, _8>{}, make_coord(0, i));
        Tensor bb = local_tile(tb, Shape<_16, _8>{}, make_coord(0, i));
        auto ra = thr.partition_fragment_A(aa);
        auto rb = thr.partition_fragment_B(bb);
        copy(AutoVectorizingCopy{}, thr.partition_A(aa), ra);
        copy(AutoVectorizingCopy{}, thr.partition_B(bb), rb);
        cute::gemm(thr, ra, rb, acc);
    }
    copy(AutoVectorizingCopy{}, acc, thr.partition_C(tc));
    __syncwarp();
}

// Keep transform traffic in aligned packets. Both Kr layouts preserve the
// low three column bits, so each logical eight-element packet is contiguous.
CUTE_DEVICE uint4 scale_packet(uint4 values, float const *gate, bool present = true) {
    auto *words = reinterpret_cast<uint32_t *>(&values);
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const auto x = __bfloat1622float2(
            *reinterpret_cast<const __ppu_bfloat162 *>(words + i));
        const float2 g = present ? *reinterpret_cast<const float2 *>(gate + 2 * i)
                                 : make_float2(1.f, 1.f);
        const auto result = __floats2bfloat162_rn(x.x * g.x, x.y * g.y);
        words[i] = *reinterpret_cast<const uint32_t *>(&result);
    }
    return values;
}

template <bool RowMajor, bool Windowed>
__global__ __launch_bounds__(64) void flashkda_paired_compose_pairs(
    BF16 const *ws, BF16 *ptr,
    float *pgt, BF16 *pinv, BF16 *pmqk, float *pbeta, int const *raw_prefix, int const *pair_prefix,
    int raw_tiles, int pair_tiles, int N, int first_pair) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    using namespace cute;
    int const pair = (Windowed ? first_pair : 0) + blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int const lane = tid % 32, warp = tid / 32;
    if (pair >= pair_prefix[N])
        return;
    int lo = 0, hi = N;
    while (lo + 1 < hi) {
        int const mid = (lo + hi) / 2;
        if (pair_prefix[mid] <= pair)
            lo = mid;
        else
            hi = mid;
    }
    int const first = raw_prefix[lo] + 2 * (pair - pair_prefix[lo]);
    bool const has_second = first + 1 < raw_prefix[lo + 1];
    int64_t const raw = int64_t(head) * raw_tiles + first;
    int64_t const dst = int64_t(head) * pair_tiles + blockIdx.x;
    __shared__ BF16 key[16 * 136], query[16 * 136], restored[16 * 136];
    __shared__ float cross[16 * 20], i0[16 * 20], i1[16 * 20], temp[16 * 20];
    const BF16 *first_slot = ws + raw * WS_SLOT_UNITS;
    const BF16 *second_slot = has_second ? first_slot + WS_SLOT_UNITS : first_slot;
    for (int packet = tid; packet < 16 * 128 / 8; packet += 64) {
        const int x = packet * 8, row = x / 128, col = x % 128;
        const int kr = RowMajor ? x : kr_offset(x);
        const float *g0 = reinterpret_cast<const float *>(first_slot + WS_OFF_GT_U) + col;
        const float *g1 = reinterpret_cast<const float *>(second_slot + WS_OFF_GT_U) + col;
        BF16 *d = ptr + dst * 96 * 128 + x;
        auto load = [](const BF16 *source) {
            return *reinterpret_cast<const uint4 *>(source);
        };
        auto store = [](BF16 *target, uint4 value) {
            *reinterpret_cast<uint4 *>(target) = value;
        };
        store(d, load(first_slot + WS_OFF_KD_U + x));
        uint4 value = has_second ? load(second_slot + WS_OFF_KD_U + x)
                                : make_uint4(0, 0, 0, 0);
        store(key + row * 136 + col, value);
        store(d + 16 * 128, scale_packet(value, g0));
        store(d + 32 * 128, load(first_slot + WS_OFF_QD_U + x));
        value = has_second ? load(second_slot + WS_OFF_QD_U + x)
                           : make_uint4(0, 0, 0, 0);
        store(query + row * 136 + col, value);
        store(d + 48 * 128, scale_packet(value, g0));
        value = load(first_slot + WS_OFF_KR_U + kr);
        store(restored + row * 136 + col, value);
        store(d + 64 * 128, scale_packet(value, g1, has_second));
        store(d + 80 * 128, has_second ? load(second_slot + WS_OFF_KR_U + kr)
                                     : make_uint4(0, 0, 0, 0));
        if (row == 0) {
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i) {
                const auto a = *reinterpret_cast<const float2 *>(g0 + 2 * i);
                const auto b = has_second ? *reinterpret_cast<const float2 *>(g1 + 2 * i)
                                          : make_float2(1.f, 1.f);
                *reinterpret_cast<float2 *>(pgt + dst * 128 + col + 2 * i) =
                    make_float2(a.x * b.x, a.y * b.y);
            }
        }
    }
    for (int x = tid; x < 256; x += 64) {
        int const row = x / 16, col = x % 16;
        BF16 const a = ws[raw * WS_SLOT_UNITS + WS_OFF_INV_U + x];
        BF16 const b = has_second ? ws[(raw + 1) * WS_SLOT_UNITS + WS_OFF_INV_U + x] : BF16(row == col ? 1.f : 0.f);
        i0[row * 20 + col] = float(a);
        i1[row * 20 + col] = float(b);
        pinv[dst * 1024 + row * 32 + col] = a;
        pinv[dst * 1024 + row * 32 + col + 16] = BF16(0.f);
        pinv[dst * 1024 + (row + 16) * 32 + col + 16] = b;
        pmqk[dst * 1024 + row * 32 + col] = ws[raw * WS_SLOT_UNITS + WS_OFF_MQK_U + x];
        pmqk[dst * 1024 + row * 32 + col + 16] = BF16(0.f);
        pmqk[dst * 1024 + (row + 16) * 32 + col + 16] =
            has_second ? ws[(raw + 1) * WS_SLOT_UNITS + WS_OFF_MQK_U + x] : BF16(0.f);
    }
    if (tid < 16) {
        pbeta[dst * 32 + tid] = reinterpret_cast<float const *>(ws + raw * WS_SLOT_UNITS + WS_OFF_BETA_U)[tid];
        pbeta[dst * 32 + 16 + tid] = has_second ? reinterpret_cast<float const *>(ws + (raw + 1) * WS_SLOT_UNITS + WS_OFF_BETA_U)[tid] : 0.f;
    }
    __syncthreads();
    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    Tensor ta = make_tensor(make_smem_ptr(warp == 0 ? key : query),
                            Layout<Shape<_16, _128>, Stride<Int<136>, _1>>{});
    Tensor tb =
        make_tensor(make_smem_ptr(restored), Layout<Shape<_16, _128>, Stride<Int<136>, _1>>{});
    Tensor ref = local_tile(ta, Shape<_16, _16>{}, make_coord(0, 0));
    auto acc = thr.make_fragment_C(thr.partition_C(ref));
    clear(acc);
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        auto aa = local_tile(ta, Shape<_16, _16>{}, make_coord(0, i));
        auto bb = local_tile(tb, Shape<_16, _16>{}, make_coord(0, i));
        auto ra = thr.partition_fragment_A(aa);
        auto rb = thr.partition_fragment_B(bb);
        copy(AutoVectorizingCopy{}, thr.partition_A(aa), ra);
        copy(AutoVectorizingCopy{}, thr.partition_B(bb), rb);
        cute::gemm(thr, ra, rb, acc);
    }
    CUTE_UNROLL
    for (int j = 0; j < decltype(size(acc))::value; ++j) {
        FragCoord const c = frag_coord_mn(lane, j);
        if (warp == 0)
            cross[c.m * 20 + c.n] = acc(j) * pbeta[dst * 32 + 16 + c.m];
        else
            pmqk[dst * 1024 + (c.m + 16) * 32 + c.n] = BF16(acc(j));
    }
    __syncwarp();
    if (warp == 0) {
        pair_gemm_fp32(i1, cross, temp, lane);
        pair_gemm_fp32(temp, i0, cross, lane);
        for (int x = lane; x < 256; x += 32)
            pinv[dst * 1024 + (x / 16 + 16) * 32 + x % 16] = BF16(-cross[x / 16 * 20 + x % 16]);
    }
#endif
}

// Paired 16-token transforms, 32-token recurrent steps.
struct SharedStorageK2Paired {
    struct Stage {
        AiuTile<BF16, 96, 64, 2> tr;
        AiuTile<BF16, 32, 64, 1> v;
        AiuTile<BF16, 32, 64, 1> inv, mqk;
        alignas(128) float gt[128], beta[32];
    } in[2];
    AsyncInPipeline<2, 32, 64> pipe;
};
// Empty requests retain public storage bits, including BF16 NaN payloads.
template <bool StateFP32>
__device__ __forceinline__ void copy_empty_state(
    void const *initial, void *final, int initial_slot, int final_slot,
    int H, int head, int dv_cta, int tid) {
    using Word = std::conditional_t<StateFP32, uint32_t, uint16_t>;
    if (!final || final_slot < 0) return;
    const int64_t src = (int64_t(initial_slot)*H+head)*128*128 + dv_cta*128;
    const int64_t dst = (int64_t(final_slot)*H+head)*128*128 + dv_cta*128;
    // Public state is staged to 16-byte alignment before paired dispatch.
    const auto* source = initial && initial_slot >= 0
        ? reinterpret_cast<const uint4*>(static_cast<const Word*>(initial) + src) : nullptr;
    auto* destination = reinterpret_cast<uint4*>(static_cast<Word*>(final) + dst);
    for (int i = tid; i < 32*128*sizeof(Word)/16; i += 96)
        destination[i] = source ? source[i] : make_uint4(0, 0, 0, 0);
}

template <bool StateFP32, bool Windowed>
__global__ __launch_bounds__(96, 3) void flashkda_paired_fwd_recurrence_paired(
    BF16 const *tr, float const *gt, BF16 const *inv, BF16 const *mqk, float const *beta,
    BF16 const *v, void const *initial, void *final, BF16 *out, int T, int H, int N, int vs,
    int32_t const *cu, int const *prefix, int total_pairs,
    int32_t const *initial_indices, int32_t const *final_indices,
    int first_pair, int window_pairs, float *carries) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    using namespace cute;
    __shared__ SharedStorageK2Paired sm;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int const seq = blockIdx.x;
    int const sequence_begin = cu ? int(cu[seq]) : seq * (T / N);
    int const eos = cu ? int(cu[seq + 1]) : (seq + 1) * (T / N);
    int const sequence_pairs = (eos - sequence_begin + 31) / 32;
    int const begin = Windowed ? max(0, first_pair - prefix[seq]) : 0;
    int const end = Windowed ? max(0, min(sequence_pairs, first_pair + window_pairs - prefix[seq])) : sequence_pairs;
    // Empty rows publish their identity once; other windows have no work here.
    if constexpr (Windowed)
        if (end <= begin && !(sequence_pairs == 0 && first_pair == 0)) return;
    bool const continuation = Windowed && begin > 0;
    bool const pending = Windowed && end < sequence_pairs;
    int const initial_slot = continuation ? seq : initial_indices ? initial_indices[seq] : seq;
    int const final_slot = pending ? seq : final_indices ? final_indices[seq] : seq;
    initial = continuation ? carries : initial;
    final = pending ? carries : final;
    int const bos = sequence_begin + begin * 32;
    int head = blockIdx.y, dv_cta = blockIdx.z * 32, dv = dv_cta + warp * 16;
    int const nt = end - begin, tile_base = prefix[seq] + begin - (Windowed ? first_pair : 0);
    if constexpr (Windowed) {
        if (sequence_pairs == 0) {
            copy_empty_state<StateFP32>(initial, final, initial_slot, final_slot, H, head, dv_cta, tid);
            return;
        }
    }
    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    Tensor ref = make_tensor(make_smem_ptr(sm.in[0].inv.data()),
                             Layout<Shape<_16, _16>, Stride<Int<40>, _1>>{});
    using AF = decltype(thr.partition_fragment_A(ref));
    using BF = decltype(thr.partition_fragment_B(ref));
    using AC = decltype(thr.make_fragment_C(thr.partition_C(ref)));
    AC state_acc[8];
    if (warp < 2) {
        CUTE_UNROLL
        for (int k = 0; k < 8; ++k) {
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(state_acc[0]))::value; ++j) {
                FragCoord c = frag_coord_mn(lane, j);
                int64_t off =
                    (int64_t(initial_slot) * H + head) * 128 * 128 + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                if (initial && initial_slot >= 0) {
                    if constexpr (StateFP32)
                        state_acc[k](j) = static_cast<float const *>(initial)[off];
                    else
                        state_acc[k](j) = continuation ? static_cast<float const *>(initial)[off]
                            : float(static_cast<BF16 const *>(initial)[off]);
                } else {
                    state_acc[k](j) = 0.f;
                }
            }
        }
    }
    sm.pipe.init(tid);
    __syncthreads();
    if (warp == 2) {
        AiuTileLoader<BF16, 96, 64, 2> lt;
        AiuTileLoader<BF16, 32, 64, 1> lv, li, lm;
        lt.init(tr, H * total_pairs * 96, 128, 128);
        // Bound the AIU load by this sequence, so padding (including NaNs)
        // never enters a recurrent MMA through a nominally zero beta.
        lv.init(v + head * 128 + (dv_cta / 64) * 64, eos, 64, vs);
        li.init(inv, H * total_pairs * 32, 32, 32);
        lm.init(mqk, H * total_pairs * 32, 32, 32);
        for (int t = 0; t < nt; ++t) {
            int st = t % 2;
            sm.pipe.producer_acquire(st, t);
            auto &z = sm.in[st];
            int64_t ix = int64_t(head) * total_pairs + tile_base + t;
            lt.load(z.tr.data(), ix * 96);
            lv.load(z.v.data(), bos + t * 32);
            li.load(z.inv.data(), ix * 32);
            lm.load(z.mqk.data(), ix * 32);
            cp_async_contiguous<32>(z.gt, gt + ix * 128, 512, lane);
            cp_async_contiguous<32>(z.beta, beta + ix * 32, 128, lane);
            cp_async_fence_all();
            kda_fence_block();
            sm.pipe.producer_commit(st);
        }
    }
    if (warp < 2) {
        // Keep fragment addresses in registers and advance by one P32 tile.
        // This removes repeated wide row/stride products from the mainloop.
        constexpr int OutputPairs = decltype(size(state_acc[0]))::value / 2;
        int64_t output_offsets[2][OutputPairs];
        const int64_t output_step = int64_t(32) * H * 128;
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i)
            CUTE_UNROLL
            for (int j = 0; j < OutputPairs; ++j) {
                const auto c = frag_coord_mn(lane, 2 * j);
                output_offsets[i][j] = (int64_t(bos) + i * 16 + c.m) * H * 128
                    + head * 128 + dv + c.n;
            }
        for (int t = 0; t < nt; ++t) {
            int st = t % 2;
            sm.pipe.consumer_wait(st, t);
            auto &z = sm.in[st];
            AC u[2], o[2];
            int const trbase = __ppu_read_firstlane(unsigned(uintptr_t(z.tr.data()) / 16));
            int const vbase = __ppu_read_firstlane(unsigned(uintptr_t(z.v.data()) / 16));
            int const ibase = __ppu_read_firstlane(unsigned(uintptr_t(z.inv.data()) / 16));
            int const mbase = __ppu_read_firstlane(unsigned(uintptr_t(z.mqk.data()) / 16));
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                clear(u[i]);
                clear(o[i]);
            }
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k) {
                AF state_k;
                narrow_into(state_k, state_acc[k]);
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    BF kb;
                    AF qa;
                    TsmOps<96, 64, 2>::load_b(&kb(0), trbase, k, i * 16);
                    TsmOps<96, 64, 2>::load_a(&qa(0), trbase, k, 32 + i * 16);
                    cute::gemm(thr, state_k, kb, u[i]);
                    cute::gemm(thr, qa, as_B(state_k), o[i]);
                }
            }
            AF before[2], U[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                BF vt;
                TsmOps<32, 64, 1>::load_bt(&vt(0), vbase, (dv_cta % 64) / 16 + warp, i * 16);
                auto vv = as_A(vt);
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(u[0]))::value; ++j) {
                    FragCoord c = frag_coord_mn(lane, j);
                    u[i](j) = (float(vv(j)) - u[i](j)) * z.beta[i * 16 + c.n];
                }
                narrow_into(before[i], u[i]);
            }

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                clear(u[i]);
                CUTE_UNROLL
                for (int j = 0; j <= i; ++j) {
                    BF b;
                    TsmOps<32, 64, 1>::load_b(&b(0), ibase, j, i * 16);
                    cute::gemm(thr, before[j], b, u[i]);
                }
                narrow_into(U[i], u[i]);
            }

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                CUTE_UNROLL
                for (int j = 0; j <= i; ++j) {
                    AF a;
                    TsmOps<32, 64, 1>::load_a(&a(0), mbase, j, i * 16);
                    cute::gemm(thr, a, as_B(U[j]), o[i]);
                }
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(o[0]))::value / 2; ++j) {
                    FragCoord c = frag_coord_mn(lane, 2 * j);
                    BF16 pair[2] = {BF16(o[i](2 * j)), BF16(o[i](2 * j + 1))};
                    if (bos + t * 32 + i * 16 + c.m < eos)
                        *reinterpret_cast<uint32_t *>(out + output_offsets[i][j]) =
                            *reinterpret_cast<uint32_t const *>(pair);
                    output_offsets[i][j] += output_step;
                }
            }
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k) {
                AC ds;
                clear(ds);
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    BF kr;
                    TsmOps<96, 64, 2>::load_bt(&kr(0), trbase, k, 64 + i * 16);
                    cute::gemm(thr, U[i], kr, ds);
                }
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(ds))::value; ++j) {
                    FragCoord c = frag_coord_mn(lane, j);
                    state_acc[k](j) = fmaf(state_acc[k](j), z.gt[k * 16 + c.n], ds(j));
                }
            }
            sm.pipe.consumer_release(st);
        }
    }
    __syncthreads();
    if constexpr (!Windowed && !StateFP32) {
        if (sequence_pairs == 0) {
            // Keep the non-windowed load/recurrence scheduling unchanged.
            copy_empty_state<StateFP32>(initial, final, initial_slot, final_slot, H, head, dv_cta, tid);
            return;
        }
    }
    if (final && final_slot >= 0 && warp < 2) {
        CUTE_UNROLL
        for (int k = 0; k < 8; ++k) {
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(state_acc[0]))::value / 2; ++j) {
                FragCoord c = frag_coord_mn(lane, 2 * j);
                int64_t off =
                    (int64_t(final_slot) * H + head) * 128 * 128 + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                if (StateFP32 || pending) {
                    *reinterpret_cast<float2 *>(static_cast<float *>(final) + off) =
                        make_float2(state_acc[k](2 * j), state_acc[k](2 * j + 1));
                } else {
                    BF16 pair[2] = {BF16(state_acc[k](2 * j)), BF16(state_acc[k](2 * j + 1))};
                    *reinterpret_cast<uint32_t *>(static_cast<BF16 *>(final) + off) =
                        *reinterpret_cast<uint32_t const *>(pair);
                }
            }
        }
    }
#endif
}

} // namespace flashkda_paired


void launch_flashkda_paired_recurrence(
    void* out, const void* v, int64_t v_stride, const void* workspace,
    const void* initial, void* final, const int32_t* initial_indices,
    const int32_t* final_indices, bool state_fp32,
    int tokens, int heads, int sequences, int raw_tiles, const int32_t* cu,
    void* scratch, flashkda::KrWorkspaceLayout layout, hggcStream_t stream) {
    using namespace flashkda_paired;
    int const pairs = (tokens + 31) / 32 + sequences;
    int const capacity = flashkda_paired_window_pairs(tokens, heads, sequences);
    int64_t const records = int64_t(capacity) * heads;
    int64_t const prefix_bytes = (int64_t(sequences + 1) * sizeof(int) + 127) / 128 * 128;
    auto* raw_prefix = static_cast<int*>(scratch);
    auto* pair_prefix = reinterpret_cast<int*>(static_cast<char*>(scratch) + prefix_bytes);
    auto* transforms = reinterpret_cast<BF16*>(static_cast<char*>(scratch) + 2 * prefix_bytes);
    auto* gates = reinterpret_cast<float*>(transforms + records * 96 * 128);
    auto* inverse = reinterpret_cast<BF16*>(gates + records * 128);
    auto* query_key = inverse + records * 32 * 32;
    auto* betas = reinterpret_cast<float*>(query_key + records * 32 * 32);
    flashkda_paired_pair_prefix<<<1, 32, 0, stream>>>(cu, tokens, sequences, raw_prefix, pair_prefix);
    auto* carries = capacity < pairs ? betas + records * 32 : nullptr;
    auto run_windows = [&](auto windowed) {
    for (int first = 0; first < pairs; first += capacity) {
        int const count = std::min(capacity, pairs - first);
        auto compose = [&](auto row_major) {
            flashkda_paired_compose_pairs<decltype(row_major)::value,decltype(windowed)::value>
                <<<dim3(count, heads), 64, 0, stream>>>(
                    static_cast<const BF16*>(workspace), transforms, gates, inverse, query_key, betas,
                    raw_prefix, pair_prefix, raw_tiles, capacity, sequences, first);
        };
        if (flashkda::kr_workspace_row_major(layout)) compose(std::true_type{});
        else compose(std::false_type{});
        auto run = [&](auto fp32) {
            flashkda_paired_fwd_recurrence_paired<decltype(fp32)::value,decltype(windowed)::value>
                <<<dim3(sequences, heads, 4), 96, 0, stream>>>(
                    transforms, gates, inverse, query_key, betas, static_cast<const BF16*>(v),
                    initial, final, static_cast<BF16*>(out), tokens, heads, sequences, int(v_stride),
                    cu, pair_prefix, capacity, initial_indices, final_indices, first, count, carries);
        };
        if (state_fp32) run(std::true_type{});
        else run(std::false_type{});
    }
    };
    if (capacity < pairs) run_windows(std::true_type{});
    else run_windows(std::false_type{});
}
