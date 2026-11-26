// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2026 MoonshotAI
// SPDX-License-Identifier: BSD-3-Clause AND MIT
// Adapted from FlashKDA.

#include "prefill/flashkdapro/common/launch_api.h"
#include "prefill/flashkdapro/common/conditioned_api.h"
#include "conditioned_repair.cuh"
#include <type_traits>
#include "flashkda_ppu_util.cuh"
#include "flashkda_f16acc.cuh"
#define ACOMPUTE_VERSION 10500
#include "prefill/flashkdapro/common/tc02/memory.cuh"

namespace flashkda_bf16_prepare {
// Independent precision tasks share the prepare launch. Each CTA owns
// disjoint value rows and prepares its own finite token window locally.
inline constexpr int RepairValues = 1;
template <int Threads> inline constexpr int RepairJobs =
    (128 + RepairValues * (Threads / 32) - 1) / (RepairValues * (Threads / 32));

using namespace flashkda_paired;
using namespace flashkda;
namespace fk = flashkda;

CUTE_DEVICE uint32_t pack_pair(float x, float y) {
    const auto packed = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&packed);
}

// Same linear 128-column b16 transfer and leader completion protocol used
// by the qualified CP merge. Tensor height clips tails; padz fills the rest.
CUTE_DEVICE void load_linear(BF16 *destination, BF16 const *source, int rows, int64_t stride) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    uint32_t address = uint32_t(__cvta_generic_to_shared(destination));
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
        " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
        :: "r"(address), "l"(source), "r"(128), "r"(rows), "r"(1),
           "r"(128), "r"(16), "r"(1), "l"(stride * int64_t(sizeof(BF16))), "l"(int64_t(0)),
           "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
#endif
}

// Normalize Q/K in their compact AIU layout. KD/QD replace the dead gate
// input; defer KI/KR publication until every compact Q/K reader has finished.
// After both Gram warps finish, KI becomes the inverse workspace.
template <int InputPitch> struct SharedStorage {
    struct Inputs {
        alignas(128) BF16 q[16 * InputPitch], k[16 * InputPitch], g[16 * InputPitch];
    };
    struct Factors {
        alignas(128) BF16 ki[16 * 136], kr[16 * 136], kd[16 * 136], qd[16 * 136];
    };
    struct Gram {
        alignas(128) float lower[16 * 20];
        alignas(128) BF16 mqk[16 * 24], inverse[16 * 24];
    };
    union { Inputs input; Factors factor; Gram gram; };
    alignas(128) float dt[128], gt[128], beta[32];
};
static_assert(offsetof(SharedStorage<128>::Gram, lower) == offsetof(SharedStorage<128>::Factors, ki));
static_assert(offsetof(SharedStorage<128>::Gram, inverse) + 16 * 24 * sizeof(BF16)
              <= offsetof(SharedStorage<128>::Factors, kr));

// A warp scans each chunk of live request lengths without a serial global
// prefix dependency. Tail lanes contribute zero and still join every shuffle.
__global__ void build_prefix(int32_t const *cu, int *prefix, int n,
                             int tokens, bool precision) {
    const int lane = threadIdx.x;
    int base = 0, count = 0;
    int* requests = prefix + n + 1;
    const int first = cu[0];
    if (lane == 0) prefix[0] = 0;
    for (int start = 0; start < n; start += 32) {
        const int seq = start + lane;
        const int bos = seq < n ? cu[seq] : 0;
        const int eos = seq < n ? cu[seq + 1] : 0;
        int tiles = seq < n ? (eos - bos + 15) / 16 : 0;
        CUTE_UNROLL
        for (int offset = 1; offset < 32; offset *= 2) {
            const int previous = __shfl_up_sync(0xffffffff, tiles, offset);
            if (lane >= offset) tiles += previous;
        }
        if (seq < n) prefix[seq + 1] = base + tiles;
        base += __shfl_sync(0xffffffff, tiles, 31);
        if (precision) {
            const bool direct = seq < n && eos > bos && eos - bos <= KDA_DIRECT_MAX_TOKENS
                && bos >= 0 && eos <= tokens && first == 0;
            const unsigned mask = __ballot_sync(0xffffffff, direct);
            const unsigned lower = lane == 0 ? 0u : (1u << lane) - 1;
            if (direct) requests[1 + count + __popc(mask & lower)] = seq;
            count += __popc(mask);
        }
    }
    if (precision && lane == 0) requests[0] = count;
}

// Halfwarps own independent inverse rows. Pivots above the largest live
// row are exact zero updates for finite L; keep only the needed solves.
template <int Threads, int Warp>
CUTE_DEVICE void solve_finite_rows(float (&inverse)[256 / Threads], float const *lower,
                                    int column, int lane) {
    constexpr int Rows = 256 / Threads, Step = Threads / 16;
    CUTE_UNROLL
    for (int pivot = 1 + 2 * Warp + (Rows - 1) * Step; pivot > 0; --pivot) {
        const float factor = lower[pivot * 20 + column];
        CUTE_UNROLL
        for (int r = Rows - 1; r >= 0; --r) {
            if (pivot <= 1 + 2 * Warp + r * Step) {
                const float x = __shfl_sync(0xffffffff, inverse[r], (lane & 16) + pivot);
                inverse[r] = fmaf(-factor, x, inverse[r]);
            }
        }
    }
}

template <bool Affine, bool Preswizzle = false, bool ResolveConditions = false, int Threads = 128, bool Cooperative = false, bool HeadFirst = false, bool ProducerPrecision = false>
__global__ __launch_bounds__(Threads, 4) void prepare(
    BF16 const *q, BF16 const *k, BF16 const *g, BF16 const *beta,
    float const *a_log, float const *dt_bias, float scale, float gate_scale,
    int tokens, int heads, int n, int tiles, int beta_stride, int64_t qs, int64_t ks,
    int32_t const *cu, int const *prefix, BF16 *ws, BF16 *keys, int32_t *aggregate_flags,
    int32_t* tile_flags, ConditionedArguments repair) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    using namespace cute;
    // Cooperative copies can scatter directly into the project's padded
    // factor pitch. Bulk AIU inputs retain their qualified compact layout.
    constexpr int InputPitch = Cooperative ? 136 : 128;
    static_assert(sizeof(SharedStorage<128>) == sizeof(SharedStorage<136>));
    __shared__ SharedStorage<InputPitch> sm;
    static_assert(Threads == 128 || Threads == 256);
    // Every launcher uses exactly Threads in a one-dimensional block.
    // Expose the same power-of-two ranges as the upstream prepare kernel.
    int const tid = int(threadIdx.x) & (Threads - 1), lane = tid % 32;
    int const warp = cutlass::canonical_warp_idx_sync() & (Threads / 32 - 1);
    int const tile = HeadFirst ? blockIdx.y : blockIdx.x;
    int const head = HeadFirst ? blockIdx.x : blockIdx.y;
    if constexpr (ResolveConditions && ProducerPrecision) {
        if (tile >= tiles) {
            // Only these precision-only CTAs reuse local token storage. They
            // contain no asynchronous input/output pipeline or named barriers.
            const int* requests = prefix ? prefix + n + 1 : nullptr;
            const int count = (requests ? requests[0] : n) * RepairJobs<Threads>;
            const int stride = (HeadFirst ? int(gridDim.y) : int(gridDim.x)) - tiles;
            for (int task = tile - tiles; task < count; task += stride) {
                const int index = task / RepairJobs<Threads>;
                const int seq = requests ? requests[1 + index] : index;
                const int part = task % RepairJobs<Threads>;
                if (seq >= n) continue;
                const int bos = cu ? cu[seq] : seq * (tokens / n);
                const int eos = cu ? cu[seq + 1] : (seq + 1) * (tokens / n);
                if (!(eos > bos && eos - bos <= KDA_DIRECT_MAX_TOKENS
                      && bos >= 0 && eos <= tokens && (!cu || cu[0] == 0))) continue;
                static_assert(sizeof(sm) >= KDA_DIRECT_MAX_TOKENS * sizeof(conditioned::RepairToken));
                auto* cached = reinterpret_cast<conditioned::RepairToken*>(&sm);
                const float a_exp = expf(repair.a_log[head]);
                if (tid < 128)
                    for (int token = bos; token < eos; ++token)
                        conditioned::prepare_repair_token(repair, cached[token - bos],
                                                         token, head, tid, lane, a_exp);
                __syncthreads();
                const int first_value = (part * (Threads / 32) + warp) * RepairValues;
                if (first_value < 128)
                    conditioned::repair_cached_values<RepairValues>(
                        repair, seq, head, first_value, lane, cached);
                if (task + stride < count) __syncthreads();
            }
            return;
        }
    }
    // This launch completes before classification. A single CTA initializes
    // every request/head, including requests with no tiles on Graph replay.
    if (aggregate_flags && tile == 0 && head == 0)
        for (int64_t i = tid; i < int64_t(n) * heads; i += Threads) aggregate_flags[i] = 0;
    BF16 *slot = ws + (int64_t(head) * tiles + tile) * WS_SLOT_UNITS;
    auto *flag = reinterpret_cast<int32_t *>(slot + WS_OFF_CONDITION_U);
    int condition = 0;
    int seq = 0, tile_base = 0;
    // A single request owns every tile. Keep its live device bounds,
    // and let the existing valid-row check reject unused padding tiles.
    if (cu && n > 1) {
        if (prefix) {
            int hi = n;
            while (seq + 1 < hi) {
                int const mid = (seq + hi) / 2;
                if (prefix[mid] <= tile) seq = mid;
                else hi = mid;
            }
            tile_base = prefix[seq];
        } else {
            for (; seq < n; ++seq) {
                int const count = (cu[seq + 1] - cu[seq] + 15) / 16;
                if (tile < tile_base + count) break;
                tile_base += count;
            }
        }
    } else if (!cu) {
        int const per_seq = ((tokens / n) + 15) / 16;
        seq = tile / per_seq;
        tile_base = seq * per_seq;
    }
    if (seq >= n) {
        if (tid == 0) {
            *flag = 0;
            if (tile_flags) tile_flags[int64_t(head) * tiles + tile] = 0;
        }
        return;
    }
    int const bos = cu ? cu[seq] : seq * (tokens / n);
    int const eos = cu ? cu[seq + 1] : (seq + 1) * (tokens / n);
    int const row0 = bos + (tile - tile_base) * 16;
    int const valid = min(16, eos - row0);
    if (valid <= 0) {
        if (tid == 0) {
            *flag = 0;
            if (tile_flags) tile_flags[int64_t(head) * tiles + tile] = 0;
        }
        return;
    }
    if constexpr (ResolveConditions) {
        const bool direct = eos - bos <= KDA_DIRECT_MAX_TOKENS
            && bos >= 0 && eos <= tokens && (!cu || cu[0] == 0);
        if (direct) {
            if constexpr (!ProducerPrecision) {
                // Compact consumer tasks share these exact tokens, prepared
                // once per request rather than once per value CTA.
                static_assert(KDA_DIRECT_MAX_TOKENS * sizeof(conditioned::RepairToken)
                              <= WS_OFF_CONDITION_U * sizeof(BF16));
                auto* cached = reinterpret_cast<conditioned::RepairToken*>(slot);
                const float a_exp = expf(repair.a_log[head]);
                if (tid < 128)
                    for (int token = bos; token < eos; ++token)
                        conditioned::prepare_repair_token(repair, cached[token - bos],
                                                         token, head, tid, lane, a_exp);
            }
            if (tid == 0) {
                *flag = 1;
                if (tile_flags) tile_flags[int64_t(head) * tiles + tile] = 1;
            }
            return;
        }
    }
    if constexpr (Cooperative) {
        auto load = [&](auto full) {
            cp_async_tile<16, 128, InputPitch, Threads, decltype(full)::value>(
                sm.input.q, q + int64_t(row0) * qs + head * 128, qs, valid, tid);
            cp_async_tile<16, 128, InputPitch, Threads, decltype(full)::value>(
                sm.input.k, k + int64_t(row0) * ks + head * 128, ks, valid, tid);
            cp_async_tile<16, 128, InputPitch, Threads, decltype(full)::value>(
                sm.input.g, g + int64_t(row0) * heads * 128 + head * 128,
                int64_t(heads) * 128, valid, tid);
        };
        if (valid == 16) load(std::true_type{});
        else load(std::false_type{});
        cp_async_fence_all();
    } else if (tid == 0) {
        load_linear(sm.input.q, q + int64_t(row0) * qs + head * 128, valid, qs);
        load_linear(sm.input.k, k + int64_t(row0) * ks + head * 128, valid, ks);
        load_linear(sm.input.g, g + int64_t(row0) * heads * 128 + head * 128, valid, int64_t(heads) * 128);
        cp_async_fence_all();
    }
    if (tid < 128) sm.dt[tid] = dt_bias[head * 128 + tid];
    if (tid < 16) sm.beta[tid] = tid < valid
        ? fk::sigmoid_tanh_approx_f32(float(beta[int64_t(row0 + tid) * beta_stride + head])) : 0.f;
    float const a = expf(a_log[head]);
    if (Cooperative || tid == 0) cp_async_wait_all<0>();
    __syncthreads();
    // Each lane owns its complete compact packet, so normalization can
    // overwrite it in place without a separate repack or packet-cache join.
    const int norm_col = (tid % 16) * 8;
    CUTE_UNROLL
    for (int r = 0; r < 256 / Threads; ++r) {
        const int row = tid / 16 + (Threads / 16) * r;
        float qv[8], kv[8], qsum = 0.f, ksum = 0.f;
        const uint4 qp = *reinterpret_cast<const uint4 *>(sm.input.q + row * InputPitch + norm_col);
        const uint4 kp = *reinterpret_cast<const uint4 *>(sm.input.k + row * InputPitch + norm_col);
        const uint32_t qw[4] = {qp.x, qp.y, qp.z, qp.w};
        const uint32_t kw[4] = {kp.x, kp.y, kp.z, kp.w};
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
            const float2 q = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162 *>(qw + j));
            const float2 k = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162 *>(kw + j));
            qv[2 * j] = q.x; qv[2 * j + 1] = q.y;
            kv[2 * j] = k.x; kv[2 * j + 1] = k.y;
            qsum += q.x * q.x; ksum += k.x * k.x;
            qsum += q.y * q.y; ksum += k.y * k.y;
        }
        CUTE_UNROLL
        for (int d = 8; d; d >>= 1) {
            qsum += __shfl_xor_sync(0xffffffff, qsum, d);
            ksum += __shfl_xor_sync(0xffffffff, ksum, d);
        }
        float const qi = rsqrtf(qsum + 1e-6f), ki = rsqrtf(ksum + 1e-6f);
        uint32_t qn[4], kn[4];
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
            qn[j] = pack_pair(qv[2 * j] * qi, qv[2 * j + 1] * qi);
            kn[j] = pack_pair(kv[2 * j] * ki, kv[2 * j + 1] * ki);
        }
        // The raw FP32 recurrence needs the gate-independent diagonal.
        // Reuse normalized BF16 packets while they are still in registers,
        // as the original prepare kernel does, instead of reloading Q/K for
        // a separate full Gram tile whose off-diagonal values are discarded.
        if constexpr (!Affine) {
            float dot = 0.f;
            CUTE_UNROLL
            for (int j = 0; j < 4; ++j) {
                const float2 q = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(qn + j));
                const float2 k = __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(kn + j));
                dot += q.x * k.x;
                dot += q.y * k.y;
            }
            CUTE_UNROLL
            for (int d = 8; d; d >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, d);
            if (norm_col == 0) sm.beta[16 + row] = float(BF16(dot * scale));
        }
        const uint4 normalized_key = make_uint4(kn[0], kn[1], kn[2], kn[3]);
        *reinterpret_cast<uint4 *>(sm.input.q + row * InputPitch + norm_col) =
            make_uint4(qn[0], qn[1], qn[2], qn[3]);
        *reinterpret_cast<uint4 *>(sm.input.k + row * InputPitch + norm_col) = normalized_key;
        // Snapshot each normalized key packet before the factor overlay.
        if (Affine && !ResolveConditions && valid >= 2 && row < 2)
            *reinterpret_cast<uint4 *>(keys+(int64_t(head)*tiles+tile)*256+row*128+norm_col) = normalized_key;
    }
    // Load every warp's bias pair before the existing CTA rendezvous.
    // Its shared storage is then dead and can hold the normalized key pair.
    constexpr int Parts = Threads / 64, PairLanes = 32 / Parts, PartRows = 16 / Parts;
    int const part = lane / PairLanes, col = 2 * (warp * PairLanes + lane % PairLanes);
    const float dt[2] = {sm.dt[col], sm.dt[col + 1]};
    asm volatile("" :: "f"(dt[0]), "f"(dt[1]));
    __syncthreads();
    if constexpr (!Affine && !ResolveConditions) {
        if (valid >= 2) {
            CUTE_UNROLL
            for (int i=tid;i<256;i+=Threads)
                keys[(int64_t(head)*tiles+tile)*256+i] =
                    sm.input.k[i];
        }
    }
    // Match FlashKDA's decayed BF16 Gram in explicit BF16 mode.
    // The FP32 producer retains its corrected, gate-independent diagonal.
    if constexpr (ResolveConditions) {
        static_assert(sizeof(sm.dt) >= 256 * sizeof(BF16));
        if (warp == 2 && valid >= 2) {
            const int row = lane / 16, column = (lane % 16) * 8;
            reinterpret_cast<uint4*>(sm.dt)[lane] =
                *reinterpret_cast<const uint4*>(sm.input.k + row * InputPitch + column);
        }
    }
    if (warp == 0) {
        bool const saturated = __any_sync(0xffffffff, lane < valid && lane < 16 && sm.beta[lane] == 1.f);
        const bool short_request = aggregate_flags && eos - bos <= KDA_DIRECT_MAX_TOKENS
            && bos >= 0 && eos <= tokens && (!cu || cu[0] == 0);
        condition = int(saturated || short_request) | (valid >= 2 ? 2 : 0);
    }
    // Split activation across available subwarps. Adjacent four-row parts
    // retain the original eight-row accumulation order before joining halves.
    auto read_pair = [](BF16 const *p) {
        const uint32_t bits = *reinterpret_cast<uint32_t const *>(p);
        return __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162 *>(&bits));
    };
    auto write_pair = [](BF16 *p, float x, float y) {
        *reinterpret_cast<uint32_t *>(p) = pack_pair(x, y);
    };
    // Keep independent shared reads ahead of activation through register
    // dependencies. An all-memory asm clobber adds a hardware fence on PPU,
    // while these input-only groups require just the existing register values.
    float2 gate_input[PartRows];
    CUTE_UNROLL
    for (int i = 0; i < PartRows; ++i)
        gate_input[i] = read_pair(sm.input.g + (part * PartRows + i) * InputPitch + col);
    static_assert(PartRows % 4 == 0);
    CUTE_UNROLL
    for (int first = 0; first < PartRows; first += 4)
        asm volatile("" :: "f"(gate_input[first].x), "f"(gate_input[first].y),
            "f"(gate_input[first + 1].x), "f"(gate_input[first + 1].y),
            "f"(gate_input[first + 2].x), "f"(gate_input[first + 2].y),
            "f"(gate_input[first + 3].x), "f"(gate_input[first + 3].y));
    float activated[2][PartRows], sums[2] = {0.f, 0.f};
    CUTE_UNROLL
    for (int i = 0; i < PartRows; ++i) {
        int const row = part * PartRows + i;
        const float values[2] = {gate_input[i].x, gate_input[i].y};
        CUTE_UNROLL
        for (int c = 0; c < 2; ++c)
            activated[c][i] = row < valid ? gate_scale * fk::sigmoid_tanh_approx_f32(
                a * (values[c] + dt[c])) : 0.f;
    }
    if constexpr (Parts == 4) {
        // Form the lower quarter's ordered carry in every lane, then run
        // both quarters through the same instruction sequence. Each upper
        // quarter still adds its four values after the original carry.
        float carry[2] = {0.f, 0.f};
        CUTE_UNROLL
        for (int i = 0; i < PartRows; ++i)
            CUTE_UNROLL
            for (int c = 0; c < 2; ++c) carry[c] += activated[c][i];
        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            const float preceding = __shfl_sync(0xffffffff, carry[c], (lane & 16) + lane % PairLanes);
            sums[c] = part & 1 ? preceding : 0.f;
        }
        CUTE_UNROLL
        for (int i = 0; i < PartRows; ++i)
            CUTE_UNROLL
            for (int c = 0; c < 2; ++c) { sums[c] += activated[c][i]; activated[c][i] = sums[c]; }
    } else {
        CUTE_UNROLL
        for (int i = 0; i < PartRows; ++i)
            CUTE_UNROLL
            for (int c = 0; c < 2; ++c) { sums[c] += activated[c][i]; activated[c][i] = sums[c]; }
    }
    float total[2];
    CUTE_UNROLL
    for (int c = 0; c < 2; ++c) {
        constexpr int FirstPart = Parts / 2 - 1;
        const float first = __shfl_sync(0xffffffff, sums[c], lane % PairLanes + FirstPart * PairLanes);
        const float last = __shfl_sync(0xffffffff, sums[c], lane % PairLanes + 32 - PairLanes);
        const float preceding = part >= Parts / 2 ? first : 0.f;
        CUTE_UNROLL
        for (int i = 0; i < PartRows; ++i) activated[c][i] += preceding;
        total[c] = kda_exp2(first + last);
        if (part == Parts - 1) sm.gt[col + c] = total[c];
    }
    __syncthreads();
    uint32_t pending_ki[PartRows], pending_kr[PartRows];
    CUTE_UNROLL
    for (int i = 0; i < PartRows; ++i) {
        int const row = part * PartRows + i;
        const int off = row * 136 + col;
        const float2 qv = read_pair(sm.input.q + row * InputPitch + col);
        const float2 kv = read_pair(sm.input.k + row * InputPitch + col);
        const float decay[2] = {kda_exp2(activated[0][i]), kda_exp2(activated[1][i])};
        const float inverse[2] = {__ppu_rcpf(decay[0]), __ppu_rcpf(decay[1])};
        write_pair(sm.factor.kd + off, kv.x * decay[0], kv.y * decay[1]);
        write_pair(sm.factor.qd + off, qv.x * decay[0] * scale, qv.y * decay[1] * scale);
        pending_ki[i] = pack_pair(kv.x * inverse[0], kv.y * inverse[1]);
        if constexpr (Affine)
            pending_kr[i] = pack_pair(kv.x * inverse[0] * total[0], kv.y * inverse[1] * total[1]);
        else
            pending_kr[i] = pack_pair(
                float(BF16(kv.x * inverse[0])) * total[0],
                float(BF16(kv.y * inverse[1])) * total[1]);
        if constexpr (Cooperative) {
            // Padded inputs and KI/KR have identical row and column owners.
            // Each pair replaces only the Q/K words this thread just read.
            static_assert(InputPitch == 136);
            static_assert(offsetof(SharedStorage<InputPitch>::Inputs, q)
                          == offsetof(SharedStorage<InputPitch>::Factors, ki));
            static_assert(offsetof(SharedStorage<InputPitch>::Inputs, k)
                          == offsetof(SharedStorage<InputPitch>::Factors, kr));
            *reinterpret_cast<uint32_t *>(sm.factor.ki + off) = pending_ki[i];
            *reinterpret_cast<uint32_t *>(sm.factor.kr + off) = pending_kr[i];
        }
    }
    if constexpr (!Cooperative) {
        // Compact AIU inputs have different owners after the padded overlay.
        // Finish every Q/K read before publishing KI/KR in that layout.
        __syncthreads();
        CUTE_UNROLL
        for (int i = 0; i < PartRows; ++i) {
            const int off = (part * PartRows + i) * 136 + col;
            *reinterpret_cast<uint32_t *>(sm.factor.ki + off) = pending_ki[i];
            *reinterpret_cast<uint32_t *>(sm.factor.kr + off) = pending_kr[i];
        }
    }
    __syncthreads();
    auto mma = make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    using Padded = Layout<Shape<_16, _128>, Stride<Int<136>, _1>>;
    using Small = Layout<Shape<_16, _16>, Stride<Int<24>, _1>>;
    Tensor small = make_tensor(make_smem_ptr(sm.gram.mqk), Small{});
    if (warp < 2) {
        Tensor ta = make_tensor(make_smem_ptr(warp == 0 ? sm.factor.kd : sm.factor.qd), Padded{});
        Tensor tb = make_tensor(make_smem_ptr(sm.factor.ki), Padded{});
        auto acc = thr.make_fragment_C(thr.partition_C(small)); clear(acc);
        CUTE_UNROLL
        for (int kb = 0; kb < 8; ++kb) {
            auto aa = local_tile(ta, Shape<_16, _16>{}, make_coord(0, kb));
            auto bb = local_tile(tb, Shape<_16, _16>{}, make_coord(0, kb));
            auto ra = thr.partition_fragment_A(aa); auto rb = thr.partition_fragment_B(bb);
            copy(AutoVectorizingCopy{}, thr.partition_A(aa), ra);
            copy(AutoVectorizingCopy{}, thr.partition_B(bb), rb);
            cute::gemm(thr, ra, rb, acc);
        }
        cutlass::arch::NamedBarrier::arrive_and_wait(64, 0);
        float row_pair_bound = 0.f;
        CUTE_UNROLL
        for (int j = 0; j < decltype(size(acc))::value; ++j) {
            FragCoord const c = frag_coord_mn(lane, j);
            if (warp == 0) {
                float const x = c.m <= c.n ? 0.f : float(half_t(float(half_t(acc(j))) * sm.beta[c.m]));
                sm.gram.lower[c.m * 20 + c.n] = x;
                row_pair_bound += isfinite(x) ? fabsf(x) : INFINITY;
            } else {
                BF16 const value = c.m < c.n ? BF16(0.f)
                    : BF16(!Affine && c.m == c.n ? sm.beta[16 + c.m] : acc(j));
                if constexpr (Affine) sm.gram.mqk[c.m * 24 + c.n] = value;
                else slot[WS_OFF_MQK_U + c.m * 16 + c.n] = value;
            }
        }
        if (warp == 0) {
            // Four lanes cover two complete rows of the rounded FP16 L.
            // Their sum bounds either row; padding is dead after Gram.
            row_pair_bound += __shfl_xor_sync(0xffffffff, row_pair_bound, 1);
            row_pair_bound += __shfl_xor_sync(0xffffffff, row_pair_bound, 2);
            CUTE_UNROLL
            for (int d = 16; d; d >>= 1)
                row_pair_bound = fmaxf(row_pair_bound,
                    __shfl_xor_sync(0xffffffff, row_pair_bound, d));
            if (lane == 0) sm.gram.lower[16] = row_pair_bound;
        }
    } else {
        if constexpr (ResolveConditions) {
            if (warp == 2) {
                const bool selected = conditioned::exact_structural_pair<BF16>(
                    reinterpret_cast<const BF16*>(sm.dt), valid, lane);
                if (lane == 0) {
                    if constexpr (Affine) sm.beta[31] = float(selected);
                    else *flag = int(selected);
                }
            }
        }
        for (int vec = tid - 64; vec < 16 * 16; vec += Threads - 64) {
            int const row = vec / 16, col = (vec % 16) * 8;
            *reinterpret_cast<uint4 *>(slot + WS_OFF_KD_U + row * 128 + col) =
                *reinterpret_cast<uint4 const *>(sm.factor.kd + row * 136 + col);
            *reinterpret_cast<uint4 *>(slot + WS_OFF_QD_U + row * 128 + col) =
                *reinterpret_cast<uint4 const *>(sm.factor.qd + row * 136 + col);
            if constexpr (!Affine) {
                const int dst = row * 128 + col;
                const int src = Preswizzle ? fk::ws_kr_swzl_src_index(dst) : dst;
                *reinterpret_cast<uint4 *>(slot + WS_OFF_KR_U + dst) =
                    *reinterpret_cast<uint4 const *>(sm.factor.kr + (src / 128) * 136 + src % 128);
            }
        }
        if constexpr (!Affine)
            if (tid - 64 < 16)
                reinterpret_cast<float *>(slot + WS_OFF_BETA_U)[tid - 64] = sm.beta[tid - 64];
        for (int c = tid - 64; c < 128; c += Threads - 64)
            reinterpret_cast<float *>(slot + WS_OFF_GT_U)[c] = sm.gt[c];
    }
    __syncthreads();
    if constexpr (ResolveConditions)
        if (warp == 0) condition = (condition & 1) | (Affine ? int(sm.beta[31]) : *flag);
    // A rounded-L row-pair norm <=15/16 certifies the existing FP16
    // L^2 row-norm remains below1, including accumulation/FTZ and FP32
    // reduction errors: its conservative upper bound is less than0.972.
    // Uncertified inputs keep the identical FP16 image and threshold. A
    // previously selected repair also needs no second selection predicate.
    if (warp == 0 && sm.gram.lower[16] > 0.9375f && !(condition & 1)) {
        auto *lower = reinterpret_cast<half_t *>(sm.factor.qd);
        auto *power = lower + 256;
        for (int i = lane; i < 256; i += 32) lower[i] = half_t(sm.gram.lower[(i / 16) * 20 + i % 16]);
        __syncwarp();
        auto l = make_tensor(make_smem_ptr(lower), Layout<Shape<_16, _16>, Stride<_16, _1>>{});
        auto lt = make_tensor(make_smem_ptr(lower), Layout<Shape<_16, _16>, Stride<_1, _16>>{});
        auto acc = fk::make_acc16_f16acc_1warp(lane);
        fk::gemm16_ss_1warp_f16acc(l, lt, lane, acc);
        fk::store_acc16_f16acc_1warp(acc, reinterpret_cast<__half *>(power), 16, lane);
        __syncwarp();
        float norm = 0.f;
        if (lane < 16) {
            #pragma unroll 1
            for (int col = 0; col < 16; ++col) norm += fabsf(float(power[lane * 16 + col]));
            if (!isfinite(norm)) norm = INFINITY;
        }
        CUTE_UNROLL
        for (int d = 16; d; d >>= 1) norm = fmaxf(norm, __shfl_xor_sync(0xffffffff, norm, d));
        if (!(norm < 1.f)) condition |= 1;
    }
    int const first_row = tid / 16, column = tid % 16;
    constexpr int InverseRows = 256 / Threads;
    float inv[InverseRows];
    CUTE_UNROLL
    for (int r = 0; r < InverseRows; ++r)
        inv[r] = first_row + r * (Threads / 16) == column ? 1.f : 0.f;
    if (isfinite(sm.gram.lower[16])) {
        if constexpr (Threads == 256) {
            // Match the upstream eight-warp solve: a common instruction
            // sequence keeps every independent inverse row ready together.
            // Extra pivots above each finite row are exact zero updates.
            CUTE_UNROLL
            for (int pivot = 15; pivot > 0; --pivot) {
                const float factor = sm.gram.lower[pivot * 20 + column];
                const float x = __shfl_sync(0xffffffff, inv[0], (lane & 16) + pivot);
                inv[0] = fmaf(-factor, x, inv[0]);
            }
        } else {
            if (warp == 0) solve_finite_rows<Threads, 0>(inv, sm.gram.lower, column, lane);
            else if (warp == 1) solve_finite_rows<Threads, 1>(inv, sm.gram.lower, column, lane);
            else if (warp == 2) solve_finite_rows<Threads, 2>(inv, sm.gram.lower, column, lane);
            else if (warp == 3) solve_finite_rows<Threads, 3>(inv, sm.gram.lower, column, lane);
        }
    } else {
        // Preserve every original per-row FMA for NaN/Inf operands. This
        // uncommon loop stays compact instead of cloning four full solves.
        #pragma unroll 1
        for (int pivot = 15; pivot > 0; --pivot) {
            const float factor = sm.gram.lower[pivot * 20 + column];
            CUTE_UNROLL
            for (int r = 0; r < InverseRows; ++r) {
                const float x = __shfl_sync(0xffffffff, inv[r], (lane & 16) + pivot);
                inv[r] = fmaf(-factor, x, inv[r]);
            }
        }
    }
    CUTE_UNROLL
    for (int r = 0; r < InverseRows; ++r) {
        int const row = first_row + r * (Threads / 16);
        if constexpr (Affine) sm.gram.inverse[column * 24 + row] = BF16(inv[r]);
        else slot[WS_OFF_INV_U + row * 16 + column] = BF16(inv[r]);
    }
    // All selection predicates are uniform in warp zero. Publish once after
    // the final predicate; consumers run only after this producer completes.
    if (tid == 0) {
        *flag = condition;
        if (tile_flags) tile_flags[int64_t(head) * tiles + tile] = condition;
    }
    if constexpr (!Affine) return;
    __syncthreads();
    Tensor ti = make_tensor(make_smem_ptr(sm.gram.inverse), Small{});
    Tensor tr = make_tensor(make_smem_ptr(sm.factor.kr), Layout<Shape<_128, _16>, Stride<_1, Int<136>>>{});
    auto ia = thr.partition_fragment_A(ti);
    copy(AutoVectorizingCopy{}, thr.partition_A(ti), ia);
    auto acc = thr.make_fragment_C(thr.partition_C(small));
    CUTE_UNROLL
    for (int kb = warp; kb < 8; kb += Threads / 32) {
        auto rb = local_tile(tr, Shape<_16, _16>{}, make_coord(kb, 0));
        auto b = thr.partition_fragment_B(rb);
        copy(AutoVectorizingCopy{}, thr.partition_B(rb), b);
        clear(acc); cute::gemm(thr, ia, b, acc);
        CUTE_UNROLL
        for (int j = 0; j < decltype(size(acc))::value / 2; ++j) {
            FragCoord const c = frag_coord_mn(lane, 2 * j);
            BF16 pair[2] = {BF16(acc(2 * j) * sm.beta[c.m]), BF16(acc(2 * j + 1) * sm.beta[c.m])};
            *reinterpret_cast<uint32_t *>(slot + WS_OFF_KR_U + c.m * 128 + kb * 16 + c.n) =
                *reinterpret_cast<uint32_t const *>(pair);
        }
    }
    if (warp == 0) {
        auto ma = thr.partition_fragment_A(small); auto ib = thr.partition_fragment_B(ti);
        copy(AutoVectorizingCopy{}, thr.partition_A(small), ma);
        copy(AutoVectorizingCopy{}, thr.partition_B(ti), ib);
        clear(acc); cute::gemm(thr, ma, ib, acc);
        CUTE_UNROLL
        for (int j = 0; j < decltype(size(acc))::value / 2; ++j) {
            FragCoord const c = frag_coord_mn(lane, 2 * j);
            BF16 pair[2] = {BF16(acc(2 * j) * sm.beta[c.n]), BF16(acc(2 * j + 1) * sm.beta[c.n + 1])};
            *reinterpret_cast<uint32_t *>(slot + WS_OFF_MQK_U + c.m * 16 + c.n) =
                *reinterpret_cast<uint32_t const *>(pair);
        }
    }
#endif
}

// Rare FP64 geometry checks retain their independent register allocation.
// Classification can also publish the same per-request repair flags, avoiding
// a second scan of the private workspace when the original state is read-only.
__global__ __launch_bounds__(256) void classify_prepared_keys(
    BF16 *ws, BF16 const *keys, int tokens, int heads, int sequences, int tiles,
    const int32_t *cu, int32_t *aggregate_flags, const int32_t* tile_flags) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    const int64_t record = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
    if (record >= int64_t(heads) * tiles) return;
    const int lane = threadIdx.x % 32;
    auto *flag = reinterpret_cast<int32_t *>(ws + record * flashkda::WS_SLOT_UNITS
                                            + flashkda::WS_OFF_CONDITION_U);
    const int pending = tile_flags ? tile_flags[record] : *flag;
    bool selected = (pending & 1) != 0;
    if (!selected && (pending & 2))
        selected = flashkda::conditioned::exact_structural_pair(keys + record * 256, 2, lane);
    if (lane != 0) return;
    *flag = int(selected);
    if (!selected || !aggregate_flags) return;

    // Only selected records need a request lookup. Ordinary records do no
    // atomic work; the preceding prepare launch has already cleared all flags.
    const int tile = int(record % tiles), head = int(record / tiles);
    int seq = 0, base = 0;
    if (cu) {
        if (sequences >= 64) {
            const int *prefix = reinterpret_cast<const int *>(
                ws + int64_t(heads) * tiles * flashkda::WS_SLOT_UNITS);
            int hi = sequences;
            while (seq + 1 < hi) {
                const int mid = (seq + hi) / 2;
                if (prefix[mid] <= tile) seq = mid;
                else hi = mid;
            }
        } else {
            for (; seq < sequences; ++seq) {
                const int count = (cu[seq + 1] - cu[seq] + 15) / 16;
                if (tile < base + count) break;
                base += count;
            }
        }
    } else {
        seq = tile / (((tokens / sequences) + 15) / 16);
    }
    if (seq < sequences) atomicOr(aggregate_flags + seq * heads + head, 1);
#endif
}
} // namespace flashkda_bf16_prepare

const int32_t* launch_flashkda_bf16_prepare(
    const void *q, const void *k, const void *g, const void *beta,
    const float *a_log, const float *bias, float scale, float gate_scale,
    int tokens, int heads, int sequences, int tiles, int beta_stride,
    int64_t qs, int64_t ks, const int32_t *cu, void *ws, void *keys,
    bool affine, hggcStream_t stream, bool preswizzle, int32_t *aggregate_flags, bool defer_classification, int32_t* tile_flags, const flashkda::ConditionedArguments* serial_repair) {
    using namespace flashkda_bf16_prepare;
    if (!tiles) return nullptr;
    // Precision consumers need live compact request IDs for every varlen
    // batch. The existing allocation reserves this table for all batch sizes.
    int *prefix = cu && (sequences >= 64 || (serial_repair && sequences > 1))
        ? reinterpret_cast<int *>(static_cast<BF16 *>(ws) + int64_t(heads) * tiles * WS_SLOT_UNITS) : nullptr;
    if (prefix) build_prefix<<<1, 32, 0, stream>>>(cu, prefix, sequences, tokens, serial_repair != nullptr);
    auto run = [&](auto mode, auto swizzle, auto resolve, auto cooperative) {
        auto launch = [&](auto threads, auto producer_precision) {
            constexpr int Threads = decltype(threads)::value;
            const int precision_blocks = decltype(producer_precision)::value
                ? sequences * RepairJobs<Threads> : 0;
            auto dispatch = [&](auto head_first) {
                constexpr bool HeadFirst = decltype(head_first)::value;
                auto kernel = prepare<decltype(mode)::value, decltype(swizzle)::value,
                    decltype(resolve)::value, Threads, decltype(cooperative)::value, HeadFirst, decltype(producer_precision)::value>;
                const int jobs = tiles + precision_blocks;
                const dim3 grid = HeadFirst ? dim3(heads, jobs) : dim3(jobs, heads);
                kernel<<<grid, Threads, 0, stream>>>(
                    static_cast<BF16 const *>(q), static_cast<BF16 const *>(k), static_cast<BF16 const *>(g),
                    static_cast<BF16 const *>(beta), a_log, bias, scale, gate_scale,
                    tokens, heads, sequences, tiles, beta_stride, qs, ks, cu, prefix,
                    static_cast<BF16 *>(ws), static_cast<BF16 *>(keys), aggregate_flags, tile_flags,
                    serial_repair ? *serial_repair : ConditionedArguments{});
            };
            if constexpr (decltype(resolve)::value && decltype(producer_precision)::value) {
                // Group producer-owned precision work with its ordinary heads.
                // Fall back to the existing grid when its transposition would
                // exceed this device's actual launch limit.
                int device = 0, max_grid_y = 0;
                if (hggcGetDevice(&device) == hggcSuccess
                    && hggcDeviceGetAttribute(&max_grid_y, hggcDevAttrMaxGridDimY, device) == hggcSuccess
                    && int64_t(tiles) + precision_blocks <= max_grid_y) dispatch(std::true_type{});
                else dispatch(std::false_type{});
            } else dispatch(std::false_type{});
        };
        // Compact consumers own precision outputs. Compile those unused
        // producer state fragments out of the ordinary prepare specialization.
        if constexpr (decltype(resolve)::value) {
            if (prefix || sequences == 1) launch(std::integral_constant<int, 256>{}, std::false_type{});
            else launch(std::integral_constant<int, 256>{}, std::true_type{});
        } else launch(std::integral_constant<int, 128>{}, std::false_type{});
    };
    if (serial_repair) {
        const bool aligned = ((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k)
            | reinterpret_cast<uintptr_t>(g)) & 15) == 0 && (qs & 7) == 0 && (ks & 7) == 0;
        auto resolved = [&](auto mode, auto swizzle) {
            if (aligned) run(mode, swizzle, std::true_type{}, std::true_type{});
            else run(mode, swizzle, std::true_type{}, std::false_type{});
        };
        if (affine) resolved(std::true_type{}, std::false_type{});
        else if (preswizzle) resolved(std::false_type{}, std::true_type{});
        else resolved(std::false_type{}, std::false_type{});
    } else if (affine) run(std::true_type{}, std::false_type{}, std::false_type{}, std::false_type{});
    else if (preswizzle) run(std::false_type{}, std::true_type{}, std::false_type{}, std::false_type{});
    else run(std::false_type{}, std::false_type{}, std::false_type{}, std::false_type{});
    if (!defer_classification) {
        const int64_t records = int64_t(heads) * tiles;
        if (records)
            classify_prepared_keys<<<(records + 7) / 8, 256, 0, stream>>>(
                static_cast<BF16 *>(ws), static_cast<const BF16 *>(keys),
                tokens, heads, sequences, tiles, cu, aggregate_flags, tile_flags);
    }
    return prefix;
}
