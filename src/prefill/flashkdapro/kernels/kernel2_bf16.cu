// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2026 MoonshotAI
// SPDX-License-Identifier: BSD-3-Clause AND MIT
// Adapted from FlashKDA.

#include "prefill/flashkdapro/common/launch_api.h"
#include <type_traits>
#include <algorithm>
#define ACOMPUTE_VERSION 10500
#include "prefill/flashkdapro/common/tc02/memory.cuh"
#include "conditioned_repair.cuh"

namespace flashkda_bf16 {
using namespace flashkda_paired;
using namespace flashkda;

enum class Pass { Serial, Prescan, Segment };

struct EmptyTile {
    alignas(128) BF16 storage[1];
    CUTE_DEVICE BF16 *data() { return storage; }
};
template <int Rows> struct InitialStorage {
    alignas(128) BF16 data[Rows * 136];
};
template <int Rows, int Stages, bool Buffered> struct OutputStorage {};
template <int Rows, int Stages> struct OutputStorage<Rows, Stages, true> {
    alignas(128) BF16 output[Stages][16 * (Rows + 8)];
    SyncOutPipeline<Stages, 32 * (Rows / 16), 32> output_pipe;
};

struct EmptyRawInput {};
struct RawInput {
    alignas(128) BF16 inverse[16 * 24];
    alignas(128) float beta[16];
};
template <int Rows, int Stages, bool Prescan, bool Homogeneous, bool HasInitial,
          bool Combined, bool Buffered, bool RawFP32 = false> struct SharedStorage : OutputStorage<Rows, Stages, Buffered> {
    static constexpr int KeyRows = Prescan ? 16 : 32;
    using ValueTile = std::conditional_t<Homogeneous, EmptyTile,
        AiuTile<BF16, 16, 64, (Rows + 63) / 64>>;
    struct SplitInput {
        AiuTile<BF16, KeyRows, 64, 2> kq;
        ValueTile v;
        alignas(128) BF16 kr[16 * 128];
    };
    struct CombinedInput {
        AiuTile<BF16, KeyRows + 16, 64, 2> kq;
        ValueTile v;
    };
    static_assert(sizeof(SplitInput) == sizeof(CombinedInput));
    // Align stage bases and strides to the TC02 1 KiB TSM group span.
    struct alignas(1024) Stage {
        // Output passes consume adjacent Kd, Qd and Kr fields together. The
        // no-output prescan keeps separate loads to avoid fetching unused Qd.
        std::conditional_t<Combined, CombinedInput, SplitInput> input;
        alignas(128) BF16 mqk[Prescan ? 1 : 16 * 24];
        alignas(128) float gt[128];
        std::conditional_t<RawFP32, RawInput, EmptyRawInput> raw;
    };
    // Empty raw storage must fit inside the existing stage alignment padding.
    static_assert(RawFP32 || sizeof(Stage) ==
        ((offsetof(Stage, gt) + 128 * sizeof(float) + 1023) / 1024) * 1024);
    static_assert(!HasInitial || sizeof(InitialStorage<Rows>) <= sizeof(Stage) * Stages);
    union {
        Stage in[Stages];
        std::conditional_t<HasInitial, AiuTile<BF16, Rows, 64, 2>, EmptyTile> initial;
        InitialStorage<HasInitial ? Rows : 1> initial_linear;
    };
    AsyncInPipeline<Stages, 32, 32 * (Rows / 16)> pipe;
    // CTA vote scratch is reused before the recurrence pipeline starts.
    int nonzero[Rows / 16 + 2];
};

struct Arguments {
    BF16 const *v, *ws;
    BF16 *out;
    void const *initial;
    void *final;
    int32_t const *input_indices, *output_indices, *cu, *segments, *tile_prefix, *direct_requests;
    int32_t *zero_slices;
    BF16 *checkpoint;
    BF16 *prepared_carries;
    const BF16 *bias_final;
    int tokens, heads, sequences, tiles;
    int64_t v_stride;
    bool fp32;
    PreparedClassification classification;
    ConditionedArguments repair;
};

CUTE_DEVICE int logical_sequence(PreparedClassification const& c, int segment) {
    int seq = 0, hi = c.sequences;
    while (seq + 1 < hi) {
        const int mid = (seq + hi) / 2;
        if (c.sequence_segments[2 * mid] <= segment) seq = mid;
        else hi = mid;
    }
    return seq;
}

// Resolve one normalized key pair; request aggregation stays with its caller.
template <bool Publish = true>
CUTE_DEVICE bool classify_record(Arguments const& a, int64_t record, int lane) {
    auto* flag = reinterpret_cast<int32_t*>(const_cast<BF16*>(a.ws)
        + record * WS_SLOT_UNITS + WS_OFF_CONDITION_U);
    const int pending = a.classification.tile_flags ? a.classification.tile_flags[record] : *flag;
    bool selected = (pending & 1) != 0;
    if (!selected && (pending & 2))
        selected = conditioned::exact_structural_pair(
            static_cast<const BF16*>(a.classification.keys) + record * 256, 2, lane);
    if constexpr (Publish)
        if (lane == 0) *flag = int(selected);
    return selected;
}

CUTE_DEVICE void classify_segment(Arguments const &a, int base, int count, int segment,
                                  int head, int warp, int lane, int warps) {
    auto const &c = a.classification;
    for (int tile = warp; tile < count; tile += warps) {
        const int64_t record = int64_t(head) * a.tiles + base + tile;
        auto *flag = reinterpret_cast<int32_t *>(const_cast<BF16 *>(a.ws)
            + record * WS_SLOT_UNITS + WS_OFF_CONDITION_U);
        const int pending = c.tile_flags ? c.tile_flags[record] : *flag;
        bool selected = (pending & 1) != 0;
        if (!selected && (pending & 2))
            selected = conditioned::exact_structural_pair(
                static_cast<BF16 const *>(c.keys) + record * 256, 2, lane);
        if (lane != 0) continue;
        *flag = int(selected);
        if (!selected) continue;
        atomicOr(c.flags + logical_sequence(c, segment) * a.heads + head, 1);
    }
}

// A pre-swizzled [16,128] W field becomes B=[128,16] without a shared transpose.
// The atom's last two positional arguments are cube-in-stage and stage.
template <class Fragment>
CUTE_DEVICE void load_w(Fragment &f, int base, int block) {
    using Atom = cute::PPU0015_TSM_LD_SWZL<BF16, 16, 32, true, true, 1>;
    static_assert(Atom::lbo == 1 && Atom::sbo == 32 && Atom::swzl_mode == 1);
    int const address = base + (block / 2) * 64 + (block % 2) * 2;
    int *r = reinterpret_cast<int *>(&f(0));
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans "
                 "{%0, %1, %2, %3}, [%4], %5, %6, %7;"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "l"(address), "r"(1), "r"(32), "n"(1));
}

// Each MMA warp owns 16 value rows. State stays in A fragments; as_B() changes
// only the register view for Q @ S^T and O @ residual^T. One dedicated warp
// fills the next shared stages while the consumers advance the recurrence.
template <Pass Mode, int Rows, int Stages, bool Homogeneous, bool RowMajor, bool Reuse, bool RawFP32 = false>
using RecurrenceStorage = SharedStorage<Rows, Stages,
    Mode == Pass::Prescan && !Reuse, Homogeneous, Mode != Pass::Prescan || Reuse,
    RowMajor && (Mode != Pass::Prescan || Reuse), Mode == Pass::Serial, RawFP32>;

CUTE_DEVICE int serial_tile_base(const Arguments& a, int seq, int bos, int eos) {
    int tile_base = 0;
    if (a.tile_prefix) {
        tile_base = a.tile_prefix[seq];
    } else if (a.cu) {
        for (int i = 0; i < seq; ++i)
            tile_base += (a.cu[i + 1] - a.cu[i] + 15) / 16;
    } else {
        tile_base = seq * ((eos - bos + 15) / 16);
    }
    return tile_base;
}

template <Pass Mode, int Rows, int Stages, bool Homogeneous = false, bool RowMajor = true, bool Reuse = false, bool FullTiles = false, bool SerialRepair = false, int PublicState = -1, bool RawFP32 = false, bool Verified = false, bool PrefixOnly = false, bool Track = false>
CUTE_DEVICE void recurrence_body(std::conditional_t<SerialRepair && !RawFP32, Arguments const&, Arguments> a,
    RecurrenceStorage<Mode, Rows, Stages, Homogeneous, RowMajor, Reuse, RawFP32>& sm, int seq, int value_slice, int summary_head = 0, const TrackState* track = nullptr) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    using namespace cute;
    // Nonnegative PublicState is dispatched only when both public states
    // exist. Specialize their I/O while retaining BF16 resident arithmetic.
    const bool public_state_fp32 = PublicState >= 0 ? bool(PublicState) : a.fp32;
    const bool has_initial = PublicState >= 0 || a.initial != nullptr;
    const bool has_final = PublicState >= 0 || a.final != nullptr;
    constexpr int Workers = Rows / 16;
    constexpr bool Outputs = Mode != Pass::Prescan || Reuse;
    constexpr int Prefix = flashkda::KDA_REPLAY_PREFIX_TILES;
    constexpr int ZeroCheckPeriod = 4;
    static_assert(KDA_REPLAY_MIN_TILES % ZeroCheckPeriod == 0);
    constexpr bool Buffered = Mode == Pass::Serial;
    constexpr bool Combined = RowMajor && Outputs;
    constexpr int KeyRows = Outputs ? 32 : 16;
    constexpr int PreparedRows = KeyRows + (Combined ? 16 : 0);
    static_assert(WS_OFF_KR_U == 32 * 128);
    int const tid = threadIdx.x;
    int const lane = Verified ? int(unsigned(threadIdx.x) & 31u) : tid % 32;
    int const warp = cutlass::canonical_warp_idx_sync();
    int const head = Verified && Mode == Pass::Prescan ? summary_head : int(blockIdx.y);
    constexpr bool homogeneous = Mode == Pass::Prescan && Homogeneous;
    int bos, eos, tile_base, output_slot, input_slot;
    if constexpr (Mode == Pass::Serial) {
        bos = a.cu ? a.cu[seq] : seq * (a.tokens / a.sequences);
        eos = a.cu ? a.cu[seq + 1] : (seq + 1) * (a.tokens / a.sequences);
        tile_base = serial_tile_base(a, seq, bos, eos);
        input_slot = a.input_indices ? a.input_indices[seq] : seq;
        output_slot = a.output_indices ? a.output_indices[seq] : seq;
    } else {
        int const *d = a.segments + seq * KDA_SEG_STRIDE;
        bos = d[KDA_SEG_TOK];
        eos = bos + d[KDA_SEG_LEN];
        tile_base = d[KDA_SEG_WSBASE];
        if constexpr (Mode == Pass::Prescan) {
            // Reuse needs a homogeneous eligibility proof for the last segment
            // too. The ordinary bias pass still leaves its unused summary alone.
            output_slot = int(homogeneous) * a.sequences + seq;
        } else {
            int const logical = d[KDA_SEG_OUTSLOT];
            output_slot = Verified ? seq
                : logical >= 0 && a.output_indices ? a.output_indices[logical] : logical;
        }
        input_slot = seq;
        if constexpr (Verified && Mode == Pass::Prescan) {
            const int logical = logical_sequence(a.classification, seq);
            input_slot = a.repair.indices ? a.repair.indices[logical] : logical;
        }
        if constexpr (Verified && Mode == Pass::Segment) input_slot = seq - 1;
    }
    int const nt = (eos - bos + 15) / 16;
    const int track_boundary = Track ? conditioned::track_token(*track, seq, eos - bos) : -1;
    if constexpr (SerialRepair) {
        if (nt == 0) {
            conditioned::copy_empty_values(a.repair, seq, head, value_slice * Rows,
                                           Rows, tid, 32 * (Workers + 2));
            return;
        }
        // Each CTA independently classifies its complete request. The loop
        // also handles mutable device offsets that outgrow the host hint.
        bool selected = Track && eos - bos <= KDA_DIRECT_MAX_TOKENS;
        for (int tile = warp; tile < nt && !(Track && eos - bos <= KDA_DIRECT_MAX_TOKENS); tile += Workers + 2) {
            const int64_t record = int64_t(head) * a.tiles + tile_base + tile;
            const int pending = a.classification.tile_flags ? a.classification.tile_flags[record]
                : *reinterpret_cast<const int32_t*>(a.ws
                    + record * WS_SLOT_UNITS + WS_OFF_CONDITION_U);
            // The serial producer resolves the identical structural test
            // before its normalized key tile is overwritten. Read one bit
            // here instead of repeating the predicate in every value CTA.
            selected |= (pending & 1) != 0;
        }
        if (lane == 0) sm.nonzero[warp] = int(selected);
        __syncthreads();
        for (int i = 0; i < Workers + 2; ++i) selected |= sm.nonzero[i] != 0;
        if (selected) {
            // Valid mandatory short repairs are handled by recurrence() from
            // the producer's prepared tokens. Other selected requests retain
            // the same standalone precision arithmetic, without a second cache.
            const int first_row = value_slice * Rows;
            for (int row = first_row + warp; row < first_row + Rows; row += Workers + 2)
                conditioned::repair_value<std::conditional_t<RawFP32, float, double>, Track>(a.repair, seq, head, row, lane, nullptr, track);
            return;
        }
    }
    if constexpr (homogeneous)
        if (a.classification.keys)
            classify_segment(a, tile_base, nt, seq, head, warp, lane, Workers + 1);
    if constexpr (Mode != Pass::Serial && !Homogeneous && !Verified) {
        const bool scheduled_repair = Mode == Pass::Segment
            ? a.classification.flags != nullptr : a.repair.q != nullptr;
        if (scheduled_repair) {
            const int logical = logical_sequence(a.classification, seq);
            if (a.classification.flags[logical * a.heads + head]) {
                if constexpr (Mode == Pass::Prescan && !Reuse) {
                    const int first = a.classification.sequence_segments[2 * logical];
                    const int count = a.classification.sequence_segments[2 * logical + 1];
                    // Planner admission assigns each value row to one segment
                    // warp, without serializing repair tasks.
                    const int value_warps = (128 + count - 1) / count;
                    const int row = (seq - first) * value_warps + warp;
                    if (warp < value_warps && row < 128)
                        conditioned::repair_value(a.repair, logical, head, row, lane);
                }
                return;
            }
        }
    }
    if constexpr (Mode == Pass::Prescan && !Verified)
        if (a.segments[seq * KDA_SEG_STRIDE + KDA_SEG_OUTSLOT] >= 0
                && (!a.checkpoint || (!Homogeneous && !Reuse))) return;
    if (nt == 0) return;
    bool checkpoint_enabled = Reuse;
    if constexpr (Mode != Pass::Serial && !Homogeneous && !Verified) {
        const bool eligible = a.checkpoint && nt >= KDA_REPLAY_MIN_TILES
            && (a.zero_slices[(seq*a.heads+head)*2]&2)
            && (a.zero_slices[(seq*a.heads+head)*2+1]&2);
        if constexpr (Reuse && Mode == Pass::Segment) {
            // One replay launch owns both eligible and ordinary segments.
            // Ineligible segments run to completion without a checkpoint read.
            checkpoint_enabled = eligible;
        } else if constexpr (Reuse) {
            if (!eligible) return;
        } else if (eligible) return;
    }
    int const dv_cta = Mode == Pass::Serial || PrefixOnly ? value_slice * Rows : 0;
    // The masked MMA coordinate exposes the same legal worker range as
    // FlashKDA, without specializing any public tensor dimensions.
    int const worker = Verified ? (warp & (Workers - 1)) : warp;
    int const dv = dv_cta + worker * 16;
    auto mma = make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{},
                              Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    Tensor ref = make_tensor(make_smem_ptr(sm.in[0].mqk),
                             Layout<Shape<_16, _16>, Stride<Int<24>, _1>>{});
    using AF = decltype(thr.partition_fragment_A(ref));
    using BF = decltype(thr.partition_fragment_B(ref));
    using AC = decltype(thr.make_fragment_C(thr.partition_C(ref)));
    AF state[8];
    AC master[RawFP32 ? 8 : 1];
    static_assert(!RawFP32 || Mode == Pass::Serial);
    // Stage public states cooperatively before the input pipeline reuses the
    // same storage. Unaligned views keep the elementwise loading path.
    const bool staged_initial = (Mode == Pass::Segment && (!Verified || PrefixOnly))
        || (Verified && Mode == Pass::Prescan && has_initial && !public_state_fp32 && input_slot >= 0
            && (reinterpret_cast<uintptr_t>(a.initial) & 127) == 0)
        || (Mode == Pass::Serial && has_initial && !public_state_fp32 && input_slot >= 0
            && (reinterpret_cast<uintptr_t>(a.initial) & 127) == 0);
    if (staged_initial) {
        if (warp == 0) {
            AiuTileLoader<BF16,Rows,64,2> loader;
            loader.init(static_cast<const BF16 *>(a.initial)
                +(int64_t(input_slot)*a.heads+head) * 128 * 128
                +int64_t(dv_cta)*128,Rows,128,128);
            loader.load(sm.initial.data(),0);
            cp_async_fence_all();
            cp_async_wait_all<0>();
        }
        __syncthreads();
        if (warp < Workers) {
            const int base=__ppu_read_firstlane(unsigned(uintptr_t(sm.initial.data())/16));
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k)
                TsmOps<Rows,64,2>::load_a(&state[k](0),base,k,worker*16);
        }
    } else if (Mode == Pass::Serial && has_initial && !public_state_fp32 && input_slot >= 0
               && (reinterpret_cast<uintptr_t>(a.initial) & 15) == 0) {
        const int64_t base = (int64_t(input_slot) * a.heads + head) * 128 * 128
            + int64_t(dv_cta) * 128;
        for (int packet = tid; packet < Rows * 128 / 8; packet += 32 * (Workers + 2)) {
            const int row = packet / 16, col = (packet % 16) * 8;
            const int64_t off = base + row * 128 + col;
            const uint4 values = *reinterpret_cast<const uint4 *>(
                static_cast<const BF16 *>(a.initial) + off);
            *reinterpret_cast<uint4 *>(sm.initial_linear.data + row * 136 + col) = values;
        }
        __syncthreads();
        if (warp < Workers) {
            auto matrix = make_tensor(make_smem_ptr(sm.initial_linear.data),
                Layout<Shape<Int<Rows>, _128>, Stride<Int<136>, _1>>{});
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k) {
                auto tile = local_tile(matrix, Shape<_16, _16>{}, make_coord(warp, k));
                copy(AutoVectorizingCopy{}, thr.partition_A(tile), state[k]);
            }
        }
    } else {
    if (warp < Workers && (Mode == Pass::Serial || PrefixOnly || (Verified && Mode == Pass::Prescan)) && has_initial && public_state_fp32 && input_slot >= 0
        && (reinterpret_cast<uintptr_t>(a.initial) & 7) == 0) {
        // A/C fragment value pairs occupy adjacent columns. Load aligned
        // public FP32 pairs directly and retain the same BF16 rounding.
        CUTE_UNROLL
        for (int k = 0; k < 8; ++k) {
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(state[0]))::value / 2; ++j) {
                const auto c = frag_coord_mn(lane, 2 * j);
                const int64_t off = (int64_t(input_slot) * a.heads + head) * 128 * 128
                    + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                const float2 value = *reinterpret_cast<const float2*>(
                    static_cast<const float*>(a.initial) + off);
                const auto packed = __floats2bfloat162_rn(value.x, value.y);
                *reinterpret_cast<uint32_t*>(&state[k](2 * j)) =
                    *reinterpret_cast<const uint32_t*>(&packed);
            }
        }
    } else if (Verified && Mode == Pass::Segment && !PrefixOnly && warp < Workers
               && has_initial && input_slot >= 0) {
        // Ordered repair reads the preceding B slot directly. Its first
        // segment is already true; private state words are naturally aligned.
        CUTE_UNROLL
        for (int k = 0; k < 8; ++k) {
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(state[0]))::value / 2; ++j) {
                const auto c = frag_coord_mn(lane,2*j);
                const int64_t off = (int64_t(input_slot)*a.heads+head)*128*128
                    + int64_t(dv+c.m)*128+k*16+c.n;
                *reinterpret_cast<uint32_t*>(&state[k](2*j)) =
                    *reinterpret_cast<const uint32_t*>(static_cast<const BF16*>(a.initial)+off);
            }
        }
    } else if (warp < Workers) {
        CUTE_UNROLL
        for (int k = 0; k < 8; ++k) {
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(state[0]))::value; ++j) {
                FragCoord const c = frag_coord_mn(lane, j);
                int64_t const off = (int64_t(input_slot) * a.heads + head) * 128 * 128
                    + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                float x = 0.f;
                if constexpr (Mode == Pass::Prescan) {
                    x = homogeneous && dv + c.m == k * 16 + c.n ? 1.f : 0.f;
                    if constexpr (Verified) {
                        if (has_initial && input_slot >= 0)
                            x = public_state_fp32 ? static_cast<float const *>(a.initial)[off]
                                : float(static_cast<BF16 const *>(a.initial)[off]);
                    }
                } else if (has_initial && input_slot >= 0) {
                    if constexpr (Mode == Pass::Segment && !PrefixOnly)
                        x = float(static_cast<BF16 const *>(a.initial)[off]);
                    else
                        x = public_state_fp32 ? static_cast<float const *>(a.initial)[off]
                                   : float(static_cast<BF16 const *>(a.initial)[off]);
                }
                state[k](j) = BF16(x);
            }
        }
    }
    }
    if constexpr (RawFP32) {
        if (warp < Workers) {
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k) {
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(state[0]))::value; ++j) {
                    const auto c = frag_coord_mn(lane, j);
                    const int64_t off = (int64_t(input_slot) * a.heads + head) * 128 * 128
                        + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                    master[k](j) = has_initial && public_state_fp32 && input_slot >= 0
                        ? static_cast<const float*>(a.initial)[off] : float(state[k](j));
                }
            }
        }
    }
    sm.pipe.init(tid);
    __syncthreads();
    if (warp == Workers) {
        AiuTileLoader<BF16, PreparedRows, 64, 2> lkq;
        AiuTileLoader<BF16, 16, 64, (Rows + 63) / 64> lv;
        lv.init(a.v + head * 128 + (dv_cta / 64) * 64, eos, 128 - (dv_cta / 64) * 64,
                a.v_stride);
        int issued = 0;
        for (int t = 0; t < nt; ++t) {
            int const st = t % Stages;
            sm.pipe.producer_acquire(st, t);
            auto &z = sm.in[st];
            BF16 const *slot = a.ws + (int64_t(head) * a.tiles + tile_base + t) * WS_SLOT_UNITS;
            lkq.init(slot, PreparedRows, 128, 128);
            lkq.load(z.input.kq.data(), 0);
            if (!homogeneous) lv.load(z.input.v.data(), bos + 16 * t);
            if constexpr (!Combined && RowMajor) {
                AiuTileLoader<BF16, 16, 64, 2> lw;
                lw.init(slot + WS_OFF_KR_U, 16, 128, 128);
                lw.load(z.input.kr, 0);
            } else if constexpr (!Combined) {
                cp_async_contiguous<32>(z.input.kr, slot + WS_OFF_KR_U, 16 * 128 * sizeof(BF16), lane);
            }
            if constexpr (Outputs)
                cp_async_tile<16, 16, 24, 32>(z.mqk, slot + WS_OFF_MQK_U, 16, 16, lane);
            cp_async_contiguous<32>(z.gt, slot + WS_OFF_GT_U, 128 * sizeof(float), lane);
            if constexpr (RawFP32) {
                cp_async_tile<16, 16, 24, 32>(z.raw.inverse, slot + WS_OFF_INV_U, 16, 16, lane);
                cp_async_contiguous<32>(z.raw.beta, slot + WS_OFF_BETA_U, 16 * sizeof(float), lane);
            }
            cp_async_fence_all();
            kda_fence_block();
            sm.pipe.producer_commit(st);
            if constexpr (Verified) issued = t + 1;
            if constexpr (Mode == Pass::Segment && Reuse) {
                // Do not submit any tail stage before the full-state comparison.
                // At this rendezvous every issued copy has a matching consumer.
                if (checkpoint_enabled && t + 1 == Prefix) {
                    kda_fence_block();
                    cutlass::arch::NamedBarrier::arrive_and_wait(32 * (Workers + 1), kOutReady);
                    int mismatch = 0;
                    CUTE_UNROLL
                    for (int w = 0; w < Workers; ++w) mismatch |= sm.nonzero[w];
                    if (!mismatch || PrefixOnly) break;
                }
            }
            if constexpr (homogeneous) {
                // The producer stops at the same checkpoint as every consumer,
                // so no future asynchronous stage can be left outstanding.
                if ((t % ZeroCheckPeriod) == ZeroCheckPeriod - 1 || t + 1 == nt) {
                    kda_fence_block();
                    cutlass::arch::NamedBarrier::arrive_and_wait(32 * (Workers + 1), kOutReady);
                    int alive = 0;
                    CUTE_UNROLL
                    for (int w = 0; w < Workers; ++w) alive |= sm.nonzero[w];
                    if (!alive) break;
                }
            }
        }
        if constexpr (Verified && Mode == Pass::Segment && !PrefixOnly) {
            // Retire the final stage-free arrivals before another segment
            // reinitializes this CTA's pipeline and reuses its barrier IDs.
            for (int t = max(0, issued - Stages); t < issued; ++t)
                sm.pipe.producer_acquire(t % Stages, t + Stages);
        }
    }
    if constexpr (Buffered) {
        if (warp == Workers + 1) {
            for (int t = 0; t < nt; ++t) {
                const int st = t % Stages;
                sm.output_pipe.consumer_wait(st, t);
                // Aggregate the MMA fragments into aligned 16-byte stores.
                // Tail rows remain masked in the public output tensor.
                for (int packet = lane; packet < 16 * Rows / 8; packet += 32) {
                    const int row = packet / (Rows / 8), col = (packet % (Rows / 8)) * 8;
                    const int64_t token = int64_t(bos) + int64_t(t) * 16 + row;
                    if (token < eos) {
                        const int64_t off = token * a.heads * 128
                            + head * 128 + dv_cta + col;
                        *reinterpret_cast<uint4 *>(a.out + off) =
                            *reinterpret_cast<const uint4 *>(sm.output[st] + row * (Rows + 8) + col);
                    }
                }
                kda_fence_block();
                sm.output_pipe.consumer_release(st);
            }
        }
    }
    if (warp < Workers) {
        bool zero = false;
        bool early_zero = false;
        bool reused = false;
        // Hoist the generic head stride and fragment coordinates out of the
        // recurrence. Keep an integer offset so masked tail rows never form
        // an out-of-bounds pointer, even when they issue no store.
        const int64_t output_stride = int64_t(a.heads) * 128;
        int64_t output_offset = 0;
        if constexpr (Outputs && Mode != Pass::Serial)
            output_offset = int64_t(bos + (lane / 4) % 8) * output_stride
                + head * 128 + dv + 2 * (lane % 4);
        for (int t = 0; t < nt; ++t) {
            int const st = t % Stages;
            if constexpr (Buffered) sm.output_pipe.producer_acquire(st, t);
            sm.pipe.consumer_wait(st, t);
            auto &z = sm.in[st];
            if (!zero) {
                AC r, o;
                clear(r);
                clear(o);
                int const kqbase = __ppu_read_firstlane(unsigned(uintptr_t(z.input.kq.data()) / 16));
                int wbase = 0;
                if constexpr (!Combined)
                    wbase = __ppu_read_firstlane(unsigned(__cvta_generic_to_shared(z.input.kr) / 16));
                CUTE_UNROLL
                for (int k = 0; k < 8; ++k) {
                    BF kb;
                    TsmOps<PreparedRows, 64, 2>::load_b(&kb(0), kqbase, k, 0);
                    cute::gemm(thr, state[k], kb, r);
                    if constexpr (Outputs) {
                        AF qa;
                        TsmOps<PreparedRows, 64, 2>::load_a(&qa(0), kqbase, k, 16);
                        cute::gemm(thr, qa, as_B(state[k]), o);
                    }
                }
                BF vt;
                if (!homogeneous) {
                    int const vb = __ppu_read_firstlane(unsigned(uintptr_t(z.input.v.data()) / 16));
                    TsmOps<16, 64, (Rows + 63) / 64>::load_bt(
                        &vt(0), vb, (dv_cta % 64) / 16 + worker, 0);
                } else {
                    clear(vt);
                }
                auto vv = as_A(vt);
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(r))::value; ++j) {
                    r(j) = float(vv(j)) - r(j);
                    if constexpr (RawFP32) r(j) *= z.raw.beta[frag_coord_mn(lane, j).n];
                }
                AF residual;
                narrow_into(residual, r);
                if constexpr (RawFP32) {
                    // R[DV,Tin] @ INV[Tout,Tin]^T -> U[DV,Tout].
                    Tensor inverse = make_tensor(make_smem_ptr(z.raw.inverse),
                        Layout<Shape<_16, _16>, Stride<Int<24>, _1>>{});
                    BF inverse_b = thr.partition_fragment_B(inverse);
                    copy(AutoVectorizingCopy{}, thr.partition_B(inverse), inverse_b);
                    AC update;
                    clear(update);
                    cute::gemm(thr, residual, inverse_b, update);
                    narrow_into(residual, update);
                }
                if constexpr (Outputs) {
                    Tensor m = make_tensor(make_smem_ptr(z.mqk),
                        Layout<Shape<_16, _16>, Stride<Int<24>, _1>>{});
                    AF ma = thr.partition_fragment_A(m);
                    copy(AutoVectorizingCopy{}, thr.partition_A(m), ma);
                    cute::gemm(thr, ma, as_B(residual), o);
                    CUTE_UNROLL
                    for (int j = 0; j < decltype(size(o))::value / 2; ++j) {
                        FragCoord const c = frag_coord_mn(lane, 2 * j);
                        BF16 pair[2] = {BF16(o(2 * j)), BF16(o(2 * j + 1))};
                        if constexpr (Buffered) {
                            *reinterpret_cast<uint32_t *>(sm.output[st]
                                + c.m * (Rows + 8) + warp * 16 + c.n) =
                                *reinterpret_cast<const uint32_t *>(pair);
                        } else {
                            static_assert(decltype(size(o))::value == 8);
                            const int64_t off = output_offset
                                + ((j & 1) ? 8 * output_stride : 0) + ((j & 2) ? 8 : 0);
                            if (FullTiles || bos + t * 16 + c.m < eos)
                                *reinterpret_cast<uint32_t *>(a.out + off) =
                                    *reinterpret_cast<const uint32_t *>(pair);
                        }
                    }
                }
                bool prefix_mismatch = false;
                CUTE_UNROLL
                for (int k = 0; k < 8; ++k) {
                    BF w;
                    if constexpr (Combined)
                        TsmOps<PreparedRows, 64, 2>::load_bt(&w(0), kqbase, k, KeyRows);
                    else if constexpr (RowMajor)
                        TsmOps<16, 64, 2>::load_bt(&w(0), wbase, k);
                    else
                        load_w(w, wbase, k);
                    AC delta;
                    clear(delta);
                    cute::gemm(thr, residual, w, delta);
                    // Keep each already-rounded BF16 pair in one register word.
                    // Decay, FMA accumulation and rounding remain FP32/BF16 as
                    // before; checkpoints consume the same word without another
                    // walk over all eight live state fragments.
                    auto words = cute::recast<uint32_t>(state[k]);
                    CUTE_UNROLL
                    for (int j = 0; j < decltype(size(delta))::value / 2; ++j) {
                        const auto c = frag_coord_mn(lane, 2 * j);
                        float x, y;
                        if constexpr (RawFP32) {
                            x = fmaf(master[k](2 * j), z.gt[k * 16 + c.n], delta(2 * j));
                            y = fmaf(master[k](2 * j + 1), z.gt[k * 16 + c.n + 1], delta(2 * j + 1));
                            master[k](2 * j) = x;
                            master[k](2 * j + 1) = y;
                        } else {
                            x = fmaf(float(state[k](2 * j)), z.gt[k * 16 + c.n], delta(2 * j));
                            y = fmaf(float(state[k](2 * j + 1)), z.gt[k * 16 + c.n + 1], delta(2 * j + 1));
                        }
                        const auto updated = __floats2bfloat162_rn(x, y);
                        const uint32_t word = *reinterpret_cast<const uint32_t *>(&updated);
                        words(j) = word;
                    }

                }
                if constexpr (Track) {
                    if ((t + 1) * 16 == track_boundary) {
                        CUTE_UNROLL
                        for (int k = 0; k < 8; ++k) {
                            CUTE_UNROLL
                            for (int j = 0; j < decltype(size(state[k]))::value; ++j) {
                                const auto c = frag_coord_mn(lane, j);
                                const int64_t offset = ((int64_t(seq) * a.heads + head) * 128 + dv + c.m) * 128 + k * 16 + c.n;
                                if constexpr (RawFP32) track->state[offset] = master[k](j);
                                else track->state[offset] = float(state[k](j));
                            }
                        }
                    }
                }
                if constexpr (Reuse) {
                    if (checkpoint_enabled && t + 1 == Prefix) {
                        // State fragments remain packed after their update.
                        // Test the checkpoint boundary once, then copy/compare
                        // every bit owned by this lane, including signed zeros.
                        static_assert(Rows == 128 || PrefixOnly);
                        const int64_t base = (int64_t(Verified ? a.sequences + seq : seq) * a.heads + head) * 128 * 128
                            + (dv_cta / 16 * 32 + tid) * 8;
                        CUTE_UNROLL
                        for (int k = 0; k < 8; ++k) {
                            auto words = cute::recast<uint32_t>(state[k]);
                            auto* buffer = Verified ? const_cast<BF16*>(a.bias_final) : a.checkpoint;
                            auto *packet = reinterpret_cast<uint4 *>(buffer + base + k * (128 * 16));
                            if constexpr (Mode == Pass::Prescan) {
                                *packet = make_uint4(words(0), words(1), words(2), words(3));
                            } else {
                                const uint4 saved = *packet;
                                prefix_mismatch |= ((words(0) ^ saved.x) | (words(1) ^ saved.y)
                                    | (words(2) ^ saved.z) | (words(3) ^ saved.w)) != 0;
                            }
                        }
                    }
                }
                if constexpr (Mode == Pass::Segment && Reuse) {
                    if (checkpoint_enabled && t + 1 == Prefix) {
                        const bool different = __any_sync(0xffffffff, prefix_mismatch);
                        if (lane == 0) sm.nonzero[warp] = different;
                    }
                }
                if constexpr (homogeneous) {
                    // Inspect every state word only at the matched producer
                    // rendezvous. Zero is still an exact BF16 magnitude test.
                    if ((t % ZeroCheckPeriod) == ZeroCheckPeriod - 1 || t + 1 == nt) {
                        uint32_t nonzero = 0;
                        CUTE_UNROLL
                        for (int k = 0; k < 8; ++k) {
                            auto words = cute::recast<uint32_t>(state[k]);
                            CUTE_UNROLL
                            for (int j = 0; j < decltype(size(words))::value; ++j)
                                nonzero |= words(j);
                        }
                        zero = __all_sync(0xffffffff, (nonzero & 0x7fff7fffu) == 0);
                        early_zero |= zero && t + 1 <= KDA_REPLAY_MIN_TILES;
                    }
                }
            }
            if constexpr (Outputs && Mode != Pass::Serial)
                output_offset += 16 * output_stride;
            if constexpr (Buffered) {
                kda_fence_block();
                sm.output_pipe.producer_commit(st);
            }
            // Even an exactly-zero homogeneous row must consume every stage.
            sm.pipe.consumer_release(st);
            if constexpr (Mode == Pass::Segment && Reuse) {
                if (checkpoint_enabled && t + 1 == Prefix) {
                    kda_fence_block();
                    cutlass::arch::NamedBarrier::arrive_and_wait(32 * (Workers + 1), kOutReady);
                    int mismatch = 0;
                    CUTE_UNROLL
                    for (int w = 0; w < Workers; ++w) mismatch |= sm.nonzero[w];
                    reused = !mismatch;
                    if (reused || PrefixOnly) break;
                }
            }
            if constexpr (homogeneous) {
                if ((t % ZeroCheckPeriod) == ZeroCheckPeriod - 1 || t + 1 == nt) {
                    if (lane == 0) sm.nonzero[warp] = !zero;
                    kda_fence_block();
                    cutlass::arch::NamedBarrier::arrive_and_wait(32 * (Workers + 1), kOutReady);
                    int alive = 0;
                    CUTE_UNROLL
                    for (int w = 0; w < Workers; ++w) alive |= sm.nonzero[w];
                    if (!alive) break;
                }
            }
        }
        int speculative_slot = -1;
        if constexpr (Verified && Mode == Pass::Prescan) {
            const int logical = a.segments[seq * KDA_SEG_STRIDE + KDA_SEG_OUTSLOT];
            if (logical >= 0)
                speculative_slot = a.repair.indices ? a.repair.indices[logical] : logical;
        }
        bool bias_copyable = true;
        BF16 *carry_target = nullptr;
        if constexpr (Mode == Pass::Prescan && !Homogeneous && !Verified)
            carry_target = a.segments[seq * KDA_SEG_STRIDE + KDA_SEG_OUTSLOT] < 0
                ? a.prepared_carries + int64_t(a.heads) * 128 * 128
                : static_cast<BF16 *>(a.final);
        // Prescan always owns allocated chain/carry storage. Other passes
        // need no final-state walk, including bias reads, for inactive slots.
        if (!PrefixOnly && (Mode == Pass::Prescan || (has_final && output_slot >= 0))) {
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k) {
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(state[0]))::value / 2; ++j) {
                    FragCoord const c = frag_coord_mn(lane, 2 * j);
                    int64_t const off = (int64_t(output_slot) * a.heads + head) * 128 * 128
                        + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                    BF16 pair[2] = {state[k](2 * j), state[k](2 * j + 1)};
                    if constexpr (Mode == Pass::Segment && Reuse) {
                        if (reused) {
                            const int64_t src = (int64_t(seq) * a.heads + head) * 128 * 128
                                + int64_t(dv+c.m)*128+k * 16 + c.n;
                            *reinterpret_cast<uint32_t *>(pair) =
                                *reinterpret_cast<const uint32_t *>(a.bias_final + src);
                        }
                    }
                    if constexpr (Mode == Pass::Prescan && !Homogeneous && !Verified) {
                        CUTE_UNROLL
                        for (int e = 0; e < 2; ++e) {
                            const uint16_t bits = reinterpret_cast<const uint16_t*>(pair)[e];
                            const uint16_t exponent = bits & 0x7f80u;
                            bias_copyable &= bits == 0 || (exponent != 0 && exponent != 0x7f80u);
                        }
                        // Publish the exact same stored bias to the next carry
                        // slot while its words are already in registers. Merge
                        // still proves copyability or overwrites every carry.
                        *reinterpret_cast<uint32_t *>(carry_target + off) =
                            *reinterpret_cast<uint32_t const *>(pair);
                    }
                    if constexpr (Mode == Pass::Prescan) {
                        *reinterpret_cast<uint32_t *>(static_cast<BF16 *>(a.final) + off) =
                            *reinterpret_cast<uint32_t const *>(pair);
                        // Publish the hypothesis while its final state is in
                        // registers. A failed checkpoint proof overwrites it.
                        if constexpr (Verified) {
                            if (a.repair.final_state && speculative_slot >= 0) {
                                const int64_t dest = (int64_t(speculative_slot) * a.heads + head) * 128 * 128
                                    + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                                if (a.repair.state_fp32)
                                    *reinterpret_cast<float2 *>(static_cast<float *>(a.repair.final_state) + dest) =
                                        make_float2(float(pair[0]), float(pair[1]));
                                else *reinterpret_cast<uint32_t *>(static_cast<BF16 *>(a.repair.final_state) + dest) =
                                    *reinterpret_cast<uint32_t const *>(pair);
                            }
                        }
                    } else if (public_state_fp32) {
                        *reinterpret_cast<float2 *>(static_cast<float *>(a.final) + off) =
                            RawFP32 ? make_float2(master[k](2 * j), master[k](2 * j + 1))
                                    : make_float2(float(pair[0]), float(pair[1]));
                    } else {
                        *reinterpret_cast<uint32_t *>(static_cast<BF16 *>(a.final) + off) =
                            *reinterpret_cast<uint32_t const *>(pair);
                    }
                }
            }
        }
        if constexpr (Mode == Pass::Prescan && !Homogeneous && !Verified) {
            const bool copyable = __all_sync(0xffffffff, bias_copyable);
            if (lane == 0) sm.nonzero[warp] = copyable;
        }
        if constexpr (Mode == Pass::Prescan) {
            if (homogeneous && lane == 0)
                sm.nonzero[warp] = int(zero) | (int(early_zero) << 1);
        }
    }
    __syncthreads();
    if constexpr (Mode == Pass::Prescan && !Homogeneous && !Verified) {
        if (tid < 2) {
            int normal = 1;
            for (int w = tid * 4; w < tid * 4 + 4; ++w) normal &= sm.nonzero[w];
            const int index = (seq * a.heads + head) * 2 + tid;
            // Preserve the boolean zero proof even for legacy merge callers.
            if ((a.zero_slices[index] & 1) && normal) a.zero_slices[index] |= 4;
        }
    }
    if constexpr (Mode == Pass::Prescan) {
        if (homogeneous && tid < 2) {
            // Bit0 preserves the existing exact-zero merge proof. Bit1 says
            // the complete row slice became exactly zero within the bounded
            // prefix budget, avoiding speculative output work for weak gates.
            int flags = 3;
            for (int i = tid * 4; i < tid * 4 + 4; ++i) flags &= sm.nonzero[i];
            a.zero_slices[(seq * a.heads + head) * 2 + tid] = flags;
        }
    }
#endif
}

template <Pass Mode, int Rows, int Stages, bool Homogeneous = false, bool RowMajor = true, bool Reuse = false, bool FullTiles = false, bool SerialRepair = false, int PublicState = -1, bool RawFP32 = false, bool CompactPrecision = false, bool RepairShort = true>
__global__ __launch_bounds__(32 * (Rows / 16 + 1 + (Mode == Pass::Serial)), 4)
void recurrence(Arguments const a) {
    // The host already validates and records the public state dtype.
    // Keep kernel parameters immutable so device helpers can borrow them.
    __shared__ RecurrenceStorage<Mode, Rows, Stages, Homogeneous, RowMajor, Reuse, RawFP32> storage;
    if constexpr (CompactPrecision) {
        static_assert(Mode == Pass::Serial && SerialRepair);
        constexpr int Warps = Rows / 16 + 2;
        constexpr int Jobs = (128 + Warps - 1) / Warps;
        constexpr int Slices = 128 / Rows;
        const int ordinary = a.sequences * Slices;
        const int work = blockIdx.x, head = blockIdx.y;
        if (work >= ordinary) {
            // Batch exact value rows only when the live precision queue
            // has enough work to fill every available CTA at that width.
            // Sparse queues keep the original one-row latency and grid.
            auto run = [&](auto values) {
                auto* cached = reinterpret_cast<conditioned::RepairToken*>(storage.in);
                static_assert(sizeof(storage.in) >= KDA_DIRECT_MAX_TOKENS * sizeof(conditioned::RepairToken));
                constexpr int Values = decltype(values)::value;
                static_assert(128 % Values == 0);
                constexpr int ValueJobs = (128 + Warps * Values - 1) / (Warps * Values);
                const int count = a.direct_requests[0] * ValueJobs;
                const int stride = int(gridDim.x) - ordinary;
                const int tid = threadIdx.x, lane = tid % 32;
                const int warp = cutlass::canonical_warp_idx_sync();
                for (int task = work - ordinary; task < count; task += stride) {
                    const int seq = a.direct_requests[1 + task / ValueJobs];
                    const int bos = a.cu ? a.cu[seq] : seq * (a.tokens / a.sequences);
                    const int eos = a.cu ? a.cu[seq + 1] : (seq + 1) * (a.tokens / a.sequences);
                    const int tile = serial_tile_base(a, seq, bos, eos);
                    const auto* prepared = reinterpret_cast<const uint64_t*>(a.ws
                        + (int64_t(head) * a.tiles + tile) * WS_SLOT_UNITS);
                    const int packets = (eos - bos) * sizeof(conditioned::RepairToken) / sizeof(uint64_t);
                    for (int packet = tid; packet < packets; packet += 32 * Warps)
                        reinterpret_cast<uint64_t*>(cached)[packet] = prepared[packet];
                    __syncthreads();
                    const int first_value = ((task % ValueJobs) * Warps + warp) * Values;
                    if (first_value < 128)
                        conditioned::repair_cached_values<Values, float>(a.repair, seq, head, first_value, lane, cached);
                    // These precision-only CTAs never enter the asynchronous
                    // recurrence pipeline. A CTA join suffices for cache reuse.
                    if (task + stride < count) __syncthreads();
                }
            };
            const int pending = a.direct_requests[0] * Jobs;
            const int available = int(gridDim.x) - ordinary;
            if (pending >= 4 * available) run(cute::Int<4>{});
            else if (pending >= 2 * available) run(cute::Int<2>{});
            else run(cute::Int<1>{});
            return;
        }
        const int seq = work / Slices, value_slice = work % Slices;
        const int bos = a.cu ? a.cu[seq] : seq * (a.tokens / a.sequences);
        const int eos = a.cu ? a.cu[seq + 1] : (seq + 1) * (a.tokens / a.sequences);
        const bool direct = eos > bos && eos - bos <= KDA_DIRECT_MAX_TOKENS
            && bos >= 0 && eos <= a.tokens && (!a.cu || a.cu[0] == 0);
        if (!direct)
            recurrence_body<Mode, Rows, Stages, Homogeneous, RowMajor, Reuse, FullTiles, SerialRepair, PublicState, RawFP32>(
                a, storage, seq, value_slice);
        return;
    }
    if constexpr (SerialRepair) {
        static_assert(Mode == Pass::Serial);
        constexpr int Warps = Rows / 16 + 2;
        constexpr int Values = 4;
        const int seq = blockIdx.x, head = blockIdx.y;
        const int bos = a.cu ? a.cu[seq] : seq * (a.tokens / a.sequences);
        const int eos = a.cu ? a.cu[seq + 1] : (seq + 1) * (a.tokens / a.sequences);
        // These requests are unconditionally selected by the producer's
        // existing direct-repair guard. Spread value rows across CTAs instead
        // of serializing several groups in each ordinary MMA block.
        const bool direct = eos > bos && eos - bos <= KDA_DIRECT_MAX_TOKENS
            && bos >= 0 && eos <= a.tokens && (!a.cu || a.cu[0] == 0);
        if (direct) {
            if constexpr (RepairShort) if (a.sequences == 1) {
                // The single-request prepare has no producer precision CTAs.
                // Read its exact cache when live device bounds change, keeping
                // ordinary captures free of extra tasks and producer registers.
                constexpr int Warps = Rows / 16 + 2;
                auto* cached = reinterpret_cast<conditioned::RepairToken*>(storage.in);
                static_assert(sizeof(storage.in) >= KDA_DIRECT_MAX_TOKENS * sizeof(conditioned::RepairToken));
                const int tid = threadIdx.x, lane = tid % 32;
                const int warp = cutlass::canonical_warp_idx_sync();
                const int tile = serial_tile_base(a, seq, bos, eos);
                const auto* prepared = reinterpret_cast<const uint64_t*>(a.ws
                    + (int64_t(head) * a.tiles + tile) * WS_SLOT_UNITS);
                const int packets = (eos - bos) * sizeof(conditioned::RepairToken) / sizeof(uint64_t);
                for (int packet = tid; packet < packets; packet += 32 * Warps)
                    reinterpret_cast<uint64_t*>(cached)[packet] = prepared[packet];
                __syncthreads();
                const int first_row = int(blockIdx.z) * Rows;
                for (int row = first_row + warp; row < 128 && row < first_row + Rows; row += Warps)
                    conditioned::repair_cached_values<1, float>(a.repair, seq, head, row, lane, cached);
            }
            return;
        }
        if (int(blockIdx.z) >= 128 / Rows) return;
    }
    recurrence_body<Mode, Rows, Stages, Homogeneous, RowMajor, Reuse, FullTiles, SerialRepair, PublicState, RawFP32>(
        a, storage, int(blockIdx.x), int(blockIdx.z));
}

// One complete CTA selects the original checkpoint or ordinary bias body.
// The shared union preserves the existing layout, copies and arithmetic in
// either family while avoiding the second, mutually exclusive bias launch.
template <bool RowMajor, bool FullTiles>
__global__ __launch_bounds__(288, 4) void hybrid_bias(Arguments a) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
    union Storage {
        RecurrenceStorage<Pass::Prescan,128,2,false,RowMajor,true> checkpoint;
        RecurrenceStorage<Pass::Prescan,128,2,false,RowMajor,false> ordinary;
    };
    __shared__ Storage storage;
    const int segment = blockIdx.x, head = blockIdx.y;
    const int nt = (a.segments[segment * KDA_SEG_STRIDE + KDA_SEG_LEN] + 15) / 16;
    bool repair = false;
    if (a.repair.q) {
        const int logical = logical_sequence(a.classification, segment);
        repair = a.classification.flags[logical * a.heads + head] != 0;
    }
    const bool checkpoint = !repair && a.checkpoint && nt >= KDA_REPLAY_MIN_TILES
        && (a.zero_slices[(segment * a.heads + head) * 2] & 2)
        && (a.zero_slices[(segment * a.heads + head) * 2 + 1] & 2);
    if (checkpoint)
        recurrence_body<Pass::Prescan,128,2,false,RowMajor,true,FullTiles>(a, storage.checkpoint, int(blockIdx.x), int(blockIdx.z));
    else
        recurrence_body<Pass::Prescan,128,2,false,RowMajor,false>(a, storage.ordinary, int(blockIdx.x), int(blockIdx.z));
#endif
}

// Independent zero-state hypotheses publish outputs and checkpoints.
// Classification uses spare resident CTA slots while independent B segments
// run. A kernel boundary completes all flags before prefixes consume them.
template <bool RowMajor, bool FullTiles, bool StateFP32, bool Indexed>
__global__ __launch_bounds__(288, 6) void verified_bias(Arguments a) {
    if constexpr (!Indexed) { a.repair.indices = nullptr; a.input_indices = nullptr; }
    const int64_t job = blockIdx.x;
    const int64_t bias_jobs = int64_t(a.sequences) * a.heads;
    if (job >= bias_jobs) {
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        const int64_t first = (job - bias_jobs) * 32;
        for (int i = warp; i < 32; i += 9) {
            const int64_t record = first + i;
            if (record >= int64_t(a.heads) * a.tiles) continue;
            const bool selected = classify_record<false>(a,record,lane);
            if (lane == 0 && selected) {
            const int tile = int(record % a.tiles), head = int(record / a.tiles);
            int segment = 0, hi = a.sequences;
            while (segment + 1 < hi) {
                const int mid = (segment + hi) / 2;
                if (a.segments[mid * KDA_SEG_STRIDE + KDA_SEG_WSBASE] <= tile) segment = mid;
                else hi = mid;
            }
            atomicOr(a.classification.flags + logical_sequence(a.classification,segment) * a.heads + head, 1);
            }
        }
        return;
    }
    __shared__ RecurrenceStorage<Pass::Prescan,128,2,false,RowMajor,true> storage;
    const int segment = int(job % a.sequences), head = int(job / a.sequences);
    if (threadIdx.x == 0) a.zero_slices[segment * a.heads + head] = 0;
    const int logical = logical_sequence(a.classification,segment);
    const bool first = segment == a.classification.sequence_segments[2 * logical];
    a.initial = first ? a.repair.initial_state : nullptr;
    a.fp32 = StateFP32;
    a.repair.state_fp32 = StateFP32;
    recurrence_body<Pass::Prescan,128,2,false,RowMajor,true,FullTiles,false,-1,false,true>(
        a, storage, segment, 0, head);
}

// Every prefix check is independent. Its exact state comparison is a proof;
// the speculative suffix stays untouched even when this hypothesis fails.
// Compact nonfirst-segment IDs avoid launching a second prefix for true B0.
template <bool RowMajor, bool FullTiles>
__global__ __launch_bounds__(288, 4) void verified_prefix(Arguments a) {
    __shared__ RecurrenceStorage<Pass::Segment,128,2,false,RowMajor,true> storage;
    int const job = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int logical = 0, hi = a.classification.sequences;
    while (logical + 1 < hi) {
        const int mid = (logical + hi) / 2;
        if (a.classification.sequence_segments[2 * mid] - mid <= job) logical = mid;
        else hi = mid;
    }
    const int segment = job + logical + 1;
    if (a.classification.flags[logical * a.heads + head]) return;
    a.initial = a.bias_final;
    a.fp32 = false;
    recurrence_body<Pass::Segment,128,2,false,RowMajor,true,FullTiles,false,-1,false,true,true>(
        a, storage, segment, 0);
    __syncthreads();
    if (tid == 0) {
        int mismatch = 0;
        for (int warp = 0; warp < 8; ++warp) mismatch |= storage.nonzero[warp];
        a.zero_slices[segment * a.heads + head] = mismatch;
    }
}

// The first failed proof starts a finite ordered repair. Matching prefixes
// after it can still reuse their untouched suffix; no CTA waits on another.
template <bool RowMajor, bool FullTiles>
__global__ __launch_bounds__(288, 2) void verified_segments(Arguments a) {
    __shared__ RecurrenceStorage<Pass::Segment,128,2,false,RowMajor,true> storage;
    int const logical = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    if (a.classification.flags[logical * a.heads + head]) {
        const int warp = tid / 32, lane = tid % 32;
        for (int row = warp; row < 128; row += 9)
            conditioned::repair_value(a.repair, logical, head, row, lane);
        return;
    }
    int const first = a.classification.sequence_segments[2 * logical];
    int const count = a.classification.sequence_segments[2 * logical + 1];
    int start = 0;
    while (start < count && !a.zero_slices[(first + start) * a.heads + head]) ++start;
    if (start == count) return;
    a.initial = a.bias_final; a.final = const_cast<BF16*>(a.bias_final); a.fp32 = false;
    for (int i = start; i < count; ++i) {
        const int segment = first + i;
        recurrence_body<Pass::Segment,128,2,false,RowMajor,true,FullTiles,false,-1,false,true>(
            a, storage, segment, 0);
        kda_fence_block();
        __syncthreads();
    }
    int const output = a.repair.indices ? a.repair.indices[logical] : logical;
    if (a.repair.final_state && output >= 0) {
        const BF16* state = a.bias_final;
        const int64_t source = (int64_t(first + count - 1) * a.heads + head) * 128 * 128;
        const int64_t dest = (int64_t(output) * a.heads + head) * 128 * 128;
        for (int word = tid; word < 128 * 128 / 2; word += 288) {
            const uint32_t value = reinterpret_cast<const uint32_t*>(state + source)[word];
            if (a.repair.state_fp32)
                reinterpret_cast<float2*>(static_cast<float*>(a.repair.final_state) + dest)[word] =
                    __bfloat1622float2(*reinterpret_cast<const __ppu_bfloat162*>(&value));
            else reinterpret_cast<uint32_t*>(static_cast<BF16*>(a.repair.final_state) + dest)[word] = value;
        }
    }
}

// Short requests share the producer's prepared tokens in a small kernel.
// Keeping this cache outside the ordinary MMA kernel avoids charging every
// request for the precision queue's control flow and shared storage.
__global__ __launch_bounds__(128, 4) void short_precision(Arguments const a) {
    constexpr int Values = 4, Warps = 4, Jobs = 128 / (Warps * Values);
    __shared__ conditioned::RepairToken cached[KDA_DIRECT_MAX_TOKENS];
    const int tid = threadIdx.x, lane = tid % 32;
    const int warp = cutlass::canonical_warp_idx_sync();
    const int count = a.direct_requests[0] * Jobs;
    const int head = blockIdx.y;
    for (int task = blockIdx.x; task < count; task += gridDim.x) {
        const int seq = a.direct_requests[1 + task / Jobs];
        const int bos = a.cu ? a.cu[seq] : seq * (a.tokens / a.sequences);
        const int eos = a.cu ? a.cu[seq + 1] : (seq + 1) * (a.tokens / a.sequences);
        const int tile = serial_tile_base(a, seq, bos, eos);
        const auto* prepared = reinterpret_cast<const uint64_t*>(a.ws
            + (int64_t(head) * a.tiles + tile) * WS_SLOT_UNITS);
        const int packets = (eos - bos) * sizeof(conditioned::RepairToken) / sizeof(uint64_t);
        for (int packet = tid; packet < packets; packet += 128)
            reinterpret_cast<uint64_t*>(cached)[packet] = prepared[packet];
        __syncthreads();
        const int first_value = ((task % Jobs) * Warps + warp) * Values;
        conditioned::repair_cached_values<Values, float>(a.repair, seq, head, first_value, lane, cached);
        if (task + int(gridDim.x) < count) __syncthreads();
    }
}

// Empty sequences are identities in the external storage type, including FP32
// values that BF16 cannot represent. Nonempty recurrent CTAs never write them.
__global__ void copy_empty_state(Arguments a) {
    int const seq = blockIdx.x, head = blockIdx.y;
    if (a.cu[seq] != a.cu[seq + 1]) return;
    int const src = a.input_indices ? a.input_indices[seq] : seq;
    int const dst = a.output_indices ? a.output_indices[seq] : seq;
    if (dst < 0) return;
    int64_t const ib = (int64_t(src) * a.heads + head) * 128 * 128;
    int64_t const ob = (int64_t(dst) * a.heads + head) * 128 * 128;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        if (a.fp32)
            static_cast<float *>(a.final)[ob + i] = a.initial && src >= 0
                ? static_cast<float const *>(a.initial)[ib + i] : 0.f;
        else
            static_cast<BF16 *>(a.final)[ob + i] = a.initial && src >= 0
                ? static_cast<BF16 const *>(a.initial)[ib + i] : BF16(0.f);
    }
}
// Checkpoint export uses the ordinary serial carry, never a replayed prefix.
// Separate instantiations keep the untracked kernels and their launch ABI intact.
__global__ void track_initial(ConditionedArguments a, TrackState track) {
    const int seq = blockIdx.x, head = blockIdx.y;
    const int64_t chunk = track.int64 ? static_cast<const int64_t*>(track.chunk_idx)[seq]
                                    : static_cast<const int32_t*>(track.chunk_idx)[seq];
    if (chunk != 0) return;
    const int slot = a.indices ? a.indices[seq] : seq;
    const int64_t src = (int64_t(slot) * a.heads + head) * 128 * 128;
    const int64_t dst = (int64_t(seq) * a.heads + head) * 128 * 128;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x)
        track.state[dst + i] = !a.initial_state ? 0.f : a.state_fp32
            ? static_cast<const float*>(a.initial_state)[src + i]
            : float(static_cast<const BF16*>(a.initial_state)[src + i]);
}

template <bool RawFP32>
__global__ void tracked_serial(Arguments a, TrackState track) {
    __shared__ RecurrenceStorage<Pass::Serial, 32, 2, false, true, false, RawFP32> sm;
    recurrence_body<Pass::Serial, 32, 2, false, true, false, false, true, -1,
        RawFP32, false, false, true>(a, sm, blockIdx.x, blockIdx.z, 0, &track);
}
} // namespace flashkda_bf16

void launch_flashkda_track_initial(const flashkda::ConditionedArguments& repair,
    flashkda::TrackState track, hggcStream_t stream) {
    flashkda_bf16::track_initial<<<dim3(repair.sequences, repair.heads), 256, 0, stream>>>(repair, track);
}

void launch_flashkda_tracked_serial(const flashkda::ConditionedArguments& repair,
    const void* ws, int tiles, const int32_t* prefix, bool fp32,
    flashkda::TrackState track, hggcStream_t stream) {
    flashkda_bf16::Arguments a{};
    a.v = static_cast<const cutlass::bfloat16_t*>(repair.v);
    a.ws = static_cast<const cutlass::bfloat16_t*>(ws);
    a.out = static_cast<cutlass::bfloat16_t*>(repair.out);
    a.initial = repair.initial_state; a.final = repair.final_state;
    a.input_indices = repair.indices; a.output_indices = repair.indices;
    a.cu = repair.cu; a.tokens = repair.tokens; a.heads = repair.heads;
    a.sequences = repair.sequences; a.tiles = tiles; a.v_stride = repair.v_stride;
    a.fp32 = repair.state_fp32; a.repair = repair; a.tile_prefix = prefix;
    const dim3 grid(repair.sequences, repair.heads, 4);
    if (fp32) flashkda_bf16::tracked_serial<true><<<grid, 128, 0, stream>>>(a, track);
    else flashkda_bf16::tracked_serial<false><<<grid, 128, 0, stream>>>(a, track);
}

void launch_flashkda_bf16_empty_state(
    const void* initial, void* final, const int32_t* input_indices,
    const int32_t* output_indices, bool fp32, const int32_t* cu,
    int sequences, int heads, hggcStream_t stream) {
    if (final && cu) {
        flashkda_bf16::Arguments a{};
        a.initial = initial; a.final = final; a.input_indices = input_indices;
        a.output_indices = output_indices; a.fp32 = fp32; a.cu = cu; a.heads = heads;
        flashkda_bf16::copy_empty_state<<<dim3(sequences, heads), 256, 0, stream>>>(a);
    }
}

namespace {
template <class Kernel> int resident_grid(Kernel kernel, int threads) {
    int active = 0, device = 0, units = 0;
    if (hggcOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, 0) == hggcSuccess
        && hggcGetDevice(&device) == hggcSuccess
        && hggcDeviceGetAttribute(&units, hggcDevAttrMultiProcessorCount, device) == hggcSuccess
        && active > 0 && units > 0)
        return active * units;
    return 0;
}

template <class Function>
void dispatch_layout(flashkda::KrWorkspaceLayout layout, bool full_tiles, Function run) {
    auto select = [&](auto row_major) {
        if (full_tiles) run(row_major, std::true_type{});
        else run(row_major, std::false_type{});
    };
    if (flashkda::kr_workspace_row_major(layout)) select(std::true_type{});
    else select(std::false_type{});
}
} // namespace

template <bool RawFP32, int Rows = (RawFP32 ? 32 : flashkda::KDA_BF16_SERIAL_ROWS)>
void launch_serial_pipeline(flashkda_bf16::Arguments a, bool row_major,
                            bool has_repair, hggcStream_t stream) {
    using namespace flashkda_bf16;
    auto run = [&](auto row_major, auto state_fp32) {
        constexpr int PublicState = decltype(state_fp32)::value;
        const dim3 grid(a.sequences, a.heads, 128 / Rows);
        constexpr int Threads = 32 * (Rows / 16 + 2);
        if (has_repair) {
            if constexpr (RawFP32) {
                auto fused = recurrence<Pass::Serial, Rows, 2, false, decltype(row_major)::value,
                    false, false, true, PublicState, RawFP32, true>;
                const int capacity = a.direct_requests ? resident_grid(fused, Threads) : 0;
                // Budget one resident wave per warp and pipeline stage.
                constexpr int Stages = 2, Warps = Threads / 32;
                const bool separate = capacity > 0 && int64_t(a.sequences) * a.heads * (128 / Rows)
                    >= int64_t(capacity) * Warps * Stages;
                if (separate) {
                    recurrence<Pass::Serial, Rows, 2, false, decltype(row_major)::value,
                        false, false, true, PublicState, RawFP32, false, false>
                        <<<grid, Threads, 0, stream>>>(a);
                    constexpr int Jobs = 128 / (4 * 4);
                    int blocks = a.sequences * Jobs;
                    const int capacity = resident_grid(short_precision, 128);
                    if (capacity > 0) blocks = std::min(blocks, std::max(1, capacity / a.heads));
                    short_precision<<<dim3(blocks, a.heads), 128, 0, stream>>>(a);
                    return;
                }
            }
            if (a.direct_requests) {
                constexpr int Warps = Rows / 16 + 2;
                constexpr int Jobs = (128 + Warps - 1) / Warps;
                auto kernel = recurrence<Pass::Serial, Rows, 2, false, decltype(row_major)::value,
                    false, false, true, PublicState, RawFP32, true>;
                int blocks = a.sequences * Jobs;
                const int capacity = resident_grid(kernel, Threads);
                if (capacity > 0) blocks = std::min(blocks, std::max(1, capacity / a.heads));
                kernel<<<dim3(a.sequences * (128 / Rows) + blocks, a.heads), Threads, 0, stream>>>(a);
                return;
            }
            recurrence<Pass::Serial, Rows, 2, false, decltype(row_major)::value, false, false, true, PublicState, RawFP32>
                <<<grid, Threads, 0, stream>>>(a);
        }
        else
            recurrence<Pass::Serial, Rows, 2, false, decltype(row_major)::value, false, false, false, PublicState, RawFP32>
                <<<grid, Threads, 0, stream>>>(a);
    };
    auto dispatch = [&](auto row_major) {
        if (!a.initial || !a.final) run(row_major, std::integral_constant<int, -1>{});
        else if (a.fp32) run(row_major, std::true_type{});
        else run(row_major, std::false_type{});
    };
    if (row_major) dispatch(std::true_type{});
    else dispatch(std::false_type{});
}

void launch_flashkda_bf16_recurrence(
    void* out, const void* v, int64_t v_stride, const void* ws,
    const void* initial, void* final, const int32_t* input_indices,
    const int32_t* output_indices, bool fp32,
    int tokens, int heads, int sequences, int tiles, const int32_t* cu,
    flashkda::KrWorkspaceLayout layout, hggcStream_t stream,
    const flashkda::ConditionedArguments* repair, const void* keys, const int32_t* tile_flags, const int32_t* tile_prefix, int value_rows) {
    using namespace flashkda_bf16;
    Arguments a{};
    a.v = static_cast<BF16 const *>(v); a.ws = static_cast<BF16 const *>(ws);
    a.out = static_cast<BF16 *>(out); a.initial = initial; a.final = final;
    a.input_indices = input_indices; a.output_indices = output_indices;
    a.cu = cu; a.tokens = tokens; a.heads = heads; a.sequences = sequences;
    a.tiles = tiles; a.v_stride = v_stride; a.fp32 = fp32;
    if (repair) a.repair = *repair;
    a.classification.keys = keys; a.classification.tile_flags = tile_flags;
    a.tile_prefix = tile_prefix;
    a.direct_requests = repair && tile_prefix ? tile_prefix + sequences + 1 : nullptr;
    auto run = [&](auto rows) {
        launch_serial_pipeline<false, decltype(rows)::value>(a,
            flashkda::kr_workspace_row_major(layout), repair != nullptr, stream);
    };
    if (value_rows == 16) run(std::integral_constant<int, 16>{});
    else if (value_rows == 32) run(std::integral_constant<int, 32>{});
    else if (value_rows == 64) run(std::integral_constant<int, 64>{});
    else run(std::integral_constant<int, flashkda::KDA_BF16_SERIAL_ROWS>{});
}

void launch_flashkda_fp32_serial(
    void* out, const void* v, int64_t v_stride, const void* ws,
    const void* initial, void* final, const int32_t* input_indices,
    const int32_t* output_indices, bool fp32,
    int tokens, int heads, int sequences, int tiles, const int32_t* cu,
    const int32_t* prefix, const flashkda::ConditionedArguments& repair, hggcStream_t stream) {
    flashkda_bf16::Arguments a{};
    a.v = static_cast<const cutlass::bfloat16_t*>(v);
    a.ws = static_cast<const cutlass::bfloat16_t*>(ws);
    a.out = static_cast<cutlass::bfloat16_t*>(out);
    a.initial = initial; a.final = final;
    a.input_indices = input_indices; a.output_indices = output_indices;
    a.cu = cu; a.tokens = tokens; a.heads = heads; a.sequences = sequences;
    a.tiles = tiles; a.v_stride = v_stride; a.fp32 = fp32;
    a.repair = repair; a.tile_prefix = prefix;
    a.direct_requests = prefix ? prefix + sequences + 1 : nullptr;
    launch_serial_pipeline<true>(a, true, true, stream);
}

void launch_flashkda_bf16_cp_prescan(
    const void* v, int64_t v_stride, const void* ws,
    void* chain, int32_t* zero_slices, int tokens, int heads, int segments,
    int tiles, const int* table, flashkda::KrWorkspaceLayout layout, hggcStream_t stream,
    void* output, void* checkpoint, bool full_tiles, flashkda::PreparedClassification classification, void* prepared_carries,
    const flashkda::ConditionedArguments* repair) {
    using namespace flashkda_bf16;
    Arguments a{};
    a.v = static_cast<BF16 const *>(v); a.ws = static_cast<BF16 const *>(ws);
    a.final = chain; a.zero_slices = zero_slices; a.segments = table;
    a.tokens = tokens; a.heads = heads; a.sequences = segments;
    a.tiles = tiles; a.v_stride = v_stride;
    a.out = static_cast<BF16 *>(output); a.checkpoint = static_cast<BF16 *>(checkpoint);
    a.classification = classification;
    if (repair) a.repair = *repair;
    a.prepared_carries = static_cast<BF16 *>(prepared_carries);
    auto run = [&](auto row_major, auto full) {
        recurrence<Pass::Prescan,128,2,true,decltype(row_major)::value>
            <<<dim3(segments,heads),288,0,stream>>>(a);
        if (checkpoint) {
            hybrid_bias<decltype(row_major)::value,decltype(full)::value>
                <<<dim3(segments,heads),288,0,stream>>>(a);
        } else {
            recurrence<Pass::Prescan,128,2,false,decltype(row_major)::value>
                <<<dim3(segments,heads),288,0,stream>>>(a);
        }
    };
    dispatch_layout(layout, full_tiles, run);
}

void launch_flashkda_bf16_cp_segment(
    void* out, const void* v, int64_t v_stride, const void* ws,
    const void* carries, void* final, const int32_t* indices, bool fp32,
    int tokens, int heads, int segments, int tiles, const int* table,
    flashkda::KrWorkspaceLayout layout, hggcStream_t stream,
    const void* checkpoint, const void* bias_final, int32_t* zero_slices, bool full_tiles, flashkda::PreparedClassification skip) {
    using namespace flashkda_bf16;
    Arguments a{};
    a.v = static_cast<BF16 const *>(v); a.ws = static_cast<BF16 const *>(ws);
    a.out = static_cast<BF16 *>(out); a.initial = carries; a.final = final;
    a.output_indices = indices; a.fp32 = fp32; a.segments = table;
    a.tokens = tokens; a.heads = heads; a.sequences = segments;
    a.tiles = tiles; a.v_stride = v_stride;
    a.checkpoint = const_cast<BF16 *>(static_cast<const BF16 *>(checkpoint));
    a.bias_final = static_cast<const BF16 *>(bias_final); a.zero_slices = zero_slices;
    a.classification = skip;
    auto run = [&](auto row_major, auto full) {
        if (checkpoint)
            recurrence<Pass::Segment,128,2,false,decltype(row_major)::value,true,decltype(full)::value>
                <<<dim3(segments,heads),288,0,stream>>>(a);
        else
            recurrence<Pass::Segment, 128, 2, false, decltype(row_major)::value, false, decltype(full)::value>
                <<<dim3(segments, heads), 288, 0, stream>>>(a);
    };
    dispatch_layout(layout, full_tiles, run);
}


hggcError_t query_flashkda_bf16_resident_blocks(
    int *blocks, bool row_major, bool verify, bool full_tiles) {
    using namespace flashkda_bf16;
    auto query = [&](auto... kernels) {
        *blocks = -1;
        for (auto kernel : {kernels...}) {
            int active = 0;
            const auto error = hggcOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 288, 0);
            if (error != hggcSuccess) return error;
            *blocks = *blocks < 0 ? active : std::min(*blocks, active);
        }
        return hggcSuccess;
    };
    auto dispatch = [&](auto linear) {
        constexpr bool RowMajor = decltype(linear)::value;
        if (verify) {
            auto run = [&](auto full) {
                constexpr bool Full = decltype(full)::value;
                return query(verified_bias<RowMajor,Full,true,false>, verified_bias<RowMajor,Full,false,false>,
                             verified_bias<RowMajor,Full,true,true>, verified_bias<RowMajor,Full,false,true>);
            };
            return full_tiles ? run(std::true_type{}) : run(std::false_type{});
        }
        return query(
            recurrence<Pass::Prescan,128,2,true,RowMajor>,
            recurrence<Pass::Prescan,128,2,false,RowMajor>,
            hybrid_bias<RowMajor,false>, hybrid_bias<RowMajor,true>,
            recurrence<Pass::Segment,128,2,false,RowMajor>,
            recurrence<Pass::Segment,128,2,false,RowMajor,true>,
            recurrence<Pass::Segment,128,2,false,RowMajor,false,true>,
            recurrence<Pass::Segment,128,2,false,RowMajor,true,true>);
    };
    return row_major ? dispatch(std::true_type{}) : dispatch(std::false_type{});
}

void launch_flashkda_bf16_cp_verify(
    void* ws, void* chain, int32_t* flags,
    const int32_t* segments, int segment_count, int tiles,
    flashkda::KrWorkspaceLayout layout, flashkda::PreparedClassification classification,
    const flashkda::ConditionedArguments& repair, bool full_tiles, hggcStream_t stream) {
    using namespace flashkda_bf16;
    Arguments a{};
    a.v = static_cast<const BF16*>(repair.v);
    a.ws = static_cast<const BF16*>(ws);
    a.out = static_cast<BF16*>(repair.out);
    a.final = chain;
    a.bias_final = static_cast<const BF16*>(chain);
    a.segments = segments;
    a.tokens = repair.tokens; a.heads = repair.heads; a.sequences = segment_count;
    a.tiles = tiles; a.v_stride = repair.v_stride;
    a.classification = classification; a.repair = repair;
    a.input_indices = repair.indices; a.zero_slices = flags;
    const int64_t jobs = int64_t(segment_count) * a.heads + (int64_t(a.heads) * tiles + 31) / 32;
    auto run = [&](auto linear, auto full) {
        constexpr bool RowMajor = decltype(linear)::value, Full = decltype(full)::value;
        auto launch = [&](auto indexed) {
            constexpr bool Indexed = decltype(indexed)::value;
            if (repair.state_fp32)
                verified_bias<RowMajor,Full,true,Indexed><<<jobs,288,0,stream>>>(a);
            else verified_bias<RowMajor,Full,false,Indexed><<<jobs,288,0,stream>>>(a);
        };
        if (repair.indices) launch(std::true_type{});
        else launch(std::false_type{});
        verified_prefix<RowMajor,Full><<<dim3(segment_count-repair.sequences,a.heads),288,0,stream>>>(a);
        verified_segments<RowMajor,Full><<<dim3(repair.sequences,a.heads),288,0,stream>>>(a);
    };
    dispatch_layout(layout,full_tiles,run);
}
