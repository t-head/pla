// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2026 MoonshotAI
// SPDX-License-Identifier: BSD-3-Clause AND MIT
// Adapted from FlashKDA.

#pragma once
#include "math.cuh"

extern "C" __device__ unsigned __ppu_read_firstlane(unsigned);

namespace flashkda_paired {
template <class Element, int CUBE_H, int CUBE_W, int NUM_CUBES> struct AiuTile {
    // Forces swzl_mode 0, matching production's bf16 tiles. Anything else selects
    // mode 1 and a different interleave than the loads below assume.
    static_assert(CUBE_W * int(sizeof(Element)) == 128,
                  "CUBE_W * sizeof(Element) must be 128 so swzl_mode == 0");
    // 256/512/576 fault on this hardware.
    static_assert(CUBE_W <= 128, "AIU CUBE_W is capped at 128 elements");

    static constexpr int kElemsPerCube = CUBE_H * CUBE_W;
    static constexpr int kElems = kElemsPerCube * NUM_CUBES;

    // The AIU faults on destination offsets that are not 128-byte aligned.
    alignas(128) Element storage[kElems];

    CUTE_DEVICE Element *data() {
        return storage;
    }
    CUTE_DEVICE Element const *data() const {
        return storage;
    }
};
// Copy atoms
#if FLASH_KDA_PPU_TC02
template <int CUBE_H, int CUBE_W, int NUM_CUBES> struct TsmOps {
    template <bool Swap, bool Trans>
    CUTE_DEVICE static void load(void *frag, int base, int block, int row) {
        int addr = base + (block / 4) * (CUBE_H * 8) + (block % 4) * 2 + row * 8;
        int *v = (int *)frag;
        constexpr int lbo = Swap != Trans ? 64 : 1, sbo = Swap != Trans ? 1 : 64;
        if constexpr (Trans) {
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, "
                         "%3}, [%4], %5, %6, %7;"
                         : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                         : "l"(addr), "r"(lbo), "r"(sbo), "n"(0));
        } else {
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, "
                         "[%4], %5, %6, %7;"
                         : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                         : "l"(addr), "r"(lbo), "r"(sbo), "n"(0));
        }
    }
    CUTE_DEVICE static void load_a(void *p, int base, int block, int row = 0) {
        load<false, false>(p, base, block, row);
    }
    CUTE_DEVICE static void load_b(void *p, int base, int block, int row = 0) {
        load<true, false>(p, base, block, row);
    }
    CUTE_DEVICE static void load_bt(void *p, int base, int block, int row = 0) {
        load<true, true>(p, base, block, row);
    }

    using LdA = cute::PPU0015_TSM_LD_SWZL<BF16, CUBE_H, CUBE_W, /*Swap=*/false,
                                          /*Trans=*/false, NUM_CUBES>;
    using LdB = cute::PPU0015_TSM_LD_SWZL<BF16, CUBE_H, CUBE_W, /*Swap=*/true,
                                          /*Trans=*/false, NUM_CUBES>;
    using LdBT = cute::PPU0015_TSM_LD_SWZL<BF16, CUBE_H, CUBE_W, /*Swap=*/true,
                                           /*Trans=*/true, NUM_CUBES>;

    static_assert(LdA::swzl_mode == 0, "expected swzl_mode 0 for a 128-byte cube row");
    static_assert(LdB::swzl_mode == 0, "expected swzl_mode 0 for a 128-byte cube row");
    static_assert(LdBT::swzl_mode == 0, "expected swzl_mode 0 for a 128-byte cube row");


};

#endif

} // namespace flashkda_paired

namespace flashkda_paired {

#if FLASH_KDA_PPU_TC02

template <class Element, int CUBE_H, int CUBE_W, int NUM_CUBES> struct AiuTileLoader {
    using Atom =
        cute::PPU0015_AIU_LOAD<cute::C<CUBE_H * CUBE_W * cute::sizeof_bits<Element>::value>,
                               Element,
                               /*Trans=*/false, CUBE_H, CUBE_W>;

    static constexpr int kElemsPerCube = CUBE_H * CUBE_W;

    cute::AiuDesc desc;

    // `base` already carries any fixed offset (e.g. head_idx * D for a
    // (H, T, D) tensor); dim_h is what padz clamps against.
    CUTE_DEVICE void init(Element const *base, int dim_h, int dim_w, int stride_w) {
        desc.init(reinterpret_cast<uint8_t const *>(base), dim_h, dim_w, stride_w);
    }

    // Warp-collective. Every lane of the issuing warp must reach this.
    CUTE_DEVICE void load(Element *smem, int coord_h) const {
        CUTE_UNROLL
        for (int c = 0; c < NUM_CUBES; ++c) {
            Atom::copy(smem + c * kElemsPerCube, desc.gmem_ptr, desc,
                       /*coord_w=*/c * CUBE_W, coord_h);
        }
    }
};

#endif // FLASH_KDA_PPU_TC02

using CpAsyncVec16 = cute::PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

CUTE_DEVICE void cp_async_fence_all() {
    cute::cp_async_fence();
}

template <int N = 0> CUTE_DEVICE void cp_async_wait_all() {
    cute::cp_async_wait<N>();
}

template <int ROWS, int COLS, int COLS_PAD, int NumThreads, bool FullRows = false, class Stride>
CUTE_DEVICE void cp_async_tile(BF16 *__restrict__ smem, BF16 const *__restrict__ gmem,
                               Stride gmem_row_stride, int rows_valid, int tid) {
    static_assert(COLS % 8 == 0, "16-byte vectors require COLS to be a multiple of 8");
    constexpr int kElemsPerVec = 16 / int(sizeof(BF16)); // 8
    constexpr int kVecsPerRow = COLS / kElemsPerVec;
    constexpr int kVecs = ROWS * kVecsPerRow;

    for (int i = tid; i < kVecs; i += NumThreads) {
        int const r = i / kVecsPerRow;
        int const c = (i % kVecsPerRow) * kElemsPerVec;
        BF16 *dst = smem + r * COLS_PAD + c;
        if (FullRows || r < rows_valid) {
            CpAsyncVec16::copy(
                *reinterpret_cast<cute::uint128_t const *>(gmem + r * gmem_row_stride + c),
                *reinterpret_cast<cute::uint128_t *>(dst));
        } else {
            CUTE_UNROLL
            for (int e = 0; e < kElemsPerVec; ++e)
                dst[e] = BF16(0);
        }
    }
}

// Contiguous 16-byte cp.async of `bytes` bytes. For the small per-tile arrays
// (INV, Mqk, g_total, beta) whose gmem side is already packed.
template <int NumThreads>
CUTE_DEVICE void cp_async_contiguous(void *__restrict__ smem, void const *__restrict__ gmem,
                                     int bytes, int tid) {
    int const vecs = bytes / 16;
    auto const *g = reinterpret_cast<cute::uint128_t const *>(gmem);
    auto *s = reinterpret_cast<cute::uint128_t *>(smem);
    for (int i = tid; i < vecs; i += NumThreads) {
        CpAsyncVec16::copy(g[i], s[i]);
    }
}

} // namespace flashkda_paired

namespace flashkda_paired {

// gmem -> smem, `Stages` buffers
template <int Stages, int NumProducerThreads, int NumConsumerThreads> struct AsyncInPipeline {
    static constexpr int kTotal = NumProducerThreads + NumConsumerThreads;
    static_assert(Stages > 0, "at least one stage");

    // Signalled by copy completion, not by an arriving thread.
    __mbarrier_t full[Stages];

    // One thread, then the caller must __syncthreads() before any use.
    CUTE_DEVICE void init(int tid) {
        if (tid == 0) {
            CUTE_UNROLL
            for (int s = 0; s < Stages; ++s) {
                __mbarrier_init(&full[s], NumProducerThreads);
            }
            kda_fence_block();
        }
    }

    // A stage is reused every `Stages` iterations, so the mbarrier phase alternates
    // on that period. Computed here rather than passed in, so a caller cannot get
    // the parity wrong.
    CUTE_DEVICE static uint32_t phase_of(int t) {
        return uint32_t((t / Stages) & 1);
    }

    // --- producer, all NumProducerThreads lanes ---

    // Wait until the consumers have finished with this stage. Skipped for the
    // first `Stages` iterations, when the buffer has never been read.
    CUTE_DEVICE void producer_acquire(int stage, int t) {
        if (t >= Stages) {
            cutlass::arch::NamedBarrier::arrive_and_wait(kTotal, kStageFree + stage);
        }
    }

    // Issue the copies first, then this: it arms the barrier to fire on their
    // completion without counting as an arrival of its own.
    CUTE_DEVICE void producer_commit(int stage) {
        uint32_t const address = cute::cast_smem_ptr_to_uint(&full[stage]);
        asm volatile("ppu.cp.async.awbar.arrive.noinc.shared::cta.b64 [%0];"
                     :: "r"(address) : "memory");
    }

    // --- consumer, all NumConsumerThreads threads ---

    CUTE_DEVICE void consumer_wait(int stage, int t) {
        uint32_t const phase = phase_of(t);
        while (!cutlass::arch::ClusterBarrier::test_wait(&full[stage], phase, 1)) {
        }
    }

    CUTE_DEVICE void consumer_release(int stage) {
        cutlass::arch::NamedBarrier::arrive(kTotal, kStageFree + stage);
    }
};

// smem -> gmem, `Stages` buffers, no async engine involved
template <int Stages, int NumProducerThreads, int NumConsumerThreads> struct SyncOutPipeline {
    static constexpr int kTotal = NumProducerThreads + NumConsumerThreads;

    CUTE_DEVICE void producer_acquire(int stage, int t) {
        if (t >= Stages) {
            cutlass::arch::NamedBarrier::arrive_and_wait(kTotal, kOutConsumed + stage);
        }
    }

    // The producing threads' own smem stores are the completion event, so this
    // only has to order them before the consumer's reads.
    CUTE_DEVICE void producer_commit(int stage) {
        cutlass::arch::NamedBarrier::arrive(kTotal, kOutReady + stage);
    }

    CUTE_DEVICE void consumer_wait(int stage, int /*t*/) {
        cutlass::arch::NamedBarrier::arrive_and_wait(kTotal, kOutReady + stage);
    }

    CUTE_DEVICE void consumer_release(int stage) {
        cutlass::arch::NamedBarrier::arrive(kTotal, kOutConsumed + stage);
    }
};

} // namespace flashkda_paired
