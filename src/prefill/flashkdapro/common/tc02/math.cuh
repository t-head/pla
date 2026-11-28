// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// Copyright (c) 2026 MoonshotAI
// SPDX-License-Identifier: BSD-3-Clause AND MIT
// Adapted from FlashKDA.

#pragma once

#ifndef CUTE_ARCH_CP_ASYNC_PPU_ENABLED
#define CUTE_ARCH_CP_ASYNC_PPU_ENABLED 1
#endif
#ifndef CUTE_ARCH_LDSM_PPU_ACTIVATED
#define CUTE_ARCH_LDSM_PPU_ACTIVATED 1
#endif

#include <cstdint>

// __mbarrier_t and friends. Without this only the unrelated 32-bit `barrier_t`
// is visible and every barrier call fails to typecheck.
#include <hggc_awbarrier.h>

#include <ppu_include.hpp>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#if !defined(ACOMPUTE_VERSION)
#error "ACOMPUTE_VERSION must be set: 10500 for tc02/ppu0015, 10000 for tc01/ppu001."
#endif

// Kernel bodies are TC02-only; keep declarations visible for the multi-target
// host launch stubs. TC01 retains its existing public dispatch.

namespace flashkda_paired {

using BF16 = cutlass::bfloat16_t;

#define FLASH_KDA_PPU_TC02 (ACOMPUTE_VERSION >= 10500)

#if !FLASH_KDA_PPU_TC02
#error "FlashKDA PPU requires tc02/ppu0015 and its FP32 MMA atom."
#endif

struct MmaBF16 : cute::PPU0015_16x16x16_F32BF16BF16F32_TN {};
struct MmaF32 : cute::PPU0015_16x16x8_F32F32F32F32_TN {};

} // namespace flashkda_paired

#if FLASH_KDA_PPU_TC02
namespace cute {
template <>
struct MMA_Traits<flashkda_paired::MmaBF16> : MMA_Traits<cute::PPU0015_16x16x16_F32BF16BF16F32_TN> {
    using ValTypeD = float;
    using ValTypeA = cutlass::bfloat16_t;
    using ValTypeB = cutlass::bfloat16_t;
    using ValTypeC = float;
};

template <>
struct MMA_Traits<flashkda_paired::MmaF32> : MMA_Traits<cute::PPU0015_16x16x8_F32F32F32F32_TN> {
    using ValTypeD = float;
    using ValTypeA = float;
    using ValTypeB = float;
    using ValTypeC = float;
};
} // namespace cute
#endif

namespace flashkda_paired {

#if FLASH_KDA_PPU_TC02
// Guard both overrides, so an upstream traits change cannot silently reintroduce
// the wrong operand type.
static_assert(
    cute::is_same<typename cute::MMA_Traits<MmaBF16>::ValTypeA, cutlass::bfloat16_t>::value,
    "MmaBF16 operand type is not bfloat16_t: the traits override was bypassed");
static_assert(cute::is_same<typename cute::MMA_Traits<MmaF32>::ValTypeA, float>::value,
              "MmaF32 operand type is not float: the traits override was bypassed");
#endif

// Math shims. Keep each call inside ONE source expression: the PPU compiler
// contracts a*b+c into v.fma.f32 only within a single expression, and no flag
// recovers a contraction split across a statement boundary.

CUTE_DEVICE float kda_exp2(float x) {
    return exp2f(x);
}

CUTE_DEVICE void kda_fence_block() {
    __threadfence_block();
}

// Warp roles and barrier ids

// User-visible NamedBarrier ids are offset by ReservedNamedBarrierCount inside
// cutlass, so these are indices, not raw hardware barrier numbers.
enum NamedBarrierId : int {
    kOutReady = 0,    // MMA warps -> store warp, per output stage
    kOutConsumed = 2, // store warp -> MMA warps, per output stage
    kStageFree = 4,   // consumers -> producer, per input stage
};

} // namespace flashkda_paired

namespace flashkda_paired {

// Compile-time proofs
#if FLASH_KDA_PPU_TC02
namespace detail {

using ABF16 = typename cute::MMA_Traits<MmaBF16>::ALayout;
using BBF16 = typename cute::MMA_Traits<MmaBF16>::BLayout;
using CBF16 = typename cute::MMA_Traits<MmaBF16>::CLayout;

// (1) C == A, stride by stride. Compared elementwise rather than with is_same so
// an equivalent layout spelled with a different tuple type still passes.
static_assert(cute::size(CBF16{}) == cute::size(ABF16{}), "bf16 atom: |C| != |A|");
static_assert(cute::stride<0, 0>(CBF16{}) == cute::stride<0, 0>(ABF16{}), "bf16: C!=A thr0");
static_assert(cute::stride<0, 1>(CBF16{}) == cute::stride<0, 1>(ABF16{}), "bf16: C!=A thr1");
static_assert(cute::stride<1, 0>(CBF16{}) == cute::stride<1, 0>(ABF16{}), "bf16: C!=A val0");
static_assert(cute::stride<1, 1>(CBF16{}) == cute::stride<1, 1>(ABF16{}), "bf16: C!=A val1");
static_assert(cute::stride<1, 2>(CBF16{}) == cute::stride<1, 2>(ABF16{}), "bf16: C!=A val2");

// (2) B == A with value modes 1 and 2 exchanged.
static_assert(cute::stride<0, 0>(BBF16{}) == cute::stride<0, 0>(ABF16{}), "bf16: B/A thr0");
static_assert(cute::stride<0, 1>(BBF16{}) == cute::stride<0, 1>(ABF16{}), "bf16: B/A thr1");
static_assert(cute::stride<1, 0>(BBF16{}) == cute::stride<1, 0>(ABF16{}), "bf16: B/A val0");
static_assert(cute::stride<1, 1>(BBF16{}) == cute::stride<1, 2>(ABF16{}),
              "bf16: B is not A with value modes 1,2 exchanged -- as_B() is invalid");
static_assert(cute::stride<1, 2>(BBF16{}) == cute::stride<1, 1>(ABF16{}),
              "bf16: B is not A with value modes 1,2 exchanged -- as_B() is invalid");

using AF32 = typename cute::MMA_Traits<MmaF32>::ALayout;
using BF32 = typename cute::MMA_Traits<MmaF32>::BLayout;
using CF32 = typename cute::MMA_Traits<MmaF32>::CLayout;

// The k=8 fp32 atom: A <-> B is still a free exchange, but of value modes 0 and
// 1 (its value shape is (2,2), not (2,2,2)).
static_assert(cute::stride<1, 0>(BF32{}) == cute::stride<1, 1>(AF32{}), "fp32: B/A val exchange");
static_assert(cute::stride<1, 1>(BF32{}) == cute::stride<1, 0>(AF32{}), "fp32: B/A val exchange");

static_assert(!(cute::stride<1, 0>(CF32{}) == cute::stride<1, 0>(AF32{}) &&
                cute::stride<1, 1>(CF32{}) == cute::stride<1, 1>(AF32{})),
              "fp32 atom now has C == A: the smem round-trips in the Neumann inverse "
              "are no longer needed and should be removed.");

} // namespace detail
#endif // FLASH_KDA_PPU_TC02

// Where a value slot lives inside a 16x16 C fragment
struct FragCoord {
    int m; // 0..15, the C fragment's row (M) index
    int n; // 0..15, the C fragment's column (N) index
};

// Used for in-fragment masking, beta scaling and g_total indexing, replacing
// three separate smem round-trips in the CUDA kernels.
CUTE_HOST_DEVICE constexpr FragCoord frag_coord_mn(int lane, int v) {
    int const t0 = lane % 4;
    int const t1 = (lane / 4) % 8;
    int const v0 = v % 2;
    int const v1 = (v / 2) % 2;
    int const v2 = (v / 4) % 2;
    return FragCoord{t1 + 8 * v1, 2 * t0 + v0 + 8 * v2};
}

#if FLASH_KDA_PPU_TC02
namespace detail {

// Check the hand-derived decoding against the traits over all 32 lanes x 8 slots
// instead of trusting the algebra in the header comment.
CUTE_HOST_DEVICE constexpr bool frag_coord_matches_traits() {
    for (int lane = 0; lane < 32; ++lane) {
        for (int v = 0; v < 8; ++v) {
            FragCoord const c = frag_coord_mn(lane, v);
            if (CBF16{}(lane, v) != c.m + 16 * c.n)
                return false;
        }
    }
    return true;
}
static_assert(frag_coord_matches_traits(),
              "frag_coord_mn() disagrees with MMA_Traits<MmaBF16>::CLayout");

} // namespace detail
#endif

// A/C-shaped fragment -> B-shaped fragment, over the same registers

template <class Engine, class Layout>
CUTE_HOST_DEVICE auto as_B(cute::Tensor<Engine, Layout> const &f) {
    auto val = cute::get<0>(f.layout());
    constexpr int kValRank = decltype(cute::rank(val))::value;
    static_assert(kValRank == 2 || kValRank == 3,
                  "as_B expects the atom's value mode, of rank 2 (k=8 fp32) or 3 (k=16 bf16)");
    static_assert(cute::rank(Layout{}) == 3, "as_B expects a (value, outer, k) fragment");

    if constexpr (kValRank == 3) {
        auto swapped = cute::make_layout(cute::get<0>(val), cute::get<2>(val), cute::get<1>(val));
        return cute::make_tensor(f.data(), cute::make_layout(swapped, cute::get<1>(f.layout()),
                                                             cute::get<2>(f.layout())));
    } else {
        auto swapped = cute::make_layout(cute::get<1>(val), cute::get<0>(val));
        return cute::make_tensor(f.data(), cute::make_layout(swapped, cute::get<1>(f.layout()),
                                                             cute::get<2>(f.layout())));
    }
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE auto as_A(cute::Tensor<Engine, Layout> const &f) {
    return as_B(f);
}
// Narrow an FP32 accumulator into an existing BF16 register fragment.
template <class Dst, class Src> CUTE_DEVICE void narrow_into(Dst &dst, Src const &src) {
    CUTE_UNROLL
    for (int i = 0; i < decltype(cute::size(src))::value; ++i) {
        dst(i) = BF16(src(i));
    }
}

} // namespace flashkda_paired
