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
 * FlashQLA — arch detection and actlize traits dispatch by __HGGC_ARCH__.
 *
 * Mirrors src/prefill/flashkda/kernels/flashkda_arch.cuh; see that file for the
 * full rationale. Summary:
 *
 *   __HGGC_ARCH__ == 150   -> PPU1.5 (ZW-M890P), tc02, actlize ppu0015
 *   __HGGC_ARCH__ == 100   -> PPU1.0/1.1 (ZW-810E), tc01, actlize ppu0010
 *
 * ``__HGGC_ARCH__`` comes from the PPU compiler driver, and actlize gates its
 * own inline asm on it directly, so no version macro has to be forced from here
 * (the older cutlass3 fork needed an ACOMPUTE_VERSION override).
 *
 * FlashQLA keeps its own copy of this header (rather than sharing FlashKDA's)
 * so the two prefill extensions stay independently buildable -- neither
 * includes anything from the other.
 *
 * After including this header, namespace ``flashqla`` exposes:
 *
 *   PpuBf16MMA   -- 16x16x16 f32<=bf16*bf16+f32 TN atom traits
 *   PpuF16MMA    -- 16x16x16 f32<=f16*f16+f32 TN atom traits
 *   PpuAiuLoad<Bits, Elem, Trans, CUBE_H, CUBE_W>
 *   PpuTsmLdSwzl<Elem, CUBE_H, CUBE_W, Swap, Trans, InstNum>
 */
#pragma once

// --- Detect PPU generation ------------------------------------------------
#if defined(__HGGC_ARCH__)
#  if __HGGC_ARCH__ >= 150
#    define FLASHQLA_ARCH_150 1
#  elif __HGGC_ARCH__ >= 100
#    define FLASHQLA_ARCH_100 1
#  else
#    error "FlashQLA prefill: unsupported __HGGC_ARCH__ (< 100). Expected 100 (PPU1.0/1.1) or 150 (PPU1.5)."
#  endif
#else
// Host / non-PPU translation units include these kernel headers through the
// pybind binding. Fall back to the 150 traits family so the CuTe types exist;
// all device asm is guarded and degenerates to (void) casts on the host side.
#  define FLASHQLA_ARCH_150 1
#endif

#include <cstdint>
#include <hggc_fp16.h>
#include <hggc_bf16.h>

#include "cute/tensor.hpp"
#include "cute/arch/mma.hpp"

// ppu_tensor_mix.hpp defines is_mix_iterator and make_mix_tensor. actlize's
// copy_traits_ppu0010_aiu.hpp uses is_mix_iterator inside a constexpr
// if in its copy_unpack, so tensor_mix MUST be included before the traits
// header, otherwise the constexpr-if fails to compile on tc01. Keep this
// block above the arch-specific copy_traits includes below.
#include "cute/ppu_tensor_mix.hpp"

// copy_ppu0010_aiu.hpp carries the shared ``AiuDesc`` definition that the
// 10500 header reuses, so it is included on both arches.
#if defined(FLASHQLA_ARCH_150)
#  include "cute/arch/mma_ppu0015.hpp"
#  include "cute/atom/mma_traits_ppu0015.hpp"
#  include "cute/arch/copy_ppu0010_aiu.hpp"  // AiuDesc
#  include "cute/arch/copy_ppu0015_aiu.hpp"
#  include "cute/atom/copy_traits_ppu0015_aiu.hpp"
#else
#  include "cute/arch/mma_ppu0010.hpp"
#  include "cute/atom/mma_traits_ppu0010.hpp"
#  include "cute/arch/copy_ppu0010_aiu.hpp"
#  include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#endif

// Standard CuTe copy algorithm (used by cute::copy expansion) and Copy_Atom /
// make_tiled_copy helpers.
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_atom.hpp>

namespace flashqla {

// ---- MMA traits aliases --------------------------------------------------
#if defined(FLASHQLA_ARCH_150)
using PpuBf16MMA = cute::PPU0015_16x16x16_F32BF16BF16F32_TN;
using PpuF16MMA  = cute::PPU0015_16x16x16_F32F16F16F32_TN;
#else
using PpuBf16MMA = cute::PPU0010_16x16x16_F32BF16BF16F32_TN;
using PpuF16MMA  = cute::PPU0010_16x16x16_F32F16F16F32_TN;
#endif

// ---- AIU / TSM traits aliases --------------------------------------------
// The traits-level copy() signatures agree across both actlize families; only
// the template shape differs. PPU1.5 takes the cube shape as template args,
// PPU1.0/1.1 carries it at runtime via AiuDesc / TSM_LD_SWZL_IMPL, so the
// extra template parameters are absorbed here and callers can name shapes
// uniformly. ``Swzl`` selects swizzled vs linear smem writes on both arches.
#if defined(FLASHQLA_ARCH_150)
template <int NumBits, typename Element, bool Trans, int CUBE_H, int CUBE_W,
          bool Swzl = true>
using PpuAiuLoad =
    cute::PPU0015_AIU_LOAD<cute::Int<NumBits>, Element, Trans, CUBE_H, CUBE_W, Swzl>;

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum>
using PpuTsmLdSwzl =
    cute::PPU0015_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum>;
#else
template <int NumBits, typename Element, bool Trans, int CUBE_H, int CUBE_W,
          bool Swzl = true>
using PpuAiuLoad =
    cute::PPU0010_AIU_LOAD<cute::Int<NumBits>, Element, Trans, Swzl>;

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum>
using PpuTsmLdSwzl =
    cute::PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum>;
#endif

}  // namespace flashqla
