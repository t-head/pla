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
 * FlashKDA — arch detection and actlize traits dispatch by __HGGC_ARCH__.
 *
 * This header is the single point of truth for which PPU generation the
 * FlashKDA prefill kernels compile against. The rule is
 *
 *   __HGGC_ARCH__ == 150   -> PPU1.5 (ZW-M890P), tc02, actlize ppu0015
 *   __HGGC_ARCH__ == 100   -> PPU1.0/1.1 (ZW-810E), tc01, actlize ppu0010
 *
 * ``__HGGC_ARCH__`` is predefined by the PPU compiler driver from the target
 * chip, so the kernel sources do not need any -D flag to distinguish
 * generations. actlize gates its own inline asm on ``__HGGC_ARCH__`` directly,
 * so nothing has to be forced from here either -- the older cutlass3 fork
 * needed an ACOMPUTE_VERSION override at this point, which is gone.
 *
 * After including this header, the following architecture-invariant
 * aliases are available under namespace ``flashkda``:
 *
 *   PpuBf16MMA     -- 16x16x16 f32<=bf16*bf16+f32 TN atom traits
 *   PpuF16MMA      -- 16x16x16 f32<=f16*f16+f32 TN atom traits
 *   PpuF16AccMMA   -- 16x16x16 f16<=f16*f16+f16 TN atom traits (Neumann)
 *   PpuAiuLoad<Bits, Elem, Trans, CUBE_H, CUBE_W>
 *   PpuTsmLdSwzl<Elem, CUBE_H, CUBE_W, Swap, Trans, InstNum>
 *
 * The AIU/TSM aliases accept the CUBE_H/CUBE_W/InstNum shape parameters that
 * PPU1.5 encodes as template arguments; on PPU1.0/1.1 those knobs are carried
 * runtime through AiuDesc / TSM_LD_SWZL_IMPL, so the extra template arguments
 * are absorbed silently.
 */
#pragma once

// --- Detect PPU generation ------------------------------------------------
#if defined(__HGGC_ARCH__)
#  if __HGGC_ARCH__ >= 150
#    define FLASHKDA_ARCH_150 1
#  elif __HGGC_ARCH__ >= 100
#    define FLASHKDA_ARCH_100 1
#  else
#    error "FlashKDA prefill: unsupported __HGGC_ARCH__ (< 100). Expected 100 (PPU1.0/1.1) or 150 (PPU1.5)."
#  endif
#else
// Host / non-PPU translation units still include these kernel headers via the
// pybind binding. Fall back to the 150 traits family so the CuTe types exist
// (all device asm is guarded and compiles to (void) casts on the host side).
#  define FLASHKDA_ARCH_150 1
#endif

#include <cstdint>
#include <hggc_fp16.h>
#include <cutlass/bfloat16.h>

#include "cute/tensor.hpp"
#include "cute/arch/mma.hpp"

// The ppu0010 copy header defines ``AiuDesc``; the ppu0015 header only
// declares it as a stripped comment and reuses the 10000 definition. Include
// both copy headers only on PPU1.5 (where kernels reference the 10500 traits
// directly); on PPU1.0/1.1 the 10000 header alone is enough and pulling in
// 10500 would drag tc02-only IMPLs into the translation unit.
#if defined(FLASHKDA_ARCH_150)
#  include "cute/arch/mma_ppu0015.hpp"
#  include "cute/atom/mma_traits_ppu0015.hpp"
#  include "cute/arch/copy_ppu0010_aiu.hpp"  // AiuDesc definition
#  include "cute/arch/copy_ppu0015_aiu.hpp"
#else
#  include "cute/arch/mma_ppu0010.hpp"
#  include "cute/atom/mma_traits_ppu0010.hpp"
#  include "cute/arch/copy_ppu0010_aiu.hpp"
#endif

namespace flashkda {

// ---- MMA traits aliases --------------------------------------------------
#if defined(FLASHKDA_ARCH_150)
using PpuBf16MMA    = cute::PPU0015_16x16x16_F32BF16BF16F32_TN;
using PpuF16MMA     = cute::PPU0015_16x16x16_F32F16F16F32_TN;
using PpuF16AccMMA  = cute::PPU0015_16x16x16_F16F16F16F16_TN;
#else
using PpuBf16MMA    = cute::PPU0010_16x16x16_F32BF16BF16F32_TN;
using PpuF16MMA     = cute::PPU0010_16x16x16_F32F16F16F32_TN;
using PpuF16AccMMA  = cute::PPU0010_16x16x16_F16F16F16F16_TN;
#endif

// ---- AIU / TSM traits variadic aliases -----------------------------------
// Both actlize families expose the same ``copy(void* smem, const void* gmem,
// AiuDesc, coord_w, coord_h, coord_n)`` and TSM ``copy(vreg, smem_base,
// coord_h, coord_w, cube_in_stage, stage)`` signatures at the traits layer;
// only the *template shape* differs. On PPU1.5 the cube shape is a compile-
// time template argument; on PPU1.0/1.1 it is carried at runtime through
// AiuDesc (for AIU_LOAD) and through the TSM_LD_SWZL_IMPL runtime helpers.
// The extra template parameters are declared here so callers can name shapes
// uniformly across arches.
#if defined(FLASHKDA_ARCH_150)
template <int NumBits, typename Element, bool Trans, int CUBE_H, int CUBE_W,
          bool Swzl = true>
using PpuAiuLoad =
    cute::PPU0015_AIU_LOAD<cute::Int<NumBits>, Element, Trans, CUBE_H, CUBE_W, Swzl>;

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum>
using PpuTsmLdSwzl =
    cute::PPU0015_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum>;
#else
// PPU1.0/1.1 actlize declares PPU0010_AIU_LOAD<Bits, Elem, Trans,
// Swzl=true, Enable> and PPU0010_TSM_LD_SWZL<Elem, CH, CW, Swap, Trans,
// InstNum=1> (no ``Num`` slot). Absorb the 10500 template surface: CUBE_H/W
// on AIU_LOAD are runtime through AiuDesc.
template <int NumBits, typename Element, bool Trans, int CUBE_H, int CUBE_W,
          bool Swzl = true>
using PpuAiuLoad =
    cute::PPU0010_AIU_LOAD<cute::Int<NumBits>, Element, Trans, Swzl>;

template <typename Element, int CUBE_H, int CUBE_W, bool Swap, bool Trans, int InstNum>
using PpuTsmLdSwzl =
    cute::PPU0010_TSM_LD_SWZL<Element, CUBE_H, CUBE_W, Swap, Trans, InstNum>;
#endif

}  // namespace flashkda
