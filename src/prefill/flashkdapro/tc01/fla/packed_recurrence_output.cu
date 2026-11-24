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

#define FLA_RECURRENCE_BODY_ONLY 1
#include "heads.h"

#ifdef FLA_PACKED_BV16
#undef FLA_RECURRENCE_BV16
#define FLA_RECURRENCE_BV16 1
#undef fla_recurrence
#define fla_recurrence FLA_JOIN_HEAD(fla_recurrence_packed_bv16, FLA_HEADS)
#undef FlaRecurrenceRawReader
#define FlaRecurrenceRawReader FLA_JOIN_HEAD(FlaRecurrenceRawReaderPackedBv16, FLA_HEADS)
#define FLA_PACKED_KERNEL FLA_JOIN_HEAD(fla_packed_recurrence_output_bv16, FLA_HEADS)
#define FLA_PACKED_CONFIGURE FLA_JOIN_HEAD(configure_fla_packed_recurrence_output_bv16, FLA_HEADS)
#define FLA_PACKED_LAUNCH FLA_JOIN_HEAD(launch_fla_packed_recurrence_output_bv16, FLA_HEADS)
#else
#define FLA_PACKED_KERNEL FLA_JOIN_HEAD(fla_packed_recurrence_output, FLA_HEADS)
#define FLA_PACKED_CONFIGURE FLA_JOIN_HEAD(configure_fla_packed_recurrence_output, FLA_HEADS)
#define FLA_PACKED_LAUNCH FLA_JOIN_HEAD(launch_fla_packed_recurrence_output, FLA_HEADS)
#endif

#include <type_traits>
#include "../../kernels/flashkda_ppu_util.cuh"
#include "recurrence.cu"
#include "cp_prescan_helpers.cuh"
#include "cp_output_store.cuh"
#include "cp_fused_helpers.cuh"
#include "cp_pretriangular_tail.cuh"
#include "cp_pretriangular_body.cuh"

__global__ __launch_bounds__(128, 1) __block_size__((128, 1, 1))
void FLA_PACKED_KERNEL(
    const uint16_t* qg, const uint16_t* a, uint16_t* output,
    const uint16_t* kg, const uint16_t* u, const uint16_t* w,
    const float* gate, const void* initial, void* final, int state_is_fp32,
    const int32_t* state_indices, const int32_t* cu, const int64_t* offsets,
    int tokens) {
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100
  const int seq = blockIdx.z;
  const int bos = cu[seq], eos = cu[seq + 1];
  if (bos < 0 || eos < bos || eos > tokens) return;
  const int length = eos - bos;
  if (state_indices) {
    state_indices += seq;
  } else {
    const int64_t slot_bytes = int64_t(FLA_HEADS) * 128 * 128 *
        (state_is_fp32 ? sizeof(float) : sizeof(uint16_t));
    if (initial) initial = static_cast<const uint8_t*>(initial) + seq * slot_bytes;
    if (final) final = static_cast<uint8_t*>(final) + seq * slot_bytes;
  }
  __shared__ __align__(128) uint16_t output_operand[8192];
  extern __shared__ __align__(128) unsigned char storage[];
  auto& stage = *reinterpret_cast<fla_recurrence::Stage*>(storage);
  const int chunks = (length + 63) / 64;
  using fla_recurrence::CpPass;
  using fla_recurrence::fused_pretriangular_output_body_pipeline5;
  if (chunks <= 32)
    fused_pretriangular_output_body_pipeline5<0, CpPass::Zero, false>(
        qg, a, output, kg, u, w, nullptr, gate, nullptr, nullptr, initial, final,
        state_is_fp32, state_indices, nullptr, nullptr, nullptr, 0, 1,
        cu + seq, offsets + seq, length, stage, output_operand);
  else if (chunks <= 128)
    fused_pretriangular_output_body_pipeline5<1, CpPass::Zero, false>(
        qg, a, output, kg, u, w, nullptr, gate, nullptr, nullptr, initial, final,
        state_is_fp32, state_indices, nullptr, nullptr, nullptr, 0, 1,
        cu + seq, offsets + seq, length, stage, output_operand);
  else
    fused_pretriangular_output_body_pipeline5<2, CpPass::Zero, false>(
        qg, a, output, kg, u, w, nullptr, gate, nullptr, nullptr, initial, final,
        state_is_fp32, state_indices, nullptr, nullptr, nullptr, 0, 1,
        cu + seq, offsets + seq, length, stage, output_operand);
#else
  __trap();
#endif
}

extern "C" hggcError_t FLA_PACKED_CONFIGURE() {
  return hggcFuncSetAttribute(FLA_PACKED_KERNEL,
      hggcFuncAttributeMaxDynamicSharedMemorySize, FLA_RECURRENCE_DYNAMIC_SHARED_BYTES);
}

extern "C" hggcError_t FLA_PACKED_LAUNCH(
    const void* qg, const void* a, void* output, const void* kg, const void* u,
    const void* w, const float* gate, const void* initial, void* final,
    int state_is_fp32, const int32_t* state_indices, const int32_t* cu,
    const int64_t* offsets, int tokens, int sequences, hggcStream_t stream) {
  using U = uint16_t;
  FLA_PACKED_KERNEL<<<
      dim3(128 / fla_recurrence::kBv, FLA_HEADS, sequences), 128,
      FLA_RECURRENCE_DYNAMIC_SHARED_BYTES, stream>>>(
      static_cast<const U*>(qg), static_cast<const U*>(a), static_cast<U*>(output),
      static_cast<const U*>(kg), static_cast<const U*>(u), static_cast<const U*>(w),
      gate, initial, final, state_is_fp32, state_indices, cu, offsets, tokens);
  return hggcGetLastError();
}
