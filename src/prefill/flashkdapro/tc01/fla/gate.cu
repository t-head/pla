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

#include "heads.h"
/* FLA-aligned GLM-5.3 H8 gate: BT64, BS32, 256 threads, mma_en=0. */
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <cstdint>

#include "gate_scan.cuh"

namespace fla_glm53_gate_aligned_m1 {

template<bool DefaultBound>
__global__ void gate_cumsum_bf16_h8_k128_bt64_bs32_aligned_m1_kernel(
    const __ppu_bfloat16* __restrict__ input,
    const float* __restrict__ a_log,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int tokens, const int32_t* cu, const int32_t* indices, int n,
    float lower_bound) {
  __shared__ float carry[16 * 32];
  fla_gate::scan<DefaultBound>(input, a_log, bias, output, tokens, cu, indices, n,
      FLA_CUTE_INT(FLA_DEVICE_HEADS(z)), lower_bound, blockIdx.y, blockIdx.x,
      blockIdx.z, carry);
}

}  // namespace fla_glm53_gate_aligned_m1

hggcError_t launch_fla_gate_cumsum_bf16_h8_aligned_m1(
    const void* g,
    const float* a,
    const float* bias,
    float* out,
    int tokens, const int32_t* cu, const int32_t* indices,
    int n, int chunks, int heads, float lower_bound,
    hggcStream_t stream) {
  if (chunks == 0) return hggcSuccess;
  const dim3 grid(4, chunks, heads);
  if (lower_bound == -5.0f)
    fla_glm53_gate_aligned_m1::gate_cumsum_bf16_h8_k128_bt64_bs32_aligned_m1_kernel<true>
        <<<grid, 256, 0, stream>>>(static_cast<const __ppu_bfloat16*>(g), a,
                                   bias, out, tokens, cu, indices, n, lower_bound);
  else
    fla_glm53_gate_aligned_m1::gate_cumsum_bf16_h8_k128_bt64_bs32_aligned_m1_kernel<false>
        <<<grid, 256, 0, stream>>>(static_cast<const __ppu_bfloat16*>(g), a,
                                   bias, out, tokens, cu, indices, n, lower_bound);
  return hggcGetLastError();
}
