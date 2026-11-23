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
 * Fixed FLA L2 stage for tc01: BT=16 rows, 8 warps (256 threads), D=128.
 *
 * This is deliberately a scalar/FMA CuTe kernel.  CuTe owns the logical
 * (rows, D) gmem tensor/layout and all indexing; no MMA atom or AIU copy is
 * introduced because the frozen FLA stage is a zero-MMA FP32 reduction.
 */
#include <hggc_bf16.h>
#include <hggc_runtime.h>

#include <cstdint>

#include "gate_scan.cuh"

namespace fla_l2_cute {

constexpr int kD = 128;
constexpr int kBlockRows = 16;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr float kEps = 1.0e-6f;

union alignas(16) Bf16x8 {
  uint4 vector;
  __ppu_bfloat162 pairs[4];
};

__global__ void l2norm_bf16_d128_bt16_kernel(
    const __ppu_bfloat16* __restrict__ x,
    __ppu_bfloat16* __restrict__ y,
    int64_t rows) {
  using namespace cute;

  // Real CuTe views over row-major [rows, 128] input and output.  The fixed
  // static feature dimension is part of the layout rather than recomputed as
  // ad-hoc byte arithmetic at each load/store.
  Tensor gX = make_tensor(
      make_gmem_ptr(x),
      make_layout(make_shape(rows, Int<kD>{}),
                  make_stride(Int<kD>{}, Int<1>{})));
  Tensor gY = make_tensor(
      make_gmem_ptr(y),
      make_layout(make_shape(rows, Int<kD>{}),
                  make_stride(Int<kD>{}, Int<1>{})));

  // Frozen FLA TTGIR layout: sizePerThread=[1,8],
  // threadsPerWarp=[2,16], warpsPerCTA=[8,1], order=[1,0].  Thus each
  // 16-lane halfwarp owns one row and each lane owns eight contiguous D
  // values; 16 halfwarps cover BT=16 rows in one 256-thread CTA.
  const int row_in_block = static_cast<int>(threadIdx.x) >> 4;
  const int lane_in_row = static_cast<int>(threadIdx.x) & 15;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kBlockRows + row_in_block;
  if (row >= rows) return;

  const int first_d = lane_in_row * 8;
  Bf16x8 packed_input;
  packed_input.vector =
      *reinterpret_cast<const uint4*>(&gX(row, first_d));
  float values[8];
  float sum_sq = 0.0f;
#pragma unroll
  for (int pair_index = 0; pair_index < 4; ++pair_index) {
    const float2 pair = __bfloat1622float2(packed_input.pairs[pair_index]);
    values[pair_index * 2] = pair.x;
    values[pair_index * 2 + 1] = pair.y;
    sum_sq += pair.x * pair.x;
    sum_sq += pair.y * pair.y;
  }

#pragma unroll
  for (int lane_mask = 8; lane_mask > 0; lane_mask >>= 1) {
    // Match Triton's four-step half-warp butterfly reduction.  Every lane
    // receives the row sum, avoiding a fifth lane-0 broadcast instruction.
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, lane_mask, 16);
  }
  const float denominator = sqrtf(sum_sq + kEps);

  Bf16x8 packed_output;
#pragma unroll
  for (int pair_index = 0; pair_index < 4; ++pair_index) {
    // One final conversion is the only output rounding point.
    packed_output.pairs[pair_index] = __float22bfloat162_rn(make_float2(
        values[pair_index * 2] / denominator,
        values[pair_index * 2 + 1] / denominator));
  }
  *reinterpret_cast<uint4*>(&gY(row, first_d)) = packed_output.vector;
}

// Fused q+k variant: one launch normalizes two [rows_each, 128] pairs; the first half of the rows selects
// (x0, y0) and the second half (x1, y1); the per-row math is identical to the single-pair kernel.
// Rows are (token, head) pairs in token-major order. With Strided the inputs are [T, heads, 128] views
// with a token stride (x0_stride / x1_stride elements) over packed head rows; the outputs stay dense.
struct ShortGate {
  const __ppu_bfloat16* raw_gate;
  const __ppu_bfloat16* raw_beta;
  const float* a_log;
  const float* bias;
  float* decay;
  float* beta;
  int64_t beta_stride;
  float lower;
};

template <bool Strided, bool Clear, bool PrepareGate = false>
__device__ __forceinline__ void normalize_pair(
    const __ppu_bfloat16* __restrict__ x0,
    __ppu_bfloat16* __restrict__ y0,
    const __ppu_bfloat16* __restrict__ x1,
    __ppu_bfloat16* __restrict__ y1,
    int64_t rows_each, int heads, int64_t x0_stride, int64_t x1_stride,
    uint4* scratch, int64_t vectors, int blocks, ShortGate gate = {}) {
  using namespace cute;
  if constexpr (Clear) {
    const int64_t thread = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t i = thread; i < vectors; i += int64_t(blocks) * blockDim.x)
      scratch[i] = make_uint4(0, 0, 0, 0);
  }

  const int row_in_block = static_cast<int>(threadIdx.x) >> 4;
  const int lane_in_row = static_cast<int>(threadIdx.x) & 15;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kBlockRows + row_in_block;
  if (row >= 2 * rows_each) return;
  const bool second = row >= rows_each;
  const int64_t local_row = second ? row - rows_each : row;

  Tensor gX = make_tensor(
      make_gmem_ptr(second ? x1 : x0),
      make_layout(make_shape(rows_each, Int<kD>{}),
                  make_stride(Int<kD>{}, Int<1>{})));
  Tensor gY = make_tensor(
      make_gmem_ptr(second ? y1 : y0),
      make_layout(make_shape(rows_each, Int<kD>{}),
                  make_stride(Int<kD>{}, Int<1>{})));

  const int first_d = lane_in_row * 8;
  const __ppu_bfloat16* src;
  if constexpr (Strided) {
    const int64_t token = local_row / heads;
    const int64_t head = local_row - token * heads;
    src = (second ? x1 : x0) + token * (second ? x1_stride : x0_stride)
        + head * kD + first_d;
  } else {
    src = &gX(local_row, first_d);
  }
  Bf16x8 packed_input;
  packed_input.vector = *reinterpret_cast<const uint4*>(src);
  float values[8];
  float sum_sq = 0.0f;
#pragma unroll
  for (int pair_index = 0; pair_index < 4; ++pair_index) {
    const float2 pair = __bfloat1622float2(packed_input.pairs[pair_index]);
    values[pair_index * 2] = pair.x;
    values[pair_index * 2 + 1] = pair.y;
    sum_sq += pair.x * pair.x;
    sum_sq += pair.y * pair.y;
  }

#pragma unroll
  for (int lane_mask = 8; lane_mask > 0; lane_mask >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, lane_mask, 16);
  }
  const float denominator = sqrtf(sum_sq + kEps);

  Bf16x8 packed_output;
#pragma unroll
  for (int pair_index = 0; pair_index < 4; ++pair_index) {
    packed_output.pairs[pair_index] = __float22bfloat162_rn(make_float2(
        values[pair_index * 2] / denominator,
        values[pair_index * 2 + 1] / denominator));
  }
  *reinterpret_cast<uint4*>(&gY(local_row, first_d)) = packed_output.vector;
  if constexpr (PrepareGate) {
    if (!second) {
      const int head = local_row % heads;
      const int64_t offset = local_row * kD + first_d;
      const int64_t token = local_row / heads;
      Bf16x8 packed_gate;
      packed_gate.vector = *reinterpret_cast<const uint4*>(gate.raw_gate + offset);
      const float a = expf(gate.a_log[head]);
      const float* bias = gate.bias + head * kD + first_d;
      float decay[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 g = __bfloat1622float2(packed_gate.pairs[i]);
        decay[i * 2] = expf(gate.lower / (1.0f + expf(-a * (g.x + bias[i * 2]))));
        decay[i * 2 + 1] = expf(gate.lower / (1.0f + expf(-a * (g.y + bias[i * 2 + 1]))));
      }
      auto* output = reinterpret_cast<float4*>(gate.decay + offset);
      output[0] = make_float4(decay[0], decay[1], decay[2], decay[3]);
      output[1] = make_float4(decay[4], decay[5], decay[6], decay[7]);
      if (lane_in_row == 0) {
        const float beta = __bfloat162float(gate.raw_beta[token * gate.beta_stride + head]);
        gate.beta[local_row] = 1.0f / (1.0f + expf(-beta));
      }
    }
  }
}

template <bool Strided>
__global__ void l2norm2_bf16_d128_bt16_kernel(
    const __ppu_bfloat16* __restrict__ x0, __ppu_bfloat16* __restrict__ y0,
    const __ppu_bfloat16* __restrict__ x1, __ppu_bfloat16* __restrict__ y1,
    int64_t rows_each, int heads, int64_t x0_stride, int64_t x1_stride) {
  normalize_pair<Strided, false>(
      x0, y0, x1, y1, rows_each, heads, x0_stride, x1_stride, nullptr, 0, gridDim.x);
}

template <bool Strided>
__global__ void prepare_short(
    const __ppu_bfloat16* q, __ppu_bfloat16* qn,
    const __ppu_bfloat16* k, __ppu_bfloat16* kn, int64_t rows_each,
    int heads, int64_t q_stride, int64_t k_stride, ShortGate gate) {
  normalize_pair<Strided, false, true>(
      q, qn, k, kn, rows_each, heads, q_stride, k_stride, nullptr, 0, gridDim.x, gate);
}

template <bool Strided, bool DefaultBound>
__global__ void prepare_front(
    const __ppu_bfloat16* q, __ppu_bfloat16* qn,
    const __ppu_bfloat16* k, __ppu_bfloat16* kn, int64_t rows_each,
    int heads, int64_t q_stride, int64_t k_stride, uint4* scratch, int64_t vectors,
    const __ppu_bfloat16* g, const float* a_log, const float* bias, float* gates,
    const int32_t* cu, const int32_t* indices, int n, int chunks, int tokens,
    float lower, int norm_blocks) {
  if (blockIdx.x < norm_blocks) {
    normalize_pair<Strided, true>(
        q, qn, k, kn, rows_each, heads, q_stride, k_stride, scratch, vectors,
        norm_blocks);
  } else {
    const int tile = blockIdx.x - norm_blocks;
    const int chunk = (tile / 4) % chunks;
    const int head = (tile / 4) / chunks;
    __shared__ float carry[16 * 32];
    fla_gate::scan<DefaultBound>(g, a_log, bias, gates, tokens, cu, indices, n, heads,
                           lower, chunk, tile % 4, head, carry);
  }
}


}  // namespace fla_l2_cute

hggcError_t launch_fla_l2norm_bf16_d128_bt16(
    const void* x, void* y, int64_t rows, hggcStream_t stream) {
  if (rows == 0) return hggcSuccess;
  const dim3 grid(rows / fla_l2_cute::kBlockRows +
                  (rows % fla_l2_cute::kBlockRows != 0));
  const dim3 block(fla_l2_cute::kThreads);
  fla_l2_cute::l2norm_bf16_d128_bt16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __ppu_bfloat16*>(x),
      static_cast<__ppu_bfloat16*>(y), rows);
  return hggcGetLastError();
}

hggcError_t launch_fla_l2norm2_bf16_d128_bt16(
    const void* x0, void* y0, const void* x1, void* y1, int64_t rows_each,
    int heads, int64_t x0_stride, int64_t x1_stride, hggcStream_t stream) {
  if (rows_each == 0) return hggcSuccess;
  const int64_t rows = 2 * rows_each;
  const dim3 grid(rows / fla_l2_cute::kBlockRows +
                  (rows % fla_l2_cute::kBlockRows != 0));
  const dim3 block(fla_l2_cute::kThreads);
  const int64_t packed = int64_t(heads) * fla_l2_cute::kD;
  auto* kernel = x0_stride == packed && x1_stride == packed
      ? fla_l2_cute::l2norm2_bf16_d128_bt16_kernel<false>
      : fla_l2_cute::l2norm2_bf16_d128_bt16_kernel<true>;
  kernel<<<grid, block, 0, stream>>>(
      static_cast<const __ppu_bfloat16*>(x0),
      static_cast<__ppu_bfloat16*>(y0),
      static_cast<const __ppu_bfloat16*>(x1),
      static_cast<__ppu_bfloat16*>(y1), rows_each, heads, x0_stride, x1_stride);
  return hggcGetLastError();
}

hggcError_t launch_fla_front_prepare(
    const void* q, void* qn, const void* k, void* kn, int64_t rows_each,
    int heads, int64_t q_stride, int64_t k_stride, void* scratch, int64_t bytes,
    const void* g, const float* a_log, const float* bias, float* gates,
    const int32_t* cu, const int32_t* indices, int n, int chunks, int tokens,
    float lower, hggcStream_t stream) {
  using namespace fla_l2_cute;
  const int norm_blocks = (2 * rows_each + kBlockRows - 1) / kBlockRows;
  const int packed = heads * kD;
  const bool strided = q_stride != packed || k_stride != packed;
  auto kernel = lower == -5.0f
      ? (strided ? prepare_front<true, true> : prepare_front<false, true>)
      : (strided ? prepare_front<true, false> : prepare_front<false, false>);
  kernel<<<dim3(norm_blocks + 4 * heads * chunks), dim3(kThreads), 0, stream>>>(
      static_cast<const __ppu_bfloat16*>(q), static_cast<__ppu_bfloat16*>(qn),
      static_cast<const __ppu_bfloat16*>(k), static_cast<__ppu_bfloat16*>(kn),
      rows_each, heads, q_stride, k_stride, static_cast<uint4*>(scratch),
      bytes / sizeof(uint4), static_cast<const __ppu_bfloat16*>(g), a_log, bias,
      gates, cu, indices, n, chunks, tokens, lower, norm_blocks);
  return hggcGetLastError();
}

hggcError_t launch_fla_short_prepare(
    const void* q, void* qn, const void* k, void* kn, int64_t rows_each,
    int heads, int64_t q_stride, int64_t k_stride, const void* raw_gate,
    const void* raw_beta, const float* a_log, const float* bias,
    int64_t beta_stride, float lower, float* decay, float* beta,
    hggcStream_t stream) {
  using namespace fla_l2_cute;
  const dim3 grid((2 * rows_each + kBlockRows - 1) / kBlockRows), block(kThreads);
  const int64_t packed = int64_t(heads) * kD;
  auto* kernel = q_stride == packed && k_stride == packed
      ? prepare_short<false>
      : prepare_short<true>;
  const ShortGate gate{static_cast<const __ppu_bfloat16*>(raw_gate),
      static_cast<const __ppu_bfloat16*>(raw_beta), a_log, bias, decay, beta,
      beta_stride, lower};
  kernel<<<grid, block, 0, stream>>>(
      static_cast<const __ppu_bfloat16*>(q), static_cast<__ppu_bfloat16*>(qn),
      static_cast<const __ppu_bfloat16*>(k), static_cast<__ppu_bfloat16*>(kn),
      rows_each, heads, q_stride, k_stride, gate);
  return hggcGetLastError();
}
