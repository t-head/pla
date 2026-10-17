/*
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * PPU1.5 port of the Kimi K3 KDA fused decode reference kernel.
 * The device code and the extern "C" launcher below are copied verbatim from
 * vllm/csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu (the golden
 * "fused_kda_decode_kernel"); only the host glue changed:
 *   - the sm90-only grid-dependency intrinsics are dropped entirely (they were
 *     guarded by an arch predicate that never holds on PPU1.5, where the
 *     compiler defines __HGGC_ARCH__ instead);
 *   - the sm90 launch-with-attributes call plus its programmatic-stream-
 *     serialization attribute is replaced with a plain <<<>>> launch;
 *   - the torch::stable wrapper is replaced with an at::Tensor op wrapper
 *     exposing the SAME I/O contract as torch.ops._C.fused_kda_decode so it can
 *     be driven from Python and compared against ppu_kda_decode_fusion.
 */

#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_runtime.h>
#include <array>
#include <optional>

#include <stdexcept>
#include <string>

namespace {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kKernelWidth = 4;
constexpr int kConvStateWidth = kKernelWidth - 1;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;

struct KdaDecodeStrides {
  int64_t x_row;
  int64_t beta_row;
  int64_t onorm_row;
  int64_t conv_slot;
  int64_t state_slot;
};

__device__ __forceinline__ float bf16_load(const __ppu_bfloat16* ptr,
                                           int64_t idx) {
  return __bfloat162float(ptr[idx]);
}

__device__ __forceinline__ float bf16_load(const float* ptr, int64_t idx) {
  return ptr[idx];
}

template <int kChannels>
__device__ __forceinline__ float conv_weight_load(const float* ptr, int channel,
                                                  int width) {
  return ptr[width * kChannels + channel];
}

__device__ __forceinline__ __ppu_bfloat16 bf16_store(float value) {
  return __float2bfloat16(value);
}

template <bool kUseCacheGlobalStore>
__device__ __forceinline__ void store_state_float4(float* ptr, float4 value) {
  if constexpr (kUseCacheGlobalStore) {
    __stcg(reinterpret_cast<float4*>(ptr), value);
  } else {
    *reinterpret_cast<float4*>(ptr) = value;
  }
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void cp_async_cg_16b(float* smem_ptr,
                                                const float* gmem_ptr) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ppu.cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("ppu.cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("ppu.cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_1() {
  asm volatile("ppu.cp.async.wait_group 1;\n" ::: "memory");
}

template <int kCopyThreads>
__device__ __forceinline__ void cp_async_state_chunk_for(float* s_state,
                                                         const float* state,
                                                         int slot, int i_hv,
                                                         int HV, int chunk) {
  constexpr int kFloat4PerChunk = kChunkV * kDimK / 4;
  const int tid = threadIdx.x;
  const int stage = chunk & 1;
  const int v_base = chunk * kChunkV;
  for (int linear4 = tid; linear4 < kFloat4PerChunk; linear4 += kCopyThreads) {
    const int elem = linear4 * 4;
    const int row = elem / kDimK;
    const int k = elem - row * kDimK;
    float* dst = s_state + (stage * kChunkV + row) * kDimK + k;
    const float* src =
        state + ((slot * HV + i_hv) * kDimV + v_base + row) * kDimK + k;
    cp_async_cg_16b(dst, src);
  }
  cp_async_commit();
}

__device__ __forceinline__ void cp_async_state_chunk(float* s_state,
                                                     const float* state,
                                                     int slot, int i_hv, int HV,
                                                     int chunk) {
  cp_async_state_chunk_for<kThreads>(s_state, state, slot, i_hv, HV, chunk);
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float warp_total = warp_reduce_sum(value);
  if (lane == 0) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kWarps ? scratch[lane] : 0.0f;
    block_total = warp_reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_for(float x, float y,
                                                      float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const float warp_x = warp_reduce_sum(x);
  const float warp_y = warp_reduce_sum(y);
  if (lane == 0) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = warp_reduce_sum(block_x);
    block_y = warp_reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

__device__ __forceinline__ Sum2 block_reduce_sum2(float x, float y,
                                                  float* scratch) {
  return block_reduce_sum2_for<kWarps>(x, y, scratch);
}

template <int kReduceWarps>
__device__ __forceinline__ float block_reduce_sum_active_for(float value,
                                                             float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_total = 0.0f;
  if (warp < kReduceWarps) {
    warp_total = warp_reduce_sum(value);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_total;
  }
  __syncthreads();

  float block_total = 0.0f;
  if (warp == 0) {
    block_total = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_total = warp_reduce_sum(block_total);
    if (lane == 0) {
      scratch[0] = block_total;
    }
  }
  __syncthreads();
  return scratch[0];
}

template <int kReduceWarps>
__device__ __forceinline__ Sum2 block_reduce_sum2_active_for(float x, float y,
                                                             float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float warp_x = 0.0f;
  float warp_y = 0.0f;
  if (warp < kReduceWarps) {
    warp_x = warp_reduce_sum(x);
    warp_y = warp_reduce_sum(y);
  }
  if (lane == 0 && warp < kReduceWarps) {
    scratch[warp] = warp_x;
    scratch[kReduceWarps + warp] = warp_y;
  }
  __syncthreads();

  float block_x = 0.0f;
  float block_y = 0.0f;
  if (warp == 0) {
    block_x = lane < kReduceWarps ? scratch[lane] : 0.0f;
    block_y = lane < kReduceWarps ? scratch[kReduceWarps + lane] : 0.0f;
    block_x = warp_reduce_sum(block_x);
    block_y = warp_reduce_sum(block_y);
    if (lane == 0) {
      scratch[0] = block_x;
      scratch[1] = block_y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

template <bool kApplyOnorm, bool kUseStaticDecodeLayout = false,
          int kFixedHeads = 0, int kFixedValueHeads = 0,
          bool kUseHeadGrid = false, bool kAccumulateOnormSumsq = false,
          bool kUseActiveQkReduction = false, bool kUseCacheGlobalStore = false,
          bool kComputeOutputBeforeStore = false, bool kSkipWarpSync = false,
          bool kPreloadOnormParams = false,
          bool kPrefetchNextStateChunk = false,
          bool kUseActiveOnormReduction = false, bool kUpdateConvState = false,
          bool kUseLowerBound = false, bool kApplyBetaSigmoid = true>
__global__
__launch_bounds__(kThreads, 2) void kda_decode_fusion_many_heads_kernel(
    const __ppu_bfloat16* __restrict__ x_q,
    const __ppu_bfloat16* __restrict__ x_k,
    const __ppu_bfloat16* __restrict__ x_v, const float* __restrict__ w_q_t,
    const float* __restrict__ w_k_t, const float* __restrict__ w_v_t,
    const float* __restrict__ bias_q, const float* __restrict__ bias_k,
    const float* __restrict__ bias_v, __ppu_bfloat16* __restrict__ cs_q,
    __ppu_bfloat16* __restrict__ cs_k, __ppu_bfloat16* __restrict__ cs_v,
    const float* __restrict__ a_log, const __ppu_bfloat16* __restrict__ g,
    const float* __restrict__ dt_bias, const __ppu_bfloat16* __restrict__ beta,
    const __ppu_bfloat16* __restrict__ onorm_g,
    const float* __restrict__ onorm_weight,
    const int* __restrict__ ssm_state_indices,
    const int* __restrict__ cu_seqlens, float* __restrict__ state,
    __ppu_bfloat16* __restrict__ out, int B, int H, int HV, float lower_bound,
    float scale, float onorm_eps, KdaDecodeStrides strides) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  int i_n;
  int i_hv;
  int i_h;
  int bos;
  int slot;
  if constexpr (kUseStaticDecodeLayout) {
    if constexpr (kUseHeadGrid) {
      i_n = blockIdx.x;
      i_hv = blockIdx.y;
    } else {
      const int nhv = blockIdx.x;
      i_n = nhv / kFixedValueHeads;
      i_hv = nhv - i_n * kFixedValueHeads;
    }
    i_h = i_hv;
    bos = i_n;
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  } else {
    const int nhv = blockIdx.x;
    i_n = nhv / HV;
    i_hv = nhv - i_n * HV;
    const int hv_per_h = HV / H;
    i_h = i_hv / hv_per_h;

    bos = cu_seqlens == nullptr ? i_n : cu_seqlens[i_n];
    const int eos = cu_seqlens == nullptr ? i_n + 1 : cu_seqlens[i_n + 1];
    if (eos <= bos) {
      return;
    }
    slot = ssm_state_indices == nullptr ? i_n : ssm_state_indices[i_n];
  }

  constexpr int kLocalDim = kFixedHeads * kDimK;
  constexpr int kPackedDim = 3 * kLocalDim;
  const int hk_off = i_h * kDimK;
  const int hv_off = i_hv * kDimV;
  constexpr int hv_count = kFixedValueHeads;

  // Framework pad slot. sglang marks CUDA-graph padding rows with
  // PAD_SLOT_ID = -1, which has no backing memory, so slot * strides.state_slot
  // would address outside the pool; emit zeros and touch neither state nor
  // conv_state. vLLM instead pads with slot 0, a real reserved null_block, so it
  // stays >= 0 and keeps its original behaviour of computing the row as usual --
  // hence this guard needs no framework flag. slot only depends on i_n, so the
  // whole CTA agrees and the early return cannot desynchronise a __syncthreads().
  if (slot < 0) {
    if (tid < kDimV) {
      out[i_n * kLocalDim + i_hv * kDimV + tid] = bf16_store(0.0f);
    }
    return;
  }

  float* const state_for_slot = state + slot * strides.state_slot;
  const int64_t conv_slot_offset = slot * strides.conv_slot;
  __ppu_bfloat16* const cs_q_for_slot = cs_q + conv_slot_offset;
  __ppu_bfloat16* const cs_k_for_slot = cs_k + conv_slot_offset;
  __ppu_bfloat16* const cs_v_for_slot = cs_v + conv_slot_offset;

  __shared__ float s_state[2][kChunkV][kDimK];
  __shared__ float s_q[kDimK];
  __shared__ float s_k[kDimK];
  __shared__ float s_decay[kDimK];
  __shared__ float s_v[kDimV];
  __shared__ float s_o[kDimV];
  __shared__ float s_reduce[kThreads];
  __shared__ float s_beta;
  float pre_onorm_gate = 0.0f;
  float pre_onorm_weight = 0.0f;

  cp_async_state_chunk(&s_state[0][0][0], state_for_slot, 0, i_hv, hv_count, 0);

  if constexpr (kUpdateConvState) {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const int64_t xq_idx = bos * strides.x_row + i_h * kDimK + k;
      const float exp_a =
          __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q == nullptr ? 0.0f : bf16_load(bias_q, hk);
      float k_acc = bias_k == nullptr ? 0.0f : bf16_load(bias_k, hk);
      __ppu_bfloat16 q_shift0 = __float2bfloat16(0.0f);
      __ppu_bfloat16 q_shift1 = __float2bfloat16(0.0f);
      __ppu_bfloat16 k_shift0 = __float2bfloat16(0.0f);
      __ppu_bfloat16 k_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __ppu_bfloat16 q_state = cs_q_for_slot[hk + w * kPackedDim];
        const __ppu_bfloat16 k_state = cs_k_for_slot[hk + w * kPackedDim];
        q_acc += __bfloat162float(q_state) *
                 conv_weight_load<kLocalDim>(w_q_t, hk, w);
        k_acc += __bfloat162float(k_state) *
                 conv_weight_load<kLocalDim>(w_k_t, hk, w);
        if (w == 1) {
          q_shift0 = q_state;
          k_shift0 = k_state;
        } else if (w == 2) {
          q_shift1 = q_state;
          k_shift1 = k_state;
        }
      }
      const __ppu_bfloat16 q_new = x_q[xq_idx];
      const __ppu_bfloat16 k_new = x_k[xq_idx];
      q_acc += __bfloat162float(q_new) *
               conv_weight_load<kLocalDim>(w_q_t, hk, kKernelWidth - 1);
      k_acc += __bfloat162float(k_new) *
               conv_weight_load<kLocalDim>(w_k_t, hk, kKernelWidth - 1);

      cs_q_for_slot[hk] = q_shift0;
      cs_q_for_slot[hk + kPackedDim] = q_shift1;
      cs_q_for_slot[hk + 2 * kPackedDim] = q_new;
      cs_k_for_slot[hk] = k_shift0;
      cs_k_for_slot[hk + kPackedDim] = k_shift1;
      cs_k_for_slot[hk + 2 * kPackedDim] = k_new;

      s_q[k] = silu_fast(q_acc);
      s_k[k] = silu_fast(k_acc);

      const int64_t gate_idx = bos * kLocalDim + i_hv * kDimK + k;
      const float g_raw = bf16_load(g, gate_idx) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
      }
    }
  } else {
    if (tid < kDimK) {
      const int k = tid;
      const int hk = hk_off + k;
      const float exp_a =
          __shfl_sync(0xffffffffu, lane == 0 ? __expf(a_log[i_h]) : 0.0f, 0);

      float q_acc = bias_q == nullptr ? 0.0f : bf16_load(bias_q, hk);
      float k_acc = bias_k == nullptr ? 0.0f : bf16_load(bias_k, hk);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int cs_idx = hk + w * kPackedDim;
        q_acc += bf16_load(cs_q_for_slot, cs_idx) *
                 conv_weight_load<kLocalDim>(w_q_t, hk, w);
        k_acc += bf16_load(cs_k_for_slot, cs_idx) *
                 conv_weight_load<kLocalDim>(w_k_t, hk, w);
      }
      q_acc += bf16_load(x_q, bos * strides.x_row + i_h * kDimK + k) *
               conv_weight_load<kLocalDim>(w_q_t, hk, kKernelWidth - 1);
      k_acc += bf16_load(x_k, bos * strides.x_row + i_h * kDimK + k) *
               conv_weight_load<kLocalDim>(w_k_t, hk, kKernelWidth - 1);

      s_q[k] = silu_fast(q_acc);
      s_k[k] = silu_fast(k_acc);

      const int64_t gate_idx = bos * kLocalDim + i_hv * kDimK + k;
      const float g_raw = bf16_load(g, gate_idx) + dt_bias[hk];
      if constexpr (kUseLowerBound) {
        s_decay[k] = __expf(lower_bound * sigmoid_fast(exp_a * g_raw));
      } else {
        s_decay[k] = __expf(-exp_a * softplus_fast(g_raw));
      }
    }
  }

  if constexpr (kUpdateConvState) {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;
      const int64_t xv_idx = bos * strides.x_row + i_hv * kDimV + v;

      float v_acc = bias_v == nullptr ? 0.0f : bf16_load(bias_v, hvv);
      __ppu_bfloat16 v_shift0 = __float2bfloat16(0.0f);
      __ppu_bfloat16 v_shift1 = __float2bfloat16(0.0f);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const __ppu_bfloat16 v_state = cs_v_for_slot[hvv + w * kPackedDim];
        v_acc += __bfloat162float(v_state) *
                 conv_weight_load<kLocalDim>(w_v_t, hvv, w);
        if (w == 1) {
          v_shift0 = v_state;
        } else if (w == 2) {
          v_shift1 = v_state;
        }
      }
      const __ppu_bfloat16 v_new = x_v[xv_idx];
      v_acc += __bfloat162float(v_new) *
               conv_weight_load<kLocalDim>(w_v_t, hvv, kKernelWidth - 1);
      cs_v_for_slot[hvv] = v_shift0;
      cs_v_for_slot[hvv + kPackedDim] = v_shift1;
      cs_v_for_slot[hvv + 2 * kPackedDim] = v_new;
      s_v[v] = silu_fast(v_acc);

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t gate_idx = i_n * strides.onorm_row + i_hv * kDimV + v;
        pre_onorm_gate = sigmoid_fast(bf16_load(onorm_g, gate_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  } else {
    if (tid < kDimV) {
      const int v = tid;
      const int hvv = hv_off + v;

      float v_acc = bias_v == nullptr ? 0.0f : bf16_load(bias_v, hvv);
#pragma unroll
      for (int w = 0; w < kConvStateWidth; ++w) {
        const int cs_idx = hvv + w * kPackedDim;
        v_acc += bf16_load(cs_v_for_slot, cs_idx) *
                 conv_weight_load<kLocalDim>(w_v_t, hvv, w);
      }
      v_acc += bf16_load(x_v, bos * strides.x_row + i_hv * kDimV + v) *
               conv_weight_load<kLocalDim>(w_v_t, hvv, kKernelWidth - 1);
      s_v[v] = silu_fast(v_acc);

      if constexpr (kApplyOnorm && kPreloadOnormParams) {
        const int64_t gate_idx = i_n * strides.onorm_row + i_hv * kDimV + v;
        pre_onorm_gate = sigmoid_fast(bf16_load(onorm_g, gate_idx));
        pre_onorm_weight = onorm_weight[v];
      }
    }
  }

  if (tid == 0) {
    const float beta_raw = bf16_load(beta, bos * strides.beta_row + i_hv);
    if constexpr (kApplyBetaSigmoid) {
      s_beta = sigmoid_fast(beta_raw);
    } else {
      s_beta = beta_raw;
    }
  }
  __syncthreads();

  if constexpr (kPrefetchNextStateChunk && kNumChunks > 1) {
    cp_async_state_chunk(&s_state[0][0][0], state_for_slot, 0, i_hv, hv_count,
                         1);
  }

  const float q_sq = tid < kDimK ? s_q[tid] * s_q[tid] : 0.0f;
  const float k_sq = tid < kDimK ? s_k[tid] * s_k[tid] : 0.0f;
  Sum2 qk_sum;
  if constexpr (kUseActiveQkReduction) {
    qk_sum = block_reduce_sum2_active_for<kDimK / 32>(q_sq, k_sq, s_reduce);
  } else {
    qk_sum = block_reduce_sum2(q_sq, k_sq, s_reduce);
  }
  if (tid < kDimK) {
    s_q[tid] *= rsqrtf(qk_sum.x + 1.0e-6f) * scale;
    s_k[tid] *= rsqrtf(qk_sum.y + 1.0e-6f);
  }
  __syncthreads();

  const int k_base = lane * 4;
  const float4 q4 = *reinterpret_cast<const float4*>(s_q + k_base);
  const float4 k4 = *reinterpret_cast<const float4*>(s_k + k_base);
  const float4 decay4 = *reinterpret_cast<const float4*>(s_decay + k_base);
  float r_q[4] = {q4.x, q4.y, q4.z, q4.w};
  float r_k[4] = {k4.x, k4.y, k4.z, k4.w};
  float r_decay[4] = {decay4.x, decay4.y, decay4.z, decay4.w};
  float o_sumsq = 0.0f;

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    if constexpr (kPrefetchNextStateChunk && kNumChunks > 1) {
      if (chunk + 1 < kNumChunks) {
        cp_async_wait_group_1();
      } else {
        cp_async_wait_all();
      }
    } else {
      cp_async_wait_all();
    }
    if constexpr (!kSkipWarpSync) {
      __syncwarp();
    }

    if constexpr (!kPrefetchNextStateChunk) {
      if (chunk + 1 < kNumChunks) {
        cp_async_state_chunk(&s_state[0][0][0], state_for_slot, 0, i_hv,
                             hv_count, chunk + 1);
      }
    }

#pragma unroll
    for (int row = 0; row < kRowsPerWarp; row += 2) {
      const int v_row_a = warp + row * kWarps;
      const int v_row_b = warp + (row + 1) * kWarps;
      const int v0 = chunk * kChunkV + v_row_a;
      const int v1 = chunk * kChunkV + v_row_b;
      float h_a_vals[4];
      float h_b_vals[4];
      float dot_hk_a = 0.0f;
      float dot_hk_b = 0.0f;

      const float4 raw_h_a = *reinterpret_cast<const float4*>(
          &s_state[chunk & 1][v_row_a][k_base]);
      const float4 raw_h_b = *reinterpret_cast<const float4*>(
          &s_state[chunk & 1][v_row_b][k_base]);
      h_a_vals[0] = raw_h_a.x * r_decay[0];
      h_a_vals[1] = raw_h_a.y * r_decay[1];
      h_a_vals[2] = raw_h_a.z * r_decay[2];
      h_a_vals[3] = raw_h_a.w * r_decay[3];
      h_b_vals[0] = raw_h_b.x * r_decay[0];
      h_b_vals[1] = raw_h_b.y * r_decay[1];
      h_b_vals[2] = raw_h_b.z * r_decay[2];
      h_b_vals[3] = raw_h_b.w * r_decay[3];
      dot_hk_a = h_a_vals[0] * r_k[0] + h_a_vals[1] * r_k[1] +
                 h_a_vals[2] * r_k[2] + h_a_vals[3] * r_k[3];
      dot_hk_b = h_b_vals[0] * r_k[0] + h_b_vals[1] * r_k[1] +
                 h_b_vals[2] * r_k[2] + h_b_vals[3] * r_k[3];

      const Sum2 dot_hk = warp_reduce_sum_pair(dot_hk_a, dot_hk_b);
      const float v_new0 = (s_v[v0] - dot_hk.x) * s_beta;
      const float v_new1 = (s_v[v1] - dot_hk.y) * s_beta;

      float dot_hq_a = 0.0f;
      float dot_hq_b = 0.0f;
      const int state_idx_a = (i_hv * kDimV + v0) * kDimK + k_base;
      const int state_idx_b = (i_hv * kDimV + v1) * kDimK + k_base;
      const float h_a_0 = h_a_vals[0] + r_k[0] * v_new0;
      const float h_a_1 = h_a_vals[1] + r_k[1] * v_new0;
      const float h_a_2 = h_a_vals[2] + r_k[2] * v_new0;
      const float h_a_3 = h_a_vals[3] + r_k[3] * v_new0;
      const float h_b_0 = h_b_vals[0] + r_k[0] * v_new1;
      const float h_b_1 = h_b_vals[1] + r_k[1] * v_new1;
      const float h_b_2 = h_b_vals[2] + r_k[2] * v_new1;
      const float h_b_3 = h_b_vals[3] + r_k[3] * v_new1;
      if constexpr (kComputeOutputBeforeStore) {
        dot_hq_a =
            h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b =
            h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
        store_state_float4<kUseCacheGlobalStore>(
            state_for_slot + state_idx_a,
            make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(
            state_for_slot + state_idx_b,
            make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
      } else {
        store_state_float4<kUseCacheGlobalStore>(
            state_for_slot + state_idx_a,
            make_float4(h_a_0, h_a_1, h_a_2, h_a_3));
        store_state_float4<kUseCacheGlobalStore>(
            state_for_slot + state_idx_b,
            make_float4(h_b_0, h_b_1, h_b_2, h_b_3));
        dot_hq_a =
            h_a_0 * r_q[0] + h_a_1 * r_q[1] + h_a_2 * r_q[2] + h_a_3 * r_q[3];
        dot_hq_b =
            h_b_0 * r_q[0] + h_b_1 * r_q[1] + h_b_2 * r_q[2] + h_b_3 * r_q[3];
      }

      const Sum2 dot_hq = warp_reduce_sum_pair(dot_hq_a, dot_hq_b);
      if (lane == 0) {
        s_o[v0] = dot_hq.x;
        s_o[v1] = dot_hq.y;
        if constexpr (kApplyOnorm && kAccumulateOnormSumsq) {
          o_sumsq += dot_hq.x * dot_hq.x + dot_hq.y * dot_hq.y;
        }
      }
    }

    if constexpr (kPrefetchNextStateChunk) {
      if (chunk + 2 < kNumChunks) {
        cp_async_state_chunk(&s_state[0][0][0], state_for_slot, 0, i_hv,
                             hv_count, chunk + 2);
      }
    }
  }
  __syncthreads();

  if constexpr (kApplyOnorm) {
    if constexpr (kAccumulateOnormSumsq) {
      if (lane == 0) {
        s_reduce[warp] = o_sumsq;
      }
      __syncthreads();

      float total_sumsq = 0.0f;
      if (warp == 0) {
        total_sumsq = lane < kWarps ? s_reduce[lane] : 0.0f;
        total_sumsq = warp_reduce_sum(total_sumsq);
        if (lane == 0) {
          s_reduce[0] = total_sumsq;
        }
      }
      __syncthreads();

      if (tid < kDimV) {
        const int64_t out_idx = i_n * kLocalDim + i_hv * kDimV + tid;
        const float raw_o = s_o[tid];
        const float rstd =
            rsqrtf(s_reduce[0] / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          const int64_t gate_idx = i_n * strides.onorm_row + i_hv * kDimV + tid;
          gate = sigmoid_fast(bf16_load(onorm_g, gate_idx));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    } else {
      const float raw_o = tid < kDimV ? s_o[tid] : 0.0f;
      const float o_sq = raw_o * raw_o;
      float sumsq;
      if constexpr (kUseActiveOnormReduction || kUseActiveQkReduction) {
        sumsq = block_reduce_sum_active_for<kDimV / 32>(o_sq, s_reduce);
      } else {
        sumsq = block_reduce_sum(o_sq, s_reduce);
      }

      if (tid < kDimV) {
        const int64_t out_idx = i_n * kLocalDim + i_hv * kDimV + tid;
        const float rstd =
            rsqrtf(sumsq / static_cast<float>(kDimV) + onorm_eps);
        float gate;
        float weight;
        if constexpr (kPreloadOnormParams) {
          gate = pre_onorm_gate;
          weight = pre_onorm_weight;
        } else {
          const int64_t gate_idx = i_n * strides.onorm_row + i_hv * kDimV + tid;
          gate = sigmoid_fast(bf16_load(onorm_g, gate_idx));
          weight = onorm_weight[tid];
        }
        const float y = raw_o * rstd * weight * gate;
        out[out_idx] = bf16_store(y);
      }
    }
  } else {
    if (tid < kDimV) {
      const int64_t out_idx = i_n * kLocalDim + i_hv * kDimV + tid;
      out[out_idx] = bf16_store(s_o[tid]);
    }
  }
}

template <int kHeads, bool kApplyOnorm, bool kUpdateConvState,
          bool kUseLowerBound, bool kApplyBetaSigmoid>
void launch_kda_decode_many_heads_raw(
    const void* x_q, const void* x_k, const void* x_v, const void* w_q_t,
    const void* w_k_t, const void* w_v_t, const void* bias_q,
    const void* bias_k, const void* bias_v, void* cs_q, void* cs_k, void* cs_v,
    const float* a_log, const void* g, const float* dt_bias, const void* beta,
    const void* onorm_g, const float* onorm_weight,
    const int* ssm_state_indices, const int* cu_seqlens, float* state,
    void* out, int B, int H, int HV, float lower_bound, float scale,
    float onorm_eps, KdaDecodeStrides strides, hggcStream_t stream) {
  auto kernel = &kda_decode_fusion_many_heads_kernel<
      kApplyOnorm, true, kHeads, kHeads, true, false, false, false, false,
      false, true, true, true, kUpdateConvState, kUseLowerBound,
      kApplyBetaSigmoid>;
  // PPU1.5 port: the sm90 launch-with-attributes path and its programmatic
  // stream serialization attribute have no PPU equivalent, so use a plain grid
  // launch. The device-side intrinsics that paired with it were removed from
  // the kernel above.
  dim3 grid(B, kHeads);
  dim3 block(kThreads);
  kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __ppu_bfloat16*>(x_q),
      reinterpret_cast<const __ppu_bfloat16*>(x_k),
      reinterpret_cast<const __ppu_bfloat16*>(x_v),
      reinterpret_cast<const float*>(w_q_t),
      reinterpret_cast<const float*>(w_k_t),
      reinterpret_cast<const float*>(w_v_t),
      reinterpret_cast<const float*>(bias_q),
      reinterpret_cast<const float*>(bias_k),
      reinterpret_cast<const float*>(bias_v),
      reinterpret_cast<__ppu_bfloat16*>(cs_q),
      reinterpret_cast<__ppu_bfloat16*>(cs_k),
      reinterpret_cast<__ppu_bfloat16*>(cs_v), a_log,
      reinterpret_cast<const __ppu_bfloat16*>(g), dt_bias,
      reinterpret_cast<const __ppu_bfloat16*>(beta),
      reinterpret_cast<const __ppu_bfloat16*>(onorm_g),
      onorm_weight, ssm_state_indices, cu_seqlens, state,
      reinterpret_cast<__ppu_bfloat16*>(out), B, H, HV,
      lower_bound, scale, onorm_eps, strides);
}

template <bool kApplyOnorm, bool kUseLowerBound, bool kApplyBetaSigmoid>
void launch_kda_decode_many_heads_selected(
    const void* x_q, const void* x_k, const void* x_v, const void* w_q_t,
    const void* w_k_t, const void* w_v_t, const void* bias_q,
    const void* bias_k, const void* bias_v, void* cs_q, void* cs_k, void* cs_v,
    const float* a_log, const void* g, const float* dt_bias, const void* beta,
    const void* onorm_g, const float* onorm_weight,
    const int* ssm_state_indices, const int* cu_seqlens, float* state,
    void* out, int B, int H, int HV, bool update_conv_cache, float lower_bound,
    float scale, float onorm_eps, KdaDecodeStrides strides,
    hggcStream_t stream) {
#define LAUNCH_KDA_DECODE(NUM_HEADS)                                        \
  do {                                                                      \
    if (update_conv_cache) {                                                \
      launch_kda_decode_many_heads_raw<NUM_HEADS, kApplyOnorm, true,        \
                                       kUseLowerBound, kApplyBetaSigmoid>(  \
          x_q, x_k, x_v, w_q_t, w_k_t, w_v_t, bias_q, bias_k, bias_v, cs_q, \
          cs_k, cs_v, a_log, g, dt_bias, beta, onorm_g, onorm_weight,       \
          ssm_state_indices, cu_seqlens, state, out, B, H, HV, lower_bound, \
          scale, onorm_eps, strides, stream);                               \
    } else {                                                                \
      launch_kda_decode_many_heads_raw<NUM_HEADS, kApplyOnorm, false,       \
                                       kUseLowerBound, kApplyBetaSigmoid>(  \
          x_q, x_k, x_v, w_q_t, w_k_t, w_v_t, bias_q, bias_k, bias_v, cs_q, \
          cs_k, cs_v, a_log, g, dt_bias, beta, onorm_g, onorm_weight,       \
          ssm_state_indices, cu_seqlens, state, out, B, H, HV, lower_bound, \
          scale, onorm_eps, strides, stream);                               \
    }                                                                       \
  } while (false)

  switch (H) {
    case 4:
      LAUNCH_KDA_DECODE(4);
      break;
    case 6:
      LAUNCH_KDA_DECODE(6);
      break;
    case 8:
      LAUNCH_KDA_DECODE(8);
      break;
    case 12:
      LAUNCH_KDA_DECODE(12);
      break;
    case 16:
      LAUNCH_KDA_DECODE(16);
      break;
    case 24:
      LAUNCH_KDA_DECODE(24);
      break;
    case 48:
      LAUNCH_KDA_DECODE(48);
      break;
    case 64:
      LAUNCH_KDA_DECODE(64);
      break;
    case 96:
      LAUNCH_KDA_DECODE(96);
      break;
    default:
      throw std::runtime_error("Unsupported number of heads: " + std::to_string(H));
  }

#undef LAUNCH_KDA_DECODE
}

struct KdaDecodeLaunchParams {
  const void* x_q;
  const void* x_k;
  const void* x_v;
  const void* w_q_t;
  const void* w_k_t;
  const void* w_v_t;
  const void* bias_q;
  const void* bias_k;
  const void* bias_v;
  void* cs_q;
  void* cs_k;
  void* cs_v;
  const float* a_log;
  const void* g;
  const float* dt_bias;
  const void* beta;
  const void* onorm_g;
  const float* onorm_weight;
  const int* ssm_state_indices;
  const int* cu_seqlens;
  float* state;
  void* out;
  int B;
  int H;
  int HV;
  bool update_conv_cache;
  float lower_bound;
  float scale;
  float onorm_eps;
  KdaDecodeStrides strides;
  hggcStream_t stream;
};

template <bool kApplyOnorm, bool kUseLowerBound, bool kApplyBetaSigmoid>
void launch_kda_decode_selected_backend(const KdaDecodeLaunchParams& p) {
  launch_kda_decode_many_heads_selected<kApplyOnorm, kUseLowerBound,
                                        kApplyBetaSigmoid>(
      p.x_q, p.x_k, p.x_v, p.w_q_t, p.w_k_t, p.w_v_t, p.bias_q, p.bias_k,
      p.bias_v, p.cs_q, p.cs_k, p.cs_v, p.a_log, p.g, p.dt_bias, p.beta,
      p.onorm_g, p.onorm_weight, p.ssm_state_indices, p.cu_seqlens, p.state,
      p.out, p.B, p.H, p.HV, p.update_conv_cache, p.lower_bound, p.scale,
      p.onorm_eps, p.strides, p.stream);
}

template <bool kApplyOnorm, bool kUseLowerBound>
void dispatch_kda_decode_beta(const KdaDecodeLaunchParams& p,
                              bool apply_beta_sigmoid) {
  if (apply_beta_sigmoid) {
    launch_kda_decode_selected_backend<kApplyOnorm, kUseLowerBound, true>(p);
  } else {
    launch_kda_decode_selected_backend<kApplyOnorm, kUseLowerBound, false>(p);
  }
}

template <bool kApplyOnorm>
void dispatch_kda_decode_decay(const KdaDecodeLaunchParams& p,
                               bool use_lower_bound, bool apply_beta_sigmoid) {
  if (use_lower_bound) {
    dispatch_kda_decode_beta<kApplyOnorm, true>(p, apply_beta_sigmoid);
  } else {
    dispatch_kda_decode_beta<kApplyOnorm, false>(p, apply_beta_sigmoid);
  }
}

void dispatch_kda_decode_features(const KdaDecodeLaunchParams& p,
                                  bool apply_onorm, bool use_lower_bound,
                                  bool apply_beta_sigmoid) {
  if (apply_onorm) {
    dispatch_kda_decode_decay<true>(p, use_lower_bound, apply_beta_sigmoid);
  } else {
    dispatch_kda_decode_decay<false>(p, use_lower_bound, apply_beta_sigmoid);
  }
}

}  // namespace

extern "C" void launch_kda_decode_many_heads_cuda(
    const void* x_q, const void* x_k, const void* x_v, const void* w_q_t,
    const void* w_k_t, const void* w_v_t, const void* bias_q,
    const void* bias_k, const void* bias_v, void* cs_q, void* cs_k, void* cs_v,
    const float* a_log, const void* g, const float* dt_bias, const void* beta,
    const void* onorm_g, const float* onorm_weight,
    const int* ssm_state_indices, const int* cu_seqlens, float* state,
    void* out, int B, int H, int HV, bool apply_onorm, bool update_conv_cache,
    bool use_lower_bound, bool apply_beta_sigmoid, float lower_bound,
    float scale, float onorm_eps, const int64_t* raw_strides,
    hggcStream_t stream) {
  const KdaDecodeStrides strides{raw_strides[0], raw_strides[1], raw_strides[2],
                                 raw_strides[3], raw_strides[4]};
  const KdaDecodeLaunchParams params{x_q,
                                     x_k,
                                     x_v,
                                     w_q_t,
                                     w_k_t,
                                     w_v_t,
                                     bias_q,
                                     bias_k,
                                     bias_v,
                                     cs_q,
                                     cs_k,
                                     cs_v,
                                     a_log,
                                     g,
                                     dt_bias,
                                     beta,
                                     onorm_g,
                                     onorm_weight,
                                     ssm_state_indices,
                                     cu_seqlens,
                                     state,
                                     out,
                                     B,
                                     H,
                                     HV,
                                     update_conv_cache,
                                     lower_bound,
                                     scale,
                                     onorm_eps,
                                     strides,
                                     stream};
  dispatch_kda_decode_features(params, apply_onorm, use_lower_bound,
                               apply_beta_sigmoid);
}

// PPU1.5 port wrapper: exposes the SAME I/O contract as
// torch.ops._C.fused_kda_decode (vllm); the at::Tensor / TORCH_CHECK wrapper that
// registers it through the pla.decode pybind now lives in the _host.cpp beside this file
// module and driven from Python. Operates in place on conv_state / state / out.
