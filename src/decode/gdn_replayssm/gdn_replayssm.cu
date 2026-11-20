// SPDX-License-Identifier: Apache-2.0
#include "gdn_replayssm_abi.h"

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_mma.h>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "gdn_replayssm_sglang_metadata.cuh"

namespace pla::gdn_replayssm {
namespace {

using BF = __ppu_bfloat16;
using HF = __half;
constexpr int THREADS = 128;
constexpr int ROWS = THREADS / 4;

template <class TT>
__device__ __forceinline__ int64_t index(const TT& t, int64_t i) {
  i *= t.stride[0];
  if (t.dtype == I64) return static_cast<const int64_t*>(t.data)[i];
  if (t.dtype == I8) return static_cast<const int8_t*>(t.data)[i];
  return static_cast<const int32_t*>(t.data)[i];
}

// Metadata known on the host to be int32 is read directly; otherwise the dynamic path is used.
template <bool I32, class TT>
__device__ __forceinline__ int64_t meta_index(const TT& t, int64_t i) {
  if constexpr (I32) return static_cast<const int32_t*>(t.data)[i * t.stride[0]];
  else return index(t, i);
}

// Compact launch parameters: full ABI tensors carry six strides (64 bytes each).
struct LiteTensor {
  void* data;
  int64_t stride[2];
  DType dtype;
};

inline LiteTensor lite(const Tensor& t) { return {t.data, {t.stride[0], t.stride[1]}, t.dtype}; }

__device__ __forceinline__ int bounded_index(const Tensor& t, int64_t row, int low, int high) {
  const int64_t value = index(t, row);
  assert(value >= low && value <= high);
  return static_cast<int>(value);
}

__device__ __forceinline__ void put_index(const Tensor& t, int64_t i, int64_t x) {
  i *= t.stride[0];
  if (t.dtype == I64) static_cast<int64_t*>(t.data)[i] = x;
  else if (t.dtype == I8) static_cast<int8_t*>(t.data)[i] = x;
  else static_cast<int32_t*>(t.data)[i] = x;
}

template <class T, class TT>
__device__ __forceinline__ float read(const TT& t, int64_t i) {
  return static_cast<float>(static_cast<const T*>(t.data)[i]);
}

template <class T, class TT>
__device__ __forceinline__ void write(const TT& t, int64_t i, float x) {
  static_cast<T*>(t.data)[i] = static_cast<T>(x);
}

template <class TT>
__device__ __forceinline__ float read_any(const TT& t, int64_t i) {
  if (t.dtype == BF16) return read<BF>(t, i);
  if (t.dtype == F16) return read<HF>(t, i);
  return read<float>(t, i);
}

// Unconditional vector loads; callers guarantee alignment on the host (see aligned_rows).
__device__ __forceinline__ void load_bf16x8(const void* src, float (&out)[8]) {
  const uint4 raw = *static_cast<const uint4*>(src);
  const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void load_bf16x4(const void* src, float (&out)[4]) {
  const uint2 raw = *static_cast<const uint2*>(src);
  out[0] = __uint_as_float(raw.x << 16);
  out[1] = __uint_as_float(raw.x & 0xffff0000u);
  out[2] = __uint_as_float(raw.y << 16);
  out[3] = __uint_as_float(raw.y & 0xffff0000u);
}

// Reads N contiguous elements with one 8- or 16-byte load when aligned and complete.
template <class T, int N>
__device__ __forceinline__ void read_vector(const Tensor& t, int64_t offset, int valid, float (&out)[N]) {
  constexpr int BYTES = N * sizeof(T);
  static_assert(BYTES == 8 || BYTES == 16, "vector loads are 8 or 16 bytes");
  const T* src = static_cast<const T*>(t.data) + offset;
  if (valid == N && (reinterpret_cast<uintptr_t>(src) & (BYTES - 1)) == 0) {
    using Raw = typename std::conditional<BYTES == 16, uint4, uint2>::type;
    const Raw raw = *reinterpret_cast<const Raw*>(src);
    const T* elements = reinterpret_cast<const T*>(&raw);
    #pragma unroll
    for (int i = 0; i < N; ++i) out[i] = static_cast<float>(elements[i]);
  } else {
    #pragma unroll
    for (int i = 0; i < N; ++i) out[i] = i < valid ? static_cast<float>(src[i]) : 0.f;
  }
}

template <class T>
__device__ __forceinline__ float rounded(float x) {
  return static_cast<float>(static_cast<T>(x));
}

__device__ __forceinline__ float operand(float x, bool tf32) {
  if (!tf32) return x;
  unsigned bits = __float_as_uint(x);
  if ((bits & 0x7f800000u) == 0x7f800000u) return x;
  // tc02 TF32 operands truncate the mantissa, rather than rounding to nearest.
  return __uint_as_float(bits & 0xffffe000u);
}

__device__ __forceinline__ float sum_warp(float x) {
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  return x;
}

__device__ __forceinline__ float sum_row(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x;
}

__device__ __forceinline__ float norm_denominator(float square_sum, int policy, float epsilon) {
  return policy == 0 ? 1.f : policy == 1 ? sqrtf(square_sum + epsilon) : sqrtf(square_sum) + epsilon;
}

__device__ __forceinline__ float warp_inclusive_sum(float x, int lane) {
  #pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const float y = __shfl_up_sync(0xffffffffu, x, offset);
    if (lane >= offset) x += y;
  }
  return x;
}

// Whole-warp call: decay[t] = exp(sum(g[t+1:count])) for ring history t < count; returns exp(sum(g)).
template <class TT>
__device__ __forceinline__ float history_decay(const TT& g, int64_t head_offset, int base,
                                               int count, int capacity, float* decay, int lane) {
  float total = 0.f;
  for (int start = 0; start < count; start += 32) {
    const int t = count - 1 - start - lane;
    const int source_t = max(t, 0);
    const float loaded = read<float>(g, head_offset + ((base + source_t) & (capacity - 1)));
    const float prefix = warp_inclusive_sum(t >= 0 ? loaded : 0.f, lane) + total;
    const float previous = __shfl_up_sync(0xffffffffu, prefix, 1);
    if (t >= 0) decay[t] = expf(lane == 0 ? total : previous);
    total = __shfl_sync(0xffffffffu, prefix, 31);
  }
  return expf(total);
}

__device__ __forceinline__ float log_gate(const Verify& p, float a, int hv) {
  if (p.processed_gates) return a;
  const float x = a + read_any(p.dt_bias, hv * p.dt_bias.stride[0]);
  return -expf(read_any(p.A_log, hv * p.A_log.stride[0])) *
         (x <= 20.f ? log1pf(expf(x)) : x);
}

// Row metadata comes first so the dependent first loads share the leading constant lines.
struct VerifyLite {
  LiteTensor slots, replay, starts, flush, wp, base;
  LiteTensor q, k, v, a, b, A_log, dt_bias, state, d, keys, g, rawv, rawk, beta, out;
  float* workspace;
  int batch, heads, value_heads, capacity, max_steps, state_slots, replay_slots, tokens, null_slot;
  float scale, epsilon;
  int norm_policy;
  bool exact, residual, tf32, processed_gates;
};

inline VerifyLite verify_lite(const Verify& v) {
  return {lite(v.slots), lite(v.replay), lite(v.starts), lite(v.flush), lite(v.wp), lite(v.base),
          lite(v.q), lite(v.k), lite(v.v), lite(v.a), lite(v.b), lite(v.A_log), lite(v.dt_bias),
          lite(v.state), lite(v.d), lite(v.keys), lite(v.g), lite(v.rawv), lite(v.rawk),
          lite(v.beta), lite(v.out), v.workspace,
          v.batch, v.heads, v.value_heads, v.capacity, v.max_steps, v.state_slots, v.replay_slots,
          v.tokens, v.null_slot, v.scale, v.epsilon, v.norm_policy,
          v.exact, v.residual, v.tf32, v.processed_gates};
}

#include "gdn_replayssm_project.cuh"

template <class A, class S, class C, int STEPS, int STATIC_K = 0>
__global__ __launch_bounds__(THREADS) void verify_kernel(Verify p) {
  const int row = blockIdx.y, hv = blockIdx.z, h = hv / (p.value_heads / p.heads);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t slot = index(p.slots, row), replay = index(p.replay, row);
  const int64_t bos = p.starts.data ? index(p.starts, row) : row;
  const int64_t eos = p.starts.data ? index(p.starts, row + 1) : row + 1;
  assert(bos >= 0 && eos >= bos && eos <= p.tokens && eos - bos <= p.max_steps);
  const int n = eos - bos;
  const int vr = blockIdx.x * ROWS + tid / 4;
  if (slot <= p.null_slot) {
    if (p.mode != 2 && (tid & 3) == 0 && vr < p.value_dim)
      for (int t = 0; t < n; ++t)
        write<A>(p.out, (bos + t) * p.out.stride[0] + hv * p.value_dim + vr, 0.f);
    return;
  }
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const bool flush = p.flush.data && index(p.flush, replay) != 0;
  if ((p.mode == 1 && flush) || (p.mode == 2 && !flush)) return;
  const int wp = bounded_index(p.wp, replay, 0, p.capacity);
  const int base = bounded_index(p.base, replay, 0, p.capacity - 1);
  assert(wp >= 0 && wp + n <= p.capacity);
  if (n == 0) return;
  const int history = flush ? 0 : wp;
  const int K = STATIC_K ? STATIC_K : p.key_dim;
  constexpr bool fp32 = std::is_same<S, float>::value;
  constexpr bool cache32 = std::is_same<C, float>::value;
  constexpr bool activation32 = std::is_same<A, float>::value;
  const bool low = p.residual && !fp32;
  const bool kk_tf32 = p.tf32 && (fp32 || cache32);
  const bool kq_tf32 = p.tf32 && (fp32 || (cache32 && activation32));
  // The original 16-bit-state Triton path uses its default dot precision here.
  const bool state_k_tf32 = fp32 ? p.tf32 : ((p.tf32 || p.legacy_dot_precision) && cache32);
  const bool state_q_tf32 = fp32 ? p.tf32 : ((p.tf32 || p.legacy_dot_precision) && cache32 && activation32);
  extern __shared__ float sm[];
  float* qs = sm;
  float* ks = qs + STEPS * K;
  float* kl = ks + STEPS * K;
  float* kk = kl + STEPS * K;
  float* kq = kk + STEPS * STEPS;
  float* hist_k = kq + STEPS * STEPS;
  float* hist_q = hist_k + p.capacity * STEPS;
  float* decay = hist_q + p.capacity * STEPS;
  float* gates = decay + p.capacity;
  float* G = gates + STEPS;
  float* beta = G + STEPS;
  float* total_decay = beta + STEPS;

  constexpr int KV = STATIC_K ? STATIC_K / 32 : 8;
  for (int t = warp; t < n; t += THREADS / 32) {
    float qnorm = 0, knorm = 0;
    float qv[KV], kv[KV];
    #pragma unroll
    for (int j = 0; j < KV; ++j) {
      const int x = lane + 32 * j;
      qv[j] = kv[j] = 0.f;
      if (x < K) {
        qv[j] = read<A>(p.q, (bos + t) * p.q.stride[0] + h * K + x);
        kv[j] = read<A>(p.k, (bos + t) * p.k.stride[0] + h * K + x);
        qnorm += qv[j] * qv[j];
        knorm += kv[j] * kv[j];
      }
    }
    qnorm = sum_warp(qnorm);
    knorm = sum_warp(knorm);
    const float qscale = p.scale * (1.f / norm_denominator(qnorm, p.norm_policy, p.epsilon));
    const float kscale = 1.f / norm_denominator(knorm, p.norm_policy, p.epsilon);
    const int phys = (base + wp + t) & (p.capacity - 1);
    #pragma unroll
    for (int j = 0; j < KV; ++j) {
      const int x = lane + 32 * j;
      if (x >= K) continue;
      const float rawk = kv[j];
      const float precise_k = rawk * kscale;
      const float hi = rounded<C>(precise_k);
      const float lo = rounded<A>(precise_k - hi);
      const float q = qv[j] * qscale;
      qs[t * K + x] = fp32 ? operand(q, p.tf32) : rounded<A>(q);
      ks[t * K + x] = fp32 ? operand(precise_k, p.tf32) : hi;
      kl[t * K + x] = low ? lo : 0.f;
      if (blockIdx.x == 0 && hv % (p.value_heads / p.heads) == 0) {
        const int64_t off = (h * p.capacity + phys) * int64_t(K) + x;
        write<C>(p.keys, replay * p.keys.stride[0] + off, hi);
        if (p.exact || p.residual)
          write<C>(p.rawk, replay * p.rawk.stride[0] + off, p.exact ? rawk : lo);
      }
    }
    if (lane == 0) {
      const float a = read_any(p.a, (bos + t) * p.a.stride[0] + hv);
      const float b = read_any(p.b, (bos + t) * p.b.stride[0] + hv);
      gates[t] = log_gate(p, a, hv);
      beta[t] = p.processed_gates ? b : 1.f / (1.f + expf(-b));
    }
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) {
      float prefix = 0.f;
      for (int t = 0; t < n; ++t) { prefix += gates[t]; G[t] = prefix; }
    }
    const float exp_total = history_decay(p.g, replay * p.g.stride[0] + hv * p.capacity,
                                          base, history, p.capacity, decay, lane);
    if (lane == 0) *total_decay = exp_total;
  }
  for (int pair = warp; pair < n * n; pair += THREADS / 32) {
    const int i = pair / n, j = pair % n;
    float a = 0.f, b = 0.f;
    for (int x = lane; x < K; x += 32) {
      const float ki = ks[i * K + x] + (low ? kl[i * K + x] : 0.f);
      const float kj = ks[j * K + x] + (low ? kl[j * K + x] : 0.f);
      if constexpr (fp32) {
        a += ki * kj;
        b += ki * qs[j * K + x];
      } else {
        a += operand(ki, kk_tf32) * operand(kj, kk_tf32);
        b += operand(ki, kq_tf32) * operand(qs[j * K + x], kq_tf32);
      }
    }
    a = sum_warp(a); b = sum_warp(b);
    if (lane == 0) { kk[i * STEPS + j] = a; kq[i * STEPS + j] = b; }
  }
  for (int old = warp; old < history; old += THREADS / 32) {
    const int phys = (base + old) & (p.capacity - 1);
    float key[KV];
    #pragma unroll
    for (int j = 0; j < KV; ++j) {
      const int x = lane + 32 * j;
      key[j] = x < K ? read<C>(p.keys, replay * p.keys.stride[0] +
                                (h * p.capacity + phys) * int64_t(K) + x) : 0.f;
    }
    for (int t = 0; t < n; ++t) {
      float a = 0.f, b = 0.f;
      #pragma unroll
      for (int j = 0; j < KV; ++j) {
        const int x = lane + 32 * j;
        if (x >= K) continue;
        const float kh = key[j];
        if constexpr (fp32) {
          const float k_operand = operand(kh, p.tf32);
          a += k_operand * ks[t * K + x];
          b += k_operand * qs[t * K + x];
        } else {
          a += operand(kh, kk_tf32) * operand(ks[t * K + x], kk_tf32);
          b += operand(kh, kq_tf32) * operand(qs[t * K + x], kq_tf32);
        }
      }
      a = sum_warp(a); b = sum_warp(b);
      if (lane == 0) {
        hist_k[old * STEPS + t] = operand(fp32 ? a : rounded<C>(a), kk_tf32);
        hist_q[old * STEPS + t] = operand(fp32 ? b : rounded<C>(b), kk_tf32);
      }
    }
  }
  __syncthreads();
  float hp[STEPS] = {}, hq[STEPS] = {};
  const int group_lane = tid & 3;
  if (vr < p.value_dim) {
    const int64_t state_off = slot * p.state.stride[0] +
                             (hv * int64_t(p.value_dim) + vr) * K;
    for (int x0 = group_lane * 4; x0 < K; x0 += 16) {
      const S* src = static_cast<const S*>(p.state.data) + state_off + x0;
      float4 values;
      if constexpr (sizeof(S) == sizeof(float)) {
        if (x0 + 3 < K && (reinterpret_cast<uintptr_t>(src) & 15) == 0) {
          values = *reinterpret_cast<const float4*>(src);
        } else {
          values = make_float4(float(src[0]), x0 + 1 < K ? float(src[1]) : 0.f,
                               x0 + 2 < K ? float(src[2]) : 0.f, x0 + 3 < K ? float(src[3]) : 0.f);
        }
      } else {
        values = make_float4(float(src[0]), x0 + 1 < K ? float(src[1]) : 0.f,
                             x0 + 2 < K ? float(src[2]) : 0.f, x0 + 3 < K ? float(src[3]) : 0.f);
      }
      #pragma unroll
      for (int z = 0; z < 4; ++z) {
        if (x0 + z < K) {
          const float element = z == 0 ? values.x : z == 1 ? values.y : z == 2 ? values.z : values.w;
          const float value = fp32 ? operand(element, p.tf32) : rounded<C>(element);
          #pragma unroll
          for (int t = 0; t < STEPS; ++t) {
            if (t < n) {
              if constexpr (fp32) {
                hp[t] += value * ks[t * K + x0 + z];
                hq[t] += value * qs[t * K + x0 + z];
              } else {
                const float key = ks[t * K + x0 + z] + (low ? kl[t * K + x0 + z] : 0.f);
                hp[t] += operand(value, state_k_tf32) * operand(key, state_k_tf32);
                hq[t] += operand(value, state_q_tf32) * operand(qs[t * K + x0 + z], state_q_tf32);
              }
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int t = 0; t < STEPS; ++t) { hp[t] = sum_row(hp[t]); hq[t] = sum_row(hq[t]); }
  if (group_lane == 0 && vr < p.value_dim) {
    float d[STEPS] = {}, hk[STEPS] = {}, hq_old[STEPS] = {};
    // History loads precede every D write below, so they can be issued back to back.
    #pragma unroll 4
    for (int old = 0; old < history; ++old) {
      const int phys = (base + old) & (p.capacity - 1);
      float scaled = read<C>(p.d, replay * p.d.stride[0] +
                    (hv * p.capacity + phys) * int64_t(p.value_dim) + vr) * decay[old];
      scaled = operand(fp32 ? scaled : rounded<C>(scaled), kk_tf32);
      #pragma unroll
      for (int t = 0; t < STEPS; ++t) {
        if (t < n) {
          hk[t] += scaled * hist_k[old * STEPS + t];
          hq_old[t] += scaled * hist_q[old * STEPS + t];
        }
      }
    }
    #pragma unroll (STEPS <= 8 ? STEPS : 1)
    for (int t = 0; t < STEPS; ++t) {
      if (t >= n) continue;
      hp[t] = *total_decay * hp[t] + hk[t];
      hq[t] = *total_decay * hq[t] + hq_old[t];
      const float rawv = read<A>(p.v, (bos + t) * p.v.stride[0] + hv * p.value_dim + vr);
      float update = beta[t] * (rawv - expf(G[t]) * hp[t]);
      #pragma unroll (STEPS <= 8 ? STEPS : 1)
      for (int j = 0; j < STEPS; ++j)
        if (j < t) update -= beta[t] * expf(G[t] - G[j]) * kk[t * STEPS + j] * d[j];
      d[t] = update;
      float output = expf(G[t]) * hq[t];
      #pragma unroll (STEPS <= 8 ? STEPS : 1)
      for (int j = 0; j < STEPS; ++j)
        if (j <= t) output += d[j] * expf(G[t] - G[j]) * kq[j * STEPS + t];
      write<A>(p.out, (bos + t) * p.out.stride[0] + hv * p.value_dim + vr, output);
      const int phys = (base + wp + t) & (p.capacity - 1);
      const int64_t off = (hv * p.capacity + phys) * int64_t(p.value_dim) + vr;
      const float hi = rounded<C>(update);
      write<C>(p.d, replay * p.d.stride[0] + off, hi);
      if (p.exact || p.residual)
        write<C>(p.rawv, replay * p.rawv.stride[0] + off, p.exact ? rawv : update - hi);
    }
  }
  if (blockIdx.x == 0 && tid < n) {
    const int phys = (base + wp + tid) & (p.capacity - 1);
    write<float>(p.g, replay * p.g.stride[0] + hv * p.capacity + phys, gates[tid]);
    if (p.exact)
      write<float>(p.beta, replay * p.beta.stride[0] + hv * p.capacity + phys, beta[tid]);
  }
}

// verify_kernel<BF16, F32, BF16, STEPS, 128> for V=128 with three dependent global-load levels:
// row metadata, then per-row cursors/state/inputs, then the whole history ring. Loads are
// unconditional (clamped rows, wrapped ring slots) because a load inside a branch must complete
// before the branch merges. Per-lane accumulation orders match verify_kernel except the Q/K norm.
template <bool BF_RAW, class TT>
__device__ __forceinline__ float input_gate(const TT& t, int64_t offset) {
  if constexpr (BF_RAW) return read<BF>(t, offset);
  else return read_any(t, offset);
}

template <int STEPS, int MAXC, bool I32, int VROWS = 32, bool BF_RAW = false, int VGROUP = 4,
          bool STATIC_TF32 = false>
__global__ __launch_bounds__(VROWS * VGROUP) void small_verify_kernel(VerifyLite p) {
  constexpr int THREADS = VROWS * VGROUP, ROWS = VROWS, WARPS = THREADS / 32;
  const bool tf32 = STATIC_TF32 || p.tf32;
  constexpr int K = 128, V = 128, TOKENS_PER_WARP = (STEPS + WARPS - 1) / WARPS;
  const int row = blockIdx.y, hv = blockIdx.z, h = hv / (p.value_heads / p.heads);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, group_lane = tid & (VGROUP - 1);
  const int local_v = tid / VGROUP, vr = blockIdx.x * ROWS + local_v;

  const int64_t slot = meta_index<I32>(p.slots, row), replay = meta_index<I32>(p.replay, row);
  const int64_t bos = p.starts.data ? meta_index<I32>(p.starts, row) : row;
  const int64_t eos = p.starts.data ? meta_index<I32>(p.starts, row + 1) : row + 1;
  assert(bos >= 0 && eos >= bos && eos <= p.tokens && eos - bos <= p.max_steps);
  const int n = static_cast<int>(eos - bos);
  if (n == 0) return;
  if (slot <= p.null_slot) {
    if (group_lane == 0)
      for (int t = 0; t < n; ++t)
        write<BF>(p.out, (bos + t) * p.out.stride[0] + hv * V + vr, 0.f);
    return;
  }
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t safe_slot = slot >= 0 && slot < p.state_slots ? slot : 0;
  const int64_t safe_replay = replay >= 0 && replay < p.replay_slots ? replay : 0;
  const bool flush = p.flush.data && index(p.flush, safe_replay) != 0;
  const int64_t wp_raw = meta_index<I32>(p.wp, safe_replay);
  const int64_t base_raw = meta_index<I32>(p.base, safe_replay);
  auto token_row = [&](int t) { return max(int64_t(0), min(bos + t, int64_t(p.tokens) - 1)); };
  const float* state_row = static_cast<const float*>(p.state.data) + safe_slot * p.state.stride[0] +
                           (hv * int64_t(V) + vr) * K + group_lane * 4;
  float4 state_values[K / (VGROUP * 4)];
  #pragma unroll
  for (int i = 0; i < K / (VGROUP * 4); ++i)
    state_values[i] = *reinterpret_cast<const float4*>(state_row + VGROUP * 4 * i);
  float raw_v[STEPS];
  #pragma unroll
  for (int t = 0; t < STEPS; ++t)
    raw_v[t] = read<BF>(p.v, token_row(t) * p.v.stride[0] + hv * V + vr);
  // Each lane owns four contiguous Q/K features of a token.
  float qv[TOKENS_PER_WARP][4], kv[TOKENS_PER_WARP][4], av[TOKENS_PER_WARP], bv[TOKENS_PER_WARP];
  #pragma unroll
  for (int i = 0; i < TOKENS_PER_WARP; ++i) {
    const int64_t r = token_row(warp + WARPS * i);
    load_bf16x4(static_cast<const BF*>(p.q.data) + r * p.q.stride[0] + h * K + lane * 4, qv[i]);
    load_bf16x4(static_cast<const BF*>(p.k.data) + r * p.k.stride[0] + h * K + lane * 4, kv[i]);
    av[i] = input_gate<BF_RAW>(p.a, r * p.a.stride[0] + hv);
    bv[i] = input_gate<BF_RAW>(p.b, r * p.b.stride[0] + hv);
  }
  const bool processed = BF_RAW ? false : p.processed_gates;
  float dt_bias = 0.f, A_log = 0.f;
  if constexpr (BF_RAW) {
    dt_bias = read<float>(p.dt_bias, hv * p.dt_bias.stride[0]);
    A_log = read<float>(p.A_log, hv * p.A_log.stride[0]);
  } else if (!processed) {
    dt_bias = read_any(p.dt_bias, hv * p.dt_bias.stride[0]);
    A_log = read_any(p.A_log, hv * p.A_log.stride[0]);
  }

  if (flush) return;
  assert(wp_raw >= 0 && wp_raw <= p.capacity && base_raw >= 0 && base_raw < p.capacity);
  const int wp = static_cast<int>(wp_raw), base = static_cast<int>(base_raw);
  assert(wp + n <= p.capacity);
  if (n == 0) return;
  const int history = wp;

  __shared__ float qs[STEPS * K], ks[STEPS * K], kk[STEPS * STEPS], kq[STEPS * STEPS];
  __shared__ float gates[STEPS], G[STEPS], beta[STEPS], total_decay;
  extern __shared__ float small_verify_mem[];
  float* hist_k = small_verify_mem;
  float* hist_q = hist_k + p.capacity * STEPS;
  float* decay = hist_q + p.capacity * STEPS;
  float* hist_key = decay + p.capacity;
  float* hist_d = hist_key + p.capacity * K;

  // 16-byte history loads: 16 threads per key row and 4 threads per 32-value D row.
  constexpr int KEY_ROWS = THREADS / 16, KEY_LOADS = (MAXC + KEY_ROWS - 1) / KEY_ROWS;
  constexpr int D_ROWS_PER_LOAD = THREADS / (ROWS / 8), D_LOADS = (MAXC + D_ROWS_PER_LOAD - 1) / D_ROWS_PER_LOAD;
  const int key_row = tid / 16, key_col = (tid % 16) * 8;
  const int d_row = tid / (ROWS / 8), d_col = (tid % (ROWS / 8)) * 8;
  float keys[KEY_LOADS][8], updates[D_LOADS][8];
  #pragma unroll
  for (int i = 0; i < KEY_LOADS; ++i) {
    const int old = min(i * KEY_ROWS + key_row, max(history - 1, 0));
    const int phys = (base + old) & (p.capacity - 1);
    const BF* src = history ? static_cast<const BF*>(p.keys.data) + replay * p.keys.stride[0] +
                    (h * p.capacity + phys) * int64_t(K) + key_col :
                    static_cast<const BF*>(p.k.data) + bos * p.k.stride[0] + h * K + key_col;
    load_bf16x8(src, keys[i]);
  }
  #pragma unroll
  for (int i = 0; i < D_LOADS; ++i) {
    const int phys = (base + i * D_ROWS_PER_LOAD + d_row) & (p.capacity - 1);
    load_bf16x8(static_cast<const BF*>(p.d.data) + replay * p.d.stride[0] +
                (hv * p.capacity + phys) * int64_t(V) + blockIdx.x * ROWS + d_col, updates[i]);
  }
  if (warp == 0) {
    const float exp_total = history_decay(p.g, replay * p.g.stride[0] + hv * p.capacity,
                                          base, history, p.capacity, decay, lane);
    if (lane == 0) total_decay = exp_total;
  }
  #pragma unroll
  for (int i = 0; i < TOKENS_PER_WARP; ++i) {
    const int t = warp + WARPS * i;
    if (t >= n) continue;
    float qnorm = 0.f, knorm = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      qnorm += qv[i][j] * qv[i][j];
      knorm += kv[i][j] * kv[i][j];
    }
    qnorm = sum_warp(qnorm);
    knorm = sum_warp(knorm);
    const float qscale = p.scale * (1.f / norm_denominator(qnorm, p.norm_policy, p.epsilon));
    const float kscale = 1.f / norm_denominator(knorm, p.norm_policy, p.epsilon);
    const int phys = (base + wp + t) & (p.capacity - 1);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int x = lane * 4 + j;
      const float precise_k = kv[i][j] * kscale;
      const float hi = rounded<BF>(precise_k);
      qs[t * K + x] = operand(qv[i][j] * qscale, tf32);
      ks[t * K + x] = operand(precise_k, tf32);
      if (blockIdx.x == 0 && hv % (p.value_heads / p.heads) == 0) {
        const int64_t off = (h * p.capacity + phys) * int64_t(K) + x;
        write<BF>(p.keys, replay * p.keys.stride[0] + off, hi);
        if (p.exact || p.residual)
          write<BF>(p.rawk, replay * p.rawk.stride[0] + off,
                    p.exact ? kv[i][j] : rounded<BF>(precise_k - hi));
      }
    }
    if (lane == 0) {
      const float x = av[i] + dt_bias;
      gates[t] = processed ? av[i] : -expf(A_log) * (x <= 20.f ? log1pf(expf(x)) : x);
      beta[t] = processed ? bv[i] : 1.f / (1.f + expf(-bv[i]));
    }
  }
  #pragma unroll
  for (int i = 0; i < KEY_LOADS; ++i) {
    const int t = i * KEY_ROWS + key_row;
    if (t < history) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) hist_key[t * K + key_col + e] = keys[i][e];
    }
  }
  #pragma unroll
  for (int i = 0; i < D_LOADS; ++i) {
    const int t = i * D_ROWS_PER_LOAD + d_row;
    if (t < history) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) hist_d[t * ROWS + d_col + e] = updates[i][e];
    }
  }
  __syncthreads();

  // Token/token and history/token dot products: one 4-lane group per pair, K split across the
  // group with a per-group rotation that keeps the eight groups of a warp on distinct banks.
  {
    const int pair_lane = tid & 3, group = tid / 4, rotation = group % 8;
    const int pairs = n * n + history * n;
    // Uniform trip count: every lane reaches the shuffles in sum_row.
    for (int first_pair = 0; first_pair < pairs; first_pair += THREADS / 4) {
      const bool valid = first_pair + group < pairs;
      const int pair = valid ? first_pair + group : 0;
      const bool token_pair = pair < n * n;
      const int first = token_pair ? pair / n : (pair - n * n) / n;
      const int second = token_pair ? pair % n : (pair - n * n) % n;
      const float* left = token_pair ? ks + first * K : hist_key + first * K;
      float a = 0.f, b = 0.f;
      if (valid) {
        #pragma unroll 8
        for (int m = 0; m < K / 4; ++m) {
          const int x = 4 * ((m + rotation) % (K / 4)) + pair_lane;
          const float l = token_pair ? left[x] : operand(left[x], tf32);
          a += l * ks[second * K + x];
          b += l * qs[second * K + x];
        }
      }
      a = sum_row(a);
      b = sum_row(b);
      if (valid && pair_lane == 0) {
        if (token_pair) {
          float gi = 0.f, gj = 0.f;
          for (int t = 0; t <= max(first, second); ++t) {
            if (t <= first) gi += gates[t];
            if (t <= second) gj += gates[t];
          }
          // kk carries beta and the decay exactly as (beta * e) * kk; kq carries its decay.
          kk[first * STEPS + second] = second < first ? (beta[first] * expf(gi - gj)) * a : 0.f;
          kq[first * STEPS + second] = first <= second ? expf(gj - gi) * b : 0.f;
        } else {
          hist_k[first * STEPS + second] = operand(a, tf32);
          hist_q[first * STEPS + second] = operand(b, tf32);
        }
      }
    }
    if (tid < n) {
      float prefix = 0.f;
      for (int t = 0; t <= tid; ++t) prefix += gates[t];
      G[tid] = expf(prefix);
    }
  }
  float hp[STEPS] = {}, hq[STEPS] = {};
  #pragma unroll
  for (int i = 0; i < K / (VGROUP * 4); ++i) {
    const int x0 = group_lane * 4 + VGROUP * 4 * i;
    const float4 s = state_values[i];
    const float s0 = operand(s.x, tf32), s1 = operand(s.y, tf32);
    const float s2 = operand(s.z, tf32), s3 = operand(s.w, tf32);
    #pragma unroll
    for (int t = 0; t < STEPS; ++t) {
      if (t < n) {
        const float4 kx = *reinterpret_cast<const float4*>(ks + t * K + x0);
        const float4 qx = *reinterpret_cast<const float4*>(qs + t * K + x0);
        hp[t] += s0 * kx.x; hp[t] += s1 * kx.y; hp[t] += s2 * kx.z; hp[t] += s3 * kx.w;
        hq[t] += s0 * qx.x; hq[t] += s1 * qx.y; hq[t] += s2 * qx.z; hq[t] += s3 * qx.w;
      }
    }
  }
  #pragma unroll
  for (int t = 0; t < STEPS; ++t) {
    if constexpr (VGROUP == 8) {
      hp[t] += __shfl_xor_sync(0xffffffffu, hp[t], 4);
      hq[t] += __shfl_xor_sync(0xffffffffu, hq[t], 4);
    }
    hp[t] = sum_row(hp[t]);
    hq[t] = sum_row(hq[t]);
  }
  __syncthreads();

  {
    // G now holds exp(G); every lane of a row group evaluates the row, lane 0 stores it.
    float d[STEPS] = {}, hk[STEPS] = {}, hq_old[STEPS] = {};
    for (int old = group_lane; old < history; old += VGROUP) {
      const float scaled = operand(hist_d[old * ROWS + local_v] * decay[old], tf32);
      #pragma unroll
      for (int t = 0; t < STEPS; ++t) {
        if (t < n) {
          hk[t] += scaled * hist_k[old * STEPS + t];
          hq_old[t] += scaled * hist_q[old * STEPS + t];
        }
      }
    }
    #pragma unroll
    for (int t = 0; t < STEPS; ++t) {
      if constexpr (VGROUP == 8) {
        hk[t] += __shfl_xor_sync(0xffffffffu, hk[t], 4);
        hq_old[t] += __shfl_xor_sync(0xffffffffu, hq_old[t], 4);
      }
      hk[t] = sum_row(hk[t]);
      hq_old[t] = sum_row(hq_old[t]);
    }
    #pragma unroll
    for (int t = 0; t < STEPS; ++t) {
      if (t >= n) continue;
      hp[t] = total_decay * hp[t] + hk[t];
      hq[t] = total_decay * hq[t] + hq_old[t];
      float update = beta[t] * (raw_v[t] - G[t] * hp[t]);
      #pragma unroll
      for (int j = 0; j < STEPS; ++j)
        if (j < t) update -= kk[t * STEPS + j] * d[j];
      d[t] = update;
      float output = G[t] * hq[t];
      #pragma unroll
      for (int j = 0; j < STEPS; ++j)
        if (j <= t) output += d[j] * kq[j * STEPS + t];
      if (group_lane == 0) {
        write<BF>(p.out, (bos + t) * p.out.stride[0] + hv * V + vr, output);
        const int phys = (base + wp + t) & (p.capacity - 1);
        const int64_t off = (hv * p.capacity + phys) * int64_t(V) + vr;
        const float hi = rounded<BF>(update);
        write<BF>(p.d, replay * p.d.stride[0] + off, hi);
        if (p.exact || p.residual)
          write<BF>(p.rawv, replay * p.rawv.stride[0] + off, p.exact ? raw_v[t] : update - hi);
      }
    }
  }
  if (blockIdx.x == 0 && tid < n) {
    const int phys = (base + wp + tid) & (p.capacity - 1);
    write<float>(p.g, replay * p.g.stride[0] + hv * p.capacity + phys, gates[tid]);
    if (p.exact)
      write<float>(p.beta, replay * p.beta.stride[0] + hv * p.capacity + phys, beta[tid]);
  }
}

template <int STEPS>
void small_verify_steps(const Verify& v, hggcStream_t stream) {
  const VerifyLite p = verify_lite(v);
  const size_t shared = sizeof(float) *
      (2 * size_t(p.capacity) * STEPS + p.capacity + size_t(p.capacity) * (128 + ROWS));
  const dim3 grid(128 / ROWS, v.batch, v.value_heads);
  const bool i32 = p.slots.dtype == I32 && p.replay.dtype == I32 && p.wp.dtype == I32 &&
                   p.base.dtype == I32 && (!p.starts.data || p.starts.dtype == I32);
  const bool bf_raw = !p.processed_gates && p.a.dtype == BF16 && p.b.dtype == BF16 &&
                      p.A_log.dtype == F32 && p.dt_bias.dtype == F32;
  const int64_t head_work = int64_t(v.batch) * v.value_heads;
  if constexpr (STEPS == 2) {
    if (v.legacy_dot_precision && i32 && bf_raw && p.tf32 && !p.exact && p.flush.dtype == I8 &&
        p.capacity == 16 && v.batch <= 8 && head_work <= 32) {
      small_verify_kernel<2, 16, true, 32, true, 4, true><<<grid, THREADS, shared, stream>>>(p);
      return;
    }
  }
  if constexpr (STEPS == 4 || STEPS == 8) {
    if (i32 && bf_raw && p.capacity <= 16 && head_work < 32) {
      if constexpr (STEPS == 4) {
        if (p.tf32) {
          small_verify_kernel<STEPS, 16, true, 32, true, 8, true>
              <<<grid, 256, shared, stream>>>(p);
          return;
        }
      }
      small_verify_kernel<STEPS, 16, true, 32, true, 8>
          <<<grid, 256, shared, stream>>>(p);
      return;
    }
  }
  if constexpr (STEPS == 4) {
    if (head_work >= 32) {
      const size_t wide_shared = sizeof(float) *
          (2 * size_t(p.capacity) * STEPS + p.capacity + size_t(p.capacity) * (128 + 64));
      const dim3 wide_grid(2, v.batch, v.value_heads);
      if (p.capacity <= 16) {
        const bool static_tf32 = head_work < 128 ||
            (v.legacy_dot_precision && v.batch <= 32 && p.capacity == 16 && head_work == 128);
        if (i32 && bf_raw && p.tf32 && static_tf32)
          small_verify_kernel<STEPS, 16, true, 64, true, 4, true><<<wide_grid, 256, wide_shared, stream>>>(p);
        else if (i32 && bf_raw)
          small_verify_kernel<STEPS, 16, true, 64, true><<<wide_grid, 256, wide_shared, stream>>>(p);
        else if (i32) small_verify_kernel<STEPS, 16, true, 64><<<wide_grid, 256, wide_shared, stream>>>(p);
        else small_verify_kernel<STEPS, 16, false, 64><<<wide_grid, 256, wide_shared, stream>>>(p);
      } else {
        if (i32) small_verify_kernel<STEPS, 32, true, 64><<<wide_grid, 256, wide_shared, stream>>>(p);
        else small_verify_kernel<STEPS, 32, false, 64><<<wide_grid, 256, wide_shared, stream>>>(p);
      }
      return;
    }
    if (i32 && bf_raw && p.capacity <= 16) {
      small_verify_kernel<STEPS, 16, true, 32, true><<<grid, THREADS, shared, stream>>>(p);
      return;
    }
  }
  if (p.capacity <= 16) {
    if (i32) small_verify_kernel<STEPS, 16, true><<<grid, THREADS, shared, stream>>>(p);
    else small_verify_kernel<STEPS, 16, false><<<grid, THREADS, shared, stream>>>(p);
  } else {
    if (i32) small_verify_kernel<STEPS, 32, true><<<grid, THREADS, shared, stream>>>(p);
    else small_verify_kernel<STEPS, 32, false><<<grid, THREADS, shared, stream>>>(p);
  }
}

template <class S, class C>
__global__ __launch_bounds__(THREADS) void exact_fold_kernel(Verify p) {
  const int row = blockIdx.y, hv = blockIdx.z, h = hv / (p.value_heads / p.heads);
  const int64_t slot = index(p.slots, row), replay = index(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  if (!index(p.flush, replay)) return;
  const int n = bounded_index(p.wp, replay, 0, p.capacity), K = p.key_dim;
  const int base = bounded_index(p.base, replay, 0, p.capacity - 1);
  assert(n >= 0 && n <= p.capacity);
  if (n == 0) return;
  const int tid = threadIdx.x, vr = blockIdx.x * ROWS + tid / 4, lane = tid & 3;
  float state[64] = {};
  const int64_t offset = slot * p.state.stride[0] + (hv * int64_t(p.value_dim) + vr) * K;
  if (vr < p.value_dim)
    for (int x = lane, i = 0; x < K; x += 4, ++i) state[i] = read<S>(p.state, offset + x);
  extern __shared__ float mem[];
  float* keys = mem;
  float* vals = keys + K;
  float* scalars = vals + ROWS;
  for (int t = 0; t < n; ++t) {
    const int phys = (base + t) & (p.capacity - 1);
    for (int x = tid; x < K; x += THREADS)
      keys[x] = read<C>(p.rawk, replay * p.rawk.stride[0] + (h * p.capacity + phys) * int64_t(K) + x);
    if (tid < ROWS)
      vals[tid] = blockIdx.x * ROWS + tid < p.value_dim ?
        read<C>(p.rawv, replay * p.rawv.stride[0] +
                (hv * p.capacity + phys) * int64_t(p.value_dim) + blockIdx.x * ROWS + tid) : 0.f;
    if (tid == 0) {
      scalars[0] = expf(read<float>(p.g, replay * p.g.stride[0] + hv * p.capacity + phys));
      scalars[1] = read<float>(p.beta, replay * p.beta.stride[0] + hv * p.capacity + phys);
    }
    __syncthreads();
    if (tid < 32) {
      float norm = 0.f;
      for (int x = tid; x < K; x += 32) norm += keys[x] * keys[x];
      norm = sum_warp(norm);
      if (tid == 0) scalars[2] = norm_denominator(norm, p.norm_policy, p.epsilon);
    }
    __syncthreads();
    float proj = 0.f;
    for (int x = lane, i = 0; x < K; x += 4, ++i) {
      state[i] *= scalars[0];
      proj += state[i] * (keys[x] / scalars[2]);
    }
    proj = sum_row(proj);
    const float update = (vals[tid / 4] - proj) * scalars[1];
    for (int x = lane, i = 0; x < K; x += 4, ++i)
      state[i] += keys[x] / scalars[2] * update;
    __syncthreads();
  }
  if (vr < p.value_dim)
    for (int x = lane, i = 0; x < K; x += 4, ++i) write<S>(p.state, offset + x, state[i]);
}

template <class TT>
__device__ __forceinline__ int64_t ring_offset(const TT& t, int layer, int64_t replay,
                                               int head, int phys, int dim, int x, int cap) {
  return layer * t.stride[0] + replay * t.stride[1] +
         (head * cap + phys) * int64_t(dim) + x;
}

#include "gdn_replayssm_commit.cuh"

template <class S, class C>
__global__ __launch_bounds__(THREADS) void compact_commit_kernel(Commit p) {
  const int row = blockIdx.y, layer = blockIdx.z / p.value_heads;
  const int hv = blockIdx.z % p.value_heads, h = hv / (p.value_heads / p.heads);
  const int64_t slot = index(p.slots, row), replay = index(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int n = bounded_index(p.wp, replay, 0, p.capacity);
  const int base = bounded_index(p.base, replay, 0, p.capacity - 1);
  const bool fold = index(p.flush, replay) != 0;
  const int64_t target = p.tracking ? index(p.track, row) : p.null_slot;
  const int64_t step = p.tracking ? index(p.steps, row) : -1;
  const bool track = target > p.null_slot && step >= 0;
  if (!fold && !track) return;
  assert(!track || step < p.capacity);
  const int accepted = track ? bounded_index(p.accepted, row, 0, p.capacity) : 0;
  const int64_t track_length = track ? n - accepted + step + 1 : 0;
  assert(track_length >= 0 && track_length <= p.capacity);
  const int nt = static_cast<int>(track_length);
  assert(!track || (target >= 0 && target < p.state_slots));
  const int count = max(fold ? n : 0, nt);
  const int K = p.key_dim, tid = threadIdx.x;
  const int vbase = blockIdx.x * 8;
  extern __shared__ float mem[];
  float* keys = mem;
  float* updates = keys + count * K;
  float* dg = updates + count * 8;
  float* dt = dg + count;
  float* totals = dt + count;
  for (int x = tid; x < count * K; x += THREADS) {
    const int t = x / K, k = x % K, phys = (base + t) & (p.capacity - 1);
    float key = read<C>(p.keys, ring_offset(p.keys, layer, replay, h, phys, K, k, p.capacity));
    if (p.residual) key += read<C>(p.kr, ring_offset(p.kr, layer, replay, h, phys, K, k, p.capacity));
    keys[x] = key;
  }
  for (int x = tid; x < count * 8; x += THREADS) {
    const int t = x / 8, v = vbase + x % 8, phys = (base + t) & (p.capacity - 1);
    float update = 0.f;
    if (v < p.value_dim) {
      update = read<C>(p.d, ring_offset(p.d, layer, replay, hv, phys, p.value_dim, v, p.capacity));
      if (p.residual)
        update += read<C>(p.dr, ring_offset(p.dr, layer, replay, hv, phys, p.value_dim, v, p.capacity));
    }
    updates[x] = update;
  }
  if (tid == 0) {
    float total = 0.f, tracked = 0.f;
    for (int t = 0; t < count; ++t) {
      const int phys = (base + t) & (p.capacity - 1);
      const float g = read<float>(p.g, layer * p.g.stride[0] + replay * p.g.stride[1] + hv * p.capacity + phys);
      if (t < n) total += g;
      if (t < nt) tracked += g;
      dg[t] = total; dt[t] = tracked;
    }
    for (int t = 0; t < count; ++t) {
      dg[t] = t < n ? expf(total - dg[t]) : 0.f;
      dt[t] = t < nt ? expf(tracked - dt[t]) : 0.f;
    }
    totals[0] = expf(total); totals[1] = expf(tracked);
  }
  __syncthreads();
  for (int x = tid; x < 8 * K; x += THREADS) {
    const int v = vbase + x / K, k = x % K;
    if (v >= p.value_dim) continue;
    const int64_t offset = layer * p.state.stride[0] + slot * p.state.stride[1] +
                           (hv * int64_t(p.value_dim) + v) * K + k;
    const float old = read<S>(p.state, offset);
    float df = 0.f, ds = 0.f;
    for (int t = 0; t < count; ++t) {
      const float key = keys[t * K + k];
      const float update = updates[t * 8 + v - vbase];
      if (fold && t < n) {
        const float scaled = update * dg[t];
        const float hi = rounded<C>(scaled), lo = rounded<C>(scaled - hi);
        df += key * hi;
        df += key * lo;
      }
      if (track && t < nt) {
        const float scaled = update * dt[t];
        const float hi = rounded<C>(scaled), lo = rounded<C>(scaled - hi);
        ds += key * hi;
        ds += key * lo;
      }
    }
    if (fold) write<S>(p.state, offset, old * totals[0] + df);
    if (track) write<S>(p.state, offset + (target - slot) * p.state.stride[1], old * totals[1] + ds);
  }
}

__global__ void finish_kernel(Commit p) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= p.batch) return;
  const int64_t slot = index(p.slots, row), replay = index(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  if (!index(p.flush, replay)) return;
  const int n = bounded_index(p.wp, replay, 0, p.capacity);
  const int base = bounded_index(p.base, replay, 0, p.capacity - 1);
  put_index(p.base, replay, (base + n) & (p.capacity - 1));
  put_index(p.wp, replay, 0);
  put_index(p.flush, replay, 0);
}

__global__ void cursor_kernel(Cursors p) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= p.batch) return;
  const int64_t replay = index(p.replay, row);
  if (replay <= p.null_slot) return;
  assert(replay >= 0 && replay < p.replay_slots);
  const int wp = bounded_index(p.wp, replay, 0, p.buffer_len);
  const int accepted = bounded_index(p.accepted, row, 0, p.max_steps);
  const int base = bounded_index(p.base, replay, 0, p.buffer_len - 1);
  const bool flush = index(p.flush, replay) != 0;
  assert(accepted >= 0 && accepted <= p.max_steps);
  const int next = flush ? accepted : wp + accepted;
  assert(next >= 0 && next <= p.buffer_len);
  put_index(p.base, replay, flush && accepted > 0 ? (base + wp) & (p.buffer_len - 1) : base);
  put_index(p.wp, replay, next);
  put_index(p.flush, replay, p.fold_every_commit ? next > 0 : next + 2 * p.max_steps > p.capacity);
}

template <class A, class S, class C, int STEPS, int STATIC_K = 0>
void verify_steps(const Verify& p, hggcStream_t stream) {
  const size_t shared = sizeof(float) * (3 * STEPS * p.key_dim + 2 * STEPS * STEPS +
                           2 * int64_t(p.capacity) * STEPS + p.capacity + 3 * STEPS + 1);
  if (shared > 48 * 1024) throw std::invalid_argument("GDN replay verify exceeds 48KiB shared memory");
  verify_kernel<A, S, C, STEPS, STATIC_K><<<dim3((p.value_dim + ROWS - 1) / ROWS, p.batch, p.value_heads), THREADS, shared, stream>>>(p);
}

template <class A, class S, class C>
void verify_typed(const Verify& p, hggcStream_t stream) {
  int steps = 1;
  while (steps < p.max_steps) steps *= 2;
  const size_t shared_bytes = sizeof(float) * (3 * steps * p.key_dim + 2 * steps * steps +
                             2 * int64_t(p.capacity) * steps + p.capacity + 3 * steps + 1);
  if (shared_bytes > 48 * 1024)
    throw std::invalid_argument("GDN replay verify exceeds 48KiB shared memory");
  if (p.mode != 1) {
    const size_t shared = sizeof(float) * (p.key_dim + ROWS + 3);
    exact_fold_kernel<S, C><<<dim3((p.value_dim + ROWS - 1) / ROWS, p.batch, p.value_heads), THREADS, shared, stream>>>(p);
  }
  if (p.max_steps <= 1) verify_steps<A, S, C, 1>(p, stream);
  else if (p.max_steps <= 2) verify_steps<A, S, C, 2>(p, stream);
  else if (p.max_steps <= 4) verify_steps<A, S, C, 4>(p, stream);
  else if (p.max_steps <= 8) verify_steps<A, S, C, 8>(p, stream);
  else if (p.max_steps <= 16) verify_steps<A, S, C, 16>(p, stream);
  else verify_steps<A, S, C, 32>(p, stream);
}

template <class S, class C>
void commit_typed(const Commit& p, hggcStream_t stream) {
  const size_t shared = sizeof(float) * (int64_t(p.capacity) * (p.key_dim + 10) + 2);
  if (shared > 48 * 1024) throw std::invalid_argument("GDN replay commit exceeds 48KiB shared memory");
  compact_commit_kernel<S, C><<<dim3((p.value_dim + 7) / 8, p.batch, p.value_heads * p.layers), THREADS, shared, stream>>>(p);
}

#include "gdn_replayssm_materialize.cuh"
#include "gdn_replayssm_cached.cuh"

#define DISPATCH(TYPE, T, BODY) do { \
  if ((TYPE) == F32) { using T = float; BODY; } \
  else if ((TYPE) == F16) { using T = HF; BODY; } \
  else { using T = BF; BODY; } \
} while (0)

}

bool launch_cached_verify(const CachedVerify& p, hggcStream_t stream) {
  if (p.batch == 0) return false;
#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150
  if (current_device_is_ppu15() && launch_cached_ppu15(p, stream)) return p.materialize_overflow;
#endif
  const dim3 grid((p.value_dim + 15) / 16, p.batch, p.value_heads);
  DISPATCH(p.q.dtype, C, DISPATCH(p.state.dtype, S,
      (cached_verify_kernel<C, S><<<grid, 128, 0, stream>>>(p))));
  return false;
}

void launch_verify(const Verify& p, hggcStream_t stream) {
  if (p.batch == 0 || launch_project_verify(p, stream)) return;
  if (p.mode == 1 && p.key_dim == 128 && p.max_steps <= 8 &&
      p.q.dtype == BF16 && p.state.dtype == F32 && p.d.dtype == BF16) {
    auto aligned = [](const Tensor& t, int bytes, int elements) {
      return t.data && (reinterpret_cast<uintptr_t>(t.data) & (bytes - 1)) == 0 &&
             t.stride[0] % elements == 0;
    };
    if (p.value_dim == 128 && p.capacity <= 32 && aligned(p.state, 16, 4) &&
        aligned(p.keys, 16, 8) && aligned(p.d, 16, 8) && aligned(p.q, 8, 4) && aligned(p.k, 16, 8)) {
      if (p.max_steps == 1) small_verify_steps<1>(p, stream);
      else if (p.max_steps == 2) small_verify_steps<2>(p, stream);
      else if (p.max_steps <= 4) small_verify_steps<4>(p, stream);
      else small_verify_steps<8>(p, stream);
      return;
    }
    if (p.max_steps == 1) verify_steps<BF, float, BF, 1, 128>(p, stream);
    else if (p.max_steps == 2) verify_steps<BF, float, BF, 2, 128>(p, stream);
    else if (p.max_steps <= 4) verify_steps<BF, float, BF, 4, 128>(p, stream);
    else verify_steps<BF, float, BF, 8, 128>(p, stream);
    return;
  }
  DISPATCH(p.q.dtype, A, DISPATCH(p.state.dtype, S, DISPATCH(p.d.dtype, C, (verify_typed<A, S, C>(p, stream)))));
}

void launch_materialize(const Materialize& p, hggcStream_t stream) {
  if (p.batch == 0 || p.layers == 0) return;
  if (p.format == CompactDKG && !p.items.data && !p.n_items.data && !p.targets.data &&
      !p.descriptors.data && p.starts.data && p.metadata_indices.data &&
      p.metadata_slots >= p.replay_slots &&
      p.metadata_indices.data == p.replay_indices.data &&
      p.metadata_indices.dtype == p.replay_indices.dtype &&
      p.metadata_indices.stride[0] == p.replay_indices.stride[0] && !p.buffers.data) {
    Commit c{};
    c.state = p.state; c.d = p.values; c.keys = p.keys; c.g = p.gates;
    c.dr = p.value_residual; c.kr = p.key_residual;
    c.slots = p.state_indices; c.replay = p.replay_indices;
    c.wp = p.lengths; c.base = p.starts; c.flush = p.enabled;
    c.layers = p.layers; c.batch = p.batch; c.heads = p.heads;
    c.value_heads = p.value_heads; c.key_dim = p.key_dim; c.value_dim = p.value_dim;
    c.capacity = p.capacity; c.state_slots = p.state_slots; c.replay_slots = p.replay_slots;
    c.null_slot = p.null_slot; c.residual = p.residual;
    if (launch_fast_commit(c, stream)) return;
  }
  DISPATCH(p.state.dtype, S, DISPATCH(p.values.dtype, C, (materialize_typed<S, C>(p, stream))));
}

void launch_sglang_materialize(const Commit& p, hggcStream_t stream) {
  if (p.batch == 0 || p.layers == 0) return;
  if (!launch_fast_commit(p, stream)) {
    DISPATCH(p.state.dtype, S, DISPATCH(p.d.dtype, C, (commit_typed<S, C>(p, stream))));
  }
}

void launch_sglang_finish(const Commit& p, hggcStream_t stream) {
  if (p.batch && p.layers && !launch_contiguous_sglang_finish(p, stream))
    finish_kernel<<<(p.batch + THREADS - 1) / THREADS, THREADS, 0, stream>>>(p);
}

void launch_cursors(const Cursors& p, hggcStream_t stream) {
  if (p.batch && !launch_contiguous_sglang_cursors(p, stream))
    cursor_kernel<<<(p.batch + THREADS - 1) / THREADS, THREADS, 0, stream>>>(p);
}

#undef DISPATCH
}
