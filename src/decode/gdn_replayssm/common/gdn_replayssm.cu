// SPDX-License-Identifier: Apache-2.0
#include "gdn_replayssm_device.cuh"

#include <hggc_mma.h>
#include <stdexcept>

namespace pla::gdn_replayssm {
namespace {

using namespace detail;

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
  const int base = physical_base(index(p.base, replay), p.capacity, p.flush.data != nullptr);
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
  assert(wp_raw >= 0 && wp_raw <= p.capacity);
  const int wp = static_cast<int>(wp_raw);
  const int base = physical_base(base_raw, p.capacity, p.flush.data != nullptr);
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

template <class A, class S, class C, int STEPS, int STATIC_K = 0>
void verify_steps(const Verify& p, hggcStream_t stream) {
  const size_t shared = verify_shared_bytes(p, STEPS);
  if (shared > 48 * 1024) throw std::invalid_argument("GDN replay verify exceeds 48KiB shared memory");
  verify_kernel<A, S, C, STEPS, STATIC_K><<<dim3((p.value_dim + ROWS - 1) / ROWS, p.batch, p.value_heads), THREADS, shared, stream>>>(p);
}

template <class A, class S, class C>
void verify_typed(const Verify& p, hggcStream_t stream) {
  if (p.max_steps <= 1) verify_steps<A, S, C, 1>(p, stream);
  else if (p.max_steps <= 2) verify_steps<A, S, C, 2>(p, stream);
  else if (p.max_steps <= 4) verify_steps<A, S, C, 4>(p, stream);
  else if (p.max_steps <= 8) verify_steps<A, S, C, 8>(p, stream);
  else if (p.max_steps <= 16) verify_steps<A, S, C, 16>(p, stream);
  else verify_steps<A, S, C, 32>(p, stream);
}

#include "gdn_replayssm_commit.cuh"
#include "gdn_replayssm_materialize.cuh"

}  // namespace

bool launch_fast_commit(const Commit& p, hggcStream_t stream) {
  if (p.tracking || p.state.dtype != F32 || p.key_dim != FAST_COMMIT_K ||
      p.capacity < 1 || p.capacity > 32 || (p.capacity & (p.capacity - 1)) != 0 ||
      p.value_dim < 1 || p.batch < 1 || p.layers < 1 || p.heads < 1 ||
      p.value_heads < 1 || p.value_heads % p.heads != 0)
    return false;
  if ((p.d.dtype != F32 && p.d.dtype != F16 && p.d.dtype != BF16) ||
      p.keys.dtype != p.d.dtype || p.g.dtype != F32 ||
      (p.residual && (p.dr.dtype != p.d.dtype || p.kr.dtype != p.d.dtype)))
    return false;
  const int64_t grid_z = int64_t(p.value_heads) * p.layers;
  if (p.batch > 65535 || grid_z > 65535) return false;
  // Few heads leave most CUs idle; 16-row tiles finish in one pass with 4x the CTAs. Beyond 32
  // heads the extra CTAs cost more in the no-op rounds than the single pass saves.
  const int64_t head_work = int64_t(p.batch) * p.value_heads * p.layers;
  const bool wide = p.d.dtype == BF16 && p.capacity == 16 && p.value_dim == 128 && head_work >= 1024;
  // The 32-position residual history needs a smaller value tile to fit shared memory.
  const int tile_v = head_work <= 32 || (p.residual && p.capacity == 32) ? 16 : wide ? 128 : 64;
  const int cache_vector = 16 / (p.d.dtype == F32 ? 4 : 2);
  auto aligned = [](const Tensor& t, int vector) {
    return t.data && (reinterpret_cast<uintptr_t>(t.data) & 15) == 0 &&
           t.stride[0] % vector == 0 && t.stride[1] % vector == 0;
  };
  if (p.value_dim % tile_v != 0 || !aligned(p.state, 4) || !aligned(p.keys, cache_vector) ||
      !aligned(p.d, cache_vector) ||
      (p.residual && (!aligned(p.kr, cache_vector) || !aligned(p.dr, cache_vector))))
    return false;
  const size_t shared = sizeof(float) *
      (size_t(p.capacity) * (p.residual ? 2 : 1) * (FAST_COMMIT_K + tile_v) + 1);
  if (shared > 48 * 1024) return false;
  const dim3 grid(static_cast<unsigned int>((int64_t(p.value_dim) + tile_v - 1) / tile_v),
                  p.batch, static_cast<unsigned int>(grid_z));
  const bool i32 = p.slots.dtype == I32 && p.replay.dtype == I32 && p.wp.dtype == I32 &&
                   p.base.dtype == I32;
  const CommitLite q{p.state_slots, p.replay_slots, p.null_slot, p.residual, p.extended_base,
                     lite(p.slots), lite(p.replay), lite(p.flush), lite(p.wp), lite(p.base),
                     lite(p.state), lite(p.d), lite(p.keys), lite(p.g), lite(p.dr), lite(p.kr),
                     p.heads, p.value_heads, p.value_dim, p.capacity};
  if (p.d.dtype == BF16 && p.value_dim == 128 && p.capacity <= 16) {
    if (head_work <= 32) {
      launch_tensor_commit<16>(q, i32, grid, stream);
      return true;
    }
    // Wider matrix tiles serve the circular adapter's int32 cursors and int8 flush flags.
    if (p.extended_base && p.residual && p.capacity == 16 && i32 && p.flush.dtype == I8) {
      if (head_work <= 128) launch_tensor_commit<32>(q, i32, grid, stream);
      else launch_tensor_commit<64>(q, i32, grid, stream);
      return true;
    }
  }
  if (wide) {
    if (i32 && p.flush.dtype == I8)
      fast_commit_kernel<BF, 16, true, 128, true, true><<<grid, FAST_COMMIT_THREADS, shared, stream>>>(q);
    else if (i32)
      fast_commit_kernel<BF, 16, true, 128, false, true><<<grid, FAST_COMMIT_THREADS, shared, stream>>>(q);
    else
      fast_commit_kernel<BF, 16, false, 128, false, true><<<grid, FAST_COMMIT_THREADS, shared, stream>>>(q);
    return true;
  }
  if (i32 && p.d.dtype == BF16 && p.capacity <= 16 && p.flush.dtype == I8) {
    if (tile_v == 16) fast_commit_kernel<BF, 16, true, 16, true>
        <<<grid, FAST_COMMIT_THREADS, shared, stream>>>(q);
    else fast_commit_kernel<BF, 16, true, 64, true>
        <<<grid, FAST_COMMIT_THREADS, shared, stream>>>(q);
    return true;
  }
  if (i32) {
    if (tile_v == 16) launch_fast_commit_capacity<true, 16>(q, grid, shared, stream);
    else launch_fast_commit_capacity<true, 64>(q, grid, shared, stream);
  } else {
    if (tile_v == 16) launch_fast_commit_capacity<false, 16>(q, grid, shared, stream);
    else launch_fast_commit_capacity<false, 64>(q, grid, shared, stream);
  }
  return true;
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
  PLA_GDN_REPLAY_DISPATCH(p.q.dtype, A, PLA_GDN_REPLAY_DISPATCH(p.state.dtype, S, PLA_GDN_REPLAY_DISPATCH(p.d.dtype, C, (verify_typed<A, S, C>(p, stream)))));
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
  PLA_GDN_REPLAY_DISPATCH(p.state.dtype, S, PLA_GDN_REPLAY_DISPATCH(p.values.dtype, C, (materialize_typed<S, C>(p, stream))));
}

}  // namespace pla::gdn_replayssm

#undef PLA_GDN_REPLAY_DISPATCH
