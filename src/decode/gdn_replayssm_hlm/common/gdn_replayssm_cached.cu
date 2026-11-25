// SPDX-License-Identifier: Apache-2.0
#include "gdn_replayssm_hlm_abi.h"
#include "../../gdn_replayssm/common/gdn_replayssm_device.cuh"

namespace pla::gdn_replayssm {
namespace {

using namespace detail;

template <class C>
__device__ __forceinline__ float cached_operand(float value, bool tf32) {
  if constexpr (std::is_same_v<C, float>) return operand(value, tf32);
  return value;
}

template <class C, class S, bool InlineFold = false>
__global__ __launch_bounds__(128) void cached_verify_kernel(CachedVerify p) {
  constexpr int ROWS = 16, GROUP = 8, MAX_K = 256, MAX_T = 8, MAX_H = 16;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.y, hv = blockIdx.z, head = hv / (p.value_heads / p.heads);
  const int vr = blockIdx.x * ROWS + tid / GROUP, klane = tid % GROUP;
  const int64_t slot = index(p.slots, row);
  if (slot <= p.null_slot) return;
  const int64_t replay = index(p.replay, row);
  const int64_t begin = p.starts.data ? index(p.starts, row) : int64_t(row) * p.max_steps;
  const int64_t end = p.starts.data ? index(p.starts, row + 1) : begin + p.max_steps;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  assert(begin >= 0 && end >= begin && end <= p.tokens && end - begin <= p.max_steps);
  const int n = int(end - begin);
  if (n == 0) return;
  assert(n <= p.capacity);
  const int64_t count_raw = index(p.lengths, replay), buffer_raw = index(p.buffers, replay);
  assert(count_raw >= 0 && count_raw <= p.capacity && buffer_raw >= 0 && buffer_raw < 2);
  if (p.offsets.data) assert(index(p.offsets, replay) == 0);
  const int count = int(count_raw), buffer = int(buffer_raw), K = p.key_dim;
  const bool overflow = count + n > p.capacity;
  const int write_buffer = overflow ? 1 - buffer : buffer, write_start = overflow ? 0 : count;
  const int64_t key_base = replay * p.keys.stride[0] + buffer * p.keys.stride[1] + head * K;
  const int64_t value_base = replay * p.values.stride[0] + buffer * p.values.stride[1] + hv * p.value_dim;
  const int64_t gate_base = replay * p.gates.stride[0] + buffer * p.gates.stride[1] + hv * p.capacity;

  __shared__ float q[MAX_T * MAX_K], k[MAX_T * MAX_K], stored_k[MAX_T * MAX_K], kh[MAX_H * MAX_K];
  __shared__ float gh[MAX_H], gates[MAX_T], cumulative[MAX_T], beta[MAX_T], origin;
  __shared__ float kk[MAX_T * MAX_T], qk[MAX_T * MAX_T];
  __shared__ float hk[MAX_T * MAX_H], hq[MAX_T * MAX_H];
  float state[MAX_K / GROUP];
  for (int x = klane, i = 0; x < K; x += GROUP, ++i)
    state[i] = vr < p.value_dim ? read<S>(p.state,
        slot * p.state.stride[0] + (int64_t(hv) * p.value_dim + vr) * K + x) : 0.f;
  for (int x = tid; x < count * K; x += 128) {
    const float key = read<C>(p.keys, key_base + (x / K) * p.keys.stride[2] + x % K);
    kh[(x / K) * MAX_K + x % K] = InlineFold ? key : cached_operand<C>(key, p.tf32);
  }
  if (tid < count) gh[tid] = read<float>(p.gates, gate_base + tid);
  if (tid == 0) origin = count ? read<float>(p.gates, gate_base + count - 1) : 0.f;
  for (int t = warp; t < n; t += 4) {
    float square_q = 0.f, square_k = 0.f;
    for (int x = lane; x < K; x += 32) {
      const float qv = read<C>(p.q, (begin + t) * p.q.stride[0] + head * K + x);
      const float kv = read<C>(p.k, (begin + t) * p.k.stride[0] + head * K + x);
      square_q += qv * qv;
      square_k += kv * kv;
    }
    square_q = sum_warp(square_q);
    square_k = sum_warp(square_k);
    const float qscale = p.scale / norm_denominator(square_q, p.norm_policy, p.epsilon);
    const float kscale = 1.f / norm_denominator(square_k, p.norm_policy, p.epsilon);
    for (int x = lane; x < K; x += 32) {
      const float qv = read<C>(p.q, (begin + t) * p.q.stride[0] + head * K + x);
      const float kv = read<C>(p.k, (begin + t) * p.k.stride[0] + head * K + x);
      q[t * MAX_K + x] = cached_operand<C>(rounded<C>(qv * qscale), p.tf32);
      stored_k[t * MAX_K + x] = rounded<C>(kv * kscale);
      k[t * MAX_K + x] = cached_operand<C>(stored_k[t * MAX_K + x], p.tf32);
    }
    if (lane == 0) {
      float g = read_any(p.a, (begin + t) * p.a.stride[0] + hv);
      float b = read_any(p.b, (begin + t) * p.b.stride[0] + hv);
      if (!p.processed_gates) {
        const float x = g + read<float>(p.dt_bias, hv * p.dt_bias.stride[0]);
        g = -expf(read<float>(p.A_log, hv * p.A_log.stride[0])) *
            (x <= 20.f ? 0.6931471805599453f * __log2f(__fadd_rn(1.f, __expf(x))) : x);
        b = 1.f / (1.f + expf(-b));
      }
      gates[t] = g;
      beta[t] = b;
    }
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.f;
    for (int t = 0; t < n; ++t) {
      sum += gates[t];
      cumulative[t] = origin + sum;
    }
  }
  __syncthreads();
  for (int pair = warp; pair < n * (n + count); pair += 4) {
    const int t = pair / (n + count), j = pair % (n + count);
    const float* key = j < n ? k + j * MAX_K : kh + (j - n) * MAX_K;
    float kd = 0.f, qd = 0.f;
    for (int x = lane; x < K; x += 32) {
      const float operand = InlineFold ? cached_operand<C>(key[x], p.tf32) : key[x];
      kd += k[t * MAX_K + x] * operand;
      qd += q[t * MAX_K + x] * operand;
    }
    kd = sum_warp(kd);
    qd = sum_warp(qd);
    if (lane == 0) {
      const float decay = expf(cumulative[t] - (j < n ? cumulative[j] : gh[j - n]));
      if (j < n) {
        kk[t * MAX_T + j] = j < t ? beta[t] * kd * decay : 0.f;
        qk[t * MAX_T + j] = j <= t ? qd * decay : 0.f;
      } else {
        hk[t * MAX_H + j - n] = cached_operand<C>(rounded<C>(kd * decay), p.tf32);
        hq[t * MAX_H + j - n] = cached_operand<C>(rounded<C>(qd * decay), p.tf32);
      }
    }
  }
  float hp[MAX_T] = {}, qp[MAX_T] = {}, hp_low[MAX_T] = {}, qp_low[MAX_T] = {};
  for (int x = klane, i = 0; x < K; x += GROUP, ++i) {
    const float hi = rounded<C>(state[i]);
    const float upper = cached_operand<C>(hi, p.tf32);
    const float lower = cached_operand<C>(rounded<C>(state[i] - hi), p.tf32);
    for (int t = 0; t < n; ++t) {
      hp[t] += upper * k[t * MAX_K + x];
      qp[t] += upper * q[t * MAX_K + x];
      if constexpr (std::is_same_v<S, float>) {
        hp_low[t] += lower * k[t * MAX_K + x];
        qp_low[t] += lower * q[t * MAX_K + x];
      }
    }
  }
  for (int t = 0; t < n; ++t) {
    hp[t] += __shfl_xor_sync(0xffffffffu, hp[t], 4);
    qp[t] += __shfl_xor_sync(0xffffffffu, qp[t], 4);
    hp_low[t] += __shfl_xor_sync(0xffffffffu, hp_low[t], 4);
    qp_low[t] += __shfl_xor_sync(0xffffffffu, qp_low[t], 4);
    hp[t] = sum_row(hp[t]) + sum_row(hp_low[t]);
    qp[t] = sum_row(qp[t]) + sum_row(qp_low[t]);
  }
  __syncthreads();
  float history_k[MAX_T] = {}, history_q[MAX_T] = {};
  for (int j = klane; j < count; j += GROUP) {
    const float u = vr < p.value_dim ? cached_operand<C>(read<C>(p.values,
        value_base + j * p.values.stride[2] + vr), p.tf32) : 0.f;
    for (int t = 0; t < n; ++t) {
      history_k[t] += hk[t * MAX_H + j] * u;
      history_q[t] += hq[t * MAX_H + j] * u;
    }
  }
  for (int t = 0; t < n; ++t) {
    history_k[t] += __shfl_xor_sync(0xffffffffu, history_k[t], 4);
    history_q[t] += __shfl_xor_sync(0xffffffffu, history_q[t], 4);
    history_k[t] = sum_row(history_k[t]);
    history_q[t] = sum_row(history_q[t]);
  }
  if (klane == 0 && vr < p.value_dim) {
    float updates[MAX_T] = {};
    const int64_t ubase = replay * p.values.stride[0] + write_buffer * p.values.stride[1] + hv * p.value_dim;
    for (int t = 0; t < n; ++t) {
      const float decay = expf(cumulative[t]);
      const float value = read<C>(p.v, (begin + t) * p.v.stride[0] + hv * p.value_dim + vr);
      float update = beta[t] * (value - decay * hp[t] - history_k[t]);
      for (int j = 0; j < t; ++j) update -= kk[t * MAX_T + j] * updates[j];
      updates[t] = update;
      float output = decay * qp[t] + history_q[t];
      for (int j = 0; j <= t; ++j) output += qk[t * MAX_T + j] * updates[j];
      write<C>(p.out, (begin + t) * p.out.stride[0] + hv * p.value_dim + vr, output);
      write<C>(p.values, ubase + (write_start + t) * p.values.stride[2] + vr, update);
    }
  }
  if (blockIdx.x == 0) {
    if (hv % (p.value_heads / p.heads) == 0) {
      const int64_t base = replay * p.keys.stride[0] + write_buffer * p.keys.stride[1] + head * K;
      for (int x = tid; x < n * K; x += 128) {
        const int t = x / K, channel = x % K;
        write<C>(p.keys, base + (write_start + t) * p.keys.stride[2] + channel,
                 stored_k[t * MAX_K + channel]);
      }
    }
    if (tid == 0) {
      const int64_t base = replay * p.gates.stride[0] + write_buffer * p.gates.stride[1] + hv * p.capacity;
      float local = 0.f;
      for (int t = 0; t < n; ++t) {
        local += gates[t];
        write<float>(p.gates, base + write_start + t, overflow ? local : cumulative[t]);
      }
    }
  }
  if constexpr (InlineFold) {
    if (!overflow) return;
    // The staged keys and source bank still hold the old prefix. Reuse the
    // query tile after Verify has finished reading it.
    __syncthreads();
    float* upper = q;
    float* lower = q + MAX_H * ROWS;
    for (int i = tid; i < count * ROWS; i += 128) {
      const int t = i / ROWS, v = blockIdx.x * ROWS + i % ROWS;
      if (v < p.value_dim) {
        const float value = read<C>(p.values, value_base + t * p.values.stride[2] + v);
        const float scaled = value * expf(origin - gh[t]);
        upper[i] = rounded<C>(scaled);
        lower[i] = rounded<C>(scaled - upper[i]);
      }
    }
    __syncthreads();
    if (vr < p.value_dim) {
      const int value_row = tid / GROUP;
      const int64_t checkpoint = slot * p.state.stride[0] + (int64_t(hv) * p.value_dim + vr) * K;
      const float decay = expf(origin);
      for (int x = klane, i = 0; x < K; x += GROUP, ++i) {
        float delta = 0.f;
        for (int t = 0; t < count; ++t) {
          const float key = kh[t * MAX_K + x];
          delta += key * upper[t * ROWS + value_row];
          delta += key * lower[t * ROWS + value_row];
        }
        write<S>(p.state, checkpoint + x, state[i] * decay + delta);
      }
    }
  }
}

}  // namespace

bool launch_cached_verify(const CachedVerify& p, hggcStream_t stream) {
  if (p.batch == 0) return false;
  const int capability = current_device_compute_capability();
  if ((capability == 80 && launch_cached_ppu10(p, stream)) ||
      (capability == 89 && launch_cached_ppu15(p, stream)))
    return p.materialize_overflow;
  const dim3 grid((p.value_dim + 15) / 16, p.batch, p.value_heads);
  if (capability == 80 && p.materialize_overflow) {
    PLA_GDN_REPLAY_DISPATCH(p.q.dtype, C, PLA_GDN_REPLAY_DISPATCH(p.state.dtype, S,
        (cached_verify_kernel<C, S, true><<<grid, 128, 0, stream>>>(p))));
    return true;
  }
  PLA_GDN_REPLAY_DISPATCH(p.q.dtype, C, PLA_GDN_REPLAY_DISPATCH(p.state.dtype, S,
      (cached_verify_kernel<C, S><<<grid, 128, 0, stream>>>(p))));
  return false;
}

}  // namespace pla::gdn_replayssm

#undef PLA_GDN_REPLAY_DISPATCH
