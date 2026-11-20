// SPDX-License-Identifier: Apache-2.0
#pragma once

template <int Group>
__device__ __forceinline__ float reduce_key_group(float value) {
  #pragma unroll
  for (int offset = Group / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

// One group reduces each value row. On overflow, retain the old checkpoint
// values in registers and reuse the keys brought in by the correlation warps.
template <class S, int Group, class Params>
__global__ __launch_bounds__(128) void tc01_decode_kernel(Params p) {
  constexpr int K = 128, Cap = 16, TileV = 128 / Group, Pieces = K / (4 * Group);
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int batch = blockIdx.y, hv = blockIdx.z, group = p.value_heads / p.heads, head = hv / group;
  const int64_t slot = index(p.slots, batch);
  if (slot <= p.null_slot) return;
  const int64_t replay = std::is_same_v<Params, CachedVerify> ? index(p.replay, batch) : slot;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t count_raw = index(p.lengths, replay), bank_raw = index(p.buffers, replay);
  assert(count_raw >= 0 && count_raw <= Cap && bank_raw >= 0 && bank_raw < 2);
  if (p.offsets.data) assert(index(p.offsets, replay) == 0);
  const int count = int(count_raw), bank = int(bank_raw);
  const bool overflow = count == Cap, fold = p.materialize_overflow && overflow;
  const int dest = overflow ? 1 - bank : bank, pos = overflow ? 0 : count;
  const int vbase = blockIdx.x * TileV, v = tid / Group, key_lane = tid % Group;
  const int64_t old_k_base = replay * p.keys.stride[0] + bank * p.keys.stride[1] + head * K;
  const int64_t old_u_base = replay * p.values.stride[0] + bank * p.values.stride[1] + hv * K + vbase;
  const int64_t gate_base = replay * p.gates.stride[0] + bank * p.gates.stride[1] + hv * Cap;
  const int64_t state_row = slot * p.state.stride[0] + (int64_t(hv) * K + vbase + v) * K;
  __shared__ __align__(32) BF old_k[Cap * K], old_u[Cap * TileV];
  __shared__ __align__(32) float qs[K], ks[K], scaled_u[Cap * TileV];
  __shared__ float old_g[Cap], kc[Cap], qc[Cap];
  __shared__ float origin, total, decay, gate, beta, qk;
  for (int i = tid * 8; i < count * TileV; i += 128 * 8)
    cached_copy_bf8(p.values, old_u_base + (i / TileV) * p.values.stride[2] + i % TileV, 8, old_u + i);
  if (tid < count) old_g[tid] = read<float>(p.gates, gate_base + tid);
  if (warp == 0) {
    float q[4], k[4];
    read_vector<BF,4>(p.q, int64_t(batch) * p.q.stride[0] + head * K + 4 * lane, 4, q);
    read_vector<BF,4>(p.k, int64_t(batch) * p.k.stride[0] + head * K + 4 * lane, 4, k);
    float sq = 0.f, sk = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
    const float qscale = p.scale / (sqrtf(sum_warp(sq)) + p.epsilon);
    const float kscale = 1.f / (sqrtf(sum_warp(sk)) + p.epsilon);
    float dot = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float qval = rounded<BF>(q[i] * qscale), kval = rounded<BF>(k[i] * kscale);
      qs[4 * lane + i] = qval;
      ks[4 * lane + i] = kval;
      dot += qval * kval;
    }
    dot = sum_warp(dot);
    if (lane == 0) qk = dot;
  }
  if (tid == 0) {
    float g = read_any(p.a, int64_t(batch) * p.a.stride[0] + hv);
    float b = read_any(p.b, int64_t(batch) * p.b.stride[0] + hv);
    if (!p.processed_gates) {
      const float arg = g + read<float>(p.dt_bias, hv * p.dt_bias.stride[0]);
      g = -expf(read<float>(p.A_log, hv * p.A_log.stride[0])) *
          (arg <= 20.f ? 0.6931471805599453f * __log2f(__fadd_rn(1.f, __expf(arg))) : arg);
      b = 1.f / (1.f + expf(-b));
    }
    origin = count ? read<float>(p.gates, gate_base + count - 1) : 0.f;
    gate = g;
    beta = b;
    total = origin + g;
    decay = expf(total);
  }
  __syncthreads();
  for (int j = warp; j < count; j += 4) {
    float values[4];
    read_vector<BF,4>(p.keys, old_k_base + j * p.keys.stride[2] + lane * 4, 4, values);
    float kd = 0.f, qd = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = 4 * lane + i;
      if (fold) old_k[j * K + c] = BF(values[i]);
      kd += ks[c] * values[i];
      qd += qs[c] * values[i];
    }
    kd = sum_warp(kd); qd = sum_warp(qd);
    if (lane == 0) {
      const float weight = expf(total - old_g[j]);
      kc[j] = rounded<BF>(kd * weight);
      qc[j] = rounded<BF>(qd * weight);
    }
  }
  float state[Pieces][4];
  float kp[4] = {}, qp[4] = {}, kl[4] = {}, ql[4] = {};
  #pragma unroll
  for (int piece = 0; piece < Pieces; ++piece) {
    const int begin = 4 * (key_lane + piece * Group);
    if constexpr (std::is_same_v<Params, CachedVerify>) {
      read_vector<S, 4>(p.state, state_row + begin, 4, state[piece]);
    } else if constexpr (std::is_same_v<S, BF>) {
      load_bf16x4(static_cast<const BF*>(p.state.data) + state_row + begin, state[piece]);
    } else {
      const float4 value = *reinterpret_cast<const float4*>(
          static_cast<const float*>(p.state.data) + state_row + begin);
      state[piece][0] = value.x;
      state[piece][1] = value.y;
      state[piece][2] = value.z;
      state[piece][3] = value.w;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float key = ks[begin + i], query = qs[begin + i];
      const float x = state[piece][i];
      if constexpr (std::is_same_v<S,float>) {
        const float high = rounded<BF>(x), low = rounded<BF>(x - high);
        kp[i] += key * high;
        qp[i] += query * high;
        kl[i] += key * low;
        ql[i] += query * low;
      } else {
        kp[i] += key * x;
        qp[i] += query * x;
      }
    }
  }
  float kproj = reduce_key_group<Group>((kp[0] + kp[1]) + (kp[2] + kp[3]));
  float qproj = reduce_key_group<Group>((qp[0] + qp[1]) + (qp[2] + qp[3]));
  if constexpr (std::is_same_v<S,float>) {
    kproj += reduce_key_group<Group>((kl[0] + kl[1]) + (kl[2] + kl[3]));
    qproj += reduce_key_group<Group>((ql[0] + ql[1]) + (ql[2] + ql[3]));
  }
  if (count) __syncthreads();
  float kh = 0.f, qh = 0.f;
  for (int j = key_lane; j < count; j += Group) {
    const float u = float(old_u[j * TileV + v]);
    kh += kc[j] * u;
    qh += qc[j] * u;
  }
  kh = reduce_key_group<Group>(kh);
  qh = reduce_key_group<Group>(qh);
  if (key_lane == 0) {
    const int64_t out_column = hv * K + vbase + v;
    const float input = read<BF>(p.v, int64_t(batch) * p.v.stride[0] + out_column);
    const float u = beta * (input - decay * kproj - kh);
    write<BF>(p.out, int64_t(batch) * p.out.stride[0] + out_column, decay * qproj + qh + qk * u);
    write<BF>(p.values, replay * p.values.stride[0] + dest * p.values.stride[1] +
              pos * p.values.stride[2] + out_column, u);
  }
  if (blockIdx.x == 0) {
    if (hv % group == 0)
      write<BF>(p.keys, replay * p.keys.stride[0] + dest * p.keys.stride[1] +
                pos * p.keys.stride[2] + head * K + tid, ks[tid]);
    if (tid == 0)
      write<float>(p.gates, replay * p.gates.stride[0] + dest * p.gates.stride[1] + hv * Cap + pos,
                   overflow ? gate : total);
  }
  if (!fold) return;
  for (int i = tid; i < Cap * TileV; i += 128) {
    scaled_u[i] = scaled_fold_value<BF>(float(old_u[i]), expf(origin - old_g[i / TileV]));
  }
  __syncthreads();
  const float d = expf(origin);
  #pragma unroll
  for (int piece = 0; piece < Pieces; ++piece) {
    const int begin = 4 * (key_lane + piece * Group);
    float delta[4] = {};
    #pragma unroll
    for (int j = 0; j < Cap; ++j) {
      const float u = scaled_u[j * TileV + v];
      #pragma unroll
      for (int i = 0; i < 4; ++i) delta[i] += float(old_k[j * K + begin + i]) * u;
    }
    float updated[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) updated[i] = d * state[piece][i] + delta[i];
    cached_fold_store_vector<S, 4>(p.state, state_row + begin, updated);
  }
}
