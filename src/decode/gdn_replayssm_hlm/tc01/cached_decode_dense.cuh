// SPDX-License-Identifier: Apache-2.0
#pragma once

__device__ __forceinline__ void tc01_unpack_bf16_pair(uint32_t word, float& first, float& second) {
  first = __uint_as_float(word << 16);
  second = __uint_as_float(word & 0xffff0000u);
}

template <int Group, bool Single, class Params>
__global__ __launch_bounds__(128) void tc01_decode_dense_kernel(Params p) {
  constexpr int K = 128, Cap = 16, TileV = 128 / Group, Pieces = K / (8 * Group);
  static_assert(Group == 4 || Group == 8);
  static_assert(!std::is_same_v<Params, CachedVerify>);
  // On a one-slot pool, state and input addresses are available before the
  // device slot check. Keep the read within the allocated slot in either case.
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = Single ? 0 : blockIdx.y, hv = blockIdx.z, head = hv / 4;
  const int slot = Single ? 0 : static_cast<const int32_t*>(p.slots.data)[row];
  if constexpr (!Single) {
    if (slot <= p.null_slot) return;
    assert(slot >= 0 && slot < p.state_slots && slot < p.replay_slots);
  }
  const int v = tid / Group, key_lane = tid % Group, vbase = blockIdx.x * TileV;
  const int column = hv * K + vbase + v;
  BF* state = static_cast<BF*>(p.state.data) + int64_t(slot) * p.state.stride[0] + column * K + key_lane * 8;

  // Issue all checkpoint reads before the CTA publishes its replay operands.
  uint4 checkpoint[Pieces];
  #pragma unroll
  for (int piece = 0; piece < Pieces; ++piece)
    checkpoint[piece] = *reinterpret_cast<const uint4*>(state + piece * Group * 8);
  const BF* input = static_cast<const BF*>(p.v.data) + int64_t(row) * p.v.stride[0] + column;
  const BF input_value = key_lane == 0 ? *input : BF(0.f);

  __shared__ __align__(32) BF kq[2][K];
  __shared__ float qk;
  if (warp == 0) {
    float q[4], k[4], sq = 0.f, sk = 0.f;
    load_bf16x4(static_cast<const BF*>(p.q.data) + int64_t(row) * p.q.stride[0] + head * K + 4 * lane, q);
    load_bf16x4(static_cast<const BF*>(p.k.data) + int64_t(row) * p.k.stride[0] + head * K + 4 * lane, k);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      sq += q[i] * q[i];
      sk += k[i] * k[i];
    }
    const float qs = p.scale / (sqrtf(sum_warp(sq)) + p.epsilon);
    const float ks = 1.f / (sqrtf(sum_warp(sk)) + p.epsilon);
    uint2 packed_k, packed_q;
    packed_k.x = cached_bf16_pair(k[0] * ks, k[1] * ks);
    packed_k.y = cached_bf16_pair(k[2] * ks, k[3] * ks);
    packed_q.x = cached_bf16_pair(q[0] * qs, q[1] * qs);
    packed_q.y = cached_bf16_pair(q[2] * qs, q[3] * qs);
    *reinterpret_cast<uint2*>(kq[0] + lane * 4) = packed_k;
    *reinterpret_cast<uint2*>(kq[1] + lane * 4) = packed_q;
    tc01_unpack_bf16_pair(packed_k.x, k[0], k[1]);
    tc01_unpack_bf16_pair(packed_k.y, k[2], k[3]);
    tc01_unpack_bf16_pair(packed_q.x, q[0], q[1]);
    tc01_unpack_bf16_pair(packed_q.y, q[2], q[3]);
    float dot = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) dot += q[i] * k[i];
    dot = sum_warp(dot);
    if (lane == 0) qk = dot;
  }
  const int count_raw = static_cast<const int32_t*>(p.lengths.data)[slot];
  const int bank_raw = static_cast<const int32_t*>(p.buffers.data)[slot];
  if constexpr (!Single) assert(count_raw >= 0 && count_raw <= Cap && bank_raw >= 0 && bank_raw < 2);
  const int count = Single ? min(Cap, max(0, count_raw)) : count_raw;
  const int bank = Single ? bank_raw & 1 : bank_raw;
  const bool fold = count == Cap;
  const int dest = fold ? 1 - bank : bank, pos = fold ? 0 : count;
  const int64_t kb = int64_t(slot) * p.keys.stride[0] + bank * p.keys.stride[1] + head * K;
  const int64_t ub = int64_t(slot) * p.values.stride[0] + bank * p.values.stride[1] + hv * K + vbase;
  const int64_t gb = int64_t(slot) * p.gates.stride[0] + bank * p.gates.stride[1] + hv * Cap;
  const BF* keys = static_cast<const BF*>(p.keys.data);
  const BF* values = static_cast<const BF*>(p.values.data);
  const float* gates = static_cast<const float*>(p.gates.data);

  __shared__ __align__(32) BF old_k[Cap * K];
  __shared__ float old_g[Cap], kc[Cap], qc[Cap], scaled_u[Cap * TileV];
  const int64_t gate_offset = int64_t(row) * p.a.stride[0] + hv;
  const float gate = static_cast<const float*>(p.a.data)[gate_offset];
  const float beta = static_cast<const float*>(p.b.data)[gate_offset];
  const float history_g = lane < count ? gates[gb + lane] : 0.f;
  const float origin = __shfl_sync(0xffffffffu, history_g, count ? count - 1 : 0);
  const float total = origin + gate;
  const float decay = expf(total);
  // One eight-lane group computes both K and Q correlations for each
  // history token. All 16 tokens can issue their two K64 loads together.
  const int history_row = tid / 8;
  const int history_col = (lane % 8) * 8;
  uint4 history_lo{}, history_hi{};
  if (history_row < count) {
    const BF* key = keys + kb + history_row * p.keys.stride[2] + history_col;
    history_lo = *reinterpret_cast<const uint4*>(key);
    history_hi = *reinterpret_cast<const uint4*>(key + 64);
  }
  BF history_u[Cap / Group];
  if (count) {
    #pragma unroll
    for (int i = 0; i < Cap / Group; ++i) {
      const int j = min(key_lane + i * Group, count - 1);
      history_u[i] = values[ub + j * p.values.stride[2] + v];
    }
  }
  if (fold && tid < Cap) old_g[tid] = history_g;
  if constexpr (Single) {
    const int active_slot = static_cast<const int32_t*>(p.slots.data)[0];
    if (active_slot <= p.null_slot) return;
    assert(active_slot == 0 && count_raw == count && bank_raw == bank);
  }
  __syncthreads();

  if (count) {
    float kd = 0.f, qd = 0.f;
    const auto accumulate = [&](uint4 packed, int offset) {
      const uint4 pk = *reinterpret_cast<const uint4*>(kq[0] + offset);
      const uint4 pq = *reinterpret_cast<const uint4*>(kq[1] + offset);
      const uint32_t bits[] = {packed.x, packed.y, packed.z, packed.w};
      const uint32_t k[] = {pk.x, pk.y, pk.z, pk.w};
      const uint32_t q[] = {pq.x, pq.y, pq.z, pq.w};
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x0, x1, k0, k1, q0, q1;
        tc01_unpack_bf16_pair(bits[e], x0, x1);
        tc01_unpack_bf16_pair(k[e], k0, k1);
        tc01_unpack_bf16_pair(q[e], q0, q1);
        kd += x0 * k0; kd += x1 * k1;
        qd += x0 * q0; qd += x1 * q1;
      }
    };
    accumulate(history_lo, history_col);
    accumulate(history_hi, history_col + 64);
    kd = reduce_key_group<8>(kd);
    qd = reduce_key_group<8>(qd);
    const float old_gate = __shfl_sync(0xffffffffu, history_g, history_row);
    if (history_col == 0 && history_row < count) {
      const float weight = expf(total - old_gate);
      kc[history_row] = rounded<BF>(kd * weight);
      qc[history_row] = rounded<BF>(qd * weight);
    }
    if (fold) {
      BF* dst = old_k + history_row * K + history_col;
      *reinterpret_cast<uint4*>(dst) = history_lo;
      *reinterpret_cast<uint4*>(dst + 64) = history_hi;
    }
  }
  float kp[2]{}, qp[2]{};
  #pragma unroll
  for (int piece = 0; piece < Pieces; ++piece) {
    const int offset = 8 * (key_lane + piece * Group);
    const uint4 k = *reinterpret_cast<const uint4*>(kq[0] + offset);
    const uint4 q = *reinterpret_cast<const uint4*>(kq[1] + offset);
    const auto update = [&](uint32_t h, uint32_t key, uint32_t query) {
      float h0, h1, k0, k1, q0, q1;
      tc01_unpack_bf16_pair(h, h0, h1);
      tc01_unpack_bf16_pair(key, k0, k1);
      tc01_unpack_bf16_pair(query, q0, q1);
      kp[0] += h0 * k0; kp[1] += h1 * k1;
      qp[0] += h0 * q0; qp[1] += h1 * q1;
    };
    const uint4 h = checkpoint[piece];
    update(h.x, k.x, q.x); update(h.y, k.y, q.y);
    update(h.z, k.z, q.z); update(h.w, k.w, q.w);
  }
  float kh = 0.f, qh = 0.f;
  if (count) {
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Cap / Group; ++i) {
      const int j = key_lane + i * Group;
      if (j < count) {
        const float u = float(history_u[i]);
        kh += kc[j] * u;
        qh += qc[j] * u;
      }
    }
  }
  // Combine history and checkpoint before the two value-row reductions.
  const float kp_all = reduce_key_group<Group>(decay * (kp[0] + kp[1]) + kh);
  const float qp_all = reduce_key_group<Group>(decay * (qp[0] + qp[1]) + qh);
  if (key_lane == 0) {
    const float u = beta * (float(input_value) - kp_all);
    static_cast<BF*>(p.out.data)[int64_t(row) * p.out.stride[0] + column] = BF(qp_all + qk * u);
    static_cast<BF*>(p.values.data)[int64_t(slot) * p.values.stride[0] + dest * p.values.stride[1] +
                                  pos * p.values.stride[2] + column] = BF(u);
  }
  if (blockIdx.x == 0) {
    if (hv % 4 == 0)
      static_cast<BF*>(p.keys.data)[int64_t(slot) * p.keys.stride[0] + dest * p.keys.stride[1] +
                                    pos * p.keys.stride[2] + head * K + tid] = kq[0][tid];
    if (tid == 0)
      static_cast<float*>(p.gates.data)[int64_t(slot) * p.gates.stride[0] + dest * p.gates.stride[1] +
                                       hv * Cap + pos] = fold ? gate : total;
  }
  if (!fold) return;
  #pragma unroll
  for (int i = 0; i < Cap / Group; ++i) {
    const int j = key_lane + i * Group;
    scaled_u[j * TileV + v] = scaled_fold_value<BF>(float(history_u[i]), expf(origin - old_g[j]));
  }
  __syncthreads();
  const float d = expf(origin);
  #pragma unroll
  for (int piece = 0; piece < Pieces; ++piece) {
    const int start = 8 * (key_lane + piece * Group);
    float delta[8]{};
    #pragma unroll 4
    for (int j = 0; j < Cap; ++j) {
      const float u = scaled_u[j * TileV + v];
      const uint4 packed = *reinterpret_cast<const uint4*>(old_k + j * K + start);
      const uint32_t words[] = {packed.x, packed.y, packed.z, packed.w};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float k0, k1;
        tc01_unpack_bf16_pair(words[i], k0, k1);
        delta[2*i] += k0 * u;
        delta[2*i+1] += k1 * u;
      }
    }
    const uint4 h = checkpoint[piece];
    const uint32_t words[] = {h.x, h.y, h.z, h.w};
    uint32_t result[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float h0, h1;
      tc01_unpack_bf16_pair(words[i], h0, h1);
      result[i] = cached_bf16_pair(d * h0 + delta[2*i], d * h1 + delta[2*i+1]);
    }
    *reinterpret_cast<uint4*>(state + piece * Group * 8) = {result[0], result[1], result[2], result[3]};
  }
}
