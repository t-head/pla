// SPDX-License-Identifier: Apache-2.0
#pragma once

// Loads are issued before the first barrier and outside data-dependent branches (a load inside a
// branch must complete before the merge): row metadata, then cursors plus every token's Q/K/V
// element, then the whole history ring. Q/K norms are reduced across the four warps.
template <int T, int MAXC, bool I32, bool CompatTf32 = false>
__global__ __launch_bounds__(128) void project_verify_prepare_kernel(VerifyLite p) {
  const int capacity = CompatTf32 ? 16 : p.capacity;
  const bool exact = !CompatTf32 && p.exact;
  const int row = blockIdx.y, hv = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int h = hv / (p.value_heads / p.heads);
  const int64_t slot = meta_index<I32>(p.slots, row), replay = meta_index<I32>(p.replay, row);
  const int64_t bos = p.starts.data ? meta_index<I32>(p.starts, row) : row;
  const int64_t eos = p.starts.data ? meta_index<I32>(p.starts, row + 1) : row + 1;
  assert(bos >= 0 && eos >= bos && eos <= p.tokens && eos - bos <= T);
  const int n = static_cast<int>(eos - bos);
  if (n == 0 || slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t safe_replay = replay >= 0 && replay < p.replay_slots ? replay : 0;
  const bool flush = p.flush.data && (CompatTf32
      ? read<int8_t>(p.flush, safe_replay * p.flush.stride[0]) != 0
      : index(p.flush, safe_replay) != 0);
  const int64_t wp_raw = meta_index<I32>(p.wp, safe_replay);
  const int64_t base_raw = meta_index<I32>(p.base, safe_replay);
  auto token_row = [&](int t) { return max(int64_t(0), min(bos + t, int64_t(p.tokens) - 1)); };
  float qv[T], kv[T], vv[T];
  #pragma unroll
  for (int t = 0; t < T; ++t) {
    const int64_t r = token_row(t);
    qv[t] = read<BF>(p.q, r * p.q.stride[0] + h * 128 + tid);
    kv[t] = read<BF>(p.k, r * p.k.stride[0] + h * 128 + tid);
    vv[t] = read<BF>(p.v, r * p.v.stride[0] + hv * 128 + tid);
  }
  const int64_t gate_row = token_row(min(tid, T - 1));
  const bool processed = !CompatTf32 && p.processed_gates;
  const int64_t a_offset = gate_row * p.a.stride[0] + hv;
  const int64_t b_offset = gate_row * p.b.stride[0] + hv;
  const float a_raw = CompatTf32 ? read<BF>(p.a, a_offset) : read_any(p.a, a_offset);
  const float b_raw = CompatTf32 ? read<BF>(p.b, b_offset) : read_any(p.b, b_offset);
  float dt_bias, A_log;
  if constexpr (CompatTf32) {
    dt_bias = read<float>(p.dt_bias, hv * p.dt_bias.stride[0]);
    A_log = read<float>(p.A_log, hv * p.A_log.stride[0]);
  } else {
    // Processed gates use `a` as a safe source for unconditional parameter loads.
    const LiteTensor& bias_source = processed ? p.a : p.dt_bias;
    const LiteTensor& log_source = processed ? p.a : p.A_log;
    dt_bias = read_any(bias_source, processed ? 0 : hv * bias_source.stride[0]);
    A_log = read_any(log_source, processed ? 0 : hv * log_source.stride[0]);
  }

  if (flush) return;
  assert(wp_raw >= 0 && wp_raw <= capacity && base_raw >= 0 && base_raw < capacity);
  const int wp = static_cast<int>(wp_raw), base = static_cast<int>(base_raw);
  assert(wp + n <= capacity);
  if (n == 0) return;
  const bool key_writer = hv % (p.value_heads / p.heads) == 0;

  extern __shared__ float project_prepare_mem[];
  float* qs = project_prepare_mem;
  float* ks = qs + T * 128;
  float* kk = ks + T * 128;
  float* kq = kk + T * T;
  float* hist_k = kq + T * T;
  float* hist_q = hist_k + capacity * T;
  float* decay = hist_q + capacity * T;
  float* gates = decay + capacity;
  float* G = gates + T;
  float* beta = G + T;
  float* norms = beta + T;
  float* hdec = norms + 2 * T;
  __shared__ float partial[4][2 * T];

  constexpr int KEY_ROWS = (MAXC + 3) / 4;
  float hist_keys[KEY_ROWS][4], hist_d[MAXC];
  #pragma unroll
  for (int i = 0; i < KEY_ROWS; ++i) {
    const int old = min(warp + 4 * i, max(wp - 1, 0));
    const int phys = (base + old) & (capacity - 1);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const BF* src = wp ? static_cast<const BF*>(p.keys.data) + replay * p.keys.stride[0] +
                         (h * capacity + phys) * int64_t(128) + lane + 32 * j :
                         static_cast<const BF*>(p.k.data) + bos * p.k.stride[0] + h * 128 + lane + 32 * j;
      hist_keys[i][j] = static_cast<float>(*src);
    }
  }
  #pragma unroll
  for (int old = 0; old < MAXC; ++old) {
    const int phys = (base + old) & (capacity - 1);
    hist_d[old] = read<BF>(p.d, replay * p.d.stride[0] + (hv * capacity + phys) * int64_t(128) + tid);
  }
  if (warp == 0) {
    const float exp_total = history_decay(p.g, replay * p.g.stride[0] + hv * capacity,
                                          base, wp, capacity, decay, lane);
    if (lane == 0) *hdec = exp_total;
  }
  #pragma unroll
  for (int t = 0; t < T; ++t) {
    const float qq = sum_warp(qv[t] * qv[t]), kk2 = sum_warp(kv[t] * kv[t]);
    if (lane == 0) { partial[warp][t] = qq; partial[warp][T + t] = kk2; }
  }
  __syncthreads();

  #pragma unroll
  for (int t = 0; t < T; ++t) {
    const float qnorm = partial[0][t] + partial[1][t] + partial[2][t] + partial[3][t];
    const float knorm = partial[0][T + t] + partial[1][T + t] + partial[2][T + t] + partial[3][T + t];
    float q = 0.f, k = 0.f;
    if (t < n) {
      q = qv[t] * (p.scale * (1.f / norm_denominator(qnorm, p.norm_policy, p.epsilon)));
      k = kv[t] * (1.f / norm_denominator(knorm, p.norm_policy, p.epsilon));
      if (key_writer) {
        const float hi = rounded<BF>(k), lo = rounded<BF>(k - hi);
        const int phys = (base + wp + t) & (capacity - 1);
        const int64_t off = (h * capacity + phys) * int64_t(128) + tid;
        write<BF>(p.keys, replay * p.keys.stride[0] + off, hi);
        if (exact || p.residual)
          write<BF>(p.rawk, replay * p.rawk.stride[0] + off, exact ? kv[t] : lo);
      }
    }
    qs[t * 128 + tid] = operand(q, CompatTf32 || p.tf32);
    ks[t * 128 + tid] = operand(k, CompatTf32 || p.tf32);
  }
  if (tid < T) {
    const float x = a_raw + dt_bias;
    const float gate = processed ? a_raw : -expf(A_log) * (x <= 20.f ? log1pf(expf(x)) : x);
    gates[tid] = tid < n ? gate : 0.f;
    beta[tid] = tid < n ? (processed ? b_raw : 1.f / (1.f + expf(-b_raw))) : 0.f;
  }
  __syncthreads();

  // Every thread forms the same sequential prefix, so no further barrier is needed for G.
  float Gr[T];
  {
    float prefix = 0.f;
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t < n) prefix += gates[t];
      Gr[t] = t < n ? prefix : 0.f;
    }
  }
  // Normalization scratch now holds exp(G) and exp(G) * history decay.
  if (tid < T) {
    norms[tid] = tid < n ? expf(Gr[tid]) : 0.f;
    norms[T + tid] = tid < n ? norms[tid] * *hdec : 0.f;
  }
  for (int pair = warp; pair < T * T; pair += 4) {
    const int t = pair / T, j = pair % T;
    float a = 0.f, f = 0.f;
    if (t < n && j <= t) {
      #pragma unroll
      for (int x = lane; x < 128; x += 32) {
        const float kj = ks[j * 128 + x];
        if (j < t) a = fmaf(ks[t * 128 + x], kj, a);
        f = fmaf(kj, qs[t * 128 + x], f);
      }
    }
    a = sum_warp(a);
    f = sum_warp(f);
    if (lane == 0) {
      float gt = 0.f, gj = 0.f;
      #pragma unroll
      for (int i = 0; i < T; ++i) { if (i == t) gt = Gr[i]; if (i == j) gj = Gr[i]; }
      const float e = t < n && j <= t ? expf(gt - gj) : 0.f;
      kk[pair] = j < t ? beta[t] * e * a : 0.f;
      kq[pair] = e * f;
    }
  }
  #pragma unroll
  for (int i = 0; i < KEY_ROWS; ++i) {
    const int old = warp + 4 * i;
    if (old >= wp) continue;
    float key[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) key[j] = operand(hist_keys[i][j], CompatTf32 || p.tf32);
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      float a = 0.f, b = 0.f;
      if (t < n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          a = fmaf(key[j], ks[t * 128 + lane + 32 * j], a);
          b = fmaf(key[j], qs[t * 128 + lane + 32 * j], b);
        }
      }
      a = sum_warp(a);
      b = sum_warp(b);
      if (lane == 0) {
        hist_k[old * T + t] = operand(a, CompatTf32 || p.tf32);
        hist_q[old * T + t] = operand(b, CompatTf32 || p.tf32);
      }
    }
  }
  __syncthreads();

  float hk[T] = {}, hq[T] = {}, W[T] = {}, U[T] = {};
  #pragma unroll
  for (int old = 0; old < MAXC; ++old) {
    if (old >= wp) break;
    const float d = operand(hist_d[old] * decay[old], CompatTf32 || p.tf32);
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t < n) {
        hk[t] = fmaf(d, hist_k[old * T + t], hk[t]);
        hq[t] = fmaf(d, hist_q[old * T + t], hq[t]);
      }
    }
  }
  float* workspace = p.workspace + (int64_t(row) * p.value_heads + hv) * (4 * T * 128);
  #pragma unroll
  for (int t = 0; t < T; ++t) {
    float qe = 0.f, bias = 0.f;
    if (t < n) {
      const float v = vv[t];
      W[t] = ((beta[t] * norms[t]) * *hdec) * ks[t * 128 + tid];
      U[t] = beta[t] * fmaf(-norms[t], hk[t], v);
      #pragma unroll
      for (int j = 0; j < T; ++j) {
        if (j < t) {
          W[t] = fmaf(-kk[t * T + j], W[j], W[t]);
          U[t] = fmaf(-kk[t * T + j], U[j], U[t]);
        }
      }
      qe = norms[T + t] * qs[t * 128 + tid];
      bias = norms[t] * hq[t];
      #pragma unroll
      for (int j = 0; j < T; ++j) {
        if (j <= t) {
          qe = fmaf(-kq[t * T + j], W[j], qe);
          bias = fmaf(kq[t * T + j], U[j], bias);
        }
      }
    }
    workspace[t * 128 + tid] = W[t];
    workspace[(T + t) * 128 + tid] = qe;
    workspace[(2 * T + t) * 128 + tid] = U[t];
    workspace[(3 * T + t) * 128 + tid] = bias;
  }
  if (tid < n) {
    const int phys = (base + wp + tid) & (capacity - 1);
    write<float>(p.g, replay * p.g.stride[0] + hv * capacity + phys, gates[tid]);
    if (exact)
      write<float>(p.beta, replay * p.beta.stride[0] + hv * capacity + phys, beta[tid]);
  }
}

// Register-lean variant for large grids, where occupancy rather than load latency limits speed.
template <int T, bool I32>
__global__ __launch_bounds__(128) void project_verify_prepare_lean_kernel(VerifyLite p) {
  const int row = blockIdx.y, hv = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t slot = meta_index<I32>(p.slots, row);
  if (slot <= p.null_slot) return;
  const int64_t replay = meta_index<I32>(p.replay, row);
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t bos = p.starts.data ? meta_index<I32>(p.starts, row) : row;
  const int64_t eos = p.starts.data ? meta_index<I32>(p.starts, row + 1) : row + 1;
  assert(bos >= 0 && eos >= bos && eos <= p.tokens && eos - bos <= T);
  const int n = eos - bos;
  if (p.flush.data && index(p.flush, replay)) return;
  const int64_t wp_raw = meta_index<I32>(p.wp, replay), base_raw = meta_index<I32>(p.base, replay);
  assert(wp_raw >= 0 && wp_raw <= p.capacity && base_raw >= 0 && base_raw < p.capacity);
  const int wp = static_cast<int>(wp_raw), base = static_cast<int>(base_raw);
  assert(wp + n <= p.capacity);
  if (n == 0) return;
  const int h = hv / (p.value_heads / p.heads);
  const bool key_writer = hv % (p.value_heads / p.heads) == 0;
  const LiteTensor& bias_source = p.processed_gates ? p.a : p.dt_bias;
  const LiteTensor& log_source = p.processed_gates ? p.a : p.A_log;

  extern __shared__ float project_prepare_mem[];
  float* qs = project_prepare_mem;
  float* ks = qs + T * 128;
  float* kk = ks + T * 128;
  float* kq = kk + T * T;
  float* hist_k = kq + T * T;
  float* hist_q = hist_k + p.capacity * T;
  float* decay = hist_q + p.capacity * T;
  float* gates = decay + p.capacity;
  float* G = gates + T;
  float* beta = G + T;
  float* norms = beta + T;
  float* hdec = norms + 2 * T;

  for (int t = warp; t < T; t += 4) {
    float qnorm = 0.f, knorm = 0.f;
    if (t < n) {
      #pragma unroll
      for (int x = lane; x < 128; x += 32) {
        const float q = read<BF>(p.q, (bos + t) * p.q.stride[0] + h * 128 + x);
        const float k = read<BF>(p.k, (bos + t) * p.k.stride[0] + h * 128 + x);
        qnorm = fmaf(q, q, qnorm);
        knorm = fmaf(k, k, knorm);
      }
    }
    qnorm = sum_warp(qnorm);
    knorm = sum_warp(knorm);
    if (lane == 0) {
      norms[t] = 0.f;
      norms[T + t] = 0.f;
      gates[t] = 0.f;
      beta[t] = 0.f;
      if (t < n) {
        norms[t] = p.scale * (1.f / norm_denominator(qnorm, p.norm_policy, p.epsilon));
        norms[T + t] = 1.f / norm_denominator(knorm, p.norm_policy, p.epsilon);
        const float a = read_any(p.a, (bos + t) * p.a.stride[0] + hv);
        const float b = read_any(p.b, (bos + t) * p.b.stride[0] + hv);
        const float dt_bias = read_any(bias_source, p.processed_gates ? 0 : hv * bias_source.stride[0]);
        const float A_log = read_any(log_source, p.processed_gates ? 0 : hv * log_source.stride[0]);
        const float x = a + dt_bias;
        gates[t] = p.processed_gates ? a : -expf(A_log) * (x <= 20.f ? log1pf(expf(x)) : x);
        beta[t] = p.processed_gates ? b : 1.f / (1.f + expf(-b));
      }
    }
  }
  __syncthreads();

  #pragma unroll
  for (int t = 0; t < T; ++t) {
    float q = 0.f, k = 0.f;
    if (t < n) {
      q = read<BF>(p.q, (bos + t) * p.q.stride[0] + h * 128 + tid) * norms[t];
      const float rawk = read<BF>(p.k, (bos + t) * p.k.stride[0] + h * 128 + tid);
      k = rawk * norms[T + t];
      if (key_writer) {
        const float hi = rounded<BF>(k), lo = rounded<BF>(k - hi);
        const int phys = (base + wp + t) & (p.capacity - 1);
        const int64_t off = (h * p.capacity + phys) * int64_t(128) + tid;
        write<BF>(p.keys, replay * p.keys.stride[0] + off, hi);
        if (p.exact || p.residual)
          write<BF>(p.rawk, replay * p.rawk.stride[0] + off, p.exact ? rawk : lo);
      }
    }
    qs[t * 128 + tid] = operand(q, p.tf32);
    ks[t * 128 + tid] = operand(k, p.tf32);
  }
  if (warp == 0) {
    if (lane == 0) {
      float prefix = 0.f;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t < n) prefix += gates[t];
        G[t] = t < n ? prefix : 0.f;
      }
    }
    const float exp_total = history_decay(p.g, replay * p.g.stride[0] + hv * p.capacity,
                                          base, wp, p.capacity, decay, lane);
    if (lane == 0) *hdec = exp_total;
  }
  __syncthreads();

  if (tid < T) {
    norms[tid] = tid < n ? expf(G[tid]) : 0.f;
    norms[T + tid] = tid < n ? norms[tid] * *hdec : 0.f;
  }
  for (int pair = warp; pair < T * T; pair += 4) {
    const int t = pair / T, j = pair % T;
    float a = 0.f, f = 0.f;
    if (t < n && j <= t) {
      #pragma unroll
      for (int x = lane; x < 128; x += 32) {
        const float kj = ks[j * 128 + x];
        if (j < t) a = fmaf(ks[t * 128 + x], kj, a);
        f = fmaf(kj, qs[t * 128 + x], f);
      }
    }
    a = sum_warp(a);
    f = sum_warp(f);
    if (lane == 0) {
      const float e = t < n && j <= t ? expf(G[t] - G[j]) : 0.f;
      kk[pair] = j < t ? beta[t] * e * a : 0.f;
      kq[pair] = e * f;
    }
  }
  for (int old = warp; old < wp; old += 4) {
    const int phys = (base + old) & (p.capacity - 1);
    float key[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      key[j] = operand(read<BF>(p.keys, replay * p.keys.stride[0] +
                       (h * p.capacity + phys) * int64_t(128) + lane + 32 * j), p.tf32);
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      float a = 0.f, b = 0.f;
      if (t < n) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          a = fmaf(key[j], ks[t * 128 + lane + 32 * j], a);
          b = fmaf(key[j], qs[t * 128 + lane + 32 * j], b);
        }
      }
      a = sum_warp(a);
      b = sum_warp(b);
      if (lane == 0) {
        hist_k[old * T + t] = operand(a, p.tf32);
        hist_q[old * T + t] = operand(b, p.tf32);
      }
    }
  }
  __syncthreads();

  float hk[T] = {}, hq[T] = {}, W[T] = {}, U[T] = {};
  #pragma unroll 4
  for (int old = 0; old < wp; ++old) {
    const int phys = (base + old) & (p.capacity - 1);
    const float d = operand(read<BF>(p.d, replay * p.d.stride[0] +
                            (hv * p.capacity + phys) * int64_t(128) + tid) * decay[old], p.tf32);
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t < n) {
        hk[t] = fmaf(d, hist_k[old * T + t], hk[t]);
        hq[t] = fmaf(d, hist_q[old * T + t], hq[t]);
      }
    }
  }
  float* workspace = p.workspace + (int64_t(row) * p.value_heads + hv) * (4 * T * 128);
  #pragma unroll
  for (int t = 0; t < T; ++t) {
    float qe = 0.f, bias = 0.f;
    if (t < n) {
      const float v = read<BF>(p.v, (bos + t) * p.v.stride[0] + hv * 128 + tid);
      W[t] = ((beta[t] * norms[t]) * *hdec) * ks[t * 128 + tid];
      U[t] = beta[t] * fmaf(-norms[t], hk[t], v);
      #pragma unroll
      for (int j = 0; j < T; ++j) {
        if (j < t) {
          W[t] = fmaf(-kk[t * T + j], W[j], W[t]);
          U[t] = fmaf(-kk[t * T + j], U[j], U[t]);
        }
      }
      qe = norms[T + t] * qs[t * 128 + tid];
      bias = norms[t] * hq[t];
      #pragma unroll
      for (int j = 0; j < T; ++j) {
        if (j <= t) {
          qe = fmaf(-kq[t * T + j], W[j], qe);
          bias = fmaf(kq[t * T + j], U[j], bias);
        }
      }
    }
    workspace[t * 128 + tid] = W[t];
    workspace[(T + t) * 128 + tid] = qe;
    workspace[(2 * T + t) * 128 + tid] = U[t];
    workspace[(3 * T + t) * 128 + tid] = bias;
  }
  if (tid < n) {
    const int phys = (base + wp + tid) & (p.capacity - 1);
    write<float>(p.g, replay * p.g.stride[0] + hv * p.capacity + phys, gates[tid]);
    if (p.exact)
      write<float>(p.beta, replay * p.beta.stride[0] + hv * p.capacity + phys, beta[tid]);
  }
}

template <int T, bool I32, bool EAGER, bool CompatTf32 = false>
__global__ __launch_bounds__(128) void project_verify_project_kernel(VerifyLite p) {
  const int capacity = CompatTf32 ? 16 : p.capacity;
  const bool exact = !CompatTf32 && p.exact;
  const int row = blockIdx.y, hv = blockIdx.z, tid = threadIdx.x;
  const int vbase = blockIdx.x * 32;
  const int64_t slot = meta_index<I32>(p.slots, row), replay = meta_index<I32>(p.replay, row);
  const int64_t bos = p.starts.data ? meta_index<I32>(p.starts, row) : row;
  const int64_t eos = p.starts.data ? meta_index<I32>(p.starts, row + 1) : row + 1;
  assert(bos >= 0 && eos >= bos && eos <= p.tokens && eos - bos <= T);
  const int n = static_cast<int>(eos - bos);
  if (n == 0) return;
  if (slot <= p.null_slot) {
    for (int i = tid; i < T * 32; i += 128) {
      const int t = i / 32, v = vbase + i % 32;
      if (t < n) write<BF>(p.out, (bos + t) * p.out.stride[0] + hv * 128 + v, 0.f);
    }
    return;
  }
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t safe_slot = slot >= 0 && slot < p.state_slots ? slot : 0;
  const int64_t safe_replay = replay >= 0 && replay < p.replay_slots ? replay : 0;
  const bool flush = p.flush.data && (CompatTf32
      ? read<int8_t>(p.flush, safe_replay * p.flush.stride[0]) != 0
      : index(p.flush, safe_replay) != 0);
  const int64_t wp_raw = meta_index<I32>(p.wp, safe_replay);
  const int64_t base_raw = meta_index<I32>(p.base, safe_replay);
  // Unconditional loads (launch_project_verify admits only 16-byte aligned state rows); a load
  // inside a branch must complete before the merge.
  const float* workspace = p.workspace + (int64_t(row) * p.value_heads + hv) * (4 * T * 128);
  __shared__ float4 vectors[2 * T][32];
  __shared__ float results[2 * T][32];
  // Few heads are latency bound: stage the workspace before the checks. Many heads are
  // occupancy bound: load it after them without staging registers.
  constexpr int STAGES = EAGER ? (2 * T * 32 + 127) / 128 : 1;
  float4 staged_vectors[STAGES];
  float staged_results[STAGES];
  if constexpr (EAGER) {
    #pragma unroll
    for (int j = 0; j < STAGES; ++j) {
      const int i = min(tid + 128 * j, 2 * T * 32 - 1);
      staged_vectors[j] = *reinterpret_cast<const float4*>(workspace + i * 4);
      staged_results[j] = workspace[(2 * T + i / 32) * 128 + vbase + i % 32];
    }
  }
  const int group_lane = tid & 7, group = tid >> 3;
  float4 state_values[2][4];
  #pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int v = vbase + pass * 16 + group;
    const float* src = static_cast<const float*>(p.state.data) + safe_slot * p.state.stride[0] +
                       (hv * int64_t(128) + v) * 128 + group_lane * 4;
    #pragma unroll
    for (int x = 0; x < 4; ++x) state_values[pass][x] = *reinterpret_cast<const float4*>(src + x * 32);
  }

  if (flush) return;
  assert(wp_raw >= 0 && wp_raw <= capacity && base_raw >= 0 && base_raw < capacity);
  const int wp = static_cast<int>(wp_raw), base = static_cast<int>(base_raw);
  assert(wp + n <= capacity);
  if (n == 0) return;

  if constexpr (EAGER) {
    #pragma unroll
    for (int j = 0; j < STAGES; ++j) {
      const int i = tid + 128 * j;
      if (i < 2 * T * 32) {
        vectors[i / 32][i % 32] = staged_vectors[j];
        results[i / 32][i % 32] = staged_results[j];
      }
    }
  } else {
    #pragma unroll
    for (int i = tid; i < 2 * T * 32; i += 128) {
      vectors[i / 32][i % 32] = *reinterpret_cast<const float4*>(workspace + i * 4);
      results[i / 32][i % 32] = workspace[(2 * T + i / 32) * 128 + vbase + i % 32];
    }
  }
  #pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    #pragma unroll
    for (int x = 0; x < 4; ++x) {
      float4& s = state_values[pass][x];
      s.x = operand(s.x, CompatTf32 || p.tf32);
      s.y = operand(s.y, CompatTf32 || p.tf32);
      s.z = operand(s.z, CompatTf32 || p.tf32);
      s.w = operand(s.w, CompatTf32 || p.tf32);
    }
  }
  __syncthreads();

  #pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    float dots[2 * T] = {};
    #pragma unroll
    for (int x = 0; x < 4; ++x) {
      const float4 s = state_values[pass][x];
      #pragma unroll
      for (int r = 0; r < 2 * T; ++r) {
        // W/Qeff already combine RTZ operands; only state is truncated here.
        const float4 w = vectors[r][group_lane + x * 8];
        dots[r] = fmaf(s.w, w.w, fmaf(s.z, w.z, fmaf(s.y, w.y, fmaf(s.x, w.x, dots[r]))));
      }
    }
    #pragma unroll
    for (int r = 0; r < 2 * T; ++r) {
      float dot = dots[r];
      dot += __shfl_xor_sync(0xffffffffu, dot, 4);
      dot = sum_row(dot);
      if (group_lane == 0) {
        const int local_v = pass * 16 + group;
        const float u_or_bias = results[r][local_v];
        results[r][local_v] = r < T ? u_or_bias - dot : u_or_bias + dot;
      }
    }
  }
  __syncthreads();

  for (int i = tid; i < T * 32; i += 128) {
    const int t = i / 32, local_v = i % 32, v = vbase + local_v;
    if (t < n) {
      const float update = results[t][local_v];
      const float hi = rounded<BF>(update);
      const int phys = (base + wp + t) & (capacity - 1);
      const int64_t off = (hv * capacity + phys) * int64_t(128) + v;
      write<BF>(p.out, (bos + t) * p.out.stride[0] + hv * 128 + v, results[T + t][local_v]);
      write<BF>(p.d, replay * p.d.stride[0] + off, hi);
      if (exact) {
        const float rawv = read<BF>(p.v, (bos + t) * p.v.stride[0] + hv * 128 + v);
        write<BF>(p.rawv, replay * p.rawv.stride[0] + off, rawv);
      } else if (p.residual) {
        write<BF>(p.rawv, replay * p.rawv.stride[0] + off, update - hi);
      }
    }
  }
}

template <int T>
inline void launch_project_verify_steps(const VerifyLite& p, hggcStream_t stream,
                                        bool sglang_compat = false) {
  const size_t shared = sizeof(float) *
      (2 * T * 128 + 2 * T * T + 2 * p.capacity * T + p.capacity + 5 * T + 1);
  const bool i32 = p.slots.dtype == I32 && p.replay.dtype == I32 && p.wp.dtype == I32 &&
                   p.base.dtype == I32 && (!p.starts.data || p.starts.dtype == I32);
  const dim3 prepare_grid(1, p.batch, p.value_heads), project_grid(4, p.batch, p.value_heads);
  const int64_t head_work = int64_t(p.batch) * p.value_heads;
  if constexpr (T == 4) {
    // Specialize the SGLang TF32 tile for its raw gates, circular capacity, and flush type.
    if (sglang_compat && i32 && p.tf32 && !p.exact && !p.processed_gates && p.capacity == 16 &&
        p.batch <= 32 && head_work >= 192 && head_work <= 256 && p.flush.dtype == I8 &&
        p.a.dtype == BF16 && p.b.dtype == BF16 && p.A_log.dtype == F32 && p.dt_bias.dtype == F32) {
      project_verify_prepare_kernel<4, 16, true, true><<<prepare_grid, 128, shared, stream>>>(p);
      project_verify_project_kernel<4, true, true, true><<<project_grid, 128, 0, stream>>>(p);
      return;
    }
  }
  // Few heads are latency bound and prefetch everything; many heads are occupancy bound.
  const bool eager = head_work < 512;
  if (i32) {
    if (!eager)
      project_verify_prepare_lean_kernel<T, true><<<prepare_grid, 128, shared, stream>>>(p);
    else if (p.capacity <= 16)
      project_verify_prepare_kernel<T, 16, true><<<prepare_grid, 128, shared, stream>>>(p);
    else
      project_verify_prepare_kernel<T, 32, true><<<prepare_grid, 128, shared, stream>>>(p);
    if (eager) project_verify_project_kernel<T, true, true><<<project_grid, 128, 0, stream>>>(p);
    else project_verify_project_kernel<T, true, false><<<project_grid, 128, 0, stream>>>(p);
  } else {
    if (!eager)
      project_verify_prepare_lean_kernel<T, false><<<prepare_grid, 128, shared, stream>>>(p);
    else if (p.capacity <= 16)
      project_verify_prepare_kernel<T, 16, false><<<prepare_grid, 128, shared, stream>>>(p);
    else
      project_verify_prepare_kernel<T, 32, false><<<prepare_grid, 128, shared, stream>>>(p);
    if (eager) project_verify_project_kernel<T, false, true><<<project_grid, 128, 0, stream>>>(p);
    else project_verify_project_kernel<T, false, false><<<project_grid, 128, 0, stream>>>(p);
  }
}

inline bool launch_project_verify(const Verify& v, hggcStream_t stream) {
  if (!v.workspace) return false;
  assert(v.mode == 1 && v.key_dim == 128 && v.value_dim == 128 &&
         v.capacity > 0 && v.capacity <= 32 && (v.capacity & (v.capacity - 1)) == 0 &&
         v.q.dtype == BF16 && v.state.dtype == F32 && v.d.dtype == BF16);
  // Unaligned state rows or workspace use the single-kernel path instead.
  if ((reinterpret_cast<uintptr_t>(v.state.data) & 15) != 0 || v.state.stride[0] % 4 != 0 ||
      (reinterpret_cast<uintptr_t>(v.workspace) & 15) != 0)
    return false;
  const VerifyLite p = verify_lite(v);
  switch (p.max_steps) {
    case 1: launch_project_verify_steps<1>(p, stream); break;
    case 2: launch_project_verify_steps<2>(p, stream); break;
    case 3: launch_project_verify_steps<3>(p, stream); break;
    case 4: launch_project_verify_steps<4>(p, stream, v.legacy_dot_precision); break;
    case 6: launch_project_verify_steps<6>(p, stream); break;
    case 8: launch_project_verify_steps<8>(p, stream); break;
    default: throw std::invalid_argument("GDN replay project supports max_steps 1, 2, 3, 4, 6 or 8");
  }
  return true;
}
