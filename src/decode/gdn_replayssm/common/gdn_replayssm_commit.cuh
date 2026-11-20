// SPDX-License-Identifier: Apache-2.0
#pragma once

constexpr int FAST_COMMIT_THREADS = 128;

// Metadata read by the early-exit path comes first so it shares the leading constant lines.
struct CommitLite {
  int state_slots, replay_slots, null_slot;
  bool residual, extended_base;
  LiteTensor slots, replay, flush, wp, base, state, d, keys, g, dr, kr;
  int heads, value_heads, value_dim, capacity;
};
static_assert(sizeof(CommitLite) == 384);

constexpr int FAST_COMMIT_K = 128;

// The update retains both cache-precision roundings before the FP32 accumulation.
__device__ __forceinline__ void fast_commit_accumulate4(float4& delta, const float4& key, float update) {
  delta.x = fmaf(key.x, update, delta.x);
  delta.y = fmaf(key.y, update, delta.y);
  delta.z = fmaf(key.z, update, delta.z);
  delta.w = fmaf(key.w, update, delta.w);
}

__device__ __forceinline__ float* fast_commit_state(const CommitLite& p, int layer, int64_t slot,
                                                   int hv, int v, int k) {
  return static_cast<float*>(p.state.data) + layer * p.state.stride[0] + slot * p.state.stride[1] +
         (hv * int64_t(p.value_dim) + v) * FAST_COMMIT_K + k;
}

template <class C, bool WIDE = false>
__device__ __forceinline__ void fast_commit_load_vector(const LiteTensor& t, int64_t offset,
                                                        float (&out)[16 / sizeof(C)]) {
  const C* source = static_cast<const C*>(t.data) + offset;
  uint4 raw;
  if constexpr (WIDE) {
    // Keep the aligned 16-byte transfer intact through unpacking.
    asm volatile("ppu.ld.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w) : "l"(source));
  } else {
    raw = *reinterpret_cast<const uint4*>(source);
  }
  const C* elements = reinterpret_cast<const C*>(&raw);
  #pragma unroll
  for (int i = 0; i < int(16 / sizeof(C)); ++i) out[i] = static_cast<float>(elements[i]);
}

// Three dependent load levels: row indices, per-row cursors, then state plus the whole ring.
template <bool I8_FLAG, class TT>
__device__ __forceinline__ bool fold_enabled(const TT& t, int64_t replay) {
  if (!t.data) return true;
  if constexpr (I8_FLAG) return static_cast<const int8_t*>(t.data)[replay * t.stride[0]] != 0;
  else return index(t, replay) != 0;
}

template <class C, int MAXC, bool I32, int TILE, bool I8_FLAG = false, bool WIDE = false>
__global__ __launch_bounds__(FAST_COMMIT_THREADS) void fast_commit_kernel(CommitLite p) {
  constexpr int TILE_V = TILE;
  static_assert(FAST_COMMIT_K == FAST_COMMIT_THREADS, "one key element per thread");
  const int row = blockIdx.y, layer = blockIdx.z / p.value_heads;
  const int hv = blockIdx.z % p.value_heads, h = hv / (p.value_heads / p.heads);
  const int64_t slot = meta_index<I32>(p.slots, row), replay = meta_index<I32>(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  if (!fold_enabled<I8_FLAG>(p.flush, replay)) return;
  const int64_t n_history = meta_index<I32>(p.wp, replay);
  assert(n_history >= 0 && n_history <= p.capacity);
  const int base = physical_base(meta_index<I32>(p.base, replay), p.capacity, p.extended_base);
  const int count = static_cast<int>(n_history), n = count;
  if (count == 0) return;
  const int tid = threadIdx.x;
  const int vbase = static_cast<int>(int64_t(blockIdx.x) * TILE_V);

  extern __shared__ float4 fast_commit_mem[];
  float* keys = reinterpret_cast<float*>(fast_commit_mem);
  float* key_low = p.residual ? keys + p.capacity * FAST_COMMIT_K : nullptr;
  float* fold_hi = keys + (p.residual ? 2 : 1) * p.capacity * FAST_COMMIT_K;
  float* fold_low = p.residual ? fold_hi + p.capacity * TILE_V : nullptr;
  float* total_decay = fold_hi + (p.residual ? 2 : 1) * p.capacity * TILE_V;

  // Unconditional aligned prefetches keep loads in flight without a branch-merge drain.
  constexpr int BUNDLE = 4;
  const int k4 = (tid % 32) * 4;
  float4 prefetched[BUNDLE];
  #pragma unroll
  for (int i = 0; i < BUNDLE; ++i)
    prefetched[i] = *reinterpret_cast<const float4*>(
        fast_commit_state(p, layer, slot, hv, vbase + tid / 32 + i * 4, k4));
  constexpr int E = 16 / sizeof(C);
  constexpr int KEY_THREADS = FAST_COMMIT_K / E, KEY_ROWS = FAST_COMMIT_THREADS / KEY_THREADS;
  constexpr int KEY_LOADS = (MAXC + KEY_ROWS - 1) / KEY_ROWS;
  const int key_row = tid / KEY_THREADS, key_col = (tid % KEY_THREADS) * E;
  const LiteTensor& key_residual = p.residual ? p.kr : p.keys;
  const LiteTensor& value_residual = p.residual ? p.dr : p.d;
  float key_values[KEY_LOADS][E], key_extra[KEY_LOADS][E];
  #pragma unroll
  for (int i = 0; i < KEY_LOADS; ++i) {
    const int phys = static_cast<int>((base + i * KEY_ROWS + key_row) & (p.capacity - 1));
    fast_commit_load_vector<C, WIDE>(p.keys, ring_offset(p.keys, layer, replay, h, phys, FAST_COMMIT_K,
                                                   key_col, p.capacity), key_values[i]);
    fast_commit_load_vector<C, WIDE>(key_residual, ring_offset(key_residual, layer, replay, h, phys,
                                                         FAST_COMMIT_K, key_col, p.capacity), key_extra[i]);
  }
  constexpr int UPDATE_THREADS = TILE_V / E, UPDATE_ROWS = FAST_COMMIT_THREADS / UPDATE_THREADS;
  constexpr int UPDATE_LOADS = (MAXC + UPDATE_ROWS - 1) / UPDATE_ROWS;
  const int update_row = tid / UPDATE_THREADS, update_col = (tid % UPDATE_THREADS) * E;
  float updates[UPDATE_LOADS][E], update_extra[UPDATE_LOADS][E];
  #pragma unroll
  for (int i = 0; i < UPDATE_LOADS; ++i) {
    const int phys = static_cast<int>((base + i * UPDATE_ROWS + update_row) & (p.capacity - 1));
    fast_commit_load_vector<C, WIDE>(p.d, ring_offset(p.d, layer, replay, hv, phys, p.value_dim,
                                                vbase + update_col, p.capacity), updates[i]);
    fast_commit_load_vector<C, WIDE>(value_residual, ring_offset(value_residual, layer, replay, hv, phys,
                                                           p.value_dim, vbase + update_col, p.capacity),
                               update_extra[i]);
  }
  const int64_t head_g = layer * p.g.stride[0] + replay * p.g.stride[1] + int64_t(hv) * p.capacity;
  const int lane = tid & 31, t_gate = count - 1 - lane;
  const float loaded_gate = read<float>(p.g, head_g + ((static_cast<int>(base) + t_gate) & (p.capacity - 1)));
  const float prefix = warp_inclusive_sum(t_gate >= 0 ? loaded_gate : 0.f, lane);
  const float previous = __shfl_up_sync(0xffffffffu, prefix, 1);
  const float total = __shfl_sync(0xffffffffu, prefix, 31);
  const float lane_decay = t_gate >= 0 ? expf(lane == 0 ? 0.f : previous) : 0.f;
  if (tid == 0) *total_decay = expf(total);
  #pragma unroll
  for (int i = 0; i < KEY_LOADS; ++i) {
    const int t = i * KEY_ROWS + key_row;
    if (t < count) {
      #pragma unroll
      for (int e = 0; e < E; ++e) {
        const int offset = t * FAST_COMMIT_K + key_col + e;
        keys[offset] = key_values[i][e];
        if (p.residual) key_low[offset] = key_extra[i][e];
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < UPDATE_LOADS; ++i) {
    const int t = i * UPDATE_ROWS + update_row;
    const float decay_t = __shfl_sync(0xffffffffu, lane_decay, (n - 1 - t) & 31);
    if (t < count) {
      #pragma unroll
      for (int e = 0; e < E; ++e) {
        const int offset = t * TILE_V + update_col + e;
        fold_hi[offset] = scaled_fold_value<C>(updates[i][e], decay_t);
        if (p.residual) fold_low[offset] = scaled_fold_value<C>(update_extra[i][e], decay_t);
      }
    }
  }
  __syncthreads();

  if (p.residual) {
    for (int v = tid / 32; v < TILE_V; v += 4) {
      float4* target = reinterpret_cast<float4*>(fast_commit_state(p, layer, slot, hv, vbase + v, k4));
      const float4 old = *target;
      float4 delta{}, v_low{}, k_low{}, cross{};
      for (int t = 0; t < count; ++t) {
        const float4 key = *reinterpret_cast<const float4*>(keys + t * FAST_COMMIT_K + k4);
        const float4 extra_key = *reinterpret_cast<const float4*>(key_low + t * FAST_COMMIT_K + k4);
        const float value = fold_hi[t * TILE_V + v], extra_value = fold_low[t * TILE_V + v];
        fast_commit_accumulate4(delta, key, value);
        fast_commit_accumulate4(v_low, key, extra_value);
        fast_commit_accumulate4(k_low, extra_key, value);
        fast_commit_accumulate4(cross, extra_key, extra_value);
      }
      const float decay = *total_decay;
      *target = make_float4((old.x * decay + delta.x) + ((v_low.x + k_low.x) + cross.x),
                            (old.y * decay + delta.y) + ((v_low.y + k_low.y) + cross.y),
                            (old.z * decay + delta.z) + ((v_low.z + k_low.z) + cross.z),
                            (old.w * decay + delta.w) + ((v_low.w + k_low.w) + cross.w));
    }
    return;
  }

  for (int pass = 0; pass < TILE_V / 4; pass += BUNDLE) {
    float4 delta[BUNDLE] = {};
    #pragma unroll 4
    for (int t = 0; t < count; ++t) {
      const float4 key = *reinterpret_cast<const float4*>(keys + t * FAST_COMMIT_K + k4);
      #pragma unroll
      for (int i = 0; i < BUNDLE; ++i) {
        const int uo = t * TILE_V + tid / 32 + (pass + i) * 4;
        fast_commit_accumulate4(delta[i], key, fold_hi[uo]);
      }
    }
    float4 next[BUNDLE];
    #pragma unroll
    for (int i = 0; i < BUNDLE; ++i) {
      const int v = vbase + tid / 32 + (pass + BUNDLE + i) * 4;
      next[i] = pass + BUNDLE < TILE_V / 4 ?
          *reinterpret_cast<const float4*>(fast_commit_state(p, layer, slot, hv, v, k4)) :
          make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const float decay = *total_decay;
    #pragma unroll
    for (int i = 0; i < BUNDLE; ++i) {
      const int v = vbase + tid / 32 + (pass + i) * 4;
      *reinterpret_cast<float4*>(fast_commit_state(p, layer, slot, hv, v, k4)) =
          make_float4(prefetched[i].x * decay + delta[i].x, prefetched[i].y * decay + delta[i].y,
                      prefetched[i].z * decay + delta[i].z, prefetched[i].w * decay + delta[i].w);
      prefetched[i] = next[i];
    }
  }

}

template <int TILE_V, bool I32, bool I8_FLAG = false>
__global__ __launch_bounds__(128) void tensor_commit_kernel(CommitLite p) {
  constexpr int K = 128, CAP = 16;
  static_assert((2 * CAP * K + 4 * CAP * TILE_V) * sizeof(BF) +
                TILE_V * K * sizeof(float) <= 48 * 1024);
  const int row = blockIdx.y, hv = blockIdx.z % p.value_heads;
  const int layer = blockIdx.z / p.value_heads, h = hv / (p.value_heads / p.heads);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t slot = meta_index<I32>(p.slots, row), replay = meta_index<I32>(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  if (!fold_enabled<I8_FLAG>(p.flush, replay)) return;
  const int64_t count_raw = meta_index<I32>(p.wp, replay);
  assert(count_raw >= 0 && count_raw <= p.capacity);
  const int count = static_cast<int>(count_raw);
  const int base = physical_base(meta_index<I32>(p.base, replay), p.capacity, p.extended_base);
  if (count == 0) return;
  const int vbase = blockIdx.x * TILE_V;
  __shared__ __align__(32) BF keys[CAP * K], key_residual[CAP * K];
  __shared__ __align__(32) BF hi[CAP * TILE_V], lo[CAP * TILE_V];
  __shared__ __align__(32) BF dr_hi[CAP * TILE_V], dr_lo[CAP * TILE_V];
  // Each warp replaces its own accumulator tile after reading the checkpoint.
  __shared__ __align__(32) float state[TILE_V * K];
  #pragma unroll
  for (int i = 0; i < TILE_V / 4; ++i) {
    const int v = warp + i * 4, k4 = lane * 4;
    *reinterpret_cast<float4*>(state + v * K + k4) =
        *reinterpret_cast<const float4*>(fast_commit_state(p, layer, slot, hv, vbase + v, k4));
  }
  const int t_gate = count - 1 - lane;
  const int gate_pos = (base + t_gate) & (p.capacity - 1);
  const float loaded_gate = read<float>(p.g, layer * p.g.stride[0] + replay * p.g.stride[1] +
                                        hv * p.capacity + gate_pos);
  const float prefix = warp_inclusive_sum(t_gate >= 0 ? loaded_gate : 0.f, lane);
  const float previous = __shfl_up_sync(0xffffffffu, prefix, 1);
  const float weight = t_gate >= 0 ? expf(lane == 0 ? 0.f : previous) : 0.f;
  const float total_decay = expf(__shfl_sync(0xffffffffu, prefix, 31));
  const uint4 zero = make_uint4(0, 0, 0, 0);
  #pragma unroll
  for (int i = 0; i < CAP * K / (8 * 128); ++i) {
    const int x = tid + i * 128, t = x / (K / 8), k = x % (K / 8) * 8;
    const int phys = (base + t) & (p.capacity - 1);
    const int off = t * K + k;
    uint4 key = zero, extra = zero;
    if (t < count) {
      const int64_t src = ring_offset(p.keys, layer, replay, h, phys, K, k, p.capacity);
      key = *reinterpret_cast<const uint4*>(static_cast<const BF*>(p.keys.data) + src);
      if (p.residual) {
        const int64_t residual_src = ring_offset(p.kr, layer, replay, h, phys, K, k, p.capacity);
        extra = *reinterpret_cast<const uint4*>(static_cast<const BF*>(p.kr.data) + residual_src);
      }
    }
    *reinterpret_cast<uint4*>(keys + off) = key;
    *reinterpret_cast<uint4*>(key_residual + off) = extra;
  }
  #pragma unroll
  for (int i = 0; i < CAP * TILE_V / 128; ++i) {
    const int x = tid + i * 128, t = x / TILE_V, v = vbase + x % TILE_V;
    const int phys = (base + t) & (p.capacity - 1);
    const float update = read<BF>(p.d, ring_offset(p.d, layer, replay, hv, phys, 128, v, p.capacity));
    const float decay = __shfl_sync(0xffffffffu, weight, (count - 1 - t) & 31);
    const float scaled = t < count ? update * decay : 0.f;
    hi[x] = BF(scaled);
    lo[x] = BF(scaled - float(hi[x]));
    if (p.residual) {
      const float extra = read<BF>(p.dr, ring_offset(p.dr, layer, replay, hv, phys, 128, v, p.capacity));
      const float scaled_extra = t < count ? extra * decay : 0.f;
      dr_hi[x] = BF(scaled_extra);
      dr_lo[x] = BF(scaled_extra - float(dr_hi[x]));
    }
  }
  __syncthreads();
  for (int tile = warp; tile < (TILE_V / 16) * (K / 16); tile += 4) {
    const int vtile = tile / (K / 16) * 16, ktile = tile % (K / 16) * 16;
    awmma::fragment<awmma::matrix_a, 16, 16, 16, BF, awmma::col_major> a_hi, a_lo;
    awmma::fragment<awmma::matrix_b, 16, 16, 16, BF, awmma::row_major> b_hi, b_lo;
    awmma::fragment<awmma::accumulator, 16, 16, 16, float> delta, correction;
    awmma::fill_fragment(delta, 0.f);
    awmma::fill_fragment(correction, 0.f);
    awmma::load_matrix_sync(a_hi, hi + vtile, TILE_V);
    awmma::load_matrix_sync(a_lo, lo + vtile, TILE_V);
    awmma::load_matrix_sync(b_hi, keys + ktile, K);
    awmma::mma_sync(delta, a_hi, b_hi, delta);
    awmma::mma_sync(delta, a_lo, b_hi, delta);
    if (p.residual) {
      awmma::load_matrix_sync(b_lo, key_residual + ktile, K);
      awmma::mma_sync(correction, a_hi, b_lo, correction);
      awmma::mma_sync(correction, a_lo, b_lo, correction);
      decltype(correction) value_low, cross;
      awmma::fill_fragment(value_low, 0.f);
      awmma::fill_fragment(cross, 0.f);
      awmma::load_matrix_sync(a_hi, dr_hi + vtile, TILE_V);
      awmma::load_matrix_sync(a_lo, dr_lo + vtile, TILE_V);
      awmma::mma_sync(value_low, a_hi, b_hi, value_low);
      awmma::mma_sync(value_low, a_lo, b_hi, value_low);
      awmma::mma_sync(cross, a_hi, b_lo, cross);
      awmma::mma_sync(cross, a_lo, b_lo, cross);
      #pragma unroll
      for (int e = 0; e < decltype(correction)::num_elements; ++e)
        correction.x[e] = (correction.x[e] + value_low.x[e]) + cross.x[e];
    }
    float* tile_state = state + vtile * K + ktile;
    decltype(delta) checkpoint;
    awmma::load_matrix_sync(checkpoint, tile_state, K, awmma::mem_row_major);
    #pragma unroll
    for (int e = 0; e < decltype(checkpoint)::num_elements; ++e)
      checkpoint.x[e] = (checkpoint.x[e] * total_decay + delta.x[e]) + correction.x[e];
    awmma::store_matrix_sync(tile_state, checkpoint, K, awmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < TILE_V / 4; ++i) {
    const int v = warp + i * 4, off = v * K + lane * 4;
    *reinterpret_cast<float4*>(fast_commit_state(p, layer, slot, hv, vbase + v, lane * 4)) =
        *reinterpret_cast<const float4*>(state + off);
  }
}

template <int TILE_V>
inline void launch_tensor_commit(const CommitLite& p, bool i32, dim3 grid, hggcStream_t stream) {
  grid.x = p.value_dim / TILE_V;
  if constexpr (TILE_V > 16) {
    tensor_commit_kernel<TILE_V, true, true><<<grid, 128, 0, stream>>>(p);
  } else if (i32 && p.flush.dtype == I8) {
    tensor_commit_kernel<TILE_V, true, true><<<grid, 128, 0, stream>>>(p);
  } else if (i32) {
    tensor_commit_kernel<TILE_V, true><<<grid, 128, 0, stream>>>(p);
  } else {
    tensor_commit_kernel<TILE_V, false><<<grid, 128, 0, stream>>>(p);
  }
}

template <int MAXC, bool I32, int TILE>
inline void launch_fast_commit_typed(const CommitLite& p, dim3 grid, size_t shared, hggcStream_t stream) {
  if (p.d.dtype == F32)
    fast_commit_kernel<float, MAXC, I32, TILE><<<grid, FAST_COMMIT_THREADS, shared, stream>>>(p);
  else if (p.d.dtype == F16)
    fast_commit_kernel<HF, MAXC, I32, TILE><<<grid, FAST_COMMIT_THREADS, shared, stream>>>(p);
  else
    fast_commit_kernel<BF, MAXC, I32, TILE><<<grid, FAST_COMMIT_THREADS, shared, stream>>>(p);
}

template <bool I32, int TILE>
inline void launch_fast_commit_capacity(const CommitLite& p, dim3 grid, size_t shared,
                                        hggcStream_t stream) {
  if (p.capacity <= 16) launch_fast_commit_typed<16, I32, TILE>(p, grid, shared, stream);
  else launch_fast_commit_typed<32, I32, TILE>(p, grid, shared, stream);
}
