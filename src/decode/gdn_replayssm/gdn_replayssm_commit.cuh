// SPDX-License-Identifier: Apache-2.0
#pragma once

constexpr int FAST_COMMIT_THREADS = 128;

// Metadata read by the early-exit path comes first so it shares the leading constant lines.
struct CommitLite {
  int state_slots, replay_slots, null_slot;
  bool residual;
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
  const int64_t n_history = meta_index<I32>(p.wp, replay), base = meta_index<I32>(p.base, replay);
  assert(n_history >= 0 && n_history <= p.capacity && base >= 0 && base < p.capacity);
  const int count = static_cast<int>(n_history), n = count;
  if (count == 0) return;
  const int tid = threadIdx.x;
  const int vbase = static_cast<int>(int64_t(blockIdx.x) * TILE_V);

  extern __shared__ float4 fast_commit_mem[];
  float* keys = reinterpret_cast<float*>(fast_commit_mem);
  float* key_low = p.residual ? keys + p.capacity * FAST_COMMIT_K : nullptr;
  float* fold_hi = keys + (p.residual ? 2 : 1) * p.capacity * FAST_COMMIT_K;
  float* totals = fold_hi + p.capacity * TILE_V;

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
  if (tid == 0) totals[0] = expf(total);
  int key_roundoff = 0;
  #pragma unroll
  for (int i = 0; i < KEY_LOADS; ++i) {
    const int t = i * KEY_ROWS + key_row;
    if (t < count) {
      #pragma unroll
      for (int e = 0; e < E; ++e) {
        const float upper = key_values[i][e];
        float merged = upper, error = 0.f;
        if (p.residual) {
          const float lower = key_extra[i][e];
          merged = __fadd_rn(upper, lower);
          if ((__float_as_uint(merged) & 0x7f800000u) != 0x7f800000u) {
            const float part = __fsub_rn(merged, upper);
            error = __fadd_rn(__fsub_rn(upper, __fsub_rn(merged, part)), __fsub_rn(lower, part));
          } else {
            merged = upper;
            error = lower;
          }
          key_low[t * FAST_COMMIT_K + key_col + e] = error;
          key_roundoff |= error != 0.f;
        }
        keys[t * FAST_COMMIT_K + key_col + e] = merged;
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
        const float update = p.residual ? updates[i][e] + update_extra[i][e] : updates[i][e];
        const float scaled = update * decay_t;
        const float hi = rounded<C>(scaled);
        fold_hi[t * TILE_V + update_col + e] = hi + rounded<C>(scaled - hi);
      }
    }
  }
  const bool compensate = __syncthreads_or(key_roundoff) != 0;

  {
    for (int pass = 0; pass < TILE_V / 4; pass += BUNDLE) {
      float4 delta[BUNDLE], correction[BUNDLE];
      #pragma unroll
      for (int i = 0; i < BUNDLE; ++i) {
        delta[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        correction[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      if (compensate) {
        for (int t = 0; t < count; ++t) {
          const float4 key = *reinterpret_cast<const float4*>(keys + t * FAST_COMMIT_K + k4);
          const float4 low = *reinterpret_cast<const float4*>(key_low + t * FAST_COMMIT_K + k4);
          #pragma unroll
          for (int i = 0; i < BUNDLE; ++i) {
            const int uo = t * TILE_V + tid / 32 + (pass + i) * 4;
            fast_commit_accumulate4(delta[i], key, fold_hi[uo]);
            fast_commit_accumulate4(correction[i], low, fold_hi[uo]);
          }
        }
      } else {
        #pragma unroll 4
        for (int t = 0; t < count; ++t) {
          const float4 key = *reinterpret_cast<const float4*>(keys + t * FAST_COMMIT_K + k4);
          #pragma unroll
          for (int i = 0; i < BUNDLE; ++i) {
            const int uo = t * TILE_V + tid / 32 + (pass + i) * 4;
            fast_commit_accumulate4(delta[i], key, fold_hi[uo]);
          }
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
      const float total = totals[0];
      #pragma unroll
      for (int i = 0; i < BUNDLE; ++i) {
        const int v = vbase + tid / 32 + (pass + i) * 4;
        *reinterpret_cast<float4*>(fast_commit_state(p, layer, slot, hv, v, k4)) =
            make_float4(prefetched[i].x * total + (delta[i].x + correction[i].x),
                        prefetched[i].y * total + (delta[i].y + correction[i].y),
                        prefetched[i].z * total + (delta[i].z + correction[i].z),
                        prefetched[i].w * total + (delta[i].w + correction[i].w));
        prefetched[i] = next[i];
      }
    }
  }
}

template <bool I32, bool I8_FLAG = false>
__global__ __launch_bounds__(128) void tensor_commit_kernel(CommitLite p) {
  constexpr int TILE_V = 16, K = 128, CAP = 16;
  const int row = blockIdx.y, hv = blockIdx.z % p.value_heads;
  const int layer = blockIdx.z / p.value_heads, h = hv / (p.value_heads / p.heads);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t slot = meta_index<I32>(p.slots, row), replay = meta_index<I32>(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  if (!fold_enabled<I8_FLAG>(p.flush, replay)) return;
  const int64_t count_raw = meta_index<I32>(p.wp, replay), base_raw = meta_index<I32>(p.base, replay);
  assert(count_raw >= 0 && count_raw <= p.capacity && base_raw >= 0 && base_raw < p.capacity);
  const int count = static_cast<int>(count_raw), base = static_cast<int>(base_raw);
  if (count == 0) return;
  const int vbase = blockIdx.x * TILE_V;
  float4 old_state[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    old_state[i] = *reinterpret_cast<const float4*>(fast_commit_state(p, layer, slot, hv,
                                                  vbase + warp + i * 4, lane * 4));
  __shared__ __align__(32) BF keys[CAP * K], key_residual[CAP * K];
  __shared__ __align__(32) BF hi[CAP * TILE_V], lo[CAP * TILE_V];
  __shared__ __align__(32) float main_delta[TILE_V * K], residual_delta[TILE_V * K];
  __shared__ float total_decay;
  const int t_gate = count - 1 - lane;
  const int gate_pos = (base + t_gate) & (p.capacity - 1);
  const float loaded_gate = read<float>(p.g, layer * p.g.stride[0] + replay * p.g.stride[1] +
                                        hv * p.capacity + gate_pos);
  const float prefix = warp_inclusive_sum(t_gate >= 0 ? loaded_gate : 0.f, lane);
  const float previous = __shfl_up_sync(0xffffffffu, prefix, 1);
  const float weight = t_gate >= 0 ? expf(lane == 0 ? 0.f : previous) : 0.f;
  const float total = __shfl_sync(0xffffffffu, prefix, 31);
  if (tid == 0) total_decay = expf(total);
  const LiteTensor& residual_keys = p.residual ? p.kr : p.keys;
  for (int i = 0; i < CAP * K / (8 * 128); ++i) {
    const int x = tid + i * 128, t = x / (K / 8), k = x % (K / 8) * 8;
    const int phys = (base + t) & (p.capacity - 1);
    float key[8], extra[8];
    fast_commit_load_vector<BF>(p.keys, ring_offset(p.keys, layer, replay, h, phys, K, k,
                                                  p.capacity), key);
    fast_commit_load_vector<BF>(residual_keys, ring_offset(residual_keys, layer, replay, h,
                                   phys, K, k, p.capacity), extra);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      keys[t * K + k + e] = BF(t < count ? key[e] : 0.f);
      key_residual[t * K + k + e] = BF(t < count && p.residual ? extra[e] : 0.f);
    }
  }
    #pragma unroll
    for (int i = 0; i < CAP * TILE_V / 128; ++i) {
      const int x = tid + i * 128, t = x / TILE_V, v = vbase + x % TILE_V;
      const int phys = (base + t) & (p.capacity - 1);
      float update = read<BF>(p.d, ring_offset(p.d, layer, replay, hv, phys, 128, v, p.capacity));
      if (p.residual)
        update += read<BF>(p.dr, ring_offset(p.dr, layer, replay, hv, phys, 128, v, p.capacity));
      const float decay = __shfl_sync(0xffffffffu, weight, (count - 1 - t) & 31);
      const float scaled = t < count ? update * decay : 0.f;
      const BF upper = BF(scaled);
      hi[x] = upper;
      lo[x] = BF(scaled - float(upper));
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
    }
    awmma::store_matrix_sync(main_delta + vtile * K + ktile, delta, K, awmma::mem_row_major);
    awmma::store_matrix_sync(residual_delta + vtile * K + ktile, correction, K, awmma::mem_row_major);
  }
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int v = warp + i * 4, off = v * K + lane * 4;
    const float4 delta = *reinterpret_cast<const float4*>(main_delta + off);
    const float4 correction = *reinterpret_cast<const float4*>(residual_delta + off);
    *reinterpret_cast<float4*>(fast_commit_state(p, layer, slot, hv, vbase + v, lane * 4)) =
        make_float4(old_state[i].x * total_decay + (delta.x + correction.x),
                    old_state[i].y * total_decay + (delta.y + correction.y),
                    old_state[i].z * total_decay + (delta.z + correction.z),
                    old_state[i].w * total_decay + (delta.w + correction.w));
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

inline bool launch_fast_commit(const Commit& p, hggcStream_t stream) {
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
  const int tile_v = head_work <= 32 ? 16 : wide ? 128 : 64;
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
      (size_t(p.capacity) * ((p.residual ? 2 : 1) * FAST_COMMIT_K + tile_v + 1) + 2);
  if (shared > 48 * 1024) return false;
  const dim3 grid(static_cast<unsigned int>((int64_t(p.value_dim) + tile_v - 1) / tile_v),
                  p.batch, static_cast<unsigned int>(grid_z));
  const bool i32 = p.slots.dtype == I32 && p.replay.dtype == I32 && p.wp.dtype == I32 &&
                   p.base.dtype == I32;
  const CommitLite q{p.state_slots, p.replay_slots, p.null_slot, p.residual,
                     lite(p.slots), lite(p.replay), lite(p.flush), lite(p.wp), lite(p.base),
                     lite(p.state), lite(p.d), lite(p.keys), lite(p.g), lite(p.dr), lite(p.kr),
                     p.heads, p.value_heads, p.value_dim, p.capacity};
  if (p.d.dtype == BF16 && p.capacity <= 16 && p.value_dim == 128 &&
      int64_t(p.batch) * p.value_heads * p.layers <= 32) {
    const dim3 tensor_grid(8, p.batch, static_cast<unsigned int>(grid_z));
    if (i32 && p.flush.dtype == I8)
      tensor_commit_kernel<true, true><<<tensor_grid, 128, 0, stream>>>(q);
    else if (i32) tensor_commit_kernel<true><<<tensor_grid, 128, 0, stream>>>(q);
    else tensor_commit_kernel<false><<<tensor_grid, 128, 0, stream>>>(q);
    return true;
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
