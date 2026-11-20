// SPDX-License-Identifier: Apache-2.0
#pragma once

struct MaterializeRange {
  int64_t source, replay, target, start, count, buffer;
  bool active;
};

__device__ __forceinline__ int64_t work_item(const Tensor& t, int row, int column) {
  const int64_t offset = row * t.stride[0] + column * t.stride[1];
  return t.dtype == I64 ? static_cast<const int64_t*>(t.data)[offset] :
                          static_cast<const int32_t*>(t.data)[offset];
}

__device__ __forceinline__ MaterializeRange materialize_range(const Materialize& p, int row) {
  MaterializeRange w{};
  if (p.n_items.data) {
    const int64_t count = index(p.n_items, 0);
    assert(count >= 0 && count <= p.batch);
    if (row >= count) return w;
  }
  if (p.items.data) {
    w.source = w.replay = work_item(p.items, row, 1);
    w.count = work_item(p.items, row, 2);
    w.buffer = work_item(p.items, row, 3);
  } else {
    w.source = index(p.state_indices, row);
    if (w.source <= p.null_slot) return w;
    w.replay = index(p.replay_indices, row);
    const int64_t metadata = p.metadata_indices.data ? index(p.metadata_indices, row) : row;
    assert(metadata >= 0 && metadata < p.metadata_slots);
    if (p.enabled.data && !index(p.enabled, metadata)) return w;
    w.count = index(p.lengths, metadata);
    w.start = p.starts.data ? index(p.starts, metadata) : 0;
    w.buffer = p.buffers.data ? index(p.buffers, metadata) : 0;
  }
  if (w.source <= p.null_slot) return w;
  w.target = p.targets.data ? index(p.targets, row) : w.source;
  if (w.target <= p.null_slot) return w;
  assert(w.source >= 0 && w.source < p.state_slots && w.target >= 0 && w.target < p.state_slots);
  assert(w.replay >= 0 && w.replay < p.replay_slots);
  assert(w.start >= 0 && w.start < p.capacity && w.count >= 0 && w.count <= p.capacity);
  assert(p.format == CompactDKG || w.start + w.count <= p.capacity);
  assert(p.format == CachedUKG ? w.buffer >= 0 && w.buffer < 2 : w.buffer == 0);
  if (p.overflow_steps > 0 && w.count + p.overflow_steps <= p.capacity) return w;
  w.active = w.count > 0 || w.source != w.target;
  return w;
}

__device__ __forceinline__ int materialize_position(const Materialize& p,
                                                   const MaterializeRange& w, int t) {
  return p.format == CompactDKG ? int((w.start + t) & (p.capacity - 1)) : int(w.start + t);
}

__device__ __forceinline__ int64_t history_vector_offset(const Materialize& p, const Tensor& t,
                                                        int layer, const MaterializeRange& w,
                                                        int head, int position, int channel) {
  int64_t offset = layer * t.stride[0] + w.replay * t.stride[1];
  if (p.format == CachedUKG)
    return offset + w.buffer * t.stride[2] + position * t.stride[3] +
           head * t.stride[4] + channel * t.stride[5];
  return offset + head * t.stride[2] + position * t.stride[3] + channel * t.stride[4];
}

__device__ __forceinline__ int64_t history_gate_offset(const Materialize& p, const Tensor& t,
                                                      int layer, const MaterializeRange& w,
                                                      int head, int position) {
  const int64_t offset = layer * t.stride[0] + w.replay * t.stride[1];
  if (p.format == CachedUKG)
    return offset + w.buffer * t.stride[2] + head * t.stride[3] + position * t.stride[4];
  return offset + head * t.stride[2] + position * t.stride[3];
}

template <class S>
__device__ __forceinline__ S* materialize_state(const Materialize& p, int layer, int64_t slot) {
  if (p.descriptors.data) {
    const auto* desc = static_cast<const int64_t*>(p.descriptors.data) + layer * p.descriptors.stride[0];
    assert(desc[0] != 0 && desc[1] >= int64_t(p.value_heads) * p.value_dim * p.key_dim);
    return reinterpret_cast<S*>(desc[0]) + slot * desc[1];
  }
  return static_cast<S*>(p.state.data) + layer * p.state.stride[0] + slot * p.state.stride[1];
}

template <class S, class C>
__global__ __launch_bounds__(128) void materialize_compact_kernel(Materialize p) {
  const int row = blockIdx.y, layer = blockIdx.z / p.value_heads;
  const int hv = blockIdx.z % p.value_heads, h = hv / (p.value_heads / p.heads);
  const MaterializeRange w = materialize_range(p, row);
  if (!w.active) return;
  const int tid = threadIdx.x, K = p.key_dim, vbase = blockIdx.x * 8;
  const S* source = materialize_state<S>(p, layer, w.source);
  S* target = materialize_state<S>(p, layer, w.target);
  if (w.count == 0) {
    for (int x = tid; x < 8 * K; x += 128) {
      const int v = vbase + x / K;
      if (v < p.value_dim) {
        const int64_t offset = (hv * int64_t(p.value_dim) + v) * K + x % K;
        target[offset] = source[offset];
      }
    }
    return;
  }
  extern __shared__ float materialize_mem[];
  float* keys = materialize_mem;
  float* hi = keys + p.capacity * K;
  float* lo = hi + p.capacity * 8;
  float* weights = lo + p.capacity * 8;
  float* total_decay = weights + p.capacity;
  for (int x = tid; x < w.count * K; x += 128) {
    const int t = x / K, channel = x % K, pos = materialize_position(p, w, t);
    keys[x] = read<C>(p.keys, history_vector_offset(p, p.keys, layer, w, h, pos, channel));
  }
  if (tid == 0) {
    float total = 0.f;
    if (p.format == CachedUKG) {
      const float origin = w.start > 0 ? read<float>(p.gates,
          history_gate_offset(p, p.gates, layer, w, hv, int(w.start - 1))) : 0.f;
      const float last = read<float>(p.gates,
          history_gate_offset(p, p.gates, layer, w, hv, int(w.start + w.count - 1)));
      for (int t = 0; t < w.count; ++t) {
        const float current = read<float>(p.gates,
            history_gate_offset(p, p.gates, layer, w, hv, int(w.start + t)));
        weights[t] = expf(last - current);
      }
      total = last - origin;
    } else {
      for (int t = w.count - 1; t >= 0; --t) {
        const int pos = materialize_position(p, w, t);
        weights[t] = expf(total);
        total += read<float>(p.gates, history_gate_offset(p, p.gates, layer, w, hv, pos));
      }
    }
    *total_decay = expf(total);
  }
  __syncthreads();
  for (int x = tid; x < w.count * 8; x += 128) {
    const int t = x / 8, v = vbase + x % 8, pos = materialize_position(p, w, t);
    float value = 0.f;
    if (v < p.value_dim) {
      value = read<C>(p.values, history_vector_offset(p, p.values, layer, w, hv, pos, v));
      if (p.residual)
        value += read<C>(p.value_residual, history_vector_offset(p, p.value_residual, layer, w, hv, pos, v));
    }
    const float scaled = value * weights[t];
    hi[x] = rounded<C>(scaled);
    lo[x] = rounded<C>(scaled - hi[x]);
  }
  __syncthreads();
  for (int x = tid; x < 8 * K; x += 128) {
    const int v = vbase + x / K, k = x % K;
    if (v >= p.value_dim) continue;
    float delta = 0.f, correction = 0.f;
    for (int t = 0; t < w.count; ++t) {
      const float key = keys[t * K + k];
      delta += key * hi[t * 8 + v - vbase];
      delta += key * lo[t * 8 + v - vbase];
      if (p.residual) {
        const int pos = materialize_position(p, w, t);
        const float residual = read<C>(p.key_residual,
            history_vector_offset(p, p.key_residual, layer, w, h, pos, k));
        correction += residual * hi[t * 8 + v - vbase];
        correction += residual * lo[t * 8 + v - vbase];
      }
    }
    const int64_t offset = (hv * int64_t(p.value_dim) + v) * K + k;
    target[offset] = static_cast<S>(float(source[offset]) * *total_decay + (delta + correction));
  }
}

template <class S, class C>
__global__ __launch_bounds__(128) void materialize_raw_kernel(Materialize p) {
  const int row = blockIdx.y, layer = blockIdx.z / p.value_heads;
  const int hv = blockIdx.z % p.value_heads, h = hv / (p.value_heads / p.heads);
  const MaterializeRange w = materialize_range(p, row);
  if (!w.active) return;
  const int tid = threadIdx.x, lane = tid & 3, v = blockIdx.x * 32 + tid / 4, K = p.key_dim;
  const S* source = materialize_state<S>(p, layer, w.source);
  S* target = materialize_state<S>(p, layer, w.target);
  const int64_t offset = (hv * int64_t(p.value_dim) + v) * K;
  if (w.count == 0) {
    if (v < p.value_dim)
      for (int k = lane; k < K; k += 4) target[offset + k] = source[offset + k];
    return;
  }
  float state[64] = {};
  if (v < p.value_dim)
    for (int k = lane, i = 0; k < K; k += 4, ++i) state[i] = float(source[offset + k]);
  extern __shared__ float raw_materialize_mem[];
  float* keys = raw_materialize_mem;
  float* values = keys + K;
  float* scalar = values + 32;
  for (int t = 0; t < w.count; ++t) {
    const int pos = materialize_position(p, w, t);
    for (int k = tid; k < K; k += 128)
      keys[k] = read<C>(p.keys, history_vector_offset(p, p.keys, layer, w, h, pos, k));
    if (tid < 32) {
      const int value_channel = blockIdx.x * 32 + tid;
      values[tid] = value_channel < p.value_dim ? read<C>(p.values,
          history_vector_offset(p, p.values, layer, w, hv, pos, value_channel)) : 0.f;
    }
    if (tid == 0) {
      scalar[0] = expf(read<float>(p.gates, history_gate_offset(p, p.gates, layer, w, hv, pos)));
      scalar[1] = read<float>(p.beta, history_gate_offset(p, p.beta, layer, w, hv, pos));
    }
    __syncthreads();
    if (tid < 32) {
      float square_sum = 0.f;
      for (int k = tid; k < K; k += 32) square_sum += keys[k] * keys[k];
      square_sum = sum_warp(square_sum);
      if (tid == 0) scalar[2] = norm_denominator(square_sum, p.norm_policy, p.epsilon);
    }
    __syncthreads();
    float predicted = 0.f;
    for (int k = lane, i = 0; k < K; k += 4, ++i) {
      state[i] *= scalar[0];
      predicted += state[i] * (keys[k] / scalar[2]);
    }
    predicted = sum_row(predicted);
    const float update = (values[tid / 4] - predicted) * scalar[1];
    for (int k = lane, i = 0; k < K; k += 4, ++i)
      state[i] += (keys[k] / scalar[2]) * update;
    __syncthreads();
  }
  if (v < p.value_dim)
    for (int k = lane, i = 0; k < K; k += 4, ++i)
      target[offset + k] = static_cast<S>(state[i]);
}

template <class S, class C>
void materialize_typed(const Materialize& p, hggcStream_t stream) {
  const int tile_v = p.format == RawWindow ? 32 : 8;
  const size_t bytes = sizeof(float) * (p.format == RawWindow ?
      p.key_dim + 35 : int64_t(p.capacity) * (p.key_dim + 17) + 1);
  if (bytes > 48 * 1024) throw std::invalid_argument("materialize exceeds 48KiB shared memory");
  const dim3 grid((p.value_dim + tile_v - 1) / tile_v, p.batch, p.layers * p.value_heads);
  if (p.format == RawWindow)
    materialize_raw_kernel<S, C><<<grid, 128, bytes, stream>>>(p);
  else
    materialize_compact_kernel<S, C><<<grid, 128, bytes, stream>>>(p);
}
