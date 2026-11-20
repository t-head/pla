// SPDX-License-Identifier: Apache-2.0
#include "gdn_replayssm_sglang_abi.h"
#include "gdn_replayssm_sglang_metadata.cuh"
#include "../../gdn_replayssm/common/gdn_replayssm_device.cuh"

#include <stdexcept>

namespace pla::gdn_replayssm {
namespace {

using namespace detail;

template <class S, class C>
__global__ __launch_bounds__(THREADS) void exact_fold_kernel(Verify p) {
  const int row = blockIdx.y, hv = blockIdx.z, h = hv / (p.value_heads / p.heads);
  const int64_t slot = index(p.slots, row), replay = index(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  if (!index(p.flush, replay)) return;
  const int n = bounded_index(p.wp, replay, 0, p.capacity), K = p.key_dim;
  const int base = physical_base(index(p.base, replay), p.capacity, true);
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

template <class S, class C>
void launch_exact_fold(const Verify& p, hggcStream_t stream) {
  const size_t shared = sizeof(float) * (p.key_dim + ROWS + 3);
  const dim3 grid((p.value_dim + ROWS - 1) / ROWS, p.batch, p.value_heads);
  exact_fold_kernel<S, C><<<grid, THREADS, shared, stream>>>(p);
}

template <class S, class C>
__global__ __launch_bounds__(THREADS) void compact_commit_kernel(Commit p) {
  const int row = blockIdx.y, layer = blockIdx.z / p.value_heads;
  const int hv = blockIdx.z % p.value_heads, h = hv / (p.value_heads / p.heads);
  const int64_t slot = index(p.slots, row), replay = index(p.replay, row);
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int n = bounded_index(p.wp, replay, 0, p.capacity);
  const int base = physical_base(index(p.base, replay), p.capacity, true);
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
    keys[x] = read<C>(p.keys, ring_offset(p.keys, layer, replay, h, phys, K, k, p.capacity));
  }
  for (int x = tid; x < count * 8; x += THREADS) {
    const int t = x / 8, v = vbase + x % 8, phys = (base + t) & (p.capacity - 1);
    float update = 0.f;
    if (v < p.value_dim)
      update = read<C>(p.d, ring_offset(p.d, layer, replay, hv, phys, p.value_dim, v, p.capacity));
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
    if (p.residual) {
      ResidualFold full, tracked;
      for (int t = 0; t < count; ++t) {
        const int phys = (base + t) & (p.capacity - 1);
        const float key = keys[t * K + k];
        const float extra_key = read<C>(p.kr, ring_offset(p.kr, layer, replay, h, phys, K, k, p.capacity));
        const float value = updates[t * 8 + v - vbase];
        const float extra_value = read<C>(p.dr, ring_offset(p.dr, layer, replay, hv, phys,
                                                             p.value_dim, v, p.capacity));
        if (fold && t < n)
          full.add(key, extra_key, scaled_fold_value<C>(value, dg[t]), scaled_fold_value<C>(extra_value, dg[t]));
        if (track && t < nt)
          tracked.add(key, extra_key, scaled_fold_value<C>(value, dt[t]), scaled_fold_value<C>(extra_value, dt[t]));
      }
      if (fold) write<S>(p.state, offset, full.result(old * totals[0]));
      if (track) write<S>(p.state, offset + (target - slot) * p.state.stride[1], tracked.result(old * totals[1]));
      continue;
    }
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
  const int base = physical_base(index(p.base, replay), p.capacity, true);
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
  const int wp = bounded_index(p.wp, replay, 0, p.capacity);
  const int accepted = bounded_index(p.accepted, row, 0, p.max_steps);
  const int base = bounded_index(p.base, replay, 0, p.buffer_len - 1);
  const bool flush = index(p.flush, replay) != 0;
  const int next = flush ? accepted : wp + accepted;
  assert(next <= p.capacity);
  put_index(p.base, replay, flush && accepted > 0 ? (base + wp) & (p.buffer_len - 1) : base);
  put_index(p.wp, replay, next);
  put_index(p.flush, replay, p.fold_every_commit ? next > 0 : next + 2 * p.max_steps > p.capacity);
}

template <class S, class C>
void commit_typed(const Commit& p, hggcStream_t stream) {
  const size_t shared = sizeof(float) * (int64_t(p.capacity) * (p.key_dim + 10) + 2);
  if (shared > 48 * 1024) throw std::invalid_argument("GDN replay commit exceeds 48KiB shared memory");
  compact_commit_kernel<S, C><<<dim3((p.value_dim + 7) / 8, p.batch, p.value_heads * p.layers), THREADS, shared, stream>>>(p);
}

}  // namespace

void launch_sglang_fold_and_verify(const Verify& p, hggcStream_t stream) {
  int steps = 1;
  while (steps < p.max_steps) steps *= 2;
  if (verify_shared_bytes(p, steps) > 48 * 1024)
    throw std::invalid_argument("GDN replay verify exceeds 48KiB shared memory");
  PLA_GDN_REPLAY_DISPATCH(p.state.dtype, S, PLA_GDN_REPLAY_DISPATCH(p.d.dtype, C, (launch_exact_fold<S, C>(p, stream))));
  launch_verify(p, stream);
}

void launch_sglang_materialize(const Commit& p, hggcStream_t stream) {
  if (p.batch == 0 || p.layers == 0) return;
  if (!launch_fast_commit(p, stream)) {
    PLA_GDN_REPLAY_DISPATCH(p.state.dtype, S, PLA_GDN_REPLAY_DISPATCH(p.d.dtype, C, (commit_typed<S, C>(p, stream))));
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

}  // namespace pla::gdn_replayssm

#undef PLA_GDN_REPLAY_DISPATCH
