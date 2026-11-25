// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gdn_replayssm_sglang_abi.h"

#include <cassert>

namespace pla::gdn_replayssm {
namespace {

constexpr int SGLANG_METADATA_THREADS = 32;

// The common small-batch path passes only the pointers and bounds each kernel needs.
__global__ __launch_bounds__(SGLANG_METADATA_THREADS) void sglang_cursors_i32_kernel(
    int32_t* write_pos, int32_t* cache_base, int8_t* is_flush,
    const int32_t* num_accepted, const int32_t* replay_indices,
    int batch, int replay_slots, int null_slot, int max_steps, int capacity,
    int buffer_len, int flush_threshold) {
  const int row = threadIdx.x;
  if (row >= batch) return;
  const int replay = replay_indices[row];
  if (replay <= null_slot) return;
  assert(replay >= 0 && replay < replay_slots);
  const int accepted = num_accepted[row];
  const int pos = write_pos[replay];
  const int base = cache_base[replay];
  const bool flush = is_flush[replay] != 0;
  assert(accepted >= 0 && accepted <= max_steps && pos >= 0 && pos <= capacity &&
         base >= 0 && base < buffer_len);
  const int next = flush ? accepted : pos + accepted;
  assert(next <= capacity);
  cache_base[replay] = flush && accepted > 0 ? (base + pos) & (buffer_len - 1) : base;
  write_pos[replay] = next;
  is_flush[replay] = next > flush_threshold;
}

__global__ __launch_bounds__(SGLANG_METADATA_THREADS) void sglang_finish_i32_kernel(
    const int32_t* state_indices, const int32_t* replay_indices,
    int32_t* write_pos, int32_t* cache_base, int8_t* is_flush,
    int batch, int state_slots, int replay_slots, int null_slot, int capacity) {
  const int row = threadIdx.x;
  if (row >= batch) return;
  const int slot = state_indices[row], replay = replay_indices[row];
  if (slot <= null_slot) return;
  assert(slot >= 0 && slot < state_slots && replay >= 0 && replay < replay_slots);
  if (!is_flush[replay]) return;
  const int pos = write_pos[replay], base = cache_base[replay];
  assert(pos >= 0 && pos <= capacity && base >= 0);
  cache_base[replay] = ((base & (capacity - 1)) + pos) & (capacity - 1);
  write_pos[replay] = 0;
  is_flush[replay] = 0;
}

inline bool contiguous_metadata(const Tensor& tensor, DType dtype) {
  return tensor.dtype == dtype && tensor.stride[0] == 1;
}

inline bool launch_contiguous_sglang_cursors(const Cursors& p, hggcStream_t stream) {
  if (p.batch > SGLANG_METADATA_THREADS || !contiguous_metadata(p.wp, I32) ||
      !contiguous_metadata(p.base, I32) || !contiguous_metadata(p.flush, I8) ||
      !contiguous_metadata(p.accepted, I32) || !contiguous_metadata(p.replay, I32))
    return false;
  const int flush_threshold = p.fold_every_commit ? 0 : p.capacity - 2 * p.max_steps;
  sglang_cursors_i32_kernel<<<1, SGLANG_METADATA_THREADS, 0, stream>>>(
      static_cast<int32_t*>(p.wp.data), static_cast<int32_t*>(p.base.data),
      static_cast<int8_t*>(p.flush.data), static_cast<const int32_t*>(p.accepted.data),
      static_cast<const int32_t*>(p.replay.data), p.batch, p.replay_slots, p.null_slot,
      p.max_steps, p.capacity, p.buffer_len, flush_threshold);
  return true;
}

inline bool launch_contiguous_sglang_finish(const Commit& p, hggcStream_t stream) {
  if (p.batch > SGLANG_METADATA_THREADS || !contiguous_metadata(p.wp, I32) ||
      !contiguous_metadata(p.base, I32) || !contiguous_metadata(p.flush, I8) ||
      !contiguous_metadata(p.slots, I32) || !contiguous_metadata(p.replay, I32))
    return false;
  sglang_finish_i32_kernel<<<1, SGLANG_METADATA_THREADS, 0, stream>>>(
      static_cast<const int32_t*>(p.slots.data), static_cast<const int32_t*>(p.replay.data),
      static_cast<int32_t*>(p.wp.data), static_cast<int32_t*>(p.base.data),
      static_cast<int8_t*>(p.flush.data), p.batch, p.state_slots, p.replay_slots, p.null_slot,
      p.capacity);
  return true;
}

}  // namespace
}  // namespace pla::gdn_replayssm
