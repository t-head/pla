// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <hggc_runtime.h>

namespace pla::gdn_replayssm {

enum DType : int { F32, F16, BF16, I32, I64, I8 };

struct Tensor {
  void* data = nullptr;
  int64_t stride[6] = {};
  DType dtype = F32;
};

struct Verify {
  Tensor q, k, v, a, b, A_log, dt_bias, state;
  Tensor d, keys, g, rawv, rawk, beta, out;
  Tensor starts, slots, replay, wp, base, flush;
  float* workspace = nullptr;
  int batch, heads, value_heads, key_dim, value_dim, capacity, max_steps;
  int state_slots, replay_slots, tokens, null_slot;
  float scale;
  bool normalize, exact, residual, tf32, processed_gates = false;
  bool legacy_dot_precision = false;
  int mode, norm_policy = 1;
  float epsilon = 1e-6f;
};

struct Commit {
  Tensor state, d, keys, g, dr, kr;
  Tensor slots, replay, wp, base, flush, accepted, track, steps;
  int layers, batch, heads, value_heads, key_dim, value_dim, capacity;
  int state_slots, replay_slots, null_slot;
  bool residual, tracking;
  bool extended_base = false;
};

enum HistoryFormat : int { CompactDKG, RawWindow, CachedUKG };

struct Materialize {
  Tensor state, values, keys, gates, value_residual, key_residual, beta;
  Tensor state_indices, replay_indices, lengths, starts, buffers, metadata_indices;
  Tensor targets, enabled, items, n_items, descriptors;
  int layers, batch, heads, value_heads, key_dim, value_dim, capacity;
  int state_slots, replay_slots, metadata_slots, null_slot;
  HistoryFormat format;
  int norm_policy = 1;
  float epsilon = 1e-6f;
  bool residual = false;
  int overflow_steps = 0;
};

void launch_verify(const Verify&, hggcStream_t);
void launch_materialize(const Materialize&, hggcStream_t);
bool launch_fast_commit(const Commit&, hggcStream_t);

}
