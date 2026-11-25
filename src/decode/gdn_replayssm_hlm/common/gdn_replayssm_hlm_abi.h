// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../gdn_replayssm/common/gdn_replayssm_abi.h"

namespace pla::gdn_replayssm {

struct CachedVerify {
  Tensor q, k, v, a, b, A_log, dt_bias, state, values, keys, gates, out;
  Tensor starts, slots, replay, lengths, offsets, buffers;
  int batch, heads, value_heads, key_dim, value_dim, capacity, max_steps;
  int state_slots, replay_slots, tokens, null_slot;
  float scale, epsilon;
  int norm_policy;
  bool tf32, processed_gates;
  bool materialize_overflow = false;
};

// True indicates that the kernel also materializes the overflowing prefix.
bool launch_cached_verify(const CachedVerify&, hggcStream_t);
bool launch_cached_ppu15(const CachedVerify&, hggcStream_t);

bool launch_cached_ppu10(const CachedVerify&, hggcStream_t);

inline int current_device_compute_capability() {
  int device = -1, major = 0, minor = 0;
  if (hggcGetDevice(&device) != hggcSuccess ||
      hggcDeviceGetAttribute(&major, hggcDevAttrComputeCapabilityMajor, device) != hggcSuccess ||
      hggcDeviceGetAttribute(&minor, hggcDevAttrComputeCapabilityMinor, device) != hggcSuccess)
    return 0;
  return 10 * major + minor;
}

}  // namespace pla::gdn_replayssm
