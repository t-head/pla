// SPDX-License-Identifier: Apache-2.0
#include "../common/gdn_replayssm_hlm_abi.h"

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 100

#include "../../gdn_replayssm/common/gdn_replayssm_device.cuh"

#include <hggc_mma.h>

namespace pla::gdn_replayssm {
namespace {

using namespace detail;

#include "../common/gdn_replayssm_cached_memory.cuh"
#include "cached_dense.cuh"
#include "cached_decode.cuh"
#include "cached_decode_dense.cuh"
#include "cached_verify.cuh"

template <class State, int Steps, class Params>
void launch_tc01_verify(const Params& p, hggcStream_t stream) {
  constexpr bool Dense = !std::is_same_v<Params, CachedVerify>;
  constexpr bool StageBF16 = std::is_same_v<State, BF>;
  const int64_t rows = int64_t(p.batch) * p.value_heads;
  if constexpr (Dense && StageBF16 && Steps == 4) {
    if (p.batch == 1 && p.state_slots == 1 && p.replay_slots == 1 &&
        ((reinterpret_cast<uintptr_t>(p.keys.data) | reinterpret_cast<uintptr_t>(p.values.data)) & 15u) == 0) {
      // The dense one-slot kernel packs Q and K in the MMA register order.
      tc01_verify_kernel<State, Steps, 64, true, false, false, false, true>
          <<<dim3(2, p.batch, p.value_heads), 128, 0, stream>>>(p);
      return;
    }
    if constexpr (Params::heads == 4) {
      if (p.batch == 128) {
        // Keep multiple checkpoint loads in flight for this large grid.
        tc01_verify_kernel<State, Steps, 128, false, true, true>
            <<<dim3(1, p.batch, p.value_heads), 128, 0, stream>>>(p);
        return;
      }
      if (p.batch == 1) {
        // Use both projection warps and compute each prefix for twice as many values.
        tc01_verify_kernel<State, Steps, 64, true, false>
            <<<dim3(2, p.batch, p.value_heads), 128, 0, stream>>>(p);
        return;
      }
    }
  }
  if constexpr (Dense && StageBF16 && Steps == 8) {
    if (rows >= 32 && rows < 128 &&
        ((reinterpret_cast<uintptr_t>(p.keys.data) | reinterpret_cast<uintptr_t>(p.values.data)) & 15u) == 0) {
      tc01_verify_kernel<State, Steps, 64, false, false, false, true>
          <<<dim3(2, p.batch, p.value_heads), 128, 0, stream>>>(p);
      return;
    }
  }
  if constexpr (Dense && Steps == 8) {
    if (rows >= 32 && rows < 128) {
      // Use two value tiles so fewer CTAs repeat the same prefix work.
      tc01_verify_kernel<State, Steps, 64, false, false>
          <<<dim3(2, p.batch, p.value_heads), 128, 0, stream>>>(p);
      return;
    }
  }
  if (rows >= (Dense ? 128 : 512)) {
    tc01_verify_kernel<State, Steps, 128, false, StageBF16>
        <<<dim3(1, p.batch, p.value_heads), 128, 0, stream>>>(p);
  } else if (rows < 128) {
    tc01_verify_kernel<State, Steps, 32, true, false>
        <<<dim3(4, p.batch, p.value_heads), 128, 0, stream>>>(p);
  } else {
    tc01_verify_kernel<State, Steps, 64, false, StageBF16>
        <<<dim3(2, p.batch, p.value_heads), 128, 0, stream>>>(p);
  }
}

template <class State, int Group, class Params>
void launch_tc01_decode(const Params& p, hggcStream_t stream) {
  tc01_decode_kernel<State, Group><<<dim3(Group, p.batch, p.value_heads), 128, 0, stream>>>(p);
}

template <class State, class Params>
void launch_fixed_steps(const Params& p, int steps, hggcStream_t stream) {
  switch (steps) {
    case 1: {
      // The scalar kernel keeps T=1 on the generic CUs, including overflow.
      const int64_t rows = int64_t(p.batch) * p.value_heads;
      if constexpr (std::is_same_v<State, BF> && !std::is_same_v<Params, CachedVerify>) {
        if (rows <= 32 && (reinterpret_cast<uintptr_t>(p.keys.data) & 15u) == 0) {
          if (p.batch == 1 && p.state_slots == 1 && p.replay_slots == 1) {
            if constexpr (Params::heads == 1)
              tc01_decode_dense_kernel<8, true><<<dim3(8, p.batch, p.value_heads), 128, 0, stream>>>(p);
            else
              tc01_decode_dense_kernel<4, true><<<dim3(4, p.batch, p.value_heads), 128, 0, stream>>>(p);
          } else {
            tc01_decode_dense_kernel<4, false><<<dim3(4, p.batch, p.value_heads), 128, 0, stream>>>(p);
          }
          return;
        }
        if (rows < 32) {
          launch_tc01_decode<State, 16>(p, stream);
          return;
        }
      }
      if (rows < 128) launch_tc01_decode<State, 8>(p, stream);
      else launch_tc01_decode<State, 4>(p, stream);
      return;
    }
    case 4:
      launch_tc01_verify<State, 4>(p, stream);
      return;
    case 8:
      launch_tc01_verify<State, 8>(p, stream);
      return;
  }
}

template <class State, int Heads>
bool try_dense_launch(const CachedVerify& p, hggcStream_t stream) {
  using Layout = Tc01Dense<Heads, std::is_same_v<State, BF> ? BF16 : F32>;
  if (!Layout::matches(p)) return false;
  launch_fixed_steps<State>(Layout::pack(p), p.max_steps, stream);
  return true;
}

template <class State>
void dispatch_tc01(const CachedVerify& p, hggcStream_t stream) {
  if (try_dense_launch<State, 1>(p, stream) || try_dense_launch<State, 4>(p, stream)) return;
  launch_fixed_steps<State>(p, p.max_steps, stream);
}

}  // namespace

bool launch_cached_ppu10(const CachedVerify& p, hggcStream_t stream) {
  if (p.q.dtype != BF16 || p.key_dim != 128 || p.value_dim != 128 || p.capacity != 16 ||
      p.starts.data || !p.tf32 || p.norm_policy != 2 ||
      (p.max_steps != 1 && p.max_steps != 4 && p.max_steps != 8) ||
      (reinterpret_cast<uintptr_t>(p.state.data) & 15u) != 0 || (p.state.stride[0] & 7))
    return false;
  if (p.state.dtype == BF16) dispatch_tc01<BF>(p, stream);
  else if (p.state.dtype == F32) dispatch_tc01<float>(p, stream);
  else return false;
  return true;
}

}  // namespace pla::gdn_replayssm
#endif  // PPU1.1 device pass
