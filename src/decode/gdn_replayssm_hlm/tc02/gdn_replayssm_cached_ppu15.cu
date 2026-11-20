// SPDX-License-Identifier: Apache-2.0
#include "../common/gdn_replayssm_hlm_abi.h"

#if !defined(__HGGC_ARCH__) || __HGGC_ARCH__ == 150

#include <hggc_bf16.h>
#include <hggc_fp16.h>
#include <hggc_mma.h>
#include <hggc_pipeline_primitives.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cached_tensor_matrix.cuh"

namespace pla::gdn_replayssm {
namespace {

#include "cached_view_common_v2.cuh"
#include "cached_operand_loaders.cuh"
#include "fold_matrix.cuh"
#include "cached_t1_regcorr.cuh"
#include "cached_t1_rtiny.cuh"
#include "cached_t1_parameters.cuh"
#include "gdn_replayssm_cached_tensor.cuh"
#include "cached_t1_metaempty.cuh"
#include "cached_register_tensor.cuh"
#include "cached_t4_grouped.cuh"
#include "cached_t4_broadcast.cuh"

// Called after the runtime PPU1.5 check and the fixed-layout routes.
bool launch_cached_tensor(const CachedVerify& p, hggcStream_t stream) {
  if (p.q.dtype == BF16 && p.key_dim == 128 && p.value_dim == 128 &&
      p.capacity == 16 && !p.starts.data && p.tf32 && p.norm_policy == 2) {
    if (p.max_steps == 1) {
      if (p.state.dtype == F32) {
        if (int64_t(p.batch)*p.value_heads < 128)
          cached_decode_vector_kernel<float,8><<<dim3(8,p.batch,p.value_heads),128,0,stream>>>(p);
        else
          cached_decode_vector_kernel<float,4><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p);
      } else {
        if (int64_t(p.batch)*p.value_heads >= 128)
          cached_tensor_kernel<BF,1,128,3,true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p);
        else
          cached_decode_vector_kernel<BF,4,false><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p);
      }
      return true;
    }
#define CACHED_STEP(N) case N: \
    if (N <= 4 && int64_t(p.batch)*p.value_heads >= 128) { \
      if (p.state.dtype == F32) \
        cached_tensor_kernel<float, N, 64, 3, true><<<dim3(2,p.batch,p.value_heads),128,0,stream>>>(p); \
      else if (int64_t(p.batch)*p.value_heads >= 512) \
        cached_tensor_kernel<BF, N, 128, 3, true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p); \
      else \
        cached_tensor_kernel<BF, N, 64, 3, true><<<dim3(2,p.batch,p.value_heads),128,0,stream>>>(p); \
    } else if (N <= 4 || int64_t(p.batch)*p.value_heads < 128) { \
      if (p.state.dtype == F32) \
        cached_tensor_kernel<float, N, 32, 3, true><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p); \
      else \
        cached_tensor_kernel<BF, N, 32, 3, true><<<dim3(4,p.batch,p.value_heads),128,0,stream>>>(p); \
    } else { \
      if (p.state.dtype == F32) \
        cached_tensor_kernel<float, N, 128, 3, true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p); \
      else \
        cached_tensor_kernel<BF, N, 128, 3, true><<<dim3(1,p.batch,p.value_heads),128,0,stream>>>(p); \
    } \
    return true
    switch (p.max_steps) {
      CACHED_STEP(1); CACHED_STEP(2); CACHED_STEP(3); CACHED_STEP(4);
      CACHED_STEP(5); CACHED_STEP(6); CACHED_STEP(7); CACHED_STEP(8);
    }
#undef CACHED_STEP
  }
  return false;
}

constexpr int kDim = 128;
constexpr int kCapacity = 16;

template <int64_t... Strides>
struct DenseView {
  void* data;
  static constexpr int64_t stride[sizeof...(Strides) + 1] = {Strides..., 0};
};

template <int H, int HV = H * 4, bool Packed = false, bool Raw = false>
struct DenseReplay {
  static constexpr int64_t query_width = H * kDim;
  static constexpr int64_t value_width = HV * kDim;
  static constexpr int64_t packed_width = 2 * query_width + value_width;
  static constexpr int head_group = HV / H;
  static constexpr bool raw_gates = Raw;
  DenseView<> slots, lengths, buffers;
  DenseView<value_width * kDim> state;
  DenseView<Packed ? packed_width : query_width> q, k;
  DenseView<Packed ? packed_width : value_width> v;
  DenseView<value_width> out;
  DenseView<HV> a, b;
  DenseView<2 * kCapacity * query_width, kCapacity * query_width, query_width> keys;
  DenseView<2 * kCapacity * value_width, kCapacity * value_width, value_width> values;
  DenseView<2 * HV * kCapacity, HV * kCapacity> gates;
  int state_slots, replay_slots, null_slot;
  float scale, epsilon;
  bool materialize_overflow;
  DenseView<1> A_log, dt_bias;
};

template <int H, int HV = H * 4, bool Packed = false, bool Raw = false>
DenseReplay<H, HV, Packed, Raw> pack_dense(const CachedVerify& p) {
  return {{p.slots.data}, {p.lengths.data}, {p.buffers.data}, {p.state.data},
          {p.q.data}, {p.k.data}, {p.v.data}, {p.out.data}, {p.a.data}, {p.b.data},
          {p.keys.data}, {p.values.data}, {p.gates.data}, p.state_slots, p.replay_slots,
          p.null_slot, p.scale, p.epsilon, p.materialize_overflow, {p.A_log.data}, {p.dt_bias.data}};
}

template <int H, int HV = H * 4, bool Packed = false>
bool dense(const CachedVerify& p) {
  constexpr int64_t query = H * kDim, value = HV * kDim, packed = 2 * query + value;
  return p.heads == H && p.value_heads == HV && p.state.stride[0] == value * kDim &&
      p.q.stride[0] == (Packed ? packed : query) && p.k.stride[0] == (Packed ? packed : query) &&
      p.v.stride[0] == (Packed ? packed : value) && p.out.stride[0] == value &&
      p.a.stride[0] == HV && p.b.stride[0] == HV &&
      p.keys.stride[0] == 2*kCapacity*query && p.keys.stride[1] == kCapacity*query && p.keys.stride[2] == query &&
      p.values.stride[0] == 2*kCapacity*value && p.values.stride[1] == kCapacity*value && p.values.stride[2] == value &&
      p.gates.stride[0] == 2*HV*kCapacity && p.gates.stride[1] == HV*kCapacity;
}

bool shared_i32_slots(const CachedVerify& p) {
  return p.slots.data == p.replay.data && p.slots.dtype == I32 && p.replay.dtype == I32 &&
      p.lengths.dtype == I32 && p.buffers.dtype == I32 && p.slots.stride[0] == 1 &&
      p.replay.stride[0] == 1 && p.lengths.stride[0] == 1 && p.buffers.stride[0] == 1;
}

bool aligned(const Tensor& view, int bytes, int dimensions = 0) {
  if (reinterpret_cast<uintptr_t>(view.data) % bytes != 0) return false;
  for (int d = 0; d < dimensions; ++d)
    if (view.stride[d] % (bytes / sizeof(BF)) != 0) return false;
  return true;
}

bool bulk_history_aligned(const CachedVerify& p) {
  return aligned(p.keys, 128, 3) && aligned(p.values, 128, 3);
}

bool dense_bulk_aligned(const CachedVerify& p) {
  return aligned(p.state, 128) && aligned(p.keys, 128) && aligned(p.values, 128);
}

bool processed_fp32_gates(const CachedVerify& p) {
  return p.processed_gates && p.a.dtype == F32 && p.b.dtype == F32;
}

template <bool Packed, bool Raw, bool Single>
void launch_tp1(const CachedVerify& p, hggcStream_t stream) {
  auto view = pack_dense<16, 48, Packed, Raw>(p);
  constexpr int Tile = 32;
  const dim3 grid(kDim / Tile, p.batch, 48);
  cached_register_tensor<float, 4, Tile, 16, Single, true, 2, 14, 2>
      <<<grid, 2 * Tile, 0, stream>>>(view);
}

bool try_tp1(const CachedVerify& p, hggcStream_t stream) {
  if (p.batch > 2 || p.heads != 16 || p.value_heads != 48 || p.max_steps != 4 ||
      p.state.dtype != F32 || !p.materialize_overflow ||
      !shared_i32_slots(p) || p.offsets.data || !dense_bulk_aligned(p) ||
      !aligned(p.q, 16) || !aligned(p.k, 16))
    return false;
  if (!p.processed_gates && p.a.dtype == BF16 && p.b.dtype == BF16 &&
      p.A_log.dtype == F32 && p.dt_bias.dtype == F32 &&
      p.A_log.data && p.dt_bias.data && p.A_log.stride[0] == 1 && p.dt_bias.stride[0] == 1 &&
      dense<16, 48, true>(p)) {
    if (p.batch == 1) launch_tp1<true, true, true>(p, stream);
    else launch_tp1<true, true, false>(p, stream);
    return true;
  }
  if (processed_fp32_gates(p) && dense<16, 48>(p)) {
    if (p.batch == 1) launch_tp1<false, false, true>(p, stream);
    else launch_tp1<false, false, false>(p, stream);
    return true;
  }
  return false;
}

bool try_single_slot_t4(const CachedVerify& p, hggcStream_t stream) {
  if (p.batch != 1 || p.max_steps != 4 || p.state.dtype != BF16 ||
      p.state_slots != 1 || p.replay_slots != 1 || !dense<1>(p))
    return false;
  constexpr int Tile = 64, Helpers = 2;
  auto view = pack_dense<1>(p);
  cached_t4_broadcast<Tile, Helpers, 4, true, true, true, true, true, false>
      <<<dim3(4 * kDim / Tile), 2 * Tile + 32 * Helpers, 0, stream>>>(view);
  return true;
}

template <class State, int Steps, int Tile, int Batch, int Heads>
bool launch_register(const CachedVerify& p, hggcStream_t stream) {
  constexpr DType Type = std::is_same_v<State, float> ? F32 : BF16;
  if (p.max_steps != Steps || p.state.dtype != Type || p.batch != Batch || !dense<Heads>(p))
    return false;
  const auto view = pack_dense<Heads>(p);
  const dim3 grid(kDim / Tile, p.batch, p.value_heads);
  constexpr bool FuseKeys = Steps == 4 || Steps == 8;
  constexpr int FloatLayout = 2;
  cached_register_tensor<State, Steps, Tile, Heads, Batch == 1, FuseKeys, FloatLayout, 0, 2>
      <<<grid, 2 * Tile, 0, stream>>>(view);
  return true;
}

bool try_register(const CachedVerify& p, hggcStream_t stream) {
  return launch_register<float, 1, 32, 1, 1>(p, stream) ||
      launch_register<BF, 4, 64, 1, 1>(p, stream) ||
      launch_register<float, 4, 64, 1, 1>(p, stream) ||
      launch_register<BF, 8, 64, 1, 1>(p, stream) ||
      launch_register<BF, 1, 64, 1, 4>(p, stream) ||
      launch_register<float, 1, 32, 1, 4>(p, stream) ||
      launch_register<float, 4, 64, 1, 4>(p, stream) ||
      launch_register<BF, 8, 64, 1, 4>(p, stream) ||
      launch_register<BF, 1, 32, 8, 1>(p, stream) ||
      launch_register<float, 1, 64, 8, 1>(p, stream) ||
      launch_register<float, 4, 64, 8, 1>(p, stream) ||
      launch_register<BF, 8, 64, 8, 1>(p, stream) ||
      launch_register<BF, 4, 64, 8, 4>(p, stream) ||
      launch_register<BF, 8, 64, 8, 4>(p, stream) ||
      launch_register<BF, 4, 64, 32, 1>(p, stream) ||
      launch_register<BF, 8, 64, 32, 1>(p, stream) ||
      launch_register<float, 1, 64, 128, 4>(p, stream) ||
      launch_register<float, 1, 64, 128, 1>(p, stream) ||
      launch_register<float, 8, 64, 8, 1>(p, stream) ||
      launch_register<float, 8, 64, 1, 4>(p, stream);
}

bool try_large_t1(const CachedVerify& p, hggcStream_t stream) {
  if (p.max_steps != 1 || p.state.dtype != BF16) return false;
  const dim3 grid(1, p.batch * p.value_heads);
  if (((p.batch == 128 && p.heads == 4) || (p.batch == 32 && p.heads == 4)) && dense<4>(p)) {
    cached_decode_t1<40, true, true, true, true, 4><<<grid, 128, 0, stream>>>(pack_t1(p));
    return true;
  }
  if (p.batch == 128 && dense<1>(p)) {
    cached_decode_t1<40, true, true, true, true, 1><<<grid, 128, 0, stream>>>(pack_t1(p));
    return true;
  }
  return false;
}

template <int Heads, bool Single>
bool launch_grouped_t4(const CachedVerify& p, hggcStream_t stream) {
  if (!dense<Heads>(p)) return false;
  constexpr int Tile = 64;
  const dim3 grid(kDim / Tile, p.batch, p.value_heads);
  cached_t4_tensor<Tile, Heads, Single, true, true, false>
      <<<grid, 2 * Tile, 0, stream>>>(pack_dense<Heads>(p));
  return true;
}

bool try_grouped_t4(const CachedVerify& p, hggcStream_t stream) {
  if (p.max_steps != 4 || p.state.dtype != BF16) return false;
  if (p.batch == 1)
    return p.heads == 1 ? launch_grouped_t4<1, true>(p, stream) : launch_grouped_t4<4, true>(p, stream);
  return p.batch == 8 && launch_grouped_t4<1, false>(p, stream);
}

bool try_small_t1(const CachedVerify& p, hggcStream_t stream) {
  if (p.max_steps != 1 || p.state.dtype != BF16) return false;
  const dim3 grid(4, p.batch, p.value_heads);
  if (p.batch == 1 && dense<1>(p)) {
    cached_decode_t1_small<32, 0, 1, false><<<grid, 64, 0, stream>>>(pack_dense<1>(p));
    return true;
  }
  if (p.batch == 1 && dense<4>(p)) {
    cached_decode_t1_tiny<32, 0, 4, true><<<grid, 64, 0, stream>>>(pack_dense<4>(p));
    return true;
  }
  if (p.batch == 8 && dense<1>(p)) {
    cached_decode_t1_tiny<32, 0, 1, false><<<grid, 64, 0, stream>>>(pack_dense<1>(p));
    return true;
  }
  return false;
}

bool try_mid_t1(const CachedVerify& p, hggcStream_t stream) {
  const int64_t work = int64_t(p.batch) * p.value_heads;
  if (p.state.dtype != BF16 || p.max_steps != 1 || work < 128 || work >= 512)
    return false;
  const CachedT1 packed = pack_t1(p);
  const dim3 grid(1, p.batch, p.value_heads);
  const bool i32 = p.slots.dtype == I32 && p.replay.dtype == I32 && p.lengths.dtype == I32 &&
      p.buffers.dtype == I32 && (!p.offsets.data || p.offsets.dtype == I32);
  if (shared_i32_slots(p) && p.value_heads == 4 * int64_t(p.heads) &&
      !p.offsets.data && processed_fp32_gates(p) && aligned(p.state, 16, 1) && bulk_history_aligned(p)) {
    cached_decode_t1<40, true, true, true, true, 0><<<grid, 128, 0, stream>>>(packed);
  } else if (i32) {
    cached_decode_t1<40, false, true, true, false, 0><<<grid, 128, 0, stream>>>(packed);
  } else {
    cached_decode_t1<40, false, true, false, false, 0><<<grid, 128, 0, stream>>>(packed);
  }
  return true;
}
}

bool launch_cached_ppu15(const CachedVerify& p, hggcStream_t stream) {
  if (p.q.dtype != BF16 || p.key_dim != kDim || p.value_dim != kDim || p.capacity != kCapacity ||
      p.starts.data || !p.tf32 || p.norm_policy != 2)
    return false;
  if (try_tp1(p, stream)) return true;
  if (!p.offsets.data && processed_fp32_gates(p) && shared_i32_slots(p) && dense_bulk_aligned(p)) {
    const bool vector_qk = aligned(p.q, 16) && aligned(p.k, 16);
    if (vector_qk && (try_single_slot_t4(p, stream) || try_register(p, stream))) return true;
    if (try_large_t1(p, stream)) return true;
    if (vector_qk && try_grouped_t4(p, stream)) return true;
    if (try_small_t1(p, stream)) return true;
  }
  if (try_mid_t1(p, stream)) return true;
  return launch_cached_tensor(p, stream);
}

}  // namespace pla::gdn_replayssm
#endif  // PPU1.5 device pass or host pass
