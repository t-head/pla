// SPDX-License-Identifier: Apache-2.0
#pragma once

// Dense TRT-LLM buffers have fixed inner strides and share the state and
// replay slot map. The typed views remove per-CTA index and dtype dispatch.
template <DType D, int64_t... Steps>
struct Tc01View {
  void* data;
  static constexpr DType dtype = D;
  static constexpr int64_t stride[sizeof...(Steps)] = {Steps...};
  __device__ __forceinline__ operator Tensor() const { return {data, {Steps...}, D}; }
};

template <int H, DType StateDType>
struct Tc01Dense {
  static constexpr int heads = H, value_heads = 4 * H, Dim = 128, Cap = 16;
  static constexpr bool processed_gates = true, materialize_overflow = true;
  static constexpr int norm_policy = 2;
  Tc01View<I32, 1> slots, lengths, buffers;
  static constexpr Tc01View<I32, 1> replay{nullptr}, offsets{nullptr};
  static constexpr Tc01View<F32, 1> A_log{nullptr}, dt_bias{nullptr};
  Tc01View<StateDType, value_heads * Dim * Dim> state;
  Tc01View<BF16, H * Dim> q, k;
  Tc01View<BF16, value_heads * Dim> v, out;
  Tc01View<F32, value_heads> a, b;
  Tc01View<BF16, 2 * Cap * value_heads * Dim, Cap * value_heads * Dim, value_heads * Dim> values;
  Tc01View<BF16, 2 * Cap * H * Dim, Cap * H * Dim, H * Dim> keys;
  Tc01View<F32, 2 * Cap * value_heads, Cap * value_heads> gates;
  int batch, state_slots, replay_slots, null_slot;
  float scale, epsilon;

  static bool matches(const CachedVerify& p) {
    return (reinterpret_cast<uintptr_t>(p.state.data) & 15u) == 0 &&
        ((reinterpret_cast<uintptr_t>(p.q.data) | reinterpret_cast<uintptr_t>(p.k.data)) & 7u) == 0 &&
        p.materialize_overflow && p.processed_gates && p.heads == H && p.value_heads == value_heads &&
        !p.offsets.data && p.slots.data == p.replay.data && p.slots.dtype == I32 && p.replay.dtype == I32 &&
        p.lengths.dtype == I32 && p.buffers.dtype == I32 && p.slots.stride[0] == 1 &&
        p.replay.stride[0] == 1 && p.lengths.stride[0] == 1 && p.buffers.stride[0] == 1 &&
        p.a.dtype == F32 && p.b.dtype == F32 && p.state.stride[0] == value_heads * Dim * Dim &&
        p.q.stride[0] == H * Dim && p.k.stride[0] == H * Dim && p.v.stride[0] == value_heads * Dim &&
        p.out.stride[0] == value_heads * Dim && p.a.stride[0] == value_heads && p.b.stride[0] == value_heads &&
        p.values.stride[0] == 2 * Cap * value_heads * Dim && p.values.stride[1] == Cap * value_heads * Dim &&
        p.values.stride[2] == value_heads * Dim && p.keys.stride[0] == 2 * Cap * H * Dim &&
        p.keys.stride[1] == Cap * H * Dim && p.keys.stride[2] == H * Dim &&
        p.gates.stride[0] == 2 * Cap * value_heads && p.gates.stride[1] == Cap * value_heads;
  }

  static Tc01Dense pack(const CachedVerify& p) {
    return {{p.slots.data}, {p.lengths.data}, {p.buffers.data}, {p.state.data},
            {p.q.data}, {p.k.data}, {p.v.data}, {p.out.data}, {p.a.data}, {p.b.data},
            {p.values.data}, {p.keys.data}, {p.gates.data}, p.batch, p.state_slots, p.replay_slots,
            p.null_slot, p.scale, p.epsilon};
  }
};
