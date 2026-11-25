// SPDX-License-Identifier: Apache-2.0
// The actual row and bank strides are preserved. Inner shapes are fixed by
// the dispatch guard and already validated by the owning cached host wrapper.

template <int Rank>
struct CachedView {
  void* data;
  int64_t stride[Rank];
  DType dtype;
};

template <int Rank>
CachedView<Rank> cached_view(const Tensor& tensor) {
  CachedView<Rank> view{};
  view.data = tensor.data;
  view.dtype = tensor.dtype;
  for (int axis = 0; axis < Rank; ++axis) view.stride[axis] = tensor.stride[axis];
  return view;
}

struct CachedT1 {
  CachedView<1> slots, replay, lengths, buffers, offsets;
  CachedView<1> state, q, k, v, a, b, A_log, dt_bias, out;
  CachedView<3> values, keys;
  CachedView<2> gates;
  int heads, value_heads, state_slots, replay_slots, null_slot, norm_policy;
  float scale, epsilon;
  bool processed_gates, materialize_overflow, shared_mapping;
};

CachedT1 pack_t1(const CachedVerify& p) {
  const bool shared_mapping = p.slots.data == p.replay.data &&
      p.slots.dtype == p.replay.dtype && p.slots.stride[0] == p.replay.stride[0];
  return CachedT1{
      cached_view<1>(p.slots), cached_view<1>(p.replay), cached_view<1>(p.lengths),
      cached_view<1>(p.buffers), cached_view<1>(p.offsets), cached_view<1>(p.state),
      cached_view<1>(p.q), cached_view<1>(p.k), cached_view<1>(p.v), cached_view<1>(p.a),
      cached_view<1>(p.b), cached_view<1>(p.A_log), cached_view<1>(p.dt_bias), cached_view<1>(p.out),
      cached_view<3>(p.values), cached_view<3>(p.keys), cached_view<2>(p.gates),
      p.heads, p.value_heads, p.state_slots, p.replay_slots, p.null_slot, p.norm_policy,
      p.scale, p.epsilon, p.processed_gates, p.materialize_overflow, shared_mapping};
}
