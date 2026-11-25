// SPDX-License-Identifier: Apache-2.0
#include "../../gdn_replayssm/common/gdn_replayssm_host.h"
#include "gdn_replayssm_hlm_abi.h"

namespace pla::gdn_replayssm::host {
namespace {

void cached_verify(T q, T k, T v, T a, T b,
                   OptionalTensor A_log, OptionalTensor dt_bias,
                   T state, T values, T keys, T gates, T out,
                   OptionalTensor starts, T slots, T indices, T lengths,
                   OptionalTensor offsets, T buffers, int64_t max_steps,
                   std::optional<double> scale, int64_t null_slot,
                   int norm_policy, double epsilon, bool processed_gates,
                   bool tf32, OptionalTensor provided_workspace,
                   bool materialize_overflow) {
  TORCH_CHECK(A_log.has_value() == dt_bias.has_value(),
              "A_log and dt_bias must be supplied together");
  TORCH_CHECK(processed_gates || A_log.has_value(),
              "raw gates require FP32 A_log and dt_bias");
  TORCH_CHECK(!materialize_overflow || !starts,
              "materialize_overflow requires fixed-T input without query_start_loc");
  replay::CachedVerify p{};
  p.materialize_overflow = materialize_overflow;
  p.max_steps = integer(max_steps, "max_verify_len", 1);
  TORCH_CHECK(p.max_steps <= 8, "cached max_verify_len must be in [1, 8]");
  p.null_slot = null_index(null_slot);
  p.epsilon = checked_epsilon(norm_policy, epsilon);
  p.norm_policy = norm_policy;
  p.processed_gates = processed_gates;
  p.tf32 = tf32;
  Checks check(q);
  p.q = check.tensor(q, "q", 3, 2, Kind::Floating);
  p.k = check.tensor(k, "k", 3, 2, Kind::Floating);
  p.v = check.tensor(v, "v", 3, 2, Kind::Floating);
  p.a = check.tensor(a, "a", 2, 1, Kind::Floating);
  p.b = check.tensor(b, "b", 2, 1, Kind::Floating);
  if (A_log) {
    p.A_log = check.tensor(*A_log, "A_log", 1, 1, Kind::Float32);
    p.dt_bias = check.tensor(*dt_bias, "dt_bias", 1, 1, Kind::Float32);
  }
  p.state = check.tensor(state, "checkpoint_state", 4, 3, Kind::Floating,
                         materialize_overflow);
  p.values = check.tensor(values, "history.values", 5, 3, Kind::Floating, true);
  p.keys = check.tensor(keys, "history.keys", 5, 3, Kind::Floating, true);
  p.gates = check.tensor(gates, "history.gates", 4, 2, Kind::Float32, true);
  p.out = check.tensor(out, "out", 3, 2, Kind::Floating, true);
  if (starts) p.starts = check.tensor(*starts, "query_start_loc", 1, 0, Kind::Index);
  p.slots = check.tensor(slots, "state_indices", 1, 0, Kind::Index);
  p.replay = check.tensor(indices, "replay_indices", 1, 0, Kind::Index);
  p.lengths = check.tensor(lengths, "lengths", 1, 0, Kind::Index);
  if (offsets) p.offsets = check.tensor(*offsets, "offsets", 1, 0, Kind::Index);
  p.buffers = check.tensor(buffers, "buffers", 1, 0, Kind::Index);
  p.tokens = integer(q.size(0), "total_tokens");
  p.batch = integer(slots.size(0), "batch");
  p.heads = integer(q.size(1), "H", 1);
  p.value_heads = integer(v.size(1), "HV", 1);
  p.key_dim = integer(q.size(2), "K", 1);
  p.value_dim = integer(v.size(2), "V", 1);
  p.state_slots = integer(state.size(0), "state slots");
  p.replay_slots = integer(values.size(0), "replay slots");
  p.capacity = integer(values.size(2), "history capacity", 1);
  TORCH_CHECK(p.capacity <= 16 && p.max_steps <= p.capacity,
              "cached history requires max_verify_len <= capacity <= 16");
  TORCH_CHECK(p.key_dim <= 256 && p.value_heads % p.heads == 0,
              "K must be <= 256 and HV must be divisible by H");
  const double actual_scale = scale.value_or(std::pow(p.key_dim, -0.5));
  TORCH_CHECK(std::isfinite(actual_scale) &&
                  std::abs(actual_scale) <= std::numeric_limits<float>::max(),
              "scale must be finite and representable as float32");
  p.scale = static_cast<float>(actual_scale);
  shape(k, {p.tokens, p.heads, p.key_dim}, "k");
  shape(v, {p.tokens, p.value_heads, p.value_dim}, "v");
  shape(a, {p.tokens, p.value_heads}, "a");
  shape(b, {p.tokens, p.value_heads}, "b");
  if (A_log) {
    shape(*A_log, {p.value_heads}, "A_log");
    shape(*dt_bias, {p.value_heads}, "dt_bias");
  }
  shape(state, {p.state_slots, p.value_heads, p.value_dim, p.key_dim}, "checkpoint_state");
  shape(values, {p.replay_slots, 2, p.capacity, p.value_heads, p.value_dim}, "history.values");
  shape(keys, {p.replay_slots, 2, p.capacity, p.heads, p.key_dim}, "history.keys");
  shape(gates, {p.replay_slots, 2, p.value_heads, p.capacity}, "history.gates");
  shape(out, {p.tokens, p.value_heads, p.value_dim}, "out");
  shape(indices, {p.batch}, "replay_indices");
  TORCH_CHECK(lengths.size(0) >= p.replay_slots && buffers.size(0) >= p.replay_slots &&
                  (!offsets || offsets->size(0) >= p.replay_slots),
              "cached metadata must cover every replay slot");
  if (starts) {
    shape(*starts, {int64_t(p.batch) + 1}, "query_start_loc");
  } else {
    TORCH_CHECK(p.tokens == multiply(p.batch, p.max_steps),
                "fixed-T cached input requires batch * max_verify_len tokens");
  }
  same_dtype(k, q, "k");
  same_dtype(v, q, "v");
  same_dtype(out, q, "out");
  same_dtype(values, q, "history.values");
  same_dtype(keys, q, "history.keys");
  TORCH_CHECK(state.scalar_type() == q.scalar_type() || state.scalar_type() == at::kFloat,
              "checkpoint_state must have the input dtype or float32");
  if (provided_workspace) {
    // Unused, but still enforce the public writable-workspace alias contract.
    check.tensor(*provided_workspace, "workspace", 1, 1, Kind::Float32, true);
    TORCH_CHECK(provided_workspace->is_contiguous(), "workspace must be contiguous");
  }
  TORCH_CHECK(p.batch <= 65535 && p.value_heads <= 65535,
              "cached verify launch grid exceeds CUDA y/z limits");
  TORCH_CHECK(p.value_dim <= std::numeric_limits<int>::max() - 15,
              "cached verify value grid exceeds the supported integer range");
  integer(multiply(p.value_heads, p.value_dim), "HV * V", 1);
  // cached_verify_kernel uses fixed MAX_T=8, MAX_H=16, MAX_K=256 arrays:
  // q/k/stored_k/kh, gates, and new/history projection coefficients.
  constexpr int64_t shared_elements = (3 * 8 + 16) * 256 + 16 + 3 * 8 + 1 +
                                      2 * 8 * 8 + 2 * 8 * 16;
  TORCH_CHECK(multiply(shared_elements, sizeof(float)) <= 48 * 1024,
              "cached verify exceeds 48KiB shared memory");
  replay::Materialize fold{};
  if (materialize_overflow) {
    // Views only: replay-slot metadata remains on device and is never advanced.
    fold.state = pack(state.unsqueeze(0));
    fold.values = pack(values.unsqueeze(0));
    fold.keys = pack(keys.unsqueeze(0));
    fold.gates = pack(gates.unsqueeze(0));
    fold.state_indices = p.slots;
    fold.replay_indices = fold.metadata_indices = p.replay;
    fold.lengths = p.lengths;
    fold.starts = p.offsets;
    fold.buffers = p.buffers;
    fold.layers = 1;
    fold.batch = p.batch;
    fold.heads = p.heads;
    fold.value_heads = p.value_heads;
    fold.key_dim = p.key_dim;
    fold.value_dim = p.value_dim;
    fold.capacity = p.capacity;
    fold.state_slots = p.state_slots;
    fold.replay_slots = p.replay_slots;
    fold.metadata_slots = integer(lengths.size(0), "metadata slots");
    fold.null_slot = p.null_slot;
    fold.format = replay::CachedUKG;
    fold.norm_policy = p.norm_policy;
    fold.epsilon = p.epsilon;
    fold.overflow_steps = p.max_steps;
    const int64_t fold_shared = add(multiply(p.capacity, p.key_dim + 17), 1);
    TORCH_CHECK(multiply(fold_shared, sizeof(float)) <= 48 * 1024,
                "cached overflow materialize exceeds 48KiB shared memory");
  }
  check.aliases();
  const c10::cuda::CUDAGuard guard(q.device());
  if (p.batch != 0) {
    const auto stream = pla::current_stream();
    const bool folded = replay::launch_cached_verify(p, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (materialize_overflow && !folded) replay::launch_materialize(fold, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void bind_hlm(pybind11::module_& m) {
  m.def("cached_verify", &cached_verify);
}

}  // namespace pla::gdn_replayssm::host
