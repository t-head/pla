// SPDX-License-Identifier: Apache-2.0
#include "gdn_replayssm_host.h"
#include "../../gdn_replayssm_sglang/common/gdn_replayssm_sglang_abi.h"

namespace pla::gdn_replayssm::host {
namespace {

void verify(T q, T k, T v, T a, T b, OptionalTensor A_log, OptionalTensor dt_bias,
            T state, T d, T keys, T g, OptionalTensor rawv, OptionalTensor rawk,
            OptionalTensor beta, T out, OptionalTensor starts, T slots, T indices,
            T wp, T base, OptionalTensor flush, int64_t length, int64_t max_steps,
            std::optional<double> scale, bool normalize, int64_t null_slot,
            int mode, bool tf32, OptionalTensor provided_workspace,
            bool allow_workspace_allocation, int norm_policy, double epsilon,
            bool processed_gates, bool legacy_dot_precision) {
  TORCH_CHECK(mode >= 0 && mode <= 2, "invalid launch mode");
  TORCH_CHECK(A_log.has_value() == dt_bias.has_value(),
              "A_log and dt_bias must be supplied together");
  TORCH_CHECK(processed_gates || A_log.has_value(),
              "raw gates require FP32 A_log and dt_bias");
  TORCH_CHECK(mode == 1 || flush.has_value(), "exact-fold launch requires is_flush");
  replay::Verify p{};
  p.mode = mode;
  p.exact = rawv.has_value() && rawk.has_value() && beta.has_value();
  p.residual = rawv.has_value() && rawk.has_value() && !beta.has_value();
  TORCH_CHECK(mode == 1 || p.exact, "raw replay buffers are required for exact-fold launch");
  p.capacity = capacity(length, "max_cache_len");
  // The 2*T reserve belongs to SGLang's flush scheduler, not common replay.
  p.max_steps = flush ? window(max_steps, length) : integer(max_steps, "max_verify_len", 1);
  TORCH_CHECK(p.max_steps <= 32, "max_verify_len must be in [1, 32]");
  p.null_slot = null_index(null_slot);
  p.epsilon = checked_epsilon(norm_policy, epsilon);
  p.norm_policy = normalize ? norm_policy : 0;
  p.normalize = p.norm_policy != 0;
  p.processed_gates = processed_gates;
  p.tf32 = tf32;
  p.legacy_dot_precision = legacy_dot_precision;
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
  p.state = check.tensor(state, "checkpoint_state", 4, 3, Kind::Floating, mode != 1);
  p.d = check.tensor(d, "d_cache", 4, 3, Kind::Floating, true);
  p.keys = check.tensor(keys, "k_cache", 4, 3, Kind::Floating, true);
  p.g = check.tensor(g, "g_cache", 3, 2, Kind::Float32, true);
  p.out = check.tensor(out, "out", 3, 2, Kind::Floating, true);
  if (starts) p.starts = check.tensor(*starts, "query_start_loc", 1, 0, Kind::Index);
  p.slots = check.tensor(slots, "ssm_state_indices", 1, 0, Kind::Index);
  p.replay = check.tensor(indices, "replay_indices", 1, 0, Kind::Index);
  p.wp = check.tensor(wp, "write_pos", 1, 0, Kind::Index);
  p.base = check.tensor(base, "cache_base", 1, 0, Kind::Index);
  if (flush) p.flush = check.tensor(*flush, "is_flush", 1, 0, Kind::Flush);
  p.tokens = integer(q.size(0), "total_tokens");
  p.batch = integer(slots.size(0), "batch");
  if (starts) {
    shape(*starts, {int64_t(p.batch) + 1}, "query_start_loc");
  } else {
    TORCH_CHECK(p.max_steps == 1 && p.tokens == p.batch,
                "decode without query_start_loc requires one token per mapping row");
  }
  p.heads = integer(q.size(1), "H", 1);
  p.value_heads = integer(v.size(1), "HV", 1);
  p.key_dim = integer(q.size(2), "K", 1);
  p.value_dim = integer(v.size(2), "V", 1);
  p.state_slots = integer(state.size(0), "state slots");
  p.replay_slots = integer(wp.size(0), "replay slots");
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
  shape(d, {p.replay_slots, p.value_heads, length, p.value_dim}, "d_cache");
  shape(keys, {p.replay_slots, p.heads, length, p.key_dim}, "k_cache");
  shape(g, {p.replay_slots, p.value_heads, length}, "g_cache");
  shape(out, {p.tokens, p.value_heads, p.value_dim}, "out");
  shape(indices, {p.batch}, "replay_indices");
  shape(base, {p.replay_slots}, "cache_base");
  if (flush) shape(*flush, {p.replay_slots}, "is_flush");
  same_dtype(k, q, "k");
  same_dtype(v, q, "v");
  same_dtype(out, q, "out");
  same_dtype(keys, d, "k_cache");
  if (rawv) {
    p.rawv = check.tensor(*rawv, "rawv_cache", 4, 3, Kind::Floating, p.exact || p.residual);
    shape(*rawv, {p.replay_slots, p.value_heads, length, p.value_dim}, "rawv_cache");
    same_dtype(*rawv, d, "rawv_cache");
  }
  if (rawk) {
    p.rawk = check.tensor(*rawk, "rawk_cache", 4, 3, Kind::Floating, p.exact || p.residual);
    shape(*rawk, {p.replay_slots, p.heads, length, p.key_dim}, "rawk_cache");
    same_dtype(*rawk, d, "rawk_cache");
  }
  if (beta) {
    p.beta = check.tensor(*beta, "beta_cache", 3, 2, Kind::Float32, p.exact);
    shape(*beta, {p.replay_slots, p.value_heads, length}, "beta_cache");
  }
  const int64_t head_work = int64_t(p.batch) * p.value_heads;
  const int threshold = p.max_steps == 4 ? 192 : p.max_steps >= 5 ? 64 : 96;
  const bool fast = mode == 1 && head_work >= threshold &&
      p.key_dim == 128 &&
      p.value_dim == 128 && (p.max_steps <= 4 || p.max_steps == 6 || p.max_steps == 8) && p.capacity <= 32 &&
      q.scalar_type() == at::kBFloat16 && state.scalar_type() == at::kFloat &&
      d.scalar_type() == at::kBFloat16;
  const int64_t workspace_elements = fast
      ? multiply(multiply(multiply(p.batch, p.value_heads), 4 * p.max_steps), 128) : 0;
  if (provided_workspace) {
    check.tensor(*provided_workspace, "workspace", 1, 1, Kind::Float32, true);
    TORCH_CHECK(provided_workspace->is_contiguous(), "workspace must be contiguous");
    if (fast) {
      TORCH_CHECK(provided_workspace->numel() >= workspace_elements,
                  "workspace is too small; expected at least ", workspace_elements,
                  " float32 elements");
      p.workspace = provided_workspace->data_ptr<float>();
    }
  }
  check.aliases();
  const c10::cuda::CUDAGuard guard(q.device());
  T flags;
  if (flush) {
    // SGLang alone supplies flags; preserve its existing int8 conversion.
    flags = flush->scalar_type() == at::kChar ? *flush : flush->to(at::kChar);
    p.flush = pack(flags);
  }
  T workspace;
  if (fast && !provided_workspace && allow_workspace_allocation) {
    workspace = torch::empty({workspace_elements}, q.options().dtype(at::kFloat));
    p.workspace = workspace.data_ptr<float>();
  }
  if (p.batch != 0) {
    const auto stream = pla::current_stream();
    if (p.mode == 1) replay::launch_verify(p, stream);
    else replay::launch_sglang_fold_and_verify(p, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void materialize(T state, T values, T keys, T gates,
                 OptionalTensor value_residual, OptionalTensor key_residual,
                 OptionalTensor beta, OptionalTensor state_indices,
                 OptionalTensor replay_indices, OptionalTensor lengths,
                 OptionalTensor starts, OptionalTensor buffers,
                 OptionalTensor metadata_indices, OptionalTensor targets,
                 OptionalTensor enabled, OptionalTensor items,
                 OptionalTensor n_items, OptionalTensor descriptors,
                 int format, int norm_policy, double epsilon, int64_t null_slot) {
  TORCH_CHECK(format >= replay::CompactDKG && format <= replay::CachedUKG,
              "invalid history format");
  TORCH_CHECK(value_residual.has_value() == key_residual.has_value(),
              "value_residual and key_residual must be supplied together");
  if (format == replay::CompactDKG) {
    TORCH_CHECK(!beta, "compact history must not contain beta");
  } else if (format == replay::RawWindow) {
    TORCH_CHECK(beta && !value_residual,
                "raw window history requires beta and forbids residuals");
  } else {
    TORCH_CHECK(!beta && !value_residual,
                "cached double history forbids residuals and extra beta");
  }
  if (items) {
    TORCH_CHECK(n_items && !state_indices && !replay_indices && !lengths &&
                    !starts && !buffers && !metadata_indices && !enabled,
                "items requires n_items and forbids indexed fields");
  } else {
    TORCH_CHECK(!n_items && state_indices && replay_indices && lengths,
                "indexed work requires state_indices, replay_indices and lengths");
  }
  replay::Materialize p{};
  p.format = static_cast<replay::HistoryFormat>(format);
  p.norm_policy = norm_policy;
  p.epsilon = checked_epsilon(norm_policy, epsilon);
  p.null_slot = null_index(null_slot);
  p.residual = value_residual.has_value();
  Checks check(state);
  p.state = check.tensor(state, "state_layers", descriptors ? 4 : 5, 3,
                         Kind::Floating, true);
  if (descriptors) {
    p.descriptors = check.tensor(*descriptors, "state_descriptors", 2, 2, Kind::Int64);
    TORCH_CHECK(descriptors->is_contiguous(), "state_descriptors must be contiguous");
    p.layers = integer(descriptors->size(0), "layers", 1);
    shape(*descriptors, {p.layers, 2}, "state_descriptors");
    p.state_slots = integer(state.size(0), "state slots");
    // Normalize the reference layout without copying state or reading GPU pointers.
    for (int i = 4; i > 0; --i) p.state.stride[i] = p.state.stride[i - 1];
    p.state.stride[0] = 0;
  } else {
    p.layers = integer(state.size(0), "layers", 1);
    p.state_slots = integer(state.size(1), "state slots");
  }
  p.value_heads = integer(state.size(-3), "HV", 1);
  p.value_dim = integer(state.size(-2), "V", 1);
  p.key_dim = integer(state.size(-1), "K", 1);
  TORCH_CHECK(p.key_dim <= 256, "K must be in [1, 256]");
  const bool double_buffer = format == replay::CachedUKG;
  const int vector_rank = double_buffer ? 6 : 5;
  const int gate_rank = double_buffer ? 5 : 4;
  p.values = check.tensor(values, "history.values", vector_rank, 3, Kind::Floating);
  p.keys = check.tensor(keys, "history.keys", vector_rank, 3, Kind::Floating);
  p.gates = check.tensor(gates, "history.gates", gate_rank, 2, Kind::Float32);
  p.replay_slots = integer(values.size(1), "replay slots");
  p.heads = integer(keys.size(double_buffer ? 4 : 2), "H", 1);
  p.capacity = format == replay::CompactDKG ? capacity(values.size(3), "history capacity")
      : integer(values.size(3), "history capacity", 1);
  TORCH_CHECK(p.capacity <= 64, "materialize history capacity must be <= 64");
  TORCH_CHECK(p.value_heads % p.heads == 0, "HV must be divisible by H");
  if (double_buffer) {
    shape(values, {p.layers, p.replay_slots, 2, p.capacity, p.value_heads, p.value_dim},
          "history.values");
    shape(keys, {p.layers, p.replay_slots, 2, p.capacity, p.heads, p.key_dim}, "history.keys");
    shape(gates, {p.layers, p.replay_slots, 2, p.value_heads, p.capacity}, "history.gates");
  } else {
    shape(values, {p.layers, p.replay_slots, p.value_heads, p.capacity, p.value_dim},
          "history.values");
    shape(keys, {p.layers, p.replay_slots, p.heads, p.capacity, p.key_dim}, "history.keys");
    shape(gates, {p.layers, p.replay_slots, p.value_heads, p.capacity}, "history.gates");
  }
  same_dtype(keys, values, "history.keys");
  if (value_residual) {
    p.value_residual = check.tensor(*value_residual, "history.value_residual", 5, 3, Kind::Floating);
    p.key_residual = check.tensor(*key_residual, "history.key_residual", 5, 3, Kind::Floating);
    shape(*value_residual, {p.layers, p.replay_slots, p.value_heads, p.capacity, p.value_dim},
          "history.value_residual");
    shape(*key_residual, {p.layers, p.replay_slots, p.heads, p.capacity, p.key_dim},
          "history.key_residual");
    same_dtype(*value_residual, values, "history.value_residual");
    same_dtype(*key_residual, values, "history.key_residual");
  }
  if (beta) {
    p.beta = check.tensor(*beta, "history.beta", 4, 2, Kind::Float32);
    shape(*beta, {p.layers, p.replay_slots, p.value_heads, p.capacity}, "history.beta");
  }
  auto optional_vector = [&](const OptionalTensor& t, const char* name,
                             int64_t size, Kind kind) {
    if (!t) return replay::Tensor{};
    const auto result = check.tensor(*t, name, 1, 0, kind);
    shape(*t, {size}, name);
    return result;
  };
  if (items) {
    p.items = check.tensor(*items, "items", 2, 0, Kind::Index);
    p.batch = integer(items->size(0), "work rows");
    shape(*items, {p.batch, 4}, "items");
    TORCH_CHECK(n_items->numel() == 1 && n_items->dim() <= 1,
                "n_items must be a scalar or a one-element vector");
    p.n_items = check.tensor(*n_items, "n_items", n_items->dim(), 0, Kind::Index);
  } else {
    // State/replay mappings always use work row; only history metadata is indirect.
    p.state_indices = check.tensor(*state_indices, "state_indices", 1, 0, Kind::Index);
    p.batch = integer(state_indices->size(0), "work rows");
    p.replay_indices = optional_vector(replay_indices, "replay_indices", p.batch, Kind::Index);
    p.lengths = check.tensor(*lengths, "lengths", 1, 0, Kind::Index);
    p.metadata_slots = integer(lengths->size(0), "metadata slots");
    TORCH_CHECK(metadata_indices || p.metadata_slots >= p.batch,
                "row-indexed metadata must cover every work row");
    p.metadata_indices = optional_vector(metadata_indices, "metadata_indices", p.batch, Kind::Index);
    p.starts = optional_vector(starts, "starts", p.metadata_slots, Kind::Index);
    p.buffers = optional_vector(buffers, "buffers", p.metadata_slots, Kind::Index);
    p.enabled = optional_vector(enabled, "enabled", p.metadata_slots, Kind::Flush);
  }
  p.targets = optional_vector(targets, "target_state_indices", p.batch, Kind::Index);
  TORCH_CHECK(p.batch <= 65535 && multiply(p.layers, p.value_heads) <= 65535,
              "materialize launch grid exceeds CUDA y/z limits");
  const int64_t shared_elements = format == replay::RawWindow ? p.key_dim + 35
      : add(multiply(p.capacity, p.key_dim + 17), 1);
  TORCH_CHECK(multiply(shared_elements, sizeof(float)) <= 48 * 1024,
              "materialize exceeds 48KiB shared memory");
  check.aliases();
  const c10::cuda::CUDAGuard guard(state.device());
  if (p.batch != 0) replay::launch_materialize(p, pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

}  // namespace pla::gdn_replayssm::host

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pla::gdn_replayssm::host;
  m.def("verify", &verify);
  m.def("materialize", &materialize);
  bind_hlm(m);
  bind_sglang(m);
}
