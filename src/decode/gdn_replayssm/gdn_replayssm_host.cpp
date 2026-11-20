// SPDX-License-Identifier: Apache-2.0
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "../../ppu_torch_stream.h"
#include "gdn_replayssm_abi.h"

namespace {
using T = torch::Tensor;
using OptionalTensor = std::optional<T>;
namespace replay = pla::gdn_replayssm;
enum class Kind { Floating, Float32, Index, Int64, Flush };

int64_t multiply(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 &&
                  (b == 0 || a <= std::numeric_limits<int64_t>::max() / b),
              "tensor size/stride product exceeds int64");
  return a * b;
}

int64_t add(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 &&
                  a <= std::numeric_limits<int64_t>::max() - b,
              "tensor address span exceeds int64");
  return a + b;
}

int integer(int64_t value, const char* name, int64_t minimum = 0) {
  TORCH_CHECK(value >= minimum && value <= std::numeric_limits<int>::max(),
              name, " is outside the supported range");
  return static_cast<int>(value);
}

int null_index(int64_t value) {
  return integer(value, "null_block_id", std::numeric_limits<int>::min());
}

int capacity(int64_t value, const char* name) {
  const int result = integer(value, name, 1);
  TORCH_CHECK((value & (value - 1)) == 0, name, " must be a power of two");
  return result;
}

int window(int64_t steps, int64_t length) {
  TORCH_CHECK(steps > 0 && steps <= 32, "max_spec_len must be in [1, 32]");
  TORCH_CHECK(length >= 2 * steps,
              "max_cache_len must be at least 2 * max_spec_len");
  return static_cast<int>(steps);
}

void shape(const T& t, std::initializer_list<int64_t> sizes, const char* name) {
  TORCH_CHECK(t.sizes() == at::IntArrayRef(sizes), name, " has an invalid shape");
}

void same_dtype(const T& a, const T& b, const char* name) {
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), name, " has an incompatible dtype");
}

replay::DType dtype(const T& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return replay::F32;
    case at::kHalf: return replay::F16;
    case at::kBFloat16: return replay::BF16;
    case at::kInt: return replay::I32;
    case at::kLong: return replay::I64;
    case at::kChar: return replay::I8;
    default: TORCH_CHECK(false, "unsupported tensor dtype");
  }
}

replay::Tensor pack(const T& t) {
  replay::Tensor result{};
  TORCH_CHECK(t.dim() <= 6, "GDN replay tensors support at most six dimensions");
  result.data = t.data_ptr();
  result.dtype = dtype(t);
  for (int i = 0; i < t.dim(); ++i) result.stride[i] = t.stride(i);
  return result;
}

float checked_epsilon(int norm_policy, double epsilon) {
  TORCH_CHECK(norm_policy >= 0 && norm_policy <= 2, "invalid norm_policy");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0 &&
                  epsilon <= std::numeric_limits<float>::max(),
              "epsilon must be finite, nonnegative and representable as float32");
  return static_cast<float>(epsilon);
}

c10::Device checked_device(const T& t) {
  TORCH_CHECK(t.defined() && t.is_cuda(), "GDN replay requires CUDA tensors");
  return t.device();
}

class Checks {
  struct Entry { T tensor; const char* name; bool writes; };
  c10::Device device_;
  std::vector<Entry> entries_;

 public:
  explicit Checks(const T& reference) : device_(checked_device(reference)) {}

  replay::Tensor tensor(const T& t, const char* name, int rank, int inner,
                        Kind kind, bool writes = false) {
    TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == device_,
                name, " must be on the same CUDA device");
    TORCH_CHECK(t.layout() == at::kStrided && t.dim() == rank &&
                    !t.is_conj() && !t.is_neg(), name, " has an unsupported layout");
    const auto type = t.scalar_type();
    const bool floating = type == at::kFloat || type == at::kHalf || type == at::kBFloat16;
    const bool index = type == at::kInt || type == at::kLong;
    TORCH_CHECK((kind == Kind::Floating && floating) ||
                    (kind == Kind::Float32 && type == at::kFloat) ||
                    (kind == Kind::Index && index) ||
                    (kind == Kind::Int64 && type == at::kLong) ||
                    (kind == Kind::Flush && (index || type == at::kChar)),
                name, " has an unsupported dtype");
    int64_t expected = 1;
    int64_t span = t.numel() == 0 ? 0 : 1;
    std::vector<std::pair<int64_t, int64_t>> axes;
    for (int i = rank - 1; i >= 0; --i) {
      TORCH_CHECK(t.stride(i) >= 0, name, " has a negative stride");
      if (i >= rank - inner) {
        TORCH_CHECK(t.size(i) <= 1 || t.stride(i) == expected,
                    name, " must have contiguous inner dimensions");
        expected = multiply(expected, t.size(i));
      }
      if (span != 0) span = add(span, multiply(t.size(i) - 1, t.stride(i)));
      if (t.size(i) > 1) axes.emplace_back(t.stride(i), t.size(i));
    }
    if (span != 0) {
      const int64_t end = multiply(add(t.storage_offset(), span), t.element_size());
      TORCH_CHECK(static_cast<uint64_t>(end) <= t.storage().nbytes(),
                  name, " exceeds its backing storage");
    }
    if (writes && span != 0) {
      std::sort(axes.begin(), axes.end());
      int64_t occupied = 1;
      for (const auto& axis : axes) {
        TORCH_CHECK(axis.first >= occupied, name, " has overlapping writable elements");
        occupied = add(occupied, multiply(axis.first, axis.second - 1));
      }
    }
    entries_.push_back({t, name, writes});
    return pack(t);
  }

  void aliases() const {
    for (size_t i = 0; i < entries_.size(); ++i) {
      for (size_t j = i + 1; j < entries_.size(); ++j) {
        const auto& a = entries_[i];
        const auto& b = entries_[j];
        if (!(a.writes || b.writes) || !a.tensor.numel() || !b.tensor.numel()) continue;
        const auto overlap = at::get_overlap_status(a.tensor, b.tensor);
        bool disjoint = overlap == at::MemOverlapStatus::No;
        if (overlap == at::MemOverlapStatus::TooHard) {
          auto interval = [](const T& t) {
            int64_t span = 1;
            for (int dim = 0; dim < t.dim(); ++dim)
              span = add(span, multiply(t.size(dim) - 1, t.stride(dim)));
            const auto start = reinterpret_cast<uintptr_t>(t.data_ptr());
            return std::pair<uintptr_t, uintptr_t>{start, start + multiply(span, t.element_size())};
          };
          const auto ai = interval(a.tensor), bi = interval(b.tensor);
          disjoint = ai.second <= bi.first || bi.second <= ai.first;
        }
        TORCH_CHECK(disjoint, a.name, " and ", b.name,
                    " overlap or cannot be proven disjoint across a writable tensor");
      }
    }
  }
};

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
  if (p.batch != 0) replay::launch_verify(p, pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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

void commit(T state, T d, T keys, T g, OptionalTensor dr, OptionalTensor kr,
            T slots, T indices, T wp, T base, T flush, T accepted,
            OptionalTensor track, OptionalTensor steps, int64_t null_slot) {
  TORCH_CHECK(dr.has_value() == kr.has_value(), "residual caches must be supplied together");
  TORCH_CHECK(track.has_value() == steps.has_value(), "tracking indices and steps must be supplied together");
  replay::Commit p{};
  p.residual = dr.has_value();
  p.tracking = track.has_value();
  p.null_slot = null_index(null_slot);
  Checks check(state);
  p.state = check.tensor(state, "checkpoint_state", 5, 3, Kind::Floating, true);
  p.d = check.tensor(d, "d_cache", 5, 3, Kind::Floating);
  p.keys = check.tensor(keys, "k_cache", 5, 3, Kind::Floating);
  p.g = check.tensor(g, "g_cache", 4, 2, Kind::Float32);
  p.slots = check.tensor(slots, "state_batch_indices", 1, 0, Kind::Index);
  p.replay = check.tensor(indices, "replay_indices", 1, 0, Kind::Index);
  p.wp = check.tensor(wp, "write_pos", 1, 0, Kind::Index, true);
  p.base = check.tensor(base, "cache_base", 1, 0, Kind::Index, true);
  p.flush = check.tensor(flush, "is_flush", 1, 0, Kind::Flush, true);
  p.accepted = check.tensor(accepted, "accept_lens", 1, 0, Kind::Index);
  p.layers = integer(state.size(0), "layers", 1);
  p.state_slots = integer(state.size(1), "state slots");
  p.replay_slots = integer(wp.size(0), "replay slots");
  p.batch = integer(slots.size(0), "batch");
  p.heads = integer(keys.size(2), "H", 1);
  p.value_heads = integer(state.size(2), "HV", 1);
  p.value_dim = integer(state.size(3), "V", 1);
  p.key_dim = integer(state.size(4), "K", 1);
  p.capacity = capacity(d.size(3), "cache capacity");
  TORCH_CHECK(p.key_dim <= 256 && (p.key_dim & (p.key_dim - 1)) == 0,
              "commit K must be a power of two in [1, 256]");
  TORCH_CHECK(p.value_heads % p.heads == 0, "HV must be divisible by H");
  shape(d, {p.layers, p.replay_slots, p.value_heads, p.capacity, p.value_dim}, "d_cache");
  shape(keys, {p.layers, p.replay_slots, p.heads, p.capacity, p.key_dim}, "k_cache");
  shape(g, {p.layers, p.replay_slots, p.value_heads, p.capacity}, "g_cache");
  shape(indices, {p.batch}, "replay_indices");
  shape(accepted, {p.batch}, "accept_lens");
  shape(base, {p.replay_slots}, "cache_base");
  shape(flush, {p.replay_slots}, "is_flush");
  same_dtype(keys, d, "k_cache");
  if (dr) {
    p.dr = check.tensor(*dr, "d_residual_cache", 5, 3, Kind::Floating);
    p.kr = check.tensor(*kr, "k_residual_cache", 5, 3, Kind::Floating);
    shape(*dr, {p.layers, p.replay_slots, p.value_heads, p.capacity, p.value_dim}, "d_residual_cache");
    shape(*kr, {p.layers, p.replay_slots, p.heads, p.capacity, p.key_dim}, "k_residual_cache");
    same_dtype(*dr, d, "d_residual_cache");
    same_dtype(*kr, d, "k_residual_cache");
  }
  if (track) {
    p.track = check.tensor(*track, "mamba_track_indices", 1, 0, Kind::Index);
    p.steps = check.tensor(*steps, "mamba_steps_to_track", 1, 0, Kind::Index);
    shape(*track, {p.batch}, "mamba_track_indices");
    shape(*steps, {p.batch}, "mamba_steps_to_track");
  }
  check.aliases();
  const c10::cuda::CUDAGuard guard(state.device());
  if (p.batch != 0) {
    replay::launch_sglang_materialize(p, pla::current_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    replay::launch_sglang_finish(p, pla::current_stream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cursors(T wp, T base, T flush, T accepted, T indices, int64_t length,
             int64_t max_steps, bool fold_every_commit,
             std::optional<int64_t> buffer_len, int64_t null_slot) {
  replay::Cursors p{};
  p.capacity = capacity(length, "max_cache_len");
  p.max_steps = window(max_steps, length);
  p.buffer_len = capacity(buffer_len.value_or(length), "cache_buf_len");
  TORCH_CHECK(p.buffer_len >= p.capacity, "cache_buf_len must be >= max_cache_len");
  p.null_slot = null_index(null_slot);
  p.fold_every_commit = fold_every_commit;
  Checks check(wp);
  p.wp = check.tensor(wp, "write_pos", 1, 0, Kind::Index, true);
  p.base = check.tensor(base, "cache_base", 1, 0, Kind::Index, true);
  p.flush = check.tensor(flush, "is_flush", 1, 0, Kind::Flush, true);
  p.accepted = check.tensor(accepted, "num_accepted", 1, 0, Kind::Index);
  p.replay = check.tensor(indices, "replay_indices", 1, 0, Kind::Index);
  p.batch = integer(indices.size(0), "batch");
  p.replay_slots = integer(wp.size(0), "replay slots");
  shape(base, {p.replay_slots}, "cache_base");
  shape(flush, {p.replay_slots}, "is_flush");
  shape(accepted, {p.batch}, "num_accepted");
  check.aliases();
  const c10::cuda::CUDAGuard guard(wp.device());
  if (p.batch != 0) replay::launch_cursors(p, pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("verify", &verify);
  m.def("cached_verify", &cached_verify);
  m.def("materialize", &materialize);
  m.def("commit", &commit);
  m.def("cursors", &cursors);
}
