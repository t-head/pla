// SPDX-License-Identifier: Apache-2.0
#pragma once

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

#include "../../../ppu_torch_stream.h"
#include "gdn_replayssm_abi.h"

namespace pla::gdn_replayssm::host {

using T = torch::Tensor;
using OptionalTensor = std::optional<T>;
namespace replay = pla::gdn_replayssm;
enum class Kind { Floating, Float32, Index, Int64, Flush };

inline int64_t multiply(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 &&
                  (b == 0 || a <= std::numeric_limits<int64_t>::max() / b),
              "tensor size/stride product exceeds int64");
  return a * b;
}

inline int64_t add(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 &&
                  a <= std::numeric_limits<int64_t>::max() - b,
              "tensor address span exceeds int64");
  return a + b;
}

inline int integer(int64_t value, const char* name, int64_t minimum = 0) {
  TORCH_CHECK(value >= minimum && value <= std::numeric_limits<int>::max(),
              name, " is outside the supported range");
  return static_cast<int>(value);
}

inline int null_index(int64_t value) {
  return integer(value, "null_block_id", std::numeric_limits<int>::min());
}

inline int capacity(int64_t value, const char* name) {
  const int result = integer(value, name, 1);
  TORCH_CHECK((value & (value - 1)) == 0, name, " must be a power of two");
  return result;
}

inline int window(int64_t steps, int64_t length) {
  TORCH_CHECK(steps > 0 && steps <= 32, "max_spec_len must be in [1, 32]");
  TORCH_CHECK(length >= 2 * steps,
              "max_cache_len must be at least 2 * max_spec_len");
  return static_cast<int>(steps);
}

inline void shape(const T& t, std::initializer_list<int64_t> sizes, const char* name) {
  TORCH_CHECK(t.sizes() == at::IntArrayRef(sizes), name, " has an invalid shape");
}

inline void same_dtype(const T& a, const T& b, const char* name) {
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), name, " has an incompatible dtype");
}

inline replay::DType dtype(const T& t) {
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

inline replay::Tensor pack(const T& t) {
  replay::Tensor result{};
  TORCH_CHECK(t.dim() <= 6, "GDN replay tensors support at most six dimensions");
  result.data = t.data_ptr();
  result.dtype = dtype(t);
  for (int i = 0; i < t.dim(); ++i) result.stride[i] = t.stride(i);
  return result;
}

inline float checked_epsilon(int norm_policy, double epsilon) {
  TORCH_CHECK(norm_policy >= 0 && norm_policy <= 2, "invalid norm_policy");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0 &&
                  epsilon <= std::numeric_limits<float>::max(),
              "epsilon must be finite, nonnegative and representable as float32");
  return static_cast<float>(epsilon);
}

inline c10::Device checked_device(const T& t) {
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

void bind_hlm(pybind11::module_&);
void bind_sglang(pybind11::module_&);

}  // namespace pla::gdn_replayssm::host
