/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// torch's CUDA compatibility headers must be included before the native hggc_runtime.h (launchers.h);
// otherwise macros defined first by the native host_defines.h break compilation of the compatibility vector_functions.h.
#include "runtime.h"
#include "../../../sanity_check.h"
#include "../../../../bypass_exec.h"
#include "../../../../op_forge_case.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "launchers.h"
#include "workspace_layout.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <tuple>

// Defined with the metadata stage; kept local to the runtime translation unit.
hggcError_t launch_fla_cp_device_geometry(const int32_t*, int32_t*, int64_t*, int, int,
                                          int, int, hggcStream_t);
hggcError_t launch_fla_zero_output_tail(const int32_t*, int, void*, int64_t, int64_t,
                                        hggcStream_t);
hggcError_t launch_fla_metadata_n1_tail(const void*, bool, int32_t*, int64_t*,
    int32_t*, int32_t*, int, int, int32_t*, int64_t*, int, int, int, int,
    void*, int64_t, hggcStream_t);
// Defined with the token stage: N=1 L2 + token fusion that also publishes qn/kn
// and, given w/u/kg, the inter stage's prepared operands and solved diagonal.
extern "C" hggcError_t launch_fla_token_l2__fla_h4(
    const void*, const void*, void*, void*, const float*, const uint16_t*, int64_t,
    void*, float*, void*, void*, void*, void*, float, const int32_t*, int, int,
    hggcStream_t);
extern "C" hggcError_t launch_fla_token_l2__fla_h8(
    const void*, const void*, void*, void*, const float*, const uint16_t*, int64_t,
    void*, float*, void*, void*, void*, void*, float, const int32_t*, int, int,
    hggcStream_t);
// Defined with the inter stage: launch_fla_inter_prepared without its prepare.
extern "C" hggcError_t launch_fla_inter_solve_prepared__fla_h4(
    const void*, const void*, const float*, const uint16_t*, int64_t, float*, void*,
    void*, void*, void*, void*, float, const int32_t*, const int32_t*, int, int, int,
    int, hggcStream_t);
extern "C" hggcError_t launch_fla_inter_solve_prepared__fla_h8(
    const void*, const void*, const float*, const uint16_t*, int64_t, float*, void*,
    void*, void*, void*, void*, float, const int32_t*, const int32_t*, int, int, int,
    int, hggcStream_t);

namespace fla_runtime {
namespace {
void cuda_check(hggcError_t status, const char* action) {
  TORCH_CHECK(status == hggcSuccess, action, ": ", hggcGetErrorString(status));
}
void tensor(const at::Tensor& x, const char* name, at::Device device,
            at::ScalarType dtype, at::IntArrayRef shape, int alignment = 16, bool contiguous = true) {
  if (!x.defined() || !x.is_cuda() || x.device() != device || x.scalar_type() != dtype ||
      x.sizes() != shape || (contiguous && !x.is_contiguous()) ||
      (x.numel() && reinterpret_cast<uintptr_t>(x.data_ptr()) % alignment))
    throw std::invalid_argument(std::string(name) + ": shape/device/dtype/stride/alignment mismatch");
}
bool alias(const at::Tensor& a, const at::Tensor& b) {
  return a.storage().data_ptr().get() == b.storage().data_ptr().get();
}
struct Shape { int64_t t, h, n, chunks; std::vector<int64_t> offsets; };
Shape validate(Arguments& a) {
  require(a.q.defined() && a.q.is_cuda() && a.q.dim() == 3 && a.q.size(2) == 128, "q must be a device tensor [T,H,128]");
  auto device = a.q.device();
  Shape s{a.q.size(0), a.q.size(1), 1, 0, {}};
  dimensions(s.t, s.h, s.n, 0);
  require(std::isfinite(a.scale) && std::abs(a.scale) <= std::numeric_limits<float>::max(), "scale must be a finite FP32");
  require(std::isfinite(a.lower_bound) && a.lower_bound >= -5 && a.lower_bound <= 0, "lower_bound must lie in [-5,0]");
  for (auto entry : {std::pair{"q", &a.q}, {"k", &a.k}, {"v", &a.v}, {"g", &a.g}, {"out", &a.out}})
    tensor(*entry.second, entry.first, device, at::kBFloat16, {s.t, s.h, 128});
  tensor(a.beta, "beta", device, at::kBFloat16, {s.t, s.h}, 2, false);
  require(a.beta.stride(1) == 1 && a.beta.stride(0) >= s.h && a.beta.stride(0) <= INT32_MAX,
          "beta only allows a token-dim stride");
  tensor(a.a_log, "A_log", device, at::kFloat, {s.h}, 4);
  tensor(a.dt_bias, "dt_bias", device, at::kFloat, {s.h, 128});
  if (a.cu.defined()) {
    require(a.cu.is_cuda() && a.cu.device() == device && a.cu.dim() == 1 && a.cu.numel() >= 2 &&
            a.cu.is_contiguous() && (a.cu.scalar_type() == at::kInt || a.cu.scalar_type() == at::kLong),
            "cu_seqlens must be a contiguous int32/int64 [N+1] on the same device");
    s.n = a.cu.numel() - 1;
  } else {
    require(!a.raw || *a.raw == std::vector<int64_t>{0, s.t}, "without cu_seqlens the raw offsets must be (0,T)");
    a.raw = std::vector<int64_t>{0, s.t};
  }
  dimensions(s.t, s.h, s.n, 0);
  if (a.raw) {
    require(a.raw->size() == size_t(s.n + 1), "raw offsets must contain N+1 elements");
    s.offsets = chunk_offsets(*a.raw, s.t);
    s.chunks = s.offsets.back();
  } else s.chunks = chunk_capacity(s.t, s.n);
  dimensions(s.t, s.h, s.n, s.chunks);
  if (a.pool.defined()) {
    require(!a.initial.defined() && !a.final.defined(), "state_pool is mutually exclusive with initial/final_state");
    require(a.indices.defined(), "state_pool requires state_indices");
    a.initial = a.final = a.pool;
  }
  if (a.indices.defined()) {
    require(a.initial.defined() || a.final.defined(), "state_indices requires a state buffer");
    tensor(a.indices, "state_indices", device, at::kInt, {s.n}, 4);
  }
  for (auto entry : {std::pair{"initial_state", &a.initial}, {"final_state", &a.final}}) {
    const auto& x = *entry.second;
    if (!x.defined()) continue;
    require(x.dim() == 4 && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat), "state must be BF16/FP32 [slots,H,128,128]");
    require(x.size(0) > 0 && (a.indices.defined() || x.size(0) == s.n), "state slot count mismatch");
    checked_mul(checked_mul(x.size(0), s.h), 128 * 128 * x.element_size());
    tensor(x, entry.first, device, x.scalar_type(), {x.size(0), s.h, 128, 128}, x.element_size());
  }
  require(!a.initial.defined() || !a.final.defined() || a.initial.scalar_type() == a.final.scalar_type(),
          "initial_state/final_state dtypes must match");
  for (const auto* x : {&a.q, &a.k, &a.v, &a.g, &a.beta, &a.a_log, &a.dt_bias}) {
    require(!a.out.numel() || !alias(a.out, *x), "out must not share storage with an input");
    for (const auto* state : {&a.initial, &a.final})
      require(!state->defined() || !alias(*state, *x), "state must not share storage with an activation");
  }
  for (const auto* state : {&a.initial, &a.final})
    require(!state->defined() || !alias(*state, a.out), "state must not share storage with out");
  return s;
}

struct CpSelection {
  int split = 0;
  int count = 0;
  int descriptor_row = -1;
  bool active() const { return count > 0; }
};

enum class CpAlgorithm { Legacy, State };
CpAlgorithm select_cp_algorithm(int segments, int heads, int64_t tokens) {
  const char* value = std::getenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO");
  if (!value || !value[0]) {
    const char* mode = std::getenv("PLA_FLASHKDA_TC01_FLA_CP");
    const bool automatic = !mode || !mode[0] || std::strcmp(mode, "auto") == 0;
    // Only promote measured AUTO buckets. Explicit algorithm
    // and split requests retain their existing semantics.
    if (automatic && tokens <= 65536 &&
        ((heads == 4 && segments == 1 && tokens > 0 && tokens <= 2048) ||
         (heads == 4 && segments == 5 && tokens >= 12288) ||
         ((heads == 6 || heads == 8) && segments == 1 && tokens > 0)))
      return CpAlgorithm::State;
    return CpAlgorithm::Legacy;
  }
  if (std::strcmp(value, "legacy") == 0) return CpAlgorithm::Legacy;
  if (std::strcmp(value, "state") == 0)
    return CpAlgorithm::State;
  // A one-segment request historically ignored other selector strings.
  if (segments == 1) return CpAlgorithm::Legacy;
  throw std::invalid_argument(
      "PLA_FLASHKDA_TC01_FLA_CP_ALGO must be legacy or state");
}

CpSelection select_cp(const Arguments& a, const Shape& s, const Launchers& table) {
  const char* mode = std::getenv("PLA_FLASHKDA_TC01_FLA_CP");
  const bool automatic = !mode || !mode[0] || std::strcmp(mode, "auto") == 0;
  if (!automatic && (std::strcmp(mode, "off") == 0 ||
                     std::strcmp(mode, "0") == 0)) return {};
  int split = 0;
  if (automatic) {
    if (s.n != 1 || !table.cp_recurrence_output) return {};
    if (!a.raw) {
      const char* algorithm = std::getenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO");
      const bool state = !algorithm || !algorithm[0] || std::strcmp(algorithm, "state") == 0;
      // H4 long device-offset buckets use a fixed CP5 launch whose descriptor
      // is rebuilt on device from the current effective length.
      if (s.h == 4 && (s.t == 16384 || s.t == 32768) && state)
        return {5, 5, cp_split_row(5)};
      if (((s.h == 8 && (s.t == 4096 || s.t == 8192)) ||
           (s.h == 8 && (s.t == 16384 || s.t == 32768))) && state)
        return {1, 1, -1};
      return {};
    }
    split = select_fla_cp_auto_split(s.h, (*a.raw)[1] - (*a.raw)[0]);
    // H4 AUTO buckets are qualified for state passing only. An explicit
    // legacy algorithm retains the serial AUTO path for both H4 buckets.
    const char* algorithm = std::getenv("PLA_FLASHKDA_TC01_FLA_CP_ALGO");
    if (s.h == 4 && split && algorithm && algorithm[0] &&
        std::strcmp(algorithm, "state") != 0) return {};
    if (!split) return {};
  } else {
    if (std::strcmp(mode, "1") == 0) split = 1;
    else if (std::strcmp(mode, "2") == 0) split = 2;
    else if (std::strcmp(mode, "3") == 0) split = 3;
    else if (std::strcmp(mode, "4") == 0) split = 4;
    else if (std::strcmp(mode, "5") == 0) split = 5;
    else if (std::strcmp(mode, "8") == 0) split = 8;
    else throw std::invalid_argument(
        "PLA_FLASHKDA_TC01_FLA_CP must be off, auto, 1, 2, 3, 4, 5, or 8");
    require(a.raw.has_value(), "forced FLA CP requires trusted raw_cu_seqlens");
    require(s.n == 1, "forced FLA CP currently supports exactly one sequence");
    require(table.cp_recurrence_output != nullptr,
            "forced FLA CP supports only H4/H6/H8/H12 static kernels");
  }
  const int64_t tokens = (*a.raw)[1] - (*a.raw)[0];
  if (tokens == 0) return {};
  if (tokens > 102400) {
    if (automatic) return {};
    throw std::invalid_argument("forced FLA CP supports effective lengths up to 102400");
  }
  const auto geometry = make_cp_geometry(tokens, split);
  return {split, geometry.count, cp_split_row(split)};
}

std::shared_mutex init_mutex;
std::set<std::pair<int, int>> initialized;
std::set<std::pair<int, int>> cp_initialized;
std::set<int> devices;
void initialize(int device, const Launchers& table, bool capture, bool require_cp) {
  const auto key = std::make_pair(device, table.heads);
  {
    std::shared_lock lock(init_mutex);
    if (initialized.count(key) && (!require_cp || cp_initialized.count(key))) return;
  }
  TORCH_CHECK(!capture, "the tc01 FLA monolithic extension and the selected CP path must be warmed up outside Graph capture");
  std::unique_lock lock(init_mutex);
  if (!devices.count(device)) {
    hggcDeviceProp prop{};
    cuda_check(hggcGetDeviceProperties(&prop, device), "device identity query failed");
    TORCH_CHECK(prop.major == 8 && prop.minor == 0 && std::string(prop.name) == "PPU-ZW810E",
                "tc01 FLA only supports PPU-ZW810E capability 8.0");
    devices.insert(device);
  }
  if (!initialized.count(key)) {
    cuda_check(table.configure_recompute(), "recompute configure");
    cuda_check(table.configure_recurrence(), "recurrence configure");
    cuda_check(table.configure_output(), "output configure");
    initialized.insert(key);
  }
  if (require_cp && !cp_initialized.count(key)) {
    TORCH_INTERNAL_ASSERT(table.configure_cp);
    cuda_check(table.configure_cp(), "CP recurrence/output configure");
    cp_initialized.insert(key);
  }
}

at::Tensor typed_view(const at::Tensor& arena, Region region, at::ScalarType dtype, at::IntArrayRef shape) {
  auto x = at::empty({0}, arena.options().dtype(dtype));
  const int64_t width = x.element_size();
  TORCH_INTERNAL_ASSERT(region.offset % width == 0);
  int64_t count = 1;
  for (auto dim : shape) count = checked_mul(count, dim);
  TORCH_INTERNAL_ASSERT(checked_mul(count, width) == region.bytes);
  // set_ holds the arena's Storage instead of building an unowned pointer with from_blob.
  return x.set_(arena.storage(), region.offset / width, shape);
}
template<class T = uint8_t> T* pointer(const at::Tensor& arena, Region region) {
  return reinterpret_cast<T*>(arena.data_ptr<uint8_t>() + region.offset);
}
void zero(const at::Tensor& arena, Region region, hggcStream_t stream) {
  if (region.bytes) cuda_check(hggcMemsetAsync(pointer(arena, region), 0, region.bytes, stream), "workspace zero-fill");
}
struct Metadata {
  at::Tensor arena;
  MetadataLayout layout;
  Metadata(at::Tensor value, MetadataLayout offsets) : arena(std::move(value)), layout(offsets) {}
};
using PlanKey = std::tuple<int, uintptr_t, int64_t, int64_t, std::vector<int64_t>>;
std::shared_mutex plans_mutex;
// No eviction: external read-only metadata referenced by a capture must stay held until the Graph is destroyed.
std::map<PlanKey, std::shared_ptr<const Metadata>> plans;
std::shared_ptr<const Metadata> trusted_plan(const Arguments& a, const Shape& s,
                                            hggcStream_t stream, bool capture) {
  PlanKey key{a.q.get_device(), reinterpret_cast<uintptr_t>(stream), s.h, s.t, *a.raw};
  { std::shared_lock lock(plans_mutex); auto it = plans.find(key); if (it != plans.end()) return it->second; }
  TORCH_CHECK(!capture, "a trusted raw_cu_seqlens plan must be warmed up in advance on the same stream");
  std::unique_lock lock(plans_mutex);
  auto found = plans.find(key);
  if (found != plans.end()) return found->second;
  MetadataLayout layout(s.n, s.chunks);
  auto host = at::empty({layout.bytes}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  auto cu = pointer<int32_t>(host, layout.cu);
  auto offsets = pointer<int64_t>(host, layout.offsets);
  auto indices = pointer<int32_t>(host, layout.indices);
  auto slots = pointer<int32_t>(host, layout.slots);
  auto cp_bounds = pointer<int32_t>(host, layout.cp_bounds);
  auto cp_offsets = pointer<int64_t>(host, layout.cp_offsets);
  for (int64_t i = 0; i <= s.n; ++i) { cu[i] = (*a.raw)[i]; offsets[i] = s.offsets[i]; }
  for (int64_t i = 0; i < s.n; ++i) {
    slots[i] = i;
    for (int64_t j = offsets[i]; j < offsets[i + 1]; ++j) {
      indices[2 * j] = i; indices[2 * j + 1] = j - offsets[i];
    }
  }
  for (int row = 0; row < static_cast<int>(kCpSplits.size()); ++row) {
    CpGeometry geometry;
    if (s.n == 1 && (*a.raw)[1] > (*a.raw)[0])
      geometry = make_cp_geometry((*a.raw)[1] - (*a.raw)[0], kCpSplits[row]);
    std::copy(geometry.bounds.begin(), geometry.bounds.end(),
              cp_bounds + row * kCpDescriptorWidth);
    std::copy(geometry.offsets.begin(), geometry.offsets.end(),
              cp_offsets + row * kCpDescriptorWidth);
  }
  auto arena = at::empty({layout.bytes}, a.q.options().dtype(at::kByte));
  // The cold path uploads synchronously, so the host may free it on return; capture/hot paths never come here.
  arena.copy_(host, false);
  auto value = std::make_shared<const Metadata>(std::move(arena), layout);
  plans.emplace(std::move(key), value);
  return value;
}
void record_inputs(const Arguments& a, const c10::cuda::CUDAStream& stream) {
  std::array<const void*, 13> seen{};
  size_t count = 0;
  for (const auto* x : {&a.q, &a.k, &a.v, &a.g, &a.beta, &a.a_log, &a.dt_bias, &a.out,
                        &a.initial, &a.final, &a.cu, &a.indices}) {
    if (!x->defined() || !x->numel()) continue;
    auto address = x->storage().data_ptr().get();
    if (std::find(seen.begin(), seen.begin() + count, address) != seen.begin() + count) continue;
    seen[count++] = address;
    c10::cuda::CUDACachingAllocator::recordStream(x->storage().data_ptr(), stream);
  }
}
}  // namespace

void forward(Arguments a) {
  if (pla::bypass::enabled()) {
    if (pla::instrument::show_params()) {
      pla::instrument::dump_metadata("flashkda_tc01_fla", {
          {"q", &a.q}, {"k", &a.k}, {"v", &a.v}, {"g", &a.g},
          {"beta", &a.beta}, {"A_log", &a.a_log}, {"dt_bias", &a.dt_bias},
          {"out", &a.out}, {"initial_state", &a.initial}, {"final_state", &a.final},
          {"state_pool", &a.pool}, {"state_indices", &a.indices}, {"cu_seqlens", &a.cu}});
      const c10::cuda::OptionalCUDAGuard guard(a.q.is_cuda()
          ? c10::optional<c10::Device>(a.q.device()) : c10::nullopt);
      pla::instrument::dump_case(a.cu.defined() ? a.cu.numel() - 1 : 1,
          a.q.size(0), a.q.size(2), a.v.size(2), a.q.size(1), a.v.size(1),
          get_dtype_string(a.q), true, a.cu.defined() ? &a.cu : nullptr,
          static_cast<float>(a.lower_bound), true);
    }
    return;
  }
  const Shape s = validate(a);
  const auto& table = launchers_for(s.h);
  const CpSelection cp = select_cp(a, s, table);
  // Explicit state also selects its single-segment implementation. AUTO uses
  // the qualified shape buckets; explicit selectors remain authoritative.
  const CpAlgorithm cp_algorithm = cp.active()
      ? (a.raw ? select_cp_algorithm(cp.count, s.h, a.raw->back() - a.raw->front())
               : CpAlgorithm::State)
      : CpAlgorithm::Legacy;
  const WorkspaceLayout layout(s.t, s.h, s.n, s.chunks, !a.raw, cp.count);
  c10::cuda::CUDAGuard guard(a.q.device());
  // FLA dispatch also keeps the branch's value diagnostics; capture only checks metadata and never reads device values.
  const pla::sanity::Context sanity("flashkda_tc01_fla", a.q);
  if (sanity) {
    namespace sp = pla::sanity::prefill;
    if (a.cu.defined()) {
      const auto values = sp::offsets(sanity, a.cu, s.t, "cu_seqlens", true);
      if (a.raw && sanity.values_enabled())
        sanity.check(values == *a.raw, "raw_cu_seqlens must equal device cu_seqlens");
    }
    if (a.indices.defined()) {
      for (const auto* state : {&a.initial, &a.final})
        if (state->defined())
          sp::slots(sanity, a.indices, s.n, state->size(0), "state_indices", a.final.defined());
    }
    if (a.initial.defined() && a.final.defined())
      sanity.check(!sp::overlaps(a.initial, a.final) || sp::same_view(a.initial, a.final),
                   "initial/final state may alias only as the exact same tensor view");
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(a.q.get_device());
  hggcStreamCaptureStatus status{};
  cuda_check(hggcStreamIsCapturing(reinterpret_cast<hggcStream_t>(stream.stream()), &status), "capture status query failed");
  const bool capture = status != hggcStreamCaptureStatusNone;
  initialize(a.q.get_device(), table, capture, cp.active());
  auto plan = a.raw ? trusted_plan(a, s, reinterpret_cast<hggcStream_t>(stream.stream()), capture) : nullptr;
  auto arena = at::empty({layout.bytes}, a.q.options().dtype(at::kByte));
  record_inputs(a, stream);
  uint8_t* meta_base = plan ? plan->arena.data_ptr<uint8_t>() : pointer(arena, layout.metadata);
  const auto& ml = plan ? plan->layout : layout.meta;
  auto cu = reinterpret_cast<int32_t*>(meta_base + ml.cu.offset);
  auto offsets = reinterpret_cast<int64_t*>(meta_base + ml.offsets.offset);
  auto chunks = reinterpret_cast<int32_t*>(meta_base + ml.indices.offset);
  auto slots = reinterpret_cast<int32_t*>(meta_base + ml.slots.offset);
  const int32_t* cp_bounds = nullptr;
  const int64_t* cp_offsets = nullptr;
  if (cp.active()) {
    if (plan) {
      TORCH_INTERNAL_ASSERT(cp.descriptor_row >= 0);
      cp_bounds = reinterpret_cast<int32_t*>(meta_base + ml.cp_bounds.offset)
          + cp.descriptor_row * kCpDescriptorWidth;
      cp_offsets = reinterpret_cast<int64_t*>(meta_base + ml.cp_offsets.offset)
          + cp.descriptor_row * kCpDescriptorWidth;
    } else if (cp.count == 1) {
      // N1 CP1 consumes the current device metadata, including empty/prefix
      // lengths on Graph replay; physical T never substitutes for effective T.
      TORCH_INTERNAL_ASSERT(s.n == 1 && cp.descriptor_row == -1);
      cp_bounds = cu;
      cp_offsets = offsets;
    } else {
      // The descriptor row is rewritten by a device kernel after metadata.
      TORCH_INTERNAL_ASSERT(s.n == 1 && cp.descriptor_row >= 0);
      cp_bounds = reinterpret_cast<int32_t*>(meta_base + ml.cp_bounds.offset)
          + cp.descriptor_row * kCpDescriptorWidth;
      cp_offsets = reinterpret_cast<int64_t*>(meta_base + ml.cp_offsets.offset)
          + cp.descriptor_row * kCpDescriptorWidth;
    }
  }
  const bool fused_metadata = !plan && s.n == 1 && (s.h == 4 || s.h == 8) &&
      (s.t == 16384 || s.t == 32768);
  if (fused_metadata) {
    cuda_check(launch_fla_metadata_n1_tail(
        a.cu.data_ptr(), a.cu.scalar_type() == at::kLong, cu, offsets, chunks,
        slots, s.t, s.chunks, const_cast<int32_t*>(cp_bounds),
        const_cast<int64_t*>(cp_offsets), cp.count, kCpDescriptorWidth,
        12288, 65536, a.out.data_ptr(), a.out.nbytes() / s.t,
        reinterpret_cast<hggcStream_t>(stream.stream())), "N1 metadata + tail");
  } else {
    if (!plan) cuda_check(launch_fla_metadata(a.cu.data_ptr(), a.cu.scalar_type() == at::kLong,
                           cu, offsets, chunks, slots, s.n, s.t, s.chunks, reinterpret_cast<hggcStream_t>(stream.stream())), "metadata");
    if (!plan && cp.count > 1)
      cuda_check(launch_fla_cp_device_geometry(
          cu, const_cast<int32_t*>(cp_bounds), const_cast<int64_t*>(cp_offsets),
          cp.count, kCpDescriptorWidth, 12288, 65536,
          reinterpret_cast<hggcStream_t>(stream.stream())), "CP device geometry");
    // Device offsets clear only rows past the current cu[n]; stages cover the rest.
    if (!plan && !a.raw && a.out.numel() && s.t > 0 && a.out.nbytes() % s.t == 0)
      cuda_check(launch_fla_zero_output_tail(cu, s.n, a.out.data_ptr(), a.out.nbytes() / s.t, s.t,
                                             reinterpret_cast<hggcStream_t>(stream.stream())), "out tail zero-fill");
    else if ((!a.raw || a.raw->back() < s.t) && a.out.numel())
      cuda_check(hggcMemsetAsync(a.out.data_ptr(), 0, a.out.nbytes(), reinterpret_cast<hggcStream_t>(stream.stream())), "out zero-fill");
  }
  const int bucket = a.raw && s.n == 1 ? (s.chunks <= 32 ? 0 : s.chunks <= 128 ? 1 : 2) : -1;
  // State read/write is fused into the recurrence kernel: initial/final are passed through, and the BF16 rounding at the read/write boundary
  // plus the zero-init semantics without initial are kept by the kernel (bit-identical to the existing host chain).
  const void* initial = a.initial.defined() ? a.initial.data_ptr() : nullptr;
  void* final = a.final.defined() ? a.final.data_ptr() : nullptr;
  const int state_fp32 = a.initial.defined() ? a.initial.scalar_type() == at::kFloat
                                             : a.final.defined() && a.final.scalar_type() == at::kFloat;
  const auto* indices = a.indices.defined() ? a.indices.data_ptr<int32_t>() : nullptr;
  if (s.t && s.chunks) {
    const bool prepared_n1 = s.n == 1 && (s.h == 4 || s.h == 8) &&
        (s.t == 4096 || s.t == 8192 || s.t == 16384 || s.t == 32768);
    // Prepared CP consumers need no workspace memset: the token stage writes
    // Aqk zeros before its diagonal tiles, the solve reads akkd's solved
    // diagonal only for live rows, and recompute zero-fills the upper inverse.
    if (!(prepared_n1 && cp.active()))
      zero(arena, layout.zeros, reinterpret_cast<hggcStream_t>(stream.stream()));
    // beta no longer does a host copy+sigmoid: the consumer kernels compute it from raw bf16 with the same inline implementation.
    const auto* beta_raw = static_cast<const uint16_t*>(a.beta.data_ptr());
    const int64_t beta_stride = a.beta.stride(0);
    auto qn = pointer(arena, layout.qn), kn = pointer(arena, layout.kn), qg = pointer(arena, layout.qg);
    auto gate = pointer<float>(arena, layout.gate);
    auto aqk = pointer(arena, layout.aqk), inverse = pointer(arena, layout.inverse);
    auto akkd = pointer<float>(arena, layout.akkd);
    auto w = pointer(arena, layout.w), u = pointer(arena, layout.u), kg = pointer(arena, layout.kg);
    if (!prepared_n1)
      cuda_check(launch_fla_l2norm2_bf16_d128_bt16(a.q.data_ptr(), qn, a.k.data_ptr(), kn,
                                                   s.t * s.h, reinterpret_cast<hggcStream_t>(stream.stream())), "L2 q+k");
    cuda_check(table.gate(a.g.data_ptr(), a.a_log.data_ptr<float>(), a.dt_bias.data_ptr<float>(), gate,
                         s.t, cu, chunks, s.n, s.chunks, s.h, a.lower_bound, reinterpret_cast<hggcStream_t>(stream.stream())), "gate");
    if (prepared_n1) {
      TORCH_INTERNAL_ASSERT(layout.w.bytes == s.t * s.h * 128 * 2 &&
                            layout.u.bytes == layout.w.bytes &&
                            layout.kg.bytes == layout.w.bytes && s.chunks <= s.t / 64);
      auto token_l2 = s.h == 4 ? launch_fla_token_l2__fla_h4 : launch_fla_token_l2__fla_h8;
      cuda_check(token_l2(a.q.data_ptr(), a.k.data_ptr(), qn, kn, gate, beta_raw, beta_stride,
                          aqk, akkd, qg, w, u, kg, a.scale, cu, s.t, s.h,
                          reinterpret_cast<hggcStream_t>(stream.stream())), "L2 + token + prepare");
    } else {
      cuda_check(table.token(qn, kn, gate, beta_raw, beta_stride, aqk, akkd, qg, a.scale, cu,
                            s.n, s.t, s.h, reinterpret_cast<hggcStream_t>(stream.stream())), "token");
    }
    if (prepared_n1) {
      auto prepared = s.h == 4 ? launch_fla_inter_solve_prepared__fla_h4
                              : launch_fla_inter_solve_prepared__fla_h8;
      cuda_check(prepared(qn, kn, gate, beta_raw, beta_stride, akkd, aqk, inverse,
                          w, u, kg, a.scale, cu, chunks, s.chunks, s.t, s.n, s.h,
                          reinterpret_cast<hggcStream_t>(stream.stream())), "inter prepared");
    } else {
      cuda_check(table.inter(qn, kn, gate, beta_raw, beta_stride, akkd, aqk, inverse, a.scale, cu, chunks,
                            s.chunks, s.t, s.n, s.h, reinterpret_cast<hggcStream_t>(stream.stream())), "inter");
    }
    cuda_check(table.recompute(kn, a.v.data_ptr(), beta_raw, beta_stride, inverse, gate, w, u, kg, cu, chunks,
                              s.chunks, s.t, s.n, s.h, reinterpret_cast<hggcStream_t>(stream.stream())), "recompute");
    if (cp.active()) {
      auto bias = pointer<float>(arena, layout.cp_bias);
      auto transition_and_carries = pointer<float>(arena, layout.cp_state_passing);
      auto reserved = pointer<int>(arena, layout.cp_executed);
      if (cp_algorithm == CpAlgorithm::State) {
        cuda_check(table.cp_recurrence_output(
            qg, aqk, a.out.data_ptr(), kg, u, w, gate, initial, final,
            state_fp32, indices, bias, transition_and_carries, reserved,
            cp_bounds, cp_offsets, cp.count, -2,
            reinterpret_cast<hggcStream_t>(stream.stream())),
            "CP deterministic state passing");
      } else {
        cuda_check(table.cp_recurrence_output(
            qg, aqk, a.out.data_ptr(), kg, u, w, gate, initial, final,
            state_fp32, indices, bias, transition_and_carries, reserved,
            cp_bounds, cp_offsets, cp.count, -1,
            reinterpret_cast<hggcStream_t>(stream.stream())),
            "CP legacy pre-scan");
        for (int segment = 1; segment < cp.count; ++segment)
          cuda_check(table.cp_recurrence_output(
              qg, aqk, a.out.data_ptr(), kg, u, w, gate, initial, final,
              state_fp32, indices, bias, transition_and_carries, reserved,
              cp_bounds, cp_offsets, cp.count, segment,
              reinterpret_cast<hggcStream_t>(stream.stream())),
              "CP legacy repair");
      }
    } else {
      auto v_new = pointer(arena, layout.v_new);
      auto history = pointer(arena, layout.history);
      cuda_check(table.recurrence(kg, u, w, v_new, gate, nullptr, history, initial, final, state_fp32,
                                 indices, cu, offsets, s.t, bucket, s.n, s.h, reinterpret_cast<hggcStream_t>(stream.stream())), "recurrence");
      cuda_check(table.output(qn, v_new, gate, qg, history, a.out.data_ptr(), aqk, cu, chunks, a.scale,
                             s.chunks, s.t, s.n, s.h, reinterpret_cast<hggcStream_t>(stream.stream())), "output");
    }
  } else if (a.final.defined()) {
    // Cold path where no kernel runs (empty batch / all-empty sequences): keep the existing publish semantics (boundary BF16 rounding).
    at::Tensor source;
    if (a.initial.defined()) {
      source = a.indices.defined() ? a.initial.index_select(0, a.indices.to(at::kLong)) : a.initial;
    } else {
      source = at::zeros({s.n, s.h, 128, 128}, a.final.options());
    }
    auto publish = source.to(at::kBFloat16).to(a.final.scalar_type());
    if (a.indices.defined()) a.final.index_copy_(0, a.indices.to(at::kLong), publish);
    else a.final.copy_(publish);
  }
}

std::vector<at::Tensor> metadata_only(const at::Tensor& input, int64_t tokens, int64_t heads) {
  if (pla::bypass::enabled()) {
    if (pla::instrument::show_params())
      pla::instrument::dump_metadata("flashkda_tc01_fla_metadata", {{"input", &input}});
    const auto n = input.numel() - 1, capacity = chunk_capacity(tokens, n);
    const auto options = input.options();
    return {pla::bypass::output({n + 1}, options.dtype(at::kInt)),
            pla::bypass::output({n + 1}, options.dtype(at::kLong)),
            pla::bypass::output({capacity, 2}, options.dtype(at::kInt)),
            pla::bypass::output({n}, options.dtype(at::kInt))};
  }
  require(input.defined() && input.is_cuda() && input.dim() == 1 && input.numel() >= 2 && input.is_contiguous() &&
          (input.scalar_type() == at::kInt || input.scalar_type() == at::kLong), "illegal metadata input");
  const auto n = input.numel() - 1, capacity = chunk_capacity(tokens, n);
  dimensions(tokens, heads, n, capacity);
  c10::cuda::CUDAGuard guard(input.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  hggcStreamCaptureStatus status{};
  cuda_check(hggcStreamIsCapturing(reinterpret_cast<hggcStream_t>(stream.stream()), &status), "capture status query failed");
  initialize(input.get_device(), launchers_for(heads),
             status != hggcStreamCaptureStatusNone, false);
  MetadataLayout layout(n, capacity);
  auto arena = at::empty({layout.bytes}, input.options().dtype(at::kByte));
  c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
  cuda_check(launch_fla_metadata(input.data_ptr(), input.scalar_type() == at::kLong,
      pointer<int32_t>(arena, layout.cu), pointer<int64_t>(arena, layout.offsets),
      pointer<int32_t>(arena, layout.indices), pointer<int32_t>(arena, layout.slots), n, tokens, capacity, reinterpret_cast<hggcStream_t>(stream.stream())), "metadata");
  return {typed_view(arena, layout.cu, at::kInt, {n + 1}), typed_view(arena, layout.offsets, at::kLong, {n + 1}),
          typed_view(arena, layout.indices, at::kInt, {capacity, 2}), typed_view(arena, layout.slots, at::kInt, {n})};
}
}  // namespace fla_runtime
