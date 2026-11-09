// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
#include <torch/extension.h>
#include <hgtx3/hgtx3.hpp>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>
#include "../../bypass_exec.h"
#include "../../op_forge_case.h"
#include "../../ppu_torch_stream.h"
#include "../../sanity_check.h"
#include "commit_plan.h"

namespace {
using torch::Tensor;
using pla::kda_mtp::IndexArg;
using pla::kda_mtp::PlanInputs;
using Optional = std::optional<Tensor>;
constexpr int D = 128;

bool perf_instrument() {
  static const bool enabled = [] {
    const char* value = std::getenv("PPU_LIB_PERF_INSTRUMENT");
    return value && std::atoi(value) != 0;
  }();
  return enabled;
}

void tensor(const Tensor& t, const Tensor& ref) {
  TORCH_CHECK(t.defined() && t.layout() == torch::kStrided && t.is_cuda() &&
              t.device() == ref.device(), "replay tensors must share a CUDA/PPU device");
  for (auto size : t.sizes())
    TORCH_CHECK(size <= INT32_MAX, "tensor dimension exceeds the native int32 ABI");
  for (auto stride : t.strides()) TORCH_CHECK(stride > 0, "strides must be positive");
}
void dense(const Tensor& t, torch::ScalarType dtype, torch::IntArrayRef shape) {
  TORCH_CHECK(t.scalar_type() == dtype && t.sizes() == shape && t.is_contiguous(),
              "invalid replay tensor shape/dtype/contiguity");
}
void indices(const Tensor& t, const Tensor& ref) {
  tensor(t, ref);
  TORCH_CHECK(t.dim() == 1 && (t.scalar_type() == torch::kInt32 ||
              t.scalar_type() == torch::kInt64), "indices must be int32/int64 vectors");
}
IndexArg index(const Tensor& t) {
  return {t.data_ptr(), t.scalar_type() == torch::kInt64, t.stride(0)};
}
IndexArg index(const Optional& t) { return t ? index(*t) : IndexArg{nullptr, false, 1}; }
void null_id(int value) {
  TORCH_CHECK(value == -1 || value == 0, "null_block_id must be -1 or 0");
}

Tensor cache(const Tensor& conv, const Tensor& state, const std::string& layout,
             int H, int slots) {
  TORCH_CHECK(state.is_cuda() && H > 0 && H <= INT32_MAX / (D * D) && slots > 0, "paged state requires CUDA/PPU, H > 0, slots > 0");
  tensor(state, state);
  tensor(conv, state);
  TORCH_CHECK(state.scalar_type() == torch::kFloat32 &&
              state.sizes() == torch::IntArrayRef({slots, H, D, D}) &&
              state.stride(3) == 1 && state.stride(2) == D && state.stride(1) == D * D &&
              state.stride(0) >= H * D * D && state.stride(0) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0,
              "state must be float4-aligned fp32 [slots,H,128,128] with contiguous pages");
  TORCH_CHECK(layout == "DS" || layout == "SD", "conv_state_layout must be DS or SD");
  TORCH_CHECK(conv.dim() == 3 && conv.size(0) == slots &&
              conv.scalar_type() == torch::kBFloat16, "conv must be bf16 with three dimensions");
  auto ds = layout == "DS" ? conv : conv.transpose(1, 2);
  TORCH_CHECK(ds.size(1) == 3 * H * D && ds.size(2) >= 3,
              "conv must contain 3*H*128 channels and at least three history items");
  TORCH_CHECK((ds.stride(2) == 1 && ds.stride(1) >= ds.size(2)) ||
              (ds.stride(1) == 1 && ds.stride(2) >= ds.size(1)),
              "conv channel/history dimensions must not overlap");
  TORCH_CHECK(ds.stride(0) >= (ds.size(1) - 1) * ds.stride(1) +
              (ds.size(2) - 1) * ds.stride(2) + 1, "conv pages must not overlap");
  return ds;
}
void records(const Tensor& rv, const Tensor& rk, const Tensor& g, const Tensor& b,
             const Tensor& windows, const Tensor& ref, int H, int Q, int R) {
  TORCH_CHECK(Q >= 1 && Q <= 32 && R >= 0, "records require Q in [1,32] and R >= 0");
  for (auto& t : {rv, rk, g, b, windows}) tensor(t, ref);
  dense(rv, torch::kBFloat16, {R, H, Q, D});
  dense(rk, torch::kBFloat16, {R, H, Q, D});
  dense(g, torch::kFloat32, {R, H, Q, D});
  dense(b, torch::kFloat32, {R, H, Q});
  dense(windows, torch::kBFloat16, {R, Q, 3 * H * D, 3});
  for (auto& t : {rv, rk, g, b})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "replay records must be 16-byte aligned");
}

// Conservative byte-span overlap check for writable outputs/records. Read-only
// projection views and the conv/SSM subviews of a common page may share storage.
std::pair<uintptr_t, uintptr_t> span(const Tensor& t) {
  const auto begin = reinterpret_cast<uintptr_t>(t.data_ptr());
  if (!t.numel()) return {begin, begin};
  int64_t extent = 1;
  for (int d = 0; d < t.dim(); ++d) extent += (t.size(d) - 1) * t.stride(d);
  return {begin, begin + extent * t.element_size()};
}
void disjoint(const std::vector<Tensor>& writes, const std::vector<Tensor>& reads) {
  auto overlap = [](const Tensor& a, const Tensor& b) {
    const auto x = span(a), y = span(b);
    return x.first < y.second && y.first < x.second;
  };
  for (size_t i = 0; i < writes.size(); ++i) {
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(!overlap(writes[i], writes[j]), "replay writes must not alias");
    for (const auto& read : reads)
      TORCH_CHECK(!overlap(writes[i], read), "replay writes must not alias inputs/cache");
  }
}

std::vector<int64_t> layer_descriptor(Tensor conv, Tensor state, Tensor rv, Tensor rk,
                                    Tensor gate, Tensor beta, Tensor windows,
                                    std::string layout, int H, int slots, int Q, int R) {
  auto ds = cache(conv, state, layout, H, slots);
  records(rv, rk, gate, beta, windows, state, H, Q, R);
  disjoint({rv, rk, gate, beta, windows}, {conv, state});
  auto ptr = [](const Tensor& t) { return reinterpret_cast<int64_t>(t.data_ptr()); };
  return {ptr(state), state.stride(0), ptr(conv), ds.stride(0), ds.stride(1),
          ds.stride(2), ptr(rv), ptr(rk), ptr(gate), ptr(beta), ptr(windows)};
}

Tensor verify(Tensor x, Tensor g, Tensor beta, Tensor weight, Tensor conv, Tensor state,
            Tensor a, Tensor dt, Optional output, Tensor rv, Tensor rk, Tensor gate,
            Tensor rb, Tensor windows, Tensor cu, Tensor source, std::string layout,
            double scale, double lower_bound, int null) {
  Tensor out = output ? *output : pla::bypass::output(
      {x.size(0), state.size(1) * D}, x.options().dtype(torch::kBFloat16));
  if (pla::instrument::show_params()) {
      pla::instrument::dump_case(source.numel(), x.dim() ? x.size(0) : 0, D, D,
          state.dim() > 1 ? state.size(1) : 0, state.dim() > 1 ? state.size(1) : 0,
          get_dtype_string(x), true, &cu, lower_bound);
      pla::instrument::dump_metadata("kda_replay_verify", {{"mixed_qkv", &x}, {"raw_g", &g},
          {"raw_beta", &beta}, {"conv_weight", &weight}, {"conv_state", &conv},
          {"checkpoint_state", &state}, {"A_log", &a}, {"dt_bias", &dt}, {"out", &out},
          {"rawv", &rv}, {"rawk", &rk}, {"g", &gate}, {"beta", &rb},
          {"conv_snapshots", &windows}, {"query_start_loc", &cu}, {"state_indices", &source}},
          "instrument=HGTX conv_layout=" + layout);
  }
  if (pla::bypass::enabled()) return out;
  std::optional<hgtx3::scoped_range> range;
  if (perf_instrument()) range.emplace("kda_replay_verify instrument=HGTX");
  null_id(null);
  TORCH_CHECK(state.dim() == 4 && rv.dim() == 4, "invalid state/records rank");
  const int H = state.size(1), slots = state.size(0), Q = rv.size(2), R = rv.size(0);
  auto cs = cache(conv, state, layout, H, slots);
  const c10::cuda::CUDAGuard guard(state.device());
  records(rv, rk, gate, rb, windows, state, H, Q, R);
  for (auto& t : {x, g, beta, weight, a, dt, out, cu, source}) tensor(t, state);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 3 * H * D && x.stride(1) == 1 &&
              x.stride(0) >= 3 * H * D && x.scalar_type() == torch::kBFloat16,
              "mixed_qkv must be bf16 [T,3*H*128] with contiguous channels");
  const int T = x.size(0);
  auto projection = [&](const Tensor& t, int channels) {
    TORCH_CHECK(t.sizes() == torch::IntArrayRef({T, channels}) && t.stride(1) == 1 &&
                t.stride(0) >= channels && t.scalar_type() == torch::kBFloat16,
                "gate/beta must be bf16 projection views");
  };
  projection(g, H * D); projection(beta, H);
  dense(weight, torch::kFloat32, {3 * H * D, 4});
  TORCH_CHECK(a.numel() == H && dt.numel() == H * D && a.is_contiguous() &&
              dt.is_contiguous() && a.scalar_type() == torch::kFloat32 &&
              dt.scalar_type() == torch::kFloat32, "gate weights must be contiguous fp32");
  dense(out, torch::kBFloat16, {T, H * D});
  TORCH_CHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)) &&
              std::isfinite(lower_bound) && lower_bound < 0 &&
              std::isfinite(static_cast<float>(lower_bound)) &&
              static_cast<float>(lower_bound) < 0, "scale/lower_bound must be finite fp32; bound < 0");
  indices(cu, state); indices(source, state);
  const int N = source.numel();
  TORCH_CHECK(cu.numel() == N + 1 && N <= R, "query index count exceeds record capacity");
  disjoint({out, rv, rk, gate, rb, windows},
           {x, g, beta, weight, conv, state, a, dt, cu, source});
  const pla::sanity::Context sanity("kda_replay_verify", state);
  if (sanity.values_enabled()) {
    const auto offsets = sanity.integers(cu), indices = sanity.integers(source);
    sanity.check(offsets.front() == 0 && offsets.back() <= T, "query offsets must cover a packed prefix");
    for (int b = 0; b < N; ++b) {
      sanity.check(offsets[b] >= 0 && offsets[b + 1] >= offsets[b] &&
                   offsets[b + 1] - offsets[b] <= Q, "invalid query length at row ", b);
      if (offsets[b] == offsets[b + 1]) continue;
      sanity.check(indices[b] < slots, "source slot out of bounds at row ", b);
      if (indices[b] <= null) continue;
      sanity.finite(x.slice(0, offsets[b], offsets[b + 1]), "mixed_qkv");
      sanity.finite(g.slice(0, offsets[b], offsets[b + 1]), "raw_g");
      sanity.finite(beta.slice(0, offsets[b], offsets[b + 1]), "raw_beta");
      sanity.finite(state.select(0, indices[b]), "checkpoint_state");
      sanity.finite(cs.select(0, indices[b]).slice(1, 0, 3), "committed conv history");
    }
    sanity.finite(weight, "conv_weight");
    sanity.finite(a, "A_log"); sanity.finite(dt, "dt_bias");
  }
  const pla::kda_mtp::ConvStrides cstr{cs.stride(0), cs.stride(1), cs.stride(2)};
  const pla::kda_mtp::WindowStrides wstr{windows.stride(0), windows.stride(1),
                                      windows.stride(2), windows.stride(3)};
  const pla::kda_mtp::VerifyStrides strides{x.stride(0), x.stride(0), x.stride(0),
      g.stride(0), beta.stride(0), cstr, cstr, cstr, wstr, wstr, wstr};
  const auto* xp = static_cast<const char*>(x.data_ptr());
  const auto* cp = static_cast<const char*>(conv.data_ptr());
  auto* wp = static_cast<char*>(windows.data_ptr());
  const int64_t channel_bytes = static_cast<int64_t>(H) * D * 2;
  // Private same-stream scratch: capture retains its allocator-pool storage,
  // and every native CTA writes its flags before the repair kernel consumes
  // them. No host metadata readback or public Python argument is introduced.
  Tensor precision_flags;
  int* precision_flags_ptr = nullptr;
  if constexpr (pla::kda_mtp::kNativeNumericalProtection) {
    if (pla::kda_mtp::current_device_is_ppu15()) {
      precision_flags = torch::empty({N, H, 4}, state.options().dtype(torch::kInt32));
      precision_flags_ptr = precision_flags.data_ptr<int>();
    }
  }
  pla::kda_mtp::launch_verify_native(xp, xp + channel_bytes, xp + 2 * channel_bytes,
      g.data_ptr(), beta.data_ptr(), weight.data_ptr<float>(),
      weight.data_ptr<float>() + H * D * 4, weight.data_ptr<float>() + 2 * H * D * 4,
      cp, cp + channel_bytes * cs.stride(1), cp + 2 * channel_bytes * cs.stride(1),
      a.data_ptr<float>(), dt.data_ptr<float>(), state.data_ptr<float>(), out.data_ptr(),
      wp, wp + channel_bytes * windows.stride(2), wp + 2 * channel_bytes * windows.stride(2),
      rv.data_ptr(), rk.data_ptr(), gate.data_ptr<float>(), rb.data_ptr<float>(),
      index(source), index(cu), H, N, Q, scale, lower_bound, strides,
      {state.stride(0), slots, T, null, precision_flags_ptr}, pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void commit(Tensor table, Tensor candidate, Tensor plan, Tensor sampled, Tensor source,
            Tensor cu, Optional mapping, Optional blocks, Optional computed,
            int64_t block_size, int H, int slots, int Q, int null) {
  if (pla::instrument::show_params()) {
      pla::instrument::dump_case(source.numel(), Q, D, D, H, H, "bf16", true, &cu);
      pla::instrument::dump_metadata("kda_replay_commit", {{"layers", &table},
          {"num_sampled", &sampled}, {"state_indices", &source}, {"query_start_loc", &cu},
          {"request_indices", mapping ? &*mapping : nullptr}, {"block_table", blocks ? &*blocks : nullptr},
          {"num_computed_tokens", computed ? &*computed : nullptr}, {"plan", &plan}}, "instrument=HGTX");
  }
  if (pla::bypass::enabled()) return;
  std::optional<hgtx3::scoped_range> range;
  if (perf_instrument()) range.emplace("kda_replay_commit instrument=HGTX");
  null_id(null);
  TORCH_CHECK(table.is_cuda() && table.dim() == 2 && table.size(0) > 0 &&
              table.size(1) == pla::kda_mtp::kReplayColumns && table.is_contiguous() &&
              table.scalar_type() == torch::kInt64, "invalid layer pointer table");
  const c10::cuda::CUDAGuard guard(table.device());
  TORCH_CHECK(plan.dim() == 2 && plan.size(0) == 4 && Q >= 1 && Q <= 32 &&
              H > 0 && slots > 0 && table.size(0) * H <= INT32_MAX,
              "invalid commit plan dimensions");
  const int R = plan.size(1), N = source.numel();
  for (auto& t : {table, candidate, plan}) tensor(t, table);
  dense(plan, torch::kInt32, {4, R}); dense(candidate, torch::kInt32, {4, R});
  for (auto& t : {sampled, source, cu}) indices(t, table);
  TORCH_CHECK(N <= R && cu.numel() == N + 1, "commit query count exceeds capacity");
  if (mapping) { indices(*mapping, table); TORCH_CHECK(mapping->numel() == N, "mapping must have N rows"); }
  else TORCH_CHECK(sampled.numel() >= N, "sampler count must cover query rows");
  TORCH_CHECK(bool(blocks) == bool(computed) && bool(blocks) == (block_size != 0),
              "align arguments must be supplied together");
  if (blocks) {
    tensor(*blocks, table); indices(*computed, table);
    TORCH_CHECK(block_size >= Q && blocks->dim() == 2 && blocks->size(1) > 0 &&
                (blocks->scalar_type() == torch::kInt32 || blocks->scalar_type() == torch::kInt64) &&
                computed->numel() >= sampled.numel(), "invalid align table/count/block size");
  }
  disjoint({candidate, plan}, {table, sampled, source, cu});
  if (mapping) disjoint({candidate, plan}, {*mapping});
  if (blocks) disjoint({candidate, plan}, {*blocks, *computed});
  PlanInputs inputs{index(sampled), index(source), index(cu), index(mapping), index(computed),
      blocks ? IndexArg{blocks->data_ptr(), blocks->scalar_type() == torch::kInt64, 1} : IndexArg{nullptr, false, 1},
      N, R, static_cast<int>(sampled.numel()), blocks ? static_cast<int>(blocks->size(0)) : 0,
      blocks ? static_cast<int>(blocks->size(1)) : 0, slots, Q, null,
      blocks ? blocks->stride(0) : 0, blocks ? blocks->stride(1) : 0, block_size};
  const pla::sanity::Context sanity("kda_replay_commit", table);
  if (sanity.values_enabled()) {
    const auto ss = sanity.integers(source), offsets = sanity.integers(cu);
    const auto counts = sanity.integers(sampled);
    const auto map = mapping ? sanity.integers(*mapping) : std::vector<int64_t>{};
    const auto positions = computed ? sanity.integers(*computed) : std::vector<int64_t>{};
    std::vector<int64_t> bt;
    if (blocks) bt = sanity.integers(blocks->reshape({-1}));
    std::vector<std::vector<int64_t>> destinations(N);
    std::vector<bool> active_source(N, false);
    sanity.check(offsets.front() == 0, "query offsets must begin at zero");
    for (int b = 0; b < N; ++b) {
      sanity.check(offsets[b] >= 0 && offsets[b + 1] >= offsets[b] &&
                   offsets[b + 1] - offsets[b] <= Q, "invalid query length");
      const auto row = mapping ? map[b] : b;
      sanity.check(row >= 0 && row < sampled.numel(), "request mapping out of bounds");
      if (offsets[b + 1] == offsets[b]) continue;
      sanity.check(ss[b] < slots, "source slot out of bounds");
      active_source[b] = ss[b] > null;
      const int64_t n = std::min(std::max(counts[row], int64_t{0}), offsets[b + 1] - offsets[b]);
      if (ss[b] <= null || !n) continue;
      int64_t f = ss[b], d = 0, bc = 0;
      if (blocks) {
        sanity.check(row < blocks->size(0), "block table request row out of bounds");
        const auto pos = positions[row];
        sanity.check(pos >= 0 && pos <= INT64_MAX - Q, "invalid computed token count");
        f = bt[row * blocks->size(1) + std::min((pos + n - 1) / block_size, blocks->size(1) - 1)];
        const auto distance = block_size - pos % block_size;
        if (n >= distance) {
          sanity.check(pos / block_size < blocks->size(1), "boundary block out of bounds");
          d = bt[row * blocks->size(1) + pos / block_size]; bc = distance;
          sanity.check(d != f || bc == n, "distinct prefix lengths require distinct destinations");
        }
      }
      sanity.check(f > null && f < slots, "final destination out of bounds");
      destinations[b].push_back(f);
      if (bc) { sanity.check(d > null && d < slots, "boundary destination out of bounds"); destinations[b].push_back(d); }
    }
    for (int b = 0; b < N; ++b) for (int other = 0; other < N; ++other) {
      if (b == other) continue;
      for (auto d : destinations[b]) {
        if (active_source[other]) sanity.check(d != ss[other], "destination overlaps another active source");
        for (auto od : destinations[other]) sanity.check(d != od, "active destinations overlap");
      }
    }
  }
  const auto stream = pla::current_stream();
  Tensor precision_flags;
  int* precision_flags_ptr = nullptr;
  if constexpr (pla::kda_mtp::kNativeNumericalProtection) {
    if (pla::kda_mtp::current_device_is_ppu15()) {
      // Initialize the contiguous buffer once instead of issuing one tiny reset
      // store from every fold CTA. The operation is captured with graph replay.
      precision_flags = torch::zeros({N, table.size(0) * H, 4},
                                    table.options().dtype(torch::kInt32));
      precision_flags_ptr = precision_flags.data_ptr<int>();
    }
  }
  pla::kda_mtp::launch_commit_plan(inputs, candidate.data_ptr<int>(), plan.data_ptr<int>(), stream);
  const int* p = plan.data_ptr<int>();
  const IndexArg lens{p, false, 1}, final{p + R, false, 1}, boundary{p + 2 * R, false, 1}, bc{p + 3 * R, false, 1};
  pla::kda_mtp::launch_fold_native(table.data_ptr<int64_t>(), index(source), lens, final, boundary, bc,
                                 table.size(0), N, H, Q, slots, null, block_size != 0,
                                 precision_flags_ptr, stream);
  pla::kda_mtp::launch_conv_native(table.data_ptr<int64_t>(), lens, final, boundary, bc,
                                 table.size(0), N, H, Q, slots, null, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  (void)pla::bypass::enabled();
  (void)pla::instrument::show_params();
  (void)perf_instrument();
  m.def("verify", &verify);
  m.def("layer_descriptor", &layer_descriptor);
  m.def("commit", &commit);
}
