// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
#include "commit_plan.h"
#include <climits>

namespace pla::kda_mtp {
namespace {
__device__ int64_t value(IndexArg a, int64_t i) {
  return a.wide ? static_cast<const int64_t*>(a.data)[i * a.stride] :
                  static_cast<const int*>(a.data)[i * a.stride];
}
__device__ bool valid_slot(int64_t slot, const PlanInputs& p) {
  return slot > p.null_id && slot >= 0 && slot < p.slots;
}
__global__ void build_plan(PlanInputs p, int* candidate) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= p.capacity) return;
  int count = 0, final = 0, boundary = 0, bc = 0;
  if (b < p.batch) {
    const int64_t row = p.mapping.data ? value(p.mapping, b) : b;
    const int64_t source = value(p.source, b);
    const int64_t begin = value(p.offsets, b), end = value(p.offsets, b + 1);
    if (value(p.offsets, 0) == 0 && row >= 0 && row < p.sampler_rows && valid_slot(source, p) &&
        begin >= 0 && end >= begin && end - begin <= p.Q) {
      const int64_t sampled = value(p.sampled, row);
      count = sampled <= 0 ? 0 : sampled > end - begin ? end - begin : sampled;
      final = source;
      if (count && p.block_size) {
        if (row >= p.block_rows) {
          count = 0;
        } else {
          const int64_t pos = value(p.computed, row);
          if (pos < 0 || pos > INT64_MAX - p.Q) {
            count = 0;
          } else {
            int64_t col = (pos + count - 1) / p.block_size;
            if (col >= p.block_cols) col = p.block_cols - 1;
            const int64_t f = value(p.blocks,
                row * p.block_row_stride + col * p.block_col_stride);
            if (!valid_slot(f, p)) count = 0;
            else final = f;
            // Derive distance without multiplying a potentially huge block index.
            const int64_t distance = p.block_size - pos % p.block_size;
            if (count && count >= distance) {
              const int64_t boundary_col = pos / p.block_size;
              if (boundary_col >= p.block_cols) {
                count = 0;
              } else {
                const int64_t dest = value(p.blocks,
                    row * p.block_row_stride + boundary_col * p.block_col_stride);
                if (!valid_slot(dest, p) || (dest == final && distance != count))
                  count = 0;
                else { boundary = dest; bc = distance; }
              }
            }
          }
        }
      }
    }
  }
  candidate[b] = count;
  candidate[p.capacity + b] = count ? final : 0;
  candidate[2 * p.capacity + b] = count ? boundary : 0;
  candidate[3 * p.capacity + b] = count ? bc : 0;
}

// Read immutable candidates, then publish a separate plan. Conflicting rows are
// masked symmetrically; no ordering-dependent read/write race in the plan itself.
template <bool PARALLEL, bool CTA_ROW = false>
__global__ void protect_plan(PlanInputs p, const int* candidate, int* plan) {
  const int lane = threadIdx.x & 31;
  const int b = CTA_ROW ? blockIdx.x :
      (PARALLEL ? blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32 :
                  blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= p.capacity) return;
  bool conflict = false;
  if (candidate[b]) {
    const int64_t source = value(p.source, b);
    const int f = candidate[p.capacity + b];
    const int d = candidate[2 * p.capacity + b];
    const bool has_d = candidate[3 * p.capacity + b] > 0;
    for (int other = CTA_ROW ? threadIdx.x : (PARALLEL ? lane : 0);
         other < p.batch; other += CTA_ROW ? blockDim.x : (PARALLEL ? 32 : 1)) {
      if (other == b) continue;
      const int64_t os = value(p.source, other);
      const int64_t begin = value(p.offsets, other), end = value(p.offsets, other + 1);
      // A real query with c=0 still owns its source checkpoint.
      if (valid_slot(os, p) && begin >= 0 && end > begin && end - begin <= p.Q)
        conflict |= f == os || (has_d && d == os);
      if (!candidate[other]) continue;
      const int of = candidate[p.capacity + other];
      const int od = candidate[2 * p.capacity + other];
      const bool has_od = candidate[3 * p.capacity + other] > 0;
      conflict |= f == os || f == of || (has_od && f == od) ||
                  (has_d && (d == os || d == of || (has_od && d == od))) ||
                  source == of || (has_od && source == od);
    }
  }
  if constexpr (PARALLEL) conflict = __any_sync(0xffffffffu, conflict);
  if constexpr (CTA_ROW) {
    __shared__ int warp_conflicts[4];
    if (lane == 0) warp_conflicts[threadIdx.x / 32] = conflict;
    __syncthreads();
    if (threadIdx.x != 0) return;
    conflict = warp_conflicts[0] || warp_conflicts[1] ||
               warp_conflicts[2] || warp_conflicts[3];
  }
  if (!PARALLEL || lane == 0)
    for (int col = 0; col < 4; ++col)
      plan[col * p.capacity + b] = conflict ? 0 : candidate[col * p.capacity + b];
}
}  // namespace

void launch_commit_plan(PlanInputs inputs, int* candidate, int* plan,
                        hggcStream_t stream) {
  if (!inputs.capacity) return;
  const int blocks = (inputs.capacity + 127) / 128;
  build_plan<<<blocks, 128, 0, stream>>>(inputs, candidate);
  if (inputs.batch >= 32 && current_device_is_ppu15()) {
    protect_plan<true, true><<<inputs.capacity, 128, 0, stream>>>(
        inputs, candidate, plan);
  } else {
    protect_plan<false><<<blocks, 128, 0, stream>>>(inputs, candidate, plan);
  }
}
}  // namespace pla::kda_mtp
