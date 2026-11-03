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

#pragma once
// torch-free ABI between the KDA MTP SGLang host wrappers and device launchers.
//
// The .cu files contain the kernels and <<<>>> launch sites and are compiled by
// hgcc against the PPU-native headers. The .cpp host wrapper may keep using the
// CUDA-compatible torch API; only raw pointers, dimensions, strides and a native
// stream handle cross this boundary.

#include <cstdint>
#include <cstdlib>

#include <hggc_runtime.h>

namespace pla {
namespace kda_mtp {

enum class ConvDType : int {
  kBFloat16 = 0,
  kFloat32,
};

// Strides (in elements) describing the q/k/v conv-state and scratch-window
// layouts the verify kernel must address. Mirrors master's device-side struct
// but lives here so it can cross the torch-free ABI boundary by value.
struct ConvStrides {
  int64_t slot, channel, window;
};

struct WindowStrides {
  int64_t row, step, channel, window;
};

struct VerifyStrides {
  int64_t x_q, x_k, x_v, g, beta;
  ConvStrides cs_q, cs_k, cs_v;
  WindowStrides ic_q, ic_k, ic_v;
};

// A 1-D index tensor exactly as the caller passed it, int32 or int64 elements.
// Kernels widen it in place, so int64 indices (the commit's
// last_correct_step_indices arrive as int64) no longer cost a cast kernel per
// call, which on small batches was as long as the scatter itself.
struct IndexArg {
  const void* data;
  bool wide;
};

void launch_fold(
    float* h0, const void* rawv, const void* rawk,
    const float* gk, const float* beta,
    IndexArg slot_idx, IndexArg accept_lens,
    IndexArg track_idx, IndexArg track_step,
    int64_t h0_s0, int64_t h0_s1,
    int64_t rawv_s0, int64_t rawv_s1,
    int64_t rawk_s0, int64_t rawk_s1,
    int64_t gk_s0, int64_t gk_s1,
    int64_t beta_s0, int64_t beta_s1,
    int layers, int B, int num_k_heads, int HV, int V, int L,
    int null_block_id, bool has_track, hggcStream_t stream);

void launch_conv_scatter(
    ConvDType dtype, int km1,
    const void* src, void* dst,
    IndexArg dst_idx, IndexArg step_idx,
    int layers, int dim, int reqs,
    int64_t src_s0, int64_t src_s1, int64_t src_s2,
    int64_t src_s3, int64_t src_s4,
    int64_t dst_s0, int64_t dst_s1, int64_t dst_s2, int64_t dst_s3,
    int src_reqs, int src_steps, int dst_reqs,
    hggcStream_t stream);

inline bool current_device_is_ppu15(int* device_out = nullptr) {
  int device = -1, major = 0, minor = 0;
  const bool supported = hggcGetDevice(&device) == hggcSuccess &&
      hggcDeviceGetAttribute(&major, hggcDevAttrComputeCapabilityMajor,
                             device) == hggcSuccess &&
      hggcDeviceGetAttribute(&minor, hggcDevAttrComputeCapabilityMinor,
                             device) == hggcSuccess &&
      major == 8 && minor == 9;
  if (supported && device_out != nullptr) *device_out = device;
  return supported;
}

// Size gates for the three PPU1.5 fast paths. Each was first set from a single
// large shape (H*N=8192); the defaults below are instead the measured crossover
// against the fallback over the caselist shape range, swept per shape with the
// cold-cache CUDA-graph contract.
//
// verify: the stepwise kernel splits its value rows over grid.z on small
// (head, request) grids (verify_steps_split below), which removed the old
// small-grid loss (steps=4 was 0.68x at H*N=4 and 0.81x at 32 with a single
// CTA per pair). Across all 2491 shapes (steps=1/3/4/5/8), full_v7 measured
// a minimum verify speedup of 1.164x over the fallback, removing the size gate.
// After issuing fold/conv index loads early, full_v7 commit speedups ranged
// from 1.099x to 1.829x. The weakest single-step fold is near the bandwidth
// limit; small grids are latency-bound. Neither path needs a size gate.
//
// The env overrides read once; a negative value disables the fast path, which
// is how the A/B sweep and any field rollback select the fallback without a
// rebuild.
inline int64_t env_gate(const char* name, int64_t fallback) {
  const char* raw = getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  char* end = nullptr;
  const long long parsed = strtoll(raw, &end, 10);
  return (end != nullptr && *end == '\0') ? parsed : fallback;
}

inline int64_t verify_fast_gate() {
  static const int64_t gate = env_gate("PLA_KDA_VERIFY_GATE", 0);
  return gate;
}

inline int64_t fold_stage_gate() {
  static const int64_t gate = env_gate("PLA_KDA_FOLD_GATE", 0);
  return gate;
}

inline int64_t conv_packed_gate() {
  static const int64_t gate = env_gate("PLA_KDA_CONV_GATE", 0);
  return gate;
}

// State rows per thread in the staged fold (1, 2 or 4; threads = 256 / rows).
// Each k/decay load feeds every row a thread holds. Outputs are bit-identical
// across rows; 2 is 9-12% faster than 1 on multi-step folds (e.g. 8x32x1000:
// 2645 -> 2345us) and level on single-step ones, 4 (168 regs) gains nothing.
inline int fold_stage_rows() {
  static const int64_t rows = env_gate("PLA_KDA_FOLD_ROWS", 2);
  return rows == 1 || rows == 4 ? static_cast<int>(rows) : 2;
}

// Largest draft window the stepwise verify kernel is instantiated for. The
// per-thread reduction terms grow as STEPS^2: STEPS=8 holds 80 of them and
// builds at 96-128 registers with no stack, STEPS=16 would hold 288 and spill.
// Wider windows keep using the generic path.
constexpr int kVerifyMaxSteps = 8;

inline bool use_verify_fast(int H, int N, int steps) {
  const int64_t gate = verify_fast_gate();
  return steps >= 1 && steps <= kVerifyMaxSteps && gate >= 0 &&
         static_cast<int64_t>(H) * N >= gate;
}

// Value-row split of the stepwise verify kernel (grid.z). PLA_KDA_VERIFY_SPLIT
// = 1/2/4/8 forces it; anything else picks it from the (head, request) grid
// and the window. The thresholds are the measured per-steps crossovers on
// ZW-M890P (H*N=4..1024, steps 1/3/4/5/8, with the weight loads issued ahead
// of the index misses): a split re-derives the cheap channel phases per CTA,
// so it pays while the grid is latency-bound and loses once VS=1 fills the
// CUs. VS=8 only pays for the wider windows at the smallest grids (its extra
// CTAs' weight loads queue with everyone's index misses: steps=1 at H*N=12 is
// 6.0us at VS=4 vs 6.2 at VS=8). VS=2 holds until its 2*H*N CTAs overflow one
// resident wave, which comes sooner for the higher-register windows: past
// H*N=192 at steps 3/4 (VS=2 is 25.0 vs 21.5us at 208), 152 at steps 5, ~112
// at steps 8. steps=1 has almost no channel work: VS=1 wins 2-3% from H*N=88
// and, eager (below) from 160, also matches or beats VS=2 to the end.
inline int64_t verify_split_override() {
  static const int64_t split = env_gate("PLA_KDA_VERIFY_SPLIT", 0);
  return split;
}

inline int verify_steps_split(int H, int N, int steps) {
  const int64_t forced = verify_split_override();
  if (forced == 1 || forced == 2 || forced == 4 || forced == 8)
    return static_cast<int>(forced);
  const int64_t ctas = static_cast<int64_t>(H) * N;
  if (ctas < (steps <= 2 ? 0 : steps <= 4 ? 8 : steps == 5 ? 12 : 16)) return 8;
  if (ctas < (steps <= 2 ? 40 : 64)) return 4;
  return ctas < (steps <= 2 ? 88 : steps <= 4 ? 200 : steps == 5 ? 160 : 120)
             ? 2 : 1;
}

// Whether a VS=1 launch keeps two passes of state rows in flight (EAGER). It
// shortens every CTA but, past steps=1, costs registers: on ZW-M890P (39 CUs)
// 72 -> 104-120 at steps 3/4, i.e. 14 -> 8 blocks/CU, so it only pays where it
// does not add a wave. cap_lazy / cap_eager are the resident CTAs per wave of
// each build (CUs x occupancy). Measured: when the lazy grid just spills into
// a second wave that the eager one also needs, eager is 6-10% faster (steps=3
// H*N=600: 40.0 vs 44.2us; steps=5 480: 44.0 vs 47.3); with the lazy grid's
// second wave over a third full it is level, and one extra wave costs 4-27%.
// steps=1 has the same registers either way; eager loses 4-6% below H*N=140
// and wins 1-3% from 160. PLA_KDA_VERIFY_EAGER = 1 / -1 forces it on / off.
inline int64_t verify_eager_override() {
  static const int64_t eager = env_gate("PLA_KDA_VERIFY_EAGER", 0);
  return eager;
}

inline bool verify_steps_eager(int64_t ctas, int64_t cap_lazy,
                               int64_t cap_eager) {
  const int64_t forced = verify_eager_override();
  if (forced != 0) return forced > 0;
  if (cap_lazy <= 0 || cap_eager <= 0) return false;
  if (cap_eager >= cap_lazy) return ctas >= 160;
  return ctas > cap_lazy && ctas <= 2 * cap_eager &&
         ctas - cap_lazy <= cap_lazy / 3;
}

void launch_verify(
    const void* x_q, const void* x_k, const void* x_v,
    const void* g_in, const void* beta_in,
    const float* w_q, const float* w_k, const float* w_v,
    const void* cs_q, const void* cs_k, const void* cs_v,
    const float* a_log, const float* dt_bias, const float* h0,
    void* out, void* ic_q, void* ic_k, void* ic_v,
    void* ring_rawv, void* ring_rawk, float* ring_g, float* ring_beta,
    IndexArg slot_idx, IndexArg scratch_idx, IndexArg cu_seqlens,
    int H, int N, int t_loop, int ring_len,
    float scale, float lower_bound, VerifyStrides strides, hggcStream_t stream);

}  // namespace kda_mtp
}  // namespace pla
