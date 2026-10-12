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

/*
 * KDA fused decode "mega" entry — host wrapper + launchers.
 *
 * Exposes fused_kda_decode_mega_forward with the vllm torch.ops._C.fused_kda_decode
 * I/O contract (operates in place on conv_state / state / out). By default the
 * call is served by the PPU-native fusion kernel; setting the environment
 * variable KDA_DECODE_IMPL=ref (or =1) routes the SAME arguments to the ported
 * NVIDIA reference (fused_kda_decode_ref_forward) instead. The selector is read
 * fresh on every call so a single process can compare both implementations.
 *
 * Supported activation dtypes: bfloat16 (at::BFloat16) and float16 (at::Half),
 * selected at runtime from x.scalar_type(). weight / a_log / dt_bias / state /
 * norm_weight stay float32; state_indices is int32.
 *
 * The device kernel lives in fused_kda_decode_mega_kernel.cuh.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

#include "k_last_mega_abi.h"
#include "fused_kda_decode_mega_kernel.cuh"

namespace {

using fused_kda_decode_ppu::fused_kda_decode_ppu_kernel;
using fused_kda_decode_ppu::kHeadElems;

// ----------------------------------------------------------------------------
// Launch helpers.
// ----------------------------------------------------------------------------
template <typename scalar_t, int WARP_NUM, int STAGE_NUM, int CORE_ROWS,
          bool APPLY_ONORM, bool USE_LOWER_BOUND, bool PARALLEL_EPILOGUE>
void launch_ppu_mega_cfg_variant(
    const float* a_log, const void* g, const float* dt_bias,
    const void* x_q, const void* x_k, const void* x_v, const void* beta,
    const float* w_q_t, const float* w_k_t, const float* w_v_t,
    const float* bias_q, const float* bias_k, const float* bias_v,
    void* cs_q, void* cs_k, void* cs_v,
    const void* g_out, const float* o_norm_weight, float norm_eps,
    float* state, const int* state_indices, void* out,
    float scale, float lower_bound, int B, int H, int HV,
    int64_t x_row, int64_t beta_row, int64_t onorm_row,
    int64_t conv_slot, int64_t state_slot, hggcStream_t stream) {
  constexpr int BV = 128;
  constexpr int K = 128;
  constexpr int PIPE_STAGES = STAGE_NUM;

  dim3 grid(1, H, B);
  dim3 block(32, WARP_NUM);

  constexpr pla::KLastSharedLayout shared_layout(
      PIPE_STAGES, K, WARP_NUM, BV, 1, true, false, 1, sizeof(scalar_t));
  // P7 appends one aligned output vector after the common KDA fixed regions.
  // The device uses this same layout for every offset, including the padding
  // between scalar beta and the float4 decay vector.
  constexpr int shared_mem_size = static_cast<int>(
      shared_layout.total_bytes + (PARALLEL_EPILOGUE ? sizeof(float) * BV : 0));

  auto kernel =
      fused_kda_decode_ppu_kernel<scalar_t, WARP_NUM, BV, STAGE_NUM, CORE_ROWS,
                                  APPLY_ONORM, USE_LOWER_BOUND,
                                  PARALLEL_EPILOGUE>;
  if (shared_mem_size > 48 * 1024) {
    hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem_size);
  }

  kernel<<<grid, block, shared_mem_size, stream>>>(
      a_log, reinterpret_cast<const scalar_t*>(g), dt_bias,
      reinterpret_cast<const scalar_t*>(x_q),
      reinterpret_cast<const scalar_t*>(x_k),
      reinterpret_cast<const scalar_t*>(x_v),
      reinterpret_cast<const scalar_t*>(beta),
      w_q_t, w_k_t, w_v_t, bias_q, bias_k, bias_v,
      reinterpret_cast<scalar_t*>(cs_q), reinterpret_cast<scalar_t*>(cs_k),
      reinterpret_cast<scalar_t*>(cs_v),
      reinterpret_cast<const scalar_t*>(g_out), o_norm_weight, norm_eps,
      state, state_indices, reinterpret_cast<scalar_t*>(out),
      scale, lower_bound, B, H, HV,
      x_row, beta_row, onorm_row, conv_slot, state_slot);
}

template <typename scalar_t, int WARP_NUM, int STAGE_NUM, int CORE_ROWS,
          bool APPLY_ONORM, bool USE_LOWER_BOUND>
void launch_ppu_mega_cfg(
    const float* a_log, const void* g, const float* dt_bias,
    const void* x_q, const void* x_k, const void* x_v, const void* beta,
    const float* w_q_t, const float* w_k_t, const float* w_v_t,
    const float* bias_q, const float* bias_k, const float* bias_v,
    void* cs_q, void* cs_k, void* cs_v,
    const void* g_out, const float* o_norm_weight, float norm_eps,
    float* state, const int* state_indices, void* out,
    float scale, float lower_bound, int B, int H, int HV,
    int64_t x_row, int64_t beta_row, int64_t onorm_row,
    int64_t conv_slot, int64_t state_slot, hggcStream_t stream) {
  // P8/P9: same-binary candidates are selected by total CTA count. After the
  // P9 WN8 Stage-A split, serial P6 still wins decisively at 384 CTAs, while
  // parallel P7 wins at 768 CTAs and above.
  const int num_ctas = B * H;
  static const int env_parallel_epilogue = []() {
    const char* s = std::getenv("KDA_PE");
    return s ? std::atoi(s) : -1;  // -1=auto, 0=P6 serial, 1=P7 parallel
  }();
  const bool auto_parallel_epilogue =
      !APPLY_ONORM || num_ctas != 384;
  const bool use_parallel_epilogue = env_parallel_epilogue == 0
      ? false
      : (env_parallel_epilogue == 1 ? true : auto_parallel_epilogue);

#define KDA_LAUNCH_EPILOGUE(PARALLEL)                                         \
  launch_ppu_mega_cfg_variant<scalar_t, WARP_NUM, STAGE_NUM, CORE_ROWS,       \
                              APPLY_ONORM, USE_LOWER_BOUND, PARALLEL>(        \
      a_log, g, dt_bias, x_q, x_k, x_v, beta, w_q_t, w_k_t, w_v_t, bias_q,   \
      bias_k, bias_v, cs_q, cs_k, cs_v, g_out, o_norm_weight, norm_eps,      \
      state, state_indices, out, scale, lower_bound, B, H, HV, x_row,         \
      beta_row, onorm_row, conv_slot, state_slot, stream)

  if (use_parallel_epilogue) {
    KDA_LAUNCH_EPILOGUE(true);
  } else {
    KDA_LAUNCH_EPILOGUE(false);
  }
#undef KDA_LAUNCH_EPILOGUE
}

template <typename scalar_t, bool APPLY_ONORM, bool USE_LOWER_BOUND>
void launch_ppu_mega(
    const float* a_log, const void* g, const float* dt_bias,
    const void* x_q, const void* x_k, const void* x_v, const void* beta,
    const float* w_q_t, const float* w_k_t, const float* w_v_t,
    const float* bias_q, const float* bias_k, const float* bias_v,
    void* cs_q, void* cs_k, void* cs_v,
    const void* g_out, const float* o_norm_weight, float norm_eps,
    float* state, const int* state_indices, void* out,
    float scale, float lower_bound, int B, int H, int HV,
    int64_t x_row, int64_t beta_row, int64_t onorm_row,
    int64_t conv_slot, int64_t state_slot, hggcStream_t stream) {
  // Production uses WARP_NUM=8 and selects both pipeline depth and recurrent
  // rows/iteration from the CTA count.  The four-row core wins at B*H>=6144,
  // where the large grid absorbs its 80-register / 75%-occupancy cost.  Below
  // that, the two-row core keeps SN3 for B*H>=1536 and SN4 otherwise. BV stays
  // pinned to 128 because gated RMSNorm needs a whole value head in one CTA.
  //
  // Benchmark overrides: KDA_WN / KDA_SN / KDA_CR select warp count, actual
  // pipeline stages, and recurrent core rows (2 or 4).  An explicit KDA_SN
  // without KDA_CR preserves the legacy two-row interpretation. KDA_PE forces
  // the P6/P7 epilogue (0=serial, 1=parallel, unset=auto).
  static const int env_wn = []() {
    const char* s = std::getenv("KDA_WN");
    return s ? std::atoi(s) : 8;
  }();
  static const int env_sn = []() {
    const char* s = std::getenv("KDA_SN");
    return s ? std::atoi(s) : 0;  // 0=auto
  }();
  static const int env_core_rows = []() {
    const char* s = std::getenv("KDA_CR");
    return s ? std::atoi(s) : 0;  // 0=auto, otherwise 2 or 4
  }();
  const int num_ctas = B * H;
  const int selected_core_rows = env_core_rows != 0
      ? env_core_rows
      : (env_sn != 0 ? 2 : (num_ctas >= 6144 ? 4 : 2));
  const int selected_sn = env_sn != 0
      ? env_sn
      : (selected_core_rows == 4 ? 4 : (num_ctas >= 1536 ? 3 : 4));

#define KDA_DISPATCH_CFG(WN, SN, CR)                                          \
  if (env_wn == (WN) && selected_sn == (SN) && selected_core_rows == (CR)) {   \
    launch_ppu_mega_cfg<scalar_t, WN, SN, CR, APPLY_ONORM, USE_LOWER_BOUND>(  \
        a_log, g, dt_bias, x_q, x_k, x_v, beta, w_q_t, w_k_t, w_v_t, bias_q,  \
        bias_k, bias_v, cs_q, cs_k, cs_v, g_out, o_norm_weight, norm_eps,     \
        state, state_indices, out, scale, lower_bound, B, H, HV, x_row,       \
        beta_row, onorm_row, conv_slot, state_slot, stream);                  \
    return;                                                                   \
  }

  KDA_DISPATCH_CFG(8, 4, 2)
  KDA_DISPATCH_CFG(8, 3, 2)
  KDA_DISPATCH_CFG(8, 5, 2)
  KDA_DISPATCH_CFG(8, 6, 2)
  KDA_DISPATCH_CFG(8, 4, 4)
  KDA_DISPATCH_CFG(4, 3, 2)
  KDA_DISPATCH_CFG(4, 4, 2)
  KDA_DISPATCH_CFG(4, 5, 2)
  KDA_DISPATCH_CFG(4, 6, 2)
#undef KDA_DISPATCH_CFG

  // Unrecognised override -> production WN8/CTA-count policy.
  if (num_ctas >= 6144) {
    launch_ppu_mega_cfg<scalar_t, 8, 4, 4, APPLY_ONORM, USE_LOWER_BOUND>(
        a_log, g, dt_bias, x_q, x_k, x_v, beta, w_q_t, w_k_t, w_v_t, bias_q,
        bias_k, bias_v, cs_q, cs_k, cs_v, g_out, o_norm_weight, norm_eps, state,
        state_indices, out, scale, lower_bound, B, H, HV, x_row, beta_row,
        onorm_row, conv_slot, state_slot, stream);
  } else if (num_ctas >= 1536) {
    launch_ppu_mega_cfg<scalar_t, 8, 3, 2, APPLY_ONORM, USE_LOWER_BOUND>(
        a_log, g, dt_bias, x_q, x_k, x_v, beta, w_q_t, w_k_t, w_v_t, bias_q,
        bias_k, bias_v, cs_q, cs_k, cs_v, g_out, o_norm_weight, norm_eps, state,
        state_indices, out, scale, lower_bound, B, H, HV, x_row, beta_row,
        onorm_row, conv_slot, state_slot, stream);
  } else {
    launch_ppu_mega_cfg<scalar_t, 8, 4, 2, APPLY_ONORM, USE_LOWER_BOUND>(
        a_log, g, dt_bias, x_q, x_k, x_v, beta, w_q_t, w_k_t, w_v_t, bias_q,
        bias_k, bias_v, cs_q, cs_k, cs_v, g_out, o_norm_weight, norm_eps, state,
        state_indices, out, scale, lower_bound, B, H, HV, x_row, beta_row,
        onorm_row, conv_slot, state_slot, stream);
  }
}

template <typename scalar_t>
void dispatch_ppu_mega(
    bool apply_onorm, bool use_lower_bound,
    const float* a_log, const void* g, const float* dt_bias,
    const void* x_q, const void* x_k, const void* x_v, const void* beta,
    const float* w_q_t, const float* w_k_t, const float* w_v_t,
    const float* bias_q, const float* bias_k, const float* bias_v,
    void* cs_q, void* cs_k, void* cs_v,
    const void* g_out, const float* o_norm_weight, float norm_eps,
    float* state, const int* state_indices, void* out,
    float scale, float lower_bound, int B, int H, int HV,
    int64_t x_row, int64_t beta_row, int64_t onorm_row,
    int64_t conv_slot, int64_t state_slot, hggcStream_t stream) {
#define LAUNCH_PPU(ONORM, LB)                                                  \
  launch_ppu_mega<scalar_t, ONORM, LB>(                                        \
      a_log, g, dt_bias, x_q, x_k, x_v, beta, w_q_t, w_k_t, w_v_t, bias_q,     \
      bias_k, bias_v, cs_q, cs_k, cs_v, g_out, o_norm_weight, norm_eps, state, \
      state_indices, out, scale, lower_bound, B, H, HV, x_row, beta_row,       \
      onorm_row, conv_slot, state_slot, stream)
  if (apply_onorm) {
    if (use_lower_bound) { LAUNCH_PPU(true, true); }
    else { LAUNCH_PPU(true, false); }
  } else {
    if (use_lower_bound) { LAUNCH_PPU(false, true); }
    else { LAUNCH_PPU(false, false); }
  }
#undef LAUNCH_PPU
}

}  // namespace

// Non-template POD entry point for the host side. The dtype decision has to be
// made on this side of the boundary: the host TU cannot name __half /
// __ppu_bfloat16 (it speaks at::Half / at::BFloat16), and this TU cannot see
// torch, so the two meet over pla::k_last::MegaDType.
void launch_kda_decode_mega(
    pla::k_last::MegaDType act_dtype,
    bool apply_onorm, bool use_lower_bound,
    const float* a_log, const void* g, const float* dt_bias,
    const void* x_q, const void* x_k, const void* x_v, const void* beta,
    const float* w_q_t, const float* w_k_t, const float* w_v_t,
    const float* bias_q, const float* bias_k, const float* bias_v,
    void* cs_q, void* cs_k, void* cs_v,
    const void* g_out, const float* o_norm_weight, float norm_eps,
    float* state, const int* state_indices, void* out,
    float scale, float lower_bound, int B, int H, int HV,
    int64_t x_row, int64_t beta_row, int64_t onorm_row,
    int64_t conv_slot, int64_t state_slot, hggcStream_t stream) {
#define PLA_MEGA_ARGS                                                          \
  apply_onorm, use_lower_bound, a_log, g, dt_bias, x_q, x_k, x_v, beta, w_q_t, \
      w_k_t, w_v_t, bias_q, bias_k, bias_v, cs_q, cs_k, cs_v, g_out,           \
      o_norm_weight, norm_eps, state, state_indices, out, scale, lower_bound,  \
      B, H, HV, x_row, beta_row, onorm_row, conv_slot, state_slot, stream
  if (act_dtype == pla::k_last::MegaDType::kBFloat16) {
    dispatch_ppu_mega<__ppu_bfloat16>(PLA_MEGA_ARGS);
  } else {
    dispatch_ppu_mega<__half>(PLA_MEGA_ARGS);
  }
#undef PLA_MEGA_ARGS
}
