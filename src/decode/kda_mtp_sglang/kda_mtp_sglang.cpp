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
 * pla.decode.kda_mtp_sglang -- pybind hub for the two SGLang KDA MTP kernels.
 *
 * One namespace = one compiled extension = one PYBIND11_MODULE. The device
 * launchers are defined in the sibling .cu files (kda_mtp_verify.cu,
 * kda_mtp_fold.cu) and only forward-declared here, mirroring how
 * fused_sigmoid_gating.cpp fronts the k-last decode extension.
 *
 * Ported from SGLang's Kimi-K3 KDA MTP-verify / commit path:
 *   verify_dspark        <- fused_kda_decode_mtp_dspark
 *                           (sglang/kernels/ops/kimi_k3/kda_decode_mtp.py)
 *   fold_all_layers      <- kda_replayssm_exact_fold_kernel
 *   conv_window_scatter  <- _fused_conv_window_scatter_with_mask_kernel
 *                           (sglang/kernels/ops/attention/fla/kda_replayssm_spec_decode.py)
 *
 * Both kernels are pure CUDA-core delta-rule recurrences (register-resident
 * state, warp-shuffle reductions); see the .cu files for the device bodies.
 * They use pla's standard __expf / rsqrtf / __bfloat16* conversions -- no
 * per-namespace compat shim -- with a local ppu_sigmoid / ppu_silu inside
 * kda_mtp_verify.cu that mirrors src/decode/k-last/fused_kda_decode_mega_kernel.cuh.
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include "../../bypass_exec.h"
#include "../../op_forge_case.h"

// ---- verify: fused_kda_decode_mtp_dspark (CACHE_RING, read-only state) ------
// Defined in kda_mtp_verify.cu.
void kda_verify_dspark(torch::Tensor x_q, torch::Tensor x_k, torch::Tensor x_v,
                       torch::Tensor g_in, torch::Tensor beta_in,
                       torch::Tensor w_q, torch::Tensor w_k, torch::Tensor w_v,
                       torch::Tensor cs_q, torch::Tensor cs_k, torch::Tensor cs_v,
                       torch::Tensor A_log, torch::Tensor dt_bias,
                       torch::Tensor h0, torch::Tensor out, torch::Tensor ic_q,
                       torch::Tensor ic_k, torch::Tensor ic_v,
                       torch::Tensor ring_rawv, torch::Tensor ring_rawk,
                       torch::Tensor ring_g, torch::Tensor ring_beta,
                       torch::Tensor slot_idx, torch::Tensor scratch_idx,
                       torch::Tensor cu_seqlens, double scale,
                       double lower_bound);

// ---- commit: exact fold + conv rollback -------------------------------------
// Defined in kda_mtp_fold.cu.
void kda_fold_all_layers(torch::Tensor h0, torch::Tensor rawv,
                         torch::Tensor rawk, torch::Tensor gk,
                         torch::Tensor beta, torch::Tensor slot_idx,
                         torch::Tensor accept_lens,
                         std::optional<torch::Tensor> track_idx,
                         std::optional<torch::Tensor> track_step,
                         int64_t num_k_heads, int64_t null_block_id);

void kda_conv_window_scatter(torch::Tensor dst, torch::Tensor src,
                             torch::Tensor dst_idx, torch::Tensor step_idx);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    (void)pla::bypass::enabled();
    (void)pla::instrument::show_params();


  m.def("verify_dspark", &kda_verify_dspark,
        "SGLang KDA MTP verify (fused_kda_decode_mtp_dspark), CACHE_RING mode; "
        "recurrent/conv state read-only, per-step raw inputs written to rings");
  m.def("fold_all_layers", &kda_fold_all_layers,
        "KDA ReplaySSM exact fold: replay the accepted prefix of the rings into "
        "the fp32 checkpoint in place (all layers in one launch)");
  m.def("conv_window_scatter", &kda_conv_window_scatter,
        "Conv-state rollback: scatter the accepted-step conv window snapshot "
        "back into the persistent conv state (byte-exact copy)");
}
