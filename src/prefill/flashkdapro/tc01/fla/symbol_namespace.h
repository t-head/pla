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

// In the monolithic extension, head-dependent types, C kernels and launchers are always isolated per head together.
#define FLA_JOIN_HEAD_IMPL(name, head) name##__fla_h##head
#define FLA_JOIN_HEAD(name, head) FLA_JOIN_HEAD_IMPL(name, head)
#define fla_glm53_gate_aligned_m1 FLA_JOIN_HEAD(fla_glm53_gate_aligned_m1, FLA_HEADS)
#define fla_inter_solve_cute FLA_JOIN_HEAD(fla_inter_solve_cute, FLA_HEADS)
#define fla_recompute_h8 FLA_JOIN_HEAD(fla_recompute_h8, FLA_HEADS)
#define fla_recurrence FLA_JOIN_HEAD(fla_recurrence, FLA_HEADS)
#define fla_glm53_token_intra_cute FLA_JOIN_HEAD(fla_glm53_token_intra_cute, FLA_HEADS)
#define fla_output FLA_JOIN_HEAD(fla_output, FLA_HEADS)
#define FlaRecomputeRawReader FLA_JOIN_HEAD(FlaRecomputeRawReader, FLA_HEADS)
#define FlaRecurrenceRawReader FLA_JOIN_HEAD(FlaRecurrenceRawReader, FLA_HEADS)
#define FlaOutputRawReader FLA_JOIN_HEAD(FlaOutputRawReader, FLA_HEADS)
#define launch_fla_gate_cumsum_bf16_h8_aligned_m1 FLA_JOIN_HEAD(launch_fla_gate_cumsum_bf16_h8_aligned_m1, FLA_HEADS)
#define fla_inter_solve_bf16_d128_bt64_bc16_h8_kernel FLA_JOIN_HEAD(fla_inter_solve_bf16_d128_bt64_bc16_h8_kernel, FLA_HEADS)
#define fla_inter_solve_cute_legacy FLA_JOIN_HEAD(fla_inter_solve_cute_legacy, FLA_HEADS)
#define fla_inter_solve_cute_wide FLA_JOIN_HEAD(fla_inter_solve_cute_wide, FLA_HEADS)
#define fla_inter_solve_bf16_d128_bt64_bc16_h8_wide_kernel FLA_JOIN_HEAD(fla_inter_solve_bf16_d128_bt64_bc16_h8_wide_kernel, FLA_HEADS)
#define launch_fla_inter_solve_bf16_d128_bt64_bc16_h8 FLA_JOIN_HEAD(launch_fla_inter_solve_bf16_d128_bt64_bc16_h8, FLA_HEADS)
#define fla_recompute_h8_bf16_d128_bt64 FLA_JOIN_HEAD(fla_recompute_h8_bf16_d128_bt64, FLA_HEADS)
#define configure_fla_recompute_h8_bf16_d128_bt64 FLA_JOIN_HEAD(configure_fla_recompute_h8_bf16_d128_bt64, FLA_HEADS)
#define launch_fla_recompute_h8_bf16_d128_bt64 FLA_JOIN_HEAD(launch_fla_recompute_h8_bf16_d128_bt64, FLA_HEADS)
#define fla_glm53_recurrence_decay_h8_d128 FLA_JOIN_HEAD(fla_glm53_recurrence_decay_h8_d128, FLA_HEADS)
#define fla_glm53_recurrence_h8_bf16_d128_bt64_nt0 FLA_JOIN_HEAD(fla_glm53_recurrence_h8_bf16_d128_bt64_nt0, FLA_HEADS)
#define fla_glm53_recurrence_h8_bf16_d128_bt64_nt1 FLA_JOIN_HEAD(fla_glm53_recurrence_h8_bf16_d128_bt64_nt1, FLA_HEADS)
#define fla_glm53_recurrence_h8_bf16_d128_bt64_nt2 FLA_JOIN_HEAD(fla_glm53_recurrence_h8_bf16_d128_bt64_nt2, FLA_HEADS)
#define fla_recurrence_packed FLA_JOIN_HEAD(fla_recurrence_packed, FLA_HEADS)
#define configure_fla_glm53_recurrence_h8_bf16_d128_bt64 FLA_JOIN_HEAD(configure_fla_glm53_recurrence_h8_bf16_d128_bt64, FLA_HEADS)
#define launch_fla_glm53_recurrence_h8_bf16_d128_bt64 FLA_JOIN_HEAD(launch_fla_glm53_recurrence_h8_bf16_d128_bt64, FLA_HEADS)
#define token_intra_bf16_d128_bt64_bc16_h8_kernel FLA_JOIN_HEAD(token_intra_bf16_d128_bt64_bc16_h8_kernel, FLA_HEADS)
#define launch_fla_glm53_token_intra_bf16_d128_bt64_bc16_h8 FLA_JOIN_HEAD(launch_fla_glm53_token_intra_bf16_d128_bt64_bc16_h8, FLA_HEADS)
#define fla_glm53_prepare_qg_h8_bf16_d128_bt64 FLA_JOIN_HEAD(fla_glm53_prepare_qg_h8_bf16_d128_bt64, FLA_HEADS)
#define fla_glm53_output_h8_bf16_d128_bt64 FLA_JOIN_HEAD(fla_glm53_output_h8_bf16_d128_bt64, FLA_HEADS)
#define fla_glm53_output_h8_bv128_pair FLA_JOIN_HEAD(fla_glm53_output_h8_bv128_pair, FLA_HEADS)
#define fla_glm53_output_h8_bv128_warpgroups FLA_JOIN_HEAD(fla_glm53_output_h8_bv128_warpgroups, FLA_HEADS)
#define configure_fla_glm53_output_h8_bf16_d128_bt64 FLA_JOIN_HEAD(configure_fla_glm53_output_h8_bf16_d128_bt64, FLA_HEADS)
#define launch_fla_glm53_output_h8_bf16_d128_bt64 FLA_JOIN_HEAD(launch_fla_glm53_output_h8_bf16_d128_bt64, FLA_HEADS)
#define fla_cp_pretriangular_output FLA_JOIN_HEAD(fla_cp_pretriangular_output, FLA_HEADS)
#define configure_fla_cp_pretriangular_output FLA_JOIN_HEAD(configure_fla_cp_pretriangular_output, FLA_HEADS)
#define launch_fla_cp_pretriangular_output FLA_JOIN_HEAD(launch_fla_cp_pretriangular_output, FLA_HEADS)
