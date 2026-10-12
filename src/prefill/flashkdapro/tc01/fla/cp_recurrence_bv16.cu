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

// Independent state-only BV16 instantiation. Keep the existing BV32/legacy
// symbols and all head-dependent helper types isolated from this translation unit.
#include "heads.h"
#undef FLA_RECURRENCE_BV16
#define FLA_RECURRENCE_BV16 1
#define FLA_CP_BV16_VARIANT 1
#undef fla_recurrence
#define fla_recurrence FLA_JOIN_HEAD(fla_recurrence_cp_bv16, FLA_HEADS)
#undef FlaRecurrenceRawReader
#define FlaRecurrenceRawReader FLA_JOIN_HEAD(FlaRecurrenceRawReaderCpBv16, FLA_HEADS)
#undef fla_recurrence_packed
#define fla_recurrence_packed FLA_JOIN_HEAD(fla_recurrence_packed_cp_bv16, FLA_HEADS)
#undef fla_cp_pretriangular_output
#define fla_cp_pretriangular_output FLA_JOIN_HEAD(fla_cp_pretriangular_output_bv16, FLA_HEADS)
#undef configure_fla_cp_pretriangular_output
#define configure_fla_cp_pretriangular_output FLA_JOIN_HEAD(configure_fla_cp_pretriangular_output_bv16, FLA_HEADS)
#undef launch_fla_cp_pretriangular_output
#define launch_fla_cp_pretriangular_output FLA_JOIN_HEAD(launch_fla_cp_pretriangular_output_bv16, FLA_HEADS)
#define fla_cp_state_prefix fla_cp_state_prefix_bv16
#include "cp_recurrence_output.cu"
