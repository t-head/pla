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
// Per-H entry points of the v-last kernels. Each is defined in its own
// translation unit (fused_sigmoid_gating_{bigbatch,smallbatch}_h<H>.cu) so the
// template cascade compiles in parallel.
//
// The entry points take the torch-free pla::v_last::Args, because their defining
// translation units are compiled against the PPU-native headers and cannot see
// torch (see v_last_args.h).
#include "v_last_args.h"

void pla_v_last_bigbatch_h32(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_bigbatch_h16(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_bigbatch_h8(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_bigbatch_h4(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_bigbatch_h2(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_bigbatch_h1(PLA_V_LAST_ENTRY_PARAMS);

void pla_v_last_smallbatch_h32(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_smallbatch_h16(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_smallbatch_h8(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_smallbatch_h4(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_smallbatch_h2(PLA_V_LAST_ENTRY_PARAMS);
void pla_v_last_smallbatch_h1(PLA_V_LAST_ENTRY_PARAMS);

// Dispatch on the runtime H to the matching per-H entry point.
#define PLA_V_LAST_DISPATCH_H(variant)                                        \
    do {                                                                      \
        if (H == 32) { pla_v_last_##variant##_h32(args); return; }             \
        if (H == 16) { pla_v_last_##variant##_h16(args); return; }             \
        if (H ==  8) { pla_v_last_##variant##_h8(args);  return; }             \
        if (H ==  4) { pla_v_last_##variant##_h4(args);  return; }             \
        if (H ==  2) { pla_v_last_##variant##_h2(args);  return; }             \
        if (H ==  1) { pla_v_last_##variant##_h1(args);  return; }             \
        TORCH_CHECK(false, "pla.decode.gdn.v_last does not support H=", H);    \
    } while (0)
