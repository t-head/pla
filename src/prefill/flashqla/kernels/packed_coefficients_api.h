// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <hggc_runtime.h>
#include <cstdint>

// Pg/Ag use the final Swizzle<2,3,3> b16 payload; Exp/Rev remain fp32.
// A is the public, ordinary inverse output. Its Element rounding precedes Ag.
bool gdn_packed_coefficients_tc01_enabled();

void launch_gdn_packed_coefficients_tc01(
    void* a, void* pg, void* ag, float* ex, float* rev,
    const void* q, const void* k, const float* beta, const float* g,
    const int* cu, const int* chunks, int num_chunks, int tokens,
    int H, int Hg, float scale, int dtype, bool fuse_kkt,
    int heads_per_group, hggcStream_t stream);

void launch_gdn_packed_fwd_tc01(
    void* o, const void* q, const void* k, const void* v,
    const void* ag, const float* rev, const float* ex, const void* pg,
    void* raw_state, const int* raw_state_indices, bool state_bf16,
    const int* cu, int tokens, int H, int Hg, float scale, int dtype,
    int64_t v_token_stride, const int* precision_flags,
    int device_index, hggcStream_t stream);
