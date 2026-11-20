// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace flashqla {

// The binding and launcher share this predicate so workspace allocation and
// the factorized prepare kernel cannot select different byte layouts.
inline bool tc02_precompute_x_allowed(
    int capability, int compute_units, int tokens,
    int heads, int key_heads, int segments) {
    if (capability != 89 || compute_units != 39) return false;
    const bool established = tokens == 8192 &&
        ((heads == 12 && key_heads == 4 && segments == 6) ||
         (heads == 6 && key_heads == 2 && segments == 12));
    const bool long_topology =
        (heads == 8 && (key_heads == 2 || key_heads == 4)) ||
        (heads == 16 && (key_heads == 2 || key_heads == 4 || key_heads == 8)) ||
        (heads == 24 && key_heads == 8);
    const bool short_cp4 = heads == 16 && key_heads == 4 &&
        tokens >= 768 && tokens < 1024 && segments == 4;
    return established || short_cp4 || (long_topology && tokens >= 1024 &&
        tokens <= 32831 && segments > 0 && segments <= 128);
}

// This kernel consumes only H-sized state/transition matrices; Hg does not
// change its work or memory layout. Offset views retain the scalar launcher.
inline bool tc02_correct_pipeline_allowed(
    int capability, int compute_units, int raw_batch, int heads, int segments) {
    if (capability != 89 || compute_units != 39) return false;
    const bool established = raw_batch == 1 &&
        ((heads == 12 && segments == 6) || (heads == 6 && segments == 12));
    const bool single = raw_batch == 1 &&
        (heads == 8 || heads == 16 || heads == 24) &&
        segments >= 4 && segments <= 128;
    const bool mixed = raw_batch == 16 && heads == 8 &&
        segments >= 19 && segments <= 64;
    const bool pair = raw_batch == 2 && heads == 8 &&
        segments >= 4 && segments <= 9;
    const bool triplet = raw_batch == 3 && heads == 8 &&
        segments >= 4 && segments <= 9;
    return established || single || mixed || pair || triplet;
}

}  // namespace flashqla
