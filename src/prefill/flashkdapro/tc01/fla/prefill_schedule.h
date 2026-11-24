// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace fla_runtime {

enum class ShortPrefill { None, Direct, Prepared, HalfWarp };
enum class PackedPrefill { None, Bv16, Bv32 };

struct PrefillSchedule {
  ShortPrefill short_path = ShortPrefill::None;
  PackedPrefill packed = PackedPrefill::None;
  bool fused_front = false;
  bool wide_inter = false;
  bool packed_tokens = false;
};

// The caller supplies full trusted offsets and an automatic CP selection.
// Partial requests and device-only layouts keep the ordinary path.
constexpr PrefillSchedule select_prefill_schedule(int heads, int sequences, int tokens) {
  if (tokens <= 0 || tokens > 32768 || sequences <= 0 || sequences > 65535) return {};
  switch (heads) {
    case 3: case 4: case 6: case 8: case 12: case 16:
    case 24: case 32: case 48: case 64: case 96:
      break;
    default:
      return {};
  }

  if (sequences == 1) {
    if (tokens <= 32)
      return {ShortPrefill::Direct};
    if ((heads == 64 && tokens <= 64) ||
        ((heads == 48 || heads == 64) && tokens >= 97 && tokens <= 128))
      return {ShortPrefill::HalfWarp};
    int short_limit = 194;
    switch (heads) {
      case 12: short_limit = 218; break;
      case 48: short_limit = 112; break;
      case 64: short_limit = 0; break;
      case 96: short_limit = 80; break;
    }
    if (tokens <= short_limit) return {ShortPrefill::Prepared};

    const bool wide_h3 = heads == 3 && tokens >= 1025 && tokens <= 1344;
    const bool wide_h12 = heads == 12 && tokens >= 219 && tokens <= 320;
    const bool fused_front = (heads == 3 && tokens >= 1000 && tokens <= 2048) ||
        (heads == 4 && tokens >= 219 && tokens <= 8192) ||
        (heads == 32 && tokens >= 513 && tokens <= 768) || wide_h12;
    const bool packed = (heads == 4 && tokens >= 2049 && tokens <= 12287) ||
        ((heads == 12 || heads == 16) && tokens <= 8191) ||
        (heads >= 24 && tokens <= 512) || (heads == 32 && fused_front);
    return {ShortPrefill::None, packed ? PackedPrefill::Bv32 : PackedPrefill::None,
            fused_front, wide_h3 || wide_h12};
  }

  const bool narrow = ((heads == 3 || heads == 4) && sequences == 2) ||
      (heads <= 12 && tokens <= 1024);
  return {ShortPrefill::None, narrow ? PackedPrefill::Bv16 : PackedPrefill::Bv32,
          false, false, heads >= 64 && sequences >= 9 && tokens >= 4096};
}

}  // namespace fla_runtime
