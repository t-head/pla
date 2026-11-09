// Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "../kda_replayssm/kda_mtp_abi.h"

namespace pla::kda_mtp {
struct PlanInputs {
  IndexArg sampled, source, offsets, mapping, computed, blocks;
  int batch, capacity, sampler_rows, block_rows, block_cols, slots, Q, null_id;
  int64_t block_row_stride, block_col_stride, block_size;
};
void launch_commit_plan(PlanInputs inputs, int* candidate, int* plan,
                        hggcStream_t stream);
}  // namespace pla::kda_mtp
