// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../gdn_replayssm/common/gdn_replayssm_abi.h"

namespace pla::gdn_replayssm {

struct Cursors {
  Tensor wp, base, flush, accepted, replay;
  int batch, replay_slots, capacity, max_steps, buffer_len, null_slot;
  bool fold_every_commit;
};

void launch_sglang_fold_and_verify(const Verify&, hggcStream_t);
void launch_sglang_materialize(const Commit&, hggcStream_t);
void launch_sglang_finish(const Commit&, hggcStream_t);
void launch_cursors(const Cursors&, hggcStream_t);

}  // namespace pla::gdn_replayssm
