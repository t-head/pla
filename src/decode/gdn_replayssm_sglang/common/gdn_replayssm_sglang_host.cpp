// SPDX-License-Identifier: Apache-2.0
#include "../../gdn_replayssm/common/gdn_replayssm_host.h"
#include "gdn_replayssm_sglang_abi.h"

namespace pla::gdn_replayssm::host {
namespace {

void commit(T state, T d, T keys, T g, OptionalTensor dr, OptionalTensor kr,
            T slots, T indices, T wp, T base, T flush, T accepted,
            OptionalTensor track, OptionalTensor steps, int64_t null_slot) {
  TORCH_CHECK(dr.has_value() == kr.has_value(), "residual caches must be supplied together");
  TORCH_CHECK(track.has_value() == steps.has_value(), "tracking indices and steps must be supplied together");
  replay::Commit p{};
  p.extended_base = true;
  p.residual = dr.has_value();
  p.tracking = track.has_value();
  p.null_slot = null_index(null_slot);
  Checks check(state);
  p.state = check.tensor(state, "checkpoint_state", 5, 3, Kind::Floating, true);
  p.d = check.tensor(d, "d_cache", 5, 3, Kind::Floating);
  p.keys = check.tensor(keys, "k_cache", 5, 3, Kind::Floating);
  p.g = check.tensor(g, "g_cache", 4, 2, Kind::Float32);
  p.slots = check.tensor(slots, "state_batch_indices", 1, 0, Kind::Index);
  p.replay = check.tensor(indices, "replay_indices", 1, 0, Kind::Index);
  p.wp = check.tensor(wp, "write_pos", 1, 0, Kind::Index, true);
  p.base = check.tensor(base, "cache_base", 1, 0, Kind::Index, true);
  p.flush = check.tensor(flush, "is_flush", 1, 0, Kind::Flush, true);
  p.accepted = check.tensor(accepted, "accept_lens", 1, 0, Kind::Index);
  p.layers = integer(state.size(0), "layers", 1);
  p.state_slots = integer(state.size(1), "state slots");
  p.replay_slots = integer(wp.size(0), "replay slots");
  p.batch = integer(slots.size(0), "batch");
  p.heads = integer(keys.size(2), "H", 1);
  p.value_heads = integer(state.size(2), "HV", 1);
  p.value_dim = integer(state.size(3), "V", 1);
  p.key_dim = integer(state.size(4), "K", 1);
  p.capacity = capacity(d.size(3), "cache capacity");
  TORCH_CHECK(p.key_dim <= 256 && (p.key_dim & (p.key_dim - 1)) == 0,
              "commit K must be a power of two in [1, 256]");
  TORCH_CHECK(p.value_heads % p.heads == 0, "HV must be divisible by H");
  shape(d, {p.layers, p.replay_slots, p.value_heads, p.capacity, p.value_dim}, "d_cache");
  shape(keys, {p.layers, p.replay_slots, p.heads, p.capacity, p.key_dim}, "k_cache");
  shape(g, {p.layers, p.replay_slots, p.value_heads, p.capacity}, "g_cache");
  shape(indices, {p.batch}, "replay_indices");
  shape(accepted, {p.batch}, "accept_lens");
  shape(base, {p.replay_slots}, "cache_base");
  shape(flush, {p.replay_slots}, "is_flush");
  same_dtype(keys, d, "k_cache");
  if (dr) {
    p.dr = check.tensor(*dr, "d_residual_cache", 5, 3, Kind::Floating);
    p.kr = check.tensor(*kr, "k_residual_cache", 5, 3, Kind::Floating);
    shape(*dr, {p.layers, p.replay_slots, p.value_heads, p.capacity, p.value_dim}, "d_residual_cache");
    shape(*kr, {p.layers, p.replay_slots, p.heads, p.capacity, p.key_dim}, "k_residual_cache");
    same_dtype(*dr, d, "d_residual_cache");
    same_dtype(*kr, d, "k_residual_cache");
  }
  if (track) {
    p.track = check.tensor(*track, "mamba_track_indices", 1, 0, Kind::Index);
    p.steps = check.tensor(*steps, "mamba_steps_to_track", 1, 0, Kind::Index);
    shape(*track, {p.batch}, "mamba_track_indices");
    shape(*steps, {p.batch}, "mamba_steps_to_track");
  }
  check.aliases();
  const c10::cuda::CUDAGuard guard(state.device());
  if (p.batch != 0) {
    replay::launch_sglang_materialize(p, pla::current_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    replay::launch_sglang_finish(p, pla::current_stream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cursors(T wp, T base, T flush, T accepted, T indices, int64_t length,
             int64_t max_steps, bool fold_every_commit,
             std::optional<int64_t> buffer_len, int64_t null_slot) {
  replay::Cursors p{};
  p.capacity = capacity(length, "max_cache_len");
  p.max_steps = window(max_steps, length);
  p.buffer_len = capacity(buffer_len.value_or(length), "cache_buf_len");
  TORCH_CHECK(p.buffer_len >= p.capacity, "cache_buf_len must be >= max_cache_len");
  p.null_slot = null_index(null_slot);
  p.fold_every_commit = fold_every_commit;
  Checks check(wp);
  p.wp = check.tensor(wp, "write_pos", 1, 0, Kind::Index, true);
  p.base = check.tensor(base, "cache_base", 1, 0, Kind::Index, true);
  p.flush = check.tensor(flush, "is_flush", 1, 0, Kind::Flush, true);
  p.accepted = check.tensor(accepted, "num_accepted", 1, 0, Kind::Index);
  p.replay = check.tensor(indices, "replay_indices", 1, 0, Kind::Index);
  p.batch = integer(indices.size(0), "batch");
  p.replay_slots = integer(wp.size(0), "replay slots");
  shape(base, {p.replay_slots}, "cache_base");
  shape(flush, {p.replay_slots}, "is_flush");
  shape(accepted, {p.batch}, "num_accepted");
  check.aliases();
  const c10::cuda::CUDAGuard guard(wp.device());
  if (p.batch != 0) replay::launch_cursors(p, pla::current_stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void bind_sglang(pybind11::module_& m) {
  m.def("commit", &commit);
  m.def("cursors", &cursors);
}

}  // namespace pla::gdn_replayssm::host
