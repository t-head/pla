# SPDX-License-Identifier: Apache-2.0
"""Caller-owned Decode cursor policy: fold once after L accepted single-token steps."""
import triton
import triton.language as tl


@triton.jit
def _advance_decode(wp, flags, replay, N: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(replay + row, row < N, other=-1).to(tl.int64)
    active = (row < N) & (slot >= 0)
    length = tl.load(wp + slot, active, other=0) + 1
    tl.store(wp + slot, length, active)
    tl.store(flags + slot, length >= L, active)


def advance_decode(write_pos, is_flush, replay_indices, capacity):
    count = replay_indices.numel()
    if count:
        _advance_decode[(triton.cdiv(count, 256),)](
            write_pos, is_flush, replay_indices, count, capacity, 256)
