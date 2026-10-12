# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Kimi K3 KDA fused decode -- pad slot (PAD_SLOT_ID = -1) behavior test.

The two frameworks fill CUDA-graph padding rows with different sentinels:

  * sglang ``MambaPool`` allocates one extra scratch row, slot 0 is valid, ``PAD_SLOT_ID = -1``;
    -1 has no backing memory, so ``slot * state_slot`` yields a negative out-of-bounds offset.
  * vLLM ``BlockPool`` reserves block 0 as the null_block, ``NULL_BLOCK_ID = 0``;
    padding rows read/write slot 0, which is real memory, so the original kernel has no check at all.

The kernel guard is an **unconditional** ``slot < 0``, which satisfies both: sglang's -1 is blocked,
vLLM's 0 does not trigger it and is computed as usual (bit-identical to its original), so no is_sglang parameter is needed.
This is exactly what sglang does when it vendors the same NVIDIA kernel
(``kda_fused_decode.cuh``: "padded cuda-graph slots (index < 0) zero the output
row and skip all pool updates").

This test asserts:
  1. the output of pad rows (slot=-1) is **exactly zero**;
  2. pad rows do not write state / conv_state (that slot keeps its initial value bit for bit);
  3. the output and state/conv updates of non-pad rows are **bit-identical** to the "no pad rows" run
     -- i.e. the guard does not disturb normal rows;
  4. slot 0 (vLLM's NULL_BLOCK_ID) is still computed normally and not treated as pad.

Covers both the ppu fused kernel and the ported reference (``KDA_DECODE_IMPL=ref``).

Usage (inside the PPU server container):
    CUDA_VISIBLE_DEVICES=1 python scripts/decode/kda_fusion/test_pad_slot.py
"""

# ruff: noqa: E501
import os
import sys

import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../scripts/decode/kda_fusion
PARENT_DIR = os.path.dirname(SCRIPTS_DIR)                  # .../scripts/decode
# NOTE: do NOT add the repo root to sys.path. The source tree's ``pla/``
# package would shadow the installed one and has no compiled ``_C``.
for _p in (SCRIPTS_DIR, PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from kda_fusion.compare_kda_decode_fusion import (
        D,
        DEVICE,
        NORM_EPS,
        _alloc_paged_cache,
        build_inputs,
    )
except ModuleNotFoundError:
    from compare_kda_decode_fusion import (  # type: ignore
        D,
        DEVICE,
        NORM_EPS,
        _alloc_paged_cache,
        build_inputs,
    )

from pla.decode.kda import fused_kda_decode_mega_forward


def _run(inp, impl, fuse_output_norm=True, lower_bound=None):
    """Run one fused decode and return (out, state, conv). impl: 'ppu' | 'ref'."""
    if impl == "ref":
        os.environ["KDA_DECODE_IMPL"] = "ref"
    else:
        os.environ.pop("KDA_DECODE_IMPL", None)

    conv, state = _alloc_paged_cache(inp, conv_dtype=torch.bfloat16)
    out = torch.empty(
        1, inp["num_seqs"], inp["num_heads"], D, dtype=torch.bfloat16, device=DEVICE
    )
    fused_kda_decode_mega_forward(
        x=inp["packed_x"],
        weight=inp["fused_weight"],
        bias=None,
        conv_state=conv,
        raw_g=inp["raw_g"],
        raw_beta=inp["raw_beta"],
        a_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        state_indices=inp["state_indices"],
        state=state,
        out=out,
        lower_bound=lower_bound,
        output_gate=inp["output_gate"] if fuse_output_norm else None,
        norm_weight=inp["norm_weight"] if fuse_output_norm else None,
        norm_eps=NORM_EPS,
    )
    torch.cuda.synchronize()
    os.environ.pop("KDA_DECODE_IMPL", None)
    return out, state, conv


def _case(impl, num_heads=12, num_seqs=8, pad_rows=(0, 3), seed=7):
    """Pad some requests and compare bit for bit against the "all valid" baseline."""
    inp = build_inputs(num_heads, num_seqs, seed)
    # Baseline: all slots valid. build_inputs gives num_seqs..1; change it to 0..num_seqs-1
    # so slot 0 (vLLM's NULL_BLOCK_ID) takes part too, verifying it is not treated as pad.
    base_indices = torch.arange(num_seqs, dtype=torch.int32, device=DEVICE)
    inp["state_indices"] = base_indices.clone()
    out_ref, state_ref, conv_ref = _run(inp, impl)

    # Pad version: the chosen rows become -1, everything else unchanged
    pad_indices = base_indices.clone()
    for r in pad_rows:
        pad_indices[r] = -1
    inp["state_indices"] = pad_indices
    out_pad, state_pad, conv_pad = _run(inp, impl)

    keep = [r for r in range(num_seqs) if r not in pad_rows]
    tag = f"[{impl}] H={num_heads} B={num_seqs} pad={list(pad_rows)}"
    ok = True

    # 1. pad row output is exactly zero
    for r in pad_rows:
        nz = torch.count_nonzero(out_pad[0, r]).item()
        if nz != 0:
            print(f"  FAIL {tag}: pad row {r} emitted {nz} non-zero outputs")
            ok = False

    # 2. the pool slot of a pad row must keep its initial value (that slot is referenced only by this row)
    for r in pad_rows:
        slot = int(base_indices[r].item())
        if not torch.equal(state_pad[slot], inp["state_seed"][slot]):
            print(f"  FAIL {tag}: state slot {slot} was written by a pad row")
            ok = False
        if not torch.equal(conv_pad[slot], inp["conv_seed"][slot]):
            print(f"  FAIL {tag}: conv slot {slot} was written by a pad row")
            ok = False

    # 3. non-pad rows must be bit-identical to the baseline (the guard must not disturb normal rows)
    for r in keep:
        if not torch.equal(out_pad[0, r], out_ref[0, r]):
            print(f"  FAIL {tag}: non-pad row {r} output changed by the guard")
            ok = False
        slot = int(base_indices[r].item())
        if not torch.equal(state_pad[slot], state_ref[slot]):
            print(f"  FAIL {tag}: non-pad row {r} state (slot {slot}) changed")
            ok = False
        if not torch.equal(conv_pad[slot], conv_ref[slot]):
            print(f"  FAIL {tag}: non-pad row {r} conv (slot {slot}) changed")
            ok = False

    # 4. if slot 0 is not a pad row it must have been computed normally (vLLM NULL_BLOCK_ID is not pad)
    if 0 in keep:
        if torch.equal(state_ref[0], inp["state_seed"][0]):
            print(f"  FAIL {tag}: slot 0 was skipped -- it must stay valid "
                  f"(vLLM NULL_BLOCK_ID is a real reserved row)")
            ok = False

    print(f"  {'PASS' if ok else 'FAIL'} {tag}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA device not available; run this on the PPU server.")
        return 0

    print("KDA fused decode pad-slot (PAD_SLOT_ID = -1) tests")
    results = []
    for impl in ("ppu", "ref"):
        results.append(_case(impl, num_heads=12, num_seqs=8, pad_rows=(0, 3)))
        results.append(_case(impl, num_heads=12, num_seqs=4, pad_rows=(3,)))
        results.append(_case(impl, num_heads=24, num_seqs=8, pad_rows=(1, 7)))

    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
