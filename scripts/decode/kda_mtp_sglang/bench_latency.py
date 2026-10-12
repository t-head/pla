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

"""Latency A/B curve for pla.decode.kda_mtp_sglang -- verify + fold over bs=1..64.

One row per (kernel, shape) -- L2 flush + median/P10/P90 -- meeting the <20s loop time
needed for CARD=6 iterations. Persisted to ``results/<tag>.json`` (relative to CWD).

Both entries are taken from the installed pla package:

  * verify  ->  ``fused_kda_decode_mtp_dspark`` (CACHE_RING, all-in wrapper)
  * fold    ->  ``commit_kda_replayssm_after_verify`` (one launch covering all layers)

Usage (inside the PPU server container):
    python scripts/decode/kda_mtp_sglang/bench_latency.py --tag mybaseline --kind both
"""
# ruff: noqa: E501
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "private"))

import torch

from _common import Cfg, SpecState, build_commit_inputs, build_verify_inputs
from _harness import Timer

from pla.decode.kda_mtp_sglang import (
    commit_kda_replayssm_after_verify,
    fused_kda_decode_mtp_dspark,
)


def bench_verify(shapes, reps, inner):
    timer = Timer(reps=reps, inner=inner)
    out = {}
    for (bs, heads, num_spec) in shapes:
        cfg = Cfg(bs=bs, num_spec=num_spec, heads=heads, onorm=False)
        vi = build_verify_inputs(cfg)
        ob = torch.empty_like(vi["x_v"])

        def fn():
            fused_kda_decode_mtp_dspark(
                x_q=vi["x_q"], x_k=vi["x_k"], x_v=vi["x_v"],
                w_q=vi["w_q"], w_k=vi["w_k"], w_v=vi["w_v"],
                cs_q=vi["cs_q"], cs_k=vi["cs_k"], cs_v=vi["cs_v"],
                g=vi["g"], beta=vi["beta"], A_log=vi["A_log"], dt_bias=vi["dt_bias"],
                recurrent_state=vi["recurrent_state"],
                intermediate_state_indices=vi["intermediate_state_indices"],
                intermediate_conv_q=vi["intermediate_conv_q"],
                intermediate_conv_k=vi["intermediate_conv_k"],
                intermediate_conv_v=vi["intermediate_conv_v"],
                ssm_state_indices=vi["ssm_state_indices"],
                cu_seqlens=vi["cu_seqlens"],
                lower_bound=float(vi["lower_bound"]),
                scale=float(vi["scale"]),
                replayssm_rawv=vi["replayssm_rawv"], replayssm_rawk=vi["replayssm_rawk"],
                replayssm_g=vi["replayssm_g"], replayssm_beta=vi["replayssm_beta"],
                out=ob,
            )

        fn(); torch.cuda.synchronize()
        r = timer.run(fn)
        key = f"verify_bs{bs}_h{heads}_ns{num_spec}"
        out[key] = r["median_ms"] * 1e3
        print(f"  {key:<28s} = {r['median_ms']*1e3:8.2f}us  "
              f"p10={r['p10_ms']*1e3:7.2f}  p90={r['p90_ms']*1e3:7.2f}  "
              f"spread={r['spread_pct']:.1f}%")
    return out


def bench_fold(shapes, reps, inner):
    timer = Timer(reps=reps, inner=inner)
    out = {}
    for (bs, heads, layers) in shapes:
        cfg = Cfg(bs=bs, num_spec=7, heads=heads, layers=layers)
        ci = build_commit_inputs(cfg)
        st = SpecState.from_commit_inputs(ci)
        st.temporal = ci["temporal"].clone()
        kw = dict(spec_state=st, state_batch_indices=ci["state_batch_indices"],
                  accept_lens=ci["accept_lens"],
                  last_correct_step_indices=ci["last_correct_step_indices"],
                  null_block_id=-1)
        commit_kda_replayssm_after_verify(**kw); torch.cuda.synchronize()
        r = timer.run(lambda: commit_kda_replayssm_after_verify(**kw))
        key = f"fold_bs{bs}_h{heads}_L{layers}"
        out[key] = r["median_ms"] * 1e3
        print(f"  {key:<28s} = {r['median_ms']*1e3:8.2f}us  "
              f"p10={r['p10_ms']*1e3:7.2f}  p90={r['p90_ms']*1e3:7.2f}  "
              f"spread={r['spread_pct']:.1f}%")
    return out


VERIFY_SHAPES = [
    (1, 12, 7), (4, 12, 7), (8, 12, 7),
    (16, 12, 7), (32, 12, 7), (64, 12, 7),
]
FOLD_SHAPES = [
    (1, 12, 69), (4, 12, 69), (8, 12, 69),
    (16, 12, 69), (32, 12, 69), (64, 12, 69),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="results/<tag>.json")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--kind", choices=["verify", "fold", "both"], default="both")
    a = ap.parse_args()
    torch.cuda.synchronize()
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count}")
    results = dict(tag=a.tag)
    if a.kind in ("verify", "both"):
        print("\n-- verify --")
        results["verify"] = bench_verify(VERIFY_SHAPES, a.reps, a.inner)
    if a.kind in ("fold", "both"):
        print("\n-- fold --")
        results["fold"] = bench_fold(FOLD_SHAPES, a.reps, a.inner)
    outp = f"results/{a.tag}.json"
    os.makedirs("results", exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
