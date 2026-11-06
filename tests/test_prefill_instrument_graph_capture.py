#!/usr/bin/env python3
# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PPU prefill instrumentation regression, run against the installed package.

Run: python tests/test_prefill_instrument_graph_capture.py
Each environment/backend combination uses a fresh process. Eager and captured
outputs/state must agree; capture must not copy cu_seqlens to CPU. The CPU
companion test validates HGTX tag contents and push/pop lifetime with doubles;
this runtime test does not parse a profiler trace.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"off": (0, 0), "perf": (1, 0), "show": (0, 1), "both": (1, 1)}


def mark(phase):
    sys.stdout.flush()
    os.write(1, f"@@PHASE {phase}\n".encode())


def worker(family):
    import torch
    from pla.prefill.flashkdapro import flashkda_fwd
    from pla.prefill.flashqla import chunk_gated_delta_rule_fwd, gdn_prefill_packed_qkv_fwd

    assert torch.cuda.is_available(), "requires a PPU device"
    torch.manual_seed(20261105)
    # Include an empty request and a physical padded suffix.
    offsets = (0, 17, 17, 65)
    total, dim = 72, 128
    kda = family.startswith("kda")
    heads, qk_heads = (12, 12) if kda else (8, 2)
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    shape = (total, heads, dim) if kda else (1, total, qk_heads, dim)
    q = (torch.randn(shape, device="cuda") * 0.1).to(torch.bfloat16)
    k = torch.randn_like(q) * 0.1
    vshape = (total, heads, dim) if kda else (1, total, heads, dim)
    v = (torch.randn(vshape, device="cuda") * 0.1).to(torch.bfloat16)
    indices = torch.tensor([3, 1, 4], dtype=torch.int32, device="cuda")
    # FP32 pool with BF16-representable inputs respects both KDA backends'
    # public rounding boundary, including empty-request state publication.
    base = (torch.randn(6, heads, dim, dim, device="cuda") * 0.01).to(torch.bfloat16).float()
    pool = base.clone()
    a_log = torch.full((heads,), -3.0, device="cuda")
    beta_raw = torch.randn(total, heads, device="cuda", dtype=torch.bfloat16)
    if kda:
        gate_raw = torch.randn_like(q)
        bias = torch.zeros(heads, dim, device="cuda")
        output = torch.empty_like(q)
    else:
        gate_raw = torch.randn(total, heads, device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros(heads, device="cuda")
        packed = torch.cat((q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)), dim=1)

    def call():
        pool.copy_(base)
        if kda:
            flashkda_fwd(
                q, k, v, gate_raw, beta_raw, a_log, bias, output,
                state_pool=pool, state_indices=indices, cu_seqlens=cu,
                raw_cu_seqlens=offsets)
            result = output
        elif family == "gdn-packed":
            result = gdn_prefill_packed_qkv_fwd(
                packed, gate_raw, beta_raw, a_log, bias, cu,
                num_qk_heads=qk_heads, num_v_heads=heads,
                initial_state=pool, initial_state_indices=indices,
                raw_cu_seqlens=offsets, auto_cp=False,
                output_final_state=False)[2]
        else:
            result = chunk_gated_delta_rule_fwd(
                q, k, v, None, None, cu_seqlens=cu,
                raw_gate_a=gate_raw, raw_beta_b=beta_raw, A_log=a_log, dt_bias=bias,
                normalize_qk=True, initial_state=pool, initial_state_indices=indices,
                raw_cu_seqlens=offsets, auto_cp=False, output_final_state=False)[2]
        return result.clone(), pool.clone()

    # Warm both the extension and trusted host-offset plans on the launch stream.
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(warm_stream)
    torch.cuda.synchronize()
    mark("eager")
    expected_output, expected_pool = call()
    torch.cuda.synchronize()
    mark("capture")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warm_stream):
        actual_output, actual_pool = call()
    mark("replay")
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(actual_pool, expected_pool, atol=0, rtol=0)
    torch.testing.assert_close(actual_pool[[0, 2, 5]], base[[0, 2, 5]], atol=0, rtol=0)
    tail = actual_output[65:] if kda else actual_output[:, 65:]
    assert not torch.count_nonzero(tail), "padded output must stay zero"
    # The empty request's selected slot is not modified either.
    torch.testing.assert_close(actual_pool[1], base[1], atol=0, rtol=0)
    mark("end")
    print(f"PASS {family}: eager == capture/replay, state slots and padding preserved")


def check_records(text, family, show):
    phases, phase = {}, None
    for line in text.splitlines():
        if line.startswith("@@PHASE "):
            phase = line.split()[1]
            phases[phase] = []
        elif phase and line.startswith(("[GDN]:", "[KDA]:")):
            phases[phase].append(line)
    assert set(phases) == {"eager", "capture", "replay", "end"}, text
    eager, capture = phases["eager"], phases["capture"]
    assert len(eager) == len(capture) == int(show), text
    assert not phases["replay"] and not phases["end"], text
    if show:
        prefix = "[KDA]:[P][3,200]" if family.startswith("kda") else "[GDN]:[P][3,200]"
        assert eager[0].startswith(prefix) and capture[0].startswith(prefix), text
        assert "cu_seqlens_q:[0,17,17,65]" in eager[0], text
        assert "cu_seqlens_q:" not in capture[0], text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("gdn", "gdn-packed", "kda-legacy", "kda-fla"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    import torch
    assert torch.cuda.is_available(), "requires a PPU device"
    families = ["gdn", "gdn-packed", "kda-legacy"]
    if torch.cuda.get_device_capability() == (8, 0):
        families.append("kda-fla")
    failures = []
    for config, (perf, show) in CONFIGS.items():
        for family in families:
            env = dict(os.environ, PPU_LIB_PERF_INSTRUMENT=str(perf),
                       PPU_LIB_SHOW_PARAMS=str(show), PLA_BYPASS_EXEC="0", PLA_SANITY_CHECK="0")
            env.pop("PYTHONPATH", None)
            env.pop("PLA_FLASHKDA_TC01_IMPL", None)
            if family.startswith("kda"):
                env["PLA_FLASHKDA_TC01_IMPL"] = family.split("-", 1)[1]
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", family],
                cwd=ROOT / "tests", env=env, capture_output=True, text=True, timeout=180)
            text = result.stdout + result.stderr
            try:
                assert result.returncode == 0, text
                check_records(result.stdout, family, show)
            except AssertionError as error:
                failures.append(f"{config}/{family}: {error}")
                print(f"FAIL {config}/{family}\n{text}")
            else:
                print(f"PASS {config}/{family}")
    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    main()
