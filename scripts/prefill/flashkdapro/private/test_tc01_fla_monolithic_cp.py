#!/usr/bin/env python3
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

"""Device checks for the monolithic tc01 FLA deterministic CP path."""
import argparse
import importlib.util
import math
import os
from pathlib import Path

import torch


SPLITS = (1, 2, 3, 4, 5, 8)
DEVICE_TARGET_TOKENS = (4096, 8192, 16384, 32768)


def device_target_lengths(tokens):
    # Cross the CP5 effective-length boundary without recapturing the graph.
    candidates = (tokens, tokens // 2 + 1, 12287, 12288, 12289,
                  65, 0, 19, tokens - 1, tokens)
    return tuple(length for length in candidates if length <= tokens)


def load_extension(path):
    spec = importlib.util.spec_from_file_location("_C", Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(tokens, heads, seed, physical_tokens=None, slow_decay=False):
    physical_tokens = physical_tokens or tokens
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (physical_tokens, heads, 128)
    q = (torch.randn(shape, device="cuda", generator=generator) * 0.1).bfloat16()
    k = (torch.randn(shape, device="cuda", generator=generator) * 0.1).bfloat16()
    v = (torch.randn(shape, device="cuda", generator=generator) * 0.1).bfloat16()
    g = (torch.randn(shape, device="cuda", generator=generator) * 0.3 - 4.5).bfloat16()
    beta = torch.randn((physical_tokens, heads), device="cuda", generator=generator).bfloat16()
    a_log = torch.full((heads,), 2.0 if slow_decay else 0.0,
                       device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros((heads, 128), device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    return q, k, v, g, beta, a_log, dt_bias, cu


def run(module, data, mode, initial=None, final_dtype=torch.bfloat16,
        pool=None, indices=None, algorithm="state"):
    q, k, v, g, beta, a_log, dt_bias, cu = data
    out = torch.empty_like(q)
    final = None
    if pool is None:
        final = torch.empty((1, q.shape[1], 128, 128), device="cuda", dtype=final_dtype)
    old = os.environ.get("PLA_FLASHKDA_TC01_FLA_CP")
    old_algo = os.environ.get("PLA_FLASHKDA_TC01_FLA_CP_ALGO")
    if mode is None:
        os.environ.pop("PLA_FLASHKDA_TC01_FLA_CP", None)
    else:
        os.environ["PLA_FLASHKDA_TC01_FLA_CP"] = mode
    if algorithm is None:
        os.environ.pop("PLA_FLASHKDA_TC01_FLA_CP_ALGO", None)
    else:
        os.environ["PLA_FLASHKDA_TC01_FLA_CP_ALGO"] = algorithm
    try:
        module.forward(q, k, v, g, beta, a_log, dt_bias, out, None, -5.0,
                       initial, final, cu, pool, indices, (0, int(cu[-1].item())))
        torch.cuda.synchronize()
    finally:
        if old is None:
            os.environ.pop("PLA_FLASHKDA_TC01_FLA_CP", None)
        else:
            os.environ["PLA_FLASHKDA_TC01_FLA_CP"] = old
        if old_algo is None:
            os.environ.pop("PLA_FLASHKDA_TC01_FLA_CP_ALGO", None)
        else:
            os.environ["PLA_FLASHKDA_TC01_FLA_CP_ALGO"] = old_algo
    return out, final


def relative_rms(actual, expected):
    delta = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return float((delta / scale).item())


def correlated_inputs(tokens=8192, heads=4, seed=20261001,
                      key_noise=0.5, random_query=True):
    """Synthetic long-memory case, independent of model weights or head IDs."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (tokens, heads, 128)
    # A common dense direction plus independent noise gives correlated keys
    # (expected pair cosine about 0.8). The inverse itself is bounded, but its
    # nilpotent polynomial has large cancelling intermediate powers. Queries
    # are independent, so the oracle also observes errors away from this axis.
    direction = (torch.arange(128, dtype=torch.float32) % 16) - 8
    k = direction.view(1, 1, 128).expand(shape).clone()
    if key_noise:
        key_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        k += (key_noise * direction.square().mean().sqrt()
              * torch.randn(shape, generator=key_generator))
    k = k.bfloat16()
    if random_query:
        query_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
        q = (torch.randn(shape, generator=query_generator) * 0.1).bfloat16()
    else:
        q = k.clone()
    v = (torch.randn(shape, generator=generator) * 0.1).bfloat16()
    g = torch.full(shape, -10.82).bfloat16()
    beta = torch.full((tokens, heads), math.log(9.0)).bfloat16()
    a_log = torch.zeros(heads, dtype=torch.float32)
    dt_bias = torch.zeros(heads, 128, dtype=torch.float32)
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    return tuple(value.cuda() for value in (q, k, v, g, beta, a_log, dt_bias, cu))


def naive_fp32_reference(data):
    """Token recurrence on CPU FP32, without chunk inverse or TensorCell math."""
    q, k, v, g, beta, a_log, dt_bias, _ = (value.cpu().float() for value in data)
    tokens, heads, dim = q.shape
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * dim ** -0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    beta = beta.sigmoid()
    decay = (-5.0 * (a_log.exp().view(1, heads, 1) * (g + dt_bias)).sigmoid()).exp()
    state = torch.zeros(heads, dim, dim, dtype=torch.float32)  # [H,K,V]
    output = torch.empty_like(q)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for token in range(tokens):
            state.mul_(decay[token, :, :, None])
            residual = v[token] - torch.bmm(k[token, :, None, :], state)[:, 0]
            state.baddbmm_(
                (beta[token, :, None] * k[token])[:, :, None], residual[:, None, :])
            output[token] = torch.bmm(q[token, :, None, :], state)[:, 0]
    finally:
        torch.set_num_threads(old_threads)
    return output.cuda(), state.transpose(-1, -2).contiguous().unsqueeze(0).cuda()


def check_correlated_keys(module, heads=4):
    data = correlated_inputs(heads=heads)
    expected_out, expected_state = naive_fp32_reference(data)
    metrics = {}
    for mode in ("off", "1"):
        out, state = run(module, data, mode)
        finite = bool(torch.isfinite(out.float()).all() and torch.isfinite(state.float()).all())
        out_rms = relative_rms(out, expected_out)
        state_rms = relative_rms(state, expected_state)
        metrics[mode] = dict(finite=finite, output_relative_rms=out_rms,
                             state_relative_rms=state_rms)
        if not finite or not out_rms < 0.01 or not state_rms < 0.01:
            raise AssertionError(f"correlated-key FP32 oracle failed at H={heads}, mode={mode}: {metrics[mode]}")
        torch.testing.assert_close(out.float(), expected_out, atol=0.02, rtol=0.01)
        torch.testing.assert_close(state.float(), expected_state, atol=0.02, rtol=0.01)
    return metrics


def check_case(module, tokens, heads, seed, physical_tokens=None, slow_decay=False,
               with_initial=True):
    data = inputs(tokens, heads, seed, physical_tokens, slow_decay)
    initial = None
    if with_initial:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + 99)
        initial = (torch.randn((1, heads, 128, 128), device="cuda", generator=generator) * 0.02).bfloat16()
    serial_out, serial_state = run(
        module, data, "off", None if initial is None else initial.clone())
    for split in SPLITS:
        out, state = run(
            module, data, str(split), None if initial is None else initial.clone())
        if split == 1:
            if not torch.equal(out, serial_out) or not torch.equal(state, serial_state):
                raise AssertionError(f"CP1 is not bitwise at T={tokens}, H={heads}")
        else:
            torch.testing.assert_close(out, serial_out, atol=0.02, rtol=0.01)
            torch.testing.assert_close(state, serial_state, atol=0.02, rtol=0.01)
        if relative_rms(out, serial_out) >= 0.01 or relative_rms(state, serial_state) >= 0.01:
            raise AssertionError(f"relative RMS failed for split={split}, T={tokens}, H={heads}")
        if not torch.isfinite(out.float()).all() or not torch.isfinite(state.float()).all():
            raise AssertionError(f"non-finite deterministic CP result for split={split}")
        if physical_tokens and physical_tokens > tokens and torch.count_nonzero(out[tokens:]).item():
            raise AssertionError("padded suffix is not zero")


def check_pool(module, tokens, heads, seed, dtype):
    data = inputs(tokens, heads, seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 101)
    pool = torch.randn((3, heads, 128, 128), device="cuda", generator=generator) * 0.02
    pool = pool.to(dtype)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    before = pool.clone()
    out, _ = run(module, data, "5", pool=pool, indices=indices)
    if not torch.equal(pool[0], before[0]) or not torch.equal(pool[2], before[2]):
        raise AssertionError("CP modified an unselected state-pool slot")
    if torch.equal(pool[1], before[1]) or not torch.isfinite(out.float()).all():
        raise AssertionError("CP did not update the selected slot or produced non-finite output")


def check_fp32_state(module, tokens, heads, seed):
    data = inputs(tokens, heads, seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 303)
    initial = torch.randn((1, heads, 128, 128), device="cuda", generator=generator) * 0.02
    serial_out, serial_state = run(module, data, "off", initial.clone(), torch.float32)
    cp_out, cp_state = run(module, data, "5", initial.clone(), torch.float32)
    if cp_state.dtype != torch.float32:
        raise AssertionError("CP changed the requested FP32 final-state dtype")
    torch.testing.assert_close(cp_out, serial_out, atol=0.02, rtol=0.01)
    torch.testing.assert_close(cp_state, serial_state, atol=0.02, rtol=0.01)


def check_determinism(module, heads, seed):
    data = inputs(1024, heads, seed, slow_decay=True)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 505)
    initial = (torch.randn((1, heads, 128, 128), device="cuda", generator=generator) * 0.02).bfloat16()
    expected_out, expected_state = run(module, data, "3", initial.clone())
    for repeat in range(19):
        out, state = run(module, data, "3", initial.clone())
        if not torch.equal(out, expected_out) or not torch.equal(state, expected_state):
            raise AssertionError(
                f"state passing is not bitwise deterministic at H={heads}, repeat={repeat + 2}")


def check_empty(module, heads, seed):
    data = inputs(0, heads, seed, physical_tokens=64)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 707)
    initial = (torch.randn((1, heads, 128, 128), device="cuda", generator=generator) * 0.02).bfloat16()
    out, state = run(module, data, "5", initial.clone())
    if torch.count_nonzero(out).item() or not torch.equal(state, initial):
        raise AssertionError("forced CP changed empty-sequence output/state semantics")


def check_cold_capture(module, heads, seed):
    q, k, v, g, beta, a_log, dt_bias, cu = inputs(64, heads, seed)
    out = torch.empty_like(q)
    os.environ["PLA_FLASHKDA_TC01_FLA_CP"] = "2"
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            module.forward(q, k, v, g, beta, a_log, dt_bias, out, None, -5.0,
                           None, None, cu, None, None, (0, 64))
    except RuntimeError as error:
        if "capture" not in str(error):
            raise
    else:
        raise AssertionError("cold CP Graph capture was not rejected")


def check_auto_route(module, heads, seed):
    # Check both sides of each new AUTO boundary, plus explicit opt-outs.
    probes = {
        4: (63, 64, 65, 96, 127, 128, 129, 2048, 2049, 12287, 12288, 32767, 32768, 65536, 65537),
        6: (63, 64, 65, 8192, 32767, 32768, 65536, 65537),
        8: (63, 64, 65, 8191, 8192, 32768, 65536, 65537),
        12: (8191, 8192, 32768, 65536, 65537),
    }[heads]
    for index, tokens in enumerate(probes):
        data = inputs(tokens, heads, seed + index)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + 100 + index)
        initial = (torch.randn((1, heads, 128, 128), device="cuda", generator=generator) * 0.02).bfloat16()
        for algorithm in (None, "state", "legacy"):
            mode = "off"
            if heads == 4 and algorithm != "legacy":
                if 128 <= tokens <= 2048:
                    mode = "1"
                elif 12288 <= tokens <= 65536:
                    mode = "5"
            elif heads in (6, 8) and tokens >= 64:
                mode = "1"
            elif heads == 12 and tokens >= 8192:
                mode = "1"
            selected_algorithm = algorithm
            if selected_algorithm is None:
                state_bucket = ((heads == 4 and mode in ("1", "5")) or
                                (heads in (6, 8) and 64 <= tokens <= 65536))
                selected_algorithm = "state" if state_bucket else "legacy"
            expected = run(module, data, mode, initial.clone(), algorithm=selected_algorithm)
            for automatic in ("auto", None):
                actual = run(module, data, automatic, initial.clone(), algorithm=algorithm)
                if not all(torch.equal(x, y) for x, y in zip(actual, expected)):
                    raise AssertionError(
                        f"AUTO/default mismatch at H={heads}, T={tokens}, algorithm={algorithm}, split={mode}")
        # A forced single segment without ALGO keeps the established legacy default.
        forced = run(module, data, "1", initial.clone(), algorithm=None)
        legacy = run(module, data, "1", initial.clone(), algorithm="legacy")
        if not all(torch.equal(x, y) for x, y in zip(forced, legacy)):
            raise AssertionError("Explicit CP1 no longer preserves the legacy default")


def check_graph(module, heads, seed):
    q, k, v, g, beta, a_log, dt_bias, cu = inputs(4096, heads, seed)
    out = torch.empty_like(q)
    os.environ["PLA_FLASHKDA_TC01_FLA_CP"] = "5"
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        module.forward(q, k, v, g, beta, a_log, dt_bias, out, None, -5.0,
                       None, None, cu, None, None, (0, 4096))
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        module.forward(q, k, v, g, beta, a_log, dt_bias, out, None, -5.0,
                       None, None, cu, None, None, (0, 4096))
    capture_stream.synchronize()
    captured = out.clone()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(out, captured):
        raise AssertionError("warm CP Graph replay did not rewrite the captured output")


def device_trusted_reference(heads, tokens, length):
    # H4 long device buckets launch a fixed CP5 graph. Inside the trusted CP5
    # AUTO bucket they must match trusted AUTO bit for bit; all other effective
    # lengths keep segment zero only and must match the serial path.
    return heads == 4 and tokens in (16384, 32768) and 12288 <= length <= 65536


def check_device_target_graph(module, heads, tokens, dtype, cu_dtype):
    data = inputs(tokens, heads, 20261028 + heads)
    q, k, v, g, beta, a_log, dt_bias, original_cu = data
    cu = original_cu.to(cu_dtype)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    initial = (torch.randn((4, heads, 128, 128), device="cuda") * 0.02).to(dtype)
    pool, serial_pool = initial.clone(), initial.clone()
    output, serial_output = torch.empty_like(q), torch.empty_like(q)
    keys = ("PLA_FLASHKDA_TC01_FLA_CP", "PLA_FLASHKDA_TC01_FLA_CP_ALGO")
    saved = {key: os.environ.get(key) for key in keys}

    def invoke(out, state):
        module.forward(q, k, v, g, beta, a_log, dt_bias, out, None, -5.0,
                       None, None, cu, state, indices, None)

    try:
        for key in keys:
            os.environ.pop(key, None)
        invoke(output, pool)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            invoke(output, pool)
        side.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=side):
            invoke(output, pool)
        lengths = device_target_lengths(tokens)
        for step, length in enumerate(lengths):
            slot = 1 + step % 2
            cu.copy_(torch.tensor([0, length], device="cuda", dtype=cu_dtype))
            indices.fill_(slot)
            serial_pool.copy_(initial)
            serial_output.fill_(float("nan"))
            if device_trusted_reference(heads, tokens, length):
                module.forward(q, k, v, g, beta, a_log, dt_bias, serial_output, None, -5.0,
                               None, None, cu, serial_pool, indices, (0, length))
            else:
                os.environ[keys[0]] = "off"
                invoke(serial_output, serial_pool)
                os.environ.pop(keys[0], None)
            for replay in (False, True, True):
                pool.copy_(initial)
                output.fill_(float("nan"))
                if replay:
                    captured.replay()
                else:
                    invoke(output, pool)
                torch.cuda.synchronize()
                label = (heads, tokens, dtype, cu_dtype, length, slot, replay)
                if not torch.equal(output, serial_output) or not torch.equal(pool, serial_pool):
                    raise AssertionError(f"device AUTO/reference mismatch: {label}")
                if not torch.isfinite(output.float()).all() or torch.count_nonzero(output[length:]):
                    raise AssertionError(f"device prefix padding is invalid: {label}")
                for guard in range(4):
                    if guard != slot and not torch.equal(pool[guard], initial[guard]):
                        raise AssertionError(f"unselected state slot changed: {label}")
                if length == 0 and not torch.equal(pool[slot], initial[slot].bfloat16().to(dtype)):
                    raise AssertionError(f"empty state boundary changed: {label}")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_device_target_states(module, heads, tokens):
    q, k, v, g, beta, a_log, dt_bias, cu = inputs(tokens, heads, 20261029)
    packed_beta = torch.zeros((tokens, 2 * heads), device="cuda", dtype=beta.dtype)
    packed_beta[:, :heads].copy_(beta)
    beta = packed_beta[:, :heads]
    keys = ("PLA_FLASHKDA_TC01_FLA_CP", "PLA_FLASHKDA_TC01_FLA_CP_ALGO")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        for dtype in (torch.bfloat16, torch.float32):
            initial = (torch.randn((1, heads, 128, 128), device="cuda") * 0.02).to(dtype)
            before = initial.clone()
            for with_initial in (False, True):
                for with_final in (False, True):
                    out, expected = torch.empty_like(q), torch.empty_like(q)
                    final = torch.empty_like(initial) if with_final else None
                    expected_final = torch.empty_like(initial) if with_final else None

                    def invoke(output, state, raw=None):
                        module.forward(q, k, v, g, beta, a_log, dt_bias, output, None, -5.0,
                                       initial if with_initial else None, state, cu, None, None, raw)

                    if device_trusted_reference(heads, tokens, tokens):
                        invoke(expected, expected_final, (0, tokens))
                    else:
                        os.environ[keys[0]] = "off"
                        invoke(expected, expected_final)
                        os.environ.pop(keys[0], None)
                    invoke(out, final)
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        invoke(out, final)
                    side.synchronize()
                    captured = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured, stream=side):
                        invoke(out, final)
                    for replay in (False, True, True):
                        out.fill_(float("nan"))
                        if final is not None:
                            final.fill_(float("nan"))
                        if replay:
                            captured.replay()
                        else:
                            invoke(out, final)
                        torch.cuda.synchronize()
                        label = (heads, tokens, dtype, with_initial, with_final, replay)
                        if not torch.equal(out, expected) or not torch.isfinite(out).all():
                            raise AssertionError(f"device state-mode output mismatch: {label}")
                        if final is not None and not torch.equal(final, expected_final):
                            raise AssertionError(f"device state-mode final mismatch: {label}")
                        if not torch.equal(initial, before):
                            raise AssertionError(f"compact initial state changed: {label}")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module")
    parser.add_argument("--device-targets-only", action="store_true")
    parser.add_argument("--heads", type=int, default=0,
                        help="one CP head, or 0 to test H4/H6/H8/H12")
    args = parser.parse_args()
    module_path = args.module
    if module_path is None:
        spec = importlib.util.find_spec("pla.prefill.flashkdapro._tc01_fla._C")
        if spec is None or spec.origin is None:
            raise RuntimeError("monolithic tc01 FLA extension is not installed")
        module_path = spec.origin
    module = load_extension(module_path)
    os.environ["PLA_FLASHKDA_TC01_FLA_CP_ALGO"] = "state"
    expected_cp_heads = (4, 6, 8, 12) if args.heads == 0 else (args.heads,)
    if (getattr(module, "runtime_abi", None) != "tc01_fla_monolithic_runtime_v2" or
            getattr(module, "kernel_abi", None) != "tc01_fla_packed_v3_cpstate_v1" or
            getattr(module, "cp_abi", None) != "tc01_fla_cp_state_passing_v2" or
            getattr(module, "manifest_schema", None) != "tc01_fla_monolithic_build_v2" or
            not set(expected_cp_heads).issubset(set(getattr(module, "cp_heads", ()))) or
            not set(expected_cp_heads).issubset(set(getattr(module, "supported_heads", ())))):
        raise RuntimeError("loaded extension is not the requested CP-capable monolithic build")
    heads_to_test = (4, 6, 8, 12) if args.heads == 0 else (args.heads,)
    if args.device_targets_only:
        for heads in heads_to_test:
            if heads not in (4, 8):
                continue
            for tokens in DEVICE_TARGET_TOKENS:
                for dtype in (torch.bfloat16, torch.float32):
                    for cu_dtype in (torch.int32, torch.int64):
                        check_device_target_graph(module, heads, tokens, dtype, cu_dtype)
                        print("DEVICE_TARGET_GRAPH_OK", heads, tokens, dtype, cu_dtype, flush=True)
                check_device_target_states(module, heads, tokens)
                print("DEVICE_TARGET_STATES_OK", heads, tokens, flush=True)
        return
    for index, heads in enumerate(heads_to_test):
        seed = 2026092401 + 100 * index
        check_cold_capture(module, heads, seed)
        check_case(module, 64, heads, seed + 1)
        check_case(module, 65, heads, seed + 2)
        check_case(module, 4096, heads, seed + 3)
        check_case(module, 4096, heads, seed + 4, physical_tokens=4160)
        check_case(module, 1024, heads, seed + 30, slow_decay=True)
        check_case(module, 1025, heads, seed + 31, with_initial=False)
        check_determinism(module, heads, seed + 32)
        check_pool(module, 4096, heads, seed + 5, torch.bfloat16)
        check_pool(module, 4096, heads, seed + 6, torch.float32)
        check_fp32_state(module, 65, heads, seed + 7)
        check_empty(module, heads, seed + 8)
        check_auto_route(module, heads, seed + 9)
        check_graph(module, heads, seed + 20)
    if 4 in heads_to_test:
        check_correlated_keys(module, 4)
    print("TC01_FLA_CP_DEVICE_OK")


if __name__ == "__main__":
    main()
