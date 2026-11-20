# SPDX-License-Identifier: Apache-2.0
"""Complete GDN state cycles, not a model-service or an OPF backend replacement."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from cycle_cases import load_models, make_case, matrix
from cycle_sources import load_cycle_sources, source_manifest


class CycleData:
    def __init__(self, case, device, periods, seed):
        import torch
        self.torch, self.case, self.device, self.periods = torch, case, device, periods
        c = case
        self.pool = c.B + 2
        self.channels = 2 * c.H * c.K + c.HV * c.V
        generator = torch.Generator(device=device).manual_seed(seed)
        def rand(*shape, dtype=torch.bfloat16, scale=1.):
            return torch.randn(*shape, dtype=dtype, device=device, generator=generator) * scale
        self.x = rand(periods, c.rounds, c.layers, c.B, c.T, self.channels)
        self.a = rand(periods, c.rounds, c.layers, c.B * c.T, c.HV, scale=0.7) - 1.5
        self.b = rand(periods, c.rounds, c.layers, c.B * c.T, c.HV)
        self.weights = rand(c.layers, self.channels, 4, scale=0.15)
        self.A_log = rand(c.layers, c.HV, dtype=torch.float32, scale=0.1)
        self.dt_bias = rand(c.layers, c.HV, dtype=torch.float32, scale=0.1)
        self.initial_state = rand(c.layers, self.pool, c.HV, c.V, c.K,
                                  dtype=torch.float32, scale=0.02)
        self.initial_conv = rand(c.layers, self.pool, self.channels, 3, scale=0.1)
        self.slots = torch.arange(1, c.B + 1, device=device, dtype=torch.int32)
        self.replay = self.slots.flip(0).contiguous()
        self.scratch = self.slots.roll(1).contiguous()
        self.starts = torch.arange(0, c.B * c.T + 1, c.T, device=device, dtype=torch.int32)
        self.accepted = torch.full((c.B,), c.A, dtype=torch.int32, device=device)
        self.accepted_step = torch.full((c.B,), c.A - 1, dtype=torch.int32, device=device)
        self.inactive = torch.tensor([0, self.pool - 1], dtype=torch.int64, device=device)

    @staticmethod
    def estimated_bytes(c, periods, backends):
        pool, channels = c.B + 2, (2 * c.H * c.K + c.HV * c.V)
        state = c.layers * pool * c.HV * c.V * c.K * 4
        inputs = periods * c.rounds * c.layers * c.B * c.T * (channels + 2 * c.HV) * 2
        rings = c.layers * pool * c.L * (c.HV * c.V + c.H * c.K) * 4
        snapshots = state * c.T if c.mode == "mtp" and "recurrent" in backends else 0
        replay_runners = sum(b != "recurrent" for b in backends) + int("triton" not in backends)
        return inputs + state * (len(backends) + 4) + rings * replay_runners + snapshots + 512 * 1024**2


class CycleRunner:
    def __init__(self, data, sources, backend):
        import torch
        from cycle_metadata import advance_decode
        self.torch, self.d, self.s, self.backend = torch, data, sources, backend
        self.advance_decode = advance_decode
        self.api = importlib.import_module("pla.decode.gdn_replayssm") if backend == "pla" else None
        c, pool = data.case, data.pool
        self.state = data.initial_state.clone()
        self.conv = data.initial_conv.clone()
        self.wp = torch.zeros(pool, dtype=torch.int32, device=data.device)
        self.base = torch.zeros_like(self.wp)
        self.flags = torch.zeros(pool, dtype=torch.int8, device=data.device)
        self.window_storage = torch.zeros(c.layers, pool, data.channels, c.T + 2,
                                          dtype=torch.bfloat16, device=data.device)
        z = self.window_storage.stride()
        self.windows = self.window_storage.as_strided(
            (c.layers, pool, c.T, data.channels, 3), (z[0], z[1], z[3], z[2], z[3]))
        self.outputs = [None] * (c.rounds * c.layers)
        self.graph = None
        self.snapshots = None
        if backend == "recurrent":
            if c.mode == "mtp":
                self.snapshots = torch.zeros(c.layers, pool, c.T, c.HV, c.V, c.K,
                                              dtype=torch.float32, device=data.device)
            self.dcache = self.kcache = self.gcache = self.dr = self.kr = None
        else:
            self.dcache = torch.zeros(c.layers, pool, c.HV, c.L, c.V,
                                     dtype=torch.bfloat16, device=data.device)
            self.kcache = torch.zeros(c.layers, pool, c.H, c.L, c.K,
                                     dtype=torch.bfloat16, device=data.device)
            self.gcache = torch.zeros(c.layers, pool, c.HV, c.L, dtype=torch.float32, device=data.device)
            self.dr, self.kr = torch.zeros_like(self.dcache), torch.zeros_like(self.kcache)
            self.out = torch.empty(c.rounds, c.layers, c.B * c.T, c.HV, c.V,
                                   dtype=torch.bfloat16, device=data.device)
        if backend == "pla":
            a = self.api
            self.mapping = a.ReplayMapping(data.slots, data.replay)
            self.step_view = a.ReplayStep(self.wp, self.base)
            self.history = a.ReplayHistory(a.HistoryFormat.COMPACT_DKG_RING,
                self.dcache, self.kcache, self.gcache, self.dr, self.kr)
            self.local_history = [a.ReplayHistory(self.history.format,
                self.dcache[i], self.kcache[i], self.gcache[i], self.dr[i], self.kr[i]) for i in range(c.layers)]
            self.work = a.MaterializeWork(state_indices=data.slots, replay_indices=data.replay,
                lengths=self.wp, starts=self.base, metadata_indices=data.replay, enabled=self.flags)
            self.workspace = []
            for layer in range(c.layers):
                packed = data.x[0, 0, layer].view(c.B * c.T, data.channels)
                q, k, v = packed.split([c.H * c.K, c.H * c.K, c.HV * c.V], dim=-1)
                probe = a.ReplayInputs(q.view(-1, c.H, c.K), k.view(-1, c.H, c.K),
                    v.view(-1, c.HV, c.V), data.a[0, 0, layer], data.b[0, 0, layer])
                count = a.gdn_replay_workspace_elements(probe, self.state[layer],
                    self.local_history[layer], self.mapping, c.T)
                self.workspace.append(torch.empty(count, device=data.device, dtype=torch.float32) if count else None)

    def reset(self):
        self.state.copy_(self.d.initial_state)
        self.conv.copy_(self.d.initial_conv)
        self.wp.zero_(); self.base.zero_(); self.flags.zero_()
        self.window_storage.zero_()
        if self.dcache is not None:
            for x in (self.dcache, self.kcache, self.gcache, self.dr, self.kr):
                x.zero_()

    def _finish(self):
        c, d = self.d.case, self.d
        block = max(1, 1 << (c.B - 1).bit_length())
        self.s.spec._finish_gdn_replayssm_circular_fold_kernel[(1,)](
            self.wp, self.base, self.flags, d.slots, d.replay, c.B,
            stride_indices=d.slots.stride(0), stride_replay_indices=d.replay.stride(0),
            MAX_CACHE_LEN=c.L, BLOCK=block, NULL_BLOCK_ID=-1)

    def step(self, period, round_index):
        c, d = self.d.case, self.d
        outputs = []
        for layer in range(c.layers):
            x = d.x[period, round_index, layer].transpose(1, 2)
            processed = self.s.conv.causal_conv1d_update(
                x, self.conv[layer], d.weights[layer], activation="silu",
                conv_state_indices=d.slots,
                intermediate_conv_window=self.windows[layer] if c.mode == "mtp" else None,
                intermediate_state_indices=d.scratch if c.mode == "mtp" else None)
            packed = processed.transpose(1, 2).view(c.B * c.T, d.channels)
            q, k, v = packed.split([c.H * c.K, c.H * c.K, c.HV * c.V], dim=-1)
            q, k, v = q.view(-1, c.H, c.K), k.view(-1, c.H, c.K), v.view(-1, c.HV, c.V)
            aa, bb = d.a[period, round_index, layer], d.b[period, round_index, layer]
            if self.backend == "pla":
                inputs = self.api.ReplayInputs(q, k, v, aa, bb, d.A_log[layer], d.dt_bias[layer])
                if c.mode == "decode":
                    out = self.api.gdn_replay_decode(inputs, self.state[layer], self.local_history[layer],
                        self.mapping, self.step_view, self.out[round_index, layer], self.workspace[layer])
                else:
                    out = self.api.gdn_replay_verify(inputs, self.state[layer], self.local_history[layer],
                        self.mapping, self.step_view, self.out[round_index, layer], self.workspace[layer],
                        query_start_loc=d.starts, max_verify_len=c.T)
            elif self.backend == "triton":
                out = self.s.spec.gdn_replayssm_spec_decode(
                    q, k, v, aa, bb, d.A_log[layer], d.dt_bias[layer], self.state[layer],
                    self.dcache[layer], self.kcache[layer], self.gcache[layer], self.dr[layer], self.kr[layer],
                    None, self.out[round_index, layer], d.starts, d.slots, d.replay,
                    self.wp, self.base, self.flags, c.L, c.T,
                    null_block_id=-1, launch_mode="verify")
            else:
                out = self.s.recurrent.fused_sigmoid_gating_delta_rule_update(
                    A_log=d.A_log[layer], a=aa, dt_bias=d.dt_bias[layer],
                    softplus_beta=1., softplus_threshold=20.,
                    q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), b=bb,
                    initial_state_source=self.state[layer], initial_state_indices=d.slots,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=d.starts,
                    disable_state_update=c.mode == "mtp",
                    intermediate_states_buffer=self.snapshots[layer] if self.snapshots is not None else None,
                    intermediate_state_indices=d.scratch if self.snapshots is not None else None,
                    cache_steps=c.T if self.snapshots is not None else None).squeeze(0)
            self.outputs[round_index * c.layers + layer] = out
            outputs.append(out)
        if self.backend == "recurrent":
            if c.mode == "mtp":
                self.s.scatter.fused_mamba_state_scatter_with_mask(
                    self.state, self.snapshots, d.slots, d.accepted_step, d.scratch)
        else:
            if c.mode == "decode":
                self.advance_decode(self.wp, self.flags, d.replay, c.L)
            else:
                self.s.spec.commit_gdn_replayssm_spec(self.wp, self.base, self.flags,
                    d.accepted, d.replay, c.L, c.T, null_block_id=-1)
            if self.backend == "pla":
                self.api.gdn_replay_materialize(self.state, self.history, self.work)
                self._finish()
            else:
                self.s.spec.commit_gdn_replayssm_circular(
                    checkpoint_state=self.state, d_cache=self.dcache, k_cache=self.kcache,
                    g_cache=self.gcache, d_residual_cache=self.dr, k_residual_cache=self.kr,
                    state_batch_indices=d.slots, replay_indices=d.replay, write_pos=self.wp,
                    cache_base=self.base, is_flush=self.flags, accept_lens=d.accepted, null_block_id=-1)
        if c.mode == "mtp":
            self.s.scatter.fused_conv_window_scatter_with_mask(
                self.conv, self.windows, d.slots, d.accepted_step, d.scratch)
        return outputs

    def cycle(self, period=0):
        for i in range(self.d.case.rounds):
            self.step(period, i)

    def capture(self):
        torch = self.torch
        stream = torch.cuda.Stream(device=self.d.device)
        self.reset()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.cycle()
        torch.cuda.current_stream().wait_stream(stream)
        self.reset()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.cycle()
        torch.cuda.synchronize()
        self.reset()


def relative_l2(actual, expected):
    import torch
    a, b = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise AssertionError("non-finite output/state")
    error = (a - b).flatten(-2).square().sum(-1).sqrt() if a.ndim >= 5 else (a - b).square().sum(-1).sqrt()
    norm = b.flatten(-2).square().sum(-1).sqrt() if b.ndim >= 5 else b.square().sum(-1).sqrt()
    return float((error / norm.clamp_min(1e-8)).max())


def validate_cycle(data, sources, runners, periods):
    import torch
    c = data.case
    reference = next((runner for runner in runners if runner.backend == "triton"), None)
    owns_reference = reference is None
    if owns_reference:
        reference = CycleRunner(data, sources, "triton")
    result = {x.backend: {"golden": "pinned_sglang_triton_compact",
                         "max_output_relative_l2": 0., "max_state_relative_l2": 0.} for x in runners}
    expected_conv = data.initial_conv.index_select(1, data.slots.long()).clone()
    for period in range(periods):
        for step in range(c.rounds):
            ref = reference.step(period, step)
            for runner in runners:
                outputs = ref if runner is reference else runner.step(period, step)
                for a, b in zip(outputs, ref):
                    error = relative_l2(a, b)
                    result[runner.backend]["max_output_relative_l2"] = max(result[runner.backend]["max_output_relative_l2"], error)
                    if error > 0.05:
                        raise AssertionError(f"{runner.backend} output relative-L2 {error} at cycle {period} step {step}")
            raw = data.x[period, step, :, :, :c.A].transpose(-1, -2)
            expected_conv = torch.cat((expected_conv, raw), dim=-1)[..., -3:]
            for runner in ([reference] if owns_reference else []) + runners:
                active = runner.conv.index_select(1, data.slots.long())
                if not torch.equal(active, expected_conv):
                    raise AssertionError(f"{runner.backend} conv rollback differs from accepted raw-input prefix")
        for runner in runners:
            error = relative_l2(runner.state, reference.state)
            result[runner.backend]["max_state_relative_l2"] = max(result[runner.backend]["max_state_relative_l2"], error)
            if error > 0.05:
                raise AssertionError(f"{runner.backend} committed state relative-L2 {error}")
            if runner.backend != "recurrent":
                if bool(runner.wp[data.replay.long()].any()) or bool(runner.flags[data.replay.long()].any()):
                    raise AssertionError("cycle did not complete its capacity fold and metadata finish")
                expected_base = ((period + 1) * c.rounds * c.A) % c.L
                if not bool((runner.base[data.replay.long()] == expected_base).all()):
                    raise AssertionError("base did not advance across complete cycles")
                for cache in (runner.dcache, runner.kcache, runner.gcache, runner.dr, runner.kr):
                    if bool(cache.index_select(1, data.inactive).any()):
                        raise AssertionError("inactive history slot modified")
            for tensor, initial in ((runner.state, data.initial_state), (runner.conv, data.initial_conv)):
                if not torch.equal(tensor.index_select(1, data.inactive), initial.index_select(1, data.inactive)):
                    raise AssertionError("inactive checkpoint/conv slot modified")
    torch.cuda.synchronize()
    if owns_reference:
        del reference
    for runner in runners:
        runner.reset()
    return result


def validate_graph(runner, periods=3):
    import torch
    eager = CycleRunner(runner.d, runner.s, runner.backend)
    runner.capture()
    for _ in range(periods):
        eager.cycle()
        runner.graph.replay()
        torch.cuda.synchronize()
        for a, b in ((runner.state, eager.state), (runner.conv, eager.conv),
                     (runner.wp, eager.wp), (runner.base, eager.base), (runner.flags, eager.flags)):
            if not torch.equal(a, b):
                raise AssertionError(f"{runner.backend} graph/eager state mismatch")
        for a, b in zip(runner.outputs, eager.outputs):
            if not torch.equal(a, b):
                raise AssertionError(f"{runner.backend} graph/eager output mismatch")
    del eager
    runner.reset()


def time_cycle(runner, iterations, warmup, timing, flush, profile=False):
    import torch
    if timing == "graph" and runner.graph is None:
        runner.capture()
    launch = runner.graph.replay if timing == "graph" else runner.cycle
    def prepare():
        runner.reset()
        flush.add_(1)
        torch.cuda.synchronize()
    for _ in range(warmup):
        prepare(); launch(); torch.cuda.synchronize()
    samples = []
    for _ in range(1 if profile else iterations):
        prepare()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("gdn_full_cycle:" + runner.backend + ":" + runner.d.case.key)
        start.record(); launch(); end.record(); end.synchronize()
        if profile:
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        samples.append(start.elapsed_time(end) * 1000.)
    median = statistics.median(samples)
    c = runner.d.case
    return {"cycle_us": median, "round_us": median / c.rounds,
            "accepted_token_us_per_batch": median / (c.rounds * c.A),
            "samples_cycle_us": samples, "timing": timing,
            "measurement": "CUDA-event elapsed for complete cycle, not OPF kernel-active-time sum"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opf-root", type=Path, required=True)
    p.add_argument("--sglang-root", type=Path, required=True)
    p.add_argument("--device")
    p.add_argument("--backends", default="pla,triton")
    p.add_argument("--mode", choices=("decode", "mtp", "all"), default="mtp")
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--model")
    p.add_argument("--tp", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--T", type=int)
    p.add_argument("--A", type=int)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--periods", type=int, default=3)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timing", choices=("eager", "graph"), default="graph")
    p.add_argument("--graph-check", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--allow-large", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    for name in ("tp", "batch", "T", "A", "layers"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name} must be positive")
    if args.mode == "decode" and (args.T not in (None, 1) or args.A not in (None, 1)):
        p.error("Decode requires T=A=1")
    if args.output and (not args.output.parent.is_dir() or args.output.exists()):
        p.error("output must be a new file in an existing directory")
    args.backends = args.backends.split(",")
    if not args.backends or len(set(args.backends)) != len(args.backends) or any(x not in ("pla", "triton", "recurrent") for x in args.backends):
        p.error("backends must be unique selections from pla,triton,recurrent")
    if args.periods < 1 or args.iterations < 1 or args.warmup < 0 or args.start < 0 or (args.limit is not None and args.limit < 1):
        p.error("invalid periods, iterations, warmup or slice")
    if not args.dry_run and (not args.device or not args.device.startswith("cuda:") or not args.device[5:].isdigit()):
        p.error("execution requires explicit --device cuda:N")
    try:
        models = load_models(args.opf_root, names=(args.model,) if args.model else None)
    except ValueError as error:
        p.error(str(error))
    if args.matrix:
        cases = matrix(models, modes=("decode", "mtp") if args.mode == "all" else (args.mode,),
            tps=(args.tp,) if args.tp else (1, 2, 4, 8, 16),
            batches=(args.batch,) if args.batch else (1, 8, 32, 128, 256),
            widths=(args.T,) if args.T else (2, 4, 6, 8), layers=args.layers)
        if args.A is not None: cases = [c for c in cases if c.A == args.A]
    else:
        if args.mode == "all": p.error("--mode all requires --matrix")
        T = 1 if args.mode == "decode" else (args.T or 4)
        A = 1 if args.mode == "decode" else (args.A if args.A is not None else T)
        cases = [make_case(models, next(iter(models)), args.tp or 8, args.batch or 8,
                           T=T, A=A, mode=args.mode, layers=args.layers)]
    total = len(cases)
    cases = cases[args.start:None if args.limit is None else args.start + args.limit]
    if not cases: p.error("case selection is empty")
    if args.profile and (len(cases) != 1 or len(args.backends) != 1):
        p.error("profiling requires exactly one case and one backend")
    if not args.dry_run and max(c.B for c in cases) >= 1024 and not args.allow_large:
        p.error("large B requires --allow-large")
    return args, cases, models, total


def main(argv=None):
    args, cases, models, total = parse_args(argv)
    manifest = source_manifest(args.sglang_root)
    meta = {"type": "metadata", "models": models, "sources": manifest,
            "selected_cases": len(cases), "total_cases_before_slice": total,
            "backends": args.backends, "timing": args.timing,
            "includes": ["native causal conv + SiLU", "strided QKV views", "SSM", "accept/cursor kernels",
                         "capacity materialize", "metadata finish", "MTP native conv rollback"],
            "excludes": ["model projections", "gated output RMSNorm", "sampling", "service scheduler", "communication"],
            "decode_protocol": "normalized-K compact DKG, caller fold after L steps; not the report's raw-key Decode kernel",
            "golden": "current source-pinned SGLang GDN Replay Triton, full state cycle",
            "matrix_threshold": "max headwise relative-L2 <= 0.05; existing strict unit-test budgets remain unchanged",
            "not_an_end_to_end_service_test": True, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if args.dry_run:
        print(json.dumps({**meta, "status": "DRY_RUN", "cases": [asdict(c) | {"rounds": c.rounds, "key": c.key} for c in cases]}, ensure_ascii=False))
        return 0
    import torch
    torch.cuda.set_device(args.device)
    sources = load_cycle_sources(args.sglang_root, args.device)
    meta["device"] = sources.provenance["device"]
    destination = args.output.open("x") if args.output else sys.stdout
    def emit(row):
        destination.write(json.dumps(row, ensure_ascii=False) + "\n"); destination.flush()
    emit(meta)
    failures = 0
    for ordinal, case in enumerate(cases, args.start):
        row = {"type": "case", "index": ordinal, "case": asdict(case), "key": case.key,
               "rounds_per_cycle": case.rounds, "backends": {}}
        try:
            free, _ = torch.cuda.mem_get_info()
            available = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            required = CycleData.estimated_bytes(case, args.periods, args.backends)
            if required > available * 0.75:
                raise MemoryError(f"estimated {required} bytes exceeds 75% of available {available}")
            seed = int.from_bytes(hashlib.sha256(case.key.encode()).digest()[:4], "little")
            data = CycleData(case, args.device, args.periods, seed)
            runners = [CycleRunner(data, sources, b) for b in args.backends]
            validation = {} if args.skip_validation else validate_cycle(data, sources, runners, args.periods)
            if args.graph_check:
                for runner in runners: validate_graph(runner)
            flush = torch.zeros(256 * 1024**2, dtype=torch.uint8, device=args.device)
            for runner in runners:
                row["backends"][runner.backend] = {
                    "validation": validation.get(runner.backend, "NOT_RUN"),
                    "graph_validation": "PASS" if args.graph_check else "NOT_RUN",
                    **time_cycle(runner, args.iterations, args.warmup, args.timing, flush, args.profile)}
            row["status"] = "MEASURED_UNVALIDATED" if args.skip_validation else "PASS"
            emit(row)
            if args.output: print(json.dumps({"index": ordinal, "key": case.key, "status": row["status"]}), flush=True)
            del runner, runners, data, flush
            gc.collect()
        except Exception as exc:
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            emit(row); failures += 1
            if args.output: print(json.dumps(row), flush=True)
            break
    emit({"type": "summary", "selected_cases": len(cases), "failures": failures,
          "status": "FAILED" if failures else "MEASURED_UNVALIDATED" if args.skip_validation else "PASS"})
    if args.output: destination.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
