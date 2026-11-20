# SPDX-License-Identifier: Apache-2.0
"""Cold-launch Replay API benchmark; effective GB/s is NOT actual HBM utilization.

Examples (no implicit device, SSH, installation or card reservation):
  python -B scripts/decode/gdn_replayssm/bench.py --dry-run --case gqa128
  python -B scripts/decode/gdn_replayssm/bench.py --device cuda:0 --phase verify
  python -B scripts/decode/gdn_replayssm/bench.py --device cuda:0 --phase commit --layers 32
  python -B scripts/decode/gdn_replayssm/bench.py --device cuda:0 --profile --phase verify

For ACU, launch this command under the user's existing profiler with profiler-API
capture enabled; --profile brackets ONE public-API invocation, after warmup,
restore and eviction. commit/both/flush may contain multiple device kernels:
this is a single API launch window, NOT a claim of one kernel. Decode profiling
requires layers=1. Use the profiler's kernel filter for individual-kernel counters.
Hardware DRAM counters and a user-supplied peak are needed to establish 80% HBM;
a logical-byte/latency ratio cannot establish it. No memory-copy proxy target.
"""
import argparse
from dataclasses import asdict
import importlib
import json
import math
import statistics
import sys

from reference import (Case, SOURCE_COMMIT, SOURCE_SHA256, UPSTREAM_CASES,
                       compact_kwargs, cursor_kwargs, decode_kwargs, make_case,
                       restore_mutated, snapshot_mutated, torch_module)


def effective_bytes(case, phase, state_dtype="float32", api="sglang"):
    """Per-stage logical compulsory bytes, not traffic or unique DRAM transactions.

    Counts active rows only, shared GQA keys once, high parts for verify history,
    high+low parts for compact materialization, raw buffers for exact folds, and
    checkpoint read/write according to phase. No tile duplication, cache lines,
    spills, internal projection workspace, allocator, eviction, reset copies, or profiler traffic is included.
    Compact source loads a full L even when history<L; this is NOT included as
    compulsory history, so this metric intentionally differs from physical loads.
    """
    c = case.validate()
    activation = 4 if c.dtype == "float32" else 2
    state_size = 4 if state_dtype == "float32" else 2
    B, T, HV, H, K, V = c.B, c.T, c.HV, c.H, c.K, c.V
    layer_count = c.layers
    state_row = HV * V * K * state_size
    record = (HV * V + H * K) * activation
    components = {}
    if phase == "cursor":
        components = {"indices_and_accept_read": B * 16,
                      "cursor_read": B * 9, "cursor_write": B * 9}
    elif phase == "commit":
        components = {"checkpoint_read": B * state_row * layer_count,
                      "checkpoint_write": B * state_row * layer_count,
                      "compact_history_read": B * c.history * (2 * record + HV * 4) * layer_count,
                      "row_metadata_read": B * (16 + 9), "cursor_write": B * 9}
    elif phase in ("verify", "decode", "flush", "both"):
        flush_rows = 0 if phase in ("verify", "decode") else B if phase == "flush" else (B + 1) // 2
        verify_rows = B - flush_rows
        exact = phase not in ("verify", "decode")
        components = {
            "checkpoint_read": (B + flush_rows) * state_row * layer_count,
            "checkpoint_write": flush_rows * state_row * layer_count,
            "qkv_and_gate_inputs_read": B * T * (2 * H * K + HV * V + 2 * HV) * activation * layer_count,
            "gate_parameters_read": 2 * HV * 4 * layer_count if B else 0,
            "verify_high_history_read": verify_rows * c.history * (record + HV * 4) * layer_count,
            "exact_raw_history_read": flush_rows * c.history * (record + 2 * HV * 4) * layer_count,
            "spec_cache_write": B * T * (2 * record + HV * (8 if exact else 4)) * layer_count,
            "output_write": B * T * HV * V * activation * layer_count,
            "row_metadata_read": (B * (16 + 9) + (B + 1) * 8) * layer_count if B else 0,
        }
    else:
        raise ValueError(f"unknown phase: {phase}")
    if api == "common":
        components.pop("cursor_write", None)
        if phase == "decode" and B:
            components["row_metadata_read"] -= (B + 1) * 8 * layer_count
    return dict(effective_bytes=sum(components.values()), components=components,
                meaning="logical compulsory per-stage bytes; NOT actual HBM utilization or measured DRAM bytes",
                exclusions=["tile rereads", "full-L masked/unused history loads", "cache-line effects",
                            "cache residency", "spills", "internal projection workspace", "restore", "eviction", "launch overhead"],
                assumptions="int64 row indices/accept; int32 wp/base; int8 flags; no tracking; all rows active")


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=(*UPSTREAM_CASES, "batch1024", "batch4096"), default="fold64")
    for name in ("B", "H", "HV", "T", "K", "V"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--state-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--api", choices=("sglang", "common"), default="sglang")
    parser.add_argument("--phase", choices=("verify", "decode", "both", "flush", "cursor", "commit"), default="verify")
    parser.add_argument("--device", help="required for execution, e.g. cuda:0; never inferred")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--flush-mib", type=int, default=256, help="evict >=256 MiB BEFORE EACH launch, outside timing")
    parser.add_argument("--peak-gbps", type=float, help="user-provided decimal GB/s; no hardware peak is assumed")
    parser.add_argument("--profile", action="store_true", help="one profiler-API/NVTX public-call window for ACU")
    parser.add_argument("--dry-run", action="store_true", help="stdlib-only JSON; no torch import or device probing")
    parser.add_argument("--allow-large", action="store_true", help="explicitly allow B>=1024 allocations")
    args = parser.parse_args(argv)
    if args.api == "common" and args.phase not in ("verify", "decode", "commit"):
        parser.error("common APIs do not own cursor updates or SGLang inline flush")
    if args.phase == "decode":
        if args.api != "common" or args.T not in (None, 1):
            parser.error("decode requires --api common and T=1")
        args.T = 1
    config = dict(UPSTREAM_CASES.get(args.case, UPSTREAM_CASES["gqa128-wide"]))
    if args.case.startswith("batch"):
        config["B"] = int(args.case[5:])
    config.update({name: getattr(args, name) for name in ("B", "H", "HV", "T", "K", "V")
                   if getattr(args, name) is not None})
    case = Case(**config, L=args.L, layers=args.layers, history=args.history, dtype=args.dtype)
    try:
        case.validate()
    except ValueError as exc:
        parser.error(str(exc))
    if args.iterations < 1 or args.warmup < 0 or args.flush_mib < 256:
        parser.error("require iterations>=1, warmup>=0, flush-mib>=256")
    if args.peak_gbps is not None and (not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0):
        parser.error("--peak-gbps must be finite and positive")
    if not args.dry_run and (not args.device or not args.device.startswith("cuda:") or not args.device[5:].isdigit()):
        parser.error("execution requires explicit --device cuda:N")
    if not args.dry_run and case.B >= 1024 and not args.allow_large:
        parser.error("B>=1024 requires --allow-large; inspect --dry-run first")
    if args.profile and args.phase in ("verify", "decode", "both", "flush") and case.layers != 1:
        parser.error("single-call decode profiling requires --layers 1")
    if case.B == 0 and not args.dry_run:
        parser.error("empty-batch correctness belongs in tests, not bandwidth timing")
    return args, case


def run(args, case):
    accounting = effective_bytes(case, args.phase, args.state_dtype, args.api)
    report = dict(case=asdict(case), phase=args.phase, api=args.api, state_dtype=args.state_dtype,
                  timing_scope="compute_only" if args.api == "common" else "sglang_compatibility_api",
                  includes_conv_rollback=False, includes_full_flush_cycle=False,
                  source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256,
                  defaults_provenance="adjacent fold64 UT; GQA128 shapes from adjacent linear UT; T=4/L=16 adapted",
                  device=args.device, accounting=accounting,
                  flush_bytes=args.flush_mib * 1024**2,
                  claim="NOT actual HBM utilization; 80% HBM is unverified without DRAM counters",
                  peak_gbps_user_provided=args.peak_gbps,
                  upstream_differential="NOT_RUN (use test_gdn_replay.py --sglang-root)",
                  correctness="NOT_RUN by benchmark; run numerical tests separately")
    if args.dry_run:
        report.update(status="DRY_RUN", latency_ms=None, effective_gbps=None)
        return report
    torch = torch_module()
    torch.cuda.set_device(torch.device(args.device))
    api = importlib.import_module(
        "pla.decode.gdn_replayssm" if args.api == "common" else "pla.decode.gdn_replayssm_sglang"
    )
    data = make_case(case, device=args.device, raw=args.phase in ("both", "flush"),
                     state_dtype=args.state_dtype, tight_pool=True)
    replay = data["replay_indices"].long()
    if args.phase in ("flush", "commit"):
        data["is_flush"][replay] = 1
    elif args.phase == "both":
        data["is_flush"][replay[::2]] = 1
    if args.phase == "commit":
        # history is the already accepted prefix for this phase (no cursor call).
        data["accept_lens"].fill_(min(case.history, case.T))
    saved = snapshot_mutated(data)
    flush = torch.zeros(report["flush_bytes"], dtype=torch.uint8, device=args.device)
    if args.api == "common":
        mapping = api.ReplayMapping(data["ssm_state_indices"], data["replay_indices"], data["null_block_id"])
        step = api.ReplayStep(data["write_pos"], data["cache_base"])
        history = api.ReplayHistory(api.HistoryFormat.COMPACT_DKG_RING,
                                    data["d_cache"], data["k_cache"], data["g_cache"],
                                    data["rawv_cache"], data["rawk_cache"])
        if args.phase == "commit":
            work = api.MaterializeWork(state_indices=mapping.state_indices,
                replay_indices=mapping.replay_indices, lengths=step.lengths,
                starts=step.starts, metadata_indices=mapping.replay_indices, enabled=data["is_flush"])
            launches = [lambda: api.gdn_replay_materialize(data["checkpoint_state"], history, work,
                                                          null_block_id=mapping.null_block_id)]
        else:
            launches = []
            for layer in range(case.layers):
                inputs = api.ReplayInputs(data["q"][layer], data["k"][layer], data["v"][layer],
                    data["a"][layer], data["b"][layer], data["A_log"][layer], data["dt_bias"][layer])
                local_history = api.ReplayHistory(history.format, history.values[layer], history.keys[layer],
                    history.gates[layer], history.value_residual[layer], history.key_residual[layer])
                state = data["checkpoint_state"][layer]
                count = api.gdn_replay_workspace_elements(inputs, state, local_history, mapping, case.T)
                workspace = torch.empty(count, dtype=torch.float32, device=args.device) if count else None
                kw = dict(inputs=inputs, state=state, history=local_history, mapping=mapping,
                          step=step, out=data["out"][layer], workspace=workspace)
                if args.phase == "decode":
                    launches.append(lambda kw=kw: api.gdn_replay_decode(**kw))
                else:
                    kw.update(query_start_loc=data["query_start_loc"], max_verify_len=case.T)
                    launches.append(lambda kw=kw: api.gdn_replay_verify(**kw))
    elif args.phase == "cursor":
        kwargs = cursor_kwargs(data)
        launches = [lambda: api.commit_gdn_replayssm_spec(**kwargs)]
    elif args.phase == "commit":
        kwargs = compact_kwargs(data)
        launches = [lambda: api.commit_gdn_replayssm_circular(**kwargs)]
    else:
        kwargs = [decode_kwargs(data, layer, mode=args.phase) for layer in range(case.layers)]
        launches = [lambda kw=kw: api.gdn_replayssm_spec_decode(**kw) for kw in kwargs]
    api_calls = len(launches)

    def prepare():
        # Restore FIRST, evict SECOND. Restoring after eviction would warm state.
        # Eviction is arithmetic on an unrelated buffer, never the timed target.
        restore_mutated(data, saved)
        flush.add_(1)
        torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(args.warmup):
            for launch in launches:
                prepare()
                launch()
                torch.cuda.synchronize()
        samples, per_launch_samples = [], []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(1 if args.profile else args.iterations):
            durations = []
            for launch in launches:
                # Even separate layer calls get their OWN restore/eviction.
                # Never wrap a warm multi-layer inner loop in one timer.
                prepare()
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("gdn_replay_single_api_launch")
                try:
                    start.record()
                    launch()
                    end.record()
                    end.synchronize()
                    durations.append(start.elapsed_time(end))
                finally:
                    if args.profile:
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.cudart().cudaProfilerStop()
            samples.append(sum(durations))
            per_launch_samples.append(durations)
    median = statistics.median(samples)
    gbps = accounting["effective_bytes"] / (median * 1e6) if median > 0 else None
    report.update(status="MEASURED", api_calls_per_sample=api_calls,
                  timed_samples=len(samples), warmup=args.warmup,
                  per_api_launch_ms=per_launch_samples,
                  timing_semantics="sum of independently restored/evicted single-API GPU-event intervals per sample",
                  profile_window="one API invocation; may contain multiple kernels" if args.profile else None,
                  latency_ms=dict(median=median, min=min(samples), p10=percentile(samples, 0.1),
                                  p90=percentile(samples, 0.9), samples=samples),
                  effective_gbps=gbps,
                  effective_peak_fraction=(gbps / args.peak_gbps if gbps is not None and args.peak_gbps else None),
                  peak_fraction_meaning="effective/logical ratio, NOT measured HBM utilization")
    return report


def main(argv=None):
    args, case = parse_args(argv)
    try:
        report = run(args, case)
    except Exception as exc:
        print(json.dumps(dict(status="NOT_MEASURED", error=f"{type(exc).__name__}: {exc}",
                              latency_ms=None, effective_gbps=None)), file=sys.stdout)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
