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

"""Replay every OperatorsForge GDN/KDA prefill record against native PLA.

The OperatorsForge lists describe packed prefill shapes, but its PPU KDA
backend currently dispatches prefill to SGLang Triton.  This runner calls
``pla.prefill.flashqla`` / ``pla.prefill.flashkdapro`` directly so a successful
replay is evidence for the extensions built from this repository.

Accuracy mode either compares PLA with the independent FLA Triton
implementation or compares the selected production ABI with its serial/base
path.  The runner discovers and reports the inventory at runtime instead of
assuming a fixed list or record count.  Use the independent oracle on top of
the canonical torch-reference suite as a cross-implementation sample.
Performance mode measures the selected PLA ABI; split and packed GDN runs can
therefore replay the same record selection for an isolated A/B comparison.
The record index is global and stable, so a long sweep can be distributed over
several devices with ``--num-shards`` and ``--shard-index``.

Example (run outside the source root after installing the wheel)::

    CUDA_VISIBLE_DEVICES=0 python generalized_prefill_replay.py \
      --list-dir /path/to/GeneralizedUTLists/inference/prefill \
      --family all --mode accuracy --output /tmp/accuracy-0.jsonl \
      --num-shards 8 --shard-index 0
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

# Source checkouts are common in the PPU containers because the packaged FLA
# extension is tied to a specific torch ABI.
if os.environ.get("FLA_SOURCE_PATH"):
    sys.path.insert(0, os.environ["FLA_SOURCE_PATH"])

import torch
import torch.nn.functional as F


D = 128
DEFAULT_TOL = 2e-2
MAX_SGLANG_FUSED_QKV_SPLIT_DIM = 8192
GDN_GATE_DECAY_DIVISOR = 16
GDN_SWA_RATIO = 0.75


def _split_fields(text: str) -> list[str]:
    fields: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            fields.append("".join(current))
            current = []
            continue
        current.append(char)
        depth += int(char == "[") - int(char == "]")
    if current:
        fields.append("".join(current))
    return fields


def _value(text: str) -> Any:
    text = text.strip().rstrip(".")
    if text.startswith("["):
        return ast.literal_eval(text)
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def parse_record(path: Path, line_number: int, line: str, family: str) -> dict[str, Any]:
    marker = f"{family}_attn:"
    body = line.split(marker, 1)[1]
    params = {}
    for field in _split_fields(body):
        if ":" not in field:
            continue
        key, value = field.split(":", 1)
        params[key] = _value(value)
    cu = tuple(int(x) for x in params["cu_seqlens_q"])
    seq_lens = tuple(b - a for a, b in zip(cu, cu[1:]))
    if len(seq_lens) != int(params["batch_size"]):
        raise ValueError(f"batch/cumulative-length mismatch in {path}:{line_number}")
    return {
        "family": family,
        "source": path.name,
        "line": line_number,
        "H": int(params["num_heads_v"]),
        "Hg": int(params["num_heads_qk"]),
        "N": len(seq_lens),
        "T": cu[-1],
        "cu": cu,
        "seq_lens": seq_lens,
        "lower": float(params.get("gate_lower_bound", -5.0)),
    }


def discover_case_lists(list_dir: Path, family: str) -> list[tuple[str, Path]]:
    """Return the deterministic case-list inventory selected by ``family``."""
    families = ("gdn", "kda") if family == "all" else (family,)
    return [
        (item_family, path)
        for item_family in families
        for path in sorted(list_dir.glob(f"*_{item_family}_op.list"))
        if not path.name.startswith(".")
    ]


def load_records(list_dir: Path, family: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item_family, path in discover_case_lists(list_dir, family):
        for line_number, raw in enumerate(path.read_text().splitlines(), 1):
            if raw.startswith(f"[{item_family.upper()}]"):
                records.append(parse_record(path, line_number, raw, item_family))
    for index, record in enumerate(records):
        record["index"] = index
        shape = (
            record["family"], record["H"], record["Hg"], record["cu"],
            record["lower"],
        )
        record["shape_id"] = hashlib.sha256(repr(shape).encode()).hexdigest()[:16]
    return records


def _seed(record: dict[str, Any], base_seed: int) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{record['shape_id']}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _make_gdn_perf_gating(T: int, H: int, seed: int):
    """Build the official FlashQLA synthetic prefill Gate distribution.

    The returned ``base_g`` still contains the slow-decay values for every
    head.  Activated callers zero the non-SWA heads directly; raw-gating
    callers keep those values in ``a`` and use ``A_log=-inf`` so SGLang's
    fused gating produces the same zero-decay heads.
    """
    gen = _generator(seed)
    base_g = F.logsigmoid(
        torch.randn(
            1, T, H, dtype=torch.float32, device="cuda", generator=gen,
        )
    ) / GDN_GATE_DECAY_DIVISOR
    beta_logits = torch.randn(
        1, T, H, dtype=torch.float32, device="cuda", generator=gen,
    )
    swa_mask = torch.zeros(H, dtype=torch.bool, device="cuda")
    swa_heads = math.ceil(GDN_SWA_RATIO * H)
    swa_mask[:swa_heads] = True
    swa_mask = swa_mask[torch.randperm(H, device="cuda", generator=gen)]
    return base_g, beta_logits, swa_mask


def make_gdn_inputs(record: dict[str, Any], accuracy: bool, seed: int):
    T, H, Hg = record["T"], record["H"], record["Hg"]
    if not accuracy:
        # Performance inputs must be finite and deterministic.  ``torch.empty``
        # can expose allocator-dependent NaNs or denormals, which may select a
        # different gate path or produce a different arithmetic timing mode in
        # otherwise identical baseline/candidate processes.
        q = torch.full(
            (1, T, Hg, D), 1.0 / math.sqrt(D),
            dtype=torch.bfloat16, device="cuda",
        )
        k = torch.full_like(q, 1.0 / math.sqrt(D))
        v = torch.full(
            (1, T, H, D), 0.01,
            dtype=torch.bfloat16, device="cuda",
        )
        g, beta_logits, swa_mask = _make_gdn_perf_gating(T, H, seed)
        g[:, :, ~swa_mask] = 0.0
        beta = beta_logits.sigmoid()
    else:
        gen = _generator(seed)
        q = F.normalize(
            torch.randn(1, T, Hg, D, device="cuda", generator=gen), dim=-1
        ).to(torch.bfloat16)
        k = F.normalize(
            torch.randn(1, T, Hg, D, device="cuda", generator=gen), dim=-1
        ).to(torch.bfloat16)
        v = (torch.randn(1, T, H, D, device="cuda", generator=gen) * 0.1).to(
            torch.bfloat16
        )
        g = F.logsigmoid(
            torch.randn(1, T, H, device="cuda", generator=gen)
        )
        beta = torch.sigmoid(
            torch.randn(1, T, H, device="cuda", generator=gen)
        )
    cu = torch.tensor(record["cu"], dtype=torch.int32, device="cuda")
    return q.contiguous(), k.contiguous(), v, g.contiguous(), beta.contiguous(), cu


def make_gdn_optimized_inputs(
    record: dict[str, Any], accuracy: bool, seed: int,
) -> dict[str, Any]:
    """Build the post-convolution GDN ABI used by the serving adapters."""
    T, H, Hg = record["T"], record["H"], record["Hg"]
    width = (2 * Hg + H) * D
    if not accuracy:
        packed_qkv = torch.full(
            (T, width), 0.01, dtype=torch.bfloat16, device="cuda",
        )
        base_g, beta_logits, swa_mask = _make_gdn_perf_gating(T, H, seed)
        # With A=1 and dt_bias=0, inverse-softplus(-g) makes the serving
        # fused-gating path reproduce ``logsigmoid(randn) / 16``.  A_log=0
        # is then replaced with -inf for the 25% zero-decay heads.
        raw_gate_a = torch.log(torch.expm1(-base_g[0])).to(torch.bfloat16)
        raw_beta_b = beta_logits[0].to(torch.bfloat16)
        A_log = torch.zeros(H, dtype=torch.float32, device="cuda")
        A_log[~swa_mask] = -torch.inf
        dt_bias = torch.zeros(H, dtype=torch.float32, device="cuda")
        state_pool = torch.zeros(
            record["N"] + 4, H, D, D, dtype=torch.float32, device="cuda",
        )
    else:
        gen = _generator(seed)
        packed_qkv = (
            torch.randn(T, width, device="cuda", generator=gen) * 0.1
        ).to(torch.bfloat16)
        raw_ba = (
            torch.randn(T, 2 * H, device="cuda", generator=gen) * 0.1
        ).to(torch.bfloat16)
        A_log = torch.linspace(
            -0.4, 0.2, H, dtype=torch.float32, device="cuda",
        )
        dt_bias = torch.linspace(
            -0.2, 0.1, H, dtype=torch.float32, device="cuda",
        )
        state_pool = (
            torch.randn(
                record["N"] + 4, H, D, D, device="cuda", generator=gen,
                dtype=torch.float32,
            )
            * 0.01
        )
        raw_beta_b, raw_gate_a = raw_ba.split(H, dim=-1)
    return {
        "packed_qkv": packed_qkv,
        "raw_gate_a": raw_gate_a,
        "raw_beta_b": raw_beta_b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cu_seqlens": torch.tensor(
            record["cu"], dtype=torch.int32, device="cuda",
        ),
        "state_pool": state_pool,
        "state_indices": (
            torch.arange(
                record["N"] - 1, -1, -1,
                dtype=torch.int32, device="cuda",
            )
            + 2
        ),
    }


def _load_gdn_split_helper() -> Callable[..., Any]:
    """Load the PLA-independent fused split shared with the OPT-07 A/B."""
    here = (
        Path(__file__).resolve().parent
        / "flashqla"
        / "private"
        / "packed_qkv_split.py"
    )
    spec = importlib.util.spec_from_file_location("pla_generalized_gdn_split", here)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load GDN split helper from {here}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_qkv


def load_gdn_optimized_ops(
    path: str,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any] | None]:
    """Resolve the explicit split/packed GDN ABI without affecting legacy."""
    from pla.prefill import flashqla

    split_op = _load_gdn_split_helper()
    legacy_op = flashqla.chunk_gated_delta_rule_fwd
    packed_op = getattr(flashqla, "gdn_prefill_packed_qkv_fwd", None)
    if path == "packed" and packed_op is None:
        raise RuntimeError(
            "--gdn-path packed requires a PLA wheel exposing "
            "gdn_prefill_packed_qkv_fwd"
        )
    return split_op, legacy_op, packed_op


def run_gdn_optimized_path(
    path: str,
    inputs: dict[str, Any],
    record: dict[str, Any],
    auto_cp: bool,
    ops: tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any] | None],
):
    """Run one explicit serving-style GDN ABI path."""
    split_op, legacy_op, packed_op = ops
    common = {
        "scale": 1.0 / math.sqrt(D),
        "initial_state": inputs["state_pool"],
        "initial_state_indices": inputs["state_indices"],
        "output_final_state": False,
        "auto_cp": auto_cp,
        "raw_cu_seqlens": tuple(record["cu"]),
    }
    if path == "split":
        packed_qkv = inputs["packed_qkv"]
        logical_width = (2 * record["Hg"] + record["H"]) * D
        if logical_width <= MAX_SGLANG_FUSED_QKV_SPLIT_DIM:
            q, k, v = split_op(
                packed_qkv, record["H"], record["Hg"],
            )
        else:
            T, H, Hg = record["T"], record["H"], record["Hg"]
            q, k, v = torch.split(
                packed_qkv, [Hg * D, Hg * D, H * D], dim=-1,
            )
            q = q.view(1, T, Hg, D)
            k = k.view(1, T, Hg, D)
            v = v.view(1, T, H, D)
        return legacy_op(
            q,
            k,
            v,
            None,
            None,
            cu_seqlens=inputs["cu_seqlens"],
            normalize_qk=True,
            raw_gate_a=inputs["raw_gate_a"],
            raw_beta_b=inputs["raw_beta_b"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            **common,
        )
    if path != "packed":
        raise ValueError(f"optimized GDN path must be split or packed, got {path!r}")
    if packed_op is None:
        raise RuntimeError("packed GDN API is unavailable")
    return packed_op(
        inputs["packed_qkv"],
        inputs["raw_gate_a"],
        inputs["raw_beta_b"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["cu_seqlens"],
        num_qk_heads=record["Hg"],
        num_v_heads=record["H"],
        **common,
    )


def make_kda_inputs(
    record: dict[str, Any], accuracy: bool, seed: int,
    *, strided_beta: bool = False,
):
    T, H = record["T"], record["H"]
    if not accuracy:
        q = torch.full(
            (T, H, D), 0.01, dtype=torch.bfloat16, device="cuda",
        )
        k = torch.full_like(q, 0.01)
        v = torch.full_like(q, 0.01)
        g = torch.full_like(q, 0.01)
        if strided_beta:
            beta_storage = torch.full(
                (T, H + 3), 0.5, dtype=torch.bfloat16, device="cuda",
            )
            beta = beta_storage[:, :H]
        else:
            beta = torch.full(
                (T, H), 0.5, dtype=torch.bfloat16, device="cuda",
            )
        A_log = torch.zeros(H, dtype=torch.float32, device="cuda")
        dt_bias = torch.zeros(H, D, dtype=torch.float32, device="cuda")
    else:
        gen = _generator(seed)
        make = lambda *shape: (
            torch.randn(*shape, device="cuda", generator=gen) * 0.1
        ).to(torch.bfloat16)
        q, k, v, g = (make(T, H, D) for _ in range(4))
        beta = make(T, H + 3)[:, :H] if strided_beta else make(T, H)
        A_log = torch.rand(H, dtype=torch.float32, device="cuda", generator=gen)
        dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda", generator=gen)
    # Both SGLang and vLLM serving metadata use int32 offsets.  Keeping replay
    # on the same dtype avoids measuring a compatibility-only narrowing copy.
    cu = torch.tensor(record["cu"], dtype=torch.int32, device="cuda")
    return q, k, v, g, beta, A_log, dt_bias, cu


def make_kda_indexed_state(
    record: dict[str, Any], accuracy: bool, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fp32 indexed state-pool ABI used by SGLang serving."""
    N, H = record["N"], record["H"]
    slots = N + 4
    indices = (
        torch.arange(N - 1, -1, -1, dtype=torch.int32, device="cuda") + 2
    )
    if accuracy:
        gen = _generator(seed ^ 0x4B44_4153)
        # KDA carries recurrence in bf16 even when the serving pool is fp32.
        # Seed the fp32 buffer with bf16-exact values so the independent FLA
        # oracle starts from the same effective state seen by the PPU kernel.
        pool = (
            torch.randn(
                slots, H, D, D, dtype=torch.bfloat16, device="cuda",
                generator=gen,
            )
            * 0.01
        ).float()
    else:
        pool = torch.zeros(
            slots, H, D, D, dtype=torch.float32, device="cuda",
        )
    return pool, indices


def load_kda() -> tuple[Callable[..., Any], bool, bool, bool]:
    try:
        from pla.prefill.flashkdapro import flashkda_fwd
        try:
            parameters = inspect.signature(flashkda_fwd).parameters.values()
            trusted_offsets = any(
                parameter.name == "raw_cu_seqlens"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            parameters = inspect.signature(flashkda_fwd).parameters.values()
            indexed_state = any(
                parameter.name == "state_pool"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            trusted_offsets = False
            indexed_state = False
        return flashkda_fwd, True, trusted_offsets, indexed_state
    except ImportError:
        from pla.prefill.flashkda import flashkda_fwd
        return flashkda_fwd, False, False, False


def run_kda_candidate(
    op: Callable[..., Any], out_params: bool, inputs: tuple[Any, ...],
    record: dict[str, Any], exact_b1_cu: bool,
    trusted_host_offsets: bool = False,
    *,
    state_pool: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    final: torch.Tensor | None = None,
):
    q, k, v, g, beta, A_log, dt_bias, cu = inputs
    cu_arg = (
        cu
        if record["N"] > 1 or exact_b1_cu or trusted_host_offsets
        else None
    )
    scale = 1.0 / math.sqrt(D)
    if not out_params:
        if trusted_host_offsets:
            raise RuntimeError(
                "--kda-trusted-host-offsets requires the flashkdapro out-param API"
            )
        return op(
            q, k, v, g, beta, A_log, dt_bias, scale, record["lower"], None,
            cu_arg,
        )
    if out is None:
        out = torch.empty_like(v)
    trusted_args = {}
    if trusted_host_offsets:
        trusted_args["raw_cu_seqlens"] = tuple(record["cu"])
    if state_pool is not None:
        if state_indices is None:
            raise ValueError("indexed KDA state_pool requires state_indices")
        if final is not None:
            raise ValueError("indexed KDA state_pool is mutually exclusive with final")
        op(
            q, k, v, g, beta, A_log, dt_bias, out, scale, record["lower"],
            None, None, cu_arg,
            state_pool=state_pool, state_indices=state_indices,
            **trusted_args,
        )
        return out, state_pool
    if state_indices is not None:
        raise ValueError("state_indices requires state_pool")
    if final is None:
        final = torch.empty(
            record["N"], record["H"], D, D,
            dtype=torch.bfloat16, device="cuda",
        )
    op(
        q, k, v, g, beta, A_log, dt_bias, out, scale, record["lower"], None,
        final, cu_arg, **trusted_args,
    )
    return out, final


def _load_kda_fla_helper() -> Callable[..., Any]:
    here = Path(__file__).resolve().parent / "flashkdapro" / "bench_vs_fla.py"
    if not here.exists():
        here = Path(__file__).resolve().parent / "flashkda" / "bench_vs_fla.py"
    private = here.parent / "private"
    sys.path.insert(0, str(private))
    spec = importlib.util.spec_from_file_location("pla_generalized_kda_fla", here)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load KDA FLA helper from {here}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_fla_chunk_kda


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    denominator = torch.linalg.vector_norm(expected).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def _max_group_relative_error(
    actual: torch.Tensor, expected: torch.Tensor, reduce_dims: tuple[int, ...],
) -> float:
    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    numerator = torch.sqrt(torch.sum(difference * difference, dim=reduce_dims))
    denominator = torch.sqrt(torch.sum(expected * expected, dim=reduce_dims))
    return float((numerator / denominator.clamp_min(1e-12)).max().item())


def accuracy_metrics(
    out: torch.Tensor,
    reference_out: torch.Tensor,
    state: torch.Tensor,
    reference_state: torch.Tensor,
) -> dict[str, float]:
    """Return global and locality-preserving metrics for an accuracy row."""
    out_reduce = tuple(index for index in range(out.ndim) if index != out.ndim - 2)
    out_difference = out.float() - reference_out.float()
    state_difference = state.float() - reference_state.float()
    return {
        "out_rel": relative_error(out, reference_out),
        "state_rel": relative_error(state, reference_state),
        "out_head_rel": _max_group_relative_error(
            out, reference_out, out_reduce,
        ),
        "state_seq_head_rel": _max_group_relative_error(
            state, reference_state, (-2, -1),
        ),
        "out_max_abs": float(out_difference.abs().max().item()),
        "state_max_abs": float(state_difference.abs().max().item()),
    }


def accuracy_gdn(
    record: dict[str, Any],
    seed: int,
    path: str = "legacy",
    ops: tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any] | None]
    | None = None,
) -> dict[str, Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    if path == "legacy":
        from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

        q, k, v, g, beta, cu = make_gdn_inputs(record, True, seed)
        candidate = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, cu_seqlens=cu, output_final_state=True,
        )
        out, state = candidate[2], candidate[4]
        del candidate
        initial_state = None
    else:
        if ops is None:
            ops = load_gdn_optimized_ops(path)
        inputs = make_gdn_optimized_inputs(record, True, seed)
        # Preserve the exact pre-call state for the independent oracle before
        # PLA updates its indexed K-last pool in place.
        pool_before = inputs["state_pool"].clone()
        selected = inputs["state_indices"].to(torch.int64)
        initial_state = inputs["state_pool"][selected].clone()
        candidate = run_gdn_optimized_path(
            path, inputs, record, auto_cp=True, ops=ops,
        )
        out, state = candidate[2], inputs["state_pool"][selected]
        q, k, v = ops[0](
            inputs["packed_qkv"], record["H"], record["Hg"],
        )
        q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
        g = (
            -inputs["A_log"].exp()[None, :]
            * F.softplus(
                inputs["raw_gate_a"].float()
                + inputs["dt_bias"][None, :]
            )
        ).unsqueeze(0).contiguous()
        beta = torch.sigmoid(inputs["raw_beta_b"].float()).unsqueeze(0).contiguous()
        cu = inputs["cu_seqlens"]

    factor = record["H"] // record["Hg"]
    ref_out, ref_state = chunk_gated_delta_rule(
        q=q.repeat_interleave(factor, dim=2).contiguous(),
        k=k.repeat_interleave(factor, dim=2).contiguous(),
        v=v, g=g, beta=beta, scale=D ** -0.5,
        initial_state=(
            initial_state.transpose(-1, -2).contiguous()
            if initial_state is not None
            else None
        ),
        output_final_state=True, use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu.to(torch.int64), chunk_size=64,
    )
    torch.cuda.synchronize()
    if path != "legacy":
        ref_state = ref_state.transpose(-1, -2).contiguous()
        touched = torch.zeros(
            inputs["state_pool"].shape[0],
            dtype=torch.bool,
            device=inputs["state_pool"].device,
        )
        touched[selected] = True
        unmapped_equal = bool(torch.equal(
            inputs["state_pool"][~touched], pool_before[~touched],
        ))
    else:
        unmapped_equal = True
    metrics = accuracy_metrics(out, ref_out, state, ref_state)
    metrics["unmapped_state_unchanged"] = unmapped_equal
    return metrics


def accuracy_gdn_serial(
    record: dict[str, Any],
    seed: int,
    path: str = "legacy",
    ops: tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any] | None]
    | None = None,
) -> dict[str, Any]:
    if path == "legacy":
        from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

        q, k, v, g, beta, cu = make_gdn_inputs(record, True, seed)
        candidate = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, cu_seqlens=cu,
            output_final_state=True, auto_cp=True,
        )
        reference = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, cu_seqlens=cu,
            output_final_state=True, auto_cp=False,
        )
        candidate_state = candidate[4]
        reference_state = reference[4]
    else:
        if ops is None:
            ops = load_gdn_optimized_ops(path)
        candidate_inputs = make_gdn_optimized_inputs(record, True, seed)
        reference_inputs = dict(candidate_inputs)
        reference_inputs["state_pool"] = candidate_inputs["state_pool"].clone()
        pool_before = candidate_inputs["state_pool"].clone()

        # The packed accuracy contract deliberately compares the production
        # automatic-CP path against the established fused-split serial path.
        # Split mode uses the same serial reference and is useful for isolating
        # CP changes from the packed-QKV adapter itself.
        candidate = run_gdn_optimized_path(
            path, candidate_inputs, record, auto_cp=True, ops=ops,
        )
        reference = run_gdn_optimized_path(
            "split", reference_inputs, record, auto_cp=False, ops=ops,
        )
        selected = candidate_inputs["state_indices"].to(torch.int64)
        candidate_state = candidate_inputs["state_pool"][selected]
        reference_state = reference_inputs["state_pool"][selected]
    torch.cuda.synchronize()
    metrics = accuracy_metrics(
        candidate[2], reference[2], candidate_state, reference_state,
    )
    if path != "legacy":
        touched = torch.zeros(
            candidate_inputs["state_pool"].shape[0],
            dtype=torch.bool,
            device=candidate_inputs["state_pool"].device,
        )
        touched[selected] = True
        metrics["unmapped_state_unchanged"] = bool(
            torch.equal(
                candidate_inputs["state_pool"][~touched], pool_before[~touched],
            )
            and torch.equal(
                reference_inputs["state_pool"][~touched], pool_before[~touched],
            )
        )
    else:
        metrics["unmapped_state_unchanged"] = True
    return metrics


def accuracy_kda(
    record: dict[str, Any], seed: int, exact_b1_cu: bool,
    kda_op: Callable[..., Any], out_params: bool,
    fla_op: Callable[..., Any], trusted_host_offsets: bool = False,
    indexed_state: bool = False,
) -> dict[str, Any]:
    inputs = make_kda_inputs(
        record, True, seed, strided_beta=indexed_state,
    )
    state_pool = state_indices = pool_before = initial_state = None
    if indexed_state:
        state_pool, state_indices = make_kda_indexed_state(record, True, seed)
        pool_before = state_pool.clone()
        initial_state = state_pool[state_indices.to(torch.int64)].clone()
    out, state = run_kda_candidate(
        kda_op, out_params, inputs, record, exact_b1_cu,
        trusted_host_offsets, state_pool=state_pool, state_indices=state_indices,
    )
    q, k, v, g, beta, A_log, dt_bias, cu = inputs
    cu_arg = (
        cu
        if record["N"] > 1 or exact_b1_cu or trusted_host_offsets
        else None
    )
    ref_out, ref_state = fla_op(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0),
        beta.contiguous().unsqueeze(0), A_log, dt_bias, D ** -0.5,
        initial_state,
        extra={"cu_seqlens": cu_arg} if cu_arg is not None else {},
    )
    torch.cuda.synchronize()
    if indexed_state:
        selected = state_indices.to(torch.int64)
        state = state[selected]
        touched = torch.zeros(
            state_pool.shape[0], dtype=torch.bool, device=state_pool.device,
        )
        touched[selected] = True
        unmapped_equal = bool(torch.equal(state_pool[~touched], pool_before[~touched]))
    else:
        unmapped_equal = True
    metrics = accuracy_metrics(out, ref_out[0], state, ref_state)
    metrics["unmapped_state_unchanged"] = unmapped_equal
    return metrics


def accuracy_kda_serial(
    record: dict[str, Any], seed: int, exact_b1_cu: bool,
    kda_op: Callable[..., Any], out_params: bool,
    trusted_host_offsets: bool = False,
    indexed_state: bool = False,
) -> dict[str, Any]:
    inputs = make_kda_inputs(
        record, True, seed, strided_beta=indexed_state,
    )
    candidate_pool = reference_pool = state_indices = pool_before = None
    if indexed_state:
        candidate_pool, state_indices = make_kda_indexed_state(record, True, seed)
        reference_pool = candidate_pool.clone()
        pool_before = candidate_pool.clone()
    keys = ("PLA_FLASHKDA_CP", "PLA_FLASHKDA_K2_NOWS")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        # Candidate is the unmodified production dispatch.
        for key in keys:
            os.environ.pop(key, None)
        out, state = run_kda_candidate(
            kda_op, out_params, inputs, record, exact_b1_cu,
            trusted_host_offsets,
            state_pool=candidate_pool, state_indices=state_indices,
        )

        # The reference disables both inter-CU sequence splitting and the
        # register-resident recurrence.  It therefore exercises the established
        # serial K1 + shared-memory K2 path on exactly the same list input.
        os.environ["PLA_FLASHKDA_CP"] = "0"
        os.environ["PLA_FLASHKDA_K2_NOWS"] = "0"
        ref_out, ref_state = run_kda_candidate(
            kda_op, out_params, inputs, record, exact_b1_cu,
            trusted_host_offsets,
            state_pool=reference_pool, state_indices=state_indices,
        )
        torch.cuda.synchronize()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if indexed_state:
        selected = state_indices.to(torch.int64)
        state = state[selected]
        ref_state = ref_state[selected]
        touched = torch.zeros(
            candidate_pool.shape[0],
            dtype=torch.bool,
            device=candidate_pool.device,
        )
        touched[selected] = True
        unmapped_equal = bool(
            torch.equal(candidate_pool[~touched], pool_before[~touched])
            and torch.equal(reference_pool[~touched], pool_before[~touched])
        )
    else:
        unmapped_equal = True
    metrics = accuracy_metrics(out, ref_out, state, ref_state)
    metrics["unmapped_state_unchanged"] = unmapped_equal
    return metrics


def _event_ms(call: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _wall_ms(call: Callable[[], Any]) -> float:
    """Measure a prepared-input API call, including synchronous Host work."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1_000.0


def median_ms(
    call: Callable[[], Any], warmup: int, repeat: int,
    *, record_first_call: bool = False,
) -> tuple[float, float | None]:
    first_call_ms = _wall_ms(call) if record_first_call else None
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        samples.append(_event_ms(call))
    return statistics.median(samples), first_call_ms


def perf_gdn(
    record: dict[str, Any], seed: int, warmup: int, repeat: int,
    auto_cp: bool, path: str = "legacy",
    ops: tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any] | None]
    | None = None,
    record_first_call: bool = False,
) -> tuple[float, float | None]:
    if path == "legacy":
        from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

        inputs = make_gdn_inputs(record, False, seed)
        q, k, v, g, beta, cu = inputs
        call = lambda: chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, cu_seqlens=cu, auto_cp=auto_cp,
        )
    else:
        if ops is None:
            ops = load_gdn_optimized_ops(path)
        optimized_inputs = make_gdn_optimized_inputs(record, False, seed)
        call = lambda: run_gdn_optimized_path(
            path, optimized_inputs, record, auto_cp=auto_cp, ops=ops,
        )
    return median_ms(
        call,
        warmup, repeat, record_first_call=record_first_call,
    )


def perf_kda(
    record: dict[str, Any], seed: int, warmup: int, repeat: int,
    exact_b1_cu: bool, kda_op: Callable[..., Any], out_params: bool,
    trusted_host_offsets: bool = False,
    indexed_state: bool = False,
    record_first_call: bool = False,
) -> tuple[float, float | None]:
    inputs = make_kda_inputs(
        record, False, seed, strided_beta=indexed_state,
    )
    state_pool = state_indices = final = None
    if indexed_state:
        state_pool, state_indices = make_kda_indexed_state(record, False, seed)
    elif out_params:
        final = torch.empty(
            record["N"], record["H"], D, D,
            dtype=torch.bfloat16, device="cuda",
        )
    out = torch.empty_like(inputs[2]) if out_params else None
    return median_ms(
        lambda: run_kda_candidate(
            kda_op, out_params, inputs, record, exact_b1_cu,
            trusted_host_offsets,
            state_pool=state_pool, state_indices=state_indices,
            out=out, final=final,
        ),
        warmup, repeat, record_first_call=record_first_call,
    )


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in (
        "index", "family", "source", "line", "shape_id", "H", "Hg", "N", "T"
    )}


def list_manifest_sha256(list_dir: Path) -> str | None:
    manifest = list_dir / "SHA256SUMS"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def extension_identity(family: str) -> dict[str, str]:
    module_name, attribute = (
        ("pla.prefill.flashqla.ops", "flashqla_cute")
        if family == "gdn"
        else ("pla.prefill.flashkdapro.ops", "flashkda_cute")
    )
    module = importlib.import_module(module_name)
    extension = getattr(module, attribute)
    extension_path = Path(extension.__file__).resolve()
    return {
        "pla_ops_path": str(Path(module.__file__).resolve()),
        "pla_extension_path": str(extension_path),
        "pla_extension_sha256": hashlib.sha256(
            extension_path.read_bytes()
        ).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("all", "gdn", "kda"), default="all")
    parser.add_argument("--mode", choices=("accuracy", "perf"), required=True)
    parser.add_argument(
        "--reference", choices=("serial", "fla"), default="serial",
        help="accuracy oracle: production-vs-base path (full sweep) or FLA Triton",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--index-file", type=Path,
        help="optional newline-delimited integer or JSONL rows containing indices",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--record-first-call", action="store_true",
        help=(
            "also record the first per-record call before warmup; for KDA this "
            "captures the sequential LRU cold/miss-or-hit workload in addition "
            "to the steady-state hot median"
        ),
    )
    parser.add_argument("--expected-list-files", type=int)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expect-list-manifest-sha256")
    parser.add_argument(
        "--gdn-auto-cp", action=argparse.BooleanOptionalAction, default=True,
        help="enable production automatic context parallelism in GDN perf mode",
    )
    parser.add_argument(
        "--gdn-path", choices=("legacy", "split", "packed"), default="legacy",
        help=(
            "GDN ABI: activated q/k/v/g/beta (legacy), fused Triton QKV split "
            "+ raw gating (split), or direct post-convolution Q|K|V (packed)"
        ),
    )
    parser.add_argument(
        "--kda-exact-b1-cu", action="store_true",
        help="pass the list's [0,T] cu_seqlens for B=1 instead of the optimized fixed path",
    )
    parser.add_argument(
        "--kda-trusted-host-offsets", action="store_true",
        help=(
            "pass the record's Host offsets as raw_cu_seqlens to the new "
            "flashkdapro API (also supplies int32 cu_seqlens for B=1)"
        ),
    )
    parser.add_argument(
        "--kda-indexed-state", action="store_true",
        help=(
            "use the SGLang serving ABI: fp32 state_pool + int32 state_indices, "
            "no compact final_state, and a raw token-strided beta view"
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    return args


def selected_abi_metadata(
    args: argparse.Namespace,
    record: dict[str, Any],
    out_params: bool | None,
) -> dict[str, Any]:
    """Describe the effective ABI in every JSONL row for auditable A/Bs."""
    if record["family"] == "gdn":
        if args.mode == "accuracy":
            reference_path = (
                "fla" if args.reference == "fla"
                else "split" if args.gdn_path != "legacy"
                else "legacy_serial"
            )
            auto_cp = True
        else:
            reference_path = None
            auto_cp = args.gdn_auto_cp
        return {
            "gdn_path": args.gdn_path,
            "gdn_reference_path": reference_path,
            "gdn_auto_cp": auto_cp,
            "gdn_gating_abi": (
                "activated" if args.gdn_path == "legacy" else "raw"
            ),
            "gdn_state_abi": (
                "legacy" if args.gdn_path == "legacy" else "indexed_pool"
            ),
            "gdn_gate_distribution": (
                "flashqla_slow_decay_swa75"
                if args.mode == "perf" else "accuracy_stress"
            ),
            "gdn_gate_decay_divisor": (
                GDN_GATE_DECAY_DIVISOR if args.mode == "perf" else None
            ),
            "gdn_swa_ratio": GDN_SWA_RATIO if args.mode == "perf" else None,
        }
    return {
        "kda_api": (
            "flashkdapro_out_params" if out_params else "flashkda_legacy_return"
        ),
        "kda_cu_dtype": "int32",
        "kda_exact_b1_cu": args.kda_exact_b1_cu,
        "kda_trusted_host_offsets": args.kda_trusted_host_offsets,
        "kda_state_abi": (
            "indexed_fp32_pool" if getattr(args, "kda_indexed_state", False)
            else "compact_bf16_final_state"
        ),
        "kda_beta_abi": (
            "raw_strided"
            if getattr(args, "kda_indexed_state", False)
            else "raw_contiguous"
        ),
        "kda_cp_plan_cache_size_env": os.environ.get(
            "PLA_FLASHKDA_CP_PLAN_CACHE_SIZE", "default",
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    inventory_files = discover_case_lists(args.list_dir, args.family)
    records = load_records(args.list_dir, args.family)
    manifest_sha256 = list_manifest_sha256(args.list_dir)
    if (
        args.expected_list_files is not None
        and len(inventory_files) != args.expected_list_files
    ):
        raise RuntimeError(
            f"case-list file count mismatch: expected "
            f"{args.expected_list_files}, got {len(inventory_files)}"
        )
    if args.expected_records is not None and len(records) != args.expected_records:
        raise RuntimeError(
            f"case-list record count mismatch: expected {args.expected_records}, "
            f"got {len(records)}"
        )
    if (
        args.expect_list_manifest_sha256 is not None
        and manifest_sha256 != args.expect_list_manifest_sha256
    ):
        raise RuntimeError(
            "case-list manifest hash mismatch: expected "
            f"{args.expect_list_manifest_sha256}, got {manifest_sha256}"
        )
    wanted = None
    if args.index_file:
        wanted = set()
        for raw in args.index_file.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            wanted.add(int(json.loads(raw)["index"] if raw.startswith("{") else raw))
    # Shard the filtered sequence rather than the global indices.  A targeted
    # rerun often contains offenders from one original shard (all indices share
    # the same modulo); applying the original modulo first would put every
    # selected record back on one device and leave the other cards idle.
    if wanted is None:
        # Preserve the full-inventory sharding contract even if an input list
        # later stops being position-indexed or contains gaps.
        selected = [
            record for record in records
            if record["index"] % args.num_shards == args.shard_index
        ]
    else:
        filtered = [record for record in records if record["index"] in wanted]
        selected = [
            record for position, record in enumerate(filtered)
            if position % args.num_shards == args.shard_index
        ]
    print(
        f"inventory_files={len(inventory_files)} loaded={len(records)} "
        f"selected={len(selected)} mode={args.mode} "
        f"gdn_path={args.gdn_path} "
        f"kda_trusted_host_offsets={args.kda_trusted_host_offsets} "
        f"kda_indexed_state={args.kda_indexed_state} "
        f"shard={args.shard_index}/{args.num_shards}", flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selection_sha256 = hashlib.sha256(
        ",".join(str(record["index"]) for record in selected).encode()
    ).hexdigest()

    gdn_ops = None
    if (
        args.gdn_path != "legacy"
        and any(record["family"] == "gdn" for record in selected)
    ):
        gdn_ops = load_gdn_optimized_ops(args.gdn_path)

    kda_op = out_params = kda_trusted_supported = kda_indexed_supported = None
    fla_kda = None
    if any(record["family"] == "kda" for record in selected):
        (
            kda_op,
            out_params,
            kda_trusted_supported,
            kda_indexed_supported,
        ) = load_kda()
        if args.kda_trusted_host_offsets and not kda_trusted_supported:
            raise RuntimeError(
                "--kda-trusted-host-offsets requires a PLA wheel whose "
                "flashkdapro API exposes raw_cu_seqlens"
            )
        if args.kda_indexed_state and not kda_indexed_supported:
            raise RuntimeError(
                "--kda-indexed-state requires a PLA wheel whose flashkdapro "
                "API exposes state_pool/state_indices"
            )
        if args.mode == "accuracy" and args.reference == "fla":
            fla_kda = _load_kda_fla_helper()

    identities = {
        family: extension_identity(family)
        for family in ("gdn", "kda")
        if any(record["family"] == family for record in selected)
    }

    passed = failed = 0
    started = time.time()
    with args.output.open("w") as output:
        for local_index, record in enumerate(selected, 1):
            row = public_record(record)
            row.update({
                "mode": args.mode,
                "reference": args.reference if args.mode == "accuracy" else None,
                "shard": args.shard_index,
                "inventory_file_count": len(inventory_files),
                "inventory_record_count": len(records),
                "list_manifest_sha256": manifest_sha256,
                "seed": args.seed,
                "num_shards": args.num_shards,
                "selection_sha256": selection_sha256,
            })
            if args.mode == "perf":
                row.update({
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "record_first_call": args.record_first_call,
                    "timing_method": "cuda_event_median_after_per_case_warmup",
                    "first_call_timing_method": (
                        "wall_clock_synchronize_before_after_prepared_inputs"
                    ),
                })
            row.update(selected_abi_metadata(args, record, out_params))
            row.update(identities[record["family"]])
            try:
                seed = _seed(record, args.seed)
                if args.mode == "accuracy":
                    if record["family"] == "gdn" and args.reference == "fla":
                        metrics = accuracy_gdn(
                            record, seed, args.gdn_path, gdn_ops,
                        )
                    elif record["family"] == "gdn":
                        metrics = accuracy_gdn_serial(
                            record, seed, args.gdn_path, gdn_ops,
                        )
                    elif args.reference == "fla":
                        metrics = accuracy_kda(
                            record, seed, args.kda_exact_b1_cu,
                            kda_op, bool(out_params), fla_kda,
                            args.kda_trusted_host_offsets,
                            args.kda_indexed_state,
                        )
                    else:
                        metrics = accuracy_kda_serial(
                            record, seed, args.kda_exact_b1_cu,
                            kda_op, bool(out_params),
                            args.kda_trusted_host_offsets,
                            args.kda_indexed_state,
                        )
                    relative_metrics = (
                        "out_rel", "state_rel", "out_head_rel",
                        "state_seq_head_rel",
                    )
                    absolute_metrics = ("out_max_abs", "state_max_abs")
                    ok = all(
                        math.isfinite(metrics[name])
                        and metrics[name] < args.tol
                        for name in relative_metrics
                    ) and all(
                        math.isfinite(metrics[name]) and metrics[name] >= 0.0
                        for name in absolute_metrics
                    ) and metrics.get("unmapped_state_unchanged", True) is True
                    row.update(metrics)
                    row.update({"tol": args.tol, "pass": ok})
                elif record["family"] == "gdn":
                    ms, first_call_ms = perf_gdn(
                        record, seed, args.warmup, args.repeat,
                        args.gdn_auto_cp, args.gdn_path, gdn_ops,
                        args.record_first_call,
                    )
                    row.update({
                        "ms": ms,
                        "pass": True,
                    })
                    if first_call_ms is not None:
                        row["first_call_ms"] = first_call_ms
                    ok = True
                else:
                    ms, first_call_ms = perf_kda(
                        record, seed, args.warmup, args.repeat,
                        args.kda_exact_b1_cu, kda_op, bool(out_params),
                        args.kda_trusted_host_offsets,
                        args.kda_indexed_state,
                        args.record_first_call,
                    )
                    row.update({
                        "ms": ms,
                        "pass": True,
                    })
                    if first_call_ms is not None:
                        row["first_call_ms"] = first_call_ms
                    ok = True
                passed += int(ok)
                failed += int(not ok)
            except Exception as exc:  # keep a full sweep auditable
                row.update({
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                failed += 1
                torch.cuda.empty_cache()
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            if local_index % 10 == 0 or local_index == len(selected):
                elapsed = time.time() - started
                rate = local_index / max(elapsed, 1e-9)
                eta = (len(selected) - local_index) / max(rate, 1e-9)
                print(
                    f"[{local_index}/{len(selected)}] pass={passed} fail={failed} "
                    f"rate={rate:.3f}/s eta={eta / 60:.1f}m",
                    flush=True,
                )

    print(f"DONE pass={passed} fail={failed} output={args.output}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
