# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Public vLLM replay contract tests, independent of a vLLM installation.

Run as a script. TP shard equivalence is not a distributed model evaluation.
The optional --distributed mode requires torchrun and two or more devices.
The explicit --phase=functional includes ordinary contracts and finite-gate
overflow checks. The default --phase=all also retains strict precision stress.
"""
import argparse
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from case_report import CaseReporter

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "kda_replay_oracle", ROOT / "scripts/decode/kda_mtp_sglang/private/_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
D = 128


def close(got, want, tol=6e-3):
    if got.numel() == 0:
        return
    result = reference.score(got, want, "replay")
    assert result["finite"] and result["max_rel"] <= tol, result


def vector(values, device, dtype=torch.int32, strided=False):
    t = torch.tensor(values, device=device, dtype=dtype)
    if strided:
        storage = torch.full((len(values) * 2 + 1,), -99, device=device, dtype=dtype)
        storage[1::2] = t
        t = storage[1::2]
    return t


def pages(slots, heads, q, layout, device, padding=256, history=None, conv_offset=0):
    c = 3 * heads * D
    history = q + 2 if history is None else history
    conv_shape = (c, history) if layout == "DS" else (history, c)
    conv_bytes = c * history * 2
    state_bytes = heads * D * D * 4
    state_offset = ((conv_offset + conv_bytes + 15) // 16) * 16
    page_bytes = ((state_offset + state_bytes + padding + 15) // 16) * 16
    raw = torch.full((slots * page_bytes,), 0x5A, device=device, dtype=torch.uint8)
    conv = raw.view(torch.bfloat16).as_strided(
        (slots, *conv_shape), (page_bytes // 2, *torch.empty(conv_shape).stride()),
        conv_offset // 2)
    state = raw.view(torch.float32).as_strided(
        (slots, heads, D, D), (page_bytes // 4, heads * D * D // heads, D, 1),
        state_offset // 4)
    conv.copy_(torch.randn(conv.shape, device=device, dtype=torch.bfloat16) * 0.2)
    state.copy_(torch.randn(state.shape, device=device) * 0.01)
    return SimpleNamespace(raw=raw, kv_cache=(conv, state), heads=heads, q=q,
                           layout=layout, page_bytes=page_bytes)


def ds(layer):
    c = layer.kv_cache[0]
    return c if layer.layout == "DS" else c.transpose(-1, -2)


def inputs(heads, capacity, device, offset=2):
    c = heads * D
    storage = torch.randn(offset + capacity + 2, 4 * c + heads + 3,
                          device=device, dtype=torch.bfloat16) * 0.4
    rows = storage[offset:offset + capacity]
    return dict(mixed_qkv=rows[:, :3 * c], raw_g=rows[:, 3 * c:4 * c],
                raw_beta=rows[:, 4 * c:4 * c + heads], storage=storage,
                conv_weight=torch.randn(3 * c, 4, device=device) * 0.2,
                A_log=torch.randn(heads, device=device) * 0.1,
                dt_bias=torch.randn(c, device=device) * 0.2, lower_bound=-5.0)


def blank_records(rows, heads, q, device):
    shape = (rows, heads, q, D)
    return SimpleNamespace(rawv=torch.full(shape, -7, dtype=torch.bfloat16, device=device),
                           rawk=torch.full(shape, -7, dtype=torch.bfloat16, device=device),
                           g=torch.full(shape, -7., device=device),
                           beta=torch.full(shape[:-1], -7., device=device),
                           conv=torch.full((rows, q, 3 * heads * D, 3), -7,
                                           dtype=torch.bfloat16, device=device))


def clone_records(records):
    return SimpleNamespace(**{k: getattr(records, k).clone()
                              for k in ("rawv", "rawk", "g", "beta", "conv")})


def oracle_verify(layer, x, offsets, slots, records, output, null_block_id=0):
    """Adapt the existing FP64 oracle per request, never use kernel records."""
    want = clone_records(records)
    out = output.clone().double()
    h, q, c = layer.heads, layer.q, layer.heads * D
    conv = ds(layer)
    for row, slot in enumerate(slots):
        begin, end = offsets[row:row + 2]
        length = end - begin
        if length == 0:
            continue
        if slot <= null_block_id:
            out[begin:end] = 0
            continue
        qkv = x["mixed_qkv"][begin:end].split(c, -1)
        vi = {"x_" + name: t.reshape(1, length, h, D)
              for name, t in zip(("q", "k", "v"), qkv)}
        vi.update(g=x["raw_g"][begin:end].reshape(1, length, h, D),
                  beta=x["raw_beta"][begin:end].unsqueeze(0),
                  A_log=x["A_log"], dt_bias=x["dt_bias"], scale=D ** -0.5,
                  lower_bound=x["lower_bound"],
                  recurrent_state=layer.kv_cache[1][slot:slot + 1],
                  cu_seqlens=torch.tensor([0, length]),
                  ssm_state_indices=torch.tensor([0]),
                  intermediate_state_indices=torch.tensor([0]))
        for name, weight, cache in zip(("q", "k", "v"),
                                       x["conv_weight"].split(c), conv[slot].split(c)):
            vi["w_" + name] = weight
            vi["cs_" + name] = cache.unsqueeze(0)[..., :3]
        cfg = SimpleNamespace(heads=h, T=length, bs=1, slots=1,
                              draft_token_num=q, ring_len=q)
        ref = reference.verify_reference(vi, cfg)
        out[begin:end] = ref["out"].reshape(length, c)
        for name in ("rawv", "rawk", "g", "beta"):
            getattr(want, name)[row, :, :length] = ref["ring_" + name][0, :, :length]
        want.conv[row, :length] = torch.cat(
            [ref["ic_" + name][0, :length] for name in ("q", "k", "v")], dim=1)
    return out, want


def folded(state, records, row, count):
    ci = dict(temporal=state.unsqueeze(0).unsqueeze(0),
              rawv=records.rawv[row:row + 1].unsqueeze(0),
              rawk=records.rawk[row:row + 1].unsqueeze(0),
              gk=records.g[row:row + 1].unsqueeze(0),
              beta=records.beta[row:row + 1].unsqueeze(0),
              state_batch_indices=torch.tensor([0]), accept_lens=torch.tensor([count]))
    return reference.fold_reference(ci, None)[0, 0].float()


def oracle_commit(layer, records, offsets, slots, counts, mapping=None,
                  block_table=None, computed=None, block_size=None, null_block_id=0):
    conv, state = ds(layer), layer.kv_cache[1]
    old = state.clone()
    expected_state, expected_conv = old.clone(), conv.clone()
    plan = [[], [], [], []]
    for b, slot in enumerate(slots):
        row = b if mapping is None else mapping[b]
        n = min(max(counts[row], 0), offsets[b + 1] - offsets[b], layer.q)
        if slot <= null_block_id:
            n = 0
        final, boundary, bc = slot, 0, 0
        if n and block_table is not None:
            p = computed[row]
            final = block_table[row][min(max(p + n - 1, 0) // block_size,
                                         len(block_table[row]) - 1)]
            next_boundary = (p // block_size + 1) * block_size
            if p + n >= next_boundary:
                bc = next_boundary - p
                boundary = block_table[row][next_boundary // block_size - 1]
        if n:
            if bc:
                expected_state[boundary] = folded(old[slot], records, b, bc)
                expected_conv[boundary, :, :3] = records.conv[b, bc - 1]
            expected_state[final] = folded(old[slot], records, b, n)
            expected_conv[final, :, :3] = records.conv[b, n - 1]
        plan[0].append(n)
        plan[1].append(final if n else 0)
        plan[2].append(boundary if n else 0)
        plan[3].append(bc if n else 0)
    return expected_state, expected_conv, plan


def invoke(layer, x, offsets, slots, out, null_block_id=0):
    from pla.decode.kda_replayssm_vllm import kda_replay_verify
    args = {k: v for k, v in x.items() if k != "storage"}
    return kda_replay_verify(**args, conv_state=layer.kv_cache[0],
                             checkpoint_state=layer.kv_cache[1],
                             records=layer.replay_records, query_start_loc=offsets,
                             state_indices=slots, conv_state_layout=layer.layout, out=out,
                             null_block_id=null_block_id)


def allocate(layer, rows):
    from pla.decode.kda_replayssm_vllm import KDAReplayRecords
    layer.replay_records = KDAReplayRecords.allocate(rows, layer.heads, layer.q,
                                                    layer.raw.device)
    for name in ("rawv", "rawk", "g", "beta", "conv"):
        getattr(layer.replay_records, name).fill_(-7)


def context(layers, rows, null_block_id=0):
    from pla.decode.kda_replayssm_vllm import KDAReplayCommitContext
    return KDAReplayCommitContext.create(layers, spec_query_len=layers[0].q,
                                        max_num_reqs=rows,
                                        conv_state_layout=layers[0].layout,
                                        null_block_id=null_block_id)


def check_plan(plan, expected, device):
    for name, values in zip(("commit_lens", "final_state_indices",
                             "boundary_state_indices", "boundary_recovery_lens"), expected):
        assert torch.equal(getattr(plan, name)[:len(values)],
                           torch.tensor(values, device=device, dtype=torch.int32)), name


def case_verify(device, layout, q, heads, index_dtype, rows=5, misaligned=False,
                null_block_id=0):
    torch.manual_seed(41 + q + heads)
    lengths = [q, max(q - 2, 0), 0, 1, 1] + [0] * (rows - 5)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    slots = [7, 2, 5, 0, -1] + [5] * (rows - 5)
    layer = pages(12, heads, q, layout, device)
    allocate(layer, len(slots))
    x = inputs(heads, len(slots) * q + 3, device)
    if misaligned:
        weight = torch.empty(x['conv_weight'].numel() + 1, device=device)
        shifted = weight[1:].view_as(x['conv_weight'])
        shifted.copy_(x['conv_weight'])
        x['conv_weight'] = shifted
        assert shifted.data_ptr() % 16 != 0
    out = torch.full((x["mixed_qkv"].shape[0], heads * D), -11,
                     dtype=torch.bfloat16, device=device)
    want_out, want = oracle_verify(layer, x, offsets, slots, layer.replay_records, out,
                                   null_block_id)
    before = layer.raw.clone()
    storage_before = x["storage"].clone()
    invoke(layer, x, vector(offsets, device, index_dtype, True),
           vector(slots, device, index_dtype, True), out, null_block_id)
    close(out[:offsets[-1]], want_out[:offsets[-1]])
    for name in ("rawv", "rawk", "g", "beta", "conv"):
        got, expected = getattr(layer.replay_records, name), getattr(want, name)
        for b, length in enumerate(lengths):
            if slots[b] > null_block_id and length:
                active = (b, slice(0, length)) if name == "conv" else (
                    b, slice(None), slice(0, length))
                if name == "conv":
                    assert torch.equal(got[active], expected[active]), name
                else:
                    close(got[active], expected[active],
                          5e-6 if name in ("g", "beta") else 6e-3)
            suffix = got[b, length:] if name == "conv" else got[b, :, length:]
            assert torch.equal(suffix, torch.full_like(suffix, -7)), name
            if slots[b] <= null_block_id or length == 0:
                assert torch.equal(got[b], torch.full_like(got[b], -7)), name
    assert torch.equal(layer.raw, before), "verify changed persistent pages"
    assert torch.equal(x["storage"], storage_before)
    assert torch.equal(out[offsets[-1]:], torch.full_like(out[offsets[-1]:], -11))
    default_out = invoke(layer, x, vector(offsets, device), vector(slots, device), None,
                         null_block_id)
    close(default_out[:offsets[-1]], want_out[:offsets[-1]])
    assert torch.equal(layer.raw, before)


def close_scalar(got, expected, tol):
    """The existing normalized-max contract against a constant FP64 oracle.

    GPU extrema cover every element without allocating a state-sized FP64
    reference tensor. The 1e-30 denominator floor matches reference.score.
    """
    assert bool(torch.isfinite(got).all()), "nonfinite precision-probe result"
    low, high = got.min().item(), got.max().item()
    error = max(abs(low - expected), abs(high - expected))
    normalized = error / max(abs(expected), 1e-30)
    assert normalized <= tol, dict(expected=expected, minimum=low, maximum=high,
                                   max_abs=error, max_rel=normalized, tolerance=tol)


def precision_fixture(device, generic=False):
    """Q4 has H*B=1024, including every valid request in the fast size gate."""
    rows, heads, q = 128, 8, 4
    layer = pages(rows, heads, q, "DS", device, history=3)
    allocate(layer, rows)
    x = inputs(heads, rows * q, device)
    x["storage"].zero_()
    x["mixed_qkv"][:, :2 * heads * D] = 1
    x["conv_weight"].zero_()
    x["conv_weight"][:, -1] = 1
    if generic:
        # Misalignment is an existing native generic-dispatch condition and
        # does not depend on a process-cached PLA_KDA_VERIFY_GATE value.
        storage = torch.empty(x["conv_weight"].numel() + 1, device=device)
        shifted = storage[1:].view_as(x["conv_weight"])
        shifted.copy_(x["conv_weight"])
        x["conv_weight"] = shifted
        assert shifted.data_ptr() % 16 != 0
    else:
        assert x["conv_weight"].data_ptr() % 16 == 0
    x["A_log"].zero_()
    x["dt_bias"].zero_()
    x["lower_bound"] = -8.
    layer.kv_cache[0].zero_()
    layer.kv_cache[1].fill_(1.)
    offsets = vector(list(range(0, rows * q + 1, q)), device)
    slots = vector(list(range(rows)), device)
    out = torch.empty((rows * q, heads * D), device=device, dtype=torch.bfloat16)
    return layer, x, offsets, slots, out


def uniform_factor(rawk, beta, gate):
    # Equivalent to exp(gate)*(1-beta*||k||^2), but FP64 retains the small
    # residual when beta and ||k||^2 approach one. eps is inside the L2 sqrt.
    norm_sq = D * rawk * rawk
    return math.exp(gate) * ((1. - beta) + beta * 1e-6 / (norm_sq + 1e-6))


def case_precision_verify(device, generic=False, overflow=False):
    layer, x, offsets, slots, out = precision_fixture(device, generic)
    before = layer.raw.clone()
    raw = 1. / (1. + math.exp(-1.))  # silu(1), latest conv tap only
    q_sum = D * raw / math.sqrt(D * raw * raw + 1e-6) * D ** -0.5
    # Preserve the zero-product overflow regression even though A_log magnitude
    # no longer selects FP64 repair; gate_product must keep this case finite.
    x["raw_g"].fill_(0. if overflow else -20.)
    x["A_log"].fill_(100. if overflow else 0.)
    gate = -4. if overflow else -8. / (1. + math.exp(20.))
    for beta_input in ((0.,) if overflow else (4., 8., 12., 16., 20.)):
        x["raw_beta"].fill_(beta_input)
        beta = 1. / (1. + math.exp(-beta_input))
        invoke(layer, x, offsets, slots, out, null_block_id=-1)
        # Per-token scoring prevents an earlier larger output from hiding the
        # cancellation residual of a later token. Keep the normal 6e-3 bound.
        factor = uniform_factor(raw, beta, gate)
        for step in range(layer.q):
            close_scalar(out.reshape(128, layer.q, -1)[:, step],
                         factor ** (step + 1) * q_sum, 6e-3)
        records = layer.replay_records
        for field, expected, tol in (("rawk", raw, 6e-3), ("rawv", 0., 6e-3),
                                      ("g", gate, 5e-6), ("beta", beta, 5e-6)):
            close_scalar(getattr(records, field), expected, tol)
        assert torch.equal(layer.raw, before), "precision verify changed cache"


def case_precision_projection(device, generic=False):
    """A well-conditioned query can still cancel inside the H0 projection."""
    layer, x, offsets, slots, out = precision_fixture(device, generic)
    channels = layer.heads * D
    key_weights = x["conv_weight"][channels:2 * channels].view(layer.heads, D, 4)
    key_weights.zero_()
    key_weights[:, 4, -1] = 1
    state = layer.kv_cache[1]
    state.zero_()
    state[..., 0] = 2. ** 26
    state[..., 1] = 1.
    state[..., 2] = -(2. ** 26)
    x["raw_g"].fill_(-20.)
    before = layer.raw.clone()
    raw = 1. / (1. + math.exp(-1.))
    query_coefficient = raw / math.sqrt(D * raw * raw + 1e-6) * D ** -0.5
    gate = -8. / (1. + math.exp(20.))
    # K is nonzero only at coordinate 4, where H0 is zero. Thus each delta is
    # exactly zero and the signed state sum remains one, scaled by gate decay.
    # FP32 projection can lose the unit term between +2**26 and -2**26 even
    # though Qeff itself is well conditioned. Check every output at both betas.
    for beta_input in (0., 4.):
        x["raw_beta"].fill_(beta_input)
        invoke(layer, x, offsets, slots, out, null_block_id=-1)
        for step in range(layer.q):
            close_scalar(out.reshape(128, layer.q, -1)[:, step],
                         query_coefficient * math.exp(gate * (step + 1)), 6e-3)
        assert torch.equal(layer.raw, before), "projection verify changed cache"


def case_precision_commit(device):
    layer, x, offsets, slots, out = precision_fixture(device)
    ctx = context([layer], 128, null_block_id=-1)
    x["raw_g"].fill_(-20.)
    for beta_input in (4., 8., 12., 16., 20.):
        x["raw_beta"].fill_(beta_input)
        layer.kv_cache[0].zero_()
        layer.kv_cache[1].fill_(1.)
        invoke(layer, x, offsets, slots, out, null_block_id=-1)
        records = layer.replay_records
        values = {}
        for name in ("rawk", "g", "beta"):
            tensor = getattr(records, name)
            assert bool(torch.isfinite(tensor).all())
            low, high = tensor.min().item(), tensor.max().item()
            assert low == high, "precision probe records must be uniform"
            values[name] = low
        close_scalar(records.rawv, 0., 0.)
        factor = uniform_factor(values["rawk"], values["beta"], values["g"])
        for count in (0, 1, 2, 4):
            layer.kv_cache[1].fill_(1.)
            layer.kv_cache[0].zero_()
            ctx.commit(vector([count] * 128, device), slots, offsets,
                       token_capacity=x["mixed_qkv"].size(0))
            # Fold oracle uses the supplied BF16/FP32 records, not the
            # unquantized verify intermediates. No tolerance relaxation.
            close_scalar(layer.kv_cache[1], factor ** count, 5e-6)
            if count:
                assert torch.equal(ds(layer), records.conv[:, count - 1])
            else:
                assert torch.equal(ds(layer), torch.zeros_like(ds(layer)))


def case_precision_graph(device):
    """Repair decisions must follow graph-replayed device inputs and metadata."""
    layer, x, offsets, slots, out = precision_fixture(device)
    ctx = context([layer], 128, null_block_id=-1)
    counts = vector([4] * 128, device)
    x["raw_g"].fill_(-20.)
    x["raw_beta"].fill_(8.)

    def run():
        invoke(layer, x, offsets, slots, out, null_block_id=-1)
        ctx.commit(counts, slots, offsets, token_capacity=x["mixed_qkv"].size(0))

    run()  # Initialize lazy runtime state before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    raw = 1. / (1. + math.exp(-1.))
    gate = -8. / (1. + math.exp(20.))
    q_sum = D * raw / math.sqrt(D * raw * raw + 1e-6) * D ** -0.5
    # Move from repair-triggering data to ordinary data, then no-write plans,
    # and back to a partial valid prefix using the same captured allocations.
    variants = ((8., 4, 4, False), (0., 4, 1, False), (8., 4, 0, False),
                (8., 0, 4, False), (8., 4, 4, True), (4., 2, 2, False))
    for beta_input, length, sampled, null in variants:
        x["raw_beta"].fill_(beta_input)
        offsets.copy_(vector([row * length for row in range(129)], device))
        slots.copy_(vector([-1] * 128 if null else list(range(128)), device))
        counts.fill_(sampled)
        layer.kv_cache[0].zero_()
        layer.kv_cache[1].fill_(1.)
        out.fill_(-11)
        for name in ("rawv", "rawk", "g", "beta", "conv"):
            getattr(layer.replay_records, name).fill_(-7)
        before = layer.raw.clone()
        graph.replay()
        torch.cuda.synchronize()
        active_tokens = 128 * length
        accepted = 0 if null else min(sampled, length)
        if null:
            assert torch.equal(out[:active_tokens], torch.zeros_like(out[:active_tokens]))
        elif length:
            beta = 1. / (1. + math.exp(-beta_input))
            factor = uniform_factor(raw, beta, gate)
            for step in range(length):
                close_scalar(out[:active_tokens].reshape(128, length, -1)[:, step],
                             factor ** (step + 1) * q_sum, 6e-3)
        assert torch.equal(out[active_tokens:], torch.full_like(out[active_tokens:], -11))
        records = layer.replay_records
        for name in ("rawv", "rawk", "g", "beta", "conv"):
            tensor = getattr(records, name)
            if null or not length:
                assert torch.equal(tensor, torch.full_like(tensor, -7)), name
            else:
                suffix = tensor[:, length:] if name == "conv" else tensor[:, :, length:]
                assert torch.equal(suffix, torch.full_like(suffix, -7)), name
                if name != "conv":
                    expected = {"rawk": raw, "rawv": 0., "g": gate,
                                "beta": 1. / (1. + math.exp(-beta_input))}[name]
                    close_scalar(tensor[:, :, :length], expected,
                                 6e-3 if name in ("rawk", "rawv") else 5e-6)
        if length and not null:
            for step in range(length):
                expected_conv = torch.zeros_like(records.conv[:, step])
                expected_conv[:, :2 * layer.heads * D, 3 - min(step + 1, 3):] = 1
                assert torch.equal(records.conv[:, step], expected_conv), "graph conv snapshot"
        if accepted:
            factor = uniform_factor(records.rawk[0, 0, 0, 0].item(),
                                    records.beta[0, 0, 0].item(), records.g[0, 0, 0, 0].item())
            close_scalar(layer.kv_cache[1], factor ** accepted, 5e-6)
            assert torch.equal(ds(layer), records.conv[:, accepted - 1])
        else:
            assert torch.equal(layer.raw, before), "no-write graph replay changed cache"
        expected_plan = [[accepted] * 128, list(range(128)) if accepted else [0] * 128,
                         [0] * 128, [0] * 128]
        check_plan(ctx.plan, expected_plan, device)
        padding = before.clone()
        for tensor in layer.kv_cache:
            view = padding.view(tensor.dtype).as_strided(tensor.shape, tensor.stride(),
                                                       tensor.storage_offset())
            view.copy_(tensor)
        assert torch.equal(layer.raw, padding), "precision graph changed page padding"


def case_commit(device, layout, align=False, rounds=4, live=False):
    torch.manual_seed(57)
    q, heads, rows = 5, 3, 3
    layers = [pages(24, heads, q, layout, device, 256 + i * 64) for i in range(2)]
    for layer in layers:
        allocate(layer, rows)
    ctx = context(layers, rows)
    x = inputs(heads, rows * q + 2, device)
    out = torch.full((rows * q + 2, heads * D), -11, device=device, dtype=torch.bfloat16)
    slot_order = [2, 9, 16]
    for turn in range(rounds):
        if live and turn == 3:
            for layer in layers:
                layer.kv_cache[1][9].zero_()
                ds(layer)[9].zero_()
        slot_order = slot_order[1:] + slot_order[:1]
        lengths = [q, 3, 1] if turn % 2 == 0 else [1, q, 0]
        offsets = [0, lengths[0], sum(lengths[:2]), sum(lengths)]
        mapping = [2, 0, 3]
        counts = ([2, 777, q, 0], [-3, 777, 1, q + 10],
                  [q, 777, 0, 1], [q + 10, 777, 1, 0])[turn % 4]
        table = [[1 + r * 6 + j for j in range(4)] for r in range(4)]
        computed = [6, 1, 7, 8]
        slots = ([table[mapping[b]][(computed[mapping[b]] - 1) // 8]
                  for b in range(rows)] if align else slot_order)
        cu, si = vector(offsets, device, torch.int64, True), vector(slots, device)
        wants = []
        for layer in layers:
            want_out, golden = oracle_verify(layer, x, offsets, slots,
                                             layer.replay_records, out)
            persistent_before, output_before = layer.raw.clone(), out.clone()
            invoke(layer, x, cu, si, out)
            close(out[:offsets[-1]], want_out[:offsets[-1]])
            assert torch.equal(layer.raw, persistent_before), "multiround verify changed cache"
            assert torch.equal(out[offsets[-1]:], output_before[offsets[-1]:])
            # Isolated fold tests use independently rounded golden rings. The
            # live pipeline retains verified records across four real commits.
            for name in ("rawv", "rawk", "g", "beta", "conv"):
                got, target = getattr(layer.replay_records, name), getattr(golden, name)
                for b, length in enumerate(lengths):
                    active = (b, slice(0, length)) if name == "conv" else (
                        b, slice(None), slice(0, length))
                    if name == "conv":
                        assert torch.equal(got[active], target[active])
                    else:
                        close(got[active], target[active],
                              5e-6 if name in ("g", "beta") else 6e-3)
                    suffix = (b, slice(length, None)) if name == "conv" else (
                        b, slice(None), slice(length, None))
                    assert torch.equal(got[suffix], target[suffix]), name
                if not live:
                    got.copy_(target)
            replay = layer.replay_records if live else golden
            wants.append(oracle_commit(layer, replay, offsets, slots, counts, mapping,
                                       table if align else None,
                                       computed if align else None, 8 if align else None))
        before = [layer.raw.clone() for layer in layers]
        kwargs = dict(request_indices=vector(mapping, device, torch.int64, True),
                      token_capacity=x["mixed_qkv"].size(0))
        if align:
            kwargs.update(block_table=torch.tensor(table, device=device, dtype=torch.int32),
                          num_computed_tokens=vector(computed, device), mamba_block_size=8)
        plan = ctx.commit(vector(counts, device, torch.int64, True), si, cu, **kwargs)
        check_plan(plan, wants[0][2], device)
        for layer, (state, conv, _), raw_before in zip(layers, wants, before):
            close(layer.kv_cache[1], state, 5e-6)
            touched = {s for n, f, b, bc in zip(*wants[0][2]) if n
                       for s in ([f, b] if bc else [f])}
            for slot in range(state.shape[0]):
                if slot not in touched:
                    assert torch.equal(layer.kv_cache[1][slot], state[slot]), slot
            assert torch.equal(ds(layer), conv), "conv snapshot scatter is not exact"
            # Reconstruct expected bytes, including untouched cache slots and page padding.
            expected_raw = raw_before.clone()
            c0, s0 = layer.kv_cache
            expected_s = expected_raw.view(torch.float32).as_strided(
                s0.shape, s0.stride(), s0.storage_offset())
            expected_c = expected_raw.view(torch.bfloat16).as_strided(
                c0.shape, c0.stride(), c0.storage_offset())
            expected_s.copy_(s0)
            expected_c.copy_(c0)
            assert torch.equal(layer.raw, expected_raw), "page padding changed"
        x["mixed_qkv"].normal_(0, 0.4)
    # Request termination and slot reuse: reset the released slot, then continue.
    layers[0].kv_cache[1][9].zero_()
    ds(layers[0])[9].zero_()
    cu, si = vector([0, 1, 1, 1], device), vector([9, 0, -1], device)
    golden_out, _ = oracle_verify(layers[0], x, [0, 1, 1, 1], [9, 0, -1],
                                 layers[0].replay_records, out)
    invoke(layers[0], x, cu, si, out)
    close(out[:1], golden_out[:1])


def case_empty(device):
    layer = pages(8, 1, 2, "SD", device)
    allocate(layer, 2)
    x = inputs(1, 0, device)
    out = torch.empty(0, D, device=device, dtype=torch.bfloat16)
    before = layer.raw.clone()
    invoke(layer, x, vector([0, 0, 0], device), vector([0, -1], device), out)
    ctx = context([layer], 2)
    plan = ctx.commit(vector([0, 9], device), vector([0, -1], device),
                      vector([0, 0, 0], device), token_capacity=x["mixed_qkv"].size(0))
    check_plan(plan, [[0, 0]] * 4, device)
    assert torch.equal(layer.raw, before)


def case_graph(device, phase, q=5, heads=1, rows=2, align=True):
    torch.manual_seed(72)
    layer = pages(12, heads, q, "DS", device)
    allocate(layer, rows)
    ctx = context([layer], rows)
    x = inputs(heads, 2 * q, device)
    def padded(values, fill):
        return values + [fill] * (rows - 2)
    cu = vector(padded([0, q, 2 * q], 2 * q), device)
    si = vector(padded([2, 7], 0), device)
    counts, mapping = vector([3, q], device), vector(padded([0, 1], 0), device)
    table = torch.tensor([[2, 3], [7, 8]], device=device, dtype=torch.int32)
    computed = vector([7, 6], device)
    out = torch.full((2 * q, heads * D), -11, device=device, dtype=torch.bfloat16)
    initial = layer.raw.clone()

    def run():
        if phase != "commit":
            invoke(layer, x, cu, si, out)
        if phase != "verify":
            kwargs = dict(request_indices=mapping, token_capacity=x["mixed_qkv"].size(0))
            if align:
                kwargs.update(block_table=table, num_computed_tokens=computed,
                              mamba_block_size=8)
            ctx.commit(counts, si, cu, **kwargs)

    invoke(layer, x, cu, si, out)
    run()
    layer.raw.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    # Python tensor-value reads are forbidden even if the runtime would allow them.
    from unittest.mock import patch
    from contextlib import ExitStack
    with ExitStack() as guard:
        for name in ("item", "tolist", "cpu", "numpy"):
            guard.enter_context(patch.object(torch.Tensor, name, side_effect=AssertionError(
                "host tensor-value read in capture")))
        with torch.cuda.graph(graph):
            run()
    for turn in range(3):
        layer.raw.copy_(initial)
        offsets, slots = ([0, 2, 2], [2, 0]) if turn == 1 else ([0, 1, 1 + q], [2, 7])
        cu.copy_(vector(padded(offsets, offsets[-1]), device))
        si.copy_(vector(padded(slots, 0), device))
        accepted = [0, q] if turn == 1 else [2, q] if turn == 2 and not align else [1, 3]
        counts.copy_(vector(accepted, device))
        mapping.copy_(vector(padded([1, 0], 0), device))
        computed.copy_(vector([7, 8], device))
        table.copy_(torch.tensor([[7, 9 + turn], [2, 4 + turn]],
                                 device=device, dtype=torch.int32))
        out.fill_(-11)
        if phase == "commit":
            invoke(layer, x, cu, si, out)
        want_out, golden = oracle_verify(layer, x, offsets, slots,
                                         layer.replay_records, out)
        # Commit-only oracle uses the existing rounded records; combined uses verify.
        records = layer.replay_records if phase == "commit" else golden
        count_values, map_values = counts.tolist(), mapping.tolist()
        table_values = table.tolist() if align else None
        computed_values = computed.tolist() if align else None
        block_size = 8 if align else None
        state, conv, _ = oracle_commit(layer, records, offsets, slots,
                                       count_values, map_values, table_values,
                                       computed_values, block_size)
        checkpoint = layer.kv_cache[1].clone()
        conv_checkpoint = ds(layer).clone()
        graph.replay()
        torch.cuda.synchronize()
        if phase != "commit":
            close(out[:offsets[-1]], want_out[:offsets[-1]])
            for b, slot in enumerate(slots):
                length = offsets[b + 1] - offsets[b]
                if slot > 0 and length:
                    for name in ("rawv", "rawk", "g", "beta"):
                        close(getattr(layer.replay_records, name)[b, :, :length],
                              getattr(golden, name)[b, :, :length],
                              5e-6 if name in ("g", "beta") else 6e-3)
                    assert torch.equal(layer.replay_records.conv[b, :length],
                                       golden.conv[b, :length]), "graph conv snapshot differs"
        if phase == "combined":
            # Verify is checked independently above. Replay its rounded records
            # from the pre-graph checkpoint to isolate the FP32 commit contract.
            saved = SimpleNamespace(kv_cache=(conv_checkpoint, checkpoint),
                                    heads=heads, q=q, layout="DS")
            state, conv, _ = oracle_commit(saved, layer.replay_records, offsets, slots,
                                           count_values, map_values, table_values,
                                           computed_values, block_size)
        assert torch.equal(out[offsets[-1]:], torch.full_like(out[offsets[-1]:], -11))
        if phase != "commit":
            for b, slot in enumerate(slots):
                length = offsets[b + 1] - offsets[b] if slot > 0 else 0
                for name in ("rawv", "rawk", "g", "beta", "conv"):
                    got, expected = getattr(layer.replay_records, name), getattr(golden, name)
                    suffix = (b, slice(length, None)) if name == "conv" else (
                        b, slice(None), slice(length, None))
                    assert torch.equal(got[suffix], expected[suffix]), name
        if phase != "verify":
            close(layer.kv_cache[1], state, 5e-6)
            assert torch.equal(ds(layer), conv)
            untouched = initial.clone()
            c, s = layer.kv_cache
            untouched.view(torch.float32).as_strided(s.shape, s.stride(), s.storage_offset()).copy_(s)
            untouched.view(torch.bfloat16).as_strided(c.shape, c.stride(), c.storage_offset()).copy_(c)
            assert torch.equal(layer.raw, untouched), "graph changed page padding"
            for slot in range(s.shape[0]):
                if torch.equal(state[slot], checkpoint[slot]):
                    assert torch.equal(s[slot], checkpoint[slot]), "unused state changed"
        else:
            assert torch.equal(layer.raw, initial)


def case_tp(device, tp, distributed_rank=None):
    torch.manual_seed(123)
    heads, q = 12, 2
    full = pages(8, heads, q, "SD", device)
    allocate(full, 2)
    x = inputs(heads, 4, device)
    assert heads % tp == 0, "TP must divide the head count"
    acceptance = vector([1, 2], device)
    if distributed_rank is not None:
        # Broadcast the same sampler result and original inputs/checkpoints.
        for tensor in (full.kv_cache[1], full.kv_cache[0], *[
                x[k] for k in ("mixed_qkv", "raw_g", "raw_beta", "conv_weight", "A_log", "dt_bias")]):
            dense = tensor.contiguous()
            torch.distributed.broadcast(dense, src=0)
            tensor.copy_(dense)
        if distributed_rank != 0:
            acceptance.zero_()
        torch.distributed.broadcast(acceptance, src=0)
    offsets, slots = [0, 2, 4], [2, 5]
    out = torch.empty(4, heads * D, device=device, dtype=torch.bfloat16)
    want, golden = oracle_verify(full, x, offsets, slots, full.replay_records, out)
    outputs, states, convs = [], [], []
    for rank in (range(tp) if distributed_rank is None else [distributed_rank]):
        lh, start = heads // tp, rank * heads // tp
        local = pages(8, lh, q, "SD", device)
        local.kv_cache[1].copy_(full.kv_cache[1][:, start:start + lh])
        for channel in range(3):
            ds(local)[:, channel * lh * D:(channel + 1) * lh * D].copy_(
                ds(full)[:, (channel * heads + start) * D:
                         (channel * heads + start + lh) * D])
        allocate(local, 2)
        lx = dict(x)
        lx["mixed_qkv"] = torch.cat([t[:, start * D:(start + lh) * D]
                                     for t in x["mixed_qkv"].split(heads * D, -1)], -1)
        lx["raw_g"] = x["raw_g"][:, start * D:(start + lh) * D]
        lx["raw_beta"] = x["raw_beta"][:, start:start + lh]
        lx["conv_weight"] = torch.cat([t[start * D:(start + lh) * D]
                                       for t in x["conv_weight"].split(heads * D)], 0)
        lx["A_log"] = x["A_log"][start:start + lh]
        lx["dt_bias"] = x["dt_bias"][start * D:(start + lh) * D]
        lo = torch.empty(4, lh * D, device=device, dtype=torch.bfloat16)
        invoke(local, lx, vector(offsets, device), vector(slots, device), lo)
        outputs.append(lo)
        # Isolated commit uses independently rounded full-head golden rings.
        for name in ("rawv", "rawk", "g", "beta"):
            getattr(local.replay_records, name).copy_(getattr(golden, name)[:, start:start + lh])
        local.replay_records.conv.copy_(torch.cat([
            golden.conv[:, :, (channel * heads + start) * D:
                        (channel * heads + start + lh) * D] for channel in range(3)], 2))
        # Same acceptance decisions for every head shard.
        context([local], 2).commit(acceptance, vector(slots, device),
                                    vector(offsets, device), token_capacity=lx["mixed_qkv"].size(0))
        states.append(local.kv_cache[1])
        convs.append(ds(local))
    if distributed_rank is not None:
        gathered = []
        for values in (outputs, states, convs):
            local_value = values[0].contiguous()
            chunks = [torch.empty_like(local_value) for _ in range(tp)]
            torch.distributed.all_gather(chunks, local_value)
            gathered.append(chunks)
        outputs, states, convs = gathered
    close(torch.cat(outputs, -1), want)
    target_state, target_conv, _ = oracle_commit(full, golden, offsets, slots, [1, 2])
    close(torch.cat(states, 1), target_state, 5e-6)
    joined_conv = torch.cat([torch.cat([c[:, channel * (heads // tp) * D:
                                                     (channel + 1) * (heads // tp) * D]
                                      for c in convs], 1) for channel in range(3)], 1)
    assert torch.equal(joined_conv, target_conv)


def serial_prefix(layer, x, slot, length):
    """Build prefill checkpoints one token at a time, with replay-record rounding."""
    for token in range(length):
        one = dict(x)
        for name in ("mixed_qkv", "raw_g", "raw_beta"):
            one[name] = x[name][token:token + 1]
        _, rings = oracle_verify(layer, one, [0, 1], [slot],
                                  blank_records(1, layer.heads, layer.q, layer.raw.device),
                                  torch.zeros(1, layer.heads * D, device=layer.raw.device))
        layer.kv_cache[1][slot].copy_(folded(layer.kv_cache[1][slot], rings, 0, 1))
        ds(layer)[slot, :, :3].copy_(rings.conv[0, 0])


def case_prefix_continuation(device, layout):
    torch.manual_seed(88)
    layer = pages(10, 1, 5, layout, device)
    allocate(layer, 1)
    x = inputs(1, 8, device)
    serial_prefix(layer, x, 2, 4)
    before = layer.raw.clone()
    out = torch.full((8, D), -11, dtype=torch.bfloat16, device=device)
    want, rings = oracle_verify(layer, x, [0, 5], [2], layer.replay_records, out)
    invoke(layer, x, vector([0, 5], device), vector([2], device), out)
    close(out[:5], want[:5])
    for name in ("rawv", "rawk", "g", "beta", "conv"):
        getattr(layer.replay_records, name).copy_(getattr(rings, name))
    ctx = context([layer], 1)
    state, conv, expected = oracle_commit(layer, rings, [0, 5], [2], [3],
                                          block_table=[[1, 3]], computed=[6], block_size=8)
    plan = ctx.commit(vector([3], device), vector([2], device), vector([0, 5], device),
                      block_table=torch.tensor([[1, 3]], device=device, dtype=torch.int64),
                      num_computed_tokens=vector([6], device), mamba_block_size=8,
                      token_capacity=x["mixed_qkv"].size(0))
    check_plan(plan, expected, device)
    close(layer.kv_cache[1][1], state[1], 5e-6)
    close(layer.kv_cache[1][3], state[3], 5e-6)
    assert torch.equal(ds(layer), conv)
    initial_state = before.view(torch.float32).as_strided(
        layer.kv_cache[1].shape, layer.kv_cache[1].stride(), layer.kv_cache[1].storage_offset())
    assert torch.equal(layer.kv_cache[1][2], initial_state[2]), "source checkpoint changed"
    # Restore the boundary and consume the remaining accepted token, then two
    # subsequent tokens. Compare against continuing directly from the final slot.
    outputs, states = [], []
    for source, destination, tokens in ((1, 4, [2, 5, 6]), (3, 5, [5, 6])):
        layer.kv_cache[1][destination].copy_(layer.kv_cache[1][source])
        ds(layer)[destination].copy_(ds(layer)[source])
        tail = dict(x)
        for name in ("mixed_qkv", "raw_g", "raw_beta"):
            tail[name] = x[name][tokens]
        length = len(tokens)
        target, golden = oracle_verify(layer, tail, [0, length], [destination],
                                        layer.replay_records, out[:length])
        invoke(layer, tail, vector([0, length], device), vector([destination], device), out[:length])
        close(out[:length], target)
        outputs.append(out[length - 2:length].clone())
        for name in ("rawv", "rawk", "g", "beta", "conv"):
            getattr(layer.replay_records, name).copy_(getattr(golden, name))
        ctx.commit(vector([length], device), vector([destination], device),
                   vector([0, length], device), token_capacity=tail["mixed_qkv"].size(0))
        states.append(layer.kv_cache[1][destination].clone())
    close(outputs[0], outputs[1])
    close(states[0], states[1], 5e-6)
    assert torch.equal(ds(layer)[4, :, :3], ds(layer)[5, :, :3])


def case_mixed_prefill(device):
    torch.manual_seed(89)
    layers = [pages(14, 1, 5, "SD", device, padding=256 + 64 * i) for i in range(3)]
    x = inputs(1, 10, device, offset=4)
    for layer in layers:
        allocate(layer, 2)
        serial_prefix(layer, x, 2, 4)
        serial_prefix(layer, x, 6, 2)
    prefix = x["storage"].clone()
    # The projection view lies between non-spec prefix/suffix rows; sampler rows
    # also contain unrelated prefill/decode requests with conspicuous counts.
    ctx = context(layers, 2)
    for layer in layers:
        out = torch.full((10, D), -11, device=device, dtype=torch.bfloat16)
        want, golden = oracle_verify(layer, x, [0, 5, 7], [6, 2], layer.replay_records, out)
        before = layer.raw.clone()
        invoke(layer, x, vector([0, 5, 7], device), vector([6, 2], device), out)
        close(out[:7], want[:7])
        assert torch.equal(layer.raw, before)
        assert torch.equal(out[7:], torch.full_like(out[7:], -11))
        for name in ("rawv", "rawk", "g", "beta", "conv"):
            getattr(layer.replay_records, name).copy_(getattr(golden, name))
    targets = [oracle_commit(l, l.replay_records, [0, 5, 7], [6, 2],
                             [999, 1, 888, 5], [3, 1]) for l in layers]
    ctx.commit(vector([999, 1, 888, 5], device), vector([6, 2], device),
               vector([0, 5, 7], device), request_indices=vector([3, 1], device),
               token_capacity=x["mixed_qkv"].size(0))
    for layer, (state, conv, _) in zip(layers, targets):
        close(layer.kv_cache[1], state, 5e-6)
        assert torch.equal(ds(layer), conv)
        for slot in set(range(14)) - {2, 6}:
            assert torch.equal(layer.kv_cache[1][slot], state[slot])
    assert torch.equal(x["storage"], prefix), "mixed projection storage changed"


def case_context_cache_overlap(device, layout):
    arena = pages(6, 1, 1, layout, device, history=6)
    conv, state = arena.kv_cache
    history_dim = 2 if layout == "DS" else 1
    first_conv, second_conv = conv[::2].split(3, dim=history_dim)
    layers = []
    # Share storage across layers. State pages alternate and conv histories
    # occupy disjoint portions of the same pages, including strided DS views.
    for kv_cache in ((first_conv, state[::2]), (second_conv, state[1::2])):
        layer = SimpleNamespace(**vars(arena))
        layer.kv_cache = kv_cache
        allocate(layer, 1)
        layers.append(layer)
    first, second = layers

    ctx = context(layers, 1)
    targets = []
    for i, layer in enumerate(layers):
        records = layer.replay_records
        records.g.fill_(-1)
        records.beta.zero_()
        records.conv.fill_(i + 1)
        targets.append(oracle_commit(layer, records, [0, 1], [1], [1]))
    ctx.commit(vector([1], device), vector([1], device), vector([0, 1], device),
               token_capacity=1)
    for layer, (want_state, want_conv, _) in zip(layers, targets):
        close(layer.kv_cache[1], want_state, 5e-6)
        assert torch.equal(ds(layer), want_conv)

    first_state, second_state = first.kv_cache[1], second.kv_cache[1]
    partial_state = first_state.as_strided(first_state.shape, first_state.stride(),
                                           first_state.storage_offset() + 4)
    partial_conv = conv[::2].narrow(history_dim, 2, 3)
    conv_in_state = second_conv.as_strided(second_conv.shape, second_conv.stride(),
                                          2 * first_state.storage_offset())
    overlapping_layers = [first]
    for kv_cache in ((second_conv, partial_state), (second_conv, state[2:5]),
                     (partial_conv, second_state), (conv_in_state, second_state)):
        overlapping_layers.append(SimpleNamespace(**(vars(second) | dict(kv_cache=kv_cache))))
    for layer in overlapping_layers:
        try:
            context([first, layer], 1)
        except ValueError as error:
            assert "overlapping writable caches" in str(error), error
        else:
            raise AssertionError("overlapping layer cache accepted")


def case_all_invalid(device):
    layer = pages(8, 1, 5, "DS", device)
    allocate(layer, 3)
    x = inputs(1, 15, device)
    out = torch.full((15, D), -11, dtype=torch.bfloat16, device=device)
    before, rings = layer.raw.clone(), clone_records(layer.replay_records)
    invoke(layer, x, vector([0, 5, 8, 8], device), vector([0, -1, 4], device), out)
    assert torch.equal(out[:8], torch.zeros_like(out[:8]))
    assert torch.equal(out[8:], torch.full_like(out[8:], -11))
    for name in ("rawv", "rawk", "g", "beta", "conv"):
        assert torch.equal(getattr(layer.replay_records, name), getattr(rings, name))
    ctx = context([layer], 3)
    plan = ctx.commit(vector([5, 5, 5], device), vector([0, -1, 4], device),
                      vector([0, 5, 8, 8], device), token_capacity=x["mixed_qkv"].size(0))
    check_plan(plan, [[0, 0, 0]] * 4, device)
    assert torch.equal(layer.raw, before)
    invoke(layer, x, vector([0], device), vector([], device), out)
    check_plan(ctx.commit(vector([], device), vector([], device), vector([0], device),
                          token_capacity=x["mixed_qkv"].size(0)),
               [[0, 0, 0]] * 4, device)
    assert torch.equal(layer.raw, before)


def case_align_shapes(device, q, layout):
    torch.manual_seed(130 + q)
    layer = pages(10, 1, q, layout, device)
    allocate(layer, 1)
    ctx = context([layer], 1)
    x = inputs(1, q, device)
    out = torch.empty(q, D, device=device, dtype=torch.bfloat16)
    # Not crossing, exactly reaching, crossing, and legal source/destination
    # coincidences. Block tables themselves are positive-stride int64 views.
    size = max(q, 4)
    variants = [(0, 1, [2, 3]), (size - 1, 1, [2, 3]),
                (size, q, [2, 3]), (size - 1, q, [4, 5])]
    for position, count, table in variants:
        want, golden = oracle_verify(layer, x, [0, q], [2], layer.replay_records, out)
        invoke(layer, x, vector([0, q], device, torch.int64), vector([2], device), out)
        close(out, want)
        for name in ("rawv", "rawk", "g", "beta", "conv"):
            getattr(layer.replay_records, name).copy_(getattr(golden, name))
        state, conv, expected = oracle_commit(layer, golden, [0, q], [2], [count],
                                              block_table=[table], computed=[position], block_size=size)
        bt_storage = torch.full((2, 5), -99, dtype=torch.int64, device=device)
        bt = bt_storage[1:2, 1::2]
        bt.copy_(torch.tensor([table], device=device))
        plan = ctx.commit(vector([count], device), vector([2], device), vector([0, q], device),
                          block_table=bt, num_computed_tokens=vector([position], device),
                          mamba_block_size=size, token_capacity=x["mixed_qkv"].size(0))
        check_plan(plan, expected, device)
        close(layer.kv_cache[1], state, 5e-6)
        assert torch.equal(ds(layer), conv)


def case_conflicts(device):
    layer = pages(8, 1, 2, "SD", device)
    allocate(layer, 2)
    ctx = context([layer], 2)
    x = inputs(1, 4, device)
    out = torch.empty(4, D, device=device, dtype=torch.bfloat16)
    cu, slots, counts = vector([0, 2, 4], device), vector([2, 3], device), vector([1, 1], device)
    invoke(layer, x, cu, slots, out)
    before = layer.raw.clone()
    kwargs = dict(block_table=torch.tensor([[2, 3], [3, 4]], device=device),
                  num_computed_tokens=vector([8, 7], device), mamba_block_size=8,
                  token_capacity=x["mixed_qkv"].size(0))
    for accepted in ([1, 1], [1, 0]):
        counts.copy_(vector(accepted, device))
        if os.environ.get("PLA_SANITY_CHECK") == "1":
            try:
                ctx.commit(counts, slots, cu, **kwargs)
            except RuntimeError as error:
                assert "overlap" in str(error)
            else:
                raise AssertionError("sanity did not reject conflicting destinations")
        else:
            check_plan(ctx.commit(counts, slots, cu, **kwargs), [[0, 0]] * 4, device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ctx.commit(counts, slots, cu, **kwargs)
        graph.replay()
        torch.cuda.synchronize()
        check_plan(ctx.plan, [[0, 0]] * 4, device)
        assert torch.equal(layer.raw, before)
    # Empty rows do not own a checkpoint, unlike nonempty c=0 requests.
    cu.copy_(vector([0, 2, 2], device))
    state, conv, expected = oracle_commit(layer, layer.replay_records, [0, 2, 2], [2, 3], [1, 0],
        block_table=[[2, 3], [3, 4]], computed=[8, 7], block_size=8)
    check_plan(ctx.commit(counts, slots, cu, **kwargs), expected, device)
    close(layer.kv_cache[1][3], state[3], 5e-6)
    assert torch.equal(ds(layer), conv)


def case_conv_packed(device, layout, history):
    torch.manual_seed(301 + history)
    q, heads, rows = 4, 3, 4
    layers = [pages(16, heads, q, layout, device, history=history, conv_offset=offset)
              for offset in (0, 2)]
    for layer in layers:
        allocate(layer, rows)
        r = layer.replay_records
        for name, val in (("rawv", .1), ("rawk", .2), ("g", -.1), ("beta", .5)):
            getattr(r, name).fill_(val)
        r.conv.copy_(torch.randn(r.conv.shape, device=device, dtype=torch.bfloat16))
    ctx = context(layers, rows)
    offsets, sources = [0, 4, 8, 12, 16], [2, 3, 4, 5]
    table, computed = [[10, 6], [11, 7], [12, 8], [13, 9]], [8, 8, 7, 6]
    si, cu = vector(sources, device), vector(offsets, device)
    accepted = vector([0, 1, 2, 4], device)
    kwargs = dict(block_table=torch.tensor(table, device=device, dtype=torch.int64),
                  num_computed_tokens=vector(computed, device), mamba_block_size=8,
                  token_capacity=offsets[-1])
    initial = [layer.raw.clone() for layer in layers]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ctx.commit(accepted, si, cu, **kwargs)
    for counts in ([0, 1, 2, 4], [4, 2, 1, 0], [0, 0, 0, 0]):
        accepted.copy_(vector(counts, device))
        for layer, before in zip(layers, initial):
            layer.raw.copy_(before)
        golden = [oracle_commit(layer, layer.replay_records, offsets, sources, counts,
                               block_table=table, computed=computed, block_size=8)
                  for layer in layers]
        graph.replay()
        torch.cuda.synchronize()
        check_plan(ctx.plan, golden[0][2], device)
        for layer, before, (state, conv, _) in zip(layers, initial, golden):
            close(layer.kv_cache[1], state, 5e-6)
            assert torch.equal(ds(layer), conv)
            c, s = layer.kv_cache
            old_state = before.view(torch.float32).as_strided(s.shape, s.stride(), s.storage_offset())
            for slot in range(6):
                assert torch.equal(s[slot], old_state[slot])
            canary = before.clone()
            canary.view(torch.float32).as_strided(s.shape, s.stride(), s.storage_offset()).copy_(s)
            canary.view(torch.bfloat16).as_strided(c.shape, c.stride(), c.storage_offset()).copy_(c)
            assert torch.equal(layer.raw, canary)
            if not any(counts):
                assert torch.equal(layer.raw, before)


def case_parallel_conflicts(device, batch, align=True):
    q, capacity = 4, batch + 3
    layer = pages(2 * batch + 8, 1, q, "DS", device)
    allocate(layer, capacity)
    ctx = context([layer], capacity)
    r = layer.replay_records
    for name, val in (("rawv", .1), ("rawk", .2), ("g", -.1), ("beta", .5), ("conv", .3)):
        getattr(r, name).fill_(val)
    before = layer.raw.clone()
    variants = ("c0_owner", "empty_owner", "duplicate_final", "boundary_source",
                "reverse_source", "disjoint", "wide_source") if align else (
                "c0_owner", "empty_owner", "duplicate_final", "disjoint",
                "wide_source", "wide_length")
    graph = None
    for variant in variants:
        layer.raw.copy_(before)
        sources = list(range(1, batch + 1))
        offsets = [i * q for i in range(batch + 1)]
        counts = [0] * batch
        counts[0] = 1
        table = [[batch + 1 + i, batch + 1 + i] for i in range(batch)]
        positions = [8] * batch
        blocked = variant in ("c0_owner", "duplicate_final", "boundary_source", "reverse_source")
        if variant in ("c0_owner", "empty_owner"):
            table[0][1] = sources[-1]
            if variant == "empty_owner":
                offsets[-1] = offsets[-2]
        elif variant == "duplicate_final":
            counts[-1] = 1
            table[-1][1] = table[0][1]
        elif variant == "boundary_source":
            counts[0], counts[-1], positions[0] = 2, 1, 7
            table[0][0] = sources[-1]
        elif variant == "reverse_source":
            counts[-1] = 1
            table[-1][1] = sources[0]
        elif variant == "disjoint":
            counts[0], counts[-1] = 2, 1
        else:
            sources[-1] = 2**32 + table[0][1]
        if not align:
            if variant in ("c0_owner", "empty_owner", "duplicate_final", "wide_length"):
                sources[-1] = sources[0]
            elif variant == "wide_source":
                sources[-1] = 2**32 + sources[0]
            if variant == "wide_length":
                offsets[-1] += 2**32
        expected_counts = [0] * batch if blocked else counts
        state, conv, expected = oracle_commit(layer, r, offsets, sources, expected_counts,
            block_table=table if align else None, computed=positions if align else None,
            block_size=8 if align else None)
        expected = [col + [0] * (capacity - batch) for col in expected]
        if graph is None:
            cu = vector(offsets, device, torch.int64, True)
            si = vector(sources, device, torch.int64, True)
            accepted = vector(counts, device, torch.int64, True)
            kwargs = dict(block_table=torch.tensor(table, device=device, dtype=torch.int64),
                          num_computed_tokens=vector(positions, device, torch.int64, True),
                          mamba_block_size=8) if align else {}
        else:
            cu.copy_(vector(offsets, device, torch.int64))
            si.copy_(vector(sources, device, torch.int64))
            accepted.copy_(vector(counts, device))
            if align:
                kwargs["block_table"].copy_(torch.tensor(table, device=device))
                kwargs["num_computed_tokens"].copy_(vector(positions, device))
        if os.environ.get("PLA_SANITY_CHECK") == "1" and (
                blocked or variant in ("wide_source", "wide_length")):
            try:
                ctx.commit(accepted, si, cu, token_capacity=batch * q, **kwargs)
            except RuntimeError:
                pass
            else:
                raise AssertionError("sanity accepted invalid metadata: " + variant)
        else:
            check_plan(ctx.commit(accepted, si, cu, token_capacity=batch * q, **kwargs),
                       expected, device)
        layer.raw.copy_(before)
        if graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ctx.commit(accepted, si, cu, token_capacity=batch * q, **kwargs)
        graph.replay()
        torch.cuda.synchronize()
        check_plan(ctx.plan, expected, device)
        close(layer.kv_cache[1], state, 5e-6)
        assert torch.equal(ds(layer), conv), variant
        canary = before.clone()
        c, s = layer.kv_cache
        canary.view(torch.float32).as_strided(s.shape, s.stride(), s.storage_offset()).copy_(s)
        canary.view(torch.bfloat16).as_strided(c.shape, c.stride(), c.storage_offset()).copy_(c)
        assert torch.equal(layer.raw, canary), variant


def case_invalid(device):
    layer = pages(8, 1, 2, "SD", device)
    allocate(layer, 1)
    ctx = context([layer], 1)
    args = (vector([1], device), vector([2], device), vector([0, 1], device))
    for kwargs in (dict(block_table=torch.ones(1, 2, device=device, dtype=torch.int32)),
                   dict(block_table=torch.ones(1, 2, device=device, dtype=torch.int32),
                        num_computed_tokens=vector([1], device), mamba_block_size=1),
                   dict(token_capacity=-1), dict(token_capacity=2**31),
                   dict(token_capacity=vector([1], device))):
        try:
            ctx.commit(*args, **({"token_capacity": 1} | kwargs))
        except (TypeError, ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid commit arguments accepted")
    try:
        ctx.commit(*args, block_table=torch.ones(1, 2, device=device, dtype=torch.int32),
                   num_computed_tokens=vector([1], device), mamba_block_size=vector([8], device),
                   token_capacity=1)
    except TypeError:
        pass
    else:
        raise AssertionError("device block-size scalar accepted")
    from pla.decode.kda_replayssm_vllm import kda_replay_verify
    x = inputs(1, 2, device)
    kwargs = {k: v for k, v in x.items() if k != "storage"}
    kwargs.update(conv_state=layer.kv_cache[0], checkpoint_state=layer.kv_cache[1],
                  records=layer.replay_records, query_start_loc=args[2],
                  state_indices=args[1], conv_state_layout="SD")
    misaligned = torch.empty(layer.kv_cache[1].numel() + 1, device=device)[1:].reshape(
        layer.kv_cache[1].shape)
    for change in (dict(mixed_qkv=x["mixed_qkv"].float()), dict(lower_bound=0),
                   dict(conv_state=layer.kv_cache[0].float()),
                   dict(checkpoint_state=misaligned), dict(conv_state_layout="invalid"),
                   dict(raw_beta=x["raw_beta"].expand(2, 2)),
                   dict(out=torch.empty(2, D, device=device, dtype=torch.float32)),
                   dict(conv_weight=x["conv_weight"][:, :3].contiguous()),
                   dict(checkpoint_state=layer.kv_cache[1].bfloat16()),
                   dict(scale=torch.tensor(1., device=device)),
                   dict(lower_bound=torch.tensor(-5., device=device))):
        try:
            kda_replay_verify(**(kwargs | change))
        except (TypeError, ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid shape/dtype/alignment accepted")


def case_null_minus_one(device):
    layer = pages(8, 1, 2, "SD", device)
    allocate(layer, 2)
    x = inputs(1, 4, device)
    out = torch.full((4, D), -11, dtype=torch.bfloat16, device=device)
    offsets, slots = [0, 2, 3], [0, -1]
    want, golden = oracle_verify(layer, x, offsets, slots, layer.replay_records, out, -1)
    before = layer.raw.clone()
    invoke(layer, x, vector(offsets, device), vector(slots, device), out, -1)
    close(out[:3], want[:3])
    assert torch.equal(layer.raw, before)
    assert torch.equal(out[3:], torch.full_like(out[3:], -11))
    for name in ("rawv", "rawk", "g", "beta", "conv"):
        got, target = getattr(layer.replay_records, name), getattr(golden, name)
        assert torch.equal(got[1], target[1]), name
        if name == "conv":
            assert torch.equal(got[0], target[0])
        else:
            close(got[0], target[0], 5e-6 if name in ("g", "beta") else 6e-3)
        got.copy_(target)
    state, conv, expected = oracle_commit(layer, golden, offsets, slots, [1, 2], null_block_id=-1)
    plan = context([layer], 2, -1).commit(vector([1, 2], device), vector(slots, device),
                                         vector(offsets, device),
                                         token_capacity=x["mixed_qkv"].size(0))
    check_plan(plan, expected, device)
    close(layer.kv_cache[1][0], state[0], 5e-6)
    assert torch.equal(ds(layer), conv)


def case_device_metadata(device, q=2, heads=1, rows=1):
    layer = pages(8, heads, q, "SD", device)
    allocate(layer, rows)
    ctx = context([layer], rows)
    x = inputs(heads, q + 1, device)
    out = torch.full((q + 1, heads * D), -11, dtype=torch.bfloat16, device=device)
    def query(begin, end):
        return vector([begin] + [end] * rows, device, torch.int64, True)
    def sources(slot):
        return vector([slot] + [0] * (rows - 1), device, torch.int64, True)
    cu, si = query(0, q), sources(2)
    counts = vector([1] + [0] * (rows - 1), device, torch.int64, True)
    mapping = vector([0] * rows, device, torch.int64, True)
    table = torch.tensor([[2, 3]] * rows, device=device, dtype=torch.int64)
    computed = vector([0] * rows, device, torch.int64, True)
    invoke(layer, x, cu, si, out)
    before = layer.raw.clone()
    rings = clone_records(layer.replay_records)
    verify_graph, commit_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(verify_graph):
        invoke(layer, x, cu, si, out)
    with torch.cuda.graph(commit_graph):
        ctx.commit(counts, si, cu, request_indices=mapping, block_table=table,
                   num_computed_tokens=computed, mamba_block_size=8,
                   token_capacity=x["mixed_qkv"].size(0))
    bad_queries = [(1, q, 2), (0, -1, 2), (0, q + 1, 2), (0, q + 2, 2),
                   (0, 2**32 + q, 2), (2**32, 2**32 + q, 2),
                   (0, q, 2**32 + 2), (0, q, 2**63 - 1)]
    for begin, end, slot in bad_queries:
        cu.copy_(query(begin, end))
        si.copy_(sources(slot))
        out.fill_(-11)
        if os.environ.get("PLA_SANITY_CHECK") == "1":
            try:
                invoke(layer, x, cu, si, out)
            except RuntimeError as error:
                assert "PLA_SANITY_CHECK" in str(error)
            else:
                raise AssertionError("invalid verify metadata accepted")
        verify_graph.replay()
        commit_graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, torch.full_like(out, -11))
        assert torch.equal(layer.raw, before)
        check_plan(ctx.plan, [[0] * rows] * 4, device)
        for name in ("rawv", "rawk", "g", "beta", "conv"):
            assert torch.equal(getattr(layer.replay_records, name), getattr(rings, name))
    cu.copy_(query(0, q))
    si.copy_(sources(2))
    for row, position, destination in ((-1, 0, 2), (2**32, 0, 2),
                                       (0, -1, 2), (0, 2**63 - 1, 2), (0, 0, 2**32)):
        mapping.fill_(row)
        computed.fill_(position)
        table.fill_(destination)
        if os.environ.get("PLA_SANITY_CHECK") == "1":
            try:
                ctx.commit(counts, si, cu, request_indices=mapping, block_table=table,
                           num_computed_tokens=computed, mamba_block_size=8,
                           token_capacity=x["mixed_qkv"].size(0))
            except RuntimeError as error:
                assert "PLA_SANITY_CHECK" in str(error)
            else:
                raise AssertionError("invalid commit metadata accepted")
        commit_graph.replay()
        torch.cuda.synchronize()
        check_plan(ctx.plan, [[0] * rows] * 4, device)
        assert torch.equal(layer.raw, before)


def case_rejected_verify_commit(device):
    """Keep stale records and rejected source owners out of graph commits."""
    torch.manual_seed(2026)
    layer = pages(8, 1, 2, "SD", device)
    allocate(layer, 2)
    ctx = context([layer], 2)
    x = inputs(1, 1, device)
    out = torch.empty(1, D, dtype=torch.bfloat16, device=device)
    cu, si, counts = vector([0, 1, 1], device), vector([2, 3], device), vector([1, 0], device)
    kwargs = dict(block_table=torch.tensor([[3], [4]], device=device),
                  num_computed_tokens=vector([0, 0], device), mamba_block_size=8,
                  token_capacity=x["mixed_qkv"].size(0))

    def run():
        invoke(layer, x, cu, si, out)
        ctx.commit(counts, si, cu, **kwargs)

    before = layer.raw.clone()
    run()
    layer.raw.copy_(before)
    records = clone_records(layer.replay_records)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Row 0 fits Q but exceeds the actual verify projection capacity.
    cu.copy_(vector([0, 2, 2], device))
    if os.environ.get("PLA_SANITY_CHECK") == "1":
        try:
            ctx.commit(counts, si, cu, **kwargs)
        except RuntimeError as error:
            assert "invalid query length" in str(error)
        else:
            raise AssertionError("sanity accepted an interval beyond verify capacity")
    out.fill_(-11)
    graph.replay()
    torch.cuda.synchronize()
    check_plan(ctx.plan, [[0, 0]] * 4, device)
    assert torch.equal(layer.raw, before)
    assert torch.equal(out, torch.full_like(out, -11))
    for name in ("rawv", "rawk", "g", "beta", "conv"):
        assert torch.equal(getattr(layer.replay_records, name), getattr(records, name))

    # Row 1 is now rejected and must not claim row 0's destination slot.
    cu.copy_(vector([0, 1, 2], device))
    state, conv, expected = oracle_commit(layer, records, [0, 1, 1], [2, 3], [1, 0],
                                          block_table=[[3], [4]], computed=[0, 0], block_size=8)
    graph.replay()
    torch.cuda.synchronize()
    check_plan(ctx.plan, expected, device)
    close(layer.kv_cache[1], state, 5e-6)
    assert torch.equal(ds(layer), conv)


def case_diagnostics(device):
    """Run in a fresh process; assert snapshots and capture-safe metadata dumps."""
    import sys
    import tempfile
    bypass = os.environ.get("PLA_BYPASS_EXEC") == "1"
    show = os.environ.get("PPU_LIB_SHOW_PARAMS") == "1"
    layer = pages(8, 1, 2, "SD", device)
    allocate(layer, 1)
    ctx = context([layer], 1)
    x = inputs(1, 2, device)
    out = torch.full((2, D), -11, dtype=torch.bfloat16, device=device)
    cu, si, counts = vector([0, 2], device, torch.int64, True), vector([2], device), vector([1], device)
    ctx._buffer.fill_(-13)
    ctx._candidate.fill_(-17)
    before = [t.clone() for t in (layer.raw, ctx._buffer, ctx._candidate)]
    rings = clone_records(layer.replay_records)
    # Flip after import, before the first call: import-time snapshots must win.
    os.environ["PLA_BYPASS_EXEC"] = "0" if bypass else "1"
    os.environ["PPU_LIB_SHOW_PARAMS"] = "0" if show else "1"
    os.environ["PPU_LIB_PERF_INSTRUMENT"] = "0"
    def run(output=out):
        result = invoke(layer, x, cu, si, output)
        ctx.commit(counts, si, cu, token_capacity=x["mixed_qkv"].size(0))
        return result
    try:
        with tempfile.TemporaryFile() as sink:
            saved = os.dup(1)
            try:
                sys.stdout.flush()
                os.dup2(sink.fileno(), 1)
                os.write(1, b"@@eager\n")
                run()
                torch.cuda.synchronize()
                if not bypass:
                    layer.raw.copy_(before[0])
                os.write(1, b"@@capture\n")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                os.write(1, b"@@replay\n")
                graph.replay()
                torch.cuda.synchronize()
            finally:
                sys.stdout.flush()
                os.dup2(saved, 1)
                os.close(saved)
            sink.seek(0)
            text = sink.read().decode()
        if show:
            eager, capture = text.split("@@capture\n")
            assert "cu_seqlens_q:[0,2]" in eager, text
            assert "cu_seqlens_q:" not in capture, text
            assert "cu_unavailable=during capture" in capture, text
            assert text.count("op=kda_replay_verify") == 2, text
            assert text.count("op=kda_replay_commit") == 2, text
            assert "instrument=HGTX" in text, text
        else:
            assert "[PLA_BYPASS]" not in text and "[KDA]" not in text, text
        if bypass:
            for got, expected in zip((layer.raw, ctx._buffer, ctx._candidate), before):
                assert torch.equal(got, expected)
            assert torch.equal(out, torch.full_like(out, -11))
            for name in ("rawv", "rawk", "g", "beta", "conv"):
                assert torch.equal(getattr(layer.replay_records, name), getattr(rings, name))
            # Deterministic allocation must not add an implicit fill to capture.
            deterministic = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                empty_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(empty_graph):
                    allocated = run(None)
                assert allocated.shape == out.shape and allocated.dtype == out.dtype
                empty_graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.use_deterministic_algorithms(deterministic)
        else:
            initial = SimpleNamespace(heads=1, q=2, layout="SD",
                kv_cache=(before[0].view(torch.bfloat16).as_strided(layer.kv_cache[0].shape,
                    layer.kv_cache[0].stride(), layer.kv_cache[0].storage_offset()),
                    before[0].view(torch.float32).as_strided(layer.kv_cache[1].shape,
                    layer.kv_cache[1].stride(), layer.kv_cache[1].storage_offset())))
            want, golden = oracle_verify(initial, x, [0, 2], [2], rings, out)
            close(out, want)
            for name in ("rawv", "rawk", "g", "beta"):
                close(getattr(layer.replay_records, name), getattr(golden, name),
                      5e-6 if name in ("g", "beta") else 6e-3)
            assert torch.equal(layer.replay_records.conv, golden.conv)
            state, conv, expected = oracle_commit(initial, layer.replay_records, [0, 2], [2], [1])
            check_plan(ctx.plan, expected, device)
            close(layer.kv_cache[1][2], state[2], 5e-6)
            assert torch.equal(ds(layer), conv)
            for slot in set(range(8)) - {2}:
                assert torch.equal(layer.kv_cache[1][slot], state[slot])
    finally:
        os.environ["PLA_BYPASS_EXEC"] = "1" if bypass else "0"
        os.environ["PPU_LIB_SHOW_PARAMS"] = "1" if show else "0"


def diagnostic_process(config, device):
    import subprocess
    import sys
    bypass, show, perf = config
    env = dict(os.environ, PLA_BYPASS_EXEC=str(bypass), PPU_LIB_SHOW_PARAMS=str(show),
               PPU_LIB_PERF_INSTRUMENT=str(perf), PLA_SANITY_CHECK="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             "--phase=diagnostics", "--device", str(device)],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    print("Fresh-process diagnostics passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "functional", "verify", "commit", "precision", "graph", "tp", "diagnostics"), default="all")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.distributed:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        assert world >= 2, "distributed mode requires at least two ranks"
        rep = CaseReporter("kda_replayssm_vllm", "distributed_contract", 1)
        if rank == 0:
            rep.header()
            rep.start(f"distributed_TP{world}")
        try:
            case_tp(device, world, rank)
            torch.distributed.barrier()
            if rank == 0:
                rep.ok()
                return rep.summary()
            return 0
        finally:
            torch.distributed.destroy_process_group()
    device = torch.device(args.device)
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    if args.oracle_only:
        torch.manual_seed(1)
        layer = pages(8, 1, 2, "SD", device)
        records = blank_records(1, 1, 2, device)
        x = inputs(1, 2, device)
        out, golden = oracle_verify(layer, x, [0, 2], [2], records,
                                   torch.zeros(2, D, device=device))
        state, conv, plan = oracle_commit(layer, golden, [0, 2], [2], [0])
        assert torch.equal(state, layer.kv_cache[1]) and torch.equal(conv, ds(layer))
        assert plan[0] == [0] and torch.isfinite(out).all()
        state, conv, plan = oracle_commit(layer, golden, [0, 2], [2], [2],
                                          block_table=[[3, 4]], computed=[7], block_size=8)
        assert plan == [[2], [4], [3], [1]]
        close(state[3], folded(layer.kv_cache[1][2], golden, 0, 1), 0)
        assert torch.equal(conv[3, :, :3], golden.conv[0, 0])
        assert torch.equal(conv[4, :, :3], golden.conv[0, 1])
        print("Oracle self-check passed")
        return 0
    from pla.decode.kda_replayssm_vllm import KDAReplayRecords  # fail loudly when unbuilt
    cases = []
    if args.phase in ("all", "functional", "verify"):
        for layout, q, h, dtype in (("DS", 1, 1, torch.int32), ("SD", 2, 3, torch.int64),
                                    ("DS", 5, 12, torch.int64), ("SD", 8, 1, torch.int32),
                                    ("DS", 32, 1, torch.int32)):
            cases.append((f"verify_{layout}_Q{q}_H{h}", lambda l=layout, n=q, h=h, d=dtype:
                          case_verify(device, l, n, h, d)))
        for layout, dtype, null in (("DS", torch.int32, 0), ("SD", torch.int64, -1)):
            cases.append((f"verify_{layout}_Q4_native", lambda l=layout, d=dtype, n=null:
                          case_verify(device, l, 4, 12, d, rows=128, null_block_id=n)))
        cases.append(("verify_Q4_unaligned_weight", lambda:
                      case_verify(device, "DS", 4, 12, torch.int64, rows=128, misaligned=True)))
        cases.append(("empty", lambda: case_empty(device)))
        cases.append(("all_invalid_and_N0", lambda: case_all_invalid(device)))
        cases.append(("invalid", lambda: case_invalid(device)))
        cases.append(("destination_conflicts", lambda: case_conflicts(device)))
        cases.append(("null_minus_one", lambda: case_null_minus_one(device)))
        cases.append(("device_metadata", lambda: case_device_metadata(device)))
        cases.append(("device_metadata_Q4_native", lambda:
                      case_device_metadata(device, q=4, heads=12, rows=128)))
    if args.phase in ("all", "functional", "verify", "precision"):
        for generic in (False, True):
            tag = "generic" if generic else "Q4"
            if args.phase != "functional":
                cases.append(("precision_cancellation_" + tag,
                              lambda g=generic: case_precision_verify(device, g)))
            cases.append(("precision_finite_gate_overflow_" + tag,
                          lambda g=generic: case_precision_verify(device, g, overflow=True)))
            if args.phase != "functional":
                cases.append(("precision_state_projection_" + tag,
                              lambda g=generic: case_precision_projection(device, g)))
    if args.phase in ("all", "commit", "precision"):
        cases.append(("precision_commit_cancellation", lambda: case_precision_commit(device)))
    if args.phase in ("all", "graph", "precision"):
        cases.append(("precision_graph_metadata", lambda: case_precision_graph(device)))
    if args.phase in ("all", "functional", "commit"):
        for batch in (31, 32, 33, 129):
            cases.append((f"parallel_conflicts_B{batch}", lambda b=batch:
                          case_parallel_conflicts(device, b)))
            cases.append((f"noalign_conflicts_B{batch}", lambda b=batch:
                          case_parallel_conflicts(device, b, align=False)))
        for layout, history in (("DS", 3), ("DS", 6), ("SD", 3)):
            cases.append((f"conv_{layout}_history{history}", lambda l=layout, h=history:
                          case_conv_packed(device, l, h)))
        for layout in ("DS", "SD"):
            cases.append(("context_cache_overlap_" + layout,
                          lambda l=layout: case_context_cache_overlap(device, l)))
            cases.append(("live_pipeline_" + layout,
                          lambda l=layout: case_commit(device, l, live=True)))
            for align in (False, True):
                cases.append((f"multiround_{layout}_align{align}",
                              lambda l=layout, a=align: case_commit(device, l, a)))
        cases.append(("mixed_prefill_spec", lambda: case_mixed_prefill(device)))
        for q, layout in ((1, "DS"), (2, "SD"), (8, "DS"), (32, "SD")):
            cases.append((f"align_Q{q}_{layout}",
                          lambda n=q, l=layout: case_align_shapes(device, n, l)))
        for layout in ("DS", "SD"):
            cases.append(("prefix_continuation_" + layout,
                          lambda l=layout: case_prefix_continuation(device, l)))
    if args.phase in ("all", "functional", "graph"):
        cases.append(("graph_rejected_verify_commit", lambda: case_rejected_verify_commit(device)))
        for phase in ("verify", "commit", "combined"):
            cases.append(("graph_" + phase, lambda p=phase: case_graph(device, p)))
            cases.append(("graph_Q4_native_" + phase, lambda p=phase:
                          case_graph(device, p, q=4, heads=12, rows=128)))
            if phase != "verify":
                cases.append(("graph_Q4_noalign_" + phase, lambda p=phase:
                              case_graph(device, p, q=4, heads=12, rows=128, align=False)))
    if args.phase in ("all", "functional", "tp"):
        for tp in (1, 2, 4):
            cases.append((f"head_shard_TP{tp}", lambda t=tp: case_tp(device, t)))
    if args.phase == "diagnostics":
        cases.append(("diagnostics", lambda: case_diagnostics(device)))
    elif args.phase in ("all", "functional") and not args.skip_diagnostics:
        for config in ((1, 0, 0), (1, 1, 1), (0, 1, 1), (0, 0, 1)):
            cases.append((f"diagnostics_{config}", lambda c=config: diagnostic_process(c, device)))
    rep = CaseReporter("kda_replayssm_vllm", "contract", len(cases))
    rep.header()
    for name, call in cases:
        rep.start(name)
        with rep.guard():
            call()
            rep.ok()
        if rep.aborted:
            break
    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(main())
