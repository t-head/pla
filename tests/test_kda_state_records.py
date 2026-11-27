# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Optional PLA-native compact/correction records, using the existing FP64 oracle."""
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

import test_kda_replayssm_vllm as base
from pla.decode.kda_replayssm_vllm import KDACompactRecords, KDACorrectionRecords


FORMATS = (KDACompactRecords, KDACorrectionRecords)


def buffers(records):
    return [getattr(records, field.name) for field in fields(records)]


def layer_with_records(record_type, heads, steps, layout, rows=7, slots=16):
    layer = base.pages(slots, heads, steps, layout, "cuda")
    layer.replay_records = record_type.allocate(rows, heads, steps, "cuda")
    for tensor in buffers(layer.replay_records):
        tensor.fill_(-7)
    return layer


def oracle(layer, x, offsets, slots, out):
    records = base.blank_records(len(slots), layer.heads, layer.q, "cuda")
    records = SimpleNamespace(**{name: value.double() for name, value in vars(records).items()})
    expected_out, records = base.oracle_verify(layer, x, offsets, slots, records, out)
    prefixes, corrections = {}, {}
    keys = records.rawk / torch.sqrt(records.rawk.square().sum(-1, keepdim=True) + 1e-6)
    decays = records.g.exp()
    for row, slot in enumerate(slots):
        length = offsets[row + 1] - offsets[row]
        if slot <= 0 or not length:
            continue
        state = layer.kv_cache[1][slot].double().clone()
        prefixes[row] = [state.clone()]
        corrections[row] = []
        for step in range(length):
            key = keys[row, :, step]
            state = state * decays[row, :, step].unsqueeze(-2)
            delta = (records.rawv[row, :, step] -
                     (state * key.unsqueeze(-2)).sum(-1)) * records.beta[row, :, step, None]
            corrections[row].append(delta)
            state = state + delta.unsqueeze(-1) * key.unsqueeze(-2)
            prefixes[row].append(state.clone())
    return expected_out, records, prefixes, corrections, keys, decays


def check_committed(layer, before, expected_state, expected_conv):
    base.close(layer.kv_cache[1], expected_state, 5e-6)
    assert torch.equal(base.ds(layer), expected_conv.bfloat16())
    # Substitute actual allowed writes into the original backing page. Every
    # other byte, including unused history and page padding, must be unchanged.
    allowed = before.clone()
    for tensor in layer.kv_cache:
        view = allowed.view(tensor.dtype).as_strided(
            tensor.shape, tensor.stride(), tensor.storage_offset())
        view.copy_(tensor)
    assert torch.equal(layer.raw, allowed), "commit changed page padding"


@pytest.mark.parametrize("record_type", FORMATS)
@pytest.mark.parametrize("layout", ("DS", "SD"))
@pytest.mark.parametrize("heads,steps", ((1, 1), (3, 3), (6, 5), (16, 8), (2, 13), (5, 32)))
def test_every_prefix(record_type, layout, heads, steps):
    torch.manual_seed(261126 + heads + steps)
    layer = layer_with_records(record_type, heads, steps, layout)
    lengths = [steps, max(steps - 2, 1), 0, 1, 0, 0, 0]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    slots = [2, 4, 5, 0, -1, 0, 0]
    x = base.inputs(heads, offsets[-1] + 2, "cuda")
    initial = layer.raw.clone()
    projection = x["storage"].clone()
    out = torch.full((x["mixed_qkv"].shape[0], heads * 128), -11,
                     device="cuda", dtype=torch.bfloat16)
    expected_out, gold, prefixes, deltas, keys, decays = oracle(layer, x, offsets, slots, out)
    cu = base.vector(offsets, "cuda", torch.int64, strided=True)
    si = base.vector(slots, "cuda", torch.int64, strided=True)
    base.invoke(layer, x, cu, si, out)
    base.close(out[:offsets[-1]], expected_out[:offsets[-1]])
    assert torch.equal(out[offsets[-1]:], torch.full_like(out[offsets[-1]:], -11))
    assert torch.equal(layer.raw, initial), "verify changed persistent pages"
    assert torch.equal(x["storage"], projection)
    records = layer.replay_records
    for row, (slot, length) in enumerate(zip(slots, lengths)):
        if slot > 0 and length:
            base.close(records.key[row, :, :length], keys[row, :, :length], 5e-6)
            base.close(records.decay[row, :, :length], decays[row, :, :length], 5e-6)
            assert torch.equal(records.conv[row, :length], gold.conv[row, :length].bfloat16())
            if record_type is KDACompactRecords:
                base.close(records.value[row, :, :length], gold.rawv[row, :, :length], 5e-6)
                base.close(records.beta[row, :, :length], gold.beta[row, :, :length], 5e-6)
                base.close(records.terminal_state[row], prefixes[row][-1], 5e-6)
            else:
                base.close(records.correction[row, :, :length], torch.stack(deltas[row], 1), 5e-6)
        for field in fields(records):
            tensor = getattr(records, field.name)
            if slot <= 0 or not length:
                suffix = tensor[row]
            elif field.name == "terminal_state":
                continue
            elif field.name == "conv":
                suffix = tensor[row, length:]
            else:
                suffix = tensor[row, :, length:]
            assert torch.equal(suffix, torch.full_like(suffix, -7)), field.name

    ctx = base.context([layer], len(slots))
    snapshots = [tensor.clone() for tensor in buffers(records)]
    for accepted in range(-1, steps + 3):
        layer.raw.copy_(initial)
        counts = [accepted, accepted - 1, steps + 7, steps + 7, 0, 0, 0]
        state, conv, plan = base.oracle_commit(
            layer, gold, offsets, slots, counts,
            fold_state=lambda _, __, row, n: prefixes[row][n].float())
        ctx.commit(base.vector(counts, "cuda", torch.int64, strided=True), si, cu,
                   token_capacity=x["mixed_qkv"].shape[0])
        base.check_plan(ctx.plan, plan, "cuda")
        check_committed(layer, initial, state, conv)
        if accepted <= 0:
            assert torch.equal(layer.raw, initial), "zero acceptance changed pages"
    for tensor, snapshot in zip(buffers(records), snapshots):
        assert torch.equal(tensor, snapshot), "commit changed immutable records"


@pytest.mark.parametrize("record_type", FORMATS)
@pytest.mark.parametrize("heads,steps", ((6, 1), (16, 5), (3, 13)))
@pytest.mark.parametrize("weight_offset", (0, 1))
def test_dynamic_graph_paged_commit(record_type, heads, steps, weight_offset):
    torch.manual_seed(261127 + heads + steps)
    rows = 64
    layers = [layer_with_records(record_type, heads, steps, "SD", rows=rows) for _ in range(2)]
    inputs = [base.inputs(heads, 4 * steps + 2, "cuda") for _ in layers]
    for x in inputs:
        weight = x["conv_weight"]
        storage = torch.empty(weight.numel() + weight_offset, device="cuda", dtype=weight.dtype)
        shifted = storage[weight_offset:].view_as(weight)
        shifted.copy_(weight)
        x["conv_weight"] = shifted
        assert shifted.is_contiguous() and shifted.data_ptr() % 16 == 4 * weight_offset
    originals = [layer.raw.clone() for layer in layers]
    outs = [torch.full((4 * steps + 2, heads * 128), -11, device="cuda",
                       dtype=torch.bfloat16) for _ in layers]
    cu = base.vector([0] * (rows + 1), "cuda", torch.int64, strided=True)
    si = base.vector([0] * rows, "cuda", torch.int64, strided=True)
    sampled = base.vector([0] * 4, "cuda", torch.int64, strided=True)
    mapping = base.vector([2, 0, 1, 3] + [0] * (rows - 4), "cuda", torch.int64)
    blocks = torch.zeros((4, 3), device="cuda", dtype=torch.int64)
    computed = base.vector([0] * 4, "cuda", torch.int64)
    block_size = max(steps, 4)
    ctx = base.context(layers, rows)

    def run():
        for layer, x, out in zip(layers, inputs, outs):
            base.invoke(layer, x, cu, si, out)
        ctx.commit(sampled, si, cu, request_indices=mapping, block_table=blocks,
                   num_computed_tokens=computed, mamba_block_size=block_size,
                   token_capacity=4 * steps + 2)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for turn in range(4):
        lengths = [steps if turn != 1 else 1, max(steps - 1, 1), 0, 1]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        offsets += [offsets[-1]] * (rows - 4)
        slots = [2 + turn % 2, 4 + turn % 2, 6, 0] + [0] * (rows - 4)
        counts = ([0, 0, 0, 0], [1, 0, 1, steps],
                  [steps + 3, 0, steps + 3, steps], [-2, 0, -1, 0])[turn]
        table = [[slots[1], 9, 13], [6, 10, 14], [slots[0], 8, 12], [0, 11, 15]]
        positions = [block_size - 1, 0, block_size - 2, 0]
        cu.copy_(base.vector(offsets, "cuda", torch.int64))
        si.copy_(base.vector(slots, "cuda", torch.int64))
        sampled.copy_(base.vector(counts, "cuda", torch.int64))
        blocks.copy_(torch.tensor(table, device="cuda", dtype=torch.int64))
        computed.copy_(base.vector(positions, "cuda", torch.int64))
        expected = []
        for layer, initial, x, out in zip(layers, originals, inputs, outs):
            layer.raw.copy_(initial)
            out.fill_(-11)
            want, gold, prefixes, _, _, _ = oracle(layer, x, offsets, slots, out)
            state, conv, plan = base.oracle_commit(
                layer, gold, offsets, slots, counts, mapping=[2, 0, 1, 3] + [0] * (rows - 4),
                block_table=table, computed=positions, block_size=block_size,
                fold_state=lambda _, __, row, n: prefixes[row][n].float())
            expected.append((want, state, conv, plan))
        graph.replay()
        for layer, initial, out, (want, state, conv, plan) in zip(layers, originals, outs, expected):
            base.close(out[:offsets[-1]], want[:offsets[-1]])
            assert torch.equal(out[offsets[-1]:], torch.full_like(out[offsets[-1]:], -11))
            check_committed(layer, initial, state, conv)
            base.check_plan(ctx.plan, plan, "cuda")


@pytest.mark.parametrize("record_type", FORMATS)
def test_rejects_aliasing_and_wrong_dtype(record_type):
    layer = layer_with_records(record_type, 3, 5, "DS", rows=16)
    records = layer.replay_records
    layer.replay_records = replace(records, key=records.key.bfloat16())
    with pytest.raises(RuntimeError, match="shape/dtype"):
        base.context([layer], 16)
    if record_type is KDACompactRecords:
        state = layer.kv_cache[1]
        terminal = state.as_strided(state.shape, (3 * 128 * 128, 128 * 128, 128, 1))
        layer.replay_records = replace(records, terminal_state=terminal)
    else:
        layer.replay_records = replace(records, correction=records.key)
    with pytest.raises(RuntimeError, match="alias"):
        base.context([layer], 16)
