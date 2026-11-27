# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent FP64 reference for BF16 CP reuse and its mandatory fallbacks."""
import pytest
import torch

from pla.prefill.flashkdapro import flashkda_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="requires TC02 PPU",
)


def _offsets(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _normalize_qk(q, k):
    q, k = q.double(), k.double()
    q *= torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1]**-.5
    k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    return q, k


def _reference(q, k, v, initial, offsets, decay, beta=.5):
    """Independent FP64 token recurrence; inputs are normalized CPU tensors."""
    state = initial.double().clone()
    out = torch.empty_like(v)
    for seq, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        for token in range(begin, end):
            state[seq] *= decay if decay.ndim == 0 else decay[token, :, None, :]
            weight = beta[token, :, None] if isinstance(beta, torch.Tensor) and beta.ndim else beta
            delta = (v[token] - (state[seq] * k[token, :, None, :]).sum(-1)) * weight
            state[seq] += delta[:, :, None] * k[token, :, None, :]
            out[token] = (state[seq] * q[token, :, None, :]).sum(-1)
    return out, state


def _assert_relative(actual, expected, dims):
    assert torch.isfinite(actual).all()
    error = ((actual - expected).square().mean(dims).sqrt()
             / expected.square().mean(dims).sqrt().clamp_min(1e-12))
    assert torch.all(error < .025), error


@pytest.mark.parametrize("resident", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("storage", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("structured", [False, True])
def test_radix_checkpoints(resident, storage, index_dtype, structured):
    """Snapshots are request-local, write-only and independent of final storage."""
    torch.manual_seed(437)
    lengths, heads = [0, 3, 65, 130, 64, 129], 3
    offsets = _offsets(lengths)
    tokens = offsets[-1]
    q, k, v = [torch.randn(tokens, heads, 128, dtype=torch.bfloat16) * .2 for _ in range(3)]
    if structured:
        k[:] = k[0].clone()
    initial = (torch.randn(len(lengths), heads, 128, 128) * .1).to(storage)
    query, key = _normalize_qk(q, k)
    values = v.double()
    q, k, v = [x.cuda() for x in (q, k, v)]
    state = initial.cuda()
    gate, beta = torch.full_like(q, -2.), torch.zeros(tokens, heads, device="cuda", dtype=torch.bfloat16)
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, 128, device="cuda")
    output, final = torch.empty_like(v), torch.empty_like(state)
    track = torch.full(state.shape, 19., device="cuda", dtype=torch.float32)
    chunks = [0, -1, 1, 2, 1, 2**40 if index_dtype == torch.int64 else 2**30]
    idx = torch.tensor(chunks, device="cuda", dtype=index_dtype)
    kwargs = dict(initial_state=state, final_state=final,
                  cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                  state_accum_dtype=resident, track_state=track, track_chunk_idx=idx)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    torch.cuda.synchronize()
    torch.testing.assert_close(state.cpu(), initial, atol=0, rtol=0)
    torch.testing.assert_close(track[0].cpu(), initial[0].float(), atol=0, rtol=0)
    for seq in [1, 4, 5]:
        assert torch.all(track[seq] == 19.)
    decay = torch.exp(-5. * torch.sigmoid(torch.tensor(-2., dtype=torch.float64)))
    for seq, boundary in [(2, 64), (3, 128)]:
        start = offsets[seq]
        _, expected = _reference(query[start:start + boundary], key[start:start + boundary],
            values[start:start + boundary], initial[seq:seq+1], [0, boundary], decay)
        _assert_relative(track[seq:seq+1].cpu().double(), expected, (2, 3))
        # A prefix ending here publishes the very same resident carry. Reading
        # the tracking destination must never affect outputs or final state.
        prefix_final = torch.empty_like(state[seq:seq+1])
        prefix_track = torch.empty_like(track[seq:seq+1])
        prefix_idx = torch.full((1,), -1, device="cuda", dtype=index_dtype)
        prefix_out = torch.empty_like(v[start:start+boundary])
        flashkda_fwd(q[start:start+boundary], k[start:start+boundary], v[start:start+boundary],
            gate[start:start+boundary], beta[start:start+boundary], alog, bias, prefix_out,
            initial_state=state[seq:seq+1], final_state=prefix_final,
            state_accum_dtype=resident, track_state=prefix_track, track_chunk_idx=prefix_idx)
        torch.testing.assert_close(track[seq:seq+1].to(storage), prefix_final, atol=0, rtol=0)
    saved_out, saved_final = output.clone(), final.clone()
    track.fill_(-73.)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    torch.testing.assert_close(output, saved_out, atol=0, rtol=0)
    torch.testing.assert_close(final, saved_final, atol=0, rtol=0)


@pytest.mark.parametrize("resident", [torch.float32, torch.bfloat16])
def test_radix_graph_pool(resident):
    """Graph replay reads live chunk IDs, offsets and pool slots, including empty rows."""
    torch.manual_seed(438)
    heads, tokens = 3, 130
    q, k, v = [torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16) * .2 for _ in range(3)]
    gate, beta = torch.full_like(q, -2.), torch.zeros(tokens, heads, device="cuda", dtype=torch.bfloat16)
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, 128, device="cuda")
    original = torch.randn(4, heads, 128, 128, device="cuda", dtype=torch.bfloat16) * .1
    pool, output = original.clone(), torch.empty_like(v)
    track = torch.empty(2, heads, 128, 128, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, 65, tokens], device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    idx = torch.tensor([1, 0], device="cuda", dtype=torch.int64)
    kwargs = dict(state_pool=pool, state_indices=slots, cu_seqlens=cu,
                  state_accum_dtype=resident, track_state=track, track_chunk_idx=idx)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    for cut, indices in [(65, [1, 0]), (0, [0, 2]), (130, [-1, 0]), (64, [1, -2])]:
        pool.copy_(original)
        track.fill_(31.)
        cu.copy_(torch.tensor([0, cut, tokens], device="cuda", dtype=torch.int32))
        idx.copy_(torch.tensor(indices, device="cuda", dtype=torch.int64))
        slots.copy_(torch.tensor([1, 3], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        actual = [x.clone() for x in (output, pool, track)]
        pool.copy_(original)
        track.fill_(31.)
        flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
        for value, expected in zip((output, pool, track), actual):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        torch.testing.assert_close(pool[[0, 2]], original[[0, 2]], atol=0, rtol=0)
        for seq, (length, chunk, slot) in enumerate(zip([cut, tokens-cut], indices, [1, 3])):
            if chunk == 0:
                torch.testing.assert_close(track[seq], original[slot].float(), atol=0, rtol=0)
            elif chunk < 0 or chunk * 64 >= length:
                assert torch.all(track[seq] == 31.)


@pytest.mark.parametrize("resident", [torch.float32, torch.bfloat16])
def test_radix_empty_and_validation(resident):
    heads = 3
    q = torch.empty(0, heads, 128, device="cuda", dtype=torch.bfloat16)
    beta = torch.empty(0, heads, device="cuda", dtype=torch.bfloat16)
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, 128, device="cuda")
    track = torch.full((2, heads, 128, 128), 19., device="cuda")
    idx = torch.tensor([0, -1], device="cuda", dtype=torch.int32)
    cu = torch.zeros(3, device="cuda", dtype=torch.int32)
    final = torch.full_like(track, 7.)
    kwargs = dict(cu_seqlens=cu, state_accum_dtype=resident,
                  track_state=track, track_chunk_idx=idx)
    flashkda_fwd(q, q, q, q, beta, alog, bias, q, final_state=final, **kwargs)
    assert torch.all(track[0] == 0.)
    assert torch.all(track[1] == 19.)
    assert torch.all(final == 0.)
    for bad in [dict(track_chunk_idx=None), dict(track_state=track.bfloat16()),
                dict(track_chunk_idx=idx.float()), dict(track_state=track.transpose(2, 3)),
                dict(track_state=track[:1]), dict(track_state=final)]:
        with pytest.raises((ValueError, RuntimeError)):
            flashkda_fwd(q, q, q, q, beta, alog, bias, q, final_state=final,
                         **(kwargs | bad))


@pytest.mark.parametrize("heads,resident", [(6, torch.float32), (16, torch.bfloat16)])
@pytest.mark.parametrize("gate_value", [-2., -15.])
def test_radix_long_prefix(heads, resident, gate_value):
    """Long, weak-decay chains export their actual carry at a late boundary."""
    torch.manual_seed(440)
    tokens, boundary = 4097, 4032
    q, k, v = [torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16) * .2 for _ in range(3)]
    gate, beta = torch.full_like(q, gate_value), torch.zeros(tokens, heads, device="cuda", dtype=torch.bfloat16)
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, 128, device="cuda")
    track = torch.empty(1, heads, 128, 128, device="cuda")
    idx = torch.tensor([boundary // 64], device="cuda", dtype=torch.int32)
    output = torch.empty_like(v)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output,
        state_accum_dtype=resident, track_state=track, track_chunk_idx=idx)
    prefix = torch.empty_like(track)
    prefix_out = torch.empty_like(v[:boundary])
    flashkda_fwd(q[:boundary], k[:boundary], v[:boundary], gate[:boundary], beta[:boundary],
        alog, bias, prefix_out, final_state=prefix, state_accum_dtype=resident,
        track_state=torch.empty_like(track), track_chunk_idx=torch.full_like(idx, -1))
    torch.testing.assert_close(track, prefix, atol=0, rtol=0)
    torch.testing.assert_close(output[:boundary], prefix_out, atol=0, rtol=0)


@pytest.mark.parametrize("input_dtype,output_dtype", [(torch.float32, torch.bfloat16),
                                                       (torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("tracking", [False, True])
def test_mixed_state_storage(input_dtype, output_dtype, tracking):
    torch.manual_seed(439)
    heads, tokens = 3, 65
    q, k, v = [torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16) * .2 for _ in range(3)]
    gate, beta = torch.full_like(q, -2.), torch.zeros(tokens, heads, device="cuda", dtype=torch.bfloat16)
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, 128, device="cuda")
    initial = torch.randn(4, heads, 128, 128, device="cuda").to(input_dtype) * .1
    final = torch.full_like(initial, 17., dtype=output_dtype)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 0, tokens], device="cuda", dtype=torch.int32)
    output, reference = torch.empty_like(v), torch.empty_like(v)
    expected = torch.empty(2, heads, 128, 128, device="cuda")
    extra = dict(track_state=torch.empty_like(expected),
                 track_chunk_idx=torch.tensor([0, 1], device="cuda", dtype=torch.int32)) if tracking else {}
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, initial_state=initial,
        final_state=final, state_indices=slots, cu_seqlens=cu, **extra)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, reference,
        initial_state=initial[slots.long()].float(), final_state=expected,
        cu_seqlens=cu, **extra)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    torch.testing.assert_close(final[slots.long()], expected.to(output_dtype), atol=0, rtol=0)
    assert torch.all(final[[0, 2]] == 17.)


def _check_graph_lengths(tokens, cuts, state_dtype, resident_dtype):
    """Graph replay can move a request across the local staging boundary."""
    torch.manual_seed(231191)
    heads, dim = 3, 128
    q, k, v = [torch.randn(tokens, heads, dim, dtype=torch.bfloat16)*.2
               for _ in range(3)]
    initial = (torch.randn(2, heads, dim, dim)*.1).to(state_dtype)
    query, key = _normalize_qk(q, k)
    decay = torch.exp(-5.*torch.sigmoid(torch.tensor(-2., dtype=torch.float64)))
    values = v.double()
    q, k, v = [x.cuda() for x in (q, k, v)]
    state = initial.cuda()
    before = state.clone()
    gate = torch.full_like(q, -2.)
    beta = torch.full((tokens, heads), 32., device="cuda", dtype=torch.bfloat16)
    alog = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    output, final = torch.empty_like(v), torch.empty_like(state)
    cu = torch.tensor([0, cuts[0], tokens], device="cuda", dtype=torch.int32)
    kwargs = dict(initial_state=state, final_state=final, cu_seqlens=cu,
                  raw_cu_seqlens=(0, cuts[0], tokens), state_accum_dtype=resident_dtype)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    for cut in cuts:
        cu.copy_(torch.tensor([0, cut, tokens], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_state = _reference(query, key, values, initial, (0, cut, tokens), decay, 1.)
        for actual, reference, dims in [(output.cpu().double(), expected, (0, 2)),
                                         (final.cpu().double(), expected_state, (2, 3))]:
            _assert_relative(actual, reference, dims)
        torch.testing.assert_close(state, before, atol=0, rtol=0)


def _check_serial_graph(monkeypatch, seed, lengths, replays, heads, state_dtype, packed, resident_dtype):
    """Prepared tile offsets follow all live device lengths on Graph replay."""
    monkeypatch.setenv("PLA_FLASHKDA_CP", "0")
    monkeypatch.setenv("PLA_FLASHKDA_VALUE_GUARD", "0")
    torch.manual_seed(seed)
    dim, tokens = 128, sum(lengths)
    projection = torch.randn(tokens, 3, heads, dim, dtype=torch.bfloat16) * .2
    initial = (torch.randn(len(lengths), heads, dim, dim) * .1).to(state_dtype)
    query, key = _normalize_qk(projection[:, 0], projection[:, 1])
    values = projection[:, 2].double()
    decay = torch.exp(-5. * torch.sigmoid(torch.tensor(-2., dtype=torch.float64)))
    projection = projection.cuda()
    q, k, v = projection.unbind(1)
    if not packed:
        q, k, v = [x.contiguous() for x in (q, k, v)]
    gate = torch.full((tokens, heads, dim), -2., device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros(tokens, heads, device="cuda", dtype=torch.bfloat16)
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, dim, device="cuda")
    state = initial.cuda()
    before = state.clone()
    output, final = torch.empty_like(v), torch.empty_like(state)

    offsets = _offsets(lengths)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    kwargs = dict(initial_state=state, final_state=final, cu_seqlens=cu,
                  raw_cu_seqlens=tuple(offsets), state_accum_dtype=resident_dtype)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    for sizes in replays:
        current = _offsets(sizes)
        cu.copy_(torch.tensor(current, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_state = _reference(query, key, values, initial, current, decay)
        for actual, reference, dims in [(output.cpu().double(), expected, (0, 2)),
                                         (final.cpu().double(), expected_state, (2, 3))]:
            _assert_relative(actual, reference, dims)
        empty = torch.tensor(sizes, device="cuda") == 0
        torch.testing.assert_close(final[empty], state[empty], atol=0, rtol=0)
        torch.testing.assert_close(state, before, atol=0, rtol=0)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("raw_gate,lower_bound,initial_scale", [
    (-2., -5., .1),      # Converging state and precomputed-output reuse.
    (-12., -5., .1),     # Weak contraction keeps the full replay.
    (-2., -5., 2.**80),  # Live incoming state cannot match the short checkpoint.
    (8., -10., .1),      # Nonfinite inverse factors require precision repair.
])
def test_bf16_cp_replay_reference(monkeypatch, state_dtype, packed,
                                  raw_gate, lower_bound, initial_scale):
    monkeypatch.setenv("PLA_FLASHKDA_CP", "2")
    monkeypatch.setenv("PLA_FLASHKDA_CP_MULT", "2")
    monkeypatch.setenv("PLA_FLASHKDA_CP_BV", "128")
    monkeypatch.setenv("PLA_FLASHKDA_VALUE_GUARD", "0")
    torch.manual_seed(231129)
    lengths, heads, dim = (0, 2048, 1539), 3, 128
    offsets = (0, 0, lengths[1], sum(lengths))
    total, padding = offsets[-1], 7
    projection = torch.randn(total+padding, 3, heads, dim, dtype=torch.bfloat16)*.2
    projection[total:] = float("nan")
    gate = torch.full((total+padding, heads, dim), raw_gate, dtype=torch.bfloat16)
    beta = torch.zeros(total+padding, heads, dtype=torch.bfloat16)
    gate[total:], beta[total:] = float("nan"), float("nan")
    initial = (torch.randn(len(lengths), heads, dim, dim)*initial_scale).to(state_dtype)
    q, k = _normalize_qk(projection[:total, 0], projection[:total, 1])
    v = projection[:total, 2].double()
    decay = (lower_bound*torch.sigmoid(torch.tensor(raw_gate, dtype=torch.float64))).exp()
    expected_out, expected_state = _reference(q, k, v, initial, offsets, decay)

    projection = projection.cuda()
    q, k, v = projection.unbind(1)
    if not packed:
        q, k, v = [tensor.contiguous() for tensor in (q, k, v)]
    gate, beta = gate.cuda(), beta.cuda()
    pool = torch.randn(6, heads, dim, dim, device="cuda", dtype=state_dtype)
    indices = torch.tensor([4, 2, 3], device="cuda", dtype=torch.int32)
    pool[indices.long()] = initial.cuda()
    before = pool.clone()
    out = torch.full_like(gate, 9.)
    a_log, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, dim, device="cuda")
    kwargs = dict(cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                  raw_cu_seqlens=offsets, state_pool=pool, state_indices=indices,
                  state_accum_dtype=torch.bfloat16, lower_bound=lower_bound)
    flashkda_fwd(q, k, v, gate, beta, a_log, bias, out, **kwargs)
    pool.copy_(before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, a_log, bias, out, **kwargs)
    for _ in range(2):
        pool.copy_(before)
        graph.replay()
    torch.cuda.synchronize()
    actual_state = pool[indices.long()].cpu().double()
    actual_out = out[:total].cpu().double()
    for seq, length in enumerate(lengths):
        pairs = [(actual_state[seq], expected_state[seq], (1, 2))]
        if length:
            sl = slice(offsets[seq], offsets[seq+1])
            pairs.append((actual_out[sl], expected_out[sl], (0, 2)))
        for actual, expected, dims in pairs:
            _assert_relative(actual, expected, dims)
    torch.testing.assert_close(pool[indices[0]], before[indices[0]], atol=0, rtol=0)
    torch.testing.assert_close(pool[[0, 1, 5]], before[[0, 1, 5]], atol=0, rtol=0)
    assert torch.count_nonzero(out[total:]).item() == 0


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("decay_exponent", [.02, .8, 1.2, 1.3, 1.4, 1.6, 2., 4., .55, .58, .6, .65])
@pytest.mark.parametrize("resident_dtype", [torch.float32, torch.bfloat16])
def test_correlated_keys_guard_reference(monkeypatch, state_dtype, decay_exponent, resident_dtype):
    """Correlated rows cross the safe-screen margin and the original risk gate."""
    import math
    monkeypatch.setenv("PLA_FLASHKDA_VALUE_GUARD", "0")
    torch.manual_seed(231163)
    tokens, heads, dim = 128, 3, 128
    key = torch.zeros(tokens, heads, dim, dtype=torch.bfloat16)
    key[:, :, 0] = math.sqrt(.75)
    for t in range(tokens): key[t, :, 1+t%16] = .5
    query = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)*.2
    value = torch.randn_like(query)*.2
    raw = torch.tensor(math.log(decay_exponent/(5.-decay_exponent)), dtype=torch.bfloat16)
    decay = torch.exp(-5.*torch.sigmoid(raw.double()))
    state = (torch.randn(1, heads, dim, dim)*.1).to(state_dtype)
    q, k = _normalize_qk(query, key)
    expected, states = _reference(q, k, value.double(), state, (0, tokens), decay)
    expected_state = states[0]
    query, key, value, state = [x.cuda() for x in [query, key, value, state]]
    gate = torch.full_like(query, float(raw))
    beta = torch.zeros(tokens, heads, device="cuda", dtype=torch.bfloat16)
    alog = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    output, final = torch.empty_like(value), torch.empty_like(state)
    kwargs = dict(initial_state=state, final_state=final,
                  state_accum_dtype=resident_dtype)
    if resident_dtype == torch.float32:
        kwargs.update(cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
                      raw_cu_seqlens=(0, tokens))
    flashkda_fwd(query, key, value, gate, beta, alog, bias, output, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(query, key, value, gate, beta, alog, bias, output, **kwargs)
    graph.replay(); torch.cuda.synchronize()
    for actual, reference, dims in [(output.cpu().double(), expected, (0, 2)),
                                     (final[0].cpu().double(), expected_state, (1, 2))]:
        _assert_relative(actual, reference, dims)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("storage_offset", [0, 1, 4, 8])
@pytest.mark.parametrize("lengths,heads,raw_gate", [
    ((17,), 24, -2.),
    ((1024, 1041, 2048), 3, -2.),
    ((1024, 1041, 2048), 3, -12.),
])
def test_bf16_initial_state_view_reference(monkeypatch, state_dtype, storage_offset,
                                          lengths, heads, raw_gate):
    """Contiguous state views may start at an unaligned storage address."""
    torch.manual_seed(231177)
    tokens, dim = sum(lengths), 128
    offsets = tuple(_offsets(lengths))
    if len(lengths) > 1:
        for name, value in (("CP", "2"), ("CP_MULT", "2"), ("CP_BV", "128"), ("VALUE_GUARD", "0")):
            monkeypatch.setenv("PLA_FLASHKDA_" + name, value)
    q, k, v = [torch.randn(tokens, heads, dim, dtype=torch.bfloat16)*.2
               for _ in range(3)]
    initial = (torch.randn(len(lengths), heads, dim, dim)*.1).to(state_dtype)
    query, key = _normalize_qk(q, k)
    decay = torch.exp(-5.*torch.sigmoid(torch.tensor(raw_gate, dtype=torch.float64)))
    expected, states = _reference(query, key, v.double(), initial, offsets, decay)

    storage = torch.full((initial.numel()+storage_offset,), 7., device="cuda",
                         dtype=state_dtype)
    state = storage[storage_offset:].view_as(initial)
    state.copy_(initial)
    before = storage.clone()
    q, k, v = [x.cuda() for x in (q, k, v)]
    gate = torch.full_like(q, raw_gate)
    beta = torch.zeros(tokens, heads, device="cuda", dtype=torch.bfloat16)
    alog = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    output, final = torch.empty_like(v), torch.empty_like(state)
    kwargs = dict(initial_state=state, final_state=final,
                  state_accum_dtype=torch.bfloat16,
                  cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                  raw_cu_seqlens=offsets)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    actual_out, actual_state = output.cpu().double(), final.cpu().double()
    for seq, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        _assert_relative(actual_out[begin:end], expected[begin:end], (0, 2))
        _assert_relative(actual_state[seq], states[seq], (1, 2))
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("resident_dtype", [torch.float32, torch.bfloat16])
def test_short_repair_cache_graph_lengths(state_dtype, resident_dtype):
    _check_graph_lengths(17, (3, 4, 1, 5), state_dtype, resident_dtype)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("resident_dtype", [torch.float32, torch.bfloat16])
def test_serial_repair_graph_lengths(state_dtype, resident_dtype):
    _check_graph_lengths(192, (96, 97, 1, 0, 191, 192), state_dtype, resident_dtype)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("storage_offset", [0, 1])
@pytest.mark.parametrize("resident_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 3, 17])
@pytest.mark.parametrize("alias", [False, True])
def test_serial_empty_state_bits(monkeypatch, state_dtype, storage_offset, resident_dtype, tokens, alias):
    """An empty request preserves storage bits across mutable Graph offsets."""
    monkeypatch.setenv("PLA_FLASHKDA_CP", "0")
    monkeypatch.setenv("PLA_FLASHKDA_VALUE_GUARD", "0")
    torch.manual_seed(231136)
    heads, dim = 1, 128
    if state_dtype == torch.float32:
        integer_dtype = torch.int32
        unsigned = [0x7fc00013, 0x7f800001, 0x7f800000, 0xff800000, 0x80000000, 0x00000001]
        width = 32
    else:
        integer_dtype = torch.int16
        unsigned = [0x7fc1, 0x7f81, 0x7f80, 0xff80, 0x8000, 0x0001]
        width = 16
    signed = [x if x < (1 << (width - 1)) else x - (1 << width) for x in unsigned]
    patterns = torch.tensor(signed, dtype=integer_dtype, device="cuda")
    elements = 2 * heads * dim * dim
    raw = torch.zeros(storage_offset + elements, dtype=integer_dtype, device="cuda")
    raw[storage_offset:storage_offset + dim * dim].copy_(
        patterns.repeat((dim * dim + len(signed) - 1) // len(signed))[:dim * dim])
    initial = raw[storage_offset:storage_offset + elements].view(state_dtype).view(2, heads, dim, dim)
    before = initial.view(torch.uint8).clone()
    q, k, v = [torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device="cuda") * .2
               for _ in range(3)]
    gate = torch.full_like(q, -2.)
    beta = torch.zeros(tokens, heads, dtype=torch.bfloat16, device="cuda")
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, dim, device="cuda")
    cu = torch.tensor([0, 0, tokens], device="cuda", dtype=torch.int32)
    output, final = torch.empty_like(v), torch.empty_like(initial)
    if alias:
        final = initial
    kwargs = dict(initial_state=initial, final_state=final, cu_seqlens=cu,
                  raw_cu_seqlens=(0, 0, tokens), state_accum_dtype=resident_dtype)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    for cut in (0, tokens, 0):
        initial.view(torch.uint8).copy_(before)
        cu.copy_(torch.tensor([0, cut, tokens], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        empty = 0 if cut == 0 else 1
        assert torch.equal(final[empty].view(torch.uint8), before[empty])
        if not alias:
            assert torch.equal(initial.view(torch.uint8), before)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("resident_dtype", [torch.float32, torch.bfloat16])
def test_serial_tile_prefix_graph(monkeypatch, state_dtype, packed, resident_dtype):
    lengths = [0 if i % 5 == 0 else 16 + i % 3 for i in range(64)]
    _check_serial_graph(monkeypatch, 231137, lengths, (lengths, lengths[::-1]), 1,
                        state_dtype, packed, resident_dtype)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("resident_dtype", [torch.float32, torch.bfloat16])
def test_serial_repair_cache_format_graph(monkeypatch, state_dtype, packed, resident_dtype):
    lengths = [1, 15, 17, 0]
    replays = (lengths, [17, 0, 1, 15], [0, 3, 16, 14],
               [0, 1, 2, 30], [0, 0, 0, 33], lengths)
    _check_serial_graph(monkeypatch, 231144, lengths, replays, 2,
                        state_dtype, packed, resident_dtype)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sequences", [65, 80, 129])
@pytest.mark.parametrize("resident_dtype", [torch.float32, torch.bfloat16])
def test_serial_mixed_batch_graph(monkeypatch, state_dtype, packed, sequences, resident_dtype):
    lengths = [0 if i % 5 == 0 else 1 + i % 33 for i in range(sequences)]
    short_mix = [1 + i % 3 for i in range(len(lengths) - 1)]
    short_mix.append(sum(lengths) - sum(short_mix))
    empty_mix = [0] * (len(lengths) - 1) + [sum(lengths)]
    replays = (lengths, lengths[::-1], short_mix, empty_mix, lengths)
    _check_serial_graph(monkeypatch, 231148, lengths, replays, 2,
                        state_dtype, packed, resident_dtype)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("raw_gate", [-1.0, -12.0])
@pytest.mark.parametrize("raw_beta", [0.0, 8.0])
@pytest.mark.parametrize("lengths", [(0, 65, 33), (257, 0, 161)])
@pytest.mark.parametrize("state_accum_dtype", [None, torch.bfloat16])
@pytest.mark.parametrize("indexed", [False, True])
def test_dense_weak_gate(state_dtype, raw_gate, raw_beta, lengths, state_accum_dtype, indexed):
    torch.manual_seed(231123)
    heads, dim = 5, 128
    total = sum(lengths)
    q, k, v = [torch.randn(total, heads, dim, dtype=torch.bfloat16) * .2 for _ in range(3)]
    gate = torch.full_like(q, raw_gate)
    beta = torch.full((total, heads), raw_beta, dtype=torch.bfloat16)
    a_log, dt_bias = torch.zeros(heads), torch.zeros(heads, dim)
    initial = torch.randn(len(lengths), heads, dim, dim).to(state_dtype) * .1
    initial_before = initial.clone()
    q64, k64 = _normalize_qk(q, k)
    v64 = v.double()
    decay = (-5. * gate.double().sigmoid()).exp()
    beta64 = beta.double().sigmoid()
    offsets = _offsets(lengths)
    expected_out, expected_state = _reference(q64, k64, v64, initial, offsets, decay, beta64)
    q, k, v, gate, beta, a_log, dt_bias, initial = [
        x.cuda() for x in (q, k, v, gate, beta, a_log, dt_bias, initial)]
    out, final = torch.empty_like(v), torch.empty_like(initial)
    kwargs = dict(cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                  raw_cu_seqlens=tuple(offsets), state_accum_dtype=state_accum_dtype)
    if indexed:
        pool = torch.randn(6, heads, dim, dim, device="cuda", dtype=state_dtype)
        indices = torch.tensor([4, 2, 3], device="cuda", dtype=torch.int32)
        pool[indices.long()] = initial
        before = pool.clone()
        kwargs.update(state_pool=pool, state_indices=indices)
        flashkda_fwd(q, k, v, gate, beta, a_log, dt_bias, out, **kwargs)
        pool.copy_(before)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flashkda_fwd(q, k, v, gate, beta, a_log, dt_bias, out, **kwargs)
        pool.copy_(before)
        graph.replay()
        final = pool[indices.long()]
        torch.testing.assert_close(pool[[0, 1, 5]], before[[0, 1, 5]], atol=0, rtol=0)
    else:
        flashkda_fwd(q, k, v, gate, beta, a_log, dt_bias, out,
                     initial_state=initial, final_state=final, **kwargs)
    torch.cuda.synchronize()
    # Match the existing random-prefill 2.5% RMS criterion for each sequence/head.
    actual_out, actual_state = out.cpu().double(), final.cpu().double()
    for seq, length in enumerate(lengths):
        pairs = [(actual_state[seq], expected_state[seq], (1, 2))]
        if length:
            sl = slice(offsets[seq], offsets[seq + 1])
            pairs.append((actual_out[sl], expected_out[sl], (0, 2)))
        for actual, expected, dims in pairs:
            _assert_relative(actual, expected, dims)
    for seq, length in enumerate(lengths):
        if not length:
            torch.testing.assert_close(final[seq].cpu(), initial_before[seq], atol=0, rtol=0)
    torch.testing.assert_close(initial.cpu(), initial_before, atol=0, rtol=0)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("raw_gate", [-12., -100.])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("value_guard", [False, True])
def test_bf16_precision_repair_layouts(monkeypatch, state_dtype, raw_gate, packed, cp, value_guard):
    """Weak decay and exact identities with projection views, padding and pools."""
    monkeypatch.setenv("PLA_FLASHKDA_CP", "2" if cp else "0")
    monkeypatch.setenv("PLA_FLASHKDA_VALUE_GUARD", "1" if value_guard else "0")
    torch.manual_seed(231125)
    lengths, heads, dim = (0, 1057, 231), 5, 128
    valid, padded = sum(lengths), sum(lengths) + 7
    if packed:
        projection = torch.zeros(padded, 3, heads, dim, device="cuda", dtype=torch.bfloat16)
        q, k, v = projection.unbind(1)
    else:
        q, k, v = [torch.zeros(padded, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q[..., 0] = 1
    g = torch.full_like(q, raw_gate, memory_format=torch.contiguous_format)
    beta = torch.full((padded, heads), -12., device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    out = torch.full_like(g, 9.)
    pool = torch.randn(6, heads, dim, dim, device="cuda").to(state_dtype) * .1
    before = pool.clone()
    indices = torch.tensor([4, 2, 3], device="cuda", dtype=torch.int32)
    offsets = (0, 0, 1057, valid)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    kwargs = dict(cu_seqlens=cu, raw_cu_seqlens=offsets, state_pool=pool,
                  state_indices=indices, state_accum_dtype=torch.bfloat16)
    flashkda_fwd(q, k, v, g, beta, a_log, bias, out, **kwargs)
    pool.copy_(before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, g, beta, a_log, bias, out, **kwargs)
    pool.copy_(before)
    graph.replay()
    torch.cuda.synchronize()
    actual_state, actual_out = pool[indices.long()].cpu().double(), out.cpu().double()
    initial = before[indices.long()].cpu().double()
    log_decay = -5 * torch.sigmoid(torch.tensor(raw_gate, dtype=torch.float64))
    for seq, length in enumerate(lengths):
        expected_state = initial[seq] * (log_decay * length).exp()
        _assert_relative(actual_state[seq], expected_state, (1, 2))
        if length:
            step = torch.arange(1, length + 1, dtype=torch.float64)
            expected_out = (initial[seq, :, :, 0][None] * (log_decay * step).exp()[:, None, None]
                            * dim ** -.5)
            actual = actual_out[offsets[seq]:offsets[seq + 1]]
            _assert_relative(actual, expected_out, (0, 2))
        if length == 0 or raw_gate == -100.:
            torch.testing.assert_close(actual_state[seq], initial[seq], atol=0, rtol=0)
    torch.testing.assert_close(pool[[0, 1, 5]], before[[0, 1, 5]], atol=0, rtol=0)
    assert torch.count_nonzero(out[valid:]).item() == 0


@pytest.mark.parametrize("sequences", [64, 79])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_bf16_large_ragged_pool(sequences, state_dtype):
    """Exercise the device tile prefix with empty rows, tails and projection views."""
    torch.manual_seed(231127)
    lengths = tuple((i * 13) % 34 for i in range(sequences))
    offsets = _offsets(lengths)
    total, heads, dim = offsets[-1], 1, 128
    projection = torch.randn(total + 7, 3, heads, dim, dtype=torch.bfloat16) * .2
    gate = torch.randn(total + 7, heads, dim, dtype=torch.bfloat16)
    beta = torch.randn(total + 7, heads, dtype=torch.bfloat16)
    projection[total:] = float("nan")
    gate[total:] = float("nan")
    beta[total:] = float("nan")
    q64, k64 = _normalize_qk(projection[:total, 0], projection[:total, 1])
    v64 = projection[:total, 2].double()
    decay = (-5 * gate[:total].double().sigmoid()).exp()
    beta64 = beta[:total].double().sigmoid()
    pool = (torch.randn(sequences + 3, heads, dim, dim, device="cuda") * .1).to(state_dtype)
    before = pool.clone()
    indices = torch.randperm(sequences + 3, device="cuda")[:sequences].int()
    initial = before[indices.long()].cpu().double()
    expected_out, expected_state = _reference(q64, k64, v64, initial, offsets, decay, beta64)
    projection = projection.cuda()
    q, k, v = projection.unbind(1)
    gate, beta = gate.cuda(), beta.cuda()
    a_log, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, dim, device="cuda")
    out = torch.empty_like(gate)
    kwargs = dict(cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                  raw_cu_seqlens=tuple(offsets), state_pool=pool, state_indices=indices,
                  state_accum_dtype=torch.bfloat16)
    flashkda_fwd(q, k, v, gate, beta, a_log, bias, out, **kwargs)
    pool.copy_(before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, a_log, bias, out, **kwargs)
    pool.copy_(before)
    graph.replay()
    torch.cuda.synchronize()
    actual_state, actual_out = pool[indices.long()].cpu().double(), out[:total].cpu().double()
    for seq, length in enumerate(lengths):
        pairs = [(actual_state[seq], expected_state[seq], (1, 2))]
        if length:
            sl = slice(offsets[seq], offsets[seq + 1])
            pairs.append((actual_out[sl], expected_out[sl], (0, 2)))
        for actual, expected, dims in pairs:
            _assert_relative(actual, expected, dims)
        if not length:
            torch.testing.assert_close(actual_state[seq], initial[seq], atol=0, rtol=0)
    unused = torch.ones(sequences + 3, device="cuda", dtype=torch.bool)
    unused[indices.long()] = False
    torch.testing.assert_close(pool[unused], before[unused], atol=0, rtol=0)
    assert torch.count_nonzero(out[total:]).item() == 0

@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("empty_first", [False, True])
@pytest.mark.parametrize("tokens", [256, 102400])
def test_paired_window_empty_state(monkeypatch, state_dtype, empty_first, tokens):
    """Both paired launches keep empty public state bits on Graph replay."""
    monkeypatch.setenv("PLA_FLASHKDA_CP", "0")
    monkeypatch.setenv("PLA_FLASHKDA_TC02_PAIRED", "1")
    torch.manual_seed(231219)
    heads, dim = 16, 128
    offsets = (0, 0 if empty_first else tokens, tokens)
    projection = torch.randn(tokens, 3, heads, dim, device="cuda", dtype=torch.bfloat16)*.2
    q, k, v = projection.unbind(1)
    gate, beta = torch.full_like(q, -2.), torch.zeros(tokens, heads, device="cuda", dtype=q.dtype)
    initial = torch.randn(2, heads, dim, dim, device="cuda", dtype=state_dtype)*.1
    word = torch.int32 if state_dtype == torch.float32 else torch.int16
    empty = 0 if empty_first else 1
    patterns = [0, -2147483648, 2143289345, 2139095040] if word == torch.int32 else [0, -32768, 32705, 32640]
    initial.view(word)[empty].reshape(-1)[:4] = torch.tensor(patterns, device="cuda", dtype=word)
    before = initial.view(word).clone()
    final = torch.full_like(initial, 7.)
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    alog, bias = torch.zeros(heads, device="cuda"), torch.zeros(heads, dim, device="cuda")
    kwargs = dict(initial_state=initial, final_state=final,
                  state_accum_dtype=torch.float32,
                  cu_seqlens=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                  raw_cu_seqlens=offsets)
    flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashkda_fwd(q, k, v, gate, beta, alog, bias, output, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(final.view(word)[empty], before[empty], atol=0, rtol=0)
    torch.testing.assert_close(initial.view(word), before, atol=0, rtol=0)
