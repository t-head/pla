# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 recurrence and fixtures, NOT extracted public-API upstream UTs.

Adjacent SGLang tests supply cases and FP32/BF16 error budgets, not this oracle.
The raw-fold test's 2*FP32-eps bound does NOT apply to compact reconstruction.
No torch import or device probing occurs until a numerical operation is requested.
"""
from dataclasses import dataclass
import importlib
import math

SOURCE_COMMIT = "40a64e1b5ba00c73b7f8a123a903a2123b7d467c"
SOURCE_SHA256 = "71a49ecdef3f598263a2681839ba03106f95dba7b0e6bdba3d0496c7da830bc6"
SOURCE_RELATIVE = "python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py"
ADJACENT_TESTS = (
    "test/registered/attention/unittests/gdn/test_gdn_replayssm_spec_fold.py",
    "test/registered/attention/unittests/gdn/test_linear_replayssm_decode.py",
)
PUBLIC_FUNCTIONS = (
    "gdn_replayssm_spec_decode", "commit_gdn_replayssm_spec",
    "commit_gdn_replayssm_circular",
)
# Fixed before any device measurements. Never adapt these to measured errors.
# FP16 is a NEW budget, not an upstream tolerance. Applies also to long chains.
TOLERANCES = {
    "float32": {"output": (1e-4, 1e-3), "state": (1.5e-3, 1e-3)},
    "bfloat16": {"output": (1e-3, 1e-2), "state": (3e-3, 1e-2)},
    "float16": {"output": (5e-4, 3e-3), "state": (1.5e-3, 3e-3)},
}
UPSTREAM_CASES = {
    "fold64": dict(B=3, H=4, HV=8, T=4, K=64, V=64),
    "gqa128": dict(B=4, H=4, HV=8, T=4, K=128, V=128),
    "gqa128-wide": dict(B=4, H=8, HV=16, T=4, K=128, V=128),
}


@dataclass(frozen=True)
class Case:
    B: int = 3
    H: int = 4
    HV: int = 8
    T: int = 4
    K: int = 64
    V: int = 64
    L: int = 16
    layers: int = 2
    history: int = 3
    dtype: str = "bfloat16"

    def validate(self):
        if self.B < 0 or min(self.H, self.HV, self.T, self.K, self.V, self.layers) < 1:
            raise ValueError("B must be nonnegative; other dimensions must be positive")
        if self.HV % self.H or self.L < 2 * self.T or self.L & (self.L - 1):
            raise ValueError("require HV % H == 0 and power-of-two L >= 2*T")
        if not 0 <= self.history <= self.L - self.T:
            raise ValueError("require 0 <= history <= L-T")
        if self.dtype not in TOLERANCES:
            raise ValueError("unsupported dtype name")
        return self


def torch_module():
    return importlib.import_module("torch")


def strided_int(values, device, dtype=None):
    torch = torch_module()
    backing = torch.full((2 * len(values),), -123, dtype=dtype or torch.int64, device=device)
    backing[::2] = torch.tensor(values, dtype=backing.dtype, device=device)
    return backing[::2]


def clone_tree(value):
    """Preserve strides, including non-contiguous QKV and int64 indices."""
    torch = torch_module()
    if isinstance(value, torch.Tensor):
        return torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                   device=value.device).copy_(value)
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    return value


MUTATED = ("checkpoint_state", "d_cache", "k_cache", "g_cache", "rawv_cache",
           "rawk_cache", "beta_cache", "out", "write_pos", "cache_base", "is_flush")


def snapshot_mutated(data):
    return {k: clone_tree(data[k]) for k in MUTATED if data[k] is not None}


def restore_mutated(data, saved):
    for name, value in saved.items():
        data[name].copy_(value)


def make_case(case=Case(), *, device="cpu", raw=False, residual=True,
              strided=True, lengths=None, seed=11, state_dtype="float32",
              tight_pool=False):
    """Cache slots and physical states deliberately use different permutations.

    History is finite, nonzero and wrapped. Unused entries contain finite canaries
    because upstream compact commit loads a full L even for a shorter history.
    """
    torch = torch_module()
    c = case.validate()
    dtype = getattr(torch, c.dtype)
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape, dt=dtype):
        return torch.randn(*shape, dtype=dt, device=device, generator=gen)

    if tight_pool:
        slots = list(range(1, c.B + 1))
        replay = slots[1:] + slots[:1]
        pool = c.B + 1
    else:
        slots = [5, 2, 7] + list(range(9, 9 + max(0, c.B - 3)))
        replay = [4, 1, 6] + list(range(8, 8 + max(0, c.B - 3)))
        slots, replay = slots[:c.B], replay[:c.B]
        pool = max([10] + slots + replay) + 1
    lengths = [c.T] * c.B if lengths is None else lengths
    if len(lengths) != c.B or any(not 0 <= n <= c.T for n in lengths):
        raise ValueError("invalid varlen lengths")
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    n = starts[-1]
    packed = rand(c.layers, n, 2 * c.H * c.K + c.HV * c.V)
    q, k, v = packed.split([c.H * c.K, c.H * c.K, c.HV * c.V], dim=-1)
    q, k, v = (q.view(c.layers, n, c.H, c.K),
               k.view(c.layers, n, c.H, c.K), v.view(c.layers, n, c.HV, c.V))
    indices = lambda values: strided_int(values, device) if strided else torch.tensor(
        values, dtype=torch.int32, device=device)
    d = rand(c.layers, pool, c.HV, c.L, c.V) * 0.1
    keys = rand(c.layers, pool, c.H, c.L, c.K) / math.sqrt(c.K)
    data = dict(
        case=c, q=q if strided else q.contiguous(), k=k if strided else k.contiguous(),
        v=v if strided else v.contiguous(), a=rand(c.layers, n, c.HV) * 0.5,
        b=rand(c.layers, n, c.HV), A_log=rand(c.layers, c.HV, dt=torch.float32) * 0.1,
        dt_bias=rand(c.layers, c.HV, dt=torch.float32) * 0.1,
        checkpoint_state=rand(c.layers, pool, c.HV, c.V, c.K,
                              dt=getattr(torch, state_dtype)),
        d_cache=d, k_cache=keys,
        g_cache=-rand(c.layers, pool, c.HV, c.L, dt=torch.float32).abs() * 0.3,
        rawv_cache=(rand(*d.shape) if raw else rand(*d.shape) * 1e-5) if raw or residual else None,
        rawk_cache=(rand(*keys.shape) if raw else rand(*keys.shape) * 1e-5) if raw or residual else None,
        beta_cache=torch.sigmoid(rand(c.layers, pool, c.HV, c.L, dt=torch.float32)) if raw else None,
        out=torch.full((c.layers, n, c.HV, c.V), 17.0, device=device, dtype=dtype),
        query_start_loc=indices(starts), ssm_state_indices=indices(slots),
        replay_indices=indices(replay),
        write_pos=torch.full((pool,), c.history, device=device, dtype=torch.int32),
        cache_base=torch.full((pool,), c.L - 2, device=device, dtype=torch.int32),
        is_flush=torch.zeros(pool, device=device, dtype=torch.int8),
        accept_lens=indices(([3, 1, 4] if c.T == 4 else [c.T] * 3)[:c.B]
                            + [c.T] * max(0, c.B - 3)),
        null_block_id=-1,
    )
    return data


def decode_kwargs(data, layer=0, *, mode="verify", normalize=True, scale=None):
    names = ("q", "k", "v", "a", "b", "A_log", "dt_bias", "checkpoint_state",
             "d_cache", "k_cache", "g_cache", "rawv_cache", "rawk_cache", "beta_cache", "out")
    kw = {k: data[k][layer] if data[k] is not None else None for k in names}
    kw.update({k: data[k] for k in ("query_start_loc", "ssm_state_indices", "replay_indices",
                                   "write_pos", "cache_base", "is_flush", "null_block_id")})
    kw.update(max_cache_len=data["case"].L, max_spec_len=data["case"].T,
              launch_mode=mode, use_qk_l2norm_in_kernel=normalize, scale=scale)
    return kw


def cursor_kwargs(data, *, fold_every=False, cache_buf_len=None):
    return dict(write_pos=data["write_pos"], cache_base=data["cache_base"],
                is_flush=data["is_flush"], num_accepted=data["accept_lens"],
                replay_indices=data["replay_indices"], max_cache_len=data["case"].L,
                max_spec_len=data["case"].T, fold_every_commit=fold_every,
                cache_buf_len=cache_buf_len, null_block_id=data["null_block_id"])


def compact_kwargs(data, track=None, steps=None, *, residual=True):
    kw = {k: data[k] for k in ("checkpoint_state", "d_cache", "k_cache", "g_cache",
                               "replay_indices", "write_pos", "cache_base", "is_flush",
                               "accept_lens", "null_block_id")}
    kw.update(state_batch_indices=data["ssm_state_indices"],
              d_residual_cache=data["rawv_cache"] if residual else None,
              k_residual_cache=data["rawk_cache"] if residual else None,
              mamba_track_indices=track, mamba_steps_to_track=steps)
    return kw


def normalized(x):
    return x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt()


def recurrent_step(state, key, value, gate, beta):
    """FP64 decay -> prediction -> delta -> outer-product; [HV,V,K]."""
    state = state.double() * gate.double().exp()[:, None, None]
    delta = beta.double()[:, None] * (value.double() - (state * key[:, None, :]).sum(-1))
    state = state + delta[:, :, None] * key[:, None, :]
    return state, delta


def history_state(data, layer, row, *, length=None, residual=False, exact=False,
                  normalize=True, initial=None):
    torch = torch_module()
    c = data["case"]
    slot, replay = int(data["ssm_state_indices"][row]), int(data["replay_indices"][row])
    state = data["checkpoint_state"][layer, slot].double().clone() if initial is None else initial.double().clone()
    count = int(data["write_pos"][replay]) if length is None else length
    base = int(data["cache_base"][replay])
    heads = torch.arange(c.HV, device=state.device) // (c.HV // c.H)
    for t in range(count):
        p = (base + t) & (c.L - 1)
        gate = data["g_cache"][layer, replay, :, p].double()
        if exact:
            key = data["rawk_cache"][layer, replay, :, p].double()
            key = normalized(key) if normalize else key
            state, _ = recurrent_step(state, key[heads], data["rawv_cache"][layer, replay, :, p],
                                      gate, data["beta_cache"][layer, replay, :, p])
        else:
            key = data["k_cache"][layer, replay, :, p].double()
            delta = data["d_cache"][layer, replay, :, p].double()
            if residual and data["rawv_cache"] is not None and data["rawk_cache"] is not None:
                key = key + data["rawk_cache"][layer, replay, :, p].double()
                delta = delta + data["rawv_cache"][layer, replay, :, p].double()
            state = state * gate.exp()[:, None, None] + delta[:, :, None] * key[heads, None, :]
    return state


def oracle_decode(data, *, mode="verify", normalize=True, scale=None):
    """Sequential oracle with real API lifetime, NOT a clone of the UT transform.

    Verify reads history HIGH parts only, even when low parts exist. Low parts
    are for compact commit. beta presence selects raw v/k instead of low D/K.
    Exact fold modifies checkpoint, NOT cursors; spec writes stay at old base+wp.
    Cursor advance later moves base after a flush with positive acceptance.
    """
    torch = torch_module()
    c = data["case"]
    exact = all(data[k] is not None for k in ("rawv_cache", "rawk_cache", "beta_cache"))
    residual = data["rawv_cache"] is not None and data["rawk_cache"] is not None and data["beta_cache"] is None
    if mode in ("both", "flush") and not exact:
        raise ValueError("raw replay buffers are required for exact-fold launch")
    if mode not in ("verify", "both", "flush"):
        raise ValueError("invalid launch mode")
    heads = torch.arange(c.HV, device=data["out"].device) // (c.HV // c.H)
    snapshots = {}
    for layer in range(c.layers):
        for row in range(c.B):
            slot, replay = int(data["ssm_state_indices"][row]), int(data["replay_indices"][row])
            bos, eos = (int(x) for x in data["query_start_loc"][row:row + 2])
            if slot <= data["null_block_id"]:
                if mode != "flush":
                    data["out"][layer, bos:eos].zero_()
                continue
            flush = bool(data["is_flush"][replay])
            if (mode == "verify" and flush) or (mode == "flush" and not flush):
                continue
            if flush:
                state = history_state(data, layer, row, exact=True, normalize=normalize)
                data["checkpoint_state"][layer, slot].copy_(state)
                state = data["checkpoint_state"][layer, slot].double()
            else:
                # Deliberately ignore low parts: upstream verify does too.
                state = history_state(data, layer, row)
            wp, base = int(data["write_pos"][replay]), int(data["cache_base"][replay])
            snapshots[layer, row] = []
            for t, token in enumerate(range(bos, eos)):
                q = data["q"][layer, token].double()
                key = data["k"][layer, token].double()
                if normalize:
                    q, key = normalized(q), normalized(key)
                q = q * (c.K ** -0.5 if scale is None else scale)
                x = data["a"][layer, token].double() + data["dt_bias"][layer].double()
                gate = -data["A_log"][layer].double().exp() * torch.nn.functional.softplus(x, threshold=20)
                beta = torch.sigmoid(data["b"][layer, token].double())
                state, delta = recurrent_step(state, key[heads], data["v"][layer, token], gate, beta)
                data["out"][layer, token].copy_((state * q[heads, None, :]).sum(-1))
                snapshots[layer, row].append(state.clone())
                if wp + t >= c.L:
                    continue
                p = (base + wp + t) & (c.L - 1)
                dhi = delta.float().to(data["d_cache"].dtype)
                khi = key.float().to(data["k_cache"].dtype)
                data["d_cache"][layer, replay, :, p].copy_(dhi)
                data["k_cache"][layer, replay, :, p].copy_(khi)
                data["g_cache"][layer, replay, :, p].copy_(gate)
                if exact:
                    data["rawv_cache"][layer, replay, :, p].copy_(data["v"][layer, token])
                    data["rawk_cache"][layer, replay, :, p].copy_(data["k"][layer, token])
                    data["beta_cache"][layer, replay, :, p].copy_(beta)
                elif residual:
                    data["rawv_cache"][layer, replay, :, p].copy_(delta.float() - dhi.float())
                    data["rawk_cache"][layer, replay, :, p].copy_(key.float() - khi.float())
    return snapshots


def oracle_cursor(data, *, fold_every=False, cache_buf_len=None):
    c = data["case"]
    ring = c.L if cache_buf_len is None else cache_buf_len
    for row, replay_tensor in enumerate(data["replay_indices"]):
        replay = int(replay_tensor)
        if replay <= data["null_block_id"]:
            continue
        accepted = int(data["accept_lens"][row])
        wp, base, flush = (int(data[k][replay]) for k in ("write_pos", "cache_base", "is_flush"))
        if accepted > 0 and flush:
            data["cache_base"][replay] = (base + wp) & (ring - 1)
        new_wp = accepted if flush else wp + accepted
        data["write_pos"][replay] = new_wp
        data["is_flush"][replay] = int(new_wp > 0 if fold_every else new_wp + 2 * c.T > c.L)


def oracle_compact(data, track=None, steps=None, *, residual=True):
    """FP64 sequential D/K/G materialization, including pre-fold tracking state."""
    c = data["case"]
    for row in range(c.B):
        slot, replay = int(data["ssm_state_indices"][row]), int(data["replay_indices"][row])
        if slot <= data["null_block_id"]:
            continue
        count = int(data["write_pos"][replay])
        fold = bool(data["is_flush"][replay])
        tracking = track is not None and steps is not None and int(track[row]) > data["null_block_id"] and int(steps[row]) >= 0
        for layer in range(c.layers):
            initial = data["checkpoint_state"][layer, slot].double().clone()
            if tracking:
                length = count - int(data["accept_lens"][row]) + int(steps[row]) + 1
                tracked = history_state(data, layer, row, length=length, residual=residual, initial=initial)
                data["checkpoint_state"][layer, int(track[row])].copy_(tracked)
            if fold:
                state = history_state(data, layer, row, residual=residual, initial=initial)
                data["checkpoint_state"][layer, slot].copy_(state)
        if fold:
            data["cache_base"][replay] = (int(data["cache_base"][replay]) + count) & (c.L - 1)
            data["write_pos"][replay] = 0
            data["is_flush"][replay] = 0
