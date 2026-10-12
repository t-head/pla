# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# ---------------------------------------------------------------------------
# VENDORED SHIM — do not edit the function bodies.
#
# These are byte-faithful copies of the helpers that the official FlashQLA
# `tests/ref_gdr.py` imports via `from flash_qla.utils import (...)`:
#
#   * pad_and_reshape, pack, unpack, fill_last_chunk_of_g
#         -> copied verbatim from flash_qla/utils/pack.py
#   * tensor_cache, prepare_chunk_offsets
#         -> copied verbatim from flash_qla/utils/index.py
#
# The only deviation from upstream is what is *omitted*: the upstream
# `flash_qla/utils/__init__.py` also re-exports profiler / math / contiguous /
# prepare_chunk_indices helpers that drag in `tilelang` and CUDA kernels which
# are neither available nor needed on PPU1.5. `ref_gdr.py` never touches those,
# so leaving them out keeps the reference path pure-torch while the five
# functions it *does* import remain identical to the official source.
# ---------------------------------------------------------------------------

import functools
from typing import Any
from collections.abc import Callable

import torch


# === from flash_qla/utils/pack.py (verbatim) ===============================

def unpack(
    x: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.Tensor,
):
    assert x.shape[0] == 1
    assert len(cu_seqlens.shape) == 1
    max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    batch_size = cu_seqlens.shape[0] - 1
    y = torch.zeros((batch_size, max_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        y[i, : end - start] = x[0, start:end]
    return y


def pack(
    x: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.Tensor,
):
    assert len(cu_seqlens.shape) == 1
    sum_len = cu_seqlens[-1].item()
    batch_size = cu_seqlens.shape[0] - 1
    y = torch.empty((1, sum_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        y[0, start:end] = x[i, : end - start]
    return y


def pad_and_reshape(
    x: torch.Tensor,
    dim: int,
    chunk_size: int = 64,
):
    sequence_length = x.shape[dim]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    zeros = [
        0,
    ] * (2 * (len(x.shape) - 1 - dim))
    padded = torch.nn.functional.pad(x, (*zeros, 0, pad_size))
    return padded.reshape((*x.shape[:dim], -1, chunk_size, *x.shape[dim + 1 :]))


def fill_last_chunk_of_g(
    g: torch.Tensor,
    num_tokens: int,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
    reverse: bool = False,
):
    if cu_seqlens is None:
        last_chunk_size = num_tokens % chunk_size
        if last_chunk_size > 0:
            if reverse:
                g[:, -1, last_chunk_size - 1] += g[:, -1, -1]
            else:
                g[:, -1, last_chunk_size:] = g[
                    :, -1, last_chunk_size - 1 : last_chunk_size
                ]
    else:
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            last_chunk_idx = (end - start) // chunk_size
            last_chunk_size = (end - start) % chunk_size
            if last_chunk_size > 0:
                if reverse:
                    g[i, last_chunk_idx, last_chunk_size - 1] += g[
                        i, last_chunk_idx, -1
                    ]
                else:
                    g[i, last_chunk_idx, last_chunk_size:] = g[
                        i, last_chunk_idx, last_chunk_size - 1 : last_chunk_size
                    ]
    return g


# === from flash_qla/utils/index.py (verbatim) ==============================

def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    cache = []
    cache_size = 4

    def _equal(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return a is b
        return a == b

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache, cache_size
        for (cached_args, cached_kwargs, cached_result) in cache:
            if all(_equal(a, b) for a, b in zip(args, cached_args, strict=False)) and \
                    all(k in cached_kwargs and _equal(v, cached_kwargs[k]) for k, v in kwargs.items()):
                return cached_result
        result = fn(*args, **kwargs)
        cache.insert(0, (args, kwargs, result))
        if len(cache) > cache_size:
            cache = cache[:cache_size]
        return result

    return wrapper


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    seqlens = torch.diff(cu_seqlens)
    num_chunks_per_seq = (seqlens + chunk_size - 1) // chunk_size
    chunk_offsets = torch.zeros_like(cu_seqlens)
    chunk_offsets[1:] = torch.cumsum(num_chunks_per_seq, dim=0)
    return chunk_offsets, chunk_offsets[-1].item()
