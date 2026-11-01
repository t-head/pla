# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    # Empty sequences have no chunks; later chunks must retain their original sequence IDs.
    pairs = [
        torch.stack([torch.full((n,), seq, dtype=torch.long), torch.arange(n)], 1)
        for seq, n in enumerate(counts) if n > 0
    ]
    if not pairs:
        return cu_seqlens.new_empty((0, 2))
    return torch.cat(pairs).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
