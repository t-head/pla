# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Utilities required by the private Triton prefill references."""

import contextlib
import functools
import inspect
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton


SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)
autotune_cache_kwargs = (
    {"cache_results": True} if SUPPORTS_AUTOTUNE_CACHE else {}
)
SUPPRESS_LEVEL = 0


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Cache the four most recent calls by tensor identity."""
    cache_entries: Tuple[Optional[Tuple], Optional[Dict], Any] = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    cache_entries = (
                        cache_entries[:i]
                        + cache_entries[i + 1 :]
                        + [(args, kwargs, last_result)]
                    )
                    return last_result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@lru_cache(maxsize=None)
def get_available_device() -> str:
    return triton.runtime.driver.active.get_current_target().backend


device = get_available_device()
device_torch_lib = getattr(torch, "cuda" if device == "hip" else device)
is_nvidia = device == "cuda"
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
is_tf32_supported = is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8
is_gather_supported = hasattr(triton.language, "gather")


def custom_device_ctx(index: int):
    if index is None:
        return contextlib.nullcontext()
    return device_torch_lib.device(index)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Make tensor arguments contiguous and run on their current device."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = tuple(
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }
        tensor = next(
            (
                value
                for value in (*args, *kwargs.values())
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        ctx = (
            custom_device_ctx(tensor.device.index)
            if tensor is not None
            else contextlib.nullcontext()
        )
        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


class Backend(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        return cls.__members__.get(arch.upper(), cls.DEFAULT).value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    properties = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)
    return properties["max_shared_mem"] >= Backend.get_shared_memory(arch)


autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type="cuda")
