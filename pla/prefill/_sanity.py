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

"""Validate prefill metadata before Python narrows offsets or builds CP plans.

Native entry points own kernel validation; this helper catches information lost
by conversions and checks host copies before a planner consumes them. Device
values are never read during CUDA graph capture.
"""

import numbers
import os

import torch


def enabled() -> bool:
    return os.environ.get("PLA_SANITY_CHECK") == "1"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"PLA_SANITY_CHECK [prefill]: {message}")


def capturing(reference: torch.Tensor) -> bool:
    if not enabled() or not reference.is_cuda:
        return False
    try:
        with torch.cuda.device(reference.device):
            return torch.cuda.is_current_stream_capturing()
    except Exception as exc:
        raise AssertionError(
            "PLA_SANITY_CHECK [prefill]: cannot determine CUDA graph capture status"
        ) from exc


def require_host_read(reference: torch.Tensor, operation: str) -> None:
    if enabled():
        check(not capturing(reference),
              f"{operation} requires host offsets during CUDA graph capture; "
              "provide raw_cu_seqlens or warm the metadata cache before capture")


def offset_values(values, capacity: int, count=None):
    values = tuple(values)
    check(len(values) >= 2 and (count is None or len(values) == count),
          "offsets must have batch+1 entries")
    check(all(isinstance(value, numbers.Integral) for value in values),
          "offsets must contain integers")
    check(values[0] == 0, "offsets must start at zero")
    check(all(0 <= left <= right <= min(capacity, 2**31 - 1)
              for left, right in zip(values, values[1:])),
          "offsets must be monotonic and within token capacity/int32 range")
    return values


def offsets(cu, capacity: int, raw=None, *, reference=None):
    """Return checked host offsets in eager mode, preserving capture metadata."""
    if not enabled():
        return raw
    check(isinstance(capacity, numbers.Integral) and 0 <= capacity <= 2**31 - 1,
          "token capacity must fit int32")
    if cu is None:
        if raw is not None:
            check(offset_values(raw, capacity, 2) == (0, capacity),
                  "fixed-length host offsets must be [0,T]")
        return raw
    check(isinstance(cu, torch.Tensor) and cu.layout == torch.strided,
          "cu_seqlens must be a strided tensor")
    check(cu.ndim == 1 and cu.numel() >= 2,
          "cu_seqlens must be [batch+1]")
    check(cu.dtype in (torch.int32, torch.int64),
          "cu_seqlens must be int32 or int64 before conversion")
    check(cu.device.type in ("cpu", "cuda"),
          "cu_seqlens must be CPU or CUDA/PPU metadata")
    host = offset_values(raw, capacity, cu.numel()) if raw is not None else None
    if capturing(cu if reference is None else reference):
        return host
    device_values = offset_values(cu.detach().cpu().tolist(), capacity, cu.numel())
    if host is not None:
        check(host == device_values, "raw_cu_seqlens must equal device cu_seqlens")
    return device_values
