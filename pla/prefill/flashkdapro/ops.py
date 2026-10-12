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

"""FlashKDA CuTe end-to-end prefill API.

By default tc01 auto-routes between monolithic FLA and legacy by the frozen rules; FLA uses native packed,
BT64, FP32 recurrence. ``PLA_FLASHKDA_TC01_IMPL=fla/legacy`` explicitly overrides the auto diversion;
tc02 always keeps the original implementation. Decisions consume only host integers and tensor shapes; unknown/uncalibrated inputs or
a missing rule table always go to monolithic FLA, with no silent fallback on runtime errors.
FLA only uses the monolithic AOT extension inside the wheel and never compiles at call time.

``out`` and ``final_state`` are caller-allocated out-params, as in upstream
``flash_kda.fwd``: the caller owns the buffers, picks the state dtype, and gets
a final state only if it asks for one.

Tensors follow the upstream FlashKDA/FLA packed-token contract — q/k/v/g are
3-D ``[total_tokens, H, 128]``. ``cu_seqlens`` is consumed as int32, matching
the SGLang/vLLM serving metadata. Legacy int64 offsets remain accepted and are
converted after an explicit range check.
"""
from typing import Optional, Tuple

import torch

from . import _C as flashkda_cute
from .. import _sanity
from ._dispatch import dispatch_tc01_impl, explicit_tc01_impl, select_backend

CHUNK_SIZE = 16          # Unified chunk size (Kernel 1 tiling, Kernel 2 recurrence)
DK = 128
DV = 128


def flashkda_fwd(
    q: torch.Tensor,                       # [total_tokens, H, DK] bf16
    k: torch.Tensor,                       # [total_tokens, H, DK] bf16
    v: torch.Tensor,                       # [total_tokens, H, DV] bf16
    g: torch.Tensor,                       # [total_tokens, H, DK] bf16 (raw gate)
    beta: torch.Tensor,                    # [total_tokens, H] bf16
    A_log: torch.Tensor,                   # [H] fp32
    dt_bias: torch.Tensor,                 # [H, DK] fp32
    out: torch.Tensor,                     # [total_tokens, H, DV] bf16   (written)
    scale: Optional[float] = None,
    lower_bound: float = -5.0,
    initial_state: Optional[torch.Tensor] = None,   # [N, H, DV, DK] bf16/fp32
    final_state: Optional[torch.Tensor] = None,     # [N, H, DV, DK] bf16/fp32 (written)
    cu_seqlens: Optional[torch.Tensor] = None,      # [N+1] int32 (int64 compatible)
    state_pool: Optional[torch.Tensor] = None,      # [slots,H,DV,DK] bf16/fp32 (in-place)
    state_indices: Optional[torch.Tensor] = None,   # [N] int32 cache slots
    raw_cu_seqlens: Optional[Tuple[int, ...]] = None,  # trusted Host offsets
) -> None:
    """FlashKDA prefill forward. Writes ``out`` (and ``final_state`` if given).

    Args:
        q, k, v, g: ``[total_tokens, H, 128]`` bf16, contiguous. ``g`` is the
            raw (pre-gate) input; the gate is applied inside Kernel 1.
        beta: raw pre-sigmoid ``[total_tokens, H]`` bf16. Each consumer kernel
            computes the fp32 sigmoid from the raw value with one shared inlined
            implementation, so all stages observe bit-identical activations. The
            head dimension must be contiguous, while a strided token dimension
            is accepted for fused-projection column slices.
        A_log: ``[H]`` fp32. ``dt_bias``: ``[H, 128]`` fp32.
        out: ``[total_tokens, H, DV]`` bf16, contiguous. Written in full; the
            caller allocates it. If ``cu_seqlens[-1] < total_tokens``, the real
            prefix is computed and the physical padded suffix is set to zero.
        scale: q scaling; defaults to ``DK ** -0.5``.
        lower_bound: KDA gate lower bound, expected in ``[-5, 0]``.
        initial_state: ``[N, H, DV, DK]`` bf16 or fp32, contiguous. ``None``
            starts from a zero state.
        final_state: ``[N, H, DV, DK]`` bf16 or fp32, contiguous. Written with
            the state after the last chunk; ``None`` skips computing it. When
            both states are given their dtypes must match.
        cu_seqlens: ``[N+1]`` int32 prefix offsets for packed/varlen input.
            Legacy int64 input is accepted when its endpoint fits in int32.
            ``None`` treats the input as a single sequence. Its last offset may
            be smaller than the physical token dimension to describe suffix
            padding; padding never contributes to the recurrent state.
        state_pool: Indexed serving cache ``[slots,H,DV,DK]``. When supplied,
            it is both the initial and final state and is updated in place;
            ``initial_state`` and ``final_state`` must be ``None``.
        state_indices: ``[N]`` contiguous int32 cache slots. Indices must be in
            range and unique within the batch. Serving owns this invariant, so
            the hot path intentionally performs no device-to-host validation.
        raw_cu_seqlens: Trusted Host copy of ``cu_seqlens`` values. Serving can
            pass the request metadata offsets so the CP planner never reads the
            device tensor back to the CPU. Plans and their device descriptor
            tables are cached by these values; the tuple must have ``N+1``
            entries and describe the same packed batch as ``cu_seqlens``.

    tc01 FLA uses an FP32 carry internally; the public state is still rounded to BF16 at the read/publish
    boundary (read/write is fused into the recurrence kernel, semantics unchanged); legacy keeps its original carry,
    and the two backends are not guaranteed to be bit-identical. beta is computed by each consumer kernel from raw
    bf16 as an FP32 sigmoid with the same inline implementation, bit-identical across kernels. Returns None; out/final_state
    are always owned by the caller.

    Before Graph capture the module must be warmed up on the target device; when using trusted raw_cu_seqlens,
    the same-valued plan must also be warmed up on the stream used by the capture. With only device cu_seqlens,
    every call and replay rebuilds the metadata on the GPU, with no hot-path D2H. All mutable scratch
    belongs to this call/Graph. The old BT64/CP environment variables only affect the legacy backend; the monolithic FLA
    tc01 FLA CP uses the separate ``PLA_FLASHKDA_TC01_FLA_CP``: unset, empty or
    ``auto`` conservatively selects only the validated single-segment fused path (H6 at >=32768 tokens;
    H8/H12 at >=8192), while H4 and uncalibrated inputs remain serial. ``off``/``0`` rolls back exactly
    to serial; forced values allow only ``1/2/3/4/5/8``. Forced mode requires N1, trusted host offsets,
    a supported head and an effective length <=102400, otherwise it raises. Forced multi-segment
    diagnostics can select ``legacy`` or ``state`` with ``PLA_FLASHKDA_TC01_FLA_CP_ALGO``.
    """
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    backend = select_backend(torch.cuda.get_device_capability(q.device))
    if backend == 'fla':
        # An explicit IMPL takes precedence over the frozen rules; rule diversion runs only without an explicit override,
        # and the decision consumes only host integers and tensor shapes (no hot-path D2H); replay does not re-select.
        if (explicit_tc01_impl()
                or dispatch_tc01_impl(q, cu_seqlens, raw_cu_seqlens, initial_state,
                                      final_state, state_pool, state_indices) == 'fla'):
            from ._tc01_fla import forward
            return forward(q, k, v, g, beta, A_log, dt_bias, out, scale, lower_bound,
                           initial_state, final_state, cu_seqlens, state_pool,
                           state_indices, raw_cu_seqlens)
        backend = 'legacy'
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta),
                         ("A_log", A_log), ("dt_bias", dt_bias), ("out", out)):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
    for name, tensor in (("initial_state", initial_state),
                         ("final_state", final_state),
                         ("cu_seqlens", cu_seqlens),
                         ("state_pool", state_pool),
                         ("state_indices", state_indices)):
        if tensor is not None and tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")

    if state_pool is not None:
        if initial_state is not None or final_state is not None:
            raise ValueError(
                "state_pool is mutually exclusive with initial_state/final_state")
        if state_indices is None:
            raise ValueError("state_pool requires state_indices")
        initial_state = state_pool
        final_state = state_pool
    elif state_indices is not None and initial_state is None and final_state is None:
        raise ValueError(
            "state_indices requires state_pool, initial_state, or final_state")

    if scale is None:
        scale = q.size(-1) ** -0.5

    raw_cu_seqlens = _sanity.offsets(
        cu_seqlens, q.size(0), raw_cu_seqlens, reference=q)
    cu32 = None
    if cu_seqlens is not None:
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be int32 or int64")
        if cu_seqlens.dtype == torch.int64:
            # Compatibility-only path. Serving frameworks already pass int32,
            # so their hot path neither synchronizes nor allocates a conversion
            # tensor. A valid prefix is monotonic, hence checking the endpoint
            # is sufficient to prevent silent narrowing overflow.
            if not _sanity.capturing(q):
                endpoint = int(cu_seqlens[-1].item())
                i32 = torch.iinfo(torch.int32)
                if endpoint < i32.min or endpoint > i32.max:
                    raise ValueError(
                        f"cu_seqlens endpoint {endpoint} exceeds int32 range")
            cu32 = cu_seqlens.to(torch.int32).contiguous()
        else:
            cu32 = cu_seqlens.contiguous()

    flashkda_cute.flashkda_fwd(
        q, k, v, g, beta, A_log, dt_bias, out,
        float(scale), float(lower_bound), initial_state, final_state, cu32,
        state_indices, raw_cu_seqlens,
    )
