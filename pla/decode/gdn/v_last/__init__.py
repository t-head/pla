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

"""``pla.decode.gdn.v_last`` — v-last (V-major) GDN decode kernels.

Compiled into its own extension (``pla/decode/gdn/v_last/_C*.so``) from
``src/decode/v-last/``, independent of the k-last kernels in :mod:`pla.decode`.
Build it alone with::

    python scripts/build.py --namespace pla.decode.gdn.v_last
"""

import torch
from typing import Optional

from ._C import fused_sigmoid_gating_delta_rule_forward


class FusedSigmoidGatingDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        softplus_beta: float,
        softplus_threshold: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        h0_source: Optional[torch.Tensor],
        o: torch.Tensor,
        h0_indices: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        scale: float,
        T: int,
        B: int,
        H: int,
        HV: int,
        K: int,
        V: int,
        use_initial_state: bool,
        use_qk_l2norm_in_kernel: bool,
        is_varlen: bool
    ) -> torch.Tensor:
        """
        Fused Sigmoid Gating Delta Rule Forward Pass
        """
        # Call CUDA kernel
        fused_sigmoid_gating_delta_rule_forward(
            A_log, a, dt_bias,
            softplus_beta, softplus_threshold,
            q, k, v, b,
            h0_source, o, h0_indices, cu_seqlens,
            scale, int(T), int(B), int(H), int(HV), int(K), int(V),
            use_initial_state, use_qk_l2norm_in_kernel, is_varlen
        )

        return o

    @staticmethod
    def backward(ctx, grad_output):
        # For now, we'll just return None for all gradients
        # A full backward implementation would be more complex
        return (None,) * 23


def fused_sigmoid_gating_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    o: torch.Tensor,
    h0_source: torch.Tensor,
    h0_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: float = 1.0,
    T: int = 1,
    B: int = 1,
    H: int = 1,
    HV: int = 1,
    K: int = 1,
    V: int = 1,
    use_initial_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    is_varlen: bool = False
) -> torch.Tensor:
    """
    Functional interface to the fused sigmoid gating delta rule operation.
    """

    return FusedSigmoidGatingDeltaRuleFunction.apply(
        A_log, a, dt_bias,
        softplus_beta, softplus_threshold,
        q, k, v, b,
        h0_source, o, h0_indices, cu_seqlens,
        scale, T, B, H, HV, K, V,
        use_initial_state, use_qk_l2norm_in_kernel, is_varlen
    )


__all__ = [
    'fused_sigmoid_gating_delta_rule',
    'fused_sigmoid_gating_delta_rule_forward',
]
