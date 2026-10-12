#!/usr/bin/env python3
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

"""Kernel runner for acu: runs one kernel type across all combos for a single B.
acu profiles each kernel launch. Warmup launches are skipped by acu --launch-skip.

Input: JSON on argv[1] with {B, H, HV, K, V, combos, kernel, mtp_T}
Each combo produces exactly 1 kernel launch (after warmup).
"""
import json, sys, torch

args = json.loads(sys.argv[1])
torch.cuda.set_device(0)
device = "cuda:0"

B = args["B"]
H, HV, K, V = args["H"], args["HV"], args["K"], args["V"]
combos = [tuple(c) for c in args["combos"]]
kernel = args["kernel"]
mtp_T = args.get("mtp_T", 4)
dtype = torch.bfloat16
packed_state_dtype = torch.bfloat16 if args.get("state_dtype", "fp32") == "bf16" else torch.float32

from pla.decode.private import fused_sigmoid_gating_delta_rule_forward_k_last_tunable
from pla.decode.private import fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable

# Pre-allocate tensors based on kernel type
A_log = torch.randn(HV, dtype=torch.float32, device=device)
dt_bias = torch.randn(HV, dtype=dtype, device=device)
indices = torch.arange(B, dtype=torch.int32, device=device)

if kernel in ("k_last", "k_last_mtp"):
    if kernel == "k_last":
        a = torch.randn(1, B, HV, dtype=dtype, device=device)
        b = torch.randn(1, B, HV, dtype=dtype, device=device)
        q = torch.randn(1, B, H, K, dtype=dtype, device=device)
        k = torch.randn(1, B, H, K, dtype=dtype, device=device)
        v = torch.randn(1, B, HV, V, dtype=dtype, device=device)
        state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=device)

        def run_kernel(bv, wn, sn, hpc):
            fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
                A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                state.clone(), indices, 1.0, True, cu, False, False,
                None, None, None, None, bv, wn, sn, hpc)
    else:  # k_last_mtp
        a = torch.randn(B * mtp_T, HV, dtype=dtype, device=device)
        b = torch.randn(B * mtp_T, HV, dtype=dtype, device=device)
        q = torch.randn(B, mtp_T, H, K, dtype=dtype, device=device)
        k = torch.randn(B, mtp_T, H, K, dtype=dtype, device=device)
        v = torch.randn(B, mtp_T, HV, V, dtype=dtype, device=device)
        state = torch.randn(B, HV, K, V, dtype=torch.float32, device=device)
        cu = torch.arange(0, B * mtp_T + 1, mtp_T, dtype=torch.int32, device=device)
        parent_list = [0] + [t for t in range(mtp_T - 1)]
        rpt = torch.tensor([parent_list] * B, dtype=torch.int64, device=device).contiguous()
        isb = torch.zeros(B * mtp_T, HV, V, K, dtype=torch.float32, device=device)

        def run_kernel(bv, wn, sn, hpc):
            fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
                A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
                state.clone(), indices, 1.0, True, cu, False, True,
                isb, indices, mtp_T, rpt, bv, wn, sn, hpc)

else:  # k_last_packed
    qkv_dim = 2 * H * K + HV * V
    mixed = torch.randn(B, qkv_dim, device=device, dtype=dtype)
    a = torch.randn(B, HV, device=device, dtype=dtype)
    bt = torch.randn(B, HV, device=device, dtype=dtype)
    state = torch.randn(B + 16, HV, V, K, device=device, dtype=packed_state_dtype) * 0.1
    cu = torch.arange(B + 1, device=device, dtype=torch.int32)
    out = mixed.new_empty(B, 1, HV, V)

    def run_kernel(bv, wn, sn, hpc):
        fused_sigmoid_gating_delta_rule_forward_k_last_packed_tunable(
            mixed, a, bt, A_log, dt_bias, 1.0, 20.0,
            K ** -0.5, state.clone(), indices, out.clone(),
            True, cu, False, bv, wn, sn, hpc)

# Warmup: run each combo once (these launches will be skipped by acu)
for bv, wn, sn, hpc in combos:
    try:
        run_kernel(bv, wn, sn, hpc)
    except:
        pass
torch.cuda.synchronize()

# Profiled: run each combo once more (acu captures these)
for bv, wn, sn, hpc in combos:
    try:
        run_kernel(bv, wn, sn, hpc)
        torch.cuda.synchronize()
    except:
        pass
