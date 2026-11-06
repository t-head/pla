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

"""
Contract tests for the Triton packed decode kernels in fused_recurrent.py.

* The flat (NV, B * HV) grid (SGLang layout) and the (NV, HV, B) fallback
  produce bit-identical output and state.
* The grid switches to the fallback exactly when B * HV exceeds 65535, and both
  sides of the boundary match a torch reference (the flat grid alone silently
  drops CTAs past the limit).
* KDA ``lower_bound`` applies the safe gate
  ``lower_bound * sigmoid(exp(A_log) * (a + dt_bias))``.
"""

import os
import sys
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import torch
import torch.nn.functional as F

import fused_recurrent as fr

K = V = 128
DEVICE = "cuda"
DTYPE = torch.bfloat16


def make_inputs(B, H, HV, is_kda, seed=0, pad_every=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, dtype=DTYPE):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    gate = HV * K if is_kda else HV
    pool = B + 3
    indices = torch.randperm(pool, device=DEVICE, generator=gen)[:B].to(torch.int32)
    if pad_every:
        indices[::pad_every] = -1
    return dict(
        mixed_qkv=randn(B, 2 * H * K + HV * V),
        a=randn(B, gate),
        b=randn(B, HV),
        A_log=randn(HV, dtype=torch.float32),
        dt_bias=randn(gate),
        initial_state=randn(pool, HV, V, K, dtype=torch.float32) * 0.1,
        ssm_state_indices=indices,
    )


def run_kernel(inp, is_kda, lower_bound=None):
    state = inp["initial_state"].clone()
    B, HV = inp["b"].shape
    out = torch.full((B, 1, HV, V), float("nan"), device=DEVICE, dtype=DTYPE)
    kwargs = dict(
        mixed_qkv=inp["mixed_qkv"], a=inp["a"], b=inp["b"], A_log=inp["A_log"],
        dt_bias=inp["dt_bias"], scale=K ** -0.5, initial_state=state, out=out,
        ssm_state_indices=inp["ssm_state_indices"], use_qk_l2norm_in_kernel=True,
    )
    if is_kda:
        fr.fused_recurrent_kda_packed_decode(**kwargs, lower_bound=lower_bound)
    else:
        fr.fused_recurrent_gated_delta_rule_packed_decode(**kwargs)
    torch.cuda.synchronize()
    return out, state


def reference(inp, is_kda, lower_bound=None, rows=128):
    """fp32 torch recurrence, chunked over the batch to bound peak memory."""
    state = inp["initial_state"].clone()
    B, HV = inp["b"].shape
    H = (inp["mixed_qkv"].shape[1] - HV * V) // (2 * K)
    out = torch.zeros(B, 1, HV, V, device=DEVICE, dtype=DTYPE)
    A = inp["A_log"].exp()[:, None]
    for begin in range(0, B, rows):
        sl = slice(begin, min(begin + rows, B))
        idx = inp["ssm_state_indices"][sl].long()
        valid = idx >= 0
        mixed = inp["mixed_qkv"][sl][valid].float()
        n = mixed.shape[0]
        q = mixed[:, :H * K].view(n, H, K)
        k = mixed[:, H * K:2 * H * K].view(n, H, K)
        v = mixed[:, 2 * H * K:].view(n, HV, V)
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q = q.repeat_interleave(HV // H, dim=1) * K ** -0.5
        k = k.repeat_interleave(HV // H, dim=1)
        x = inp["a"][sl][valid].float().view(n, HV, -1) + inp["dt_bias"].float().view(HV, -1)
        if lower_bound is not None:
            g = lower_bound * torch.sigmoid(A * x)
        else:
            g = -A * F.softplus(x, beta=1.0, threshold=20.0)
        beta = torch.sigmoid(inp["b"][sl][valid].float())
        if not is_kda:
            beta = beta.to(DTYPE).float()
        h = state[idx[valid]] * g.exp()[:, :, None, :]
        delta = (v - (h * k[:, :, None, :]).sum(-1)) * beta[..., None]
        h = h + delta[..., None] * k[:, :, None, :]
        rows_out = torch.zeros(sl.stop - sl.start, HV, V, device=DEVICE, dtype=DTYPE)
        rows_out[valid] = (h * q[:, :, None, :]).sum(-1).to(DTYPE)
        out[sl, 0] = rows_out
        state[idx[valid]] = h
    return out, state


class PackedDecodeGridTest(unittest.TestCase):

    def assert_matches_reference(self, inp, is_kda, lower_bound=None):
        out, state = run_kernel(inp, is_kda, lower_bound)
        ref_out, ref_state = reference(inp, is_kda, lower_bound)
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(state, ref_state, atol=1e-4, rtol=1e-4)
        return out, state

    def test_grid_selection(self):
        self.assertEqual(fr.packed_decode_grid(4, 511, 128), ((4, 65408), True))
        self.assertEqual(fr.packed_decode_grid(4, 512, 128), ((4, 128, 512), False))
        self.assertEqual(fr.packed_decode_grid(4, 682, 96), ((4, 65472), True))
        self.assertEqual(fr.packed_decode_grid(4, 683, 96), ((4, 96, 683), False))

    def test_small_batch_matches_reference(self):
        for is_kda, lower_bound in ((False, None), (True, None), (True, -5.0)):
            with self.subTest(is_kda=is_kda, lower_bound=lower_bound):
                inp = make_inputs(37, 4, 8, is_kda, seed=1, pad_every=5)
                out, state = self.assert_matches_reference(inp, is_kda, lower_bound)
                pad = inp["ssm_state_indices"] < 0
                self.assertEqual(int(torch.count_nonzero(out[pad])), 0)
                live = torch.zeros(state.shape[0], dtype=torch.bool, device=DEVICE)
                live[inp["ssm_state_indices"][~pad].long()] = True
                self.assertTrue(torch.equal(state[~live], inp["initial_state"][~live]))

    def test_flat_and_fallback_grids_are_bit_identical(self):
        for is_kda, lower_bound in ((False, None), (True, None), (True, -5.0)):
            with self.subTest(is_kda=is_kda, lower_bound=lower_bound):
                inp = make_inputs(64, 8, 32, is_kda, seed=2, pad_every=7)
                flat = run_kernel(inp, is_kda, lower_bound)
                saved = fr.MAX_GRID_DIM_Y
                fr.MAX_GRID_DIM_Y = 0
                try:
                    fallback = run_kernel(inp, is_kda, lower_bound)
                finally:
                    fr.MAX_GRID_DIM_Y = saved
                self.assertTrue(torch.equal(flat[0], fallback[0]))
                self.assertTrue(torch.equal(flat[1], fallback[1]))

    def test_lower_bound_changes_gate(self):
        inp = make_inputs(8, 4, 4, True, seed=3)
        default = run_kernel(inp, True)[1]
        bounded = run_kernel(inp, True, -5.0)[1]
        self.assertFalse(torch.equal(default, bounded))

    def test_grid_limit_boundary_matches_reference(self):
        for is_kda, H, HV, batches in ((False, 16, 128, (511, 512)),
                                       (True, 96, 96, (682, 683))):
            for B in batches:
                with self.subTest(is_kda=is_kda, HV=HV, B=B):
                    inp = make_inputs(B, H, HV, is_kda, seed=4)
                    self.assert_matches_reference(inp, is_kda, -5.0 if is_kda else None)
                    del inp
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
