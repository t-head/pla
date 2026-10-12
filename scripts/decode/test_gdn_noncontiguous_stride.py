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
Tests that fused_sigmoid_gating_delta_rule_forward_k_last produces correct results
when a/b inputs are non-contiguous, as happens with Qwen3.5-27B (v_per_group=3)
via mixed_ba.split().

Ported from sglang PR #22312, adapted to use fused_sigmoid_gating_delta_rule_forward_k_last
instead of fused_sigmoid_gating_delta_rule_update.
"""

import sys
import os
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import torch
from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last


def _make_noncontiguous_ab(batch, num_heads, dtype=torch.bfloat16, device="cuda"):
    """
    Simulate Qwen3.5 fallback: mixed_ba.split([nv_tp, nv_tp], dim=-1).
    Returns (b, a) as split views with stride(0) = 2 * num_heads.
    Also returns contiguous copies for reference comparison.
    """
    mixed_ba = torch.randn(batch, 2 * num_heads, dtype=dtype, device=device)
    b, a = mixed_ba.split([num_heads, num_heads], dim=-1)

    if batch > 1:
        assert not a.is_contiguous(), "a should be non-contiguous from split"
        assert not b.is_contiguous(), "b should be non-contiguous from split"
    assert a.stride(0) == 2 * num_heads
    assert b.stride(0) == 2 * num_heads
    return b, a, b.contiguous(), a.contiguous()


def _run_k_last(A_log, dt_bias, q, k, v, a, b, ssm_states, state_indices,
                cu_seqlens, use_qk_l2norm=True, is_kda=False,
                disable_state_update=False):
    """Run fused_sigmoid_gating_delta_rule_forward_k_last with given inputs."""
    return fused_sigmoid_gating_delta_rule_forward_k_last(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b,
        ssm_states, state_indices, 1.0, use_qk_l2norm,
        cu_seqlens, is_kda, disable_state_update,
        None, None, None, None
    )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKLastNonContiguousDecodeGDN(unittest.TestCase):
    """Test k_last decode (T=1) with non-contiguous a/b (GDN, non-KDA)."""

    def _run_test(self, batch, num_k_heads, num_v_heads, head_k_dim=128, head_v_dim=128):
        HV = num_v_heads
        K = head_k_dim
        V = head_v_dim
        T = 1

        torch.manual_seed(42)
        A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(1, batch, num_k_heads, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, batch, num_k_heads, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, batch, HV, V, dtype=torch.bfloat16, device="cuda")

        # Non-contiguous a/b from split
        mixed_ba = torch.randn(batch, 2 * HV, dtype=torch.bfloat16, device="cuda")
        b_nc, a_nc = mixed_ba.split([HV, HV], dim=-1)
        b_c, a_c = b_nc.contiguous(), a_nc.contiguous()

        cu_seqlens = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda")
        cache_len = batch + 4
        ssm_states = torch.randn(cache_len, HV, K, V, dtype=torch.float32, device="cuda")
        state_indices = torch.arange(batch, dtype=torch.int32, device="cuda")

        # Reference: contiguous a/b
        ssm_ref = ssm_states.clone()
        out_ref = _run_k_last(A_log, dt_bias, q, k, v, a_c, b_c, ssm_ref,
                              state_indices, cu_seqlens)

        # Test: non-contiguous a/b
        ssm_test = ssm_states.clone()
        out_test = _run_k_last(A_log, dt_bias, q, k, v, a_nc, b_nc, ssm_test,
                               state_indices, cu_seqlens)

        self.assertTrue(
            torch.allclose(out_test, out_ref, rtol=0, atol=0),
            f"output mismatch: max diff = {(out_test - out_ref).abs().max().item()}")
        self.assertTrue(
            torch.allclose(ssm_test, ssm_ref, rtol=0, atol=0),
            f"state mismatch: max diff = {(ssm_test - ssm_ref).abs().max().item()}")

    def test_small(self):
        self._run_test(batch=4, num_k_heads=4, num_v_heads=8)

    def test_qwen35_max_tp2(self):
        """Qwen3.5-Max TP=2: H=8, HV=32."""
        self._run_test(batch=16, num_k_heads=8, num_v_heads=32)

    def test_qwen35_max_tp4(self):
        """Qwen3.5-Max TP=4: H=4, HV=16."""
        self._run_test(batch=32, num_k_heads=4, num_v_heads=16)

    def test_single_batch(self):
        self._run_test(batch=1, num_k_heads=16, num_v_heads=32)

    def test_large_batch(self):
        self._run_test(batch=64, num_k_heads=16, num_v_heads=64)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKLastNonContiguousMTP(unittest.TestCase):
    """Test k_last MTP (T>1) with non-contiguous a/b."""

    def _run_test(self, batch, T, num_k_heads, num_v_heads, head_k_dim=128, head_v_dim=128):
        HV = num_v_heads
        K = head_k_dim
        V = head_v_dim

        torch.manual_seed(42)
        A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(HV, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(batch, T, num_k_heads, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, T, num_k_heads, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, T, HV, V, dtype=torch.bfloat16, device="cuda")

        # Non-contiguous a/b from split
        mixed_ba = torch.randn(batch * T, 2 * HV, dtype=torch.bfloat16, device="cuda")
        b_nc, a_nc = mixed_ba.split([HV, HV], dim=-1)
        b_c, a_c = b_nc.contiguous(), a_nc.contiguous()

        cu_seqlens = torch.arange(0, batch * T + 1, T, dtype=torch.int32, device="cuda")
        cache_len = batch + 4
        ssm_states = torch.randn(cache_len, HV, K, V, dtype=torch.float32, device="cuda")
        state_indices = torch.arange(batch, dtype=torch.int32, device="cuda")

        # Reference: contiguous a/b
        ssm_ref = ssm_states.clone()
        out_ref = _run_k_last(A_log, dt_bias, q, k, v, a_c, b_c, ssm_ref,
                              state_indices, cu_seqlens, disable_state_update=True)

        # Test: non-contiguous a/b
        ssm_test = ssm_states.clone()
        out_test = _run_k_last(A_log, dt_bias, q, k, v, a_nc, b_nc, ssm_test,
                               state_indices, cu_seqlens, disable_state_update=True)

        self.assertTrue(
            torch.allclose(out_test, out_ref, rtol=0, atol=0),
            f"output mismatch: max diff = {(out_test - out_ref).abs().max().item()}")
        self.assertTrue(
            torch.allclose(ssm_test, ssm_ref, rtol=0, atol=0),
            f"state mismatch: max diff = {(ssm_test - ssm_ref).abs().max().item()}")

    def test_decode_single_token(self):
        self._run_test(batch=4, T=1, num_k_heads=4, num_v_heads=8)

    def test_multi_token(self):
        self._run_test(batch=4, T=4, num_k_heads=4, num_v_heads=8)

    def test_qwen35_max_decode(self):
        self._run_test(batch=8, T=1, num_k_heads=8, num_v_heads=32)

    def test_qwen35_max_mtp(self):
        self._run_test(batch=4, T=4, num_k_heads=4, num_v_heads=16)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestKLastKDANonContiguous(unittest.TestCase):
    """Test k_last KDA path with non-contiguous a."""

    def _run_test(self, batch, T, num_k_heads, num_v_heads, head_k_dim=128, head_v_dim=128):
        HV = num_v_heads
        K = head_k_dim
        V = head_v_dim

        torch.manual_seed(42)
        A_log = torch.randn(HV, dtype=torch.float32, device="cuda")
        # T == 1 is pure decode (dt_bias at the activation dtype); T > 1 sets
        # disable_state_update below, which makes it an MTP-verify call reading fp32.
        dt_bias = torch.randn(HV, K, device="cuda",
                              dtype=torch.float32 if T > 1 else torch.bfloat16)
        q = torch.randn(1, batch * T, num_k_heads, K, dtype=torch.bfloat16, device="cuda") if T == 1 else \
            torch.randn(batch, T, num_k_heads, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn(1, batch * T, HV, V, dtype=torch.bfloat16, device="cuda") if T == 1 else \
            torch.randn(batch, T, HV, V, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(1, batch * T, HV, dtype=torch.bfloat16, device="cuda") if T == 1 else \
            torch.randn(batch, T, HV, dtype=torch.bfloat16, device="cuda")

        # Non-contiguous a from split (KDA: a is [tokens, HV*K])
        tokens = batch * T
        mixed_a = torch.randn(tokens, 2 * HV * K, dtype=torch.bfloat16, device="cuda")
        a_nc, _ = mixed_a.split([HV * K, HV * K], dim=-1)
        a_c = a_nc.contiguous()

        cu_seqlens = torch.arange(0, batch * T + 1, T, dtype=torch.int32, device="cuda") if T == 1 else \
                     torch.arange(0, batch * T + 1, T, dtype=torch.int32, device="cuda")
        cache_len = batch + 4
        ssm_states = torch.randn(cache_len, HV, K, V, dtype=torch.float32, device="cuda")
        state_indices = torch.arange(batch, dtype=torch.int32, device="cuda")

        disable_su = T > 1

        # Reference: contiguous a
        ssm_ref = ssm_states.clone()
        out_ref = _run_k_last(A_log, dt_bias, q, k, v, a_c, b, ssm_ref,
                              state_indices, cu_seqlens, is_kda=True,
                              disable_state_update=disable_su)

        # Test: non-contiguous a
        ssm_test = ssm_states.clone()
        out_test = _run_k_last(A_log, dt_bias, q, k, v, a_nc, b, ssm_test,
                               state_indices, cu_seqlens, is_kda=True,
                               disable_state_update=disable_su)

        self.assertTrue(
            torch.allclose(out_test, out_ref, rtol=0, atol=0),
            f"KDA output mismatch: max diff = {(out_test - out_ref).abs().max().item()}")
        self.assertTrue(
            torch.allclose(ssm_test, ssm_ref, rtol=0, atol=0),
            f"KDA state mismatch: max diff = {(ssm_test - ssm_ref).abs().max().item()}")

    def test_kda_decode(self):
        self._run_test(batch=4, T=1, num_k_heads=4, num_v_heads=8)

    def test_kda_mtp(self):
        self._run_test(batch=2, T=4, num_k_heads=4, num_v_heads=8)

    def test_kda_decode_large(self):
        self._run_test(batch=16, T=1, num_k_heads=8, num_v_heads=32)


if __name__ == "__main__":
    unittest.main()
