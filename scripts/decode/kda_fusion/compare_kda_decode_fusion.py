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

"""Kimi K3 KDA decode accuracy comparison -- PPU fused kernel vs reference implementation.

Two precision modes, selected with ``--dtype``:

  * ``--dtype bf16`` (default): reference = the **inlined vllm triton chain**
        1) ``causal_conv1d_update``                (kda_fusion._vllm_triton_ref)
        2) ``fused_recurrent_kda_packed_decode``   (kda_fusion._vllm_triton_ref)
        if fuse_output_norm=True, gated RMSNorm is then run manually in torch.
        Criteria: output atol=1e-2/rtol=1e-2, state atol=1e-3/rtol=1e-3, conv 0/0.
        (Tightened from vllm's 3e-2; measured upper bounds on the 24-case HxB grid: output 1.56e-2, state 5.5e-4.)

  * ``--dtype fp16``: vllm triton does not support fp16; the reference becomes a **torch-naive f32
        golden** (the bf16 tensors from build_inputs are cast to fp16 and fed to PPU; computed per head, per token
        in f32, with a final cast to fp16). Criteria: atol=1e-3, rtol=1e-2; conv allows
        fp16 rounding.

The path under test is always the PPU fused kernel ``fused_kda_decode_mega_forward`` (default impl).

Usage:
    python scripts/decode/kda_fusion/compare_kda_decode_fusion.py            # bf16
    python scripts/decode/kda_fusion/compare_kda_decode_fusion.py --dtype fp16
    python scripts/decode/kda_fusion/compare_kda_decode_fusion.py --json out.json
"""

# ruff: noqa: E501
import argparse
import json
import math
import os
import sys

import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))                # .../scripts/decode/kda_fusion
PARENT_DIR = os.path.dirname(SCRIPTS_DIR)                               # .../scripts/decode
# NOTE: do NOT add the repo root to sys.path. The source tree's ``pla/``
# package would shadow the installed one and has no compiled ``_C``.
for _p in (SCRIPTS_DIR, PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Allow importing the module either as ``kda_fusion._vllm_triton_ref`` or directly from the same directory.
try:
    from kda_fusion._vllm_triton_ref import (  # noqa: E402
        causal_conv1d_update,
        fused_recurrent_kda_packed_decode,
    )
except ModuleNotFoundError:
    from _vllm_triton_ref import (  # noqa: E402
        causal_conv1d_update,
        fused_recurrent_kda_packed_decode,
    )

# Path under test -- PPU fused decode
from pla.decode.kda import fused_kda_decode_mega_forward  # noqa: E402


def fused_kda_decode_ppu_forward(*args, **kwargs):
    """PPU path of the unified mega entry (default impl)."""
    os.environ.pop("KDA_DECODE_IMPL", None)
    return fused_kda_decode_mega_forward(*args, **kwargs)


DEVICE = "cuda"
D = 128         # head dim
W = 4           # conv width
NORM_EPS = 1e-5

# bf16 thresholds (tightened from vllm's 3e-2; empirical ceiling across 24-case H×B grid):
#   output max_abs ~1.56e-2 (at H=24/96 B=64), state max_abs ~5.5e-4.
VLLM_OUT_ATOL = 1e-2
VLLM_OUT_RTOL = 1e-2
VLLM_STATE_ATOL = 1e-3
VLLM_STATE_RTOL = 1e-3

# CUDA PLA fp16 thresholds (aligned with the fp16 branch of vllm tests/models/kimi_k3/test_kda.py)
FP16_ATOL = 1e-3
FP16_RTOL = 1e-2

SCALE = 1.0 / math.sqrt(D)

# (num_heads, num_seqs, lower_bound, fuse_output_norm) -- supported-head smoke matrix
K3_CASES = [
    (4, 1, -5.0, True),
    (6, 1, -5.0, True),
    (6, 4, None, False),
    (6, 16, -5.0, True),
    (8, 1, -5.0, True),
    (12, 1, -5.0, True),
    (12, 4, None, False),
    (16, 1, -5.0, True),
    (24, 4, None, False),
    (48, 1, -5.0, True),
    (64, 1, -5.0, True),
    (96, 1, -5.0, True),
]


# ---------------------------------------------------------------------------
# Input construction: field-by-field identical to vllm test_fused_kda_decode_correctness (bf16 layout)
# ---------------------------------------------------------------------------
def build_inputs(num_heads, num_seqs, seed):
    torch.manual_seed(seed)
    dim = num_heads * D
    slots = num_seqs + 2

    packed_x_storage = torch.randn(
        num_seqs, 3 * dim + 17, dtype=torch.bfloat16, device=DEVICE
    )
    packed_x = packed_x_storage[:, : 3 * dim]
    weight = 0.1 * torch.randn(3 * dim, W, dtype=torch.float32, device=DEVICE)
    conv_seed = 0.1 * torch.randn(
        slots, W - 1, 3 * dim, dtype=torch.bfloat16, device=DEVICE
    ).transpose(1, 2)
    raw_g = torch.randn(
        1, num_seqs, num_heads, D, dtype=torch.bfloat16, device=DEVICE
    )
    raw_beta_storage = torch.randn(
        1, num_seqs, num_heads + 1, dtype=torch.bfloat16, device=DEVICE
    )
    raw_beta = raw_beta_storage[:, :, :num_heads]
    output_gate_storage = torch.randn(
        num_seqs, dim + 7, dtype=torch.bfloat16, device=DEVICE
    )
    output_gate = output_gate_storage[:, :dim].view(num_seqs, num_heads, D)
    norm_weight = torch.randn(D, dtype=torch.float32, device=DEVICE)
    A_log = 0.5 * torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
    dt_bias = 0.1 * torch.randn(dim, dtype=torch.float32, device=DEVICE)
    state_indices = torch.arange(
        num_seqs, 0, -1, dtype=torch.int32, device=DEVICE
    )
    state_seed = 0.01 * torch.randn(
        slots, num_heads, D, D, dtype=torch.float32, device=DEVICE
    )
    # PPU mega kernel takes fused width-major weight [3, W, dim]
    fused_weight = weight.reshape(3, dim, W).transpose(1, 2).contiguous()

    return {
        "num_heads": num_heads,
        "num_seqs": num_seqs,
        "dim": dim,
        "slots": slots,
        "packed_x": packed_x,
        "weight": weight,
        "fused_weight": fused_weight,
        "conv_seed": conv_seed,
        "raw_g": raw_g,
        "raw_beta": raw_beta,
        "output_gate": output_gate,
        "norm_weight": norm_weight,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_indices": state_indices,
        "state_seed": state_seed,
    }


def _to_fp16(inp):
    """Cast the bf16 activation tensors from build_inputs to fp16 (f32/int stay unchanged)."""
    out = dict(inp)
    for k in ("packed_x", "conv_seed", "raw_g", "raw_beta", "output_gate"):
        out[k] = inp[k].to(torch.float16)
    return out


# ---------------------------------------------------------------------------
# Reference path A: bf16 vllm triton chain (default)
# ---------------------------------------------------------------------------
def vllm_reference(inp, lower_bound, fuse_output_norm):
    """Run the inlined vllm triton chain and return (expected_out, state_ref, conv_ref)."""
    conv_ref = inp["conv_seed"].clone()
    state_ref = inp["state_seed"].clone()

    mixed_qkv = causal_conv1d_update(
        inp["packed_x"],
        conv_ref,
        inp["weight"],
        activation="silu",
        conv_state_indices=inp["state_indices"],
        validate_data=True,
        out=torch.empty_like(inp["packed_x"]),
    )
    expected, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed_qkv,
        raw_g=inp["raw_g"],
        raw_beta=inp["raw_beta"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        lower_bound=lower_bound,
        initial_state=state_ref,
        state_indices=inp["state_indices"],
    )

    if fuse_output_norm:
        expected_float = expected.float()
        expected = (
            expected_float
            * torch.rsqrt(
                expected_float.square().mean(dim=-1, keepdim=True) + NORM_EPS
            )
            * inp["norm_weight"]
            * inp["output_gate"].float().sigmoid().unsqueeze(0)
        ).to(expected.dtype)

    torch.cuda.synchronize()
    return expected, state_ref, conv_ref


# ---------------------------------------------------------------------------
# Reference path B: fp16 torch-naive f32 golden
# ---------------------------------------------------------------------------
def _silu(x):
    return x * torch.sigmoid(x)


def torch_reference_fp16(inp, lower_bound, fuse_output_norm):
    """Step-by-step f32 reference; output/conv are cast to fp16 at the end."""
    num_seqs = inp["num_seqs"]
    H = inp["num_heads"]
    dim = H * D

    packed_x = inp["packed_x"].float()
    weight = inp["fused_weight"].float()
    conv = inp["conv_seed"].float().clone()
    raw_g = inp["raw_g"].float()
    raw_beta = inp["raw_beta"].float()
    A_log = inp["A_log"].float()
    dt_bias = inp["dt_bias"].float()
    state = inp["state_seed"].float().clone()
    state_indices = inp["state_indices"].long()
    output_gate = inp["output_gate"]
    norm_weight = inp["norm_weight"].float()

    out = torch.empty(1, num_seqs, H, D, dtype=torch.float16, device=DEVICE)

    for b in range(num_seqs):
        slot = int(state_indices[b].item())
        mixed = torch.empty(3 * dim, dtype=torch.float32, device=DEVICE)
        for seg in range(3):
            for ch in range(dim):
                c = seg * dim + ch
                taps = conv[slot, c, :]
                x_new = packed_x[b, c]
                window = torch.cat([taps, x_new.view(1)])
                acc = torch.dot(weight[seg, :, ch], window)
                mixed[c] = _silu(acc)
                conv[slot, c, 0] = taps[1]
                conv[slot, c, 1] = taps[2]
                conv[slot, c, 2] = x_new
        q = mixed[:dim].view(H, D)
        k = mixed[dim:2 * dim].view(H, D)
        v = mixed[2 * dim:].view(H, D)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        exp_a = A_log.exp()
        g_raw = raw_g[0, b] + dt_bias.view(H, D)
        if lower_bound is not None:
            decay = torch.exp(lower_bound * torch.sigmoid(exp_a[:, None] * g_raw))
        else:
            sp = torch.nn.functional.softplus(g_raw)
            decay = torch.exp(-exp_a[:, None] * sp)
        beta = torch.sigmoid(raw_beta[0, b])
        o = torch.empty(H, D, dtype=torch.float32, device=DEVICE)
        for h in range(H):
            S = state[slot, h]
            S = S * decay[h][None, :]
            dot_hk = S @ k[h]
            v_new = (v[h] - dot_hk) * beta[h]
            S = S + torch.outer(v_new, k[h])
            o[h] = S @ q[h]
            state[slot, h] = S
        if fuse_output_norm:
            rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + NORM_EPS)
            gate = torch.sigmoid(output_gate[b].float())
            o = o * rstd * norm_weight[None, :] * gate
        out[0, b] = o.to(torch.float16)

    return out, state, conv.to(torch.float16)


# ---------------------------------------------------------------------------
# PPU paged cache allocation (bf16 conv / f32 state or fp16 conv / f32 state)
# ---------------------------------------------------------------------------
def _alloc_paged_cache(inp, conv_dtype):
    num_seqs = inp["num_seqs"]
    num_heads = inp["num_heads"]
    dim = inp["dim"]
    slots = inp["slots"]
    conv_slot_elements = 3 * dim * (W - 1)
    state_slot_elements = num_heads * D * D
    conv_slot_bytes = conv_slot_elements * conv_dtype.itemsize
    page_bytes = conv_slot_bytes + state_slot_elements * torch.float32.itemsize
    cache_storage = torch.empty(slots * page_bytes, dtype=torch.uint8, device=DEVICE)
    conv_actual = torch.as_strided(
        cache_storage.view(conv_dtype),
        size=(slots, 3 * dim, W - 1),
        stride=(page_bytes // conv_dtype.itemsize, 1, 3 * dim),
    )
    state_actual = torch.as_strided(
        cache_storage.view(torch.float32),
        size=(slots, num_heads, D, D),
        stride=(page_bytes // torch.float32.itemsize, D * D, D, 1),
        storage_offset=conv_slot_bytes // torch.float32.itemsize,
    )
    conv_actual.copy_(inp["conv_seed"])
    state_actual.copy_(inp["state_seed"])
    return conv_actual, state_actual


def run_kernel(fn, inp, lower_bound, fuse_output_norm, out_dtype):
    conv, state = _alloc_paged_cache(inp, conv_dtype=out_dtype)
    out = torch.empty(
        1, inp["num_seqs"], inp["num_heads"], D,
        dtype=out_dtype, device=DEVICE,
    )
    fn(
        x=inp["packed_x"], weight=inp["fused_weight"], bias=None,
        conv_state=conv, raw_g=inp["raw_g"], raw_beta=inp["raw_beta"],
        a_log=inp["A_log"], dt_bias=inp["dt_bias"], state_indices=inp["state_indices"],
        state=state, out=out, lower_bound=lower_bound,
        output_gate=inp["output_gate"] if fuse_output_norm else None,
        norm_weight=inp["norm_weight"] if fuse_output_norm else None,
        norm_eps=NORM_EPS,
    )
    torch.cuda.synchronize()
    return out, state, conv


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------
def diff(a, b):
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    return d.max().item(), (d / denom).max().item()


def _allclose(a, b, atol, rtol):
    try:
        torch.testing.assert_close(a.detach(), b.detach(), atol=atol, rtol=rtol)
        return True
    except AssertionError:
        return False


def vllm_out_allclose(a, b):
    return _allclose(a, b, atol=VLLM_OUT_ATOL, rtol=VLLM_OUT_RTOL)


def vllm_state_allclose(a, b):
    return _allclose(a, b, atol=VLLM_STATE_ATOL, rtol=VLLM_STATE_RTOL)


def fp16_allclose(a, b):
    return _allclose(a, b, atol=FP16_ATOL, rtol=FP16_RTOL)


def conv_exact(a, b):
    return _allclose(a, b, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                    help="bf16: inline vllm triton chain (default); "
                         "fp16: torch-naive f32 golden")
    ap.add_argument("--json", default=None, help="dump results JSON here")
    args = ap.parse_args()

    use_fp16 = args.dtype == "fp16"
    out_dtype = torch.float16 if use_fp16 else torch.bfloat16
    if use_fp16:
        out_atol, out_rtol = FP16_ATOL, FP16_RTOL
        state_atol, state_rtol = FP16_ATOL, FP16_RTOL
        conv_judge = fp16_allclose        # conv allows fp16 rounding
    else:
        out_atol, out_rtol = VLLM_OUT_ATOL, VLLM_OUT_RTOL
        state_atol, state_rtol = VLLM_STATE_ATOL, VLLM_STATE_RTOL
        conv_judge = conv_exact           # conv is bit-exact

    print("=" * 108)
    if use_fp16:
        print("Kimi K3 KDA decode accuracy comparison: PPU fused_kda_decode_mega vs torch-naive f32 golden (fp16 cast)")
        print(f"  judge          : atol={out_atol:.0e}, rtol={out_rtol:.0e} (conv uses the same tolerance)")
    else:
        print("Kimi K3 KDA decode accuracy comparison: PPU fused_kda_decode_mega vs inline vllm triton chain")
        print("  ref chain      : causal_conv1d_update -> fused_recurrent_kda_packed_decode (+torch RMSNorm)")
        print(f"  judge          : out atol={out_atol:.0e}/rtol={out_rtol:.0e}, "
              f"state atol={state_atol:.0e}/rtol={state_rtol:.0e} (conv 0/0)")
    print("=" * 108)

    results = []
    all_ok = True
    for (H, B, lb, fon) in K3_CASES:
        seed = 967 + H + B
        inp = build_inputs(H, B, seed)
        if use_fp16:
            inp = _to_fp16(inp)

        if use_fp16:
            expected_out, ref_state, ref_conv = torch_reference_fp16(inp, lb, fon)
        else:
            expected_out, ref_state, ref_conv = vllm_reference(inp, lb, fon)

        ppu_out, ppu_state, ppu_conv = run_kernel(
            fused_kda_decode_ppu_forward, inp, lb, fon, out_dtype=out_dtype,
        )

        o_abs, o_rel = diff(ppu_out, expected_out)
        s_abs, s_rel = diff(ppu_state, ref_state)
        c_abs, _ = diff(ppu_conv, ref_conv)

        judge_out = _allclose(ppu_out, expected_out, atol=out_atol, rtol=out_rtol)
        judge_state = _allclose(ppu_state, ref_state, atol=state_atol, rtol=state_rtol)
        judge_conv = conv_judge(ppu_conv, ref_conv)
        ok = judge_out and judge_state and judge_conv
        all_ok = all_ok and ok

        tag = f"H={H:<3} B={B} lb={str(lb):<5} fon={fon}"
        flags = "" if ok else f" [out={judge_out},state={judge_state},conv={judge_conv}]"
        print(
            f"[{'PASS' if ok else 'FAIL'}] {tag} | "
            f"o_abs={o_abs:.2e} o_rel={o_rel:.2e}  "
            f"s_abs={s_abs:.2e} s_rel={s_rel:.2e}  "
            f"conv_abs={c_abs:.1e}{flags}",
            flush=True,
        )

        results.append({
            "num_heads": H, "num_seqs": B,
            "lower_bound": lb, "fuse_output_norm": fon,
            "out_abs": o_abs, "out_rel": o_rel,
            "state_abs": s_abs, "state_rel": s_rel,
            "conv_abs": c_abs,
            "out_ok": judge_out, "state_ok": judge_state, "conv_ok": judge_conv,
            "pass": ok,
        })

    print("=" * 108)
    print(
        f"OVERALL ({args.dtype}): {'ALL PASS' if all_ok else 'SOME FAILED'}  "
        f"(out atol={out_atol:.0e}/rtol={out_rtol:.0e}, "
        f"state atol={state_atol:.0e}/rtol={state_rtol:.0e}"
        + ("" if use_fp16 else ", conv exact") + ")"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "cases": results, "all_pass": all_ok,
                "dtype": args.dtype,
                "out_atol": out_atol, "out_rtol": out_rtol,
                "state_atol": state_atol, "state_rtol": state_rtol,
            }, f, indent=2)
        print(f"wrote {args.json}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
