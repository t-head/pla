# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
"""Vendored KDA reference implementations (pure PyTorch).

Copied/adapted from fla-org/flash-linear-attention:
  - fla/ops/kda/naive.py  : naive_recurrent_kda
  - fla/ops/kda/gate.py   : naive_kda_gate, naive_kda_lowerbound_gate

These are the numerical *golden references* that vLLM's vendored KDA op
(``vllm.third_party.flash_linear_attention.ops.kda``) is validated against.
We vendor them here so that ``scripts/decode/test_kda_decode.py`` can exercise the PLA PPU
CUDA implementation (``pla.decode.kda.fused_recurrent_kda``) without depending on an
upstream ``fla`` install (which would collide with this workspace's ``pla``
package name).

License: MIT (see fla-org/flash-linear-attention).
"""

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gate references (fla/ops/kda/gate.py)
# ---------------------------------------------------------------------------
def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """g = -A_log.exp() * softplus(g + dt_bias)."""
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = (-A_log.view(H, 1).float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: float = -5.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


# ---------------------------------------------------------------------------
# Recurrent / chunk references (fla/ops/kda/naive.py)
# ---------------------------------------------------------------------------
def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    """Reference recurrent KDA. q/k [B,T,H,K], v/g [B,T,HV,V/K], beta [B,T,HV].

    Returns ``(o [B,T,HV,V], S [B,HV,K,V] or None)``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            'b h k, b h v -> b h k v',
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S

# ---------------------------------------------------------------------------
# Test helper (mirrors fla.utils.assert_close)
# ---------------------------------------------------------------------------
def assert_close(prefix: str, ref: torch.Tensor, tri: torch.Tensor, ratio: float,
                 warning: bool = False):
    ref = ref.float()
    tri = tri.float()
    # relative L2 error (stable near zero, matches fla-org's assert_close spirit)
    denom = ref.norm().clamp_min(1e-8)
    rel = ((ref - tri).norm() / denom).item()
    msg = f"{prefix}: relative L2 diff = {rel:.6f} (tol={ratio})"
    if rel > ratio:
        if warning:
            print(f"[WARN] {msg}")
        else:
            raise AssertionError(msg)
    else:
        print(msg)
