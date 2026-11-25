# SPDX-License-Identifier: Apache-2.0
"""SGLang-compatible signatures, exact-fold verify, commit and cursors."""

from __future__ import annotations

import torch

from ..gdn_replayssm._C import commit as _commit
from ..gdn_replayssm._C import cursors as _cursors
from ..gdn_replayssm._C import verify as _verify

__all__ = [
    "gdn_replayssm_spec_decode",
    "commit_gdn_replayssm_spec",
    "commit_gdn_replayssm_circular",
]

is_tf32_supported = (
    torch.version.hip is None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8
)


def gdn_replayssm_spec_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    rawv_cache: torch.Tensor | None,
    rawk_cache: torch.Tensor | None,
    beta_cache: torch.Tensor | None,
    out: torch.Tensor,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    max_cache_len: int,
    max_spec_len: int,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
    block_v: int = 64,
    num_warps: int = 1,
    num_stages: int = 2,
    nk: int = 2,
    bs_min: int = 4,
    block_v_flush: int = 64,
    num_warps_flush: int = 1,
    num_stages_flush: int = 2,
    nk_flush: int = 2,
    launch_mode: str = "both",
    dot_precision: str = "tf32" if is_tf32_supported else "ieee",
):
    """Verify in place; paired raw buffers without beta encode residuals."""
    modes = {"both": 0, "verify": 1, "flush": 2}
    if launch_mode not in modes:
        raise ValueError("launch_mode must be 'both', 'verify', or 'flush'")
    if dot_precision not in ("tf32", "ieee"):
        raise ValueError("CUDA GDN replay supports dot_precision='tf32' or 'ieee'")
    if launch_mode in ("both", "flush") and any(
        buf is None for buf in (rawv_cache, rawk_cache, beta_cache)
    ):
        raise ValueError("raw replay buffers are required for exact-fold launch")
    _verify(
        q, k, v, a, b, A_log, dt_bias, checkpoint_state,
        d_cache, k_cache, g_cache, rawv_cache, rawk_cache, beta_cache, out,
        query_start_loc, ssm_state_indices, replay_indices,
        write_pos, cache_base, is_flush, max_cache_len, max_spec_len,
        scale, use_qk_l2norm_in_kernel, null_block_id,
        modes[launch_mode], dot_precision == "tf32",
        None, True, 1, 1e-6, False, True,
    )
    return out


def commit_gdn_replayssm_spec(
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    num_accepted: torch.Tensor,
    replay_indices: torch.Tensor,
    max_cache_len: int,
    max_spec_len: int,
    fold_every_commit: bool = False,
    cache_buf_len: int | None = None,
    null_block_id: int = 0,
):
    """Advance request-keyed cursors in place, without host metadata reads."""
    _cursors(
        write_pos, cache_base, is_flush, num_accepted, replay_indices,
        max_cache_len, max_spec_len, fold_every_commit, cache_buf_len, null_block_id,
    )


def commit_gdn_replayssm_circular(
    *,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
    state_batch_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accept_lens: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:
    """Fold all layers and optional snapshots, then finish SGLang metadata."""
    _commit(
        checkpoint_state, d_cache, k_cache, g_cache,
        d_residual_cache, k_residual_cache, state_batch_indices, replay_indices,
        write_pos, cache_base, is_flush, accept_lens,
        mamba_track_indices, mamba_steps_to_track, null_block_id,
    )
