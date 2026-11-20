# SPDX-License-Identifier: Apache-2.0
"""Framework-independent GDN replay tensors and numerical policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

import torch


class HistoryFormat(str, Enum):
    COMPACT_DKG_RING = "compact_dkg_ring"
    RAW_VKG_BETA_WINDOW = "raw_vkg_beta_window"
    CACHED_UKG_DOUBLE = "cached_ukG_double"


@dataclass(frozen=True)
class NumericPolicy:
    """TF32 preserves source operand rounding, not an unquantized IEEE error bound."""

    qk_norm: str = "l2_inside_sqrt"
    epsilon: float = 1e-6
    gate_mode: str = "raw"
    dot_precision: str = "tf32"  # Projection precision; materialization accumulates in FP32.

    def __post_init__(self):
        if self.qk_norm not in ("none", "l2_inside_sqrt", "l2_outside_sqrt"):
            raise ValueError("qk_norm must be 'none', 'l2_inside_sqrt', or 'l2_outside_sqrt'")
        if self.gate_mode not in ("raw", "processed"):
            raise ValueError("gate_mode must be 'raw' or 'processed'")
        if self.dot_precision not in ("ieee", "tf32"):
            raise ValueError("dot_precision must be 'ieee' or 'tf32'")
        if (not isinstance(self.epsilon, Real) or isinstance(self.epsilon, bool)
                or not math.isfinite(self.epsilon)
                or not 0 <= self.epsilon <= 3.4028234663852886e38):
            raise ValueError("epsilon must be a finite, nonnegative float32 host scalar")


@dataclass(frozen=True)
class ReplayInputs:
    # q/k: [tokens, H, K]; v: [tokens, HV, V]; gate/beta: [tokens, HV].
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor
    A_log: torch.Tensor | None = None
    dt_bias: torch.Tensor | None = None
    scale: float | None = None


@dataclass(frozen=True)
class ReplayHistory:
    # Flush/reset before changing format; populated buffers cannot be reinterpreted.
    format: HistoryFormat
    # Materialize uses layered buffers; decode/verify consume one layer.
    values: torch.Tensor  # Compact D, cached U, or raw V; never interchangeable.
    keys: torch.Tensor  # Normalized except for raw-window history.
    gates: torch.Tensor  # Per-step g, or checkpoint-relative cumulative G for cached double.
    value_residual: torch.Tensor | None = None  # Low parts are consumed by materialization, not history projection.
    key_residual: torch.Tensor | None = None
    beta: torch.Tensor | None = None


@dataclass(frozen=True)
class ReplayMapping:
    # Both mappings are indexed by work row, not replay slot.
    state_indices: torch.Tensor
    replay_indices: torch.Tensor
    null_block_id: int = -1


@dataclass(frozen=True)
class ReplayStep:
    # Replay-slot metadata is read-only; cached starts, if supplied, must be zero.
    lengths: torch.Tensor
    starts: torch.Tensor | None = None
    buffers: torch.Tensor | None = None


@dataclass(frozen=True)
class MaterializeWork:
    # The source checkpoint must already precede the selected history range.
    state_indices: torch.Tensor | None = None
    replay_indices: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    starts: torch.Tensor | None = None
    buffers: torch.Tensor | None = None
    # Selects lengths/starts/buffers/enabled; None selects the work row.
    metadata_indices: torch.Tensor | None = None
    target_state_indices: torch.Tensor | None = None
    enabled: torch.Tensor | None = None
    # Native Holmes [B, 4]: ignored, shared state/replay slot, count, buffer.
    items: torch.Tensor | None = None
    n_items: torch.Tensor | None = None
