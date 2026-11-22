"""Measured tc01 serial regions for independent coefficient preparation."""


def tc01_packed_coefficients_mode(
    major: int, minor: int, num_tokens: int, heads: int, key_heads: int,
    *, batch_size: int, indexed_state: bool, full_sequence: bool,
    force_cp: bool = False, group_override: bool = False, available: bool = True,
) -> int:
    """Return 0 for ordinary, 1 for short fused KKT, 2 for packed-only."""
    if ((major, minor) != (8, 0) or batch_size != 1 or not indexed_state or
            not full_sequence or force_cp or group_override or not available):
        return 0
    # The complete 1..512 sweep found tiny cases at parity/slightly slower;
    # retain their ordinary path and admit the measured two-chunk boundary.
    if heads == 8 and key_heads in (2, 4) and 64 <= num_tokens <= 512:
        return 1
    # The six-process canonical and boundary sweeps qualify the long region.
    # Keep short sequences and lengths beyond the measured upper bound ordinary.
    if heads == 16 and key_heads == 2 and 1291 <= num_tokens <= 32768:
        return 2
    return 0
