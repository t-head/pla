"""Geometry-based folded GDN scheduling for 39-CU tc02."""

from functools import lru_cache


def _raw_admitted(heads, key_heads, lengths, tokens):
    batch = len(lengths)
    if batch == 1:
        if (heads, key_heads) == (12, 4):
            return 24576 <= tokens <= 32768
        if (heads, key_heads) == (32, 16):
            return 512 <= tokens <= 24576
        if (heads, key_heads) == (64, 8):
            return 1024 <= tokens <= 12288
        return ((heads, key_heads) in ((8, 2), (8, 4), (16, 4), (24, 8))
                and 576 <= tokens <= 32768)

    if (heads, key_heads) == (64, 8):
        return batch == 2 and 8192 <= tokens <= 12288
    if (heads, key_heads) not in ((8, 2), (8, 4), (16, 4), (24, 8)) or batch > 8:
        return False
    longest = max(lengths)
    if longest < 768:
        return False
    if batch == 2 and lengths[-1] == longest and tokens >= 448 * batch:
        return True
    if ((heads == 8 and tokens >= 768 * batch)
            or (heads == 24 and tokens >= 512 * batch)):
        return True
    return (longest >= 1024 and 8192 <= tokens <= 32768 + batch - 1
            and (tokens >= 1024 * batch or tokens - longest <= 3 * (batch - 1)))


def _preactivated_cap(heads, key_heads, chunks):
    batch = len(chunks)
    longest = max(chunks)
    total = sum(chunks)
    if heads >= 32:
        return 0
    if heads == 24:
        return -1 if batch > 1 or longest <= 80 else 0
    if heads == 16:
        # Smaller, balanced packs avoid paying for CP prepare and correct.
        if batch == 2 and 4 * longest <= 3 * total:
            return -1
        if batch == 3 and longest >= 160 and 20 * longest <= 11 * total:
            return -1
        if longest >= 256 and total >= 128 * batch:
            return 0
        return -1 if longest <= 48 or longest * 3 <= total * 2 else 0
    if heads == 12:
        if batch == 1:
            if 64 < longest <= 96:
                return 16
            return 32 if 96 < longest <= 256 else 0
        # Shorter CP segments distribute the longest chain across more CUs.
        if 20 * longest >= 11 * total:
            if batch == 2 and longest >= 64 and (longest >= 160 or 5 * longest >= 3 * total):
                return 32
            if batch == 3 and longest >= 160:
                return 32
        return 0
    if heads == 8:
        if longest <= 16:
            return -1
        # The small-batch serial path needs both short and balanced chains.
        if batch == 3:
            return -1 if longest <= 192 and 50 * longest <= 23 * total else 0
        if batch == 4:
            return -1 if 50 * longest < 19 * total else 0
        if batch >= 5 and 2 * longest <= total:
            return -1
        if key_heads == 2 and batch == 1 and 32 < longest <= 36:
            return 10
        if total <= 128:
            return 16 if longest > 64 else 0
        return 32 if total <= 256 else 0
    # H=6 benefits from more CTAs on sparse medium-length packs.
    if batch == 1 and 160 < longest <= 448:
        return 32
    if batch == 2 and 48 <= longest <= 128 and 5 * longest >= 3 * total:
        return 16
    if batch == 3 and 80 <= longest <= 320 and 20 * longest >= 7 * total:
        return 32
    if batch == 4 and 160 <= longest <= 256 and 50 * longest <= 21 * total:
        return 32
    if longest >= 256 and total >= 128 * batch:
        return 0
    if batch >= 3 and longest * 5 <= total * 2:
        return -1
    if batch > 4:
        return -1
    if batch == 1 and 144 <= longest <= 160:
        return 16
    if batch > 1 and longest >= 96 and total <= 384:
        return 32
    if total <= 256:
        return -1 if total <= 24 * batch else 0
    return (32 if batch <= 3 else -1) if total <= 512 else 0


@lru_cache(maxsize=128)
def folded_schedule(heads, key_heads, offsets, raw_gating, auto_cp, calibrated):
    """Return a CP chunk cap for folded, or None for the established kernel."""
    batch = len(offsets) - 1
    tokens = offsets[-1]
    if tokens < 24 or (heads == key_heads and batch != 1):
        return None
    if (auto_cp and batch == 1 and tokens == 8192
            and (heads, key_heads) in ((6, 2), (12, 4))):
        return None
    if heads == 8 and key_heads == 4 and batch == 1 and 1088 <= tokens <= 1408:
        return None
    if tokens < 48 and heads >= 32 and (heads, key_heads) not in ((32, 4), (32, 16)):
        return None

    lengths = (tokens,) if batch == 1 else tuple(b - a for a, b in zip(offsets, offsets[1:]))
    if min(lengths) <= 0:
        return None
    if raw_gating:
        if not _raw_admitted(heads, key_heads, lengths, tokens):
            return None
        return 10 if (heads, key_heads, batch) == (8, 2, 1) and 1024 < tokens <= 1152 else 0
    if heads == 8 and key_heads == 2 and batch > 32:
        longest = max(lengths)
        if longest < 16384 and tokens - longest <= 3 * (batch - 1):
            return None
    if calibrated and heads >= 6 and heads != key_heads:
        return _preactivated_cap(heads, key_heads, tuple((length + 31) // 32 for length in lengths))

    # Padded and explicitly configured calls retain conservative admission.
    min_length = 512 if (heads, key_heads) == (24, 8) else (1024 if batch == 1 else 4096)
    longest = max(lengths)
    if (tokens >= min_length * batch
            or (heads, key_heads, batch) == (32, 4, 1) and tokens >= 768
            or batch == 1 and tokens == 24 and heads in (8, 16)
            or (heads, key_heads) == (48, 16) and 11 <= batch <= 128 and longest <= 256
            or heads == 8 and key_heads in (2, 4) and (
                batch <= 4 and longest >= 768 and tokens >= 512 * batch
                or key_heads == 2 and batch <= 32 and longest >= 8192
                and tokens - longest <= 3 * (batch - 1))):
        return 0
    return None
