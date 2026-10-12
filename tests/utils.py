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

"""Common utilities for tests/test_op_forge.py and tests/test_op_forge_perf.py.

Key points extracted:
1. caselist / format string parsing (parse_pla_string_re / read_cmds_from_file / ...)
2. dtype parsing (parse_dtype)
3. KDA auto-detection (detect_is_kda_* / resolve_is_kda)
4. MTP mode expansion: T=1 -> single_chain (topk=1), T>1 -> eagle_tree (topk=eagle_topk)
5. seqlen_q normalization / replacement helpers
"""
import os
import re
import ast

import torch


# ============================================================
# dtype parsing
# ============================================================
_DTYPE_MAP = {
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
}


def parse_dtype(dtype_str):
    """Parse dtype string (case-insensitive) into torch.dtype.
    Returns torch.float16 by default if input is None/empty.
    """
    if dtype_str is None or str(dtype_str).strip() == "":
        return torch.float16
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    key = str(dtype_str).lower().strip()
    if key not in _DTYPE_MAP:
        print(f"WARNING: unsupported dtype '{dtype_str}', fallback to fp16")
        return torch.float16
    return _DTYPE_MAP[key]


# ============================================================
# Safe numeric conversion
# ============================================================
def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')


def safe_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ============================================================
# KDA / Kimi auto-detection
# ============================================================
# Example caselist line: [KDA]:[D][1]kda_attn:seqlen_q:1,...
KDA_TAG_RE = re.compile(r"\[KDA\]|kda_attn", re.IGNORECASE)


def detect_is_kda_in_text(text):
    """Detect whether the string contains a KDA marker (works for both caselist lines and format strings)."""
    if text is None:
        return False
    return bool(KDA_TAG_RE.search(str(text)))


def detect_is_kda_in_path(path):
    """Detect whether the caselist path contains a KDA model marker (Kimi / kda).
    e.g. Kimi-Linear-48B-A3B_decode_tp1_..._kda_op.list
    """
    if path is None:
        return False
    base = os.path.basename(str(path)).lower()
    return ("kimi" in base) or ("kda" in base)


def resolve_is_kda(force_flag=False, raw_line=None, format_str=None, caselist_path=None):
    """Decide is_kda from all signals combined. True if any of them is true.
    Priority (short-circuit evaluation):
      1. force_flag (forced by the --is_kda CLI)
      2. raw_line contains [KDA]/kda_attn  (a single caselist line)
      3. format_str contains [KDA]/kda_attn (--format mode)
      4. caselist_path basename contains Kimi/kda (file-name fallback)
    """
    if force_flag:
        return True
    if detect_is_kda_in_text(raw_line):
        return True
    if detect_is_kda_in_text(format_str):
        return True
    if detect_is_kda_in_path(caselist_path):
        return True
    return False


# ============================================================
# caselist / format string parsing
# ============================================================
def parse_pla_string_re(s):
    """Parse a parameter dict from a caselist line or a --format string.
    Example input: [GDN]:[D][1]gdn_attn:seqlen_q:1,head_dim_qk:128,...,batch_size:1,dtype:bf16,tp:2.
    Unequal-length packed inputs use cu_seqlens_q:[0,24,184,358] (or cu_seqlens), and seqlen_q may be absent.
    """
    result = {}
    op_match = re.search(r"(?:gdn|kda)_attn\s*:", s, re.IGNORECASE)
    if op_match:
        s = s[op_match.end():]
    elif "seqlen_q" in s:
        s = s[s.index("seqlen_q"):]
    else:
        print("ERROR: not found gdn_attn, kda_attn, or seqlen_q")
        return result
    s = s.strip(".")
    param_pattern = r"(\w+):(\[.*?\]|[^,]+)"

    match_string = re.findall(param_pattern, s)
    if len(match_string) == 0:
        print("ERROR: wrong pla case format input, please check!!")
    for key, value in match_string:
        if value.startswith('[') and value.endswith(']'):
            result[key] = ast.literal_eval(value)
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


def resolve_cu_seqlens(case):
    """Return and validate explicit packed offsets from an OpForge case."""
    values = case.get("cu_seqlens")
    values_q = case.get("cu_seqlens_q")
    if values is not None and values_q is not None and values != values_q:
        raise ValueError("cu_seqlens and cu_seqlens_q must match when both are provided")
    if values is None:
        values = values_q
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("cu_seqlens must be a list with at least two offsets")

    try:
        offsets = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("cu_seqlens offsets must be integers") from exc
    if any(isinstance(value, bool) or offset != value for value, offset in zip(values, offsets)):
        raise ValueError("cu_seqlens offsets must be integers")
    if offsets[0] != 0:
        raise ValueError("cu_seqlens must start at 0")
    if any(end <= start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be strictly increasing")

    batch_size = case.get("batch_size")
    if batch_size is not None and int(batch_size) != len(offsets) - 1:
        raise ValueError(
            f"batch_size={batch_size} does not match cu_seqlens batch size {len(offsets) - 1}"
        )
    return offsets


def read_cmds_from_file(casefile):
    """Read a caselist file and return a list of case dicts.
    Each dict additionally carries a '_raw_line' field (the raw line string) for downstream KDA auto-detection.
    """
    pla_cases = []
    with open(casefile, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            case = parse_pla_string_re(line)
            if case:
                case['_raw_line'] = line
                pla_cases.append(case)
    return pla_cases


def read_cmds_raw_from_file(casefile):
    """Read a caselist file and return the raw string lines (unparsed, for perf to feed into acu commands)."""
    pla_cases = []
    with open(casefile, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            pla_cases.append(line)
    return pla_cases


def collect_cases_from_path(path):
    """Collect .list caselists from a single file or a directory recursively. Returns [{file, cases}, ...]."""
    all_cases = []
    case_files = []

    if os.path.isfile(path):
        case_files = [path]
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('.list'):
                    case_files.append(os.path.join(root, filename))
    else:
        print(f"ERROR: path not found: {path}")
        return [], []

    for case_file in case_files:
        cases = read_cmds_from_file(case_file)
        if cases:
            all_cases.append({'file': case_file, 'cases': cases})

    return all_cases, case_files


# ============================================================
# seqlen_q helpers
# ============================================================
def normalize_seqlen_q(case_str, kernel):
    """Force seqlen_q=1 for plain decode cases.
    MTP cases and varlen prefill cases with explicit cu_seqlens keep their original values.
    """
    if kernel == "k_last_mtp" or (
        kernel == "k_last" and re.search(r"\bcu_seqlens(?:_q)?\s*:", case_str)
    ):
        return case_str
    return re.sub(r"(seqlen_q\s*:\s*)\d+", r"\g<1>1", case_str)


def replace_seqlen_q(case_str, new_T):
    """Replace seqlen_q:<n> in case_str with seqlen_q:<new_T>."""
    return re.sub(r"(seqlen_q\s*:\s*)\d+", rf"\g<1>{int(new_T)}", case_str)


# ============================================================
# MTP mode expansion
# ============================================================
# sglang semantics (see srt/speculative/dflash_info.py and srt/server_args.py):
#   * the kernel's T == spec_info.draft_token_num, i.e. the verify window length,
#     and is **independent** of topk; topk only decides whether tree metadata is enabled.
#   * DFLASH (single chain / linear verify): forces topk=1, num_steps=1, but
#     draft_token_num = dflash_block_size, 16 by default. So **the T of a single chain is also >=3**.
#   * EAGLE (tree): topk>1, and T is likewise draft_token_num.
# Hence T=1 is not a valid MTP config (T=1 is plain decode and should go through k_last).
# The decode cases in the caselists all have seqlen_q:1, which is meaningless for MTP, so it must be
# replaced by draft_token_num.
DEFAULT_MTP_EAGLE_TOPK = 2  # for historical callers, single-value default topk
DEFAULT_MTP_EAGLE_T = 4     # for historical callers, single-value default T
# topk=1 single_chain (run for every case); topk>1 eagle_tree (run only for boundary cases).
# eagle takes only the two boundary topk values 2/5 by default (to avoid the heavy cost of the tree reload path).
DEFAULT_MTP_TOPKS = [1, 2, 5]
# draft_token_num (T) covered by default: 4 = small window, 16 = sglang DFLASH default block_size
DEFAULT_MTP_DRAFT_TOKEN_NUMS = [4, 16]
MIN_MTP_DRAFT_TOKEN_NUM = 3  # T >= 3 in real MTP usage


def resolve_mtp_topk(T, eagle_topk=DEFAULT_MTP_EAGLE_TOPK):
    """Legacy-compatible interface: guess topk from T.

    Note: under the new semantics T and topk are decoupled; this function is kept only for old callers.
    """
    return 1 if int(T) == 1 else int(eagle_topk)


def _topk_label(topk, T=None):
    base = "single_chain" if int(topk) == 1 else f"eagle_tree_topk{int(topk)}"
    return base if T is None else f"{base}_T{int(T)}"


def normalize_mtp_draft_token_num(T):
    """Raise an invalid MTP window length (T<MIN, typically seqlen_q=1 from a caselist)
    to MIN_MTP_DRAFT_TOKEN_NUM.
    """
    T = int(T)
    return T if T >= MIN_MTP_DRAFT_TOKEN_NUM else MIN_MTP_DRAFT_TOKEN_NUM


def mtp_shape_key(case):
    """Shape sort key for picking boundary cases: (batch_size, num_heads_v, num_heads_qk).
    H is usually constant within one list, in which case this is equivalent to sorting by batch_size;
    H is kept as a secondary key to support lists with mixed H.
    """
    return (int(case.get('batch_size', 1)),
            int(case.get('num_heads_v', case.get('num_heads_qk', 0))),
            int(case.get('num_heads_qk', 0)))


def select_eagle_boundary_indices(cases):
    """Return the set of case indices that should run eagle tree: the min(B,H) and max(B,H) boundaries.

    cases: list of parsed case dicts (in the caselist's original order).
    The functional and perf sides compute it with the same function and order, so the boundary sets match.
    """
    if not cases:
        return set()
    keys = [mtp_shape_key(c) for c in cases]
    imin = min(range(len(keys)), key=lambda i: keys[i])
    imax = max(range(len(keys)), key=lambda i: keys[i])
    return {imin, imax}


def mtp_subcase_specs(seqlen_q, has_explicit_topk=False, explicit_topk=None,
                      topks=None, draft_token_nums=None, include_eagle=True):
    """The **single source of truth** for MTP sub-item expansion (shared by test_op_forge.py and test_op_forge_perf.py).

    Returns [(T, topk, label), ...]; the order must be stable: the k-th kernel instance of perf-side ACU
    must map strictly 1:1 to the k-th sub-case on the functional side.

    Rules:
      1. explicit topk: respect it and generate a single sub-item; T is seqlen_q (raised to MIN if too small).
      2. single_chain (topk=1): respect the verify window carried by the caselist (seqlen_q>1,
         e.g. seqlen_q:3); pure decode cases with seqlen_q<=1 use draft_token_nums instead.
      3. eagle_tree (topk>1): **always uses draft_token_nums** (default 4/16), independent of the case's own
         seqlen_q, so the T combinations covered by eagle are deterministic.
    With include_eagle=False only single_chain is kept and eagle is dropped; used for "eagle tree runs only on boundary
    cases". Explicit topk is not affected by include_eagle.
    """
    topks = [int(k) for k in (topks if topks else DEFAULT_MTP_TOPKS)]
    dtns = [normalize_mtp_draft_token_num(t)
            for t in (draft_token_nums if draft_token_nums else DEFAULT_MTP_DRAFT_TOKEN_NUMS)]
    # deduplicate while preserving order
    seen = set()
    dtns = [t for t in dtns if not (t in seen or seen.add(t))]

    seqlen_q = int(seqlen_q or 1)

    if has_explicit_topk:
        topk = int(explicit_topk)
        T = normalize_mtp_draft_token_num(seqlen_q)
        return [(T, topk, _topk_label(topk, T))]

    # T of single_chain: respect the window carried by the caselist, otherwise use draft_token_nums
    sc_ts = [normalize_mtp_draft_token_num(seqlen_q)] if seqlen_q > 1 else list(dtns)
    # T of eagle: always draft_token_nums, never drifting with the case's seqlen_q
    eagle_ts = list(dtns)
    eagle_ks = [k for k in topks if k > 1]

    out = []
    if 1 in topks:
        for T in sc_ts:
            out.append((T, 1, _topk_label(1, T)))
    if include_eagle:
        for T in eagle_ts:
            for k in eagle_ks:
                out.append((T, k, _topk_label(k, T)))
    return out


def expand_mtp_subcases(case_dict, topks=None, eagle_T=None, draft_token_nums=None,
                        include_eagle=True):
    """Expand one caselist case into a list of MTP sub-items [(T, topk, label), ...].

    See mtp_subcase_specs. eagle_T is only for historical compatibility (equivalent to a single-element
    draft_token_nums=[eagle_T]). With include_eagle=False only single_chain is generated.
    """
    if draft_token_nums is None and eagle_T is not None:
        draft_token_nums = [eagle_T]
    return mtp_subcase_specs(
        seqlen_q=case_dict.get('seqlen_q', 1),
        has_explicit_topk=('topk' in case_dict),
        explicit_topk=case_dict.get('topk'),
        topks=topks,
        draft_token_nums=draft_token_nums,
        include_eagle=include_eagle,
    )
