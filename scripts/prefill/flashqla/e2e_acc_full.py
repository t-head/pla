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

"""Comprehensive E2E accuracy: full pipeline (chunk_gated_delta_rule_fwd) with
all supported dispatch shapes, exercising both non-CP and CP paths.

Tests:
  1. All 16 (H,Hg) dispatch combos at T=2048 (automatic planner)
  2. GVA deployment shapes at T={2048,4096,8192,16384,32768} (exercises CP)
  3. CP-path validation with fast-decay gates (ensures fallback_mask=0)
  4. h0 (initial state) path
  5. Optional generalized-prefill replay over the repository's complete
     20-list inventory (16 GDN lists / 20,890 GDN records).  The replay checks
     production auto-CP against the serial path and writes auditable JSONL.

The generalized replay is opt-in because it is a long-running exhaustive gate::

    CUDA_VISIBLE_DEVICES=0 python scripts/prefill/flashqla/e2e_acc_full.py \
      --generalized-gdn only --generalized-gdn-num-shards 8 \
      --generalized-gdn-shard-index 0 \
      --generalized-gdn-output /tmp/gdn-accuracy-shard0.jsonl

Gate: primary O comparison uses the pla prefill kernels; the numpy reference is
retained for failure/debug localization and final-state diagnostics when the
kernel path does not expose ht.
"""
# The pure-torch oracle is an internal test dependency, not an installed API.
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from pla.prefill.flashqla import chunk_gated_delta_rule_fwd
from pla.prefill.flashqla import private as flashqla_cute
from pla.prefill.flashqla.private.cp_preprocess import calc_cp_seqs, MULTI_PROCESSOR_COUNT
# Accuracy oracle: the vendored, self-contained upstream FlashQLA torch reference
# (private/ref_gdr.py, depending only on the vendored pure-torch helpers).
# It is GVA-native and needs no external flash-linear-attention install, so the
# test is fully self-contained on this repo.
from private import ref_gdr

CS = 32
DK = 128
DV = 128
LOG2E = 1.4426950408889634
# Oracle is the fp32 self-contained FlashQLA torch reference (ref_gdr); with
# fp16 kernel inputs the observed worst case across all shapes is o_rel~1.1e-3,
# ht_rel~6e-4, cos=1.00000. Tolerances are tightened to align with that, leaving
# a small margin for seed/shape jitter (NOT the old loose FLA-Triton bounds).
REF_COS_MIN = 0.999
REF_REL_MAX = 5e-3
BF16_REF_REL_MAX = 1e-2
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREFILL_LIST_DIR = REPO_ROOT / "caselist" / "prefill"
EXPECTED_PREFILL_LIST_FILES = 20
EXPECTED_GDN_LIST_FILES = 16
EXPECTED_GDN_RECORDS = 20_890


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FlashQLA canonical and generalized-list E2E accuracy",
    )
    parser.add_argument(
        "--generalized-gdn", choices=("off", "append", "only"), default="off",
        help=(
            "off: canonical suite only; append: canonical suite followed by "
            "the complete GDN list replay; only: run only the GDN list replay"
        ),
    )
    parser.add_argument(
        "--generalized-gdn-list-dir", type=Path,
        default=DEFAULT_PREFILL_LIST_DIR,
        help="directory containing the 20 generalized prefill case lists",
    )
    parser.add_argument(
        "--generalized-gdn-output", type=Path,
        help=(
            "JSONL destination; defaults to "
            "/tmp/flashqla-gdn-accuracy-shard<index>.jsonl"
        ),
    )
    parser.add_argument("--generalized-gdn-num-shards", type=int, default=1)
    parser.add_argument("--generalized-gdn-shard-index", type=int, default=0)
    parser.add_argument(
        "--expected-prefill-list-files", type=int,
        default=EXPECTED_PREFILL_LIST_FILES,
    )
    parser.add_argument(
        "--expected-gdn-list-files", type=int,
        default=EXPECTED_GDN_LIST_FILES,
    )
    parser.add_argument(
        "--expected-gdn-records", type=int, default=EXPECTED_GDN_RECORDS,
    )
    args = parser.parse_args(argv)
    if args.generalized_gdn_num_shards <= 0:
        parser.error("--generalized-gdn-num-shards must be positive")
    if not 0 <= args.generalized_gdn_shard_index < args.generalized_gdn_num_shards:
        parser.error(
            "--generalized-gdn-shard-index must be in "
            "[0, --generalized-gdn-num-shards)"
        )
    return args


def _load_generalized_prefill_replay():
    path = REPO_ROOT / "scripts" / "prefill" / "generalized_prefill_replay.py"
    spec = importlib.util.spec_from_file_location(
        "pla_flashqla_generalized_prefill_replay", path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generalized prefill replay from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_generalized_gdn_case_lists(args):
    """Replay every GDN record from the complete 20-list prefill inventory."""
    replay = _load_generalized_prefill_replay()
    list_dir = args.generalized_gdn_list_dir.resolve()
    all_lists = replay.discover_case_lists(list_dir, "all")
    gdn_lists = replay.discover_case_lists(list_dir, "gdn")
    gdn_records = replay.load_records(list_dir, "gdn")

    inventory_counts = {
        "all lists": (len(all_lists), args.expected_prefill_list_files),
        "GDN lists": (len(gdn_lists), args.expected_gdn_list_files),
        "GDN records": (len(gdn_records), args.expected_gdn_records),
    }
    mismatches = [
        f"{name}: expected {expected}, got {actual}"
        for name, (actual, expected) in inventory_counts.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError("generalized prefill inventory mismatch: " + "; ".join(mismatches))

    output = args.generalized_gdn_output
    if output is None:
        output = Path(
            "/tmp/flashqla-gdn-accuracy-"
            f"shard{args.generalized_gdn_shard_index}.jsonl"
        )
    output = output.resolve()
    runner_args = [
        "--list-dir", str(list_dir),
        "--family", "gdn",
        "--mode", "accuracy",
        "--reference", "serial",
        "--gdn-path", "legacy",
        "--output", str(output),
        "--num-shards", str(args.generalized_gdn_num_shards),
        "--shard-index", str(args.generalized_gdn_shard_index),
        "--expected-list-files", str(args.expected_gdn_list_files),
        "--expected-records", str(args.expected_gdn_records),
    ]
    print(
        "  inventory: "
        f"all={len(all_lists)} GDN_lists={len(gdn_lists)} "
        f"GDN_records={len(gdn_records)}",
        flush=True,
    )
    status = replay.main(runner_args)

    expected_indices = {
        record["index"] for record in gdn_records
        if record["index"] % args.generalized_gdn_num_shards
        == args.generalized_gdn_shard_index
    }
    actual_indices = set()
    row_count = pass_count = 0
    for line_number, raw in enumerate(output.read_text().splitlines(), 1):
        row = json.loads(raw)
        index = row.get("index")
        if not isinstance(index, int):
            raise RuntimeError(f"{output}:{line_number}: missing integer index")
        if index in actual_indices:
            raise RuntimeError(f"{output}:{line_number}: duplicate index {index}")
        actual_indices.add(index)
        row_count += 1
        pass_count += int(row.get("pass") is True)

    coverage_ok = actual_indices == expected_indices
    ok = status == 0 and coverage_ok and pass_count == row_count
    print(
        f"  generalized GDN shard {args.generalized_gdn_shard_index}/"
        f"{args.generalized_gdn_num_shards}: pass={pass_count}/{row_count} "
        f"coverage={'PASS' if coverage_ok else 'FAIL'} output={output} "
        f"{'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    if not coverage_ok:
        print(
            f"    missing={sorted(expected_indices - actual_indices)[:20]} "
            f"extra={sorted(actual_indices - expected_indices)[:20]}",
            flush=True,
        )
    return ok, pass_count, row_count, len(gdn_lists), output


def dtype_name(dtype):
    name = str(dtype)
    return name[len("torch."):] if name.startswith("torch.") else name


def f16(x):
    return x.astype(np.float16).astype(np.float32)


def cos_sim(a, b):
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0)).item()


def tensor_metrics(a, b):
    finite = torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    if not finite:
        return 0.0, 1.0, False
    cos = cos_sim(a, b)
    rel = ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-8)).item()
    return cos, float(rel), True


def official_ref(q, k, v, g, beta, scale, h0=None):
    """Self-contained upstream FlashQLA torch reference (fp32, chunk_size=CS).

    Wraps the vendored ref_gdr.chunk_gated_delta_rule_fwd. It is GVA-native
    (q/k carry Hg heads, v carries H heads; the internal repeat_interleave
    expands them) and runs its own chunk-local gate cumsum, so it consumes the
    SAME raw inputs the CuTe op consumes. q/k are already L2-normalized by the
    caller (matching the CuTe pipeline), so no extra normalization here.

    Returns (o, ht) as float torch tensors:
      o  = [1, T, H, DV]     ht = [1, H, DK, DV]
    """
    h0_ref = h0.float() if h0 is not None else None
    # ref_gdr.chunk_gated_delta_rule_fwd -> (g_cum, o, A, h, final_state)
    _, o_ref, _, _, ht_ref = ref_gdr.chunk_gated_delta_rule_fwd(
        q=q.float(), k=k.float(), v=v.float(),
        g=g.float(), beta=beta.float(),
        cu_seqlens=None, initial_state=h0_ref,
        scale=scale, chunk_size=CS,
    )
    return o_ref.float(), ht_ref.float()


def ref_kkt(k, beta, T, H, Hg):
    A = np.zeros((T, H, CS), np.float32)
    nchunks = (T + CS - 1) // CS
    for c in range(nchunks):
        cs0 = c * CS
        act = min(CS, T - cs0)
        for h in range(H):
            hg = h // (H // Hg)
            Kt = k[cs0:cs0 + act, hg]
            bt = beta[cs0:cs0 + act, h]
            KKT = (Kt @ Kt.T) * bt[None, :]
            M = np.eye(act, dtype=np.float32) - np.tril(KKT, -1)
            Ac = np.linalg.inv(M).astype(np.float32)
            A[cs0:cs0 + act, h, :act] = Ac
    return f16(A)


def ref_fused_fwd(q, k, v, a, beta, g, T, H, Hg, scale, h0=None):
    o = np.zeros((T, H, DV), np.float32)
    ht = np.zeros((H, DK, DV), np.float32)
    nchunks = (T + CS - 1) // CS
    tri = np.tril(np.ones((CS, CS), np.float32))
    for h in range(H):
        hg = h // (H // Hg)
        state = h0[h].astype(np.float32).copy() if h0 is not None \
            else np.zeros((DK, DV), np.float32)
        for c in range(nchunks):
            cs0 = c * CS
            act = min(CS, T - cs0)
            K = np.zeros((CS, DK), np.float32); V = np.zeros((CS, DV), np.float32)
            Q = np.zeros((CS, DK), np.float32); A = np.zeros((CS, CS), np.float32)
            gc = np.zeros(CS, np.float32); bc = np.zeros(CS, np.float32)
            K[:act] = k[cs0:cs0 + act, hg]; V[:act] = v[cs0:cs0 + act, h]
            Q[:act] = q[cs0:cs0 + act, hg]; A[:act, :] = a[cs0:cs0 + act, h]
            gc[:act] = g[cs0:cs0 + act, h]
            if act < CS:
                gc[act:] = gc[act - 1] if act > 0 else 0.0
            bc[:act] = beta[cs0:cs0 + act, h]
            g_last = gc[act - 1] if act > 0 else 0.0
            g_exp = np.exp2(gc * LOG2E)
            g_rev = np.exp2((g_last - gc) * LOG2E)
            decay = np.exp2(g_last * LOG2E)
            gate_delta = gc[:, None] - gc[None, :]
            gij = np.exp2(np.minimum(gate_delta, 0.0) * LOG2E)
            h_stm = f16(state)
            U = f16(K) @ h_stm
            W = f16(V - g_exp[:, None] * U)
            Ag = f16((A * gij * bc[None, :]) * tri)
            Vd = f16(Ag @ W)
            O_inter = scale * g_exp[:, None] * (f16(Q) @ h_stm)
            P = f16(Q) @ f16(K).T
            Pg = f16((scale * gij * P) * tri)
            O_intra = f16(Pg) @ f16(Vd)
            o[cs0:cs0 + act, h] = (O_inter + O_intra)[:act]
            Vd_g = f16(g_rev[:, None] * Vd)
            state = decay * state + f16(K).T @ f16(Vd_g)
        ht[h] = state
    return o, ht


def run_e2e(
    seq_len,
    H,
    Hg,
    use_h0=False,
    fast_decay=False,
    slow_decay=False,
    dtype=torch.float16,
    compare_no_cp=False,
    expected_fallback=None,
    seed=42,
):
    """Run full pipeline E2E and compare primarily against the vendored FlashQLA
    torch reference (ref_gdr).

    The hand-written numpy reference is retained as a secondary debug/localization
    cross-check.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    scale = DK ** -0.5

    if fast_decay and slow_decay:
        raise ValueError("fast_decay and slow_decay are mutually exclusive")
    if expected_fallback not in (None, "zero", "nonzero", "serial"):
        raise ValueError(f"invalid expected_fallback: {expected_fallback}")

    q = (torch.randn(1, seq_len, Hg, DK, device=dev) * 0.1).to(dtype)
    k = (torch.randn(1, seq_len, Hg, DK, device=dev) * 0.1).to(dtype)
    # Match FLA prefill semantics: q/k are L2-normalized before the delta-rule
    # recurrence. CuTe consumes the same normalized tensors; FLA is called with
    # use_qk_l2norm_in_kernel=False to avoid normalizing twice.
    q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    v = (torch.randn(1, seq_len, H, DV, device=dev) * 0.1).to(dtype)
    if fast_decay:
        # Force a chunk-local cumulative gate below the fp32 exp overflow
        # boundary. The kernel must evaluate exp(g_i - g_j) directly.
        g = -(torch.abs(torch.randn(1, seq_len, H, device=dev)) * 0.5 + 4.0).float()
    elif slow_decay:
        # Keep every CP segment above the -10 warmup threshold. This forces
        # fallback_mask>0 and validates the mt @ h correction path.
        g = torch.full((1, seq_len, H), -1e-3, device=dev, dtype=torch.float32)
    else:
        g = F.logsigmoid(torch.randn(1, seq_len, H, device=dev).float())
    beta = torch.rand(1, seq_len, H, device=dev).float().sigmoid()
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=dev)
    h0 = (torch.randn(1, H, DK, DV, device=dev) * 0.02).float() if use_h0 else None

    # Check CP decision
    use_cp, cp_cu, _, _, ht_mask = calc_cp_seqs(
        cu, CS, H, num_k_heads=Hg
    )
    cp_batch = (cp_cu.numel() - 1) if use_cp else 1

    # Run CuTe PPU pipeline
    g_cum, A, o_cute, _, ht_cute = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale=scale, initial_state=h0, cu_seqlens=cu, auto_cp=True)

    serial_metrics = None
    if compare_no_cp:
        _, _, o_serial, _, ht_serial = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale=scale, initial_state=h0,
            cu_seqlens=cu, auto_cp=False)
        serial_o_cos, serial_o_rel, serial_o_finite = tensor_metrics(o_cute, o_serial)
        serial_ht_cos, serial_ht_rel, serial_ht_finite = tensor_metrics(ht_cute, ht_serial)
        serial_metrics = (
            serial_o_cos, serial_o_rel, serial_o_finite,
            serial_ht_cos, serial_ht_rel, serial_ht_finite,
        )

    # Vendored FlashQLA torch reference is the primary accuracy oracle for O and
    # ht. It is GVA-native, so q/k keep their Hg heads (no repeat_interleave).
    o_ref, ht_ref = official_ref(q, k, v, g, beta, scale, h0=h0)

    # Check fallback_mask if CP active
    fb_sum = -1
    if use_cp:
        nw, fb = flashqla_cute.get_warmup_chunks(
            g_cum, ht_mask.to(torch.int8), cp_cu.to(torch.int32))
        fb_sum = int(fb.sum().item())

    o_c_t = o_cute.reshape(1, seq_len, H, DV).float()
    ht_c_t = ht_cute.reshape(1, H, DK, DV).float()
    o_r_t = o_ref.reshape(1, seq_len, H, DV).float()
    ht_r_t = ht_ref.reshape(1, H, DK, DV).float()

    o_cos, o_rel, o_finite = tensor_metrics(o_c_t, o_r_t)
    ht_cos, ht_rel, ht_finite = tensor_metrics(ht_c_t, ht_r_t)
    ht_primary = True

    # Numpy debug reference: not the primary gate, but printed when useful for
    # diagnosing whether a failure is FLA/CuTe mismatch or algorithm-local.
    k_np = k[0].float().cpu().numpy(); beta_np = beta[0].cpu().numpy()
    A_ref = ref_kkt(k_np, beta_np, seq_len, H, Hg)
    h0_np = h0[0].cpu().numpy() if use_h0 else None
    o_np, ht_np = ref_fused_fwd(
        q[0].float().cpu().numpy(), k_np, v[0].float().cpu().numpy(),
        A_ref, beta_np, g_cum[0].cpu().numpy(), seq_len, H, Hg, scale, h0_np)
    o_np_t = torch.from_numpy(o_np).reshape(1, seq_len, H, DV)
    ht_np_t = torch.from_numpy(ht_np).reshape(1, H, DK, DV)
    dbg_o_cos, dbg_o_rel, _ = tensor_metrics(o_c_t.cpu(), o_np_t)
    dbg_ht_cos, dbg_ht_rel, _ = tensor_metrics(ht_c_t.cpu(), ht_np_t)

    # BF16 has a 7.8125e-3 unit roundoff near one. Keep the stricter FP16 gate,
    # but allow one BF16 quantization step plus margin for the CP reordering.
    rel_max = BF16_REF_REL_MAX if dtype == torch.bfloat16 else REF_REL_MAX
    ok = (o_finite and o_cos > REF_COS_MIN and o_rel < rel_max)
    ok = ok and ht_finite and ht_cos > REF_COS_MIN and ht_rel < rel_max
    if serial_metrics is not None:
        so_cos, so_rel, so_finite, sh_cos, sh_rel, sh_finite = serial_metrics
        ok = ok and so_finite and so_cos > REF_COS_MIN and so_rel < rel_max
        ok = ok and sh_finite and sh_cos > REF_COS_MIN and sh_rel < rel_max
    if expected_fallback == "zero":
        ok = ok and use_cp and fb_sum == 0
    elif expected_fallback == "nonzero":
        ok = ok and use_cp and fb_sum > 0
    elif expected_fallback == "serial":
        ok = ok and not use_cp

    label = "CP" if use_cp else "no-CP"
    h0_s = " h0" if use_h0 else ""
    fd_s = " fast-decay" if fast_decay else ""
    sd_s = " slow-decay" if slow_decay else ""
    dtype_s = dtype_name(dtype)
    serial_s = ""
    if serial_metrics is not None:
        so_cos, so_rel, _, sh_cos, sh_rel, _ = serial_metrics
        serial_s = (
            f"serial_o_cos={so_cos:.5f} serial_o_rel={so_rel:.2e} "
            f"serial_ht_cos={sh_cos:.5f} serial_ht_rel={sh_rel:.2e} "
        )
    print(f"  [{label}] seq={seq_len:>5} H={H:>3} Hg={Hg:>2} "
          f"dtype={dtype_s}{h0_s}{fd_s}{sd_s} | "
          f"cp_batch={cp_batch:>2} o_ref_cos={o_cos:.5f} o_ref_rel={o_rel:.2e} "
          f"ht_ref_cos={ht_cos:.5f} ht_ref_rel={ht_rel:.2e} "
          f"{serial_s}np_o_cos={dbg_o_cos:.5f} np_o_rel={dbg_o_rel:.2e} "
          f"fb={fb_sum} "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    debug numpy: o_cos={dbg_o_cos:.6f} o_rel={dbg_o_rel:.3e} "
              f"ht_cos={dbg_ht_cos:.6f} ht_rel={dbg_ht_rel:.3e}")
        print(
            "    finite/max: "
            f"o={torch.isfinite(o_c_t).float().mean().item():.3f}/"
            f"{torch.nan_to_num(o_c_t).abs().max().item():.3e} "
            f"ht={torch.isfinite(ht_c_t).float().mean().item():.3f}/"
            f"{torch.nan_to_num(ht_c_t).abs().max().item():.3e}"
        )
    return ok


def run_indexed_state_case(lengths, H, Hg, dtype, auto_cp, seed):
    """Compare the indexed K-last state pool with the legacy batch state ABI."""
    torch.manual_seed(seed)
    total = sum(lengths)
    q = F.normalize(
        torch.randn(1, total, Hg, DK, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(
        torch.randn(1, total, Hg, DK, device="cuda"), dim=-1).to(dtype)
    v = (torch.randn(1, total, H, DV, device="cuda") * 0.1).to(dtype)
    g = F.logsigmoid(torch.randn(1, total, H, device="cuda").float())
    beta = torch.rand(1, total, H, device="cuda").float().sigmoid()
    cu = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        dtype=torch.int32, device="cuda")

    batch = len(lengths)
    num_slots = batch + 3
    slots = torch.tensor(
        list(range(num_slots - 1, num_slots - batch - 1, -1)),
        dtype=torch.int32, device="cuda")
    pool_before = (
        torch.randn(num_slots, H, DV, DK, device="cuda") * 0.02).float()
    legacy_h0 = pool_before[slots].transpose(-1, -2).contiguous()

    _, legacy_a, legacy_o, _, legacy_ht = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, initial_state=legacy_h0, cu_seqlens=cu,
        auto_cp=auto_cp)
    indexed_pool = pool_before.clone()
    _, indexed_a, indexed_o, _, indexed_ht = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, initial_state=indexed_pool, cu_seqlens=cu,
        auto_cp=auto_cp, initial_state_indices=slots)

    expected_pool = pool_before.clone()
    expected_pool[slots] = legacy_ht.transpose(-1, -2)
    rel_max = BF16_REF_REL_MAX if dtype == torch.bfloat16 else REF_REL_MAX
    metrics = {
        "a": tensor_metrics(indexed_a, legacy_a),
        "o": tensor_metrics(indexed_o, legacy_o),
        "returned_state": tensor_metrics(
            indexed_ht, legacy_ht.transpose(-1, -2)),
        "pool": tensor_metrics(indexed_pool, expected_pool),
    }
    ok = all(
        finite and cos > REF_COS_MIN and rel < rel_max
        for cos, rel, finite in metrics.values())

    # Serial dispatch must enforce the same legacy-state contract as the CP
    # preprocessing binding; otherwise an invalid state can become an OOB read
    # solely because the planner selected a different path.  bf16 is a supported
    # state dtype, so only layout and shape violations are rejected here.
    for bad_legacy in (
        legacy_h0.transpose(-1, -2),
        legacy_h0[:max(0, batch - 1)],
    ):
        try:
            chunk_gated_delta_rule_fwd(
                q, k, v, g, beta, initial_state=bad_legacy,
                cu_seqlens=cu, auto_cp=False)
        except ValueError:
            pass
        else:
            ok = False

    summary = " ".join(
        f"{name}={cos:.5f}/{rel:.2e}"
        for name, (cos, rel, _) in metrics.items())
    print(
        f"  lengths={lengths} H={H} Hg={Hg} "
        f"dtype={dtype_name(dtype)} auto_cp={auto_cp} "
        f"{summary} {'PASS' if ok else 'FAIL'}")
    return ok


def run_indexed_no_return_state_case(lengths, H, Hg, dtype, auto_cp, seed):
    """Check indexed pool writeback when final state is not returned."""
    torch.manual_seed(seed)
    total = sum(lengths)
    q = F.normalize(
        torch.randn(1, total, Hg, DK, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(
        torch.randn(1, total, Hg, DK, device="cuda"), dim=-1).to(dtype)
    v = (torch.randn(1, total, H, DV, device="cuda") * 0.1).to(dtype)
    g = F.logsigmoid(torch.randn(1, total, H, device="cuda").float())
    beta = torch.rand(1, total, H, device="cuda").float().sigmoid()
    cu = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        dtype=torch.int32, device="cuda")

    batch = len(lengths)
    num_slots = batch + 3
    slot_values = list(range(num_slots - 1, num_slots - batch - 1, -1))
    slots = torch.tensor(slot_values, dtype=torch.int32, device="cuda")
    unmapped = torch.tensor(
        [slot for slot in range(num_slots) if slot not in slot_values],
        dtype=torch.int64, device="cuda")
    pool_before = (
        torch.randn(num_slots, H, DV, DK, device="cuda") * 0.02).float()

    return_pool = pool_before.clone()
    _, return_a, return_o, _, return_ht = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, initial_state=return_pool, cu_seqlens=cu,
        auto_cp=auto_cp, initial_state_indices=slots,
        output_final_state=True)

    no_return_pool = pool_before.clone()
    pool_ptr = no_return_pool.data_ptr()
    _, no_return_a, no_return_o, _, no_return_ht = (
        chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, initial_state=no_return_pool,
            cu_seqlens=cu, auto_cp=auto_cp,
            initial_state_indices=slots, output_final_state=False))

    checks = {
        "ht_none": no_return_ht is None,
        "same_storage": no_return_pool.data_ptr() == pool_ptr,
        "pool_matches_true": torch.equal(no_return_pool, return_pool),
        "mapped_updated": not torch.equal(
            no_return_pool[slots], pool_before[slots]),
        "unmapped_unchanged": torch.equal(
            no_return_pool[unmapped], pool_before[unmapped]),
        "true_return_matches_pool": (
            return_ht is not None and torch.equal(return_ht, return_pool[slots])),
        "a_unchanged": torch.equal(no_return_a, return_a),
        "o_unchanged": torch.equal(no_return_o, return_o),
    }
    ok = all(checks.values())
    summary = " ".join(
        f"{name}={'PASS' if passed else 'FAIL'}"
        for name, passed in checks.items())
    print(
        f"  lengths={lengths} H={H} Hg={Hg} "
        f"dtype={dtype_name(dtype)} auto_cp={auto_cp} "
        f"output_final_state=False {summary} "
        f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    args = parse_cli_args()
    if args.generalized_gdn == "only":
        generalized_ok, _, _, _, _ = run_generalized_gdn_case_lists(args)
        sys.exit(0 if generalized_ok else 1)

    all_ok = True
    results = []
    generalized_result = None

    # === Section 1: All 16 dispatch shapes, T=2048, automatic planner ===
    print("=" * 80)
    print("SECTION 1: All 16 dispatch combos (T=2048, pipeline E2E)")
    print("=" * 80)
    DISPATCH_COMBOS = [
        (128, 16), (64, 16), (64, 8), (48, 16), (32, 16),
        (32, 8), (32, 4), (24, 8), (16, 8), (16, 4),
        (16, 2), (12, 4), (8, 4), (8, 2), (6, 2), (4, 2),
    ]
    for (H, Hg) in DISPATCH_COMBOS:
        ok = run_e2e(2048, H, Hg)
        results.append(("dispatch", 2048, H, Hg, ok))
        all_ok &= ok

    # === Section 2: GVA deployment shapes, longer sequences (exercises CP) ===
    print("\n" + "=" * 80)
    print("SECTION 2: GVA deployment shapes, variable T (exercises CP decisions)")
    print("=" * 80)
    GVA_CASES = [
        # (T, H, Hg) — covers major Qwen deployment configs
        (4096, 16, 8),   # Qwen3.7-Max tp8
        (8192, 16, 8),
        (16384, 16, 8),
        (32768, 16, 8),
        (4096, 32, 16),  # Qwen3.7-Max tp4
        (8192, 32, 16),
        (16384, 32, 16),
        (4096, 64, 8),   # Qwen3.7-Max tp2
        (8192, 64, 8),
    ]
    for (T, H, Hg) in GVA_CASES:
        ok = run_e2e(T, H, Hg)
        results.append(("gva", T, H, Hg, ok))
        all_ok &= ok

    # === Section 3: CP-path with fast-decay gates (validates CP correctness) ===
    print("\n" + "=" * 80)
    print("SECTION 3: CP/serial dispatch validation (fast-decay gates)")
    print("=" * 80)
    CP_CASES = [
        # H16 activates CP on both tc01 and tc02 at these lengths.
        (8192, 16, 8, torch.float16, "zero"),
        (16384, 16, 8, torch.float16, "zero"),
        (32768, 16, 8, torch.float16, "zero"),
    ]
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (int(props.major), int(props.minor)) == (8, 9):
        # tc01 deliberately keeps H32 serial. The latest tc02 calibration also
        # keeps 8K serial (the CP threshold is 9440 tokens), then requires CP
        # from the measured 10K production point onward.
        CP_CASES += [
            (8192, 32, 16, torch.float16, "serial"),
            (10240, 32, 16, torch.float16, "zero"),
            (10240, 32, 16, torch.bfloat16, "zero"),
            (16384, 32, 16, torch.float16, "zero"),
        ]
    for (T, H, Hg, dtype, expected_path) in CP_CASES:
        ok = run_e2e(
            T,
            H,
            Hg,
            fast_decay=True,
            expected_fallback=expected_path,
            dtype=dtype,
        )
        results.append(("cp", T, H, Hg, ok))
        all_ok &= ok

    # === Section 4: h0 (initial state) path ===
    print("\n" + "=" * 80)
    print("SECTION 4: Initial state (h0) path")
    print("=" * 80)
    H0_CASES = [
        (2048, 32, 16),
        (4096, 16, 8),
        (8192, 16, 8),   # with CP + h0
    ]
    for (T, H, Hg) in H0_CASES:
        ok = run_e2e(T, H, Hg, use_h0=True)
        results.append(("h0", T, H, Hg, ok))
        all_ok &= ok

    # === Section 5: partial tail chunk (actual < CS) ===
    # Sections 1-4 use only multiples of CS=32, so the actual < CS branch in
    # fused_fwd_cp_cute_kernel was never executed by this suite at all -- a full
    # PASS said nothing about the tail path. These cases exercise it: the tail
    # chunk skips the prefetch, is issued inline, and relies on the AIU zero-
    # filling smem rows [actual, CS). Paired with fast_decay because a stale
    # (possibly NaN) tail row only becomes visible once it survives a GEMM.
    print("\n" + "=" * 80)
    print("SECTION 5: Partial tail chunk (T %% CS != 0, CS=%d)" % CS)
    print("=" * 80)
    TAIL_CASES = [
        # (T, H, Hg, fast_decay)
        (5555, 32, 16, False),  # 173*32 + 19 -> 19-token tail
        (5555, 16, 8,  True),   # same, with strong decay
        (2047, 32, 16, False),  # 31-token tail (largest partial)
        (2049, 32, 16, False),  # 1-token tail (smallest partial)
        (2049, 16, 8,  True),
        (33,   16, 8,  False),  # one full chunk + 1-token tail
        (17,   16, 8,  False),  # shorter than one chunk: smem never written before
    ]
    for (T, H, Hg, fd) in TAIL_CASES:
        ok = run_e2e(T, H, Hg, fast_decay=fd)
        results.append(("tail", T, H, Hg, ok))
        all_ok &= ok

    # === Section 6: indexed K-last state pool ===
    print("\n" + "=" * 80)
    print("SECTION 6: Indexed K-last state pool (CP/no-CP + state writeback)")
    print("=" * 80)
    INDEXED_CASES = [
        ((65, 47), 16, 8, torch.float16, False, 42),
        ((8192, 2048), 16, 8, torch.bfloat16, True, 123),
    ]
    for lengths, H, Hg, dtype, auto_cp, seed in INDEXED_CASES:
        ok = run_indexed_state_case(
            lengths, H, Hg, dtype, auto_cp, seed)
        results.append(("indexed", sum(lengths), H, Hg, ok))
        all_ok &= ok
    NO_RETURN_CASE = ((65, 47), 16, 8, torch.bfloat16, True, 2026)
    ok = run_indexed_no_return_state_case(*NO_RETURN_CASE)
    results.append((
        "indexed", sum(NO_RETURN_CASE[0]), NO_RETURN_CASE[1],
        NO_RETURN_CASE[2], ok))
    all_ok &= ok

    # === Section 7: all GDN records from the complete 20-list inventory ===
    if args.generalized_gdn == "append":
        print("\n" + "=" * 80)
        print("SECTION 7: Generalized GDN prefill inventory (20 lists total)")
        print("=" * 80)
        generalized_result = run_generalized_gdn_case_lists(args)
        all_ok &= generalized_result[0]

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    sections = {"dispatch": "Dispatch combos", "gva": "GVA deployment",
                "cp": "CP-path", "h0": "h0 path", "tail": "Partial tail",
                "indexed": "Indexed state"}
    for sec_key, sec_name in sections.items():
        sec_results = [(T, H, Hg, ok) for (s, T, H, Hg, ok) in results if s == sec_key]
        passed = sum(1 for r in sec_results if r[3])
        total = len(sec_results)
        status = "ALL PASS" if passed == total else f"{total - passed} FAILED"
        print(f"  {sec_name:20s}: {passed}/{total} {status}")
        if passed < total:
            for (T, H, Hg, ok) in sec_results:
                if not ok:
                    print(f"    FAIL: T={T} H={H} Hg={Hg}")

    if generalized_result is not None:
        generalized_ok, passed, record_count, list_count, output = generalized_result
        print(
            f"  {'GDN case-list replay':20s}: "
            f"{passed}/{record_count} "
            f"{'ALL PASS' if generalized_ok else 'FAILED'} "
            f"({list_count} GDN lists, output={output})"
        )

    total_pass = sum(1 for r in results if r[4]) + int(
        generalized_result is not None and generalized_result[0]
    )
    total_tests = len(results) + int(generalized_result is not None)
    print(f"\n  TOTAL: {total_pass}/{total_tests} "
          f"{'ALL PASS' if all_ok else 'SOME FAILED'}")
    print("=" * 80)
    sys.exit(0 if all_ok else 1)
