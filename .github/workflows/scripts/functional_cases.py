# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass


BASELINE = "38646e904784fbd3beeadafa2b9485dabc5cc09e"
GDN_MODELS = (
    "Qwen3.5-27B", "Qwen3.5-35B-A3B", "Qwen3.5-397B-A17B",
    "Qwen3-Next-80B-A3B", "Qwen3.7-Max",
)


def caselist(model, phase, family="gdn", tp=1):
    return f"caselist/{phase}/{model}_{phase}_tp{tp}_kvlimit_200GB_model_len_128K_{family}_op.list"


@dataclass(frozen=True)
class Case:
    name: str
    phase: str
    argv: tuple
    timeout: int = 3600
    report: str = "exit"
    legacy: bool = False
    lists: tuple = ()


def cases():
    result = []
    for model in GDN_MODELS:
        kernels = ("k_last", "k_last_packed", "k_last_mtp")
        if model != "Qwen3.7-Max":
            kernels = ("cuda",) + kernels
        for kernel in kernels:
            timeout = 5400 if kernel == "k_last_mtp" and model in (
                "Qwen3.5-27B", "Qwen3.7-Max",
            ) else 3600
            result.append(Case(
                f"{model}-{kernel}", "decode",
                ("tests/test_op_forge.py", f"--caselist={caselist(model, 'decode')}",
                 f"--kernel={kernel}"), timeout, "markers",
            ))
    kimi = caselist("Kimi-K3", "decode", "kda")
    for kernel in ("k_last", "k_last_packed", "k_last_mtp"):
        result.append(Case(
            f"Kimi-K3-{kernel}", "decode",
            ("tests/test_op_forge.py", f"--caselist={kimi}", f"--kernel={kernel}"),
            5400 if kernel == "k_last_mtp" else 3600, "markers",
        ))
    result.extend([
        Case("gdn-noncontiguous-stride", "decode",
             ("scripts/decode/test_gdn_noncontiguous_stride.py",), 900),
        Case("kda-mega-fusion", "decode", ("tests/test_kda_decode_mega.py",
             "--dtype=bf16", f"--caselist={kimi}", "--max-batches=4"), 3600, "markers"),
        Case("kda-mtp-sglang", "decode", ("tests/test_kda_mtp_sglang.py",
             f"--caselist={kimi}", "--phase=all", "--max-bs-cases=3",
             "--bs-cap=64", "--layers=2,4"), 2400, "markers"),
        Case("kda-spec-recurrent", "decode", ("scripts/decode/test_kda_decode.py",), 1200),
        Case("kda-pad-slot", "decode", ("scripts/decode/kda_fusion/test_pad_slot.py",), 900),
        Case("kda-commit-layouts", "decode",
             ("scripts/decode/kda_mtp_sglang/test_commit_layouts.py",), 900),
        Case("kda-verify-layouts", "decode",
             ("scripts/decode/kda_mtp_sglang/test_verify_layouts.py",), 900),
        Case("sanity-check", "decode", ("test_sanity_check.py",), 1800, "pytest"),
        Case("k-last-state-dtype", "decode", ("test_k_last_state_dtype.py",), 1800, "pytest"),
        Case("generated-k-last", "decode", ("tests/test_generated_k_last.py",), 600),
        Case("instrument-graph-capture", "decode",
             ("tests/test_instrument_graph_capture.py",), 1200),
    ])
    for model in GDN_MODELS:
        result.append(Case(
            f"{model}-flashqla", "prefill",
            ("tests/test_op_forge_prefill.py", f"--caselist={caselist(model, 'prefill')}"),
            3600, "markers",
        ))
    result.append(Case(
        "flashkda-dispatch", "prefill",
        ("tests/test_kda_prefill_flashkdapro.py", "--phase=dispatch"), 3600, "markers", True,
    ))
    for name, model, tps in (("glm-kda-full", "GLM-5.3-Flash", (1,)),
                             ("kimi-kda-full", "Kimi-K3", (1, 2))):
        result.append(Case(
            name, "prefill", ("scripts/prefill/generalized_prefill_replay.py",
                              "--family=kda", "--mode=accuracy", "--reference=serial"),
            3600, "jsonl", True,
            tuple(caselist(model, "prefill", "kda", tp) for tp in tps),
        ))
    for name, script, timeout in (
        ("flashkda-legacy-cp", "test_cp_correctness.py", 1200),
        ("flashkda-indexed-state", "test_indexed_state_pool.py", 900),
        ("flashkda-int32-cu", "test_int32_cu_seqlens.py", 900),
        ("flashkda-cp-plan-cache", "test_cp_plan_cache.py", 900),
        ("flashkda-kernel1", "test_kernel1.py", 900),
        ("flashkda-tc01-auto-cp", "test_cp_tc01_auto.py", 1800),
    ):
        result.append(Case(name, "prefill",
                           (f"scripts/prefill/flashkdapro/private/{script}",),
                           timeout, "exit", True))
    result.extend([
        Case("flashkda-bt64-contract", "prefill",
             ("scripts/prefill/flashkdapro/private/test_bt64_contract.py",), 600),
        Case("flashkda-fla-dispatch-contract", "prefill",
             ("tests/test_tc01_fla_dispatch.py",), 600),
        Case("flashqla-cp-planner-contract", "prefill",
             ("scripts/prefill/flashqla/private/test_cp_planner_cpu.py",), 600),
    ])
    return result
