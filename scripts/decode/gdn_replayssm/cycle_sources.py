# SPDX-License-Identifier: Apache-2.0
"""Load pinned original GDN/conv/scatter functions on PPU1.5 without installing SGLang."""

import ast
import builtins
import functools
import hashlib
import linecache
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import typing

__all__ = ["load_cycle_sources", "source_manifest"]

_SOURCE_COMMIT = "c843608d390ac416791e49e0252d2f17fafdc489"
_OPS = "python/sglang/kernels/ops/"
_SPEC_PUBLIC = (
    "gdn_replayssm_spec_decode",
    "commit_gdn_replayssm_spec",
    "commit_gdn_replayssm_circular",
)
_PROFILES = {
    "conv": {
        "relative": _OPS + "mamba/causal_conv1d_triton.py",
        "sha256": "6c05f0534a3330ccfddb75378a8e13a2d08631f80eb77108473f060dbe2b217d",
        "required": ("_causal_conv1d_update_kernel", "causal_conv1d_update"),
        "constants": ("PAD_SLOT_ID",),
    },
    "scatter": {
        "relative": _OPS + "mamba/mamba_state_scatter_triton.py",
        "sha256": "47989f7434e7f7f06892fb7005306a003e49bf850916085775c56f12629b4784",
        "required": (
            "_require_entry_contiguous_dst",
            "_prepare_src_rows",
            "_fused_conv_window_scatter_with_mask_kernel",
            "fused_conv_window_scatter_with_mask",
            "_fused_mamba_state_scatter_with_mask_kernel",
            "fused_mamba_state_scatter_with_mask",
        ),
        "constants": (),
    },
    "recurrent": {
        "relative": _OPS + "attention/linear/fused_sigmoid_gating_recurrent.py",
        "sha256": "4d1e5109fa8d50cf5b0202fe3b948dac742d1c4804c835e1d02d08c4a2fdcbdf",
        "required": (
            "_select_recurrent_launch_config",
            "fused_sigmoid_gating_delta_rule_update_kernel",
            "fused_sigmoid_gating_delta_rule_update",
        ),
        "constants": (
            "_EX2_APPROX_FTZ_F32",
            "_LG2_APPROX_FTZ_F32",
            "_UNARY_F32_REGISTERS",
        ),
    },
    "spec": {
        "relative": _OPS + "attention/linear/gdn_replayssm_spec_decode.py",
        "sha256": "71a49ecdef3f598263a2681839ba03106f95dba7b0e6bdba3d0496c7da830bc6",
        "required": _SPEC_PUBLIC + (
            "_finish_gdn_replayssm_circular_fold_kernel",
            "_advance_gdn_spec_cursors_kernel",
        ),
        "constants": (),
    },
}
_LOADER_SHA = "71cdcfd130731b6a9e6d08355b452df226ee86709eb95ac2c9e7b5211e449323"
_EVIDENCE = {
    "python/sglang/kernels/arch/base.py": (
        "e1b8cb284bc1dda8abb770b58e694ebf24ceaa34b9cc8ecc2bab3e304696651a",
        70,
        "DeviceArch.supports_pdl defaults to False",
    ),
    "python/sglang/kernels/arch/_cuda.py": (
        "7d877f9f52490e34b1ead0ea255a8d4b75395df50777807ed6ce2bf9eba7d17d",
        43,
        "CudaArch.supports_pdl requires major >= 9; CC8.9 does not qualify",
    ),
}
_EXCLUSIONS = (
    "all devices except ZW-M890P / PPU1.5 / CUDA API CC8.9",
    "training/autograd",
    "tree attention (all retrieve_* arguments must be None)",
    "KDA, lower_bound gating, match_cutedsl_decode and NVIDIA PTX parity paths",
    "scatter multi/ptr_table, tracking and unrelated SGLang APIs",
    "GPU correctness/performance validation (not performed by this loader)",
)


def _verified_text(path, expected):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(
            f"source SHA256 mismatch at {path}: got {actual}, expected {expected}; "
            f"refusing the {_SOURCE_COMMIT} cycle profile"
        )
    return raw.decode("utf-8")


def _select_ast(source, path, required, constants=(), *, whole=False):
    """Follow local function/constant references, without editing any node."""
    tree = ast.parse(source, filename=str(path))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    missing = set(required) - functions.keys()
    if missing:
        raise ValueError(f"missing source functions at {path}: {sorted(missing)}")
    if whole:
        return tree
    available = dict(functions)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    available[target.id] = node
    missing_constants = set(constants) - available.keys()
    if missing_constants:
        raise ValueError(f"missing source constants at {path}: {missing_constants}")
    selected = set(required)
    pending = list(required)
    while pending:
        name = pending.pop()
        references = {
            node.id for node in ast.walk(available[name])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        dependencies = (references & available.keys()) - selected
        selected.update(dependencies)
        pending.extend(dependencies)
    retained = {id(available[name]) for name in selected}
    return ast.Module(
        body=[node for node in tree.body if id(node) in retained], type_ignores=[]
    )


def _node_records(tree):
    records = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names = [node.name]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            start = node.lineno
        else:
            continue
        records.append({
            "names": names, "kind": type(node).__name__,
            "lineno": node.lineno, "source_start": start,
            "end_lineno": node.end_lineno,
        })
    return records


def _feature_overrides(name):
    if name == "scatter":
        return {}
    if name == "spec":
        return {
            "is_tf32_supported": True, "is_gfx95_supported()": False,
            "basis": "explicitly authorized ZW-M890P profile; no global TF32 setting",
        }
    result = {
        "is_arch_support_pdl()": False,
        "pdl_basis": [f"{path}:{entry[1]}" for path, entry in _EVIDENCE.items()],
        "target_limit": "ZW-M890P PPU1.5, CC8.9 only; no PDL/GDC",
    }
    if name == "recurrent":
        result.update({
            "_is_hip": False, "_is_gfx95": False, "_is_sm90": False,
            "_is_sm103": False, "_is_sm107": False,
            "is_cuda()": True, "get_device_sm()": 89,
            "envs": "fail-fast object, no environment values or FP8BlockGEMV implementation",
            "_select_recurrent_launch_config": (
                "GDN-only admission guard delegates unchanged to original selector; "
                "original available in ORIGINAL_FUNCTIONS and via inspect.unwrap"
            ),
            "asm_constants": (
                "original three tl.constexpr ASM strings retained for Triton AST "
                "dependency resolution, not an authorization to execute PTX"
            ),
        })
    return result


def _validated_sources(root):
    root = Path(root).expanduser().resolve()
    snapshots = {}
    profiles = {}
    for name, config in _PROFILES.items():
        path = root / config["relative"]
        source = _verified_text(path, config["sha256"])
        tree = _select_ast(
            source, path, config["required"], config["constants"],
            whole=name == "spec",
        )
        # Static syntax validation only, never imports or executes source here.
        compile(tree, str(path), "exec", dont_inherit=True)
        snapshots[name] = (path, source, tree)
        profiles[name] = {
            "source_path": str(path), "source_relative": config["relative"],
            "source_commit": _SOURCE_COMMIT,
            "source_sha256": config["sha256"], "sha_verified": True,
            "mode": "upstream.py::load_upstream (whole source)" if name == "spec"
                    else "unchanged AST dependency closure",
            "function_names": [n.name for n in tree.body if isinstance(n, ast.FunctionDef)],
            "required_functions": list(config["required"]),
            "retained_nodes": _node_records(tree),
            "feature_overrides": _feature_overrides(name),
            "scope_exclusions": list(_EXCLUSIONS),
            "original_header": source.splitlines()[:6] if name == "conv" else [],
        }
    evidence = []
    for relative, (sha, line, meaning) in _EVIDENCE.items():
        path = root / relative
        _verified_text(path, sha)
        evidence.append({
            "source_path": str(path), "source_commit": _SOURCE_COMMIT,
            "source_sha256": sha, "sha_verified": True,
            "line": line, "meaning": meaning,
        })
    loader_path = Path(__file__).resolve().with_name("upstream.py")
    loader_source = _verified_text(loader_path, _LOADER_SHA)
    loader_tree = _select_ast(
        loader_source, loader_path, ("check_source", "load_upstream")
    )
    compile(loader_tree, str(loader_path), "exec", dont_inherit=True)
    snapshots["loader"] = (loader_path, loader_source, loader_tree)
    provenance = {
        "profile": "ZW-M890P / PPU1.5 / CC8.9 GDN linear-chain inference",
        "source_root": str(root), "source_commit": _SOURCE_COMMIT,
        "commit_semantics": (
            "audited local baseline commit; exact full-file SHA256 checks are "
            "authoritative, no git checkout or git executable required"
        ),
        "sources": profiles, "feature_evidence": evidence,
        "scope_exclusions": list(_EXCLUSIONS),
        "gpu_validated": False,
        "loader": {
            "source_path": str(loader_path), "source_sha256": _LOADER_SHA,
            "sha_verified": True, "retained_nodes": _node_records(loader_tree),
            "bindings": (
                "stdlib imports plus pinned PUBLIC_FUNCTIONS, SOURCE_RELATIVE, "
                "SOURCE_SHA256; SOURCE_COMMIT metadata set to current audited "
                "baseline instead of reference.py's historical "
                "40a64e1b5ba00c73b7f8a123a903a2123b7d467c"
            ),
            "package_imports": "no reference.py import and no global sglang stubs",
        },
    }
    return snapshots, provenance


def source_manifest(root):
    """Return a JSON-serializable, SHA-verified manifest; stdlib only, no device probe."""
    return _validated_sources(root)[1]


def _compile_module(label, snapshot, bindings):
    path, source, tree = snapshot
    identity = hashlib.sha256((str(path) + source).encode("utf-8")).hexdigest()[:20]
    name = f"_gdn_cycle_{label}_{identity}"
    filename = str(path)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = ""
    module.__dict__.update(bindings)
    previous_module = sys.modules.get(name)
    previous_cache = linecache.cache.get(filename)
    linecache.cache[filename] = (
        len(source.encode("utf-8")), None, source.splitlines(keepends=True), filename
    )
    sys.modules[name] = module
    try:
        exec(compile(tree, filename, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module
        if previous_cache is None:
            linecache.cache.pop(filename, None)
        else:
            linecache.cache[filename] = previous_cache
        raise
    module.ORIGINAL_FUNCTIONS = {
        node.name: module.__dict__[node.name]
        for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    return module


class _GDNOnlyEnvs:
    """Known unsupported dependency fails explicitly; unknown attributes also fail."""

    __slots__ = ()

    @property
    def SGLANG_ENABLE_FP8_BLOCK_GEMV(self):
        raise NotImplementedError("FP8BlockGEMV/KDA is outside the GDN-only cycle profile")


def _false():
    return False


def _cuda_api():
    return True


def _sm89():
    return 89


def _gdn_only_selector(original):
    @functools.wraps(original)
    def select(n, h, hv, k, v, is_kda, target_verify=False, *, cache_steps=0):
        if is_kda:
            raise NotImplementedError("cycle_sources supports GDN only, not KDA")
        return original(
            n, h, hv, k, v, is_kda, target_verify, cache_steps=cache_steps
        )
    return select


def _check_device(torch, device):
    if isinstance(device, int) and not isinstance(device, bool):
        device = f"cuda:{device}"
    requested = torch.device(device)
    if requested.type != "cuda" or getattr(torch.version, "hip", None):
        raise NotImplementedError("only ZW-M890P with CUDA device API / CC8.9 is authorized")
    if not torch.cuda.is_available():
        raise RuntimeError("ZW-M890P CUDA device API is unavailable; use source_manifest for dry-run")
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)
    if props.name.strip() != "ZW-M890P" or capability != (8, 9):
        raise NotImplementedError(
            f"unauthorized device {props.name!r}, CC{capability}; "
            "only ZW-M890P / CC8.9 is supported (not generic NVIDIA SM89)"
        )
    return {"device": f"cuda:{index}", "name": props.name, "capability": list(capability)}


def load_cycle_sources(root, device):
    """Load original GDN cycle kernels; the caller owns state, streams and linear-chain execution."""
    snapshots, provenance = _validated_sources(root)
    import torch

    provenance["device"] = _check_device(torch, device)
    import triton
    import triton.language as tl

    real = {"torch": torch, "triton": triton, "tl": tl}
    conv = _compile_module("conv", snapshots["conv"], {
        **real, "List": typing.List, "Optional": typing.Optional,
        "Union": typing.Union, "is_arch_support_pdl": _false,
    })
    # Deliberately no make_ptr_table or common import, real or fake.
    scatter = _compile_module("scatter", snapshots["scatter"], real)
    recurrent = _compile_module("recurrent", snapshots["recurrent"], {
        **real, "Optional": typing.Optional, "functools": functools,
        "is_arch_support_pdl": _false, "envs": _GDNOnlyEnvs(),
        "_is_hip": False, "_is_gfx95": False, "_is_sm90": False,
        "_is_sm103": False, "_is_sm107": False,
        "is_cuda": _cuda_api, "get_device_sm": _sm89,
    })
    recurrent._select_recurrent_launch_config = _gdn_only_selector(
        recurrent.ORIGINAL_FUNCTIONS["_select_recurrent_launch_config"]
    )
    loader = _compile_module("loader", snapshots["loader"], {
        "ast": ast, "builtins": builtins, "hashlib": hashlib,
        "linecache": linecache, "Path": Path, "sys": sys, "types": types,
        "PUBLIC_FUNCTIONS": _SPEC_PUBLIC, "SOURCE_COMMIT": _SOURCE_COMMIT,
        "SOURCE_RELATIVE": _PROFILES["spec"]["relative"],
        "SOURCE_SHA256": _PROFILES["spec"]["sha256"],
    })
    # Reuses its scoped platform-import hook, not process-wide SGLang stubs.
    spec = loader.load_upstream(
        provenance["source_root"], tf32_supported=True, gfx95_supported=False
    )
    provenance["sources"]["spec"]["upstream_loader_provenance"] = dict(spec.PROVENANCE)
    for name, module in (
        ("conv", conv), ("scatter", scatter), ("recurrent", recurrent), ("spec", spec)
    ):
        module.PROVENANCE = provenance["sources"][name]
    return SimpleNamespace(
        conv=conv, scatter=scatter, recurrent=recurrent, spec=spec,
        provenance=provenance,
    )
