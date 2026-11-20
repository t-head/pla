# SPDX-License-Identifier: Apache-2.0
"""Source-validated GDN per-rank cycle cases; defaults describe one operator layer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

__all__ = ["CycleCase", "load_models", "matrix", "make_case"]

_TPS = (1, 2, 4, 8, 16)
_MODELS = {
    "Qwen3-Next-80B-A3B": {
        "H": 16, "HV": 32, "K": 128, "V": 128,
        "conv_width": 4, "linear_layers": 36,
    },
    "Qwen3.8-Flash-Next": {
        "H": 16, "HV": 48, "K": 128, "V": 128,
        "conv_width": 4, "linear_layers": 36,
    },
    "Qwen3.8-Max": {
        "H": 16, "HV": 128, "K": 128, "V": 128,
        "conv_width": 4, "linear_layers": 69,
    },
}
# Max is opt-in so the established default matrix and its evidence stay unchanged.
_DEFAULT_MODELS = ("Qwen3-Next-80B-A3B", "Qwen3.8-Flash-Next")
_SCOPES = {
    "Qwen3-Next-80B-A3B": (None, "qwen3_next", "qwen3_next"),
    "Qwen3.8-Flash-Next": ("text_config", "qwen4_exp", "qwen4_exp_text"),
    "Qwen3.8-Max": ("text_config", "qwen3_5_moe", "qwen3_5_moe_text"),
}
_TOTAL_LAYERS = {
    "Qwen3-Next-80B-A3B": 48,
    "Qwen3.8-Flash-Next": 48,
    "Qwen3.8-Max": 92,
}
_SOURCE_FIELDS = {
    "H": "linear_num_key_heads",
    "HV": "linear_num_value_heads",
    "K": "linear_key_head_dim",
    "V": "linear_value_head_dim",
    "conv_width": "linear_conv_kernel_dim",
}


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _known_model(model):
    if not isinstance(model, str) or model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {tuple(_MODELS)}")


def _expect(source, field, actual, expected):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{source}: {field} expected {expected!r}, got {actual!r}; "
            "refusing a changed cycle matrix"
        )


def _model_metadata(models, model):
    _known_model(model)
    if model not in models:
        raise ValueError(f"model {model!r} is missing from models")
    info = models[model]
    if not isinstance(info, dict):
        raise ValueError(f"{model}: metadata must be a dict")
    for field, expected in _MODELS[model].items():
        _expect(model, field, info.get(field), expected)
    return info


@dataclass(frozen=True)
class CycleCase:
    model: str
    tp: int
    B: int
    T: int
    A: int
    mode: str
    H: int
    HV: int
    K: int = 128
    V: int = 128
    L: int = 16
    layers: int = 1

    def validate(self) -> CycleCase:
        _known_model(self.model)
        for name in ("tp", "B", "T", "A", "H", "HV", "K", "V", "L", "layers"):
            _positive_int(name, getattr(self, name))
        if self.tp not in _TPS:
            raise ValueError(f"tp must be one of {_TPS}, got {self.tp}")
        if self.mode not in ("decode", "mtp"):
            raise ValueError(f"mode must be 'decode' or 'mtp', got {self.mode!r}")
        if self.HV % self.H:
            raise ValueError("HV must be an integer multiple of H (GQA)")
        if self.K != 128 or self.V != 128:
            raise ValueError("GDN cycle cases require K=V=128")
        if self.L & (self.L - 1):
            raise ValueError("L must be a power of two")
        if self.mode == "decode":
            if self.T != 1 or self.A != 1:
                raise ValueError("decode requires T=A=1")
        elif self.A > self.T or self.L < 2 * self.T:
            raise ValueError("mtp requires 0 < A <= T and L >= 2*T")
        return self

    @property
    def rounds(self) -> int:
        self.validate()
        if self.mode == "decode":
            return self.L
        return max(1, (self.L - 2 * self.T) // self.A + 1)

    @property
    def key(self) -> str:
        self.validate()
        return (
            f"{self.model}_tp{self.tp}_b{self.B}_t{self.T}_a{self.A}"
            f"_{self.mode}_l{self.L}_layers{self.layers}"
            f"_h{self.H}_hv{self.HV}_k{self.K}_v{self.V}"
        )


def load_models(opf_root, names=None) -> dict:
    """Validate canonical model JSON parameters and return their source SHA256."""
    root = Path(opf_root).expanduser().resolve()
    names = _axis("model", _DEFAULT_MODELS if names is None else names)
    models = {}
    for model in names:
        _known_model(model)
        expected = _MODELS[model]
        path = root / "opf" / "models" / "config" / f"{model}.json"
        raw = path.read_bytes()
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"invalid model JSON at {path}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"{path}: expected a JSON object")
        scope, root_type, config_type = _SCOPES[model]
        _expect(path, "model_type", document.get("model_type"), root_type)
        config = document if scope is None else document.get(scope)
        if not isinstance(config, dict):
            raise ValueError(f"{path}: missing object {scope}")
        _expect(path, "config.model_type", config.get("model_type"), config_type)
        info = {}
        for field, source_field in _SOURCE_FIELDS.items():
            value = config.get(source_field)
            _expect(path, source_field, value, expected[field])
            info[field] = value
        dtype_fields = [key for key in ("dtype", "torch_dtype") if key in config]
        if not dtype_fields:
            raise ValueError(f"{path}: missing bf16 input dtype (dtype/torch_dtype)")
        for field in dtype_fields:
            _expect(path, field, config[field], "bfloat16")
        total = config.get("num_hidden_layers")
        interval = config.get("full_attention_interval")
        _expect(path, "num_hidden_layers", total, _TOTAL_LAYERS[model])
        _expect(path, "full_attention_interval", interval, 4)
        schedule = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(total)
        ]
        if "layer_types" in config:
            _expect(path, "layer_types", config["layer_types"], schedule)
            schedule = config["layer_types"]
        info["linear_layers"] = schedule.count("linear_attention")
        _expect(path, "linear_layers", info["linear_layers"], expected["linear_layers"])
        info.update(source_path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        models[model] = info
    return models


def make_case(models, model, tp, B, T=4, A=4, mode="mtp", L=16, layers=1) -> CycleCase:
    """Build one case, including synthetic large B; decode callers pass T=A=1."""
    info = _model_metadata(models, model)
    _positive_int("tp", tp)
    # Match OPF generator/attention/decode.py:45-46; never floor-divide heads.
    return CycleCase(
        model=model, tp=tp, B=B, T=T, A=A, mode=mode,
        H=(info["H"] + tp - 1) // tp,
        HV=(info["HV"] + tp - 1) // tp,
        K=info["K"], V=info["V"], L=L, layers=layers,
    ).validate()


def _axis(name, values):
    values = tuple(values)
    for index, value in enumerate(values):
        if value in values[:index]:
            raise ValueError(f"duplicate {name} value {value!r}")
    return values


def matrix(
    models, *, modes=("decode", "mtp"), tps=(1, 2, 4, 8, 16),
    batches=(1, 8, 32, 128, 256), widths=(2, 4, 6, 8), L=16, layers=1,
) -> list[CycleCase]:
    """Ordered model/TP/B/mode/T/A grid; defaults yield 50 decode + 1000 MTP."""
    modes = _axis("mode", modes)
    tps = _axis("tp", tps)
    batches = _axis("batch", batches)
    widths = _axis("width", widths)
    for mode in modes:
        if mode not in ("decode", "mtp"):
            raise ValueError(f"unknown mode {mode!r}")
    for tp in tps:
        _positive_int("tp", tp)
        if tp not in _TPS:
            raise ValueError(f"tp must be one of {_TPS}, got {tp}")
    for B in batches:
        _positive_int("B", B)
    for T in widths:
        _positive_int("T", T)
        if "mtp" in modes and (type(L) is not int or L < 2 * T):
            raise ValueError("mtp requires L >= 2*T for every requested width")
    _positive_int("L", L)
    _positive_int("layers", layers)
    if L & (L - 1):
        raise ValueError("L must be a power of two")
    cases = []
    for model in models:
        _model_metadata(models, model)
        for tp in tps:
            for B in batches:
                for mode in modes:
                    if mode == "decode":
                        cases.append(make_case(models, model, tp, B, 1, 1, mode, L, layers))
                    else:
                        for T in widths:
                            for A in range(1, T + 1):
                                cases.append(make_case(models, model, tp, B, T, A, mode, L, layers))
    return cases
