"""Catch host declarations that silently drift from CUDA launch definitions."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "src/prefill/flashqla"


def _parameters(path, name):
    source = path.read_text()
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    matches = list(re.finditer(r"\bvoid\s+" + re.escape(name) + r"\s*\(", source))
    assert len(matches) == 1, (path, name, len(matches))
    start = matches[0].end()
    end = source.index(")", start)
    parameters = []
    for argument in source[start:end].split(","):
        argument = argument.split("=", 1)[0].strip()
        # These launch APIs contain primitive/pointer parameters, without
        # function pointers or templates. Every declaration names its args.
        assert re.search(r"\b[A-Za-z_]\w*$", argument), (path, argument)
        type_name = re.sub(r"\b[A-Za-z_]\w*$", "", argument).strip()
        assert type_name, (path, argument)
        parameters.append(re.sub(r"\s+", "", type_name))
    return parameters


@pytest.mark.parametrize("name,definition", [
    ("launch_prepare_h_cp_cute", "kernels/prepare_h_cute.cu"),
    ("launch_fused_fwd_cp_cute", "kernels/fused_fwd_cute.cu"),
])
def test_host_cuda_launch_signature(name, definition):
    assert _parameters(SOURCE / "host/torch_extension_cute.cc", name) == _parameters(
        SOURCE / definition, name)


def test_packed_header_cuda_launch_signature():
    name = "launch_gdn_packed_fwd_tc01"
    assert _parameters(SOURCE / "kernels/packed_coefficients_api.h", name) == _parameters(
        SOURCE / "kernels/fused_fwd_cute.cu", name)


@pytest.mark.parametrize("name", ["launch_snapshot", "launch_guard", "launch_repair"])
def test_precision_header_cuda_launch_signature(name):
    assert _parameters(SOURCE / "kernels/precision_repair_api.h", name) == _parameters(
        SOURCE / "kernels/precision_repair.cu", name)
