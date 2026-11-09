"""CPU checks for consistent native KDA host/device numerical build modes."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = ("pla.decode.kda_replayssm_sglang", "pla.decode.kda_replayssm_vllm")


def kda_extensions(value):
    # Evaluate the real KDA extension construction without importing torch or
    # invoking setuptools/build_ext. The compiler argument lists are the result
    # under test, not a separate reimplementation of their construction.
    tree = ast.parse((ROOT / "setup.py").read_text())
    loop = next(node for node in tree.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == "kda_namespace")
    module = ast.Module(body=[loop], type_ignores=[])
    captured = []

    def extension(**kwargs):
        captured.append(kwargs)
        return kwargs

    namespace = dict(os=os, here=str(ROOT), NAMESPACES=NAMESPACES,
                     HGCC_THREADS="2", ext_modules=[], Extension=extension)
    with patch.dict(os.environ):
        os.environ.pop("PLA_KDA_NATIVE_FP64_REPAIR", None)
        if value is not None:
            os.environ["PLA_KDA_NATIVE_FP64_REPAIR"] = value
        exec(compile(module, str(ROOT / "setup.py"), "exec"), namespace)
    return captured


class KDANumericalBuildModeTests(unittest.TestCase):
    def test_default_and_explicit_modes_match_host_and_device(self):
        for value, expected in ((None, "0"), ("0", "0"), ("1", "1")):
            with self.subTest(value=value):
                extensions = kda_extensions(value)
                self.assertEqual([e["name"] for e in extensions],
                                 [n + "._C" for n in NAMESPACES])
                for extension in extensions:
                    for compiler in ("cxx", "hgcc"):
                        defines = [flag for flag in extension["extra_compile_args"][compiler]
                                   if flag.startswith("-DPLA_KDA_NATIVE_FP64_REPAIR")]
                        self.assertEqual(defines, ["-DPLA_KDA_NATIVE_FP64_REPAIR=" + expected])

    def test_invalid_mode_is_rejected_before_extension_creation(self):
        for value in ("", "2", "true", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                    kda_extensions(value)


if __name__ == "__main__":
    unittest.main()
