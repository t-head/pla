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

"""CPU regression: FlashQLA prefill head-config surfaces must agree.

Three lists gate one (H, Hg) shape and drift silently when only one is edited:
``SUPPORTED_HEAD_CONFIGS`` (Python validation), ``is_supported_head_config``
(host TORCH_CHECK) and the ``FF_DISPATCH_ALL`` table (kernel instantiation).
A whitelist entry without a dispatch entry passes validation and then aborts
inside the launcher, so compare all three here.

Run: python3 tests/test_prefill_head_config_contract.py
"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "pla/prefill/flashqla/ops.py"
HOST_CC = ROOT / "src/prefill/flashqla/host/torch_extension_cute.cc"
FUSED_FWD = ROOT / "src/prefill/flashqla/kernels/fused_fwd_cute.cu"


def python_configs():
    tree = ast.parse(OPS.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "SUPPORTED_HEAD_CONFIGS" not in targets:
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not call.args:
            raise AssertionError("SUPPORTED_HEAD_CONFIGS must wrap a literal")
        literal = ast.literal_eval(call.args[0])
        return {tuple(pair) for pair in literal}
    raise AssertionError("SUPPORTED_HEAD_CONFIGS not found in ops.py")


def _returned_expression(expression):
    """Flatten the C++ ``return`` statement into one Python expression."""
    lines = [line.split("//", 1)[0] for line in expression.splitlines()]
    text = " ".join(" ".join(lines).split()).rstrip(";")
    if text.startswith("return"):
        text = text[len("return"):].strip()
    return text


def _predicate(expression):
    """Parse the C++ predicate as Python and keep only comparison nodes."""
    expression = expression.replace("||", " or ").replace("&&", " and ")
    expression = re.sub(r"\btrue\b", "True", expression)
    expression = re.sub(r"\bfalse\b", "False", expression)
    tree = ast.parse(_returned_expression(expression), mode="eval").body
    allowed = (ast.BoolOp, ast.Compare, ast.Name, ast.Constant, ast.UnaryOp)
    for node in ast.walk(tree):
        if not isinstance(node, allowed + (ast.And, ast.Or, ast.Eq, ast.NotEq,
                                           ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                                           ast.Not, ast.Load)):
            raise AssertionError(f"unsupported node in head predicate: {node!r}")
    return tree


def _evaluate(node, scope):
    if isinstance(node, ast.BoolOp):
        values = [_evaluate(value, scope) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, scope)
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, scope)
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, scope)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            else:
                ok = left >= right
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        return scope[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unsupported node in head predicate: {node!r}")


def host_cc_configs():
    source = HOST_CC.read_text(encoding="utf-8")
    start = source.index("bool is_supported_head_config(")
    body_start = source.index("{", start) + 1
    body_end = source.index("\n}", body_start)
    predicate = _predicate(source[body_start:body_end])
    configs = set()
    for heads in range(1, 257):
        for key_heads in range(1, heads + 1):
            if _evaluate(predicate, {"H": heads, "Hg": key_heads}):
                configs.add((heads, key_heads))
    return configs


def fused_fwd_configs():
    source = FUSED_FWD.read_text(encoding="utf-8")
    start = source.index("#define FF_DISPATCH_ALL(")
    end = source.index("#undef FF_DISPATCH_ALL", start)
    table = source[start:end]
    pairs = re.findall(
        r"FF_DISPATCH_E\(\s*ElementType\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", table)
    return {(int(h), int(hg)) for h, hg in pairs}


class HeadConfigContractTest(unittest.TestCase):
    def test_python_and_host_check_agree(self):
        self.assertEqual(python_configs(), host_cc_configs())

    def test_dispatch_table_covers_every_supported_shape(self):
        missing = python_configs() - fused_fwd_configs()
        self.assertEqual(missing, set(), f"no FF_DISPATCH_E entry for {missing}")

    def test_ratio_one_is_supported(self):
        self.assertIn((16, 16), python_configs())
        self.assertIn((16, 16), fused_fwd_configs())

    def test_every_shape_divides(self):
        for heads, key_heads in python_configs():
            with self.subTest(heads=heads, key_heads=key_heads):
                self.assertEqual(heads % key_heads, 0)


if __name__ == "__main__":
    unittest.main()
