# SPDX-License-Identifier: Apache-2.0
"""Load a SHA-verified original SGLang module in memory, without installing SGLang.

Only the two platform-capability imports are stubbed. Original Triton kernels,
wrappers, source line numbers and defaults are retained. No vendored kernel copy,
source-file edits, persistent generated files, or global SGLang package stubs.
"""
import ast
import builtins
import hashlib
import linecache
from pathlib import Path
import sys
import types

from reference import PUBLIC_FUNCTIONS, SOURCE_COMMIT, SOURCE_RELATIVE, SOURCE_SHA256


def check_source(root):
    path = Path(root).expanduser().resolve() / SOURCE_RELATIVE
    source = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != SOURCE_SHA256:
        raise ValueError(f"unknown upstream source SHA256 {sha}; expected {SOURCE_SHA256} at {SOURCE_COMMIT}")
    tree = ast.parse(source, filename=str(path))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = set(PUBLIC_FUNCTIONS) - functions.keys()
    if missing:
        raise ValueError(f"missing public functions: {sorted(missing)}")
    imports = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module and n.module.startswith("sglang")}
    expected = {"sglang.kernels.ops.attention.fla.utils", "sglang.srt.utils"}
    if imports != expected:
        raise ValueError(f"unexpected SGLang dependencies: {imports}")
    return source, {"source_path": str(path), "source_commit": SOURCE_COMMIT,
                    "source_sha256": sha, "sha_verified": True,
                    "provenance": "original public API differential, not extracted upstream numerical UT"}


def load_upstream(root, *, tf32_supported=False, gfx95_supported=False):
    source, provenance = check_source(root)
    name = "_gdn_replay_upstream_" + SOURCE_SHA256[:16]
    # Triton uses inspect.getsourcelines: a private in-memory source entry gives
    # it the unmodified file, without a 1400-line vendor file or a temp disk file.
    filename = f"<{name}.py>"
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = ""
    fla = types.ModuleType("sglang.kernels.ops.attention.fla.utils")
    fla.is_tf32_supported = bool(tf32_supported)
    utils = types.ModuleType("sglang.srt.utils")
    utils.is_gfx95_supported = lambda: bool(gfx95_supported)
    stubs = {fla.__name__: fla, utils.__name__: utils}
    original_import = builtins.__import__

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in stubs:
            return stubs[name]
        return original_import(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=scoped_import)
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        linecache.cache.pop(filename, None)
        raise
    module.PROVENANCE = provenance
    return module
