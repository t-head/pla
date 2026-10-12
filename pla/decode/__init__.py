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

"""``pla.decode`` — k-last fused sigmoid gating delta rule (GDN) decode kernels.

The k-last initial state and MTP intermediate states support FP32/BF16; when both are given their dtypes must match.
State is stored natively and written back in place; the recurrence accumulates in FP32 and rounds only when writing snapshots/the final state.

Related namespaces:

    pla.decode.kda              Kimi Delta Attention decode
    pla.decode.kda_mtp_sglang   SGLang KDA MTP verify + commit kernels
    pla.decode.private          tunable entry points (auto-tune only)
    pla.decode.gdn.v_last       v-last (V-major) GDN layout, separate extension

The ops are resolved lazily (PEP 562) rather than imported eagerly. Importing
this package must not require this namespace's own extension, because
``pla.decode.gdn.v_last`` is nested underneath it and Python imports the parent
first: an eager ``from ._C import ...`` here would make a v_last-only build
(``--namespace pla.decode.gdn.v_last``) install a package that cannot be
imported at all.
"""

import importlib

# MUST come before any import of ._C, and is why this import is not "unused".
# The extension links against libtorch/libc10 but is built with no RPATH, so its
# NEEDED entries only resolve if those libraries are already loaded into the
# process -- which is exactly what importing torch does. Without this, a plain
# `from pla.decode import ...` dies with
#     ImportError: libc10.so: cannot open shared object file
# on any host whose LD_LIBRARY_PATH does not happen to include torch/lib.
import torch  # noqa: F401

__all__ = [
    'fused_sigmoid_gating_delta_rule_forward_k_last',
    'fused_sigmoid_gating_delta_rule_forward_k_last_packed',
]

_EXTENSION_MISSING = (
    "pla.decode's compiled extension (pla.decode._C) is not present, so "
    "{name!r} is unavailable. This namespace was not built -- build it with\n"
    "    python scripts/build.py --namespace pla.decode\n"
    "or build everything with a plain `pip install .`. Note that "
    "pla.decode.gdn.v_last and pla.prefill.flashqla are separate extensions and "
    "do not provide this op."
)


def __getattr__(name):
    if name == '_C' or name in __all__:
        # importlib, not `from . import _C`: when the extension is absent the
        # latter falls back to getattr(pla.decode, '_C'), which re-enters this
        # very function and recurses until RecursionError.
        try:
            mod = importlib.import_module('._C', __name__)
        except ModuleNotFoundError as exc:
            # The extension really is not installed -- this namespace was not
            # built. Anything else (a dlopen failure such as a missing
            # libc10.so) must propagate untouched: claiming "not built" for a
            # link error sends people off to rebuild something that is already
            # there.
            if exc.name in (f'{__name__}._C', '_C'):
                raise ImportError(_EXTENSION_MISSING.format(name=name)) from exc
            raise
        globals()['_C'] = mod          # cache: __getattr__ is not consulted again
        return mod if name == '_C' else getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
