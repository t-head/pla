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

"""``pla.prefill.flashkdapro.private`` — prefill building blocks, not public API.

``flashkda_prepare`` is Kernel 1 on its own: it returns the six workspace
tensors that :func:`pla.prefill.flashkdapro.flashkda_fwd` normally consumes
internally, so a unit test can check them against a torch reference. Calling it
directly means owning the workspace contract (chunk size 16, the packed
``[total_tokens, H, 128]`` bf16 layouts, int64 offsets); no compatibility is
promised across versions.

The recurrence stage is not exposed separately — it reads the workspace through
an internal byte-image ABI rather than a stable tensor signature.
"""

# MUST come before importing the extension: it links against libtorch/libc10 with
# no RPATH, so those libraries have to be loaded into the process first. Without
# this the import dies with `ImportError: libc10.so: cannot open shared object
# file` on any host whose LD_LIBRARY_PATH does not include torch/lib.
import torch  # noqa: F401

from .._C import flashkda_prepare

__all__ = [
    'flashkda_prepare',
]
