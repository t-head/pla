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

"""``pla.decode.kda`` — Kimi Delta Attention decode entry points.

Both ops are implemented entirely in C++/CUDA (``src/decode``):

``fused_recurrent_kda``
    Recurrent KDA from raw gate/beta inputs. Signature-compatible with vLLM's
    Triton ``fused_recurrent_kda``, plus a trailing ``is_sglang`` switch so
    sglang can share the entry point. Used for speculative decoding only.

``fused_kda_decode_mega_forward``
    Single-kernel KDA decode exposing the ``torch.ops._C.fused_kda_decode``
    I/O contract (conv + gating + recurrence + norm), operating in place.
"""

# MUST come before importing the extension: it links against libtorch/libc10 with
# no RPATH, so those libraries have to be loaded into the process first. Without
# this the import dies with `ImportError: libc10.so: cannot open shared object
# file` on any host whose LD_LIBRARY_PATH does not include torch/lib.
import torch  # noqa: F401

from .._C import fused_recurrent_kda
from .._C import fused_kda_decode_mega_forward

__all__ = [
    'fused_recurrent_kda',
    'fused_kda_decode_mega_forward',
]
