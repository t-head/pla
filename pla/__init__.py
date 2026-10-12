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

"""PLA — PPU Linear Attention kernels.

The public API is organised by namespace; nothing is exported from the root
package. Import the namespace you need:

    pla.decode                    fused sigmoid gating delta rule (GDN) decode
    pla.decode.kda                Kimi Delta Attention decode
    pla.decode.kda_mtp_sglang     SGLang KDA MTP verify + commit kernels
    pla.decode.private            tunable entry points (auto-tune only)
    pla.prefill.flashqla          FlashQLA prefill forward
    pla.prefill.flashqla.private  FlashQLA prefill building blocks
    pla.prefill.flashkdapro        FlashKDA (Kimi Delta Attention) prefill forward
    pla.prefill.flashkdapro.private  FlashKDA prefill building blocks

Each namespace maps onto one compiled extension: the decode families and the two
prefill families all live in separate shared objects, so a build restricted to
one namespace still yields an importable package for that namespace.
"""

__version__ = "1.0.0"

__all__ = []
