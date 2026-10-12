/*
 * Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "launchers.h"
#include "runtime_heads.h"
#include <stdexcept>
#include <vector>

namespace fla_runtime {
const Launchers& head_adapter__fla_h0();
#define FLA_DECLARE(H) const Launchers& head_adapter__fla_h##H();
FLA_EACH_STATIC_HEAD(FLA_DECLARE)
#undef FLA_DECLARE
const Launchers& launchers_for(int heads) {
  if (heads < 1 || heads > 65535) throw std::invalid_argument("head exceeds the device grid range");
  switch (heads) {
#define FLA_CASE(H) case H: return head_adapter__fla_h##H();
    FLA_EACH_STATIC_HEAD(FLA_CASE)
#undef FLA_CASE
    default: return head_adapter__fla_h0();
  }
}
std::vector<int> supported_heads() {
#define FLA_VALUE(H) H,
  return {FLA_EACH_STATIC_HEAD(FLA_VALUE)};
#undef FLA_VALUE
}
std::vector<int> supported_cp_heads() {
#define FLA_VALUE(H) H,
  return {FLA_EACH_CP_HEAD(FLA_VALUE)};
#undef FLA_VALUE
}
}  // namespace fla_runtime
