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

#pragma once

#include <cstddef>

#if defined(__CUDACC__) || defined(__HGGC__)
#define PLA_K_LAST_HD __host__ __device__
#else
#define PLA_K_LAST_HD
#endif

namespace pla {
// One byte-offset contract for host allocation and both device kernels. Every
// region starts at 16 bytes: float4 and cp.async require more than float alignment.
struct KLastSharedLayout {
    std::size_t h, q, k, bv, reduction_q, reduction_k, beta, bg;
    std::size_t v_mtp, q_mtp, k_mtp, q_raw, k_raw, beta_mtp, bg_mtp, total_bytes;

    PLA_K_LAST_HD static constexpr std::size_t align16(std::size_t n) {
        return (n + 15) & ~std::size_t(15);
    }

    PLA_K_LAST_HD constexpr KLastSharedLayout(
        int stages, int key_dim, int warps, int tile_v, int heads,
        bool is_kda, bool mtp, int token_capacity, int scalar_bytes = 2)
        : h(0), q(align16(4ull * stages * key_dim * warps)),
          k(align16(q + 4ull * key_dim)),
          bv(align16(k + 4ull * key_dim)),
          reduction_q(align16(bv + std::size_t(scalar_bytes) * tile_v * heads)),
          reduction_k(align16(reduction_q + 4ull * warps)),
          beta(align16(reduction_k + 4ull * warps)),
          bg(align16(beta + 4ull * heads)),
          v_mtp(align16(bg + 4ull * heads * (is_kda ? key_dim : 1))),
          q_mtp(align16(v_mtp + std::size_t(scalar_bytes) * token_capacity * tile_v * heads)),
          k_mtp(align16(q_mtp + 4ull * token_capacity * key_dim)),
          q_raw(align16(k_mtp + 4ull * token_capacity * key_dim)),
          k_raw(align16(q_raw + std::size_t(scalar_bytes) * token_capacity * key_dim)),
          beta_mtp(align16(k_raw + std::size_t(scalar_bytes) * token_capacity * key_dim)),
          bg_mtp(align16(beta_mtp + 4ull * token_capacity * heads)),
          total_bytes(mtp ? align16(bg_mtp + 4ull * token_capacity * heads * (is_kda ? key_dim : 1)) : v_mtp) {}
};
}  // namespace pla

#undef PLA_K_LAST_HD
