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

/*
 * KDA (Kimi Delta Attention) recurrent decode entry point for vLLM and sglang.
 *
 * Exposes `fused_recurrent_kda` with the **same signature** as vLLM's Triton
 * implementation in
 * `vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py`, plus a
 * trailing `is_sglang` switch so both frameworks can share the entry point.
 *
 * This op is **only used for speculative decoding** -- plain decode goes
 * through the packed kernels in both frameworks.
 *
 * Design:
 *   * We reuse the *existing* IS_KDA template path of the CUDA kernel
 *     `fused_sigmoid_gating_delta_rule_forward_k_last` -- no new kernel
 *     instantiation is added.
 *   * `fuse_gate=true` (vLLM's default) maps onto the kernel's native mode,
 *     where the KDA decay and the beta sigmoid are computed **inside** the
 *     kernel:
 *
 *         g_gate = -exp(A_log) * softplus(raw_g + dt_bias)   // per-K decay
 *         beta   = sigmoid(raw_beta)
 *
 *     The pre-materialised gate path (`fuse_gate=false`) is not supported.
 *   * `ssm_state_indices` is forwarded to the kernel **unchanged**; the kernel
 *     understands both layouts, so no index juggling is needed here:
 *
 *       - 1D [N]: the slot is read directly and the recurrence commits the
 *         final state back to it once, at the end.
 *       - 2D [N, S] (vLLM spec decode): every draft token owns a slot. The
 *         kernel picks the starting column via num_accepted_tokens - 1 and
 *         writes each timestep's state back into ssm_state_indices[n, t].
 *
 *   * `is_sglang` selects the pad-slot convention, the one place where the two
 *     frameworks genuinely disagree:
 *
 *       - vLLM (false, the default because this signature mirrors vLLM's):
 *         BlockPool reserves block 0 as null_block, so slot <= 0 is padding.
 *       - sglang (true): MambaPool allocates an extra scratch row, so slot 0 is
 *         valid and PAD_SLOT_ID is -1.
 *
 *     Padded requests emit zeros and leave the state pool untouched.
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <limits>

#include "../../sanity_check.h"
#include "../../bypass_exec.h"
#include "../../op_forge_case.h"
#include "sanity_checks.h"

namespace py = pybind11;

// The Python implementation this replaces raised ValueError / NotImplementedError
// for its argument validation, and callers (including scripts/decode/test_vllm_kda.py
// and the vLLM / sglang integrations) catch those types. TORCH_CHECK would raise
// RuntimeError instead, so raise the original exception types explicitly.
[[noreturn]] static void raise_value_error(const std::string& msg) {
    throw py::value_error(msg);
}

[[noreturn]] static void raise_not_implemented(const std::string& msg) {
    PyErr_SetString(PyExc_NotImplementedError, msg.c_str());
    throw py::error_already_set();
}

// The kernel computes softplus with these constants (matching vLLM's Triton
// kernel, which hardcodes a threshold of 20.0 and unit beta).
static constexpr float kSoftplusBeta = 1.0f;
static constexpr float kSoftplusThreshold = 20.0f;

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last(
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    float softplus_beta,
    float softplus_threshold,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor b,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    std::optional<float> scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    bool disable_state_update,
    std::optional<torch::Tensor> intermediate_states_buffer,
    std::optional<torch::Tensor> intermediate_state_indices,
    std::optional<std::variant<torch::Tensor, int64_t>> cache_steps_or_num_accept,
    std::optional<torch::Tensor> retrieve_parent_token,
    std::optional<float> lower_bound,
    std::optional<bool> is_sglang,
    std::optional<torch::Tensor> out
);

// Defined next to the kernel entry point: does a call with this shape take the
// MTP-verify path? That is what decides which dtype dt_bias is read as.
bool k_last_takes_mtp_path(
    const torch::Tensor& k,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& initial_state_indices,
    bool disable_state_update
);

// Returns (out, initial_state), mirroring vLLM's inplace_final_state=True
// contract where the returned state *is* the in-place updated pool.
std::tuple<torch::Tensor, torch::Tensor> fused_recurrent_kda(
    torch::Tensor q,                 // [1, T, H, K]
    torch::Tensor k,                 // [1, T, H, K]
    torch::Tensor v,                 // [1, T, H, V]
    torch::Tensor raw_g,             // [1, T, H, K] raw gate projection
    torch::Tensor raw_beta,          // [1, T, H] raw beta logits
    torch::Tensor A_log,             // [H]
    std::optional<torch::Tensor> dt_bias,   // [H, K] or [H*K]
    std::optional<double> lower_bound,      // Kimi K3 safe-gate lower bound
    torch::Tensor initial_state,     // [num_slots, H, V, K] fp32, k-last, in-place
    torch::Tensor cu_seqlens,        // [N+1]
    torch::Tensor ssm_state_indices, // [N] or [N, S] (vLLM spec decode)
    std::optional<torch::Tensor> num_accepted_tokens,  // [N], with a 2D table
    std::optional<torch::Tensor> out,                  // [1, T, H, V] to fill
    std::optional<bool> fuse_gate,                     // must be true / unset
    bool is_sglang
) {
    // This wrapper normally narrows/copies gates and metadata on the device.
    // Bypass here, before preprocessing or the downstream native entry point.
    if (pla::bypass::enabled()) {
        const int batch = cu_seqlens.numel() > 0 ? cu_seqlens.numel() - 1 : 0;
        const int tokens = q.dim() >= 2 ? q.size(1) : 0;
        const int heads = q.dim() >= 3 ? q.size(2) : 0;
        const int key_dim = q.dim() >= 4 ? q.size(3) : 0;
        const int value_heads = v.dim() >= 3 ? v.size(2) : 0;
        const int value_dim = v.dim() >= 4 ? v.size(3) : 0;
        const int max_tokens = ssm_state_indices.dim() == 2
            ? ssm_state_indices.size(1) : (batch > 0 ? tokens / batch : tokens);
        if (pla::instrument::show_params()) pla::instrument::dump_case(batch, max_tokens, key_dim, value_dim, heads,
                              value_heads, get_dtype_string(q), true, &cu_seqlens,
                              static_cast<float>(lower_bound.value_or(-5.0)));
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("fused_recurrent_kda", {
            {"q", &q}, {"k", &k}, {"v", &v}, {"raw_g", &raw_g},
            {"raw_beta", &raw_beta}, {"A_log", &A_log},
            {"dt_bias", dt_bias ? &*dt_bias : nullptr},
            {"initial_state", &initial_state}, {"cu_seqlens", &cu_seqlens},
            {"ssm_state_indices", &ssm_state_indices},
            {"num_accepted_tokens", num_accepted_tokens ? &*num_accepted_tokens : nullptr},
            {"out", out ? &*out : nullptr}},
            "is_sglang=" + std::to_string(is_sglang) +
            " fuse_gate=" + std::to_string(fuse_gate.value_or(true)));
        auto result = out.has_value() ? *out
                                     : pla::bypass::output(v.sizes(), v.options());
        return std::make_tuple(result, initial_state);
    }
    sanity_recurrent_kda_inputs(q, k, v, raw_g, raw_beta, A_log, initial_state,
                                cu_seqlens, ssm_state_indices, dt_bias,
                                num_accepted_tokens, out, lower_bound);
    if (!fuse_gate.value_or(true)) {
        raise_not_implemented(
            "pla.decode.kda.fused_recurrent_kda only implements the fused-gate "
            "path (fuse_gate=True): pass the raw gate projection plus "
            "A_log/dt_bias and drop the separate _fused_kda_gate_beta call.");
    }
    if (!dt_bias.has_value()) {
        raise_value_error("dt_bias is required by the fused-gate kernel path.");
    }

    if (q.dim() != 4) {
        raise_value_error("q must be packed as [1, T, H, K], got " +
                          std::to_string(q.dim()) + "D.");
    }
    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t H = q.size(2);
    const int64_t K = q.size(3);
    const int64_t V = v.size(-1);
    if (B != 1) {
        raise_value_error("q must be packed as [1, T, H, K], got batch " +
                          std::to_string(B) + ".");
    }
    // The PPU CUDA kernel is instantiated for head_dim K == V == 128 only.
    if (K != 128 || V != 128) {
        raise_not_implemented(
            "pla.decode.kda.fused_recurrent_kda requires head_dim K == V == 128, got K=" +
            std::to_string(K) + ", V=" + std::to_string(V) + ".");
    }
    if (initial_state.scalar_type() != torch::kFloat32) {
        raise_value_error(std::string("initial_state must be float32, got ") +
                          c10::toString(initial_state.scalar_type()) + ".");
    }
    if (ssm_state_indices.dim() != 1 && ssm_state_indices.dim() != 2) {
        raise_value_error("ssm_state_indices must be 1D or 2D, got " +
                          std::to_string(ssm_state_indices.dim()) + "D");
    }

    // A 2D slot table means vLLM spec decode: the kernel writes every timestep's
    // state into its own slot, so it must not additionally commit to the start
    // slot. A 1D table commits once, at the end of the recurrence.
    const bool per_token_slots = ssm_state_indices.dim() == 2;
    if (per_token_slots && is_sglang) {
        raise_value_error(
            "a 2D ssm_state_indices is the vLLM spec-decode layout; sglang uses "
            "a 1D table plus its own intermediate-state buffer.");
    }

    // ---- reshape inputs into the k_last layout the CUDA kernel expects ------
    // The kernel treats a / b as `scalar_t` (== q dtype); A_log keeps its own
    // dtype (dispatched separately). Cast accordingly.
    const auto scalar_type = q.scalar_type();
    auto a = raw_g.reshape({B * T, H * K}).to(scalar_type).contiguous();
    auto b = raw_beta.reshape({B * T, H}).to(scalar_type).contiguous();
    auto A_log_flat = A_log.reshape({-1}).contiguous();

    auto k_contig = k.contiguous();
    auto cu_seqlens_i32 = cu_seqlens.to(torch::kInt32);
    auto state_indices = ssm_state_indices.to(torch::kInt32).contiguous();

    // Only the KDA MTP-verify kernel reads dt_bias as fp32 -- there the per-K decay
    // is recomputed for every draft token, so rounding it compounds through the
    // recurrence. The pure-decode kernel evaluates the gate once and reads dt_bias
    // at q's dtype. Both frameworks hold the parameter in fp32, so require that and
    // narrow it only for the decode path. Asking the kernel entry point which path
    // this call takes keeps the T_max rule in one place; if it ever disagreed, the
    // entry point's own dtype check would reject the call rather than misread it.
    if (dt_bias->scalar_type() != torch::kFloat32) {
        raise_value_error(std::string("dt_bias must be float32, got ") +
                          c10::toString(dt_bias->scalar_type()) + ".");
    }
    auto dt_bias_kda = dt_bias->reshape({H, K});
    if (!k_last_takes_mtp_path(k_contig, cu_seqlens_i32, state_indices,
                               /*disable_state_update=*/per_token_slots)) {
        dt_bias_kda = dt_bias_kda.to(scalar_type);
    }
    dt_bias_kda = dt_bias_kda.contiguous();

    std::optional<std::variant<torch::Tensor, int64_t>> num_accepted;
    if (num_accepted_tokens.has_value()) {
        num_accepted = num_accepted_tokens->to(torch::kInt32).contiguous();
    }

    std::optional<float> lower_bound_f;
    if (lower_bound.has_value()) {
        lower_bound_f = static_cast<float>(lower_bound.value());
    }

    torch::Tensor o = fused_sigmoid_gating_delta_rule_forward_k_last(
        A_log_flat,
        a,
        dt_bias_kda,
        kSoftplusBeta,
        kSoftplusThreshold,
        q.contiguous(),
        k_contig,
        v.contiguous(),
        b,
        initial_state,
        state_indices,
        static_cast<float>(std::pow(static_cast<double>(K), -0.5)),
        /*use_qk_l2norm_in_kernel=*/true,
        cu_seqlens_i32,
        /*is_kda=*/true,
        /*disable_state_update=*/per_token_slots,  // per-step slots already committed
        std::nullopt,   // intermediate_states_buffer (sglang-only target-verify cache)
        std::nullopt,   // intermediate_state_indices
        num_accepted,   // cache_steps_or_num_accept
        std::nullopt,   // retrieve_parent_token (sglang-only eagle tree)
        lower_bound_f,
        is_sglang,
        std::nullopt    // out: copied into the caller's buffer below
    );

    if (out.has_value()) {
        out->copy_(o);
        o = out.value();
    }

    return std::make_tuple(o, initial_state);
}
