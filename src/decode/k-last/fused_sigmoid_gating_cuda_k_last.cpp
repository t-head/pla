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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../ppu_torch_stream.h"
#include <hggc_runtime.h>
#include <cmath>
#include <variant>

#include "common.cuh"
#include "k_last_shared_layout.h"
#include "sanity_check.h"
#include "bypass_exec.h"
#include "op_forge_case.h"
#include "sanity_checks.h"
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "generated/fused_sigmoid_gating_dispatch_k_last.h"
#include "generated/tuned_dispatch.h"

#define CHECK_DTYPE(x, dataType) TORCH_CHECK(x.dtype() == dataType, #x "'s data type should be " #dataType)


// Only checks metadata and never synchronizes the device; the plain and tunable entries share one head-count contract.
static void check_k_last_heads(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& A_log, const torch::Tensor& dt_bias,
    const std::optional<torch::Tensor>& state,
    const std::optional<torch::Tensor>& cache, bool is_kda) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must have shape [B, T, heads, dim]");
    const auto H = k.size(2), HV = v.size(2), K = k.size(3), V = v.size(3);
    TORCH_CHECK(H > 0 && HV >= H && HV % H == 0,
                "HV must be a positive multiple of H, got H=", H, ", HV=", HV);
    TORCH_CHECK(q.sizes() == k.sizes() && v.size(0) == k.size(0) && v.size(1) == k.size(1),
                "q/k shapes and q/k/v token dimensions must match");
    const auto gate_width = is_kda ? HV * K : HV;
    TORCH_CHECK(a.dim() >= 2 &&
                ((a.dim() == 4 && a.size(2) == HV && a.size(3) == (is_kda ? K : 1)) ||
                 (is_kda && a.dim() == 3 && a.size(1) == HV && a.size(2) == K) ||
                 (a.dim() != 4 && a.size(-1) == gate_width)),
                "a head dimensions must match HV and the gate type");
    TORCH_CHECK(b.dim() >= 2 && b.size(-1) == HV,
                "b head dimension must match HV");
    TORCH_CHECK(A_log.numel() == HV && dt_bias.numel() == gate_width,
                "A_log/dt_bias head dimensions must match HV and the gate type");
    for (const auto* tensor : {&state, &cache}) {
        if (tensor->has_value()) {
            const auto& t = tensor->value();
            // The kernel addresses one slot as h0_idx * t.stride(0) + hv*V*K + v*K + k, so
            // besides the physical [.., HV, V, K] layout it also accepts vLLM's flattened pool rows (e.g. warmup's
            // 2D [num_slots, HV*V*K], where stride(0) is the slot pitch): it only requires the innermost dim to be contiguous
            // and one slot to hold a [HV, V, K] block; the pitch is still controlled by the caller.
            const bool physical = t.dim() >= 4 && t.size(-3) == HV
                                  && t.size(-2) == V && t.size(-1) == K;
            const bool flattened = t.dim() >= 2 && t.stride(-1) == 1
                                   && t.stride(0) >= (int64_t)HV * V * K;
            TORCH_CHECK(physical || flattened,
                        "state/cache head dimensions must match [HV, V, K], or be "
                        "a flattened pool whose stride(0) covers HV*V*K with a "
                        "unit-stride innermost dim");
        }
    }
}

// Split the shared parameter into the two things the kernel actually needs.
static void unpack_cache_steps_or_num_accept(
    const std::optional<CacheStepsOrNumAccept>& value,
    int* cache_steps_out,
    const int32_t** num_accepted_tokens_out,
    bool* has_cache_steps_out
) {
    *cache_steps_out = 0;
    *num_accepted_tokens_out = nullptr;
    *has_cache_steps_out = false;
    if (!value.has_value()) return;

    if (std::holds_alternative<int64_t>(value.value())) {
        *cache_steps_out = static_cast<int>(std::get<int64_t>(value.value()));
        *has_cache_steps_out = true;
    } else {
        const torch::Tensor& t = std::get<torch::Tensor>(value.value());
        CHECK_DTYPE(t, torch::kInt32);
        *num_accepted_tokens_out = t.data_ptr<int32_t>();
    }
}

// vLLM spec decode hands us a 2D [N, S] slot table; sglang always uses 1D.
static int state_indices_row_stride(const std::optional<torch::Tensor>& indices) {
    if (!indices.has_value() || indices.value().dim() != 2) return 0;
    return static_cast<int>(indices.value().stride(0));
}


// The initial state and per-step snapshots share one storage dtype; no temporary FP32 pool is allocated and in-place write-back semantics are unchanged.
static at::ScalarType check_state_dtype(
    const std::optional<torch::Tensor>& initial_state,
    const std::optional<torch::Tensor>& intermediate_states
) {
    const auto dtype = initial_state.has_value() ? initial_state->scalar_type()
        : intermediate_states.has_value() ? intermediate_states->scalar_type()
        : at::ScalarType::Float;
    TORCH_CHECK(dtype == at::ScalarType::Float || dtype == at::ScalarType::BFloat16,
                "k-last state dtype must be float32 or bfloat16, got ", dtype);
    TORCH_CHECK(!intermediate_states.has_value() || intermediate_states->scalar_type() == dtype,
                "initial_state_source and intermediate_states_buffer must have the same dtype");
    return dtype;
}

static void launch_k_last_impl(
    k_last_launcher_t launcher, int shared_mem_size, dim3 grid, hggcStream_t stream,
    torch::Tensor& A_log, torch::Tensor& a, torch::Tensor& dt_bias,
    float softplus_beta_or_lower_bound, float softplus_threshold,
    bool use_lower_bound,
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, torch::Tensor& b,
    torch::Tensor& o,
    void* h0_ptr, const int32_t* cu_seq_ptr,
    void* inter_buf_ptr,
    const int32_t* h0_idx_ptr, const int32_t* inter_idx_ptr, int cache_steps_val,
    const int64_t* parent_ptr, int stride_parent_seq, int stride_parent_token,
    float scale_val, int T_max,
    int stride_a, int stride_q, int stride_k, int stride_v, int stride_b,
    int64_t h0_slot_stride,
    int B, int H, int HV, int K, int V,
    bool cache_intermediate_states, bool has_eagle_tree,
    bool use_qk_l2norm_in_kernel, bool disable_state_update,
    const int32_t* num_accepted_tokens, int stride_h0_indices_seq, bool is_sglang,
    bool is_kda
) {
    launcher(
        A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(),
        softplus_beta_or_lower_bound, softplus_threshold,
        use_lower_bound,
        q.data_ptr(), k.data_ptr(), v.data_ptr(), b.data_ptr(),
        o.data_ptr(), h0_ptr, cu_seq_ptr, inter_buf_ptr,
        h0_idx_ptr, inter_idx_ptr, cache_steps_val,
        parent_ptr, stride_parent_seq, stride_parent_token,
        scale_val, T_max,
        stride_a, stride_q, stride_k, stride_v, stride_b,
        h0_slot_stride,
        B, H, HV, K, V,
        cache_intermediate_states, has_eagle_tree,
        use_qk_l2norm_in_kernel, disable_state_update,
        num_accepted_tokens, stride_h0_indices_seq, is_sglang,
        is_kda,
        shared_mem_size, grid, stream
    );
}

// dt_bias is the one gate input whose dtype is not q's: the kernel reads it as
// fp32 on the KDA MTP-verify path (where the per-K decay is recomputed per draft
// token) and as the activation dtype everywhere else. Check rather than cast -- a
// silent conversion would both hide caller bugs and add a per-call launch to a
// decode entry point.
static void check_dt_bias_dtype(
    const torch::Tensor& dt_bias, const torch::Tensor& q,
    bool is_kda, bool mtp_verify
) {
    const bool wants_fp32 = is_kda && mtp_verify;
    const auto want = wants_fp32 ? at::ScalarType::Float : q.scalar_type();
    TORCH_CHECK(dt_bias.scalar_type() == want,
                "dt_bias must be ", want, " here (is_kda=", is_kda,
                ", mtp_verify=", mtp_verify, "): only KDA MTP verify reads it as "
                "float32, every other path reads it as q's dtype. Got ",
                dt_bias.scalar_type());
}

// Whether a k_last call with this shape will take the MTP-verify path, which is
// what decides dt_bias's dtype (see check_dt_bias_dtype above). Exposed so the KDA
// wrapper in fused_recurrent_kda.cpp can hand dt_bias over in the dtype this call
// actually reads, without duplicating the T_max derivation. Assumes the caller
// passes no intermediate-state cache and no eagle tree, which is true of that
// wrapper -- its only target-verify trigger is a 2D slot table.
bool k_last_takes_mtp_path(
    const torch::Tensor& k, const torch::Tensor& cu_seqlens,
    const torch::Tensor& initial_state_indices, bool disable_state_update
) {
    const int stride_h0_indices_seq = state_indices_row_stride(initial_state_indices);
    const bool target_verify = stride_h0_indices_seq > 0;
    const int logical_slot_columns = target_verify ? initial_state_indices.size(1) : 0;
    const int B = static_cast<int>(cu_seqlens.size(0) - 1);
    const int T_max = get_packed_t_max(k, B, target_verify, /*cache_steps=*/0,
                                       logical_slot_columns);
    return T_max > 1 || disable_state_update || target_verify;
}


torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias,
    float softplus_beta, float softplus_threshold,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor b,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    std::optional<float> scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    bool disable_state_update,
    std::optional<torch::Tensor> intermediate_states_buffer,
    std::optional<torch::Tensor> intermediate_state_indices,
    std::optional<CacheStepsOrNumAccept> cache_steps_or_num_accept,
    std::optional<torch::Tensor> retrieve_parent_token,
    std::optional<float> lower_bound,
    std::optional<bool> is_sglang
) {
    if (pla::bypass::enabled()) {
        const auto* accepted = cache_steps_or_num_accept
            ? std::get_if<torch::Tensor>(&*cache_steps_or_num_accept) : nullptr;
        std::ostringstream flags;
        flags << "is_kda=" << is_kda << " is_sglang=" << is_sglang.value_or(true)
              << " disable_state_update=" << disable_state_update
              << " use_qk_l2norm=" << use_qk_l2norm_in_kernel
              << " softplus_beta=" << softplus_beta << " softplus_threshold=" << softplus_threshold;
        if (scale) flags << " scale=" << *scale;
        if (cache_steps_or_num_accept && !accepted)
            flags << " cache_steps=" << std::get<int64_t>(*cache_steps_or_num_accept);
        if (lower_bound) flags << " lower_bound=" << *lower_bound;
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("k_last", {
            {"q", &q}, {"k", &k}, {"v", &v}, {"a", &a}, {"b", &b},
            {"A_log", &A_log}, {"dt_bias", &dt_bias},
            {"state", initial_state_source ? &*initial_state_source : nullptr},
            {"slots", initial_state_indices ? &*initial_state_indices : nullptr},
            {"cu", cu_seqlens ? &*cu_seqlens : nullptr}, {"accepted", accepted},
            {"cache", intermediate_states_buffer ? &*intermediate_states_buffer : nullptr},
            {"cache_indices", intermediate_state_indices ? &*intermediate_state_indices : nullptr},
            {"parent", retrieve_parent_token ? &*retrieve_parent_token : nullptr}}, flags.str());
        if (k.dim() == 4 && v.dim() == 4) {
            const int B = cu_seqlens && cu_seqlens->dim() == 1 ? cu_seqlens->numel() - 1 : k.size(0);
            const int T = cu_seqlens && initial_state_indices && initial_state_indices->dim() == 2
                ? initial_state_indices->size(1) : k.size(1);
            if (pla::instrument::show_params()) pla::instrument::dump_case(B, T, k.size(3), v.size(3), k.size(2), v.size(2),
                get_dtype_string(q), is_kda, cu_seqlens ? &*cu_seqlens : nullptr, lower_bound.value_or(-5.0f));
        }
        return pla::bypass::output(v.sizes(), v.options());
    }
    sanity_k_last_inputs(A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, initial_state_source, initial_state_indices, scale, cu_seqlens,
        is_kda, disable_state_update, intermediate_states_buffer, intermediate_state_indices,
        cache_steps_or_num_accept, retrieve_parent_token, lower_bound, is_sglang.value_or(true));
    TORCH_CHECK(!lower_bound.has_value() || is_kda, "lower_bound is only supported for KDA");
    float softplus_beta_or_lower_bound = lower_bound.value_or(softplus_beta);

    int cache_steps_val = 0;
    const int32_t* num_accepted_ptr = nullptr;
    bool has_cache_steps = false;
    unpack_cache_steps_or_num_accept(cache_steps_or_num_accept, &cache_steps_val,
                                     &num_accepted_ptr, &has_cache_steps);

    const int stride_h0_indices_seq = state_indices_row_stride(initial_state_indices);
    const bool is_sglang_val = is_sglang.value_or(true);
    // A 2D initial_state_indices means vLLM spec decode: one state slot per draft
    // token, so the kernel walks the table instead of using an intermediate buffer.
    const bool vllm_spec = stride_h0_indices_seq > 0;
    TORCH_CHECK(!vllm_spec || !is_sglang_val,
                "2D initial_state_indices is the vLLM spec-decode layout; pass is_sglang=False");
    TORCH_CHECK(!(vllm_spec && has_cache_steps),
                "vLLM spec decode expects num_accepted_tokens (a tensor) in "
                "cache_steps_or_num_accept, but received an int");

    check_k_last_heads(q, k, v, a, b, A_log, dt_bias,
                       initial_state_source, intermediate_states_buffer, is_kda);
    const auto state_dtype = check_state_dtype(initial_state_source, intermediate_states_buffer);
    int B = cu_seqlens.has_value() ? (cu_seqlens.value().size(0) - 1) : k.size(0);
    bool target_verify = intermediate_states_buffer.has_value()
                         || intermediate_state_indices.has_value()
                         || retrieve_parent_token.has_value()
                         || has_cache_steps
                         || vllm_spec;
    int T_max = cu_seqlens.has_value()
                ? get_packed_t_max(
                      k, B, target_verify, cache_steps_val,
                      vllm_spec ? initial_state_indices->size(1) : 0)
                : k.size(1);
    int H = k.size(2);
    int K = k.size(3);
    int HV = v.size(2);
    int V = v.size(3);
    TORCH_CHECK(K == 128 && V == 128,
                "fused_sigmoid_gating_delta_rule_forward_k_last requires "
                "head_dim K == V == 128, got K=", K, ", V=", V);

    if (g_ppu_lib_perf_instrument || g_ppu_lib_show_params) {
        const torch::Tensor* dump_cu_seqlens =
            cu_seqlens.has_value() ? &cu_seqlens.value() : nullptr;
        std::string nvtx_tag = generate_nvtx_tag(
            B, T_max, K, V, H, HV, get_dtype_string(q), is_kda,
            dump_cu_seqlens, lower_bound.value_or(-5.0f));
        push_nvtx_and_print(nvtx_tag, true);
    }

    float scale_val = scale.has_value() ? static_cast<float>(scale.value()) : (1.0f / sqrtf((float)K));
    auto o = torch::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, v.options());

    // KDA accepts both [tokens, HV, K] and the flattened [tokens, HV*K].
    int stride_a = a.dim() == 4 ? a.stride(1)
        : (is_kda && a.dim() == 3 && a.size(1) == HV && a.size(2) == K)
            ? a.stride(0) : a.stride(a.dim() - 2);
    int stride_q = q.dim() == 4 ? q.stride(1) : q.stride(0);
    int stride_k = k.dim() == 4 ? k.stride(1) : k.stride(0);
    int stride_v = v.dim() == 4 ? v.stride(1) : v.stride(0);
    int stride_b = b.stride(b.dim() - 2);

    bool mtp_verify = T_max > 1 || disable_state_update || target_verify;
    check_dt_bias_dtype(dt_bias, q, is_kda, mtp_verify);

    // The MTP tables are split by chain/tree mode: an eagle-tree call always
    // carries retrieve_parent_token while a chain call never does. The host
    // cannot cheaply inspect the device-side parent values, so has_value is
    // the mode signal -- a framework that passes linear parents for a chain
    // conservatively lands on the tree table (correctness is unaffected).
    TunedParams tp = mtp_verify ? get_tuned_k_last_mtp(B, H, HV, K, V, retrieve_parent_token.has_value())
                                : get_tuned_k_last(B, H, HV, K, V);

    int hv_per_h = HV / H;
    TORCH_CHECK(tp.bv > 0 && tp.wn > 0 && tp.sn > 0 && V % tp.bv == 0,
                "invalid k-last tile parameters");
    TORCH_CHECK(tp.hv_per_cta > 0 && tp.hv_per_cta <= hv_per_h && hv_per_h % tp.hv_per_cta == 0,
                "HV_PER_CTA must divide HV/H");
    int NV = V / tp.bv;
    dim3 grid(NV, hv_per_h / tp.hv_per_cta, B * H);

    const int32_t* cu_seq_ptr = nullptr;
    const int32_t* h0_idx_ptr = nullptr;
    const int32_t* inter_idx_ptr = nullptr;
    const int64_t* parent_ptr = nullptr;
    torch::Tensor parent_i64;  // keep upconverted int64 alive until launch
    int stride_parent_seq = 0, stride_parent_token = 0;

    if (cu_seqlens.has_value()) { CHECK_DTYPE(cu_seqlens.value(), torch::kInt32); cu_seq_ptr = cu_seqlens.value().data_ptr<int32_t>(); }
    if (initial_state_indices.has_value()) { CHECK_DTYPE(initial_state_indices.value(), torch::kInt32); h0_idx_ptr = initial_state_indices.value().data_ptr<int32_t>(); }
    if (intermediate_state_indices.has_value()) { CHECK_DTYPE(intermediate_state_indices.value(), torch::kInt32); inter_idx_ptr = intermediate_state_indices.value().data_ptr<int32_t>(); }
    if (retrieve_parent_token.has_value()) {
        auto pt = retrieve_parent_token.value();
        TORCH_CHECK(pt.scalar_type() == torch::kInt64 || pt.scalar_type() == torch::kInt32,
                    "retrieve_parent_token dtype should be int32 or int64");
        // sglang CUDA-graph path allocates this buffer as int32; upconvert so the
        // kernel (which reads int64) works without a framework-side change.
        parent_i64 = (pt.scalar_type() == torch::kInt64) ? pt : pt.to(torch::kInt64);
        parent_ptr = parent_i64.data_ptr<int64_t>();
        stride_parent_seq = parent_i64.stride(0);
        stride_parent_token = parent_i64.stride(1);
    }

    const pla::KLastSharedLayout shared_layout(tp.sn, K, tp.wn, tp.bv, tp.hv_per_cta,
        is_kda, mtp_verify, T_max);
    TORCH_CHECK(shared_layout.total_bytes <= std::numeric_limits<int>::max(), "shared allocation exceeds int range");
    const int shared_mem_size = static_cast<int>(shared_layout.total_bytes);

    k_last_launcher_t launcher = get_k_last_launcher_dispatch(
        tp.bv, tp.wn, tp.sn, tp.hv_per_cta,
        q.scalar_type(), A_log.scalar_type(), mtp_verify, state_dtype);
    TORCH_CHECK(launcher != nullptr, "No kernel instantiation for BV=", tp.bv, " WN=", tp.wn, " SN=", tp.sn, " HV_PER_CTA=", tp.hv_per_cta);

    // Request-dim stride must be the buffer's ALLOCATED step dim, not the runtime
    // draft_token_num passed as cache_steps (they differ under --speculative-adaptive:
    // buffer is [req, maxT, HV, K, V] but cache_steps=runtime T). Derive from the
    // buffer's explicit step dim (ndim>=5) to stay aligned with sglang's commit.
    int cache_steps_eff = (intermediate_states_buffer.has_value() && intermediate_states_buffer.value().dim() >= 5)
        ? static_cast<int>(intermediate_states_buffer.value().size(1))
        : cache_steps_val;

    launch_k_last_impl(launcher, shared_mem_size, grid, pla::current_stream(),
        A_log, a, dt_bias, softplus_beta_or_lower_bound, softplus_threshold,
        lower_bound.has_value(),
        q, k, v, b, o,
        initial_state_source.has_value() ? initial_state_source.value().data_ptr() : nullptr,
        cu_seq_ptr,
        intermediate_states_buffer.has_value() ? intermediate_states_buffer.value().data_ptr() : nullptr,
        h0_idx_ptr, inter_idx_ptr, cache_steps_eff,
        parent_ptr, stride_parent_seq, stride_parent_token,
        scale_val, T_max, stride_a, stride_q, stride_k, stride_v, stride_b,
        // vLLM packs conv_state and ssm_state into one page, so the slot
        // stride of its ssm_state view exceeds HV*V*K; SGLang is contiguous.
        initial_state_source.has_value() ? initial_state_source.value().stride(0) : 0,
        B, H, HV, K, V,
        intermediate_states_buffer.has_value(), retrieve_parent_token.has_value(),
        use_qk_l2norm_in_kernel, disable_state_update,
        num_accepted_ptr, stride_h0_indices_seq, is_sglang_val,
        is_kda
    );
    return o;
}

torch::Tensor fused_sigmoid_gating_delta_rule_forward_k_last_tunable(
    torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias,
    float softplus_beta, float softplus_threshold,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor b,
    std::optional<torch::Tensor> initial_state_source,
    std::optional<torch::Tensor> initial_state_indices,
    std::optional<float> scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> cu_seqlens,
    bool is_kda,
    bool disable_state_update,
    std::optional<torch::Tensor> intermediate_states_buffer,
    std::optional<torch::Tensor> intermediate_state_indices,
    std::optional<CacheStepsOrNumAccept> cache_steps_or_num_accept,
    std::optional<torch::Tensor> retrieve_parent_token,
    int bv, int wn, int sn, int hv_per_cta,
    std::optional<bool> is_sglang,
    std::optional<float> lower_bound
) {
    if (pla::bypass::enabled()) {
        const auto* accepted = cache_steps_or_num_accept
            ? std::get_if<torch::Tensor>(&*cache_steps_or_num_accept) : nullptr;
        std::ostringstream flags;
        flags << "is_kda=" << is_kda << " is_sglang=" << is_sglang.value_or(true)
              << " disable_state_update=" << disable_state_update
              << " use_qk_l2norm=" << use_qk_l2norm_in_kernel
              << " softplus_beta=" << softplus_beta << " softplus_threshold=" << softplus_threshold;
        if (scale) flags << " scale=" << *scale;
        if (cache_steps_or_num_accept && !accepted)
            flags << " cache_steps=" << std::get<int64_t>(*cache_steps_or_num_accept);
        flags << " bv=" << bv << " wn=" << wn << " sn=" << sn << " hv_per_cta=" << hv_per_cta;
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("k_last_tunable", {
            {"q", &q}, {"k", &k}, {"v", &v}, {"a", &a}, {"b", &b},
            {"A_log", &A_log}, {"dt_bias", &dt_bias},
            {"state", initial_state_source ? &*initial_state_source : nullptr},
            {"slots", initial_state_indices ? &*initial_state_indices : nullptr},
            {"cu", cu_seqlens ? &*cu_seqlens : nullptr}, {"accepted", accepted},
            {"cache", intermediate_states_buffer ? &*intermediate_states_buffer : nullptr},
            {"cache_indices", intermediate_state_indices ? &*intermediate_state_indices : nullptr},
            {"parent", retrieve_parent_token ? &*retrieve_parent_token : nullptr}}, flags.str());
        if (k.dim() == 4 && v.dim() == 4) {
            const int B = cu_seqlens && cu_seqlens->dim() == 1 ? cu_seqlens->numel() - 1 : k.size(0);
            const int T = cu_seqlens && initial_state_indices && initial_state_indices->dim() == 2
                ? initial_state_indices->size(1) : k.size(1);
            if (pla::instrument::show_params()) pla::instrument::dump_case(B, T, k.size(3), v.size(3), k.size(2), v.size(2),
                get_dtype_string(q), is_kda, cu_seqlens ? &*cu_seqlens : nullptr, -5.0f);
        }
        return pla::bypass::output(v.sizes(), v.options());
    }
    sanity_k_last_inputs(A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, initial_state_source, initial_state_indices, scale, cu_seqlens,
        is_kda, disable_state_update, intermediate_states_buffer, intermediate_state_indices,
        cache_steps_or_num_accept, retrieve_parent_token, std::nullopt, is_sglang.value_or(true));
    TORCH_CHECK(!lower_bound.has_value() || is_kda, "lower_bound is only supported for KDA");
    int cache_steps_val = 0;
    const int32_t* num_accepted_ptr = nullptr;
    bool has_cache_steps = false;
    unpack_cache_steps_or_num_accept(cache_steps_or_num_accept, &cache_steps_val,
                                     &num_accepted_ptr, &has_cache_steps);

    const int stride_h0_indices_seq = state_indices_row_stride(initial_state_indices);
    const bool is_sglang_val = is_sglang.value_or(true);
    const bool vllm_spec = stride_h0_indices_seq > 0;
    TORCH_CHECK(!vllm_spec || !is_sglang_val,
                "2D initial_state_indices is the vLLM spec-decode layout; pass is_sglang=False");
    TORCH_CHECK(!(vllm_spec && has_cache_steps),
                "vLLM spec decode expects num_accepted_tokens rather than integer cache_steps");

    check_k_last_heads(q, k, v, a, b, A_log, dt_bias,
                       initial_state_source, intermediate_states_buffer, is_kda);
    const auto state_dtype = check_state_dtype(initial_state_source, intermediate_states_buffer);
    int B = cu_seqlens.has_value() ? (cu_seqlens.value().size(0) - 1) : k.size(0);
    bool target_verify = intermediate_states_buffer.has_value()
                         || intermediate_state_indices.has_value()
                         || retrieve_parent_token.has_value()
                         || has_cache_steps
                         || vllm_spec;
    int T_max = cu_seqlens.has_value()
                ? get_packed_t_max(
                      k, B, target_verify, cache_steps_val,
                      vllm_spec ? initial_state_indices->size(1) : 0)
                : k.size(1);
    int H = k.size(2);
    int K = k.size(3);
    int HV = v.size(2);
    int V = v.size(3);
    TORCH_CHECK(K == 128 && V == 128,
                "fused_sigmoid_gating_delta_rule_forward_k_last_tunable requires "
                "head_dim K == V == 128, got K=", K, ", V=", V);

    float scale_val = scale.has_value() ? static_cast<float>(scale.value()) : (1.0f / sqrtf((float)K));
    auto o = torch::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, v.options());

    // KDA accepts both [tokens, HV, K] and the flattened [tokens, HV*K].
    int stride_a = a.dim() == 4 ? a.stride(1)
        : (is_kda && a.dim() == 3 && a.size(1) == HV && a.size(2) == K)
            ? a.stride(0) : a.stride(a.dim() - 2);
    int stride_q = q.dim() == 4 ? q.stride(1) : q.stride(0);
    int stride_k = k.dim() == 4 ? k.stride(1) : k.stride(0);
    int stride_v = v.dim() == 4 ? v.stride(1) : v.stride(0);
    int stride_b = b.stride(b.dim() - 2);

    bool mtp_verify = T_max > 1 || disable_state_update || target_verify;
    check_dt_bias_dtype(dt_bias, q, is_kda, mtp_verify);

    int hv_per_h = HV / H;
    TORCH_CHECK(bv > 0 && wn > 0 && sn > 0 && V % bv == 0,
                "invalid k-last tile parameters");
    TORCH_CHECK(hv_per_cta > 0 && hv_per_cta <= hv_per_h && hv_per_h % hv_per_cta == 0,
                "HV_PER_CTA must divide HV/H");
    int NV = V / bv;
    dim3 grid(NV, hv_per_h / hv_per_cta, B * H);

    const int32_t* cu_seq_ptr = nullptr;
    const int32_t* h0_idx_ptr = nullptr;
    const int32_t* inter_idx_ptr = nullptr;
    const int64_t* parent_ptr = nullptr;
    torch::Tensor parent_i64;  // keep upconverted int64 alive until launch
    int stride_parent_seq = 0, stride_parent_token = 0;

    if (cu_seqlens.has_value()) { CHECK_DTYPE(cu_seqlens.value(), torch::kInt32); cu_seq_ptr = cu_seqlens.value().data_ptr<int32_t>(); }
    if (initial_state_indices.has_value()) { CHECK_DTYPE(initial_state_indices.value(), torch::kInt32); h0_idx_ptr = initial_state_indices.value().data_ptr<int32_t>(); }
    if (intermediate_state_indices.has_value()) { CHECK_DTYPE(intermediate_state_indices.value(), torch::kInt32); inter_idx_ptr = intermediate_state_indices.value().data_ptr<int32_t>(); }
    if (retrieve_parent_token.has_value()) {
        auto pt = retrieve_parent_token.value();
        TORCH_CHECK(pt.scalar_type() == torch::kInt64 || pt.scalar_type() == torch::kInt32,
                    "retrieve_parent_token dtype should be int32 or int64");
        // sglang CUDA-graph path allocates this buffer as int32; upconvert so the
        // kernel (which reads int64) works without a framework-side change.
        parent_i64 = (pt.scalar_type() == torch::kInt64) ? pt : pt.to(torch::kInt64);
        parent_ptr = parent_i64.data_ptr<int64_t>();
        stride_parent_seq = parent_i64.stride(0);
        stride_parent_token = parent_i64.stride(1);
    }

    const pla::KLastSharedLayout shared_layout(sn, K, wn, bv, hv_per_cta,
        is_kda, mtp_verify, T_max);
    TORCH_CHECK(shared_layout.total_bytes <= std::numeric_limits<int>::max(), "shared allocation exceeds int range");
    const int shared_mem_size = static_cast<int>(shared_layout.total_bytes);

    k_last_launcher_t launcher = get_k_last_launcher_dispatch(
        bv, wn, sn, hv_per_cta, q.scalar_type(), A_log.scalar_type(), mtp_verify, state_dtype);
    TORCH_CHECK(launcher != nullptr, "No kernel instantiation for BV=", bv, " WN=", wn, " SN=", sn, " HV_PER_CTA=", hv_per_cta);

    // Request-dim stride must be the buffer's ALLOCATED step dim, not the runtime
    // draft_token_num passed as cache_steps (they differ under --speculative-adaptive:
    // buffer is [req, maxT, HV, K, V] but cache_steps=runtime T). Derive from the
    // buffer's explicit step dim (ndim>=5) to stay aligned with sglang's commit.
    int cache_steps_eff = (intermediate_states_buffer.has_value() && intermediate_states_buffer.value().dim() >= 5)
        ? static_cast<int>(intermediate_states_buffer.value().size(1))
        : cache_steps_val;

    launch_k_last_impl(launcher, shared_mem_size, grid, pla::current_stream(),
        A_log, a, dt_bias, lower_bound.value_or(softplus_beta), softplus_threshold,
        lower_bound.has_value(),
        q, k, v, b, o,
        initial_state_source.has_value() ? initial_state_source.value().data_ptr() : nullptr,
        cu_seq_ptr,
        intermediate_states_buffer.has_value() ? intermediate_states_buffer.value().data_ptr() : nullptr,
        h0_idx_ptr, inter_idx_ptr, cache_steps_eff,
        parent_ptr, stride_parent_seq, stride_parent_token,
        scale_val, T_max, stride_a, stride_q, stride_k, stride_v, stride_b,
        // vLLM packs conv_state and ssm_state into one page, so the slot
        // stride of its ssm_state view exceeds HV*V*K; SGLang is contiguous.
        initial_state_source.has_value() ? initial_state_source.value().stride(0) : 0,
        B, H, HV, K, V,
        intermediate_states_buffer.has_value(), retrieve_parent_token.has_value(),
        use_qk_l2norm_in_kernel, disable_state_update,
        num_accepted_ptr, stride_h0_indices_seq, is_sglang_val,
        is_kda
    );
    return o;
}
