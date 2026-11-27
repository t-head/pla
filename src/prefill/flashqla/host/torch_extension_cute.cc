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
 * FlashQLA PPU1.5 CuTe — PyTorch extension binding for pla.prefill.flashqla.
 *
 * Compiled into pla/prefill/flashqla/_C. The Python facades split these ops
 * across two namespaces:
 *
 * All six are exposed under pla.prefill.flashqla.private; the public
 * pla.prefill.flashqla namespace only carries the end-to-end pipeline
 * chunk_gated_delta_rule_fwd (pure Python, in pla/prefill/flashqla/ops.py).
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <set>
#include "../../../ppu_torch_stream.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <mutex>

#include "op_forge_case.h"
#include "../../sanity_check.h"
#include "../../../bypass_exec.h"
#include "../../../op_forge_case.h"
#include "../../instrument.h"
#include "sanity_checks.h"
#include "precision_repair_api.h"
#include "packed_coefficients_api.h"
#include "prefill/flashqla/common/prepare_policy.h"
#include "prefill/flashqla/common/graph_payload_owner.h"

// ---- forward decls from kernels/*.cu ----
void launch_get_warmup_chunks_cute(
    const float* g, const signed char* ht_mask, const int* cu_seqlens,
    int* num_warmup_chunks, signed char* fallback_mask,
    int batch_size, int num_tokens, int H, float threshold,
    hggcStream_t stream);

void launch_kkt_solve_cute(
    const void* K, const float* beta, void* A, const int* cu_seqlens,
    const int* chunk_indices,
    int H, int Hg, int num_chunks, int num_tokens, int dtype_tag,
    int device_index, int num_cu, int compute_capability, int sequences,
    hggcStream_t stream);

void launch_correct_h0_no_raw_cute(
    void* cp_h0, const void* ht_buffer, const void* mt_buffer,
    const signed char* fallback_mask, const int* seq_map_r2c,
    int cp_batch_size, int raw_batch_size, int H, bool cp_h0_is_bf16,
    int dtype_tag, int device_index, int num_cu, int compute_capability,
    const int* precision_flags, int key_heads, hggcStream_t stream);

void launch_correct_h0_raw_cute(
    void* cp_h0, const void* raw_h0, const void* ht_buffer,
    const void* mt_buffer, const signed char* fallback_mask,
    const int* seq_map_r2c, const int* raw_h0_indices,
    int cp_batch_size, int raw_batch_size, int H, int raw_h0_k_last,
    bool raw_h0_is_bf16, bool cp_h0_is_bf16,
    int dtype_tag, int device_index, int num_cu, int compute_capability,
    const int* precision_flags, int key_heads, hggcStream_t stream);

void launch_prepare_h_cp_cute(
    const void* k, const void* v, const void* a,
    const float* g, const float* beta, void* ht, void* mt,
    const int* cu_seqlens, const int* num_warmup_chunks,
    int batch_size, int num_tokens, int H, int Hg, int dtype_tag,
    int64_t v_token_stride,
    int device_index, int num_cu, int compute_capability,
    void* precomputed_x, const int* precision_flags, const int* cp_to_raw,
    const int* raw_to_cp, const int* raw_cu_seqlens, int raw_batch_size,
    hggcStream_t stream);

void launch_fused_fwd_cp_cute(
    void* o, const void* q, const void* k, const void* v,
    const void* a, const float* beta, const float* g,
    const void* h0, void* ht_out, void* raw_state,
    const void* raw_state_input, bool state_bf16,
    const int* raw_state_indices,
    const int* cu_seqlens, const int* cp_seq_map, const int* raw_cu_seqlens,
    const int* num_warmup_chunks, const signed char* fallback_mask,
    int batch_size, int num_tokens, int raw_batch_size,
    int H, int Hg, float scale, int dtype_tag,
    int64_t v_token_stride, const int* precision_flags,
    int device_index, int num_cu, int compute_capability,
    hggcStream_t stream);

void launch_chunk_local_cumsum_cute(
    const float* g, const int* cu_seqlens, const int* chunk_indices,
    float* g_cum, int num_chunks, int H, hggcStream_t stream);

void launch_chunk_local_cumsum_qk_norm_cute(
    const float* g, const int* cu_seqlens, const int* chunk_indices,
    float* g_cum, const void* q, const void* k, void* q_norm, void* k_norm,
    int num_chunks, int T, int H, int Hg, int dtype_tag,
    hggcStream_t stream, bool stage_g);

void launch_chunk_local_cumsum_qk_norm_gated_cute(
    const void* a, const void* b, const float* A_log, const float* dt_bias,
    const int* cu_seqlens, const int* chunk_indices,
    float* g_cum, float* beta,
    const void* q, const void* k, void* q_norm, void* k_norm,
    int num_chunks, int T, int H, int Hg,
    int64_t stride_a, int64_t stride_b,
    int64_t stride_q, int64_t stride_k, int dtype_tag,
    hggcStream_t stream);

// Folded kernels use the same extension, stream and tensor ownership as the
// established stages. Coefficients are an optional internal workspace.
void launch_folded_coefficients_cute(const void*, const void*, void*, const float*,
    const float*, void*, const int*, const int*, int, int, int, int, float, bool, hggcStream_t);
void launch_folded_prepare_h_cute(const void*, const void*, const void*, const int*,
    const int*, void*, void*, int, int, int, int, int64_t, bool, hggcStream_t,
    const float*, const int*, float*);
void launch_folded_fwd_cute(void*, const void*, const void*, const void*, const void*,
    const int*, const int*, const int*, const float*, float*, const int*, int, int, int,
    int, int, int, int64_t, float, const int*, hggcStream_t);
void launch_gdn_folded_gated_scan(const void*, const void*, const float*, const float*,
    const int*, const int*, float*, float*, const void*, const void*, void*, void*, int,
    int, int, int, int64_t, int64_t, int64_t, int64_t, hggcStream_t);
void launch_gdn_correct_native(void*, const void*, const void*, const void*,
    const signed char*, const int*, const int*, int, int, int, int, hggcStream_t, bool);
void launch_folded_correct_fp32(void*, const void*, const void*, const void*,
    const signed char*, const int*, const int*, int, int, hggcStream_t, bool);

namespace {

constexpr int kMaxCachedDevices = 64;

struct LogicalDeviceInfo {
    int index;
    int num_cu;
    int compute_capability;
};

LogicalDeviceInfo logical_device_info(torch::Tensor const& tensor) {
    const int device = tensor.get_device();
    auto query_device_info = [device]() {
        int num_cu = 0;
        int major = 0;
        int minor = 0;
        const hggcError_t cu_status = hggcDeviceGetAttribute(
            &num_cu, hggcDevAttrMultiProcessorCount, device);
        TORCH_CHECK(
            cu_status == hggcSuccess,
            "failed to query multiprocessor count for PPU device ", device,
            ": ", hggcGetErrorString(cu_status));
        TORCH_CHECK(
            num_cu > 0,
            "PPU device ", device,
            " reported invalid multiprocessor count ", num_cu);

        const hggcError_t major_status = hggcDeviceGetAttribute(
            &major, hggcDevAttrComputeCapabilityMajor, device);
        TORCH_CHECK(
            major_status == hggcSuccess,
            "failed to query compute capability major for PPU device ",
            device, ": ", hggcGetErrorString(major_status));
        const hggcError_t minor_status = hggcDeviceGetAttribute(
            &minor, hggcDevAttrComputeCapabilityMinor, device);
        TORCH_CHECK(
            minor_status == hggcSuccess,
            "failed to query compute capability minor for PPU device ",
            device, ": ", hggcGetErrorString(minor_status));

        const int compute_capability = major * 10 + minor;
        TORCH_CHECK(
            compute_capability == 80 || compute_capability == 89,
            "unsupported PPU compute capability ", major, ".", minor,
            " on PPU device ", device,
            "; expected 8.0 (tc01) or 8.9 (tc02)");
        return LogicalDeviceInfo{device, num_cu, compute_capability};
    };
    if (device < 0 || device >= kMaxCachedDevices) {
        return query_device_info();
    }
    // Device properties are stable for the lifetime of a logical device. The
    // once_flag both removes the launch-path runtime query and avoids a race
    // when multiple host threads first use the same device concurrently.
    static std::once_flag configured[kMaxCachedDevices];
    static LogicalDeviceInfo cached_info[kMaxCachedDevices] = {};
    std::call_once(
        configured[device],
        [&]() { cached_info[device] = query_device_info(); });
    return cached_info[device];
}

// Pure C++ metadata ownership. Graph user objects hold this whole entry so
// both the Tensor storage and the cross-stream readiness event outlive replay.
struct ChunkMetadataEntry {
    torch::Tensor indices;  // Primary metadata tensor establishes device provenance.
    std::vector<torch::Tensor> metadata;
    std::shared_ptr<at::cuda::CUDAEvent> ready;
    std::shared_ptr<std::atomic<bool>> ready_complete = std::make_shared<std::atomic<bool>>(false);
    std::set<std::pair<int, int64_t>> active_streams;
    std::mutex mutex;
};

void mark_metadata_ready(void* data) {
    // This callback runs on the upload stream after the readiness event.
    // Its payload owns no tensors or runtime resources.
    std::unique_ptr<std::shared_ptr<std::atomic<bool>>> ready(
        static_cast<std::shared_ptr<std::atomic<bool>>*>(data));
    (*ready)->store(true, std::memory_order_release);
}

std::shared_ptr<ChunkMetadataEntry> make_metadata_bundle_entry(std::vector<torch::Tensor> metadata) {
    TORCH_CHECK(!metadata.empty(), "metadata bundle cannot be empty");
    const auto indices = metadata.front();
    for (const auto& tensor : metadata)
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == indices.device() && tensor.is_contiguous() &&
                    (tensor.scalar_type() == torch::kInt || tensor.scalar_type() == torch::kLong ||
                     tensor.scalar_type() == torch::kChar),
                    "metadata bundle requires contiguous same-device int8/int32/int64 tensors");
    const c10::cuda::CUDAGuard guard(indices.device());
    const auto device = logical_device_info(indices);
    TORCH_CHECK(device.compute_capability == 89 && device.num_cu == 39,
                "private metadata owner requires current tc02 / 39 CU");
    const auto stream = c10::cuda::getCurrentCUDAStream(indices.get_device());
    hggcStreamCaptureStatus status{};
    unsigned long long capture_id = 0;
    hggcGraph_t graph{};
    TORCH_CHECK(hggcStreamGetCaptureInfo(reinterpret_cast<hggcStream_t>(stream.stream()),
                    &status, &capture_id, &graph, nullptr, nullptr, nullptr) == hggcSuccess &&
                status == hggcStreamCaptureStatusNone,
                "chunk metadata cold creation must be warmed before capture");
    auto entry = std::make_shared<ChunkMetadataEntry>();
    entry->indices = indices;
    entry->metadata = std::move(metadata);
    entry->ready = std::make_shared<at::cuda::CUDAEvent>();
    entry->ready->record(stream);
    auto notification = std::make_unique<std::shared_ptr<std::atomic<bool>>>(entry->ready_complete);
    const auto queued = hggcLaunchHostFunc(reinterpret_cast<hggcStream_t>(stream.stream()),
                                           mark_metadata_ready, notification.get());
    TORCH_CHECK(queued == hggcSuccess, "failed to enqueue metadata readiness: ",
                hggcGetErrorString(queued));
    notification.release();
    for (const auto& tensor : entry->metadata)
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    entry->active_streams.insert({int(stream.device_index()), int64_t(stream.id())});
    return entry;
}

void prepare_chunk_metadata_for_capture(const std::shared_ptr<ChunkMetadataEntry>& entry) {
    if (entry->ready_complete->load(std::memory_order_acquire)) return;
    const c10::cuda::CUDAGuard guard(entry->indices.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(entry->indices.get_device());
    hggcStreamCaptureStatus status{};
    TORCH_CHECK(hggcStreamIsCapturing(reinterpret_cast<hggcStream_t>(stream.stream()),
                                     &status) == hggcSuccess &&
                status == hggcStreamCaptureStatusNone,
                "prepare metadata before entering Graph capture");
    entry->ready->synchronize();
    entry->ready_complete->store(true, std::memory_order_release);
}

std::shared_ptr<ChunkMetadataEntry> make_chunk_metadata_entry(torch::Tensor indices) {
    TORCH_CHECK(indices.scalar_type() == torch::kInt,
                "chunk metadata must be contiguous device int32");
    return make_metadata_bundle_entry({std::move(indices)});
}

void activate_chunk_metadata_entry(const std::shared_ptr<ChunkMetadataEntry>& entry) {
    const c10::cuda::CUDAGuard guard(entry->indices.device());
    // Raw handle and stream device are obtained from this same typed object.
    const auto stream = c10::cuda::getCurrentCUDAStream(entry->indices.get_device());
    const std::pair<int, int64_t> key{int(stream.device_index()), int64_t(stream.id())};
    hggcStreamCaptureStatus status{};
    unsigned long long capture_id = 0;
    hggcGraph_t graph{};
    TORCH_CHECK(hggcStreamGetCaptureInfo(reinterpret_cast<hggcStream_t>(stream.stream()),
                    &status, &capture_id, &graph, nullptr, nullptr, nullptr) == hggcSuccess,
                "failed to query chunk metadata activation capture status");
    TORCH_CHECK(status == hggcStreamCaptureStatusNone || status == hggcStreamCaptureStatusActive,
                "chunk metadata activation requires a valid stream/capture");
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (status == hggcStreamCaptureStatusActive) {
            // The eager factory completed the one-time upload before caching.
            // No query/wait/record or permanent stream activation during capture.
            TORCH_CHECK(entry->ready_complete->load(std::memory_order_acquire),
                        "prefill metadata upload is pending; prepare it before capture");
        } else if (!entry->active_streams.count(key)) {
            entry->ready->block(stream);
            for (const auto& tensor : entry->metadata)
                c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
            entry->active_streams.insert(key);
        }
    }
    // Even a previously activated stream must attach a fresh graph reference.
    pla_graph_owner_experiment::owner().retain_if_capturing(
        reinterpret_cast<hggcStream_t>(stream.stream()), int(stream.device_index()),
        entry->indices.get_device(), entry);
}

pybind11::dict chunk_metadata_owner_stats() {
    const auto s = pla_graph_owner_experiment::owner().stats();
    pybind11::dict result;
    result["created"] = s.created; result["attached"] = s.attached;
    result["callbacks"] = s.callbacks; result["released"] = s.released;
    result["create_errors"] = s.create_errors; result["attach_errors"] = s.attach_errors;
    result["release_errors"] = s.release_errors; result["device_errors"] = s.device_errors;
    result["shutdown"] = s.shutdown;
    return result;
}

int dtype_tag(torch::Tensor const& x) {
    if (x.dtype() == torch::kFloat16) return 0;
    if (x.dtype() == torch::kBFloat16) return 1;
    TORCH_CHECK(false, "tensor must be fp16 or bf16");
}

void* data_ptr_b16(torch::Tensor& x) {
    return x.data_ptr();
}

const void* data_ptr_b16(torch::Tensor const& x) {
    return x.data_ptr();
}

void check_same_b16_dtype(torch::Tensor const& ref, torch::Tensor const& x, const char* name) {
    TORCH_CHECK(x.dtype() == ref.dtype(), name, " must have same dtype as reference");
}

void check_same_cuda_device(torch::Tensor const& ref, torch::Tensor const& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(x.device() == ref.device(), name, " must be on ", ref.device(),
                ", got ", x.device());
}

bool is_supported_head_config(int64_t H, int64_t Hg) {
    return (H == 128 && Hg == 16) ||
           (H == 64 && (Hg == 16 || Hg == 8)) ||
           (H == 48 && Hg == 16) ||
           (H == 32 && (Hg == 16 || Hg == 8 || Hg == 4)) ||
           (H == 24 && Hg == 8) ||
           (H == 16 && (Hg == 16 || Hg == 8 || Hg == 4 || Hg == 2)) ||
           (H == 12 && Hg == 4) ||
           (H == 8 && (Hg == 4 || Hg == 2 || Hg == 1)) ||
           (H == 6 && Hg == 2) ||
           (H == 4 && Hg == 2);
}

namespace sp = pla::sanity::prefill;

// chunk_local_cumsum: forward chunk-local cumsum of the gate. Returns g_cum
// [1, T, H] fp32. Replaces the host-side Python per-chunk loop.
torch::Tensor chunk_local_cumsum(
    torch::Tensor g,           // [1, T, H] fp32 (raw log-gate)
    torch::Tensor cu_seqlens,  // [batch+1] int32
    torch::Tensor chunk_indices // [num_chunks,2] int32
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.chunk_local_cumsum",
            {{"g", &g}, {"cu_seqlens", &cu_seqlens}, {"chunk_indices", &chunk_indices}});
        return pla::bypass::output(g.sizes(), g.options());
    }
    const pla::sanity::Context sanity("flashqla.chunk_local_cumsum", g);
    if (sanity) {
        sanity.check(g.dim() == 3, "g must be [1,T,H]");
        sp::int_extent(sanity, g.size(1), "T");
        sp::int_extent(sanity, g.size(2), "H");
        sanity_gate(sanity, g, "g", g.size(1), g.size(2));
        const auto cu = sp::offsets(sanity, cu_seqlens, g.size(1));
        sp::chunks(sanity, chunk_indices, cu);
    }
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(g.dim() == 3, "g must be [1,T,H]");
    TORCH_CHECK(g.stride(2) == 1, "g must be contiguous in H");
    check_same_cuda_device(g, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(g, chunk_indices, "chunk_indices");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.dtype() == torch::kInt32,
                "chunk_indices must be int32");
    TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2,
                "chunk_indices must be [num_chunks,2]");
    TORCH_CHECK(chunk_indices.is_contiguous(), "chunk_indices must be contiguous");
    at::cuda::CUDAGuard device_guard(g.device());

    int H = g.size(2);
    int num_chunks = chunk_indices.size(0);

    auto g_cum = torch::empty_like(g);

    const hggcStream_t stream = pla::current_stream();
    launch_chunk_local_cumsum_cute(
        g.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
        chunk_indices.data_ptr<int>(), g_cum.data_ptr<float>(),
        num_chunks, H, stream);

    return g_cum;
}

std::vector<torch::Tensor> chunk_local_cumsum_qk_norm(
    torch::Tensor g,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices,
    torch::Tensor q,
    torch::Tensor k
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.chunk_local_cumsum_qk_norm",
            {{"g", &g}, {"q", &q}, {"k", &k}, {"cu_seqlens", &cu_seqlens},
             {"chunk_indices", &chunk_indices}});
        return {pla::bypass::output(g.sizes(), g.options()),
                pla::bypass::output(q.sizes(), q.options()),
                pla::bypass::output(k.sizes(), k.options())};
    }
    const pla::sanity::Context sanity("flashqla.chunk_local_cumsum_qk_norm", g);
    if (sanity) {
        sanity.check(g.dim() == 3 && q.dim() == 4, "g/q rank mismatch");
        sp::int_extent(sanity, g.size(1), "T");
        sp::int_extent(sanity, g.size(2), "H");
        sp::int_extent(sanity, q.size(2), "Hg");
        sanity_gate(sanity, g, "g", g.size(1), g.size(2));
        sanity_feature(sanity, q, "q", g.size(1), q.size(2), 128);
        sanity_feature(sanity, k, "k", g.size(1), q.size(2), 128);
        const auto cu = sp::offsets(sanity, cu_seqlens, g.size(1));
        sp::chunks(sanity, chunk_indices, cu);
    }
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(g.dim() == 3 && g.stride(2) == 1,
                "g must be [1,T,H] and contiguous in H");
    check_same_cuda_device(g, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(g, chunk_indices, "chunk_indices");
    check_same_cuda_device(g, q, "q");
    check_same_cuda_device(g, k, "k");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32,
                "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.dtype() == torch::kInt32,
                "chunk_indices must be int32");
    TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                chunk_indices.is_contiguous(),
                "chunk_indices must be contiguous [num_chunks,2]");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.sizes() == k.sizes(),
                "q and k must have the same [1,T,Hg,D] shape");
    TORCH_CHECK(q.size(0) == 1 && q.size(1) == g.size(1) && q.size(3) == 128,
                "q and k must be [1,T,Hg,128]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),
                "q and k must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q and k must be fp16 or bf16");
    check_same_b16_dtype(q, k, "k");
    at::cuda::CUDAGuard device_guard(g.device());

    auto g_cum = torch::empty_like(g);
    auto q_norm = torch::empty_like(q);
    auto k_norm = torch::empty_like(k);
    const hggcStream_t stream = pla::current_stream();
    const auto norm_device = logical_device_info(q);
    const bool medium_lowhead_prefix = cu_seqlens.numel() == 3 && g.size(2) == 8 &&
        ((q.size(2) == 2 && g.size(1) == 1041) || (q.size(2) == 4 && g.size(1) == 920));
    const bool stage_g = norm_device.compute_capability == 89 && norm_device.num_cu == 39 &&
        (medium_lowhead_prefix || (
        cu_seqlens.numel() == 2 &&
        (((g.size(2) == 24 && q.size(2) == 8 && g.size(1) == 823) ||
          (g.size(2) == 16 && q.size(2) == 8 && g.size(1) == 508)) ||
         (g.size(2) == 16 && (q.size(2) == 2 || q.size(2) == 4 || q.size(2) == 8) &&
          g.size(1) >= 768 && g.size(1) <= 1408) ||
         (g.size(1) == 24 && g.size(2) == 32 && q.size(2) == 8) ||
         (g.size(1) == 48 && ((q.size(2) == 16 &&
           (g.size(2) == 48 || g.size(2) == 64 || g.size(2) == 128)) ||
           (q.size(2) == 8 && g.size(2) == 64))))));
    launch_chunk_local_cumsum_qk_norm_cute(
        g.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
        chunk_indices.data_ptr<int>(), g_cum.data_ptr<float>(),
        data_ptr_b16(q), data_ptr_b16(k), data_ptr_b16(q_norm),
        data_ptr_b16(k_norm), chunk_indices.size(0), g.size(1), g.size(2),
        q.size(2), dtype_tag(q), stream, stage_g);
    return {g_cum, q_norm, k_norm};
}

std::vector<torch::Tensor> chunk_local_cumsum_qk_norm_gated(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices,
    torch::Tensor q,
    torch::Tensor k,
    int64_t valid_num_tokens,
    bool folded
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.chunk_local_cumsum_qk_norm_gated",
            {{"a", &a}, {"b", &b}, {"q", &q}, {"k", &k}, {"A_log", &A_log},
             {"dt_bias", &dt_bias}, {"cu_seqlens", &cu_seqlens}, {"chunk_indices", &chunk_indices}});
        auto options = a.options().dtype(torch::kFloat32);
        return {pla::bypass::output({1, a.size(0), a.size(1)}, options),
                pla::bypass::output({1, a.size(0), a.size(1)}, options),
                pla::bypass::output(q.sizes(), q.options()),
                pla::bypass::output(k.sizes(), k.options())};
    }
    const pla::sanity::Context sanity("flashqla.chunk_local_cumsum_qk_norm_gated", a);
    if (sanity) {
        sanity.check(a.dim() == 2 && q.dim() == 4, "a/q rank mismatch");
        sp::int_extent(sanity, a.size(0), "T");
        sp::int_extent(sanity, a.size(1), "H");
        sp::int_extent(sanity, q.size(2), "Hg");
        for (const auto& named : std::initializer_list<std::pair<const char*, torch::Tensor>>{
                 {"a", a}, {"b", b}, {"A_log", A_log}, {"dt_bias", dt_bias}})
            sanity.tensor(named.second, named.first);
        sanity_feature(sanity, q, "q", a.size(0), q.size(2), 128, true);
        sanity_feature(sanity, k, "k", a.size(0), q.size(2), 128, true);
        const auto cu = sp::offsets(sanity, cu_seqlens, a.size(0));
        sp::chunks(sanity, chunk_indices, cu);
        if (sanity.values_enabled())
            sanity.check(cu.back() == (valid_num_tokens < 0 ? a.size(0) : valid_num_tokens),
                         "valid_num_tokens must match cu_seqlens endpoint");
    }
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.sizes() == b.sizes(),
                "a and b must have the same [T,H] shape");
    TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1 &&
                    a.stride(0) >= a.size(1) && b.stride(0) >= b.size(1),
                "a and b must be head-contiguous with a positive token stride");
    TORCH_CHECK(a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16,
                "a and b must be fp16 or bf16");
    check_same_b16_dtype(a, b, "b");
    check_same_b16_dtype(a, q, "q");
    check_same_b16_dtype(a, k, "k");
    check_same_cuda_device(a, b, "b");
    check_same_cuda_device(a, A_log, "A_log");
    check_same_cuda_device(a, dt_bias, "dt_bias");
    check_same_cuda_device(a, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(a, chunk_indices, "chunk_indices");
    check_same_cuda_device(a, q, "q");
    check_same_cuda_device(a, k, "k");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32 &&
                dt_bias.dtype() == torch::kFloat32,
                "A_log and dt_bias must be fp32");
    TORCH_CHECK(A_log.dim() == 1 && dt_bias.dim() == 1 &&
                A_log.numel() == a.size(1) && dt_bias.numel() == a.size(1) &&
                A_log.is_contiguous() && dt_bias.is_contiguous(),
                "A_log and dt_bias must be contiguous [H]");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32,
                "cu_seqlens must be int32");
    TORCH_CHECK(chunk_indices.dtype() == torch::kInt32 &&
                chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                chunk_indices.is_contiguous(),
                "chunk_indices must be contiguous int32 [num_chunks,2]");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.sizes() == k.sizes(),
                "q and k must have the same [1,T,Hg,D] shape");
    TORCH_CHECK(q.size(0) == 1 && q.size(1) == a.size(0) && q.size(3) == 128,
                "q and k must be [1,T,Hg,128]");
    const int64_t qk_tokens = valid_num_tokens < 0
        ? a.size(0) : valid_num_tokens;
    TORCH_CHECK(qk_tokens > 0 && qk_tokens <= a.size(0),
                "valid_num_tokens must be in (0,T]");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 &&
                    q.stride(2) == 128 && k.stride(2) == 128 &&
                    q.stride(1) >= q.size(2) * q.size(3) &&
                    k.stride(1) >= k.size(2) * k.size(3),
                "q and k must be head-contiguous with a positive token stride");
    at::cuda::CUDAGuard device_guard(a.device());

    auto fp32_options = a.options().dtype(torch::kFloat32);
    auto g_cum = torch::empty({1, a.size(0), a.size(1)}, fp32_options);
    auto beta = torch::empty_like(g_cum);
    // Raw Q/K may be views into one packed post-convolution row. The consumers
    // (KKT and recurrence) deliberately retain their original contiguous
    // normalized layout, so never preserve the packed input strides here.
    auto q_norm = torch::empty(q.sizes(), q.options());
    auto k_norm = torch::empty(k.sizes(), k.options());
    const hggcStream_t stream = pla::current_stream();
    if (folded) {
        TORCH_CHECK(q.dtype() == torch::kBFloat16 &&
                    ((a.size(1) == 24 && q.size(2) == 8) ||
                     (a.size(1) == 12 && q.size(2) == 4)) &&
                    logical_device_info(q).compute_capability == 89,
                    "folded gating requires M890P BF16 with H/Hg=24/8 or 12/4");
        launch_gdn_folded_gated_scan(
            a.data_ptr(), b.data_ptr(), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
            cu_seqlens.data_ptr<int>(), chunk_indices.data_ptr<int>(),
            g_cum.data_ptr<float>(), beta.data_ptr<float>(), q.data_ptr(), k.data_ptr(),
            q_norm.data_ptr(), k_norm.data_ptr(), chunk_indices.size(0), qk_tokens,
            a.size(1), q.size(2), a.stride(0), b.stride(0), q.stride(1), k.stride(1), stream);
    } else launch_chunk_local_cumsum_qk_norm_gated_cute(
        data_ptr_b16(a), data_ptr_b16(b), A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
        chunk_indices.data_ptr<int>(), g_cum.data_ptr<float>(),
        beta.data_ptr<float>(), data_ptr_b16(q), data_ptr_b16(k),
        data_ptr_b16(q_norm), data_ptr_b16(k_norm),
        chunk_indices.size(0), qk_tokens, a.size(1), q.size(2),
        a.stride(0), b.stride(0), q.stride(1), k.stride(1),
        dtype_tag(q), stream);
    return {g_cum, beta, q_norm, k_norm};
}

// get_warmup_chunks: returns (num_warmup_chunks[batch,H] int32, fallback_mask[batch,H] int8)
std::vector<torch::Tensor> get_warmup_chunks(
    torch::Tensor g,           // [1, T, H] fp32 chunk-local cumsum gate
    torch::Tensor ht_mask,     // [batch] int8
    torch::Tensor cu_seqlens,  // [batch+1] int32
    double threshold
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.get_warmup_chunks",
            {{"g", &g}, {"ht_mask", &ht_mask}, {"cu_seqlens", &cu_seqlens}});
        const auto batch = cu_seqlens.numel() - 1;
        return {pla::bypass::output({batch, g.size(2)}, g.options().dtype(torch::kInt32)),
                pla::bypass::output({batch, g.size(2)}, g.options().dtype(torch::kInt8))};
    }
    const pla::sanity::Context sanity("flashqla.get_warmup_chunks", g);
    if (sanity) {
        sanity.check(g.dim() == 3, "g must be [1,T,H]");
        sp::int_extent(sanity, g.size(1), "T");
        sp::int_extent(sanity, g.size(2), "H");
        sanity_gate(sanity, g, "g", g.size(1), g.size(2));
        const auto cu = sp::offsets(sanity, cu_seqlens, g.size(1));
        const auto mask = sp::binary_mask(sanity, ht_mask, cu_seqlens.numel() - 1, "ht_mask");
        if (sanity.values_enabled())
            for (size_t i = 0; i < mask.size(); ++i)
                sanity.check(mask[i] || (cu[i + 1] - cu[i]) % 32 == 0,
                             "warmup segments without an external state must contain full 32-token chunks");
    }
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(g.dim() == 3, "g must be [1,T,H]");
    check_same_cuda_device(g, ht_mask, "ht_mask");
    check_same_cuda_device(g, cu_seqlens, "cu_seqlens");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(ht_mask.dtype() == torch::kInt8, "ht_mask must be int8");
    TORCH_CHECK(std::isfinite(threshold) && threshold < 0.0 &&
                    std::isfinite(static_cast<float>(threshold)),
                "threshold must be finite, negative, and representable as fp32");
    at::cuda::CUDAGuard device_guard(g.device());

    int T = g.size(1);
    int H = g.size(2);
    int batch_size = cu_seqlens.size(0) - 1;

    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(g.device());
    auto opt_i8  = torch::TensorOptions().dtype(torch::kInt8).device(g.device());
    auto num_warmup = torch::empty({batch_size, H}, opt_i32);
    auto fallback   = torch::empty({batch_size, H}, opt_i8);

    const hggcStream_t stream = pla::current_stream();
    launch_get_warmup_chunks_cute(
        g.data_ptr<float>(),
        reinterpret_cast<const signed char*>(ht_mask.data_ptr<int8_t>()),
        cu_seqlens.data_ptr<int>(),
        num_warmup.data_ptr<int>(),
        reinterpret_cast<signed char*>(fallback.data_ptr<int8_t>()),
        batch_size, T, H, static_cast<float>(threshold), stream);

    return {num_warmup, fallback};
}

// kkt_solve: A = (I + StrictLower(-beta * K K^T))^{-1}, per (chunk, head).
// Returns A [1, T, H, CS] fp16.
torch::Tensor kkt_solve(
    torch::Tensor K,           // [1, T, Hg, DK] fp16
    torch::Tensor beta,        // [1, T, H] fp32
    torch::Tensor cu_seqlens,  // [batch+1] int32
    int64_t H,
    c10::optional<torch::Tensor> chunk_indices_opt
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.kkt_solve",
            {{"K", &K}, {"beta", &beta}, {"cu_seqlens", &cu_seqlens},
             {"chunk_indices", chunk_indices_opt.has_value() ? &chunk_indices_opt.value() : nullptr}});
        return pla::bypass::output({1, K.size(1), H, 32}, K.options());
    }
    const pla::sanity::Context sanity("flashqla.kkt_solve", K);
    if (sanity) {
        sanity.check(K.dim() == 4, "K must be [1,T,Hg,128]");
        sp::int_extent(sanity, K.size(1), "T");
        sp::int_extent(sanity, H, "H");
        sanity.check(K.size(2) > 0 && H % K.size(2) == 0,
                     "H must be divisible by positive Hg");
        sanity_feature(sanity, K, "K", K.size(1), K.size(2), 128);
        sanity_gate(sanity, beta, "beta", K.size(1), H);
        const auto cu = sp::offsets(sanity, cu_seqlens, K.size(1));
        if (chunk_indices_opt.has_value())
            sp::chunks(sanity, chunk_indices_opt.value(), cu);
        sanity.check((chunk_indices_opt.has_value() ? chunk_indices_opt->size(0)
                                                    : (K.size(1) + 31) / 32) * H <=
                         std::numeric_limits<int>::max(),
                     "chunk/head task count exceeds kernel int32 ABI");
    }
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(K.dtype() == torch::kFloat16 || K.dtype() == torch::kBFloat16, "K must be fp16 or bf16");
    TORCH_CHECK(K.dim() == 4, "K must be [1,T,Hg,DK]");
    check_same_cuda_device(K, beta, "beta");
    check_same_cuda_device(K, cu_seqlens, "cu_seqlens");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be fp32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    at::cuda::CUDAGuard device_guard(K.device());

    constexpr int CS = 32;
    int T  = K.size(1);
    int Hg = K.size(2);
    int DK = K.size(3);
    TORCH_CHECK(DK == 128, "kkt_solve expects DK=128");
    int batch_size = cu_seqlens.size(0) - 1;
    int num_chunks = (T + CS - 1) / CS;
    const int* chunk_indices_ptr = nullptr;
    if (chunk_indices_opt.has_value()) {
        auto chunk_indices = chunk_indices_opt.value();
        check_same_cuda_device(K, chunk_indices, "chunk_indices");
        TORCH_CHECK(chunk_indices.dtype() == torch::kInt32,
                    "chunk_indices must be int32");
        TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2,
                    "chunk_indices must be [num_chunks,2]");
        num_chunks = chunk_indices.size(0);
        chunk_indices_ptr = chunk_indices.data_ptr<int>();
    } else {
        TORCH_CHECK(batch_size == 1,
                    "packed multi-sequence kkt_solve requires chunk_indices");
    }

    auto opt_e = torch::TensorOptions().dtype(K.dtype()).device(K.device());
    auto A = torch::empty({1, T, (int)H, CS}, opt_e);

    const auto device = logical_device_info(K);
    const hggcStream_t stream = pla::current_stream();
    launch_kkt_solve_cute(
        data_ptr_b16(K), beta.data_ptr<float>(), data_ptr_b16(A),
        cu_seqlens.data_ptr<int>(),
        chunk_indices_ptr,
        (int)H, Hg, num_chunks, T, dtype_tag(K),
        device.index, device.num_cu, device.compute_capability, batch_size, stream);

    return A;
}

// Independent packed coefficients. Omitting A fuses the measured short KKT
// solve; supplying A preserves the ordinary solver and its public return value.
std::vector<torch::Tensor> packed_coefficients_tc01(
    torch::Tensor q, torch::Tensor k, torch::Tensor beta, torch::Tensor g,
    torch::Tensor cu, torch::Tensor chunks, int64_t H, double scale,
    c10::optional<torch::Tensor> a_opt) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.size(0) == 1 &&
                q.size(3) == 128 && q.is_contiguous(), "q must be contiguous [1,T,Hg,128]");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 ||
                q.scalar_type() == torch::kBFloat16, "q must be fp16 or bf16");
    const int T = q.size(1), Hg = q.size(2);
    const bool fused = !a_opt.has_value();
    TORCH_CHECK(T > 0 && std::isfinite(scale) &&
                std::isfinite(static_cast<float>(scale)), "invalid packed GDN extent/scale");
    TORCH_CHECK((fused && H == 8 && (Hg == 2 || Hg == 4) && T <= 512) ||
                (!fused && H == 16 && Hg == 2), "unsupported packed GDN region");
    for (const auto& value : {k, beta, g, cu, chunks})
        check_same_cuda_device(q, value, "packed coefficient input");
    TORCH_CHECK(k.sizes() == q.sizes() && k.scalar_type() == q.scalar_type() &&
                k.is_contiguous(), "k must match contiguous q");
    for (const auto& value : {beta, g})
        TORCH_CHECK(value.scalar_type() == torch::kFloat32 && value.is_contiguous() &&
                    value.dim() == 3 && value.size(0) == 1 &&
                    value.size(1) == T && value.size(2) == H,
                    "packed coefficient gates must be contiguous fp32 [1,T,H]");
    TORCH_CHECK(cu.scalar_type() == torch::kInt32 && cu.is_contiguous() &&
                cu.dim() == 1 && cu.numel() == 2,
                "packed coefficients require single-sequence int32 offsets");
    TORCH_CHECK(chunks.scalar_type() == torch::kInt32 && chunks.is_contiguous() &&
                chunks.dim() == 2 && chunks.size(1) == 2 &&
                chunks.size(0) <= (T + 31) / 32,
                "packed coefficients require contiguous int32 chunk pairs");
    at::cuda::CUDAGuard guard(q.device());
    auto a = a_opt.has_value() ? a_opt.value() : torch::empty({1,T,H,32},q.options());
    check_same_cuda_device(q, a, "A");
    TORCH_CHECK(a.scalar_type() == q.scalar_type() && a.is_contiguous() &&
                a.dim() == 4 && a.size(0) == 1 && a.size(1) == T &&
                a.size(2) == H && a.size(3) == 32, "A must be [1,T,H,32]");
    auto pg = torch::empty_like(a), ag = torch::empty_like(a);
    auto ex = torch::empty({1,T,H},g.options()), rev = torch::empty_like(ex);
    if (pla::bypass::enabled()) return {a,pg,ag,ex,rev};
    TORCH_CHECK(gdn_packed_coefficients_tc01_enabled(),
                "packed GDN requires the register-chain reordered schedule");
    const pla::sanity::Context sanity("flashqla.packed_coefficients_tc01", q);
    if (sanity) {
        sanity_feature(sanity, q, "q", T, Hg, 128);
        sanity_feature(sanity, k, "k", T, Hg, 128);
        sanity_gate(sanity, beta, "beta", T, H);
        sanity_gate(sanity, g, "g", T, H);
        const auto offsets = sp::offsets(sanity, cu, T);
        sp::chunks(sanity, chunks, offsets);
        sanity.check(chunks.size(0) * H <= std::numeric_limits<int>::max(),
                     "chunk/head task count exceeds kernel int32 ABI");
    }
    const auto device = logical_device_info(q);
    TORCH_CHECK(device.compute_capability == 80, "packed coefficients require tc01");
    const int heads_per_group = T <= 512 ? 1 : 2;
    launch_gdn_packed_coefficients_tc01(a.data_ptr(),pg.data_ptr(),ag.data_ptr(),
        ex.data_ptr<float>(),rev.data_ptr<float>(),q.data_ptr(),k.data_ptr(),
        beta.data_ptr<float>(),g.data_ptr<float>(),cu.data_ptr<int>(),chunks.data_ptr<int>(),
        chunks.size(0),T,H,Hg,static_cast<float>(scale),dtype_tag(q),fused,
        heads_per_group,pla::current_stream());
    return {a,pg,ag,ex,rev};
}

void check_folded_coefficients(torch::Tensor const& reference, torch::Tensor const& coefficients, int H) {
    check_same_cuda_device(reference, coefficients, "coefficients");
    TORCH_CHECK(reference.dtype() == torch::kBFloat16 && H > 0 && reference.size(2) > 0 &&
                H % reference.size(2) == 0 &&
                logical_device_info(reference).compute_capability == 89,
                "folded prefill requires M890P BF16 with grouped value heads");
    TORCH_CHECK(coefficients.dtype() == torch::kBFloat16 && coefficients.is_contiguous() &&
                coefficients.dim() == 4 && coefficients.size(0) == 1 &&
                coefficients.size(1) == reference.size(1) && coefficients.size(2) == H &&
                coefficients.size(3) == 162, "invalid folded coefficient workspace");
}

std::tuple<torch::Tensor, torch::Tensor> folded_coefficients(torch::Tensor q, torch::Tensor k,
    torch::Tensor beta, torch::Tensor g, torch::Tensor cu, torch::Tensor chunks, double scale, bool precise) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.folded_coefficients",
            {{"q", &q}, {"k", &k}, {"beta", &beta}, {"g", &g}, {"cu", &cu}, {"chunks", &chunks}});
        return {pla::bypass::output({1, q.size(1), beta.size(2), 32}, q.options()),
                pla::bypass::output({1, q.size(1), beta.size(2), 162}, q.options())};
    }
    TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.size(0) == 1 && q.size(3) == 128 &&
                q.is_contiguous() && q.sizes() == k.sizes() && k.is_contiguous(),
                "folded coefficients require contiguous [1,T,Hg,128] q/k");
    check_same_cuda_device(q, k, "k");
    check_same_b16_dtype(q, k, "k");
    TORCH_CHECK(beta.dim() == 3 && beta.size(2) > 0, "beta must be [1,T,H]");
    const int H = beta.size(2), Hg = q.size(2);
    TORCH_CHECK(Hg > 0 && H % Hg == 0 && (precise || (H == 24 && Hg == 8)),
                "unsupported folded coefficient head configuration");
    for (const auto& tensor : {beta, g}) {
        check_same_cuda_device(q, tensor, "gate");
        TORCH_CHECK(tensor.dtype() == torch::kFloat32 && tensor.is_contiguous() &&
                    tensor.dim() == 3 && tensor.size(0) == 1 && tensor.size(1) == q.size(1) &&
                    tensor.size(2) == H, "g/beta must be contiguous FP32 [1,T,H]");
    }
    for (const auto& tensor : {cu, chunks}) {
        check_same_cuda_device(q, tensor, "metadata");
        TORCH_CHECK(tensor.dtype() == torch::kInt32 && tensor.is_contiguous(),
                    "folded metadata must be contiguous int32");
    }
    TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2 && chunks.dim() == 2 &&
                chunks.size(1) == 2, "invalid folded sequence metadata");
    at::cuda::CUDAGuard device_guard(q.device());
    // SoA storage: BF16 R[H,T,128], BF16 M[H,T,32], FP32 decay[H,T].
    auto a = torch::empty({1, q.size(1), H, 32}, q.options());
    auto coefficients = torch::empty({1, q.size(1), H, 162}, q.options());
    check_folded_coefficients(q, coefficients, H);
    launch_folded_coefficients_cute(q.data_ptr(), k.data_ptr(), a.data_ptr(),
        beta.data_ptr<float>(), g.data_ptr<float>(), coefficients.data_ptr(), cu.data_ptr<int>(),
        cu.numel() == 2 ? nullptr : chunks.data_ptr<int>(), chunks.size(0),
        q.size(1), H, Hg, (float)scale, precise, pla::current_stream());
    return {a, coefficients};
}

// correct_h0: affine-semigroup recurrence over CP segments. Returns cp_h0
// [cp_batch, H, DK, DV], fp32 or bf16 per out_bf16. raw_h0 optional (empty
// tensor -> no_raw variant). Indexed raw_h0 uses [slots,H,DV,DK]; legacy raw_h0
// uses [states,H,DK,DV].
//
// cp_h0 is fed straight back to fused_fwd as its h0, and fused_fwd carries a
// single compile-time state dtype, so the caller keeps cp_h0 in the pool's own
// dtype rather than a wider temporary.
torch::Tensor correct_h0_impl(
    torch::Tensor ht_buffer,     // [cp_batch, H, DK, DV] fp16
    torch::Tensor mt_buffer,     // [cp_batch, H, DK, DK] fp16
    torch::Tensor fallback_mask, // [cp_batch * H] int8
    torch::Tensor seq_map_r2c,   // [raw_batch+1] int32
    int64_t raw_batch_size,
    c10::optional<torch::Tensor> raw_h0_opt,
    c10::optional<torch::Tensor> raw_h0_indices_opt,
    bool raw_h0_k_last,
    bool out_bf16,
    c10::optional<torch::Tensor> precision_flags_opt,
    int64_t key_heads,
    bool folded = false,
    c10::optional<torch::Tensor> seeded_cp_h0 = c10::nullopt
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.correct_h0",
            {{"ht_buffer", &ht_buffer}, {"mt_buffer", &mt_buffer},
             {"fallback_mask", &fallback_mask}, {"seq_map_r2c", &seq_map_r2c},
             {"raw_h0", raw_h0_opt.has_value() ? &raw_h0_opt.value() : nullptr},
             {"raw_h0_indices", raw_h0_indices_opt.has_value() ? &raw_h0_indices_opt.value() : nullptr}});
        return pla::bypass::output(ht_buffer.sizes(), ht_buffer.options().dtype(
            out_bf16 ? torch::kBFloat16 : torch::kFloat32));
    }
    const pla::sanity::Context sanity("flashqla.correct_h0", ht_buffer);
    if (sanity) {
        sanity.tensor(ht_buffer, "ht_buffer");
        sanity.tensor(mt_buffer, "mt_buffer");
        sanity.check(ht_buffer.dim() == 4 && ht_buffer.is_contiguous() &&
                         ht_buffer.size(2) == 128 && ht_buffer.size(3) == 128,
                     "ht_buffer must be contiguous [cp_batch,H,128,128]");
        sp::int_extent(sanity, ht_buffer.size(0), "cp_batch");
        sp::int_extent(sanity, ht_buffer.size(1), "H");
        sp::int_extent(sanity, raw_batch_size, "raw_batch_size");
        sanity.check(mt_buffer.sizes() == ht_buffer.sizes() && mt_buffer.is_contiguous(),
                     "mt_buffer must match contiguous ht_buffer shape");
        sp::binary_mask(sanity, fallback_mask, ht_buffer.size(0) * ht_buffer.size(1), "fallback_mask");
        const auto map = sp::offsets(sanity, seq_map_r2c, ht_buffer.size(0), "seq_map_r2c");
        sanity.check(seq_map_r2c.numel() == raw_batch_size + 1,
                     "seq_map_r2c must have raw_batch_size+1 entries");
        if (sanity.values_enabled()) {
            sanity.check(map.back() == ht_buffer.size(0),
                         "seq_map_r2c must cover all CP segments");
            for (size_t i = 1; i < map.size(); ++i)
                sanity.check(map[i] > map[i - 1],
                             "each raw sequence must own at least one CP segment");
        }
        if (raw_h0_opt.has_value() && raw_h0_opt->numel() > 0) {
            sanity.tensor(raw_h0_opt.value(), "raw_h0");
            sanity.check(raw_h0_opt->dim() == 4, "raw_h0 must have rank four");
            if (raw_h0_indices_opt.has_value())
                sp::slots(sanity, raw_h0_indices_opt.value(), raw_batch_size,
                          raw_h0_opt->size(0), "raw_h0_indices", false);
        }
    }
    TORCH_CHECK(ht_buffer.is_cuda(), "ht_buffer must be CUDA");
    TORCH_CHECK(ht_buffer.dtype() == torch::kFloat16 || ht_buffer.dtype() == torch::kBFloat16 ||
                (folded && ht_buffer.dtype() == torch::kFloat32), "invalid ht_buffer dtype");
    check_same_cuda_device(ht_buffer, mt_buffer, "mt_buffer");
    check_same_cuda_device(ht_buffer, fallback_mask, "fallback_mask");
    check_same_cuda_device(ht_buffer, seq_map_r2c, "seq_map_r2c");
    TORCH_CHECK(ht_buffer.dtype() == mt_buffer.dtype(), "mt_buffer must match ht_buffer dtype");
    TORCH_CHECK(seq_map_r2c.dtype() == torch::kInt32, "seq_map_r2c must be int32");
    TORCH_CHECK(fallback_mask.dtype() == torch::kInt8, "fallback_mask must be int8");

    int cp_batch = ht_buffer.size(0);
    int H  = ht_buffer.size(1);
    int DK = ht_buffer.size(2);
    int DV = ht_buffer.size(3);
    TORCH_CHECK(DK == 128 && DV == 128, "correct_h0 expects DK=DV=128");
    at::cuda::CUDAGuard device_guard(ht_buffer.device());

    auto opt_f = torch::TensorOptions()
        .dtype(out_bf16 ? torch::kBFloat16 : torch::kFloat32)
        .device(ht_buffer.device());
    // The kernel writes the first state and every following CP-segment state.
    auto cp_h0 = seeded_cp_h0 ? *seeded_cp_h0
                              : torch::empty({cp_batch, H, DK, DV}, opt_f);
    if (seeded_cp_h0) {
        TORCH_CHECK(folded &&
                    (ht_buffer.dtype() == torch::kBFloat16 || ht_buffer.dtype() == torch::kFloat32) &&
                    H == 24 && raw_h0_indices_opt && raw_batch_size == 1 &&
                    cp_batch >= 2 && !out_bf16,
                    "seeded correction requires a split indexed FP32 sequence");
        check_same_cuda_device(ht_buffer, cp_h0, "seeded_cp_h0");
        TORCH_CHECK(cp_h0.dtype() == torch::kFloat32 && cp_h0.is_contiguous() &&
                    cp_h0.sizes() == ht_buffer.sizes(),
                    "seeded_cp_h0 must match the FP32 corrected state layout");
    }

    const auto device = logical_device_info(ht_buffer);
    const int* precision_flags = nullptr;
    if (precision_flags_opt.has_value()) {
        const auto flags = precision_flags_opt.value();
        check_same_cuda_device(ht_buffer, flags, "precision_flags");
        TORCH_CHECK(device.compute_capability == 89 && device.num_cu == 39 &&
                    raw_batch_size == 64 && H == 8 && (key_heads == 2 || key_heads == 4),
                    "private selected correct requires tc02/39CU/N64/H8/Hg2 or4");
        TORCH_CHECK(H > 0 && key_heads > 0 && H % key_heads == 0 &&
                    flags.scalar_type() == torch::kInt32 && flags.is_contiguous() &&
                    flags.dim() == 2 && flags.size(0) == raw_batch_size &&
                    flags.size(1) == key_heads,
                    "precision_flags must be contiguous int32 [raw_batch,key_heads]");
        precision_flags = flags.data_ptr<int>();
    }
    const hggcStream_t stream = pla::current_stream();
    const signed char* fb = reinterpret_cast<signed char*>(fallback_mask.data_ptr<int8_t>());

    const bool has_raw_h0 = raw_h0_opt.has_value() && raw_h0_opt->numel() > 0;
    TORCH_CHECK(!raw_h0_indices_opt.has_value() || has_raw_h0,
                "raw_h0_indices requires a non-empty raw_h0");
    TORCH_CHECK(!raw_h0_k_last || has_raw_h0,
                "raw_h0_k_last requires a non-empty raw_h0");
    if (has_raw_h0) {
        auto raw_h0 = raw_h0_opt.value();
        check_same_cuda_device(ht_buffer, raw_h0, "raw_h0");
        TORCH_CHECK(raw_h0.dtype() == torch::kFloat32 ||
                    raw_h0.dtype() == torch::kBFloat16,
                    "raw_h0 must be fp32 or bf16");
        TORCH_CHECK(raw_h0.is_contiguous(), "raw_h0 must be contiguous");
        if (raw_h0_k_last) {
            TORCH_CHECK(raw_h0.dim() == 4 && raw_h0.size(0) > 0 &&
                        raw_h0.size(1) == H && raw_h0.size(2) == DV &&
                        raw_h0.size(3) == DK,
                        "indexed raw_h0 must be [num_slots,H,DV,DK]");
        } else {
            TORCH_CHECK(raw_h0.dim() == 4 && raw_h0.size(0) > 0 &&
                        raw_h0.size(1) == H && raw_h0.size(2) == DK &&
                        raw_h0.size(3) == DV,
                        "raw_h0 must be [num_states,H,DK,DV]");
        }
        const int* raw_h0_indices_ptr = nullptr;
        if (raw_h0_indices_opt.has_value()) {
            auto raw_h0_indices = raw_h0_indices_opt.value();
            check_same_cuda_device(ht_buffer, raw_h0_indices, "raw_h0_indices");
            TORCH_CHECK(raw_h0_indices.dtype() == torch::kInt32,
                        "raw_h0_indices must be int32");
            TORCH_CHECK(raw_h0_indices.is_contiguous(),
                        "raw_h0_indices must be contiguous");
            TORCH_CHECK(raw_h0_indices.numel() == raw_batch_size,
                        "raw_h0_indices must have raw_batch_size entries");
            raw_h0_indices_ptr = raw_h0_indices.data_ptr<int>();
        } else {
            TORCH_CHECK(raw_h0.size(0) >= raw_batch_size,
                        "raw_h0 must have at least raw_batch_size states");
        }
        if (folded) {
            TORCH_CHECK((ht_buffer.dtype() == torch::kFloat32 || ht_buffer.dtype() == torch::kBFloat16) &&
                        raw_h0.dtype() == torch::kFloat32 && raw_h0_k_last &&
                        raw_h0_indices_ptr && !out_bf16 && device.compute_capability == 89,
                        "folded correction requires an indexed M890P FP32 state pool");
            if (ht_buffer.dtype() == torch::kFloat32) {
                launch_folded_correct_fp32(cp_h0.data_ptr(), raw_h0.data_ptr(),
                    ht_buffer.data_ptr(), mt_buffer.data_ptr(), fb,
                    seq_map_r2c.data_ptr<int>(), raw_h0_indices_ptr,
                    (int)raw_batch_size, H, stream, seeded_cp_h0.has_value());
            } else {
                launch_gdn_correct_native(cp_h0.data_ptr(), raw_h0.data_ptr(),
                    ht_buffer.data_ptr(), mt_buffer.data_ptr(), fb,
                    seq_map_r2c.data_ptr<int>(), raw_h0_indices_ptr,
                    cp_batch, (int)raw_batch_size, H, device.index, stream,
                    seeded_cp_h0.has_value());
            }
        } else launch_correct_h0_raw_cute(
            cp_h0.data_ptr(), raw_h0.data_ptr(),
            data_ptr_b16(ht_buffer), data_ptr_b16(mt_buffer),
            fb, seq_map_r2c.data_ptr<int>(), raw_h0_indices_ptr,
            cp_batch, (int)raw_batch_size, H, raw_h0_k_last ? 1 : 0,
            raw_h0.dtype() == torch::kBFloat16, out_bf16,
            dtype_tag(ht_buffer), device.index, device.num_cu,
            device.compute_capability, precision_flags, static_cast<int>(key_heads), stream);
    } else {
        launch_correct_h0_no_raw_cute(
            cp_h0.data_ptr(),
            data_ptr_b16(ht_buffer), data_ptr_b16(mt_buffer),
            fb, seq_map_r2c.data_ptr<int>(),
            cp_batch, (int)raw_batch_size, H, out_bf16, dtype_tag(ht_buffer),
            device.index, device.num_cu, device.compute_capability, precision_flags, static_cast<int>(key_heads), stream);
    }
    return cp_h0;
}

// Public native entry preserves complete cp_h0 initialization.
torch::Tensor correct_h0(
    torch::Tensor ht_buffer, torch::Tensor mt_buffer, torch::Tensor fallback_mask,
    torch::Tensor seq_map_r2c, int64_t raw_batch_size,
    c10::optional<torch::Tensor> raw_h0_opt,
    c10::optional<torch::Tensor> raw_h0_indices_opt,
    bool raw_h0_k_last, bool out_bf16, bool folded,
    c10::optional<torch::Tensor> seeded_cp_h0) {
    return correct_h0_impl(ht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
        raw_batch_size, raw_h0_opt, raw_h0_indices_opt, raw_h0_k_last, out_bf16,
        c10::nullopt, 0, folded, seeded_cp_h0);
}

// Private opt-in: selected cp_h0 slices are deliberately unwritten and must
// only be consumed by the matching precision-flagged fused/repair pipeline.
torch::Tensor correct_h0_selected_skip(
    torch::Tensor ht_buffer, torch::Tensor mt_buffer, torch::Tensor fallback_mask,
    torch::Tensor seq_map_r2c, int64_t raw_batch_size,
    c10::optional<torch::Tensor> raw_h0_opt,
    c10::optional<torch::Tensor> raw_h0_indices_opt,
    bool raw_h0_k_last, bool out_bf16, torch::Tensor precision_flags,
    int64_t key_heads) {
    return correct_h0_impl(ht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
        raw_batch_size, raw_h0_opt, raw_h0_indices_opt, raw_h0_k_last, out_bf16,
        precision_flags, key_heads);
}

// prepare_h: warmup-chunk state recurrence per (batch, head). Returns
// (ht [batch*H, DK, DV] fp16, mt [batch*H, DK, DK] fp16). PPU1.5 uses one
// combined CTA for the additive and homogeneous recurrences while sharing
// K/A/g inputs. PPU1.0/1.1 retains the validated two-pass implementation.
// Inactive transition tiles are not consumed by correct_h0.
std::vector<torch::Tensor> prepare_h_impl(
    torch::Tensor k,                 // [1, T, Hg, DK] fp16
    torch::Tensor v,                 // [1, T, H, DV] fp16
    torch::Tensor a,                 // [1, T, H, CS] fp16
    torch::Tensor g,                 // [1, T, H] fp32
    torch::Tensor beta,              // [1, T, H] fp32
    torch::Tensor cu_seqlens,        // [batch+1] int32
    torch::Tensor num_warmup_chunks, // [batch*H] int32
    int64_t H,
    int64_t Hg,
    c10::optional<torch::Tensor> precision_flags_opt,
    c10::optional<torch::Tensor> cp_to_raw_opt,
    c10::optional<torch::Tensor> raw_to_cp_opt,
    c10::optional<torch::Tensor> raw_cu_seqlens_opt,
    c10::optional<torch::Tensor> coefficients_opt = c10::nullopt,
    bool precise = false,
    bool skip_last_segment = false,
    c10::optional<torch::Tensor> seeded_cp_h0 = c10::nullopt,
    c10::optional<torch::Tensor> initial_pool = c10::nullopt,
    c10::optional<torch::Tensor> first_slot = c10::nullopt
) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.prepare_h",
            {{"k", &k}, {"v", &v}, {"a", &a}, {"g", &g}, {"beta", &beta},
             {"cu_seqlens", &cu_seqlens}, {"num_warmup_chunks", &num_warmup_chunks}});
        const auto batch_heads = (cu_seqlens.numel() - 1) * H;
        const auto options = k.options().dtype(
            coefficients_opt.has_value() && precise ? torch::kFloat32 : k.scalar_type());
        return {pla::bypass::output({batch_heads, 128, 128}, options),
                pla::bypass::output({batch_heads, 128, 128}, options)};
    }
    const pla::sanity::Context sanity("flashqla.prepare_h", k);
    if (sanity) {
        sanity.check(k.dim() == 4, "k must have rank four");
        sp::int_extent(sanity, k.size(1), "T");
        sp::int_extent(sanity, H, "H");
        sanity.check(Hg > 0 && H % Hg == 0, "H must be divisible by positive Hg");
        sanity_feature(sanity, k, "k", k.size(1), Hg, 128);
        sanity_feature(sanity, v, "v", k.size(1), H, 128, true);
        sanity_feature(sanity, a, "a", k.size(1), H, 32);
        sanity_gate(sanity, g, "g", k.size(1), H);
        sanity_gate(sanity, beta, "beta", k.size(1), H);
        const auto cu = sp::offsets(sanity, cu_seqlens, k.size(1));
        sanity.tensor(num_warmup_chunks, "num_warmup_chunks");
        sanity.check(num_warmup_chunks.scalar_type() == torch::kInt32 &&
                         num_warmup_chunks.is_contiguous() &&
                         num_warmup_chunks.numel() == (cu_seqlens.numel() - 1) * H,
                     "num_warmup_chunks must contain contiguous int32 [batch*H] entries");
        sanity.check((cu_seqlens.numel() - 1) * H <= std::numeric_limits<int>::max(),
                     "batch*H exceeds kernel int32 ABI");
        if (sanity.values_enabled()) {
            const auto warmup = sanity.integers(num_warmup_chunks);
            for (size_t i = 0; i < warmup.size(); ++i) {
                const int64_t seq = i / H;
                sanity.check(warmup[i] >= 0 && warmup[i] <= (cu[seq + 1] - cu[seq] + 31) / 32,
                             "num_warmup_chunks out of range at entry ", i);
            }
        }
    }
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(k.dtype() == torch::kFloat16 || k.dtype() == torch::kBFloat16, "k must be fp16 or bf16");
    check_same_cuda_device(k, v, "v");
    check_same_cuda_device(k, a, "a");
    check_same_cuda_device(k, g, "g");
    check_same_cuda_device(k, beta, "beta");
    check_same_cuda_device(k, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(k, num_warmup_chunks, "num_warmup_chunks");
    check_same_b16_dtype(k, v, "v");
    check_same_b16_dtype(k, a, "a");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be fp32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(num_warmup_chunks.dtype() == torch::kInt32,
                "num_warmup_chunks must be int32");
    TORCH_CHECK(k.dim() == 4, "k must be [1,T,Hg,DK]");
    at::cuda::CUDAGuard device_guard(k.device());

    constexpr int DK = 128, DV = 128;
    int T = k.size(1);
    int batch_size = cu_seqlens.size(0) - 1;
    TORCH_CHECK(k.size(3) == DK, "prepare_h expects DK=128");
    TORCH_CHECK(v.dim() == 4 && v.size(0) == 1 && v.size(1) == T &&
                    v.size(2) == H && v.size(3) == DV &&
                    v.stride(3) == 1 && v.stride(2) == DV &&
                    v.stride(1) >= H * DV,
                "v must be [1,T,H,128], head-contiguous, with a positive "
                "token stride");

    auto opt_e = torch::TensorOptions().dtype(
        coefficients_opt.has_value() && precise ? torch::kFloat32 : k.scalar_type()).device(k.device());
    auto ht = torch::empty({batch_size * (int)H, DK, DV}, opt_e);
    auto mt = torch::empty({batch_size * (int)H, DK, DK}, opt_e);

    if (seeded_cp_h0) {
        TORCH_CHECK(coefficients_opt && H == 24 && Hg == 8 &&
                    initial_pool && first_slot && batch_size >= 2,
                    "seeded prepare requires a short folded 24/8 sequence");
        check_same_cuda_device(k, *seeded_cp_h0, "seeded_cp_h0");
        check_same_cuda_device(k, *initial_pool, "initial_pool");
        check_same_cuda_device(k, *first_slot, "first_slot");
        TORCH_CHECK(seeded_cp_h0->dtype() == torch::kFloat32 &&
                    seeded_cp_h0->is_contiguous() &&
                    seeded_cp_h0->sizes() == at::IntArrayRef({batch_size, H, DK, DV}),
                    "seeded_cp_h0 must be contiguous FP32 [segments,H,128,128]");
        TORCH_CHECK(initial_pool->dtype() == torch::kFloat32 &&
                    initial_pool->is_contiguous() && initial_pool->dim() == 4 &&
                    initial_pool->size(0) > 0 && initial_pool->size(1) == H &&
                    initial_pool->size(2) == DV && initial_pool->size(3) == DK &&
                    first_slot->dtype() == torch::kInt32 && first_slot->is_contiguous() &&
                    first_slot->numel() == 1,
                    "seeded prepare requires a contiguous FP32 state pool and one slot");
    }

    const auto device = logical_device_info(k);
    const int* precision_flags = nullptr;
    const int* cp_to_raw = nullptr;
    const int* raw_to_cp = nullptr;
    const int* raw_cu_seqlens = nullptr;
    int raw_batch_size = 0;
    if (precision_flags_opt.has_value()) {
        TORCH_CHECK(cp_to_raw_opt.has_value() && raw_to_cp_opt.has_value() &&
                    raw_cu_seqlens_opt.has_value(), "selected prepare requires all CP maps");
        const auto flags = precision_flags_opt.value();
        const auto c2r = cp_to_raw_opt.value();
        const auto r2c = raw_to_cp_opt.value();
        const auto raw_cu = raw_cu_seqlens_opt.value();
        check_same_cuda_device(k, flags, "precision_flags");
        check_same_cuda_device(k, c2r, "cp_to_raw");
        check_same_cuda_device(k, r2c, "raw_to_cp");
        check_same_cuda_device(k, raw_cu, "raw_cu_seqlens");
        TORCH_CHECK(device.compute_capability == 89 && device.num_cu == 39 &&
                    H == 8 && (Hg == 2 || Hg == 4),
                    "private selected prepare requires tc02/39CU/H8/Hg2 or4");
        TORCH_CHECK(flags.scalar_type() == torch::kInt32 && flags.is_contiguous() &&
                    flags.dim() == 2 && flags.size(0) == 64 && flags.size(1) == Hg,
                    "precision_flags must be contiguous int32 [64,Hg]");
        TORCH_CHECK(c2r.dtype() == torch::kInt32 && c2r.is_contiguous() &&
                    c2r.dim() == 1 && c2r.numel() == batch_size,
                    "cp_to_raw must be contiguous int32 [cp_batch]");
        TORCH_CHECK(r2c.dtype() == torch::kInt32 && r2c.is_contiguous() &&
                    r2c.dim() == 1 && r2c.numel() == 65 &&
                    raw_cu.dtype() == torch::kInt32 && raw_cu.is_contiguous() &&
                    raw_cu.dim() == 1 && raw_cu.numel() == 65,
                    "raw_to_cp and raw_cu must be contiguous int32 [65]");
        TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.is_contiguous() &&
                    num_warmup_chunks.is_contiguous() &&
                    num_warmup_chunks.numel() == static_cast<int64_t>(batch_size) * H,
                    "selected prepare requires contiguous CP offsets and warmup");
        precision_flags = flags.data_ptr<int>();
        cp_to_raw = c2r.data_ptr<int>();
        raw_to_cp = r2c.data_ptr<int>();
        raw_cu_seqlens = raw_cu.data_ptr<int>();
        raw_batch_size = 64;
    } else {
        TORCH_CHECK(!cp_to_raw_opt.has_value() && !raw_to_cp_opt.has_value() &&
                    !raw_cu_seqlens_opt.has_value(), "CP skip maps require precision_flags");
    }
    const hggcStream_t stream = pla::current_stream();
    const bool precompute_x = !coefficients_opt.has_value() && flashqla::tc02_precompute_x_allowed(
        device.compute_capability, device.num_cu, T, H, Hg, batch_size);
    torch::Tensor x_scratch;
    if (precompute_x) {
        x_scratch = torch::empty({(T + 31) / 32 + batch_size, (int)H, DK, 32}, opt_e);
    }
    if (coefficients_opt.has_value()) {
        auto coefficients = coefficients_opt.value();
        check_folded_coefficients(k, coefficients, H);
        // The last segment has no continuation. Retain the full scratch shape
        // for correct_h0, which reads only the nonterminal transitions.
        const int prepare_batch = batch_size - (skip_last_segment && batch_size > 1);
        launch_folded_prepare_h_cute(k.data_ptr(), v.data_ptr(), coefficients.data_ptr(),
            cu_seqlens.data_ptr<int>(), num_warmup_chunks.data_ptr<int>(), ht.data_ptr(),
            mt.data_ptr(), prepare_batch,
            T, (int)H, (int)Hg, v.stride(1), precise, stream,
            seeded_cp_h0 ? initial_pool->data_ptr<float>() : nullptr,
            seeded_cp_h0 ? first_slot->data_ptr<int>() : nullptr,
            seeded_cp_h0 ? seeded_cp_h0->data_ptr<float>() : nullptr);
    } else launch_prepare_h_cp_cute(
        data_ptr_b16(k), data_ptr_b16(v), data_ptr_b16(a),
        g.data_ptr<float>(), beta.data_ptr<float>(), data_ptr_b16(ht), data_ptr_b16(mt),
        cu_seqlens.data_ptr<int>(), num_warmup_chunks.data_ptr<int>(),
        batch_size, T, (int)H, (int)Hg, dtype_tag(k),
        v.stride(1),
        device.index, device.num_cu, device.compute_capability,
        precompute_x ? data_ptr_b16(x_scratch) : nullptr, precision_flags,
        cp_to_raw, raw_to_cp, raw_cu_seqlens, raw_batch_size, stream);

    return {ht, mt};
}

// Public native prepare preserves its original initialized-scratch contract.
std::vector<torch::Tensor> prepare_h(
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor g,
    torch::Tensor beta, torch::Tensor cu, torch::Tensor warmup, int64_t H, int64_t Hg,
    c10::optional<torch::Tensor> coefficients, bool precise, bool skip_last_segment,
    c10::optional<torch::Tensor> seeded_cp_h0, c10::optional<torch::Tensor> initial_pool,
    c10::optional<torch::Tensor> first_slot) {
    return prepare_h_impl(k, v, a, g, beta, cu, warmup, H, Hg,
        c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt,
        coefficients, precise, skip_last_segment, seeded_cp_h0, initial_pool, first_slot);
}

// Internal only: selected T1 scratch may be unwritten. Matching selected-correct
// and fused consumers must skip exactly the same raw-sequence/key-head flags.
std::vector<torch::Tensor> prepare_h_selected_t1_skip(
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor g,
    torch::Tensor beta, torch::Tensor cu, torch::Tensor warmup, int64_t H, int64_t Hg,
    torch::Tensor flags, torch::Tensor cp_to_raw, torch::Tensor raw_to_cp,
    torch::Tensor raw_cu) {
    return prepare_h_impl(k, v, a, g, beta, cu, warmup, H, Hg,
        flags, cp_to_raw, raw_to_cp, raw_cu);
}

// fused_fwd: main forward pass. Returns (o [1,T,H,DV] fp16, ht_out
// [raw_batch,H,DK,DV] fp32 or bf16 per final_state_bf16). When raw_state is
// supplied, final state is written there in place, in raw_state's own dtype,
// and the returned ht_out is empty. Otherwise ht_out may be omitted when the
// caller does not consume the final state. h0 optional (empty -> zero).
//
// The state dtype is a single compile-time kernel parameter, so h0, raw_state
// and ht_out all share it: every state tensor handed in must agree with
// final_state_bf16.
std::vector<torch::Tensor> fused_fwd(
    torch::Tensor q,               // [1, T, Hg, DK] fp16
    torch::Tensor k,               // [1, T, Hg, DK] fp16
    torch::Tensor v,               // [1, T, H, DV] fp16
    torch::Tensor a,               // [1, T, H, CS] fp16
    torch::Tensor beta,            // [1, T, H] fp32
    torch::Tensor g,               // [1, T, H] fp32
    torch::Tensor cu_seqlens,      // [cp_batch+1] int32
    c10::optional<torch::Tensor> cp_seq_map_opt,  // [cp_batch] int32; absent for identity
    torch::Tensor raw_cu_seqlens,  // [raw_batch+1] int32
    int64_t valid_num_tokens,
    int64_t H,
    int64_t Hg,
    double scale,
    c10::optional<torch::Tensor> h0_opt,
    c10::optional<torch::Tensor> raw_state_opt,
    c10::optional<torch::Tensor> raw_state_indices_opt,
    bool output_final_state,
    bool final_state_bf16,
    c10::optional<torch::Tensor> raw_state_input_opt,
    c10::optional<torch::Tensor> num_warmup_opt,
    c10::optional<torch::Tensor> fallback_mask_opt,
    c10::optional<torch::Tensor> packed_pg_opt,
    c10::optional<torch::Tensor> precision_flags_opt,
    c10::optional<torch::Tensor> coefficients_opt,
    bool skip_folded_repair_heads
) {
    if (pla::bypass::enabled()) {
        const c10::cuda::OptionalCUDAGuard guard(q.is_cuda()
            ? c10::optional<c10::Device>(q.device()) : c10::nullopt);
        const auto raw_batch = raw_cu_seqlens.numel() - 1;
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashqla.fused_fwd",
            {{"q", &q}, {"k", &k}, {"v", &v}, {"a", &a}, {"beta", &beta}, {"g", &g},
             {"cu_seqlens", &cu_seqlens}, {"raw_cu_seqlens", &raw_cu_seqlens},
             {"cp_seq_map", cp_seq_map_opt.has_value() ? &cp_seq_map_opt.value() : nullptr},
             {"h0", h0_opt.has_value() ? &h0_opt.value() : nullptr},
             {"raw_state", raw_state_opt.has_value() ? &raw_state_opt.value() : nullptr},
             {"raw_state_indices", raw_state_indices_opt.has_value() ? &raw_state_indices_opt.value() : nullptr},
             {"raw_state_input", raw_state_input_opt.has_value() ? &raw_state_input_opt.value() : nullptr},
             {"num_warmup", num_warmup_opt.has_value() ? &num_warmup_opt.value() : nullptr},
             {"fallback_mask", fallback_mask_opt.has_value() ? &fallback_mask_opt.value() : nullptr}},
            "valid_num_tokens=" + std::to_string(valid_num_tokens) +
            " scale=" + std::to_string(scale) +
            " output_final_state=" + std::to_string(output_final_state));
        if (pla::instrument::show_params()) pla::instrument::dump_case(raw_batch, q.size(1), q.size(3), v.size(3), Hg, H,
            get_dtype_string(q), false, &raw_cu_seqlens, -5.0f, true);
        auto state_options = q.options().dtype(
            final_state_bf16 ? torch::kBFloat16 : torch::kFloat32);
        auto ht = (!raw_state_opt.has_value() && output_final_state)
            ? pla::bypass::output({raw_batch, H, q.size(3), v.size(3)}, state_options)
            : pla::bypass::output({0}, state_options);
        return {pla::bypass::output({1, q.size(1), H, v.size(3)}, q.options()), ht};
    }
    const pla::sanity::Context sanity("flashqla.fused_fwd", q);
    if (sanity) {
        sanity.check(q.dim() == 4, "q must have rank four");
        sp::int_extent(sanity, q.size(1), "T");
        sp::int_extent(sanity, H, "H");
        sanity.check(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)),
                     "scale must be finite and representable as fp32");
        for (const auto& named : std::initializer_list<std::pair<const char*, torch::Tensor>>{
                 {"q", q}, {"k", k}, {"v", v}, {"a", a}, {"beta", beta}, {"g", g}})
            sanity.tensor(named.second, named.first);
        const auto cu = sp::offsets(sanity, cu_seqlens, q.size(1));
        const auto raw = sp::offsets(sanity, raw_cu_seqlens, q.size(1), "raw_cu_seqlens");
        std::vector<int64_t> map;
        if (cp_seq_map_opt.has_value()) {
            sanity.tensor(cp_seq_map_opt.value(), "cp_seq_map");
            sanity.check(cp_seq_map_opt->scalar_type() == torch::kInt32 &&
                             cp_seq_map_opt->dim() == 1 && cp_seq_map_opt->is_contiguous() &&
                             cp_seq_map_opt->numel() == cu_seqlens.numel() - 1,
                         "cp_seq_map must be contiguous int32 [cp_batch]");
            if (sanity.values_enabled()) map = sanity.integers(cp_seq_map_opt.value());
        }
        if (sanity.values_enabled()) {
            sanity.check(cu.back() == valid_num_tokens && raw.back() == valid_num_tokens,
                         "CP/raw cu endpoints must equal valid_num_tokens");
            if (!cp_seq_map_opt.has_value()) {
                sanity.check(cu == raw, "omitted cp_seq_map requires identical CP/raw offsets");
            } else {
                for (size_t i = 0; i < map.size(); ++i) {
                    const int64_t r = map[i];
                    sanity.check(r >= 0 && r < int64_t(raw.size()) - 1,
                                 "cp_seq_map raw sequence out of range at entry ", i);
                    sanity.check(cu[i] >= raw[r] && cu[i + 1] <= raw[r + 1],
                                 "CP segment must lie within its mapped raw sequence");
                    sanity.check((cu[i] - raw[r]) % 32 == 0 &&
                                     (cu[i + 1] == raw[r + 1] || (cu[i + 1] - raw[r]) % 32 == 0),
                                 "CP segment boundaries must preserve raw 32-token chunks");
                }
            }
        }
        if (packed_pg_opt.has_value()) sanity.tensor(packed_pg_opt.value(), "packed_pg");
        if (h0_opt.has_value() && h0_opt->numel() > 0) sanity.tensor(h0_opt.value(), "h0");
        if (raw_state_opt.has_value()) {
            sanity.tensor(raw_state_opt.value(), "raw_state");
            sanity.check(raw_state_opt->dim() == 4, "raw_state must have rank four");
            sanity.check(raw_state_indices_opt.has_value(), "raw_state_indices required with raw_state");
            sp::slots(sanity, raw_state_indices_opt.value(), raw_cu_seqlens.numel() - 1,
                      raw_state_opt->size(0), "raw_state_indices", true, raw);
            // These buffers are consumed while other CP CTAs write the pool.
            for (const auto& input : {q, k, v, a, beta, g})
                sanity.check(!sp::overlaps(raw_state_opt.value(), input),
                             "raw_state must not overlap token inputs");
            if (packed_pg_opt.has_value())
                sanity.check(!sp::overlaps(raw_state_opt.value(), packed_pg_opt.value()),
                             "raw_state must not overlap packed_pg");
            if (h0_opt.has_value() && h0_opt->numel() > 0)
                sanity.check(!sp::overlaps(raw_state_opt.value(), h0_opt.value()) ||
                                 (!cp_seq_map_opt.has_value() &&
                                  sp::same_view(raw_state_opt.value(), h0_opt.value())),
                             "raw_state/h0 overlap is unsafe for CP or partial state views");
            if (raw_state_input_opt.has_value()) {
                sanity.tensor(raw_state_input_opt.value(), "raw_state_input");
                sanity.check(!sp::overlaps(raw_state_opt.value(),
                                           raw_state_input_opt.value()),
                             "raw_state_input must be a disjoint overlap snapshot");
            }
        }
    }
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16, "q must be fp16 or bf16");
    TORCH_CHECK(is_supported_head_config(H, Hg),
                "FlashQLA does not support head configuration (H=", H,
                ", Hg=", Hg, ")");
    TORCH_CHECK(q.dim() == 4, "q must be [1,T,Hg,128]");
    TORCH_CHECK(k.dim() == 4, "k must be [1,T,Hg,128]");
    TORCH_CHECK(v.dim() == 4, "v must be [1,T,H,128]");
    TORCH_CHECK(a.dim() == 4, "a must be [1,T,H,32]");
    TORCH_CHECK(beta.dim() == 3, "beta must be [1,T,H]");
    TORCH_CHECK(g.dim() == 3, "g must be [1,T,H]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.stride(3) == 1 && v.stride(2) == 128 &&
                    v.stride(1) >= H * 128,
                "v must be head-contiguous with a positive token stride");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");
    check_same_cuda_device(q, k, "k");
    check_same_cuda_device(q, v, "v");
    check_same_cuda_device(q, a, "a");
    check_same_cuda_device(q, beta, "beta");
    check_same_cuda_device(q, g, "g");
    check_same_cuda_device(q, cu_seqlens, "cu_seqlens");
    check_same_cuda_device(q, raw_cu_seqlens, "raw_cu_seqlens");
    check_same_b16_dtype(q, k, "k");
    check_same_b16_dtype(q, v, "v");
    check_same_b16_dtype(q, a, "a");
    TORCH_CHECK(g.dtype() == torch::kFloat32, "g must be fp32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be fp32");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(raw_cu_seqlens.dtype() == torch::kInt32, "raw_cu_seqlens int32");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.is_contiguous() &&
                cu_seqlens.numel() >= 2,
                "cu_seqlens must be contiguous int32 [cp_batch+1]");
    TORCH_CHECK(raw_cu_seqlens.dim() == 1 && raw_cu_seqlens.is_contiguous() &&
                raw_cu_seqlens.numel() >= 2,
                "raw_cu_seqlens must be contiguous int32 [raw_batch+1]");
    at::cuda::CUDAGuard device_guard(q.device());

    constexpr int DK = 128, DV = 128, CS = 32;
    int T = q.size(1);
    TORCH_CHECK(T > 0, "FlashQLA does not support an empty token dimension");
    TORCH_CHECK(valid_num_tokens > 0 && valid_num_tokens <= T,
                "valid_num_tokens must be in (0, q.size(1)], got ",
                valid_num_tokens, " for ", T, " physical tokens");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == Hg && q.size(3) == DK,
                "q must be [1,T,Hg,128]");
    TORCH_CHECK(k.size(0) == 1 && k.size(1) == T &&
                k.size(2) == Hg && k.size(3) == DK,
                "k must be [1,T,Hg,128] and match q's token dimension");
    TORCH_CHECK(v.size(0) == 1 && v.size(1) == T &&
                v.size(2) == H && v.size(3) == DV,
                "v must be [1,T,H,128] and match q's token dimension");
    TORCH_CHECK(a.size(0) == 1 && a.size(1) == T &&
                a.size(2) == H && a.size(3) == CS,
                "a must be [1,T,H,32] and match q's token dimension");
    TORCH_CHECK(beta.size(0) == 1 && beta.size(1) == T && beta.size(2) == H,
                "beta must be [1,T,H]");
    TORCH_CHECK(g.size(0) == 1 && g.size(1) == T && g.size(2) == H,
                "g must be [1,T,H]");
    int cp_batch = cu_seqlens.size(0) - 1;
    int raw_batch = raw_cu_seqlens.size(0) - 1;
    const int* cp_seq_map_ptr = nullptr;
    torch::Tensor cp_seq_map;
    if (cp_seq_map_opt.has_value()) {
        cp_seq_map = cp_seq_map_opt.value();
        check_same_cuda_device(q, cp_seq_map, "cp_seq_map");
        TORCH_CHECK(cp_seq_map.dtype() == torch::kInt32,
                    "cp_seq_map must be int32");
        TORCH_CHECK(cp_seq_map.is_contiguous(),
                    "cp_seq_map must be contiguous");
        TORCH_CHECK(cp_seq_map.dim() == 1 && cp_seq_map.numel() == cp_batch,
                    "cp_seq_map must be [cp_batch]");
        cp_seq_map_ptr = cp_seq_map.data_ptr<int>();
    } else {
        TORCH_CHECK(cp_batch == raw_batch,
                    "cp_seq_map may be omitted only for an identity plan");
    }
    TORCH_CHECK(raw_state_opt.has_value() == raw_state_indices_opt.has_value(),
                "raw_state and raw_state_indices must be provided together");
    TORCH_CHECK(num_warmup_opt.has_value() == fallback_mask_opt.has_value(),
                "num_warmup and fallback_mask must be provided together");
    TORCH_CHECK(!num_warmup_opt.has_value() || cp_seq_map_opt.has_value(),
                "overlap warmup metadata requires a CP sequence map");
    TORCH_CHECK(!raw_state_input_opt.has_value() ||
                    (raw_state_opt.has_value() && num_warmup_opt.has_value()),
                "raw_state_input is only valid for indexed overlap warmup");
    TORCH_CHECK(!num_warmup_opt.has_value() || !raw_state_opt.has_value() ||
                    raw_state_input_opt.has_value(),
                "indexed overlap warmup requires a read-only raw_state_input");
    TORCH_CHECK(!num_warmup_opt.has_value() || !h0_opt.has_value() ||
                    h0_opt->numel() == 0,
                "overlap warmup cannot consume precomputed CP h0");

    const int* num_warmup_ptr = nullptr;
    const signed char* fallback_mask_ptr = nullptr;
    torch::Tensor num_warmup;
    torch::Tensor fallback_mask;
    if (num_warmup_opt.has_value()) {
        num_warmup = num_warmup_opt.value();
        fallback_mask = fallback_mask_opt.value();
        check_same_cuda_device(q, num_warmup, "num_warmup");
        check_same_cuda_device(q, fallback_mask, "fallback_mask");
        TORCH_CHECK(num_warmup.dtype() == torch::kInt32 &&
                        num_warmup.is_contiguous() &&
                        num_warmup.numel() == cp_batch * H,
                    "num_warmup must be contiguous int32 [cp_batch*H]");
        TORCH_CHECK(fallback_mask.dtype() == torch::kInt8 &&
                        fallback_mask.is_contiguous() &&
                        fallback_mask.numel() == cp_batch * H,
                    "fallback_mask must be contiguous int8 [cp_batch*H]");
        num_warmup_ptr = num_warmup.data_ptr<int>();
        fallback_mask_ptr = reinterpret_cast<const signed char*>(
            fallback_mask.data_ptr<int8_t>());
    }

    const pla::instrument::PrefillRange profile_range(
        raw_batch, T, DK, DV, (int)Hg, (int)H, get_dtype_string(q),
        false, &raw_cu_seqlens);

    auto opt_h = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    // The compute kernels only visit the prefix covered by cu_seqlens. Allocate
    // the physical shape required by the framework; only its padded suffix
    // needs initialization.
    auto o = torch::empty({1, T, (int)H, DV}, opt_h);
    // The final state leaves the kernel in the pool's own dtype: bf16 halves the
    // resident state pool, fp32 keeps the historical contract.
    auto opt_state = torch::TensorOptions()
        .dtype(final_state_bf16 ? torch::kBFloat16 : torch::kFloat32)
        .device(q.device());
    // Indexed mode always updates raw_state in place. In the non-indexed path,
    // do not allocate or globally store a state that the public API will drop.
    auto ht_out = (!raw_state_opt.has_value() && output_final_state)
        ? torch::empty({raw_batch, (int)H, DK, DV}, opt_state)
        : torch::empty({0}, opt_state);

    const void* h0_ptr = nullptr;
    torch::Tensor h0;
    if (h0_opt.has_value() && h0_opt->numel() > 0) {
        h0 = h0_opt.value();
        check_same_cuda_device(q, h0, "h0");
        TORCH_CHECK(h0.dtype() == (final_state_bf16 ? torch::kBFloat16
                                                    : torch::kFloat32),
                    "h0 dtype must match the state pool selected by "
                    "final_state_bf16");
        TORCH_CHECK(h0.is_contiguous(), "h0 must be contiguous");
        TORCH_CHECK(h0.dim() == 4 && h0.size(0) >= cp_batch &&
                    h0.size(1) == H && h0.size(2) == DK && h0.size(3) == DV,
                    "h0 must be [num_states>=cp_batch,H,DK,DV]");
        h0_ptr = h0.data_ptr();
    }

    void* raw_state_ptr = nullptr;
    const void* raw_state_input_ptr = nullptr;
    const int* raw_state_indices_ptr = nullptr;
    torch::Tensor raw_state;
    torch::Tensor raw_state_input;
    torch::Tensor raw_state_indices;
    if (raw_state_opt.has_value()) {
        raw_state = raw_state_opt.value();
        check_same_cuda_device(q, raw_state, "raw_state");
        TORCH_CHECK(raw_state.dtype() == (final_state_bf16 ? torch::kBFloat16
                                                            : torch::kFloat32),
                    "raw_state dtype must match the state pool selected by "
                    "final_state_bf16");
        TORCH_CHECK(raw_state.is_contiguous(), "raw_state must be contiguous");
        TORCH_CHECK(raw_state.dim() == 4 && raw_state.size(0) > 0 &&
                    raw_state.size(1) == H &&
                    raw_state.size(2) == DV && raw_state.size(3) == DK,
                    "raw_state must be [num_slots,H,DV,DK]");
        raw_state_indices = raw_state_indices_opt.value();
        check_same_cuda_device(q, raw_state_indices, "raw_state_indices");
        TORCH_CHECK(raw_state_indices.dtype() == torch::kInt32,
                    "raw_state_indices must be int32");
        TORCH_CHECK(raw_state_indices.is_contiguous(),
                    "raw_state_indices must be contiguous");
        TORCH_CHECK(raw_state_indices.dim() == 1 &&
                    raw_state_indices.numel() == raw_batch,
                    "raw_state_indices must be [raw_batch]");
        raw_state_ptr = raw_state.data_ptr();
        raw_state_indices_ptr = raw_state_indices.data_ptr<int>();
    }
    if (raw_state_input_opt.has_value()) {
        raw_state_input = raw_state_input_opt.value();
        check_same_cuda_device(q, raw_state_input, "raw_state_input");
        TORCH_CHECK(raw_state_input.dtype() ==
                        (final_state_bf16 ? torch::kBFloat16 : torch::kFloat32),
                    "raw_state_input dtype must match raw_state");
        TORCH_CHECK(raw_state_input.is_contiguous(),
                    "raw_state_input must be contiguous");
        TORCH_CHECK(raw_state_input.dim() == 4 &&
                        raw_state_input.size(0) == raw_batch &&
                        raw_state_input.size(1) == H &&
                        raw_state_input.size(2) == DV &&
                        raw_state_input.size(3) == DK,
                    "raw_state_input must be [raw_batch,H,DV,DK]");
        raw_state_input_ptr = raw_state_input.data_ptr();
    }

    const auto device = logical_device_info(q);
    const hggcStream_t stream = pla::current_stream();
    if (valid_num_tokens < T) {
        const int64_t tail_offset = valid_num_tokens * H * DV * q.element_size();
        const int64_t tail_bytes =
            (int64_t(T) - valid_num_tokens) * H * DV * q.element_size();
        auto* tail = reinterpret_cast<unsigned char*>(data_ptr_b16(o)) + tail_offset;
        const hggcError_t status = hggcMemsetAsync(tail, 0, tail_bytes, stream);
        TORCH_CHECK(status == hggcSuccess,
                    "failed to zero FlashQLA padded output tail: ",
                    hggcGetErrorString(status));
    }
    const int* precision_flags_ptr = nullptr;
    if (precision_flags_opt.has_value()) {
        const auto& flags = precision_flags_opt.value();
        TORCH_CHECK(flags.device() == q.device() && flags.scalar_type() == torch::kInt &&
                    flags.is_contiguous() && flags.dim() == 2 &&
                    flags.size(0) == raw_batch && flags.size(1) == Hg,
                    "precision_flags must be contiguous device int32 [raw_batch,Hg]");
        TORCH_CHECK(!sp::overlaps(flags, o), "precision_flags must not overlap output");
        if (raw_state_opt.has_value())
            TORCH_CHECK(!sp::overlaps(flags, raw_state_opt.value()),
                        "precision_flags must not overlap mutable raw_state");
        precision_flags_ptr = flags.data_ptr<int>();
    }
    TORCH_CHECK(!skip_folded_repair_heads || (coefficients_opt && precision_flags_ptr),
                "skipping folded heads requires coefficients and precision flags");
    if (packed_pg_opt.has_value()) {
        TORCH_CHECK(gdn_packed_coefficients_tc01_enabled(),
                    "packed GDN requires the register-chain reordered schedule");
        const auto& pg = packed_pg_opt.value();
        check_same_cuda_device(q, pg, "packed_pg");
        TORCH_CHECK(device.compute_capability == 80 && raw_state_ptr != nullptr &&
                    cp_seq_map_ptr == nullptr && cp_batch == 1 && raw_batch == 1 &&
                    h0_ptr == nullptr && raw_state_input_ptr == nullptr &&
                    ((H == 8 && (Hg == 2 || Hg == 4) && T <= 512) ||
                     (H == 16 && Hg == 2)), "unsupported packed GDN main region");
        TORCH_CHECK(pg.scalar_type() == q.scalar_type() && pg.is_contiguous() &&
                    pg.sizes() == a.sizes(), "packed_pg must match contiguous A");
        launch_gdn_packed_fwd_tc01(data_ptr_b16(o),data_ptr_b16(q),data_ptr_b16(k),
            data_ptr_b16(v),data_ptr_b16(a),beta.data_ptr<float>(),g.data_ptr<float>(),
            pg.data_ptr(),raw_state_ptr,raw_state_indices_ptr,final_state_bf16,
            cu_seqlens.data_ptr<int>(),T,H,Hg,static_cast<float>(scale),dtype_tag(q),
            v.stride(1),precision_flags_ptr,device.index,stream);
    } else if (coefficients_opt.has_value()) {
        auto coefficients = coefficients_opt.value();
        check_folded_coefficients(q, coefficients, H);
        TORCH_CHECK(raw_state_ptr && raw_state_indices_ptr &&
                    !final_state_bf16 && !num_warmup_ptr,
                    "folded forward requires an indexed FP32 state pool without overlap replay");
        launch_folded_fwd_cute(o.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
            coefficients.data_ptr(), cu_seqlens.data_ptr<int>(), cp_seq_map_ptr,
            raw_cu_seqlens.data_ptr<int>(), static_cast<const float*>(h0_ptr),
            static_cast<float*>(raw_state_ptr), raw_state_indices_ptr, cp_batch, T,
            raw_batch, device.index, (int)H, (int)Hg, v.stride(1), (float)scale,
            skip_folded_repair_heads ? precision_flags_ptr : nullptr, stream);
    } else {
    launch_fused_fwd_cp_cute(
        data_ptr_b16(o), data_ptr_b16(q), data_ptr_b16(k), data_ptr_b16(v), data_ptr_b16(a),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        h0_ptr, ht_out.numel() > 0 ? ht_out.data_ptr() : nullptr,
        raw_state_ptr, raw_state_input_ptr, final_state_bf16,
        raw_state_indices_ptr, cu_seqlens.data_ptr<int>(), cp_seq_map_ptr,
        raw_cu_seqlens.data_ptr<int>(), num_warmup_ptr, fallback_mask_ptr,
        cp_batch, T, raw_batch, (int)H, (int)Hg, (float)scale, dtype_tag(q),
        v.stride(1), precision_flags_ptr,
        device.index, device.num_cu, device.compute_capability, stream);
    }

    return {o, ht_out};
}

// Structural selection and FP64 recovery keep scratch local to a call/Graph.
// A missing selection tensor retains the private force-all diagnostic mode.
torch::Tensor precision_guard(torch::Tensor k, torch::Tensor cu, bool include_empty,
    c10::optional<torch::Tensor> beta, bool complete_layout) {
    TORCH_CHECK(k.is_cuda() && k.dim() == 4 && k.size(0) == 1 && k.size(3) == 128 &&
                k.is_contiguous() && (k.scalar_type() == torch::kHalf ||
                k.scalar_type() == torch::kBFloat16),
                "precision guard k must be contiguous FP16/BF16 [1,T,Hg,128]");
    TORCH_CHECK(cu.device() == k.device() && cu.scalar_type() == torch::kInt &&
                cu.is_contiguous() && cu.dim() == 1 && cu.numel() >= 2,
                "precision guard cu must be contiguous device int32 [N+1]");
    const int64_t t = k.size(1), hg = k.size(2), n = cu.numel()-1;
    TORCH_CHECK(t > 0 && t <= INT32_MAX && hg > 0 && n*hg <= INT32_MAX,
                "invalid precision guard extent");
    if (beta) TORCH_CHECK(beta->device() == k.device() &&
        beta->scalar_type() == torch::kFloat && beta->is_contiguous() &&
        beta->dim() == 3 && beta->size(0) == 1 && beta->size(1) == t &&
        beta->size(2) > 0 && beta->size(2) <= INT32_MAX && beta->size(2)%hg == 0,
        "precision guard beta must be contiguous FP32 [1,T,H]");
    const c10::cuda::CUDAGuard guard(k.device());
    const auto device = logical_device_info(k);
    const int64_t heads = beta ? beta->size(2) : 0;
    const int64_t guard_tasks = n * hg;
    const bool tc01_short_windows = n > 1 && heads >= 24 && hg <= 8 &&
        guard_tasks <= device.num_cu && t >= 256 && t <= 1024 && t >= n * 64;
    const bool tc01_two_guard_waves = n <= 8 && guard_tasks <= 2 * device.num_cu &&
        t >= n * 512;
    const bool tc01_long_windows = hg <= 16 && t >= 2048 && t >= n * 256 && t <= 32831 &&
        (guard_tasks <= device.num_cu || tc01_two_guard_waves || (hg <= 4 && n <= 64 && t >= 8192));
    // The window grid spans T. The host plan knows whether T includes padding.
    const bool tc01_windows = complete_layout &&
        device.compute_capability == 80 && device.num_cu == 64 &&
        (tc01_short_windows || tc01_long_windows);
    // One CTA per sequence and key head can underfill M890P on a long packed batch.
    const bool tc02_underfilled_packed = k.scalar_type() == torch::kBFloat16 && beta &&
        n > 3 && t >= 8192 &&
        (guard_tasks <= device.num_cu ||
         (complete_layout && heads == 8 && hg == 2 && n >= 40 && n <= 48 &&
          t >= 16384 && t <= 20480));
    const bool tc02_windows = device.compute_capability == 89 && device.num_cu == 39 &&
        (n <= 3 || tc02_underfilled_packed ||
         ((n == 16 || n == 32 || n == 64) && heads == 8 && (hg == 2 || hg == 4))) &&
        hg <= 16 && ((t >= 1024 && t <= 32831) ||
        (n == 1 && heads == 16 && hg == 4 && t >= 768 && t < 1024));
    auto flags = torch::empty({n,hg}, k.options().dtype(torch::kInt));
    if (!pla::bypass::enabled()) {
        flashqla_precision::launch_guard(k.data_ptr(),cu.data_ptr<int32_t>(),
            flags.data_ptr<int32_t>(),int(t),int(hg),int(n),dtype_tag(k),include_empty,
            pla::current_stream(k.get_device()),
            beta ? beta->data_ptr<float>() : nullptr, int(heads),
            tc01_windows || tc02_windows);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return flags;
}

torch::Tensor precision_snapshot(torch::Tensor initial,
    c10::optional<torch::Tensor> indices, torch::Tensor cu, int64_t heads,
    c10::optional<torch::Tensor> flags) {
    TORCH_CHECK(initial.is_cuda() && initial.dim() == 4 && initial.is_contiguous() &&
                (initial.scalar_type() == torch::kFloat || initial.scalar_type() == torch::kBFloat16) &&
                initial.size(0) > 0 && initial.size(0) <= INT32_MAX &&
                initial.size(1) == heads && initial.size(2) == 128 && initial.size(3) == 128,
                "repair initial state must be contiguous FP32/BF16 [slots,H,128,128]");
    TORCH_CHECK(cu.device() == initial.device() && cu.scalar_type() == torch::kInt &&
                cu.is_contiguous() && cu.dim() == 1 && cu.numel() >= 2,
                "repair cu must be contiguous device int32 [N+1]");
    const int64_t n = cu.numel() - 1;
    TORCH_CHECK(heads > 0 && n * heads <= INT32_MAX / 32,
                "repair task count overflows int32");
    if (flags) TORCH_CHECK(flags->device() == initial.device() &&
        flags->scalar_type() == torch::kInt && flags->is_contiguous() &&
        flags->dim() == 2 && flags->size(0) == n && flags->size(1) > 0 &&
        heads % flags->size(1) == 0, "invalid precision snapshot flags");
    if (indices) {
        TORCH_CHECK(indices->device() == initial.device() &&
        indices->scalar_type() == torch::kInt && indices->is_contiguous() &&
        indices->dim() == 1 && indices->numel() == n,
        "repair indices must be contiguous device int32 [N]");
    } else {
        TORCH_CHECK(initial.size(0) >= n, "repair legacy state has too few sequences");
    }
    const c10::cuda::CUDAGuard guard(initial.device());
    const pla::sanity::Context sanity("flashqla.precision_snapshot", initial);
    if (sanity && indices) {
        const auto raw = sp::offsets(sanity, cu, INT32_MAX);
        sp::slots(sanity, indices.value(), n, initial.size(0),
                  "initial_state_indices", true, raw);
    }
    const auto snapshot_device = logical_device_info(initial);
    const bool allow_tiled = snapshot_device.compute_capability == 89 && snapshot_device.num_cu == 39;
    auto result = torch::empty({n, heads, 128, 128}, initial.options().dtype(torch::kFloat));
    if (!pla::bypass::enabled()) {
        flashqla_precision::launch_snapshot(initial.data_ptr(), result.data_ptr<float>(),
            indices ? indices->data_ptr<int32_t>() : nullptr, cu.data_ptr<int32_t>(), int(n), int(heads),
            int(initial.size(0)), initial.scalar_type() == torch::kBFloat16,
            pla::current_stream(initial.get_device()),
            flags ? flags->data_ptr<int32_t>() : nullptr, flags ? int(flags->size(1)) : 0,
            allow_tiled);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return result;
}

static bool tc02_quick_guard_shape(int64_t t, int64_t heads, int64_t hg) {
    if (hg <= 0 || hg > 128 || (hg & (hg - 1)) != 0 || heads % hg != 0) return false;
    return ((heads == 8 || heads == 16 || heads == 24 || heads == 32) && t > 0 && t <= 640) ||
           ((heads == 48 || heads == 64 || heads == 128) && t > 0 && t <= 192) ||
           (heads == 64 && (hg == 8 || hg == 16) && t > 192 && t <= 256) ||
           (heads == 8 && hg == 2 && t >= 768 && t < 1024);
}

std::vector<torch::Tensor> precision_guard_snapshot_short(
    torch::Tensor k, torch::Tensor cu, bool include_empty,
    c10::optional<torch::Tensor> beta, torch::Tensor initial,
    c10::optional<torch::Tensor> indices, int64_t heads) {
    TORCH_CHECK(k.is_cuda() && k.dim() == 4 && k.size(0) == 1 && k.size(3) == 128 &&
        k.is_contiguous() && (k.scalar_type() == torch::kHalf || k.scalar_type() == torch::kBFloat16),
        "short guard k must be contiguous FP16/BF16 [1,T,Hg,128]");
    TORCH_CHECK(cu.device() == k.device() && cu.scalar_type() == torch::kInt &&
        cu.is_contiguous() && cu.dim() == 1 && cu.numel() == 2,
        "short guard cu must be device int32 [2]");
    const int64_t t = k.size(1), hg = k.size(2);
    TORCH_CHECK(tc02_quick_guard_shape(t, heads, hg), "invalid tc02 single-sequence guard shape");
    if (beta) TORCH_CHECK(beta->device() == k.device() && beta->scalar_type() == torch::kFloat &&
        beta->is_contiguous() && beta->dim() == 3 && beta->size(0) == 1 &&
        beta->size(1) == t && beta->size(2) == heads, "invalid short guard beta");
    TORCH_CHECK(initial.device() == k.device() && initial.dim() == 4 && initial.is_contiguous() &&
        (initial.scalar_type() == torch::kFloat || initial.scalar_type() == torch::kBFloat16) &&
        initial.size(0) > 0 && initial.size(0) <= INT32_MAX && initial.size(1) == heads &&
        initial.size(2) == 128 && initial.size(3) == 128, "invalid short guard initial state");
    if (indices) TORCH_CHECK(indices->device() == k.device() && indices->scalar_type() == torch::kInt &&
        indices->is_contiguous() && indices->dim() == 1 && indices->numel() == 1,
        "short guard indices must be device int32 [1]");
    const c10::cuda::CUDAGuard guard(k.device());
    const auto device = logical_device_info(k);
    TORCH_CHECK(device.compute_capability == 89 && device.num_cu == 39,
        "private short guard requires tc02 with 39 CUs");
    const pla::sanity::Context sanity("flashqla.precision_guard_snapshot_short", initial);
    if (sanity && indices) {
        const auto raw = sp::offsets(sanity, cu, t);
        sp::slots(sanity, indices.value(), 1, initial.size(0), "initial_state_indices", true, raw);
    }
    auto flags = torch::empty({1,hg}, k.options().dtype(torch::kInt));
    auto snapshot = torch::empty({1,heads,128,128}, initial.options().dtype(torch::kFloat));
    if (!pla::bypass::enabled()) {
        flashqla_precision::launch_guard_snapshot_short(k.data_ptr(),cu.data_ptr<int32_t>(),
            flags.data_ptr<int32_t>(),int(t),int(hg),dtype_tag(k),include_empty,
            beta ? beta->data_ptr<float>() : nullptr,int(heads),initial.data_ptr(),snapshot.data_ptr<float>(),
            indices ? indices->data_ptr<int32_t>() : nullptr,int(initial.size(0)),
            initial.scalar_type() == torch::kBFloat16,pla::current_stream(k.get_device()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {flags,snapshot};
}

std::vector<torch::Tensor> precision_guard_snapshot_medium(
    torch::Tensor k, torch::Tensor cu, bool include_empty,
    c10::optional<torch::Tensor> beta, torch::Tensor initial,
    c10::optional<torch::Tensor> indices, int64_t heads) {
    TORCH_CHECK(k.is_cuda() && k.dim() == 4 && k.size(0) == 1 && k.size(3) == 128 &&
        k.is_contiguous() && (k.scalar_type() == torch::kHalf || k.scalar_type() == torch::kBFloat16),
        "short guard k must be contiguous FP16/BF16 [1,T,Hg,128]");
    TORCH_CHECK(cu.device() == k.device() && cu.scalar_type() == torch::kInt &&
        cu.is_contiguous() && cu.dim() == 1 && cu.numel() >= 2 && cu.numel() <= 33,
        "packed guard requires device int32 offsets for one to 32 sequences");
    const int64_t t = k.size(1), hg = k.size(2);
    const int sequences = int(cu.numel() - 1);
    const bool compact = (sequences == 11 || sequences == 22) &&
                         heads == 8 && (hg == 2 || hg == 4);
    const bool medium = compact ||
        (sequences == 32 && t == 4964 && heads == 48 && hg == 16) ||
        (sequences == 2 && ((t == 920 && heads == 8 && hg == 4) ||
                            (t == 1041 && heads == 8 && hg == 2))) ||
        (sequences == 1 && heads == 16 && ((hg == 2 && t == 847) || (hg == 4 && t == 823))) ||
        (sequences == 1 && heads == 8 && (hg == 2 || hg == 4) && t >= 768 && t <= 1023);
    TORCH_CHECK(hg > 0 && heads % hg == 0 && medium,
        "unsupported tc02 packed guard shape");
    if (beta) TORCH_CHECK(beta->device() == k.device() && beta->scalar_type() == torch::kFloat &&
        beta->is_contiguous() && beta->dim() == 3 && beta->size(0) == 1 &&
        beta->size(1) == t && beta->size(2) == heads, "invalid short guard beta");
    TORCH_CHECK(initial.device() == k.device() && initial.dim() == 4 && initial.is_contiguous() &&
        (initial.scalar_type() == torch::kFloat || initial.scalar_type() == torch::kBFloat16) &&
        initial.size(0) > 0 && initial.size(0) <= INT32_MAX && initial.size(1) == heads &&
        initial.size(2) == 128 && initial.size(3) == 128, "invalid short guard initial state");
    TORCH_CHECK(indices || initial.size(0) >= sequences, "dense state is smaller than the sequence count");
    if (indices) TORCH_CHECK(indices->device() == k.device() && indices->scalar_type() == torch::kInt &&
        indices->is_contiguous() && indices->dim() == 1 && indices->numel() == sequences,
        "short guard indices must be device int32 [N]");
    const c10::cuda::CUDAGuard guard(k.device());
    const auto device = logical_device_info(k);
    TORCH_CHECK(device.compute_capability == 89 && device.num_cu == 39,
        "private short guard requires tc02 with 39 CUs");
    const pla::sanity::Context sanity("flashqla.precision_guard_snapshot_medium", initial);
    if (sanity && indices) {
        const auto raw = sp::offsets(sanity, cu, t);
        sp::slots(sanity, indices.value(), sequences, initial.size(0), "initial_state_indices", true, raw);
    }
    auto flags = torch::empty({sequences,hg}, k.options().dtype(torch::kInt));
    auto snapshot = torch::empty({sequences,heads,128,128}, initial.options().dtype(torch::kFloat));
    if (!pla::bypass::enabled()) {
        flashqla_precision::launch_guard_snapshot_medium(k.data_ptr(),cu.data_ptr<int32_t>(),
            flags.data_ptr<int32_t>(),int(t),int(hg),sequences,dtype_tag(k),include_empty,
            beta ? beta->data_ptr<float>() : nullptr,int(heads),initial.data_ptr(),snapshot.data_ptr<float>(),
            indices ? indices->data_ptr<int32_t>() : nullptr,int(initial.size(0)),
            initial.scalar_type() == torch::kBFloat16,pla::current_stream(k.get_device()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {flags,snapshot};
}

void precision_repair(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor beta, c10::optional<torch::Tensor> g,
    c10::optional<torch::Tensor> raw_a, c10::optional<torch::Tensor> a_log,
    c10::optional<torch::Tensor> dt_bias, torch::Tensor out,
    torch::Tensor snapshot, c10::optional<torch::Tensor> final,
    torch::Tensor cu, c10::optional<torch::Tensor> indices, double scale,
    int dv_per_warp, int staged_tokens, c10::optional<torch::Tensor> flags,
    bool fuse_suffix, bool load_indexed_state, c10::optional<torch::Tensor> live_initial) {
    if (pla::bypass::enabled()) return;
    TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.size(0) == 1 && q.size(3) == 128 &&
                q.is_contiguous() && (q.scalar_type() == torch::kHalf || q.scalar_type() == torch::kBFloat16),
                "repair q must be contiguous FP16/BF16 [1,T,Hg,128]");
    const auto device = q.device();
    const int64_t t = q.size(1), hg = q.size(2);
    TORCH_CHECK(t <= INT32_MAX && hg > 0 && k.device() == device &&
                k.scalar_type() == q.scalar_type() && k.sizes() == q.sizes() && k.is_contiguous(),
                "repair k must match q");
    TORCH_CHECK(v.device() == device && v.scalar_type() == q.scalar_type() &&
                v.dim() == 4 && v.size(0) == 1 && v.size(1) == t && v.size(3) == 128 &&
                v.stride(3) == 1 && v.stride(2) == 128 && v.stride(1) >= v.size(2) * 128,
                "repair v must be head-contiguous [1,T,H,128]");
    const int64_t h = v.size(2);
    TORCH_CHECK(h > 0 && h <= INT32_MAX && h % hg == 0 &&
                out.device() == device && out.scalar_type() == q.scalar_type() &&
                out.sizes() == v.sizes() && out.is_contiguous(), "invalid repair output or head mapping");
    TORCH_CHECK(beta.device() == device && beta.scalar_type() == torch::kFloat &&
                beta.is_contiguous() && beta.dim() == 3 && beta.size(0) == 1 &&
                beta.size(1) == t && beta.size(2) == h, "repair beta must be FP32 [1,T,H]");
    TORCH_CHECK(cu.device() == device && cu.scalar_type() == torch::kInt &&
                cu.is_contiguous() && cu.dim() == 1 && cu.numel() >= 2,
                "repair cu must be contiguous int32 [N+1]");
    const int64_t n = cu.numel() - 1;
    TORCH_CHECK(dv_per_warp == 1 || dv_per_warp == 2 || dv_per_warp == 4, "repair DV per warp must be 1, 2 or 4");
    TORCH_CHECK(staged_tokens == 0 || staged_tokens == 32, "repair staged tokens must be 0 or 32");
    TORCH_CHECK(n * h <= INT32_MAX / 32 && std::isfinite(scale), "invalid repair task count/scale");
    if (flags) {
        TORCH_CHECK(flags->device() == device && flags->scalar_type() == torch::kInt &&
                    flags->is_contiguous() && flags->dim() == 2 &&
                    flags->size(0) == n && flags->size(1) == hg,
                    "repair flags must be contiguous int32 [N,Hg]");
        TORCH_CHECK(!sp::overlaps(flags.value(), out) &&
                    (!final || !sp::overlaps(flags.value(),final.value())),
                    "repair flags must not overlap mutable outputs");
    }
    TORCH_CHECK(snapshot.device() == device && snapshot.scalar_type() == torch::kFloat &&
                snapshot.is_contiguous() && (snapshot.numel() == 0 || snapshot.numel() == n*h*128*128),
                "repair snapshot must be dense FP32 [N,H,DV,DK]");
    if (indices) TORCH_CHECK(indices->device() == device && indices->scalar_type() == torch::kInt &&
        indices->is_contiguous() && indices->dim() == 1 && indices->numel() == n,
        "repair indices must be contiguous int32 [N]");
    if (g) {
        TORCH_CHECK(g->device() == device && g->scalar_type() == torch::kFloat &&
            g->is_contiguous() && g->sizes() == beta.sizes(), "repair g must match activated beta layout");
    } else {
        TORCH_CHECK(raw_a && a_log && dt_bias, "repair requires activated g or all raw gate inputs");
        TORCH_CHECK(raw_a->device() == device && raw_a->scalar_type() == q.scalar_type() &&
            raw_a->dim() == 2 && raw_a->size(0) == t && raw_a->size(1) == h &&
            raw_a->stride(1) == 1 && raw_a->stride(0) >= h, "invalid repair raw gate layout");
        for (const auto* x : {&a_log, &dt_bias}) TORCH_CHECK(x->value().device() == device &&
            x->value().scalar_type() == torch::kFloat && x->value().is_contiguous() &&
            x->value().dim() == 1 && x->value().numel() == h, "repair gate parameters must be FP32 [H]");
    }
    if (final) TORCH_CHECK(final->device() == device && final->is_contiguous() &&
        (final->scalar_type() == torch::kFloat || final->scalar_type() == torch::kBFloat16) &&
        final->dim() == 4 && final->size(0) > 0 && final->size(0) <= INT32_MAX &&
        (indices || final->size(0) == n) && final->size(1) == h &&
        final->size(2) == 128 && final->size(3) == 128, "invalid repair final state");
    if (live_initial) {
        const auto& state = *live_initial;
        TORCH_CHECK(flags && snapshot.numel() == 0 && !load_indexed_state &&
                    state.device() == device && state.is_contiguous() &&
                    (state.scalar_type() == torch::kFloat || state.scalar_type() == torch::kBFloat16) &&
                    state.dim() == 4 && state.size(0) > 0 && state.size(0) <= INT32_MAX &&
                    (indices || state.size(0) >= n) && state.size(1) == h &&
                    state.size(2) == 128 && state.size(3) == 128 &&
                    !sp::overlaps(state, out), "invalid live precision state");
        TORCH_CHECK(indices ? final && sp::same_view(state, *final)
                            : !final || !sp::overlaps(state, *final),
                    "live precision state must retain the original carry");
    }
    flashqla_precision::Arguments a{};
    a.q=q.data_ptr(); a.k=k.data_ptr(); a.v=v.data_ptr(); a.beta=beta.data_ptr<float>();
    a.g=g ? g->data_ptr<float>() : nullptr;
    a.raw_a=raw_a ? raw_a->data_ptr() : nullptr;
    a.a_log=a_log ? a_log->data_ptr<float>() : nullptr;
    a.dt_bias=dt_bias ? dt_bias->data_ptr<float>() : nullptr;
    a.snapshot=snapshot.numel() ? snapshot.data_ptr<float>() : nullptr;
    if (live_initial) {
        a.live_initial = live_initial->data_ptr();
        a.initial_slots = int(live_initial->size(0));
        a.initial_bf16 = live_initial->scalar_type() == torch::kBFloat16;
    }
    a.cu=cu.data_ptr<int32_t>(); a.indices=indices ? indices->data_ptr<int32_t>() : nullptr;
    a.flags=flags ? flags->data_ptr<int32_t>() : nullptr;
    a.out=out.data_ptr(); a.final_state=final ? final->data_ptr() : nullptr;
    a.tokens=int(t); a.heads=int(h); a.key_heads=int(hg); a.sequences=int(n);
    a.dv_per_warp=dv_per_warp;
    a.staged_tokens=staged_tokens;
    a.final_slots=final ? int(final->size(0)) : 0;
    a.v_token_stride=v.stride(1); a.a_token_stride=raw_a ? raw_a->stride(0) : 0;
    a.scale=scale; a.indexed=indices.has_value();
    a.state_bf16=final && final->scalar_type() == torch::kBFloat16;
    const c10::cuda::CUDAGuard guard(device);
    bool suffix_shape = false;
    if (fuse_suffix) {
        const auto repair_device = logical_device_info(q);
        suffix_shape = repair_device.compute_capability == 89 && repair_device.num_cu == 39 &&
            ((n == 1 && (tc02_quick_guard_shape(t, h, hg) ||
                (h == 8 && (hg == 2 || hg == 4) && t >= 768 && t <= 1023) ||
                (h == 16 && (hg == 2 || hg == 4 || hg == 8) && t >= 768 && t <= 1408))) ||
             (n == 2 && ((t == 920 && h == 8 && hg == 4) ||
                         (t == 1041 && h == 8 && hg == 2))) ||
             ((n == 11 || n == 22) && h == 8 && (hg == 2 || hg == 4)));
    }
    // Opt-in only: the old native entry permits aliases. Preserve its ordered
    // two-launch behavior whenever suffix stores could overlap any reader/state.
    bool suffix_disjoint = false;
    if (fuse_suffix && suffix_shape) {
        suffix_disjoint = !sp::overlaps(out, q) && !sp::overlaps(out, k) &&
            !sp::overlaps(out, v) && !sp::overlaps(out, beta) &&
            !sp::overlaps(out, snapshot) && !sp::overlaps(out, cu);
        for (const auto* input : {&g, &raw_a, &a_log, &dt_bias, &final, &indices, &flags, &live_initial})
            if (input->has_value()) suffix_disjoint = suffix_disjoint && !sp::overlaps(out, input->value());
    }
    a.fuse_suffix = fuse_suffix && suffix_shape && suffix_disjoint;
    TORCH_CHECK(!load_indexed_state || (n == 1 && t > 0 && t <= 3 && final && indices &&
        !snapshot.numel() && !flags && a.fuse_suffix), "invalid tiny indexed repair arguments");
    a.load_indexed_state = load_indexed_state;
    int repair_block_cap = 256;
    if (staged_tokens == 32 && h == 8 &&
        ((n == 64 && (hg == 2 || hg == 4)) ||
         (hg == 2 && a.live_initial && n >= 40 && n <= 48 && t >= 16384 && t <= 20480))) {
        const auto n64_device = logical_device_info(q);
        if (n64_device.compute_capability == 89 && n64_device.num_cu == 39)
            repair_block_cap = 2048;
    }
    flashqla_precision::launch_repair(a, dtype_tag(q), pla::current_stream(q.get_device()),
                                    repair_block_cap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<ChunkMetadataEntry, std::shared_ptr<ChunkMetadataEntry>>(m, "_ChunkMetadataEntry")
        .def("activate", &activate_chunk_metadata_entry)
        .def("prepare_for_capture", &prepare_chunk_metadata_for_capture,
             pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("_make_chunk_metadata_entry", &make_chunk_metadata_entry);
    m.def("_make_plan_metadata_entry", &make_metadata_bundle_entry);
    m.def("_metadata_owner_stats", &chunk_metadata_owner_stats);
    m.def("_metadata_owner_wait", [](uint64_t count, unsigned timeout_ms) {
        pybind11::gil_scoped_release release;
        return pla_graph_owner_experiment::owner().wait_for_releases(count, timeout_ms);
    });
    m.def("_metadata_owner_shutdown", [] {
        pybind11::gil_scoped_release release;
        pla_graph_owner_experiment::owner().shutdown_before_runtime_teardown();
    });

    (void)pla::bypass::enabled();
    (void)pla::instrument::show_params();
    (void)pla::instrument::perf_instrument();

    m.def("_precision_guard", &precision_guard,
          py::arg("k"), py::arg("cu"), py::arg("include_empty") = false,
          py::arg("beta") = c10::nullopt, py::arg("complete_layout") = false);
    m.def("_precision_guard_snapshot_short", &precision_guard_snapshot_short,
          py::arg("k"), py::arg("cu"), py::arg("include_empty"), py::arg("beta"),
          py::arg("initial"), py::arg("indices"), py::arg("heads"));
    m.def("_precision_guard_snapshot_medium", &precision_guard_snapshot_medium,
          py::arg("k"), py::arg("cu"), py::arg("include_empty"), py::arg("beta"),
          py::arg("initial"), py::arg("indices"), py::arg("heads"));
    m.def("_precision_snapshot", &precision_snapshot,
          py::arg("initial"), py::arg("indices"), py::arg("cu"), py::arg("heads"),
          py::arg("flags") = c10::nullopt);
    m.def("_precision_repair", &precision_repair,
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("beta"), py::arg("g"),
          py::arg("raw_a"), py::arg("a_log"), py::arg("dt_bias"), py::arg("out"),
          py::arg("snapshot"), py::arg("final"), py::arg("cu"), py::arg("indices"),
          py::arg("scale"), py::arg("dv_per_warp") = 1, py::arg("staged_tokens") = 0,
          py::arg("flags") = c10::nullopt, py::arg("fuse_suffix") = false,
          py::arg("load_indexed_state") = false,
          py::arg("live_initial") = c10::nullopt);
    m.def("_precision_bypass_enabled", [] { return pla::bypass::enabled(); });
    m.def("chunk_local_cumsum", &chunk_local_cumsum,
          "FlashQLA CuTe chunk-local cumsum of gate",
          py::arg("g"), py::arg("cu_seqlens"), py::arg("chunk_indices"));
    m.def("chunk_local_cumsum_qk_norm", &chunk_local_cumsum_qk_norm,
          "FlashQLA fused raw q/k L2 normalization and gate cumsum",
          py::arg("g"), py::arg("cu_seqlens"), py::arg("chunk_indices"),
          py::arg("q"), py::arg("k"));
    m.def("chunk_local_cumsum_qk_norm_gated",
          &chunk_local_cumsum_qk_norm_gated,
          "FlashQLA fused raw GDN gating, cumsum, and q/k L2 normalization",
          py::arg("a"), py::arg("b"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("cu_seqlens"), py::arg("chunk_indices"),
          py::arg("q"), py::arg("k"), py::arg("valid_num_tokens") = -1,
          py::arg("folded") = false);
    m.def("get_warmup_chunks", &get_warmup_chunks,
          "FlashQLA CuTe get_warmup_chunks (Task 2)",
          py::arg("g"), py::arg("ht_mask"), py::arg("cu_seqlens"),
          py::arg("threshold") = -10.0);
    m.def("kkt_solve", &kkt_solve,
          "FlashQLA CuTe kkt_solve (Task 3)",
          py::arg("K"), py::arg("beta"), py::arg("cu_seqlens"), py::arg("H"),
          py::arg("chunk_indices") = c10::optional<torch::Tensor>());
    m.def("_packed_coefficients_tc01", &packed_coefficients_tc01,
          "tc01 independent packed coefficients with optional short KKT fusion",
          py::arg("q"), py::arg("k"), py::arg("beta"), py::arg("g"),
          py::arg("cu_seqlens"), py::arg("chunk_indices"), py::arg("H"),
          py::arg("scale"), py::arg("a") = c10::optional<torch::Tensor>());
    m.attr("_packed_coefficients_tc01_enabled") =
        py::bool_(gdn_packed_coefficients_tc01_enabled());
    m.def("folded_coefficients", &folded_coefficients,
          "FlashQLA M890P folded coefficient workspace",
          py::arg("q"), py::arg("k"), py::arg("beta"), py::arg("g"),
          py::arg("cu_seqlens"), py::arg("chunk_indices"), py::arg("scale"), py::arg("precise") = false);
    m.def("correct_h0", &correct_h0,
          "FlashQLA CuTe correct_h0; raw_h0_indices must index raw_h0 slots",
          py::arg("ht_buffer"), py::arg("mt_buffer"), py::arg("fallback_mask"),
          py::arg("seq_map_r2c"), py::arg("raw_batch_size"),
          py::arg("raw_h0") = c10::optional<torch::Tensor>(),
          py::arg("raw_h0_indices") = c10::optional<torch::Tensor>(),
          py::arg("raw_h0_k_last") = false,
          py::arg("out_bf16") = false, py::arg("folded") = false,
          py::arg("seeded_cp_h0") = c10::optional<torch::Tensor>());
    m.def("_correct_h0_selected_skip", &correct_h0_selected_skip,
          "Private selected-head scratch skip; only flagged fused/repair may consume",
          py::arg("ht_buffer"), py::arg("mt_buffer"), py::arg("fallback_mask"),
          py::arg("seq_map_r2c"), py::arg("raw_batch_size"), py::arg("raw_h0"),
          py::arg("raw_h0_indices"), py::arg("raw_h0_k_last"), py::arg("out_bf16"),
          py::arg("precision_flags"), py::arg("key_heads"));
    m.def("_prepare_h_selected_t1_skip", &prepare_h_selected_t1_skip,
          "Private selected T1 prepare; selected scratch remains unwritten",
          py::arg("k"), py::arg("v"), py::arg("a"), py::arg("g"), py::arg("beta"),
          py::arg("cu_seqlens"), py::arg("num_warmup_chunks"),
          py::arg("H"), py::arg("Hg"), py::arg("precision_flags"),
          py::arg("cp_to_raw"), py::arg("raw_to_cp"), py::arg("raw_cu_seqlens"));
    m.def("prepare_h", &prepare_h,
          "FlashQLA CuTe prepare_h (Task 5)",
          py::arg("k"), py::arg("v"), py::arg("a"), py::arg("g"), py::arg("beta"),
          py::arg("cu_seqlens"), py::arg("num_warmup_chunks"),
          py::arg("H"), py::arg("Hg"),
          py::arg("coefficients") = c10::optional<torch::Tensor>(),
          py::arg("precise") = false, py::arg("skip_last_segment") = false,
          py::arg("seeded_cp_h0") = c10::optional<torch::Tensor>(),
          py::arg("initial_pool") = c10::optional<torch::Tensor>(),
          py::arg("first_slot") = c10::optional<torch::Tensor>());
    m.def("fused_fwd", &fused_fwd,
          "FlashQLA CuTe fused_fwd; raw_state and raw_state_indices are paired",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("a"),
          py::arg("beta"), py::arg("g"), py::arg("cu_seqlens"),
          py::arg("cp_seq_map"), py::arg("raw_cu_seqlens"),
          py::arg("valid_num_tokens"), py::arg("H"), py::arg("Hg"),
          py::arg("scale"),
          py::arg("h0") = c10::optional<torch::Tensor>(),
          py::arg("raw_state") = c10::optional<torch::Tensor>(),
          py::arg("raw_state_indices") = c10::optional<torch::Tensor>(),
          py::arg("output_final_state") = true,
          py::arg("final_state_bf16") = false,
          py::arg("raw_state_input") = c10::optional<torch::Tensor>(),
          py::arg("num_warmup") = c10::optional<torch::Tensor>(),
          py::arg("fallback_mask") = c10::optional<torch::Tensor>(),
          py::arg("packed_pg") = c10::optional<torch::Tensor>(),
          py::arg("precision_flags") = c10::optional<torch::Tensor>(),
          py::arg("coefficients") = c10::optional<torch::Tensor>(),
          py::arg("skip_folded_repair_heads") = false);
}
