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
 * FlashKDApro launch interface shared by the extension host and kernel TUs.
 *
 * Keep launcher declarations here rather than duplicating forward declarations
 * at call sites.  This is an internal C++ ABI; the Python binding remains the
 * public extension surface.
 */
#pragma once
#include "conditioned_api.h"

#include <hggc_runtime.h>

#include <cstdint>

#include "prefill/flashkdapro/common/workspace_abi.h"

// q_stride, k_stride and v_stride are token strides in elements; the head and
// feature dimensions of q/k/v are always packed ([T, H, 128] with strides
// (stride, 128, 1)).  Dense inputs pass H * 128.
void launch_flashkda_prepare(
    const void* q, const void* k, const void* g_in, const void* beta,
    const float* A_log, const float* dt_bias, float scale, float gate_scale,
    int T_total, int H, int beta_stride, int64_t q_stride, int64_t k_stride,
    int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen,
    int tile_offset, int tile_count, int head_offset, int head_count,
    int* ready, void* ws, bool force_plain, bool cp_fma_front,
    bool fp16_kd_qd, flashkda::KrWorkspaceLayout kr_layout,
    hggcStream_t stream, const void* value_input = nullptr, bool short_k1_lb8 = false);

namespace flashkda {
// Optional same-stream classification work for the homogeneous CP pass.
// Keys and request mapping remain owned by the public forward call.
inline constexpr int KDA_BF16_SERIAL_ROWS = 128;
inline constexpr int64_t KDA_MAX_AUX_BYTES = int64_t(1) << 30;

struct PreparedClassification {
    const void *keys;
    int32_t *flags;
    const int32_t *sequence_segments;
    int sequences;
    const int32_t* tile_flags = nullptr;
};
} // namespace flashkda

// Padded TC02 affine preparation and its existing exact structural classifier.
const int32_t* launch_flashkda_bf16_prepare(
    const void *q, const void *k, const void *g, const void *beta,
    const float *a_log, const float *bias, float scale, float gate_scale,
    int tokens, int heads, int sequences, int tiles, int beta_stride,
    int64_t q_stride, int64_t k_stride, const int32_t *cu, void *ws, void *keys,
    bool affine, hggcStream_t stream, bool preswizzle = false, int32_t *aggregate_flags = nullptr, bool defer_classification = false, int32_t* tile_flags = nullptr, const flashkda::ConditionedArguments* serial_repair = nullptr);

void launch_flashkda_zero_padded_output_tail(
    void* out, int T_total, int H, int N,
    const int32_t* cu_seqlens, int valid_tokens,
    hggcStream_t stream);

void launch_flashkda_recurrence(
    void* o, const void* v, int64_t v_stride, const void* beta,
    const void* ws,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices, const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream, int preferred_dv = 0);

// Faithful register-resident recurrence implementation called by the public K2
// dispatcher.  It remains in the shared interface so its definition and the
// cross-TU call are checked against the same declaration.
void launch_flashkda_recurrence_nows(
    void* o, const void* v, int64_t v_stride, const void* beta,
    const void* ws,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices, const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream, int preferred_dv = 0);

// Read-only small requests keep FP32 resident math and finish precision
// selection inside the serial recurrence, using the completed prepare flags.
void launch_flashkda_fp32_serial(
    void* out, const void* v, int64_t v_stride, const void* ws,
    const void* initial, void* final, const int32_t* input_indices,
    const int32_t* output_indices, bool state_fp32,
    int tokens, int heads, int sequences, int tiles, const int32_t* cu,
    const int32_t* prefix, const flashkda::ConditionedArguments& repair, hggcStream_t stream);

void launch_flashkda_track_initial(const flashkda::ConditionedArguments& repair,
    flashkda::TrackState track, hggcStream_t stream);
void launch_flashkda_tracked_serial(const flashkda::ConditionedArguments& repair,
    const void* ws, int tiles, const int32_t* prefix, bool fp32,
    flashkda::TrackState track, hggcStream_t stream);

// tc01-only four-stage BT64 path. The public operator ABI is unchanged; these
// buffers are private, same-stream intermediates owned by the host call.
void launch_flashkda_bt64_prefill_tc01(
    void* o, const void* v, int64_t v_stride, const void* ws16,
    void* bt_ws, void* x, void* h_checkpoint,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices,
    const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu_seqlens, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream);

// tc01-only BT64xCP composed pipeline.  The host interleaves these stages with
// the CH16 CP pre_scan/merge passes:
//   bt64 front -> cp prescan -> cp merge -> bt64 segment -> bt64 back.
// seg_bt is the BT64-aligned segment descriptor table (KDA_SEG_BT_STRIDE
// int32 lanes per segment); carries is the bf16 [n_seg, H, DV, DK] merge
// output consumed as each segment's init state.
void launch_flashkda_bt64_cp_front_tc01(
    const void* v, int64_t v_stride, const void* ws16, void* bt_ws,
    void* h_checkpoint_scratch,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu_seqlens, flashkda::KrWorkspaceLayout kr_layout,
    hggcStream_t stream);

void launch_flashkda_bt64_cp_segment_tc01(
    const void* ws16, const void* bt_ws, void* x, void* h_checkpoint,
    const void* carries, void* final_state,
    const int32_t* final_state_indices, bool final_state_is_fp32,
    int T_total, int H, int total_tiles, int total_bt,
    const int32_t* seg_bt, int n_seg, bool has_state_out,
    hggcStream_t stream);

void launch_flashkda_bt64_cp_back_tc01(
    void* o, const void* ws16, void* bt_ws, const void* x,
    const void* h_checkpoint,
    int T_total, int H, int N, int total_tiles, int total_bt,
    const int32_t* cu_seqlens, hggcStream_t stream);

// The debug prepare confirms the KR layout via a device probe and caches it per device; a cold capture must be warmed up eagerly first.
bool flashkda_kr_is_preswizzled(hggcStream_t stream);

// Short tc02 CP publishes one exact-zero flag per 64-row transition slice.
void launch_flashkda_cp_prescan_zero_flags(
    const void* v, const void* beta, const void* ws, void* chain, int32_t* zero_slices,
    int tokens, int heads, int segments, int tiles, const int* table, hggcStream_t stream);
void launch_flashkda_cp_merge_zero_flags(
    const void* chain, const void* initial, void* carries, const int32_t* indices,
    bool state_fp32, const int* mapping, const int32_t* zero_slices,
    int heads, int sequences, int segments, hggcStream_t stream, bool prepared_bias=false, const int32_t* skip_heads=nullptr);

// CP pass 1/3 (template modes of the register-resident recurrence kernel).
void launch_flashkda_cp_prescan(
    const void* v, int64_t v_stride, const void* beta, const void* ws,
    void* chain_state, bool fp16_state,
    int T_total, int H, int n_seg, int total_tiles,
    const int* seg_tbl, int bv, hggcStream_t stream, bool automatic_nt256 = false);

void launch_flashkda_cp_segment(
    void* o, const void* v, int64_t v_stride, const void* beta, const void* ws,
    const void* carries, bool fp16_state, void* final_state,
    const int32_t* final_state_indices, bool final_state_is_fp32,
    int T_total, int H, int n_seg, int total_tiles,
    const int* seg_tbl, int bv, hggcStream_t stream, bool automatic_nt256 = false);

// CP pass 2 (the only serial stage; serial over segments, not chunks).
void launch_flashkda_cp_merge(
    const void* chain, const void* init_state, void* carries, bool fp16_state,
    const int32_t* init_state_indices, bool init_state_is_fp32,
    const int* seq_seg, int H, int N, int n_seg, hggcStream_t stream);

void launch_flashkda_cp_merge_short_auto(
    const void* chain, const void* init_state, void* carries, bool fp16_state,
    const int32_t* init_state_indices, bool init_state_is_fp32,
    const int* seq_seg, int H, int N, int n_seg, hggcStream_t stream);


// Reuse CH32 scratch over bounded token windows. Every head stays parallel;
// FP32 carries preserve the recurrence across private workspace boundaries.
inline constexpr int64_t KDA_PAIRED_RECORD_BYTES =
    96 * 128 * 2 + 128 * 4 + 2 * 32 * 32 * 2 + 32 * 4;

inline int flashkda_paired_window_pairs(int tokens, int heads, int sequences) {
    int64_t const pairs = (int64_t(tokens) + 31) / 32 + sequences;
    int64_t const prefix_bytes = (int64_t(sequences + 1) * sizeof(int) + 127) / 128 * 128;
    if (heads <= 0 || pairs <= 0) return 0;
    int64_t const bytes = pairs * heads * KDA_PAIRED_RECORD_BYTES;
    if (bytes + 2 * prefix_bytes <= flashkda::KDA_MAX_AUX_BYTES) return int(pairs);
    int64_t const carries = int64_t(sequences) * heads * 128 * 128 * sizeof(float);
    int64_t const capacity = (flashkda::KDA_MAX_AUX_BYTES - 2 * prefix_bytes - carries)
        / (heads * KDA_PAIRED_RECORD_BYTES);
    return int(capacity > 0 ? capacity : 1);
}

inline int64_t flashkda_paired_workspace_bytes(int tokens, int heads, int sequences) {
    int const window = flashkda_paired_window_pairs(tokens, heads, sequences);
    int64_t const pairs = (int64_t(tokens) + 31) / 32 + sequences;
    int64_t const prefix_bytes = (int64_t(sequences + 1) * sizeof(int) + 127) / 128 * 128;
    return 2 * prefix_bytes + int64_t(window) * heads * KDA_PAIRED_RECORD_BYTES
        + (window < pairs ? int64_t(sequences) * heads * 128 * 128 * sizeof(float) : 0);
}

void launch_flashkda_paired_recurrence(
    void* out, const void* v, int64_t v_stride, const void* workspace,
    const void* initial, void* final, const int32_t* initial_indices,
    const int32_t* final_indices, bool state_fp32,
    int tokens, int heads, int sequences, int raw_tiles, const int32_t* cu,
    void* scratch, flashkda::KrWorkspaceLayout layout, hggcStream_t stream);


namespace flashkda {
// Compare a complete BF16 state after four C16 chunks. This is an exact proof,
// not a contraction threshold. Short segments keep the ordinary replay.
inline constexpr int KDA_REPLAY_PREFIX_TILES=3;
inline constexpr int KDA_REPLAY_MIN_TILES=32;
}

// Explicit BF16 resident-state mode; external BF16/FP32 storage is independent.
// Speculative segment outputs are reused only after a full checkpoint match.
void launch_flashkda_bf16_cp_verify(
    void* ws, void* chain, int32_t* flags,
    const int32_t* segments, int segment_count, int tiles,
    flashkda::KrWorkspaceLayout layout, flashkda::PreparedClassification classification,
    const flashkda::ConditionedArguments& args, bool full_tiles, hggcStream_t stream);

void launch_flashkda_bf16_empty_state(
    const void* initial, void* final, const int32_t* input_indices,
    const int32_t* output_indices, bool fp32, const int32_t* cu,
    int sequences, int heads, hggcStream_t stream);
void launch_flashkda_bf16_recurrence(
    void* out, const void* v, int64_t v_stride, const void* ws,
    const void* initial, void* final, const int32_t* input_indices,
    const int32_t* output_indices, bool fp32,
    int tokens, int heads, int sequences, int tiles, const int32_t* cu,
    flashkda::KrWorkspaceLayout layout, hggcStream_t stream,
    const flashkda::ConditionedArguments* repair = nullptr, const void* keys = nullptr,
    const int32_t* tile_flags = nullptr, const int32_t* tile_prefix = nullptr, int value_rows = 0);
void launch_flashkda_bf16_cp_prescan(
    const void* v, int64_t v_stride, const void* ws,
    void* chain, int32_t* zero_slices, int tokens, int heads, int segments,
    int tiles, const int* table, flashkda::KrWorkspaceLayout layout, hggcStream_t stream,
    void* output, void* checkpoint, bool full_tiles,
    flashkda::PreparedClassification classification, void* prepared_carries,
    const flashkda::ConditionedArguments* repair=nullptr);
void launch_flashkda_bf16_cp_segment(
    void* out, const void* v, int64_t v_stride, const void* ws,
    const void* carries, void* final, const int32_t* indices, bool fp32,
    int tokens, int heads, int segments, int tiles, const int* table,
    flashkda::KrWorkspaceLayout layout, hggcStream_t stream,
    const void* checkpoint = nullptr, const void* bias_final = nullptr, int32_t* zero_slices = nullptr, bool full_tiles = false,
    flashkda::PreparedClassification skip = {});

// Actual TC02 occupancy of the full-value BF16 summary/replay kernels.
hggcError_t query_flashkda_bf16_resident_blocks(
    int *blocks, bool row_major, bool verify = false, bool full_tiles = false);
