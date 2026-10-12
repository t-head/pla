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

#include <hggc_runtime.h>

#include <cstdint>

#include "prefill/flashkdapro/common/workspace_abi.h"

void launch_flashkda_prepare(
    const void* q, const void* k, const void* g_in, const void* beta,
    const float* A_log, const float* dt_bias, float scale, float gate_scale,
    int T_total, int H, int beta_stride,
    int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen,
    int tile_offset, int tile_count, int head_offset, int head_count,
    int* ready, void* ws, bool force_plain, bool cp_fma_front,
    bool fp16_kd_qd, flashkda::KrWorkspaceLayout kr_layout,
    hggcStream_t stream);

void launch_flashkda_zero_padded_output_tail(
    void* out, int T_total, int H, int N,
    const int32_t* cu_seqlens, int valid_tokens,
    hggcStream_t stream);

void launch_flashkda_recurrence(
    void* o, const void* v, const void* beta,
    const void* ws,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices, const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream);

// Faithful register-resident recurrence implementation called by the public K2
// dispatcher.  It remains in the shared interface so its definition and the
// cross-TU call are checked against the same declaration.
void launch_flashkda_recurrence_nows(
    void* o, const void* v, const void* beta,
    const void* ws,
    const void* init_state, void* final_state,
    const int32_t* init_state_indices, const int32_t* final_state_indices,
    bool init_state_is_fp32, bool final_state_is_fp32,
    int T_total, int H, int N, int total_tiles, const int32_t* cu_seqlens,
    bool is_varlen, bool has_state_in, bool has_state_out,
    flashkda::KrWorkspaceLayout kr_layout, hggcStream_t stream);

// tc01-only four-stage BT64 path. The public operator ABI is unchanged; these
// buffers are private, same-stream intermediates owned by the host call.
void launch_flashkda_bt64_prefill_tc01(
    void* o, const void* v, const void* ws16,
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
    const void* v, const void* ws16, void* bt_ws, void* h_checkpoint_scratch,
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

// CP pass 1/3 (template modes of the register-resident recurrence kernel).
void launch_flashkda_cp_prescan(
    const void* v, const void* beta, const void* ws,
    void* chain_state, bool fp16_state,
    int T_total, int H, int n_seg, int total_tiles,
    const int* seg_tbl, int bv, hggcStream_t stream);

void launch_flashkda_cp_segment(
    void* o, const void* v, const void* beta, const void* ws,
    const void* carries, bool fp16_state, void* final_state,
    const int32_t* final_state_indices, bool final_state_is_fp32,
    int T_total, int H, int n_seg, int total_tiles,
    const int* seg_tbl, int bv, hggcStream_t stream);

// CP pass 2 (the only serial stage; serial over segments, not chunks).
void launch_flashkda_cp_merge(
    const void* chain, const void* init_state, void* carries, bool fp16_state,
    const int32_t* init_state_indices, bool init_state_is_fp32,
    const int* seq_seg, int H, int N, int n_seg, hggcStream_t stream);
