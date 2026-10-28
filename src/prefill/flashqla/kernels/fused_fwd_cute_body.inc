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

  constexpr int H  = H_TPL;
  constexpr int Hg = Hg_TPL;
  constexpr int FF_DVH = DVH_TPL;
  constexpr int FF_HELEMS = FF_DVH * FF_HLD;
  constexpr int FF_WELEMS = FF_DVH * FF_WLD;
  constexpr bool FF_AUX_P = AUX_P_TPL;
  constexpr bool FF_AUX_U = FF_AUX_P && H_TPL == 12;
  constexpr int FF_NThreads = FF_WG * NGRP;
  // H=24 is the validated tc02 specialization: its register chain makes the
  // h/W/Vd/gVd backing tiles redundant, while Ag/Pg/g-expansion are shared.
  using Smem = FfSmemLayout<NGRP, FF_DVH, H_TPL == 24>;
#if FF_REG_CHAIN
  using MmaO = FfMmaM4T<Element>;
  using MmaH = FfMmaM4T<Element>;
#else
  using MmaO = fq::TiledMmaM32T<Element>;
  using MmaH = fq::TiledMmaM32T<Element>;
#endif
#if defined(FF_BODY_HEAD_MAJOR)
  const int seq_idx  = blockIdx.y;
  const int head_idx = blockIdx.x;
#elif defined(FLASHQLA_ARCH_150)
  // Order the flattened CTA grid by request, then head on tc02. This only
  // permutes tasks; it does not control CTA residency. CP indexes segments
  // rather than requests and keeps its original mapping. Restrict the mapping
  // to head configurations whose full-caselist rerun had no confirmed tail
  // regressions. H=16/Hg=8/N=22 keeps sequence-major order because its
  // full-caselist high-sample rerun confirmed a tail regression.
  const int task_idx = int(blockIdx.y) * int(gridDim.x) + int(blockIdx.x);
  const bool safe_head_shape =
      H >= 64 || (H == 16 && Hg >= 4 && !(Hg == 8 && batch_size == 22));
  const bool head_contiguous =
      cp_seq_map == nullptr && batch_size > 1 && safe_head_shape;
  const int seq_idx = head_contiguous ? task_idx / H : int(blockIdx.x);
  const int head_idx = head_contiguous ? task_idx % H : int(blockIdx.y);
#else
  int seq_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  if constexpr (H == 16 && Hg == 2 && NGRP == 2) {
    if (cp_seq_map == nullptr && (batch_size == 2 || batch_size == 3) &&
        cu_seqlens[batch_size] >= 16384 && cu_seqlens[batch_size] <= 32768) {
      const int task = int(blockIdx.y) * int(gridDim.x) + int(blockIdx.x);
      const int ordinal = task / H;
      head_idx = task % H;
      int lengths[3];
      lengths[0] = cu_seqlens[1] - cu_seqlens[0];
      lengths[1] = cu_seqlens[2] - cu_seqlens[1];
      lengths[2] = batch_size == 3 ? cu_seqlens[3] - cu_seqlens[2] : -1;
      CUTE_UNROLL
      for (int i = 0; i < 3; ++i) {
        int rank = 0;
        CUTE_UNROLL
        for (int j = 0; j < 3; ++j)
          rank += lengths[j] > lengths[i] || (lengths[j] == lengths[i] && j < i);
        if (rank == ordinal) seq_idx = i;
      }
    }
  }
#endif
  const int kv_head_idx = head_idx / (H / Hg);
  // 2D block: threadIdx.x = warp-group-local tid (0..FF_WG-1 = 0..127),
  //           threadIdx.y = warp-group ID (0 or 1).
#if defined(FLASHQLA_ARCH_100)
  constexpr bool FF_SPLIT_DV_CTA = NGRP == 1;
  const int aux_gid = FF_SPLIT_DV_CTA && FF_AUX_P ? threadIdx.y : 0;
  const bool is_aux = FF_AUX_P && aux_gid == 1;
  const int gid = FF_SPLIT_DV_CTA ? 0 : threadIdx.y;
#else
  const bool is_aux = false;
  const int gid = threadIdx.y;                     // DV-half group (was tx / FF_WG)
#endif
  const int l   = threadIdx.x;                     // MMA group-local tid (was tx % FF_WG)
#if defined(FLASHQLA_ARCH_100)
  const int tx  = (FF_SPLIT_DV_CTA ? aux_gid : gid) * FF_WG + l;
#else
  const int tx  = gid * FF_WG + l;                 // flat CTA tid for cooperative loops
#endif

  const int output_start = cu_seqlens[seq_idx];
  const int seq_end = cu_seqlens[seq_idx + 1];
  const int raw_batch_idx = cp_seq_map != nullptr ? cp_seq_map[seq_idx] : seq_idx;
  int seq_start = output_start;
  if (cp_seq_map != nullptr && num_warmup_chunks != nullptr &&
      fallback_mask != nullptr && raw_batch_idx >= 0 &&
      raw_batch_idx < raw_batch_size &&
      output_start > raw_cu_seqlens[raw_batch_idx]) {
    int predecessor = seq_idx - 1;
    while (predecessor >= 0 && cp_seq_map[predecessor] == raw_batch_idx) {
      const int metadata_offset = predecessor * H + head_idx;
      if (!static_cast<bool>(fallback_mask[metadata_offset])) {
        const int predecessor_start = cu_seqlens[predecessor];
        const int predecessor_end = cu_seqlens[predecessor + 1];
        const int predecessor_chunks =
            (predecessor_end - predecessor_start + FF_CS - 1) / FF_CS;
        int warmup_chunks = num_warmup_chunks[metadata_offset];
        warmup_chunks = max(0, min(warmup_chunks, predecessor_chunks));
        seq_start = max(predecessor_start,
                        predecessor_end - warmup_chunks * FF_CS);
        break;
      }
      seq_start = cu_seqlens[predecessor];
      --predecessor;
    }
  }
  const int seq_len = seq_end - seq_start;
  if (seq_len == 0) return;
  const int num_chunks = (seq_len + FF_CS - 1) / FF_CS;
  const int full_chunks = seq_len / FF_CS;  // chunks with actual==FF_CS (no partial)
#if defined(FF_BODY_INDEXED_TC01)
  // The mapped raw-batch slot is CTA-uniform. Reject a malformed private-call
  // map before reading raw_state_indices and before the first CTA barrier.
  if (raw_batch_idx < 0 || raw_batch_idx >= raw_batch_size) return;
#endif
#if !defined(FF_BODY_LEGACY_TC01)
  const int raw_state_idx =
      (raw_state_indices != nullptr && raw_batch_idx >= 0)
      ? raw_state_indices[raw_batch_idx] : raw_batch_idx;
  const auto* state_input =
      raw_state_input != nullptr ? raw_state_input : raw_state;
  const int state_input_idx =
      raw_state_input != nullptr ? raw_batch_idx : raw_state_idx;
#endif
  bool need_store_ht = false;
  bool load_raw_state = cp_seq_map == nullptr;
  if (raw_batch_idx >= 0 && raw_batch_idx < raw_batch_size) {
    need_store_ht = (seq_end >= raw_cu_seqlens[raw_batch_idx + 1]);
    load_raw_state = load_raw_state ||
        (seq_start <= raw_cu_seqlens[raw_batch_idx]);
  }

  extern __shared__ char smem[];
  // Input tiles are (double-)buffered: K_all/... are the slot-0 base pointers;
  // the per-chunk K_buf/... below index the active slot (p = c % FF_NBUF).
  Element* K_all    = reinterpret_cast<Element*>(smem + Smem::K);
  Element* V_all    = reinterpret_cast<Element*>(smem + Smem::V);
  Element* Q_all    = reinterpret_cast<Element*>(smem + Smem::Q);
  Element* A_all    = reinterpret_cast<Element*>(smem + Smem::A);
#if defined(FLASHQLA_ARCH_100)
  // tc01: KT_buf aliases the hstmT region (FF_HSTMT, 32KB) rather than a
  // separate allocation. hstmT is last read by GEMM1 (U = K @ hstmT); GEMM2,
  // GEMM6 and GEMM5 never touch it, so it is dead from after GEMM1 through the
  // end of the chunk (the next chunk re-snapshots it at the top). KT is a
  // block-global [DK,CS] tile, so it uses the raw FF_HSTMT base and is filled
  // by a block-wide cooperative loop after GEMM1/GEMM2, just before GEMM6.
  // FF_KT_BYTES stays 0 so FF_TOTAL matches tc02 and fits the TSM window.
  // See docs/flashqla_ppu1.0_port_notes.md section 8.
  Element* KT_buf   = reinterpret_cast<Element*>(smem + Smem::HSTMT);
#else
  Element* KT_buf   = reinterpret_cast<Element*>(smem + Smem::KT);
#endif
  Element* Ag_swz   = reinterpret_cast<Element*>(smem + Smem::AGSWZ);
  Element* hstmT_a  = reinterpret_cast<Element*>(smem + Smem::HSTMT);
  Element* Wswz_a   = reinterpret_cast<Element*>(smem + Smem::WSWZ);
  Element* Vdswz_a  = reinterpret_cast<Element*>(smem + Smem::VDSWZ);
  Element* gVdswz_a = reinterpret_cast<Element*>(smem + Smem::GVDSWZ);
  Element* Pg_a     = reinterpret_cast<Element*>(smem + Smem::PG);
  float*  g_all     = reinterpret_cast<float*>(smem + Smem::G);
  float*  b_all     = reinterpret_cast<float*>(smem + Smem::B);
  float*  g_exp     = reinterpret_cast<float*>(smem + Smem::GEXP);
  float*  g_rev     = reinterpret_cast<float*>(smem + Smem::GREV);
  float*  g_log     = reinterpret_cast<float*>(smem + Smem::GLOG);

  const int dv_base = (blockIdx.z * NGRP + gid) * FF_DVH;

  // per-group intermediate buffers
  Element* hstmT_g  = hstmT_a  + gid * FF_HELEMS;
  Element* Wswz_g   = Wswz_a   + gid * FF_WELEMS;
  Element* Vdswz_g  = Vdswz_a  + gid * FF_WELEMS;
  Element* gVdswz_g = gVdswz_a + gid * FF_WELEMS;
#if FF_REG_CHAIN
  // Pg depends on Q/K and the per-head gate, but not on the DV half.  Both
  // NGRP=2 groups therefore consume one shared tile instead of building an
  // identical copy for each half.
  Element* Pg_g     = Pg_a;
#else
  Element* Pg_g     = Pg_a     + gid * FF_PGELEMS;
#endif

  // ---- MMA partition helpers. tc02 uses the transposed [DVh,CS] O chain and
  //      [DVh,DK] state layout; tc01 retains the original [CS,DVh]/[DK,DVh]
  //      layouts. ----
  MmaO mmaO;
  auto thr  = mmaO.get_thread_slice(l);       // O-chain thread slice (also GEMM4 P)
  MmaH mmaH;
  auto thrH = mmaH.get_thread_slice(l);        // state (GEMM6 / register-h) slice
#if FF_REG_CHAIN
  auto cId_csdv = cute::make_identity_tensor(
      cute::make_shape(cute::Int<FF_DVH>{}, cute::Int<FF_CS>{}));
  auto tc_csdv = thr.partition_C(cId_csdv);          // C coords for [DVh,CS]
  auto cId_dkdv = cute::make_identity_tensor(
      cute::make_shape(cute::Int<FF_DVH>{}, cute::Int<FF_DK>{}));
  auto tc_dkdv = thrH.partition_C(cId_dkdv);         // C coords for [DVh,DK]
#else
  auto cId_csdv = cute::make_identity_tensor(
      cute::make_shape(cute::Int<FF_CS>{}, cute::Int<FF_DVH>{}));
  auto tc_csdv = thr.partition_C(cId_csdv);          // C coords for [CS,DVh]
  auto cId_dkdv = cute::make_identity_tensor(
      cute::make_shape(cute::Int<FF_DK>{}, cute::Int<FF_DVH>{}));
  auto tc_dkdv = thrH.partition_C(cId_dkdv);         // C coords for [DK,DVh]
#endif

  // ---- register-resident state. tc02 keeps h^T[DVh,DK] in the 4x1 layout;
  //      tc01 retains h[DK,DVh] in its validated M32 layout. ----
#if FF_REG_CHAIN
  auto h_reg = fq::make_acc<FF_DVH, FF_DK, MmaH, Element>(l);
#else
  auto h_reg = fq::make_acc<FF_DK, FF_DVH, MmaH, Element>(l);
#endif
  cute::Tensor rHreg = cute::recast<float>(h_reg);
  if (!is_aux) {
    int64_t hbase = (int64_t)seq_idx * H * FF_DK * FF_DV +
                    (int64_t)head_idx * FF_DK * FF_DV;
    if constexpr (STATE_BF16) {
      // Two-phase the read: issue a whole innermost-vv batch into distinct
      // staging registers before converting any of it.  With a narrowed pool the
      // convert forces each load to land in a scratch register, and at 240/256
      // registers the allocator recycles the same one across unroll steps, so
      // the compiler must emit s.wait vldcnt(0) after EVERY load and in-flight
      // loads collapse from one batch to one.
      using SElem = fq::StatePool<STATE_BF16>;
      constexpr int NVV = decltype(cute::size<0>(tc_dkdv))::value;
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
          SElem raw[NVV];
          CUTE_UNROLL
          for (int vv = 0; vv < NVV; ++vv) {
            auto crd = tc_dkdv(vv, m, n);
#if FF_REG_CHAIN
            int dv = cute::get<0>(crd), dk = cute::get<1>(crd);  // h^T [DVh,DK]
#else
            int dk = cute::get<0>(crd), dv = cute::get<1>(crd);  // h [DK,DVh]
#endif
#if defined(FF_BODY_LEGACY_TC01)
            if (h0 != nullptr) {
              raw[vv] = h0[hbase + dk * FF_DV + (dv_base + dv)];
            } else {
              fq::state_from_f32(raw[vv], 0.0f);
            }
#else
            if (h0 != nullptr) {
              raw[vv] = h0[hbase + dk * FF_DV + (dv_base + dv)];
            } else if (load_raw_state && state_input != nullptr &&
                       state_input_idx >= 0) {
              const int64_t raw_off =
                  ((int64_t)state_input_idx * H + head_idx) * FF_DV * FF_DK +
                  (int64_t)(dv_base + dv) * FF_DK + dk;
              raw[vv] = state_input[raw_off];
            } else {
              fq::state_from_f32(raw[vv], 0.0f);
            }
#endif
          }
          CUTE_UNROLL
          for (int vv = 0; vv < NVV; ++vv)
            rHreg(vv, m, n) = fq::state_to_f32(raw[vv]);
        }
      }
    } else {
      // fp32 state: state_to_f32 is the identity, so the direct store already
      // keeps every load in flight.  Staging here only widens the live range at
      // the kernel prologue -- it costs ~1% end-to-end on tc01 and wins nothing.
      // Keep this loop byte-identical to the pre-staging form.
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
            auto crd = tc_dkdv(vv, m, n);
#if FF_REG_CHAIN
            int dv = cute::get<0>(crd), dk = cute::get<1>(crd);  // h^T [DVh,DK]
#else
            int dk = cute::get<0>(crd), dv = cute::get<1>(crd);  // h [DK,DVh]
#endif
#if defined(FF_BODY_LEGACY_TC01)
            rHreg(vv, m, n) =
                (h0 != nullptr)
                ? fq::state_to_f32(h0[hbase + dk * FF_DV + (dv_base + dv)])
                : 0.0f;
#else
            if (h0 != nullptr) {
              rHreg(vv, m, n) =
                  fq::state_to_f32(h0[hbase + dk * FF_DV + (dv_base + dv)]);
            } else if (load_raw_state && state_input != nullptr &&
                       state_input_idx >= 0) {
              const int64_t raw_off =
                  ((int64_t)state_input_idx * H + head_idx) * FF_DV * FF_DK +
                  (int64_t)(dv_base + dv) * FF_DK + dk;
              rHreg(vv, m, n) = fq::state_to_f32(state_input[raw_off]);
            } else {
              rHreg(vv, m, n) = 0.0f;
            }
#endif
          }
        }
      }
    }
  }

  // ---- prologue: seed the input pipeline with multi-warp split AIU ----
  if (!is_aux) {
    const int warp = __shfl_sync(0xffffffff, tx >> 5, 0);
    const int lane = tx & 31;
    ff_preload_chunks_split_aiu<Element, H, Hg, NGRP>(
        FF_NBUF, seq_start, 0, seq_end, warp, lane,
        Q_all, K_all, V_all, A_all, g_all, b_all,
        q, k, v, a, g, beta, kv_head_idx, head_idx, v_token_stride);
  }
  __syncthreads();  // both input slots are visible before the chunk loop

#pragma unroll 1
#pragma clang loop licm(disable)
  for (int c = 0; c < num_chunks; ++c) {
    const int chunk_start = seq_start + c * FF_CS;
    const int actual = min(FF_CS, seq_end - chunk_start);
    const Element* K_src = k + (int64_t)chunk_start * Hg * FF_DK + kv_head_idx * FF_DK;
    const Element* Q_src = q + (int64_t)chunk_start * Hg * FF_DK + kv_head_idx * FF_DK;
    const Element* V_src =
        v + (int64_t)chunk_start * v_token_stride + head_idx * FF_DV;
    const Element* A_src = a + (int64_t)chunk_start * H  * FF_CS + head_idx    * FF_CS;

    // ---- active input slot for this chunk; other slot receives the prefetch ----
#if defined(FLASHQLA_ARCH_100) && !FF_REG_CHAIN
    const int p = 0;
#else
    const int p = c % FF_NBUF;
#endif
    Element* K_buf = K_all + p * FF_KELEMS;
    Element* V_buf = V_all + p * FF_VELEMS;
    Element* Q_buf = Q_all + p * FF_KELEMS;
    Element* A_buf = A_all + p * FF_AELEMS;
    float*  g_buf = g_all + p * FF_CS;   // g/beta for chunk c: prefetched by the
    float*  b_buf = b_all + p * FF_CS;   // prior iteration (or prologue for c=0).

    // ================= PHASE A: drain the active chunk's DMA =====================
    if (!is_aux) {
    if (actual == FF_CS) {
#if defined(FLASHQLA_ARCH_100) && !FF_REG_CHAIN
      // One tc01 input slot: c was issued only after the preceding chunk fence.
      ff_aiu_wait_group<0>();
#else
      // Drain chunk c while retaining the next full chunk in flight.  The
      // last full chunk flushes the FIFO before a possible partial tail.
      if (c < full_chunks - 1)
        ff_aiu_wait_group<1>();
      else
        ff_aiu_wait_group<0>();
#endif
    } else {
      // Tail (partial) chunk: no prefetch was issued; FIFO is empty. Inline-issue
      // the single flat group and drain immediately.
      if (tx == 0) {
        const float* g_srcp = g    + (int64_t)chunk_start * H + head_idx;
        const float* b_srcp = beta + (int64_t)chunk_start * H + head_idx;
        ff_issue_chunk_aiu<Element>(K_buf, V_buf, Q_buf, A_buf,
                           K_src, V_src, Q_src, A_src, H, Hg,
                           v_token_stride, actual,
                           g_buf, b_buf, g_srcp, b_srcp);
      }
      ff_aiu_wait_group<0>();
    }
    }
    // The next safe prefetch is deferred to the chunk fence (see loop end).

    // ============ PHASE A-tail: g_exp / g_rev / stable g snapshot =========
    __syncthreads();  // publish AIU-drained chunk data (K/V/Q/A/g/b) to all
    // NOTE (tc01): K^T is materialised into KT_buf (which aliases hstmT) AFTER
    // GEMM1, not here -- hstmT is live from the Phase-C snapshot through GEMM1,
    // so filling KT now would clobber it. See the KT fill just before GEMM6.
    {
      const int w = tx >> 5;
      const int i = tx & 31;
      if (w < 3 && i < FF_CS) {
        const float g_last_w = g_buf[actual > 0 ? actual - 1 : 0];
        const float gi = (i < actual) ? g_buf[i] : g_last_w;
        if (w == 0)      g_exp[i]     = exp2f(gi * FF_LOG2E);
        else if (w == 1) g_rev[i]     = exp2f((g_last_w - gi) * FF_LOG2E);
        else             g_log[i]     = gi;
      }
#if !FF_REG_CHAIN
      // Phase C: snapshot register-h -> hstmT_g (pre-decay). Reads only regs,
      // writes per-group smem; runs in parallel with g_exp compute above.
      if (!is_aux) {
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
            auto crd = tc_dkdv(vv, m, n);
            int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
            hstmT_g[dv * FF_HLD + dk] = fq::float_to_element<Element>(rHreg(vv, m, n));
          }
        }
      }
      }
#endif
    }
    const float g_last = g_buf[actual > 0 ? actual - 1 : 0];
    const float decay  = exp2f(g_last * FF_LOG2E);
    // Phase B: Ag = tri(exp(gi-gj)) * A * beta -> Ag_swz (Swizzle<2,3,3>, int4 stores)
    //
    // Reads g_buf rather than the g_log snapshot so that it does not depend on the
    // block above, which lets a single barrier below publish both. Inside the
    // r >= c2 && r < actual && c2 < actual guard the two hold the same value:
    // g_log[i] is g_buf[i] for i < actual and g_last_w only past it.
    {
#if !(defined(FLASHQLA_ARCH_100) && FF_VD_REG_B && FF_GVD_REG_B && !FF_REG_CHAIN)
      auto sAg_sw = fq::make_smem_swzA_cs32<FF_CS, FF_CS>(Ag_swz);
#endif
#if defined(FLASHQLA_ARCH_100)
      // tc01: A_buf is AIU-native cube-major; read/write through the view.
      auto sA_ain = ff_a_view_tc01<Element>(A_buf);
#endif
      constexpr int GRP_PER_ROW = FF_CS / 8;                 // 4 int4 groups per row
      for (int grp = tx; grp < FF_CS * GRP_PER_ROW; grp += FF_NThreads) {
        const int r  = grp / GRP_PER_ROW;
        const int c0 = (grp % GRP_PER_ROW) * 8;
#if defined(FLASHQLA_ARCH_100)
        const float g_row = (r < actual) ? g_buf[r] : 0.0f;
#endif
        Element packed[8];
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) {
          const int c2 = c0 + j;
          if (r >= c2 && r < actual && c2 < actual) {
#if defined(FLASHQLA_ARCH_100)
            float av  = fq::element_to_float<Element>(
                *reinterpret_cast<const Element*>(&sA_ain(r, c2)));
#else
            float av  = fq::element_to_float<Element>(A_buf[r * FF_CS + c2]);
#endif
#if defined(FLASHQLA_ARCH_100)
            float gij = exp2f(fminf(g_row - g_buf[c2], 0.0f) * FF_LOG2E);
#else
            float gij = ff_gate_ratio(g_buf, r, c2);
#endif
            packed[j] = fq::float_to_element<Element>(av * gij * b_buf[c2]);
          } else {
            packed[j] = fq::float_to_element<Element>(0.0f);
          }
#if defined(FLASHQLA_ARCH_100)
#if !FF_REG_CHAIN
          *reinterpret_cast<Element*>(&sA_ain(r, c2)) = packed[j];
#endif
#else
          A_buf[r * FF_CS + c2] = packed[j];
#endif
        }
#if !(defined(FLASHQLA_ARCH_100) && FF_VD_REG_B && FF_GVD_REG_B && !FF_REG_CHAIN)
        *reinterpret_cast<int4*>(&sAg_sw(r, c0)) =
            *reinterpret_cast<const int4*>(packed);
#endif
      }
    }

#if defined(FLASHQLA_ARCH_100)
    // The optional second warp-group overlaps the DV-independent QK^T/Pg work
    // with the main group's state snapshot and gated-WY preparation.  It writes
    // the same Pg tile that the main group consumes after the existing block
    // publication barrier, so no extra shared memory or synchronization phase
    // is introduced.
    if constexpr (FF_AUX_P && !FF_REG_CHAIN) {
      if (is_aux) {
        using MmaP = MmaO;
        MmaP mmaP;
        auto thrP = mmaP.get_thread_slice(l);
        auto p_acc = fq::make_acc<FF_CS, FF_CS, MmaP, Element>(l);
        fq::flashqla_gemm_ss_swzlAB_m32<
            FF_CS, FF_CS, FF_DK, decltype(p_acc), Element>(
            Q_buf, K_buf, l, p_acc);
        auto cP = cute::make_identity_tensor(
            cute::make_shape(cute::Int<FF_CS>{}, cute::Int<FF_CS>{}));
        auto tcP = thrP.partition_C(cP);
        auto rP = cute::recast<float>(p_acc);
        CUTE_UNROLL
        for (int m = 0; m < cute::size<1>(tcP); ++m) {
          CUTE_UNROLL
          for (int n = 0; n < cute::size<2>(tcP); ++n) {
            CUTE_UNROLL
            for (int vv = 0; vv < cute::size<0>(tcP); ++vv) {
              auto crd = tcP(vv, m, n);
              int row = cute::get<0>(crd), col = cute::get<1>(crd);
              float pg = 0.0f;
              if (row >= col && row < actual && col < actual)
                pg = scale * ff_gate_ratio(g_buf, row, col) *
                     rP(vv, m, n);
              Pg_g[row * FF_PGLD + col] =
                  fq::float_to_element<Element>(pg);
            }
          }
        }
      }
    }
#endif

#if FF_REG_CHAIN
    // P and Pg are identical for the two DV halves.  In the two-group kernel,
    // gid0 already owns all g/Ag work above because those loops cover at most
    // 128 threads; gid1 would otherwise wait idle at the block barrier.  Let
    // gid1 build the single shared Pg tile in parallel.  NGRP=1 builds it here
    // after Ag, still before the same existing barrier.
    if (!is_aux && (NGRP == 1 || gid == 1)) {
      auto sPg_write = fq::make_smem_swzA_cs32<FF_CS, FF_CS>(Pg_g);
      using MmaP = fq::TiledMmaM32T<Element>;
      MmaP mmaP;
      auto thrP = mmaP.get_thread_slice(l);
      auto p_acc = fq::make_acc<FF_CS, FF_CS, MmaP, Element>(l);
      // Warp 2 owns rows [0,16) x columns [16,32), which are strictly above
      // the causal diagonal and are overwritten with zero below. Skip its QK^T
      // fragment instead of issuing tensor-cell work whose result is discarded.
      // The tiled-MMA warp placement is the same on tc01 and tc02 even though
      // their atom-level A/C lane layouts differ, so this pruning is valid on
      // both architectures.
      if ((l >> 5) == 2)
        cute::clear(p_acc);
      else
      fq::flashqla_gemm_ss_swzlAB_m32<FF_CS, FF_CS, FF_DK,
          decltype(p_acc), Element>(Q_buf, K_buf, l, p_acc);
      auto cP = cute::make_identity_tensor(
          cute::make_shape(cute::Int<FF_CS>{}, cute::Int<FF_CS>{}));
      auto tcP = thrP.partition_C(cP);
      auto rP = cute::recast<float>(p_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tcP); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tcP); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tcP); ++vv) {
            auto crd = tcP(vv, m, n);
            int row = cute::get<0>(crd), col = cute::get<1>(crd);
            float pg = 0.0f;
            if (row >= col && row < actual && col < actual)
              pg = scale * ff_gate_ratio(g_buf, row, col) * rP(vv, m, n);
            sPg_write(row, col) =
                typename fq::ElementTraits<Element>::CuteType(pg);
          }
        }
      }
    }
#endif

#if FF_REG_CHAIN
    // Publishes g_exp/g_rev/g_log, Ag (Ag_swz and the gated A_buf), and the
    // shared Pg tile.  This remains the only cross-group publication point in
    // the chunk body, so overlapping Pg with gid0's preprocessing adds no new
    // synchronization.
#else
    // Publishes g_exp/g_rev/g_log, the register-h snapshot, and Ag.
#endif
    //
    // Ag has to be covered by something: Phase B is a tx-strided loop over 128
    // items. In the two-group specialization only gid==0 builds the single copy
    // that both groups read; the one-group specialization builds that same copy
    // in its sole group. The group barriers below cannot publish gid0's writes to
    // gid1, so the block barrier remains required for the two-group form.
    //
    // Measured on 890P against a dedicated barrier here: relocating is ~0.4%
    // faster.  Giving each group its own Ag copy avoids the rendezvous but has
    // to build Ag twice and is slower; the shared Ag/Pg tiles therefore keep
    // this single block publication point.
    __syncthreads();

    if (!is_aux || FF_AUX_U) {
#if FF_GEMM_REORDER
#if FF_REG_CHAIN
    // ============ PHASE D: DV-major warp-owned register chain ================
    // The 4x1 MMA layout gives each warp one fixed 16-row DV tile. tc02 A/C
    // layouts are identical, so h^T, W^T and Vd^T become the next GEMM's A
    // fragments with lane-local packing and no cross-lane shuffle.

    // Register h^T -> architecture-native A fragments, one tile at a time.
    uint32_t h_a_regs[FF_DK / 16][1][4];
    CUTE_UNROLL
    for (int kt = 0; kt < FF_DK / 16; ++kt) {
      CUTE_UNROLL
      uint32_t h_c_regs[4];
      for (int i = 0; i < 4; ++i) {
        uint32_t lo = fq::element_bits16<Element>(
            fq::float_to_element<Element>(rHreg(2 * i, 0, kt)));
        uint32_t hi = fq::element_bits16<Element>(
            fq::float_to_element<Element>(rHreg(2 * i + 1, 0, kt)));
        h_c_regs[i] = fq::pack_b16x2_u32(lo, hi);
      }
      fq::cfrag_to_afrag_mma_u32<Element>(h_c_regs, h_a_regs[kt][0], l & 31);
    }

    // ---- GEMM3^T/GEMM1^T: O_int^T = h^T@Q^T, U^T = h^T@K^T ----
    auto o_acc = fq::make_acc<FF_DVH, FF_CS, MmaO, Element>(l);
    auto u_acc = fq::make_acc<FF_DVH, FF_CS, MmaO, Element>(l);
    ff_gemm_regA_qkB_m4<Element>(
        h_a_regs, Q_buf, K_buf, l, o_acc, u_acc);
    auto rO = cute::recast<float>(o_acc);
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
          int cs = cute::get<1>(tc_csdv(vv, m, n));
          rO(vv, m, n) *= scale * g_exp[cs];
        }
      }
    }

    // U^T -> W^T = V^T - U^T*diag(g_exp), packed directly as the A
    // fragments consumed by W^T @ Ag^T.
    uint32_t w_a_regs[FF_CS / 16][1][4];
    {
      auto rU = cute::recast<float>(u_acc);
#if defined(FLASHQLA_ARCH_100)
      auto sV_ain = ff_qkv_view_tc01<Element>(V_buf);
#endif
      CUTE_UNROLL
      for (int kt = 0; kt < FF_CS / 16; ++kt) {
        CUTE_UNROLL
        uint32_t w_c_regs[4];
        for (int i = 0; i < 4; ++i) {
          auto crd0 = tc_csdv(2 * i, 0, kt);
          auto crd1 = tc_csdv(2 * i + 1, 0, kt);
          int dv0 = cute::get<0>(crd0), cs0 = cute::get<1>(crd0);
          int dv1 = cute::get<0>(crd1), cs1 = cute::get<1>(crd1);
          float w0 = fq::element_to_float<Element>(
#if defined(FLASHQLA_ARCH_100)
                         *reinterpret_cast<const Element*>(
                             &sV_ain(cs0, dv_base + dv0)))
#else
                         V_buf[cs0 * FF_DV + dv_base + dv0])
#endif
                     - g_exp[cs0] * rU(2 * i, 0, kt);
          float w1 = fq::element_to_float<Element>(
#if defined(FLASHQLA_ARCH_100)
                         *reinterpret_cast<const Element*>(
                             &sV_ain(cs1, dv_base + dv1)))
#else
                         V_buf[cs1 * FF_DV + dv_base + dv1])
#endif
                     - g_exp[cs1] * rU(2 * i + 1, 0, kt);
          uint32_t lo = fq::element_bits16<Element>(
              fq::float_to_element<Element>(w0));
          uint32_t hi = fq::element_bits16<Element>(
              fq::float_to_element<Element>(w1));
          w_c_regs[i] = fq::pack_b16x2_u32(lo, hi);
        }
        fq::cfrag_to_afrag_mma_u32<Element>(w_c_regs, w_a_regs[kt][0], l & 31);
      }
    }

    // ---- GEMM2^T: Vd^T = W^T @ Ag^T ----
    // With TN MMA semantics Ag is a natural [N=CS,K=CS] B view; using a
    // physical transposed load here would compute W^T@Ag and is incorrect.
    auto vd_acc = fq::make_acc<FF_DVH, FF_CS, MmaO, Element>(l);
    ff_gemm_regA_swzlB_m4<FF_CS, FF_CS, Element>(
        w_a_regs, Ag_swz, l, vd_acc);

    // Vd^T feeds Vd^T @ Pg^T unchanged and Vd^T*diag(g_rev) @ K after
    // column scaling. Both are direct tc02 A fragments; no shuffle is needed.
    uint32_t vd_a_regs[FF_CS / 16][1][4];
#if !defined(FLASHQLA_ARCH_100) || !defined(FF_VD_INPLACE)
    uint32_t gvd_a_regs[FF_CS / 16][1][4];
#endif
    {
      auto rVd = cute::recast<float>(vd_acc);
      CUTE_UNROLL
      for (int kt = 0; kt < FF_CS / 16; ++kt) {
        CUTE_UNROLL
        uint32_t vd_c_regs[4];
        for (int i = 0; i < 4; ++i) {
          float vd0 = rVd(2 * i, 0, kt);
          float vd1 = rVd(2 * i + 1, 0, kt);
          uint32_t lo = fq::element_bits16<Element>(
              fq::float_to_element<Element>(vd0));
          uint32_t hi = fq::element_bits16<Element>(
              fq::float_to_element<Element>(vd1));
          vd_c_regs[i] = fq::pack_b16x2_u32(lo, hi);
        }
        fq::cfrag_to_afrag_mma_u32<Element>(vd_c_regs, vd_a_regs[kt][0], l & 31);
#if !defined(FLASHQLA_ARCH_100) || !defined(FF_VD_INPLACE)
        // gVd has the same A-fragment ownership as Vd.  Scale the already
        // converted Vd fragment at its destination CS coordinates instead of
        // building a second C fragment and paying another eight cross-lane
        // shuffles on tc01.
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) {
          // Destination CS coordinate of the A fragment, per architecture.
          // Both atoms place the pair of halves of one u32 at adjacent cs
          // (cs, cs+1), but the tile-column stride of value bit i differs:
          //   tc01 A ((4,8),(2,2,2)):((32,1),(16,128,8)): cs += 8*(i&1)
          //   tc02 A ((4,8),(2,2,2)):((32,1),(16,8,128)): cs += 8*(i>>1)
          // Using the tc01 form on tc02 scales g_rev at the wrong CS
          // positions and corrupts the recurrent-state update (48b1dc6).
#if defined(FLASHQLA_ARCH_100)
          const int cs0 = kt * 16 + 2 * (l & 3) + 8 * (i & 1);
#else
          const int cs0 = kt * 16 + 2 * (l & 3) + 8 * (i >> 1);
#endif
          const int cs1 = cs0 + 1;
          const uint32_t vd_pack = vd_a_regs[kt][0][i];
          const float vd0 = fq::element_to_float<Element>(
              fq::element_from_bits16<Element>(
                  fq::get_b16_from_pack(vd_pack, 0)));
          const float vd1 = fq::element_to_float<Element>(
              fq::element_from_bits16<Element>(
                  fq::get_b16_from_pack(vd_pack, 1)));
          const uint32_t lo = fq::element_bits16<Element>(
              fq::float_to_element<Element>(g_rev[cs0] * vd0));
          const uint32_t hi = fq::element_bits16<Element>(
              fq::float_to_element<Element>(g_rev[cs1] * vd1));
          gvd_a_regs[kt][0][i] = fq::pack_b16x2_u32(lo, hi);
        }
#endif
      }
    }

#if defined(FLASHQLA_ARCH_100) && defined(FF_VD_INPLACE)
    // Consume the unscaled Vd fragment before reusing the same registers for
    // gVd. This shortens the tc01 live range by one A-fragment array.
    ff_gemm_regA_swzlB_m4<FF_CS, FF_CS, Element>(
        vd_a_regs, Pg_g, l, o_acc, /*clear_acc=*/false);
    CUTE_UNROLL
    for (int kt = 0; kt < FF_CS / 16; ++kt) {
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        const int cs0 = kt * 16 + 2 * (l & 3) + 8 * (i & 1);
        const int cs1 = cs0 + 1;
        const uint32_t vd_pack = vd_a_regs[kt][0][i];
        const float vd0 = fq::element_to_float<Element>(
            fq::element_from_bits16<Element>(
                fq::get_b16_from_pack(vd_pack, 0)));
        const float vd1 = fq::element_to_float<Element>(
            fq::element_from_bits16<Element>(
                fq::get_b16_from_pack(vd_pack, 1)));
        const uint32_t lo = fq::element_bits16<Element>(
            fq::float_to_element<Element>(g_rev[cs0] * vd0));
        const uint32_t hi = fq::element_bits16<Element>(
            fq::float_to_element<Element>(g_rev[cs1] * vd1));
        vd_a_regs[kt][0][i] = fq::pack_b16x2_u32(lo, hi);
      }
    }
#endif

    // ---- decay h^T; GEMM6^T h^T += gVd^T@K; GEMM5^T O^T += Vd^T@Pg^T ----
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv)
          rHreg(vv, m, n) *= decay;
      }
    }
#if defined(FLASHQLA_ARCH_100) && defined(FF_VD_INPLACE)
    ff_gemm_regA_ktB_m4_acc<Element>(
        vd_a_regs, K_buf, l, h_reg);
#else
    ff_gemm_regA_ktB_m4_acc<Element>(
        gvd_a_regs, K_buf, l, h_reg);
    ff_gemm_regA_swzlB_m4<FF_CS, FF_CS, Element>(
        vd_a_regs, Pg_g, l, o_acc, /*clear_acc=*/false);
#endif

    // The DV-major fragment owns a [DV16,CS32] O^T tile per warp.  Transpose it
    // once in registers and publish two aligned int4 vectors per lane.  This
    // removes the Wswz_g scatter, group barrier and TSM reload from the final
    // output path without reintroducing scalar/uncoalesced global stores.
    if (chunk_start >= output_start) {
      Element* O_ptr = o + (int64_t)chunk_start * H * FF_DV
                       + head_idx * FF_DV;
      ff_store_ot_register_direct<Element>(
          o_acc, O_ptr, (int64_t)H * FF_DV, dv_base, l, actual);
    }
#else
    // ============ PHASE D (FF_GEMM_REORDER): GEMM4->3->1->2->6->5 =============
    auto sH  = fq::make_smem_tensor_h_ld<FF_DVH, FF_DK, FF_HLD>(hstmT_g);

    // ---- GEMM4: P = Q @ K^T -> Pg = scale*tri(exp(gi-gj))*P ----
    if constexpr (!FF_AUX_P) {
    using MmaP = MmaO;
    const int lp = l;
    MmaP mmaP;
    auto thrP = mmaP.get_thread_slice(lp);
    auto p_acc = fq::make_acc<FF_CS, FF_CS, MmaP, Element>(lp);
    fq::flashqla_gemm_ss_swzlAB_m32<
        FF_CS, FF_CS, FF_DK, decltype(p_acc), Element>(
        Q_buf, K_buf, lp, p_acc);
    cute::Tensor cP  = cute::make_identity_tensor(
        cute::make_shape(cute::Int<FF_CS>{}, cute::Int<FF_CS>{}));
    cute::Tensor tcP = thrP.partition_C(cP);
    cute::Tensor rP  = cute::recast<float>(p_acc);
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tcP); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tcP); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tcP); ++vv) {
          auto crd = tcP(vv, m, n);
          int row = cute::get<0>(crd), col = cute::get<1>(crd);
          float pg = 0.0f;
          if (row >= col && row < actual && col < actual)
            pg = scale * ff_gate_ratio(g_log, row, col) * rP(vv, m, n);
          Pg_g[row * FF_PGLD + col] = fq::float_to_element<Element>(pg);
        }
      }
    }
    }

    // ---- GEMM3: O_int = Q @ hstmT_g ; O_int *= scale*g_exp ----
    // o_acc produced here, accumulated into by GEMM5 at the very end.
    auto o_acc = fq::make_acc<FF_CS, FF_DVH, MmaO, Element>(l);
    auto u_acc = fq::make_acc<FF_CS, FF_DVH, MmaO, Element>(l);
#if defined(FLASHQLA_ARCH_100)
    if constexpr (DVH_TPL == 32) {
      if constexpr (FF_AUX_U) {
        if (is_aux) {
          fq::flashqla_gemm_ss_swzlA_m32<
              FF_CS, FF_DK, decltype(sH), decltype(u_acc), Element>(
              K_buf, sH, l, u_acc);
        } else {
          fq::flashqla_gemm_ss_swzlA_m32<
              FF_CS, FF_DK, decltype(sH), decltype(o_acc), Element>(
              Q_buf, sH, l, o_acc);
        }
      } else {
        fq::fq_gemm_qk_aiuA_ss_tc01<FF_CS, FF_DK>(
            Q_buf, K_buf, sH, l, o_acc, u_acc);
      }
    } else {
      fq::flashqla_gemm_ss_swzlA_m32<
          FF_CS, FF_DK, decltype(sH), decltype(o_acc), Element>(
          Q_buf, sH, l, o_acc);
    }
#else
    fq::flashqla_gemm_ss_swzlA_m32<FF_CS, FF_DK, decltype(sH), decltype(o_acc), Element>(
        Q_buf, sH, l, o_acc);
#endif
    if (!is_aux) {
      cute::Tensor rO = cute::recast<float>(o_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            int cs = cute::get<0>(tc_csdv(vv, m, n));
            rO(vv, m, n) *= scale * g_exp[cs];
          }
        }
      }
    }

    // ---- GEMM1: U = K @ hstmT_g -> W = V[:,half] - g_exp*U -> W_swz ----
#if defined(FLASHQLA_ARCH_100)
    if constexpr (DVH_TPL != 32) {
      fq::flashqla_gemm_ss_swzlA_m32<
          FF_CS, FF_DK, decltype(sH), decltype(u_acc), Element>(
          K_buf, sH, l, u_acc);
    }
#else
    fq::flashqla_gemm_ss_swzlA_m32<FF_CS, FF_DK, decltype(sH), decltype(u_acc), Element>(
        K_buf, sH, l, u_acc);
#endif
    if (!FF_AUX_U || is_aux) {
      auto rU = cute::recast<float>(u_acc);
#if defined(FLASHQLA_ARCH_100)
      auto sV_ain = ff_qkv_view_tc01<Element>(V_buf);  // AIU-native V read
#endif
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
#if defined(FLASHQLA_ARCH_100)
            float w = fq::element_to_float<Element>(
                          *reinterpret_cast<const Element*>(&sV_ain(cs, dv_base + dv)))
                      - g_exp[cs] * rU(vv, m, n);
#else
            float w = fq::element_to_float<Element>(V_buf[cs * FF_DV + (dv_base + dv)])
                      - g_exp[cs] * rU(vv, m, n);
#endif
            Wswz_g[dv * FF_WLD + cs] = fq::float_to_element<Element>(w);
          }
        }
      }
    }
    if constexpr (FF_AUX_U) {
      __syncthreads();   // auxiliary U/W is visible to the main warp-group
    } else {
      ff_group_bar(gid); // W (transposed [DVh,CS]) visible within the group
    }

    if (!is_aux) {

    // ---- GEMM2: Vd = Ag @ W -> optional smem outputs for fallback paths ----
#if !(FF_VD_REG_B && FF_GVD_REG_B)
    auto sAg = fq::make_smem_swzA_cs32<FF_CS, FF_CS>(Ag_swz);
    auto sWT = fq::make_smem_tensor_h_ld<FF_DVH, FF_CS, FF_WLD>(Wswz_g);
    auto vd_acc = fq::make_acc<FF_CS, FF_DVH, MmaO, Element>(l);
    fq::flashqla_gemm_ss<decltype(sAg), decltype(sWT), decltype(vd_acc), MmaO, Element>(sAg, sWT, l, vd_acc);
    {
      auto rVd  = cute::recast<float>(vd_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
            float vd = rVd(vv, m, n);
#if !FF_VD_REG_B
            Vdswz_g[dv * FF_CS + cs]  = fq::float_to_element<Element>(vd);
#endif
#if !FF_GVD_REG_B
            gVdswz_g[dv * FF_CS + cs] = fq::float_to_element<Element>(g_rev[cs] * vd);
#endif
          }
        }
      }
    }
#endif
#if !(FF_VD_REG_B && FF_GVD_REG_B)
    ff_group_bar(gid);   // Vd/gVd fallback smem visible within group
#endif

    // ---- decay register-h (v7: before GEMM6, GEMM5+O_write) ----
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv)
          rHreg(vv, m, n) *= decay;
      }
    }

    // ---- GEMM6: h += K^T @ (g_rev*Vd) (LAST K reader) ----
    // tc01 (and tc02) now read K^T directly from the swizzled K_buf via the
    // Trans=true TSM_LD_SWZL helper -- no KT_buf materialisation / transpose.
#if FF_VD_REG_B
    // Preserve the unscaled Vd B fragments across GEMM6 so GEMM5 can consume
    // the same single-warp GEMM2 result instead of recomputing Vd.
    uint32_t vd_b_regs[2][2][4];
#endif
#if FF_GVD_REG_B
    {
      uint32_t gvd_b_regs[2][2][4];
      const int warp = l / 32;
      const int warp_n = warp / 2;
      const int lane = l & 31;
      CUTE_UNROLL
      for (int k_half = 0; k_half < 2; ++k_half) {
        CUTE_UNROLL
        for (int n_tile = 0; n_tile < FF_DVH / 32; ++n_tile) {
          const int dv_block = warp_n + n_tile * 2;
          auto vd16_acc = fq::make_acc16_1warp<Element>(lane);
          cute::clear(vd16_acc);
          CUTE_UNROLL
          for (int kk = 0; kk < 2; ++kk) {
            auto sAg16 = fq::block16_view<FF_CS>(A_buf, k_half, kk);
            auto sWT16 = fq::block16_view<FF_WLD>(Wswz_g, dv_block, kk);
            fq::gemm16_ss_1warp_acc<decltype(sAg16), decltype(sWT16), decltype(vd16_acc), Element>(
                sAg16, sWT16, lane, vd16_acc);
          }
#if FF_VD_REG_B
          fq::acc16_to_bfrag_swizzle_u32<Element>(
              vd16_acc, vd_b_regs[k_half][n_tile], lane);
#endif
          fq::scale_acc16_rows<Element>(vd16_acc, g_rev, k_half * 16, lane);
          fq::acc16_to_bfrag_swizzle_u32<Element>(vd16_acc, gvd_b_regs[k_half][n_tile], lane);
        }
      }
      fq::flashqla_gemm_swzlA_trans_regB_acc_m32<
          FF_DK, FF_CS, decltype(h_reg), Element, FF_DVH>(
          K_buf, gvd_b_regs, l, h_reg);
    }
#else
    {
      auto sVg = fq::make_smem_tensor_h_ld<FF_DVH, FF_CS, FF_WLD>(gVdswz_g);
      fq::flashqla_gemm_ss_swzlA_trans_acc_m32<FF_DK, FF_CS, decltype(sVg), decltype(h_reg), Element>(
          K_buf, sVg, l, h_reg);
    }
#endif

    // ---- GEMM5: O += Pg @ Vd ; write O[:,half] ----
#if FF_VD_REG_B
    {
      auto sPg = fq::make_smem_tensor_h_ld<FF_CS, FF_CS, FF_PGLD>(Pg_g);
      fq::flashqla_gemm_ss_regB_acc_m32<FF_DVH, FF_CS, decltype(sPg), decltype(o_acc), Element>(
          sPg, vd_b_regs, l, o_acc);
    }
#else
    auto sPg  = fq::make_smem_tensor_h_ld<FF_CS, FF_CS, FF_PGLD>(Pg_g);
    auto sVdT = fq::make_smem_tensor_h_ld<FF_DVH, FF_CS, FF_WLD>(Vdswz_g);
    fq::flashqla_gemm_ss_acc<decltype(sPg), decltype(sVdT), decltype(o_acc), MmaO, Element>(
        sPg, sVdT, l, o_acc);                                    // accumulate on scaled O_int
#endif
    if (chunk_start >= output_start) {
      Element* O_ptr = o + (int64_t)chunk_start * H * FF_DV + head_idx * FF_DV;
      cute::Tensor rO2 = cute::recast<float>(o_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
            if (cs < actual) {
              O_ptr[(int64_t)cs * H * FF_DV + (dv_base + dv)] =
                  fq::float_to_element<Element>(rO2(vv, m, n));
            }
          }
        }
      }
    }
    }
#endif  // FF_REG_CHAIN
#else
    // ================= PHASE D: per-group GEMMs on the DV half ================
    // ---- GEMM1: U = K @ hstmT_g -> W = V[:,half] - g_exp*U -> W_swz ----
    auto sK  = fq::make_smem_tensor_h<FF_CS, FF_DK>(K_buf);
    auto sH  = fq::make_smem_tensor_h_ld<FF_DVH, FF_DK, FF_HLD>(hstmT_g);
    auto u_acc = fq::make_acc<FF_CS, FF_DVH, MmaO, Element>(l);
    fq::flashqla_gemm_ss_swzlA_m32<FF_CS, FF_DK, decltype(sH), decltype(u_acc), Element>(
        K_buf, sH, l, u_acc, fq::SmemCopyAtomBTransT<Element>{});

    {
      auto rU = cute::recast<float>(u_acc);
#if defined(FLASHQLA_ARCH_100)
      auto sV_ain = ff_qkv_view_tc01<Element>(V_buf);  // AIU-native V read
#endif
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
#if defined(FLASHQLA_ARCH_100)
            float w = fq::element_to_float<Element>(
                          *reinterpret_cast<const Element*>(&sV_ain(cs, dv_base + dv)))
                      - g_exp[cs] * rU(vv, m, n);
#else
            float w = fq::element_to_float<Element>(V_buf[cs * FF_DV + (dv_base + dv)])
                      - g_exp[cs] * rU(vv, m, n);
#endif
            Wswz_g[dv * FF_WLD + cs] = fq::float_to_element<Element>(w);
          }
        }
      }
    }

    ff_group_bar(gid);   // W (transposed [DVh,CS]) visible within group before GEMM2

    // ---- GEMM2: Vd = Ag @ W -> Vd + gVd (transposed [DVh,CS]) ----
    auto sAg = fq::make_smem_swzA_cs32<FF_CS, FF_CS>(Ag_swz);
    auto sWT = fq::make_smem_tensor_h_ld<FF_DVH, FF_CS, FF_WLD>(Wswz_g);
    auto vd_acc = fq::make_acc<FF_CS, FF_DVH, MmaO, Element>(l);
    fq::flashqla_gemm_ss<decltype(sAg), decltype(sWT), decltype(vd_acc), MmaO, Element>(sAg, sWT, l, vd_acc);
    {
      auto rVd  = cute::recast<float>(vd_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
            float vd = rVd(vv, m, n);
            Vdswz_g[dv * FF_WLD + cs]  = fq::float_to_element<Element>(vd);
            gVdswz_g[dv * FF_WLD + cs] = fq::float_to_element<Element>(g_rev[cs] * vd);
          }
        }
      }
    }

    // ---- GEMM3: O_int = Q @ hstmT_g ; O_int *= scale*g_exp ----
    auto sQ = fq::make_smem_tensor_h<FF_CS, FF_DK>(Q_buf);
    auto o_acc = fq::make_acc<FF_CS, FF_DVH, MmaO, Element>(l);
    fq::flashqla_gemm_ss_swzlA_m32<FF_CS, FF_DK, decltype(sH), decltype(o_acc), Element>(
        Q_buf, sH, l, o_acc, fq::SmemCopyAtomBTransT<Element>{});
    {
      cute::Tensor rO = cute::recast<float>(o_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            int cs = cute::get<0>(tc_csdv(vv, m, n));
            rO(vv, m, n) *= scale * g_exp[cs];
          }
        }
      }
    }

    // ---- GEMM4: P = Q @ K^T -> Pg = scale*tri(exp(gi-gj))*P (DV-agnostic) ----
    using MmaP = MmaO;
    const int  lp = l;
    const bool p_active = true;
    MmaP mmaP;
    auto thrP = mmaP.get_thread_slice(lp);
    auto p_acc = fq::make_acc<FF_CS, FF_CS, MmaP, Element>(lp);
    if (p_active) {
      fq::flashqla_gemm_ss_swzlAB_m32<FF_CS, FF_CS, FF_DK, decltype(p_acc), Element>(
          Q_buf, K_buf, lp, p_acc);
    }
    auto dummyA = fq::make_smem_tensor_h<FF_CS, FF_CS>(reinterpret_cast<Element*>(16));
    if (p_active) {

      cute::Tensor cP  = cute::make_identity_tensor(
          cute::make_shape(cute::Int<FF_CS>{}, cute::Int<FF_CS>{}));
      cute::Tensor tcP = thrP.partition_C(cP);
      cute::Tensor rP  = cute::recast<float>(p_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tcP); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tcP); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tcP); ++vv) {
            auto crd = tcP(vv, m, n);
            int row = cute::get<0>(crd), col = cute::get<1>(crd);
            float pg = 0.0f;
            if (row >= col && row < actual && col < actual)
              pg = scale * ff_gate_ratio(g_log, row, col) * rP(vv, m, n);
            Pg_g[row * FF_PGLD + col] = fq::float_to_element<Element>(pg);
          }
        }
      }
    }
    (void)dummyA;
    ff_group_bar(gid);   // Pg_g + Vd_swz_g visible within group before GEMM5

    // ---- GEMM5: O += Pg @ Vd ; write O[:,half] ----
    auto sPg  = fq::make_smem_tensor_h_ld<FF_CS, FF_CS, FF_PGLD>(Pg_g);
    auto sVdT = fq::make_smem_tensor_h_ld<FF_DVH, FF_CS, FF_WLD>(Vdswz_g);
    fq::flashqla_gemm_ss_acc<decltype(sPg), decltype(sVdT), decltype(o_acc), MmaO, Element>(
        sPg, sVdT, l, o_acc);                                    // accumulate on scaled O_int
    if (chunk_start >= output_start) {
      Element* O_ptr = o + (int64_t)chunk_start * H * FF_DV + head_idx * FF_DV;
      cute::Tensor rO2 = cute::recast<float>(o_acc);
      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tc_csdv); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tc_csdv); ++n) {
          CUTE_UNROLL
          for (int vv = 0; vv < cute::size<0>(tc_csdv); ++vv) {
            auto crd = tc_csdv(vv, m, n);
            int cs = cute::get<0>(crd), dv = cute::get<1>(crd);
            if (cs < actual)
              O_ptr[(int64_t)cs * H * FF_DV + (dv_base + dv)] =
                  fq::float_to_element<Element>(rO2(vv, m, n));
          }
        }
      }
    }

    // ---- GEMM6: h += K^T @ (g_rev*Vd) (LAST K reader) ----
    {
      auto sVg = fq::make_smem_tensor_h_ld<FF_DVH, FF_CS, FF_WLD>(gVdswz_g);
      fq::flashqla_gemm_ss_swzlA_trans_acc_m32<FF_DK, FF_CS, decltype(sVg), decltype(h_reg), Element>(
#if defined(FLASHQLA_ARCH_100)
          KT_buf,
#else
          K_buf,
#endif
          sVg, l, h_reg);
    }
    // ---- decay register-h ----
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv)
          rHreg(vv, m, n) *= decay;
      }
    }
#endif  // FF_GEMM_REORDER
    }

    // ---- chunk fence + deferred prefetch ----
    __syncthreads();   // all input-buffer readers done for this chunk
#if defined(FLASHQLA_ARCH_100) && !FF_REG_CHAIN
    if (!is_aux && (c + 1) < num_chunks) {
      const int warp = tx >> 5;
      const int lane = tx & 31;
      ff_preload_chunks_split_aiu<Element, H, Hg, NGRP>(
          1, chunk_start + FF_CS, 0, seq_end, warp, lane,
          Q_all, K_all, V_all, A_all, g_all, b_all,
          q, k, v, a, g, beta, kv_head_idx, head_idx, v_token_stride);
    }
#else
    if ((c + FF_NBUF) < num_chunks) {
      const int warp = tx >> 5;
      const int lane = tx & 31;
      ff_preload_chunks_split_aiu<Element, H, Hg, NGRP>(
          1, chunk_start + FF_NBUF * FF_CS, (c + FF_NBUF) % FF_NBUF,
          seq_end, warp, lane,
          Q_all, K_all, V_all, A_all, g_all, b_all,
          q, k, v, a, g, beta, kv_head_idx, head_idx, v_token_stride);
    }
#endif
  }

  // ---- store final state ht (last CP segment only) from register-h ----
#if defined(FF_BODY_LEGACY_TC01)
  if (!is_aux && need_store_ht && ht_out != nullptr) {
    int64_t hbase = (int64_t)raw_batch_idx * H * FF_DK * FF_DV +
                    (int64_t)head_idx * FF_DK * FF_DV;
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
          auto crd = tc_dkdv(vv, m, n);
#if FF_REG_CHAIN
          int dv = cute::get<0>(crd), dk = cute::get<1>(crd);
#else
          int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
#endif
          fq::state_from_f32(ht_out[hbase + dk * FF_DV + dv_base + dv],
                             rHreg(vv, m, n));
        }
      }
    }
  }
#elif defined(FF_BODY_INDEXED_TC01)
#if FF_REG_CHAIN
  if (!is_aux && need_store_ht && raw_state != nullptr) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
          auto crd = tc_dkdv(vv, m, n);
          int dv = cute::get<0>(crd), dk = cute::get<1>(crd);
          const int64_t raw_off =
              ((int64_t)raw_state_idx * H + head_idx) * FF_DV * FF_DK +
              (int64_t)(dv_base + dv) * FF_DK + dk;
          fq::state_from_f32(raw_state[raw_off], rHreg(vv, m, n));
        }
      }
    }
  }
#else
  if (!is_aux && need_store_ht && raw_state != nullptr) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
          auto crd = tc_dkdv(vv, m, n);
          int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
          const int64_t raw_off =
              ((int64_t)raw_state_idx * H + head_idx) * FF_DV * FF_DK +
              (int64_t)(dv_base + dv) * FF_DK + dk;
          raw_state[raw_off] = rHreg(vv, m, n);
        }
      }
    }
  }
#endif
#else
  if (!is_aux && need_store_ht &&
      (ht_out != nullptr || raw_state != nullptr)) {
    int64_t hbase = (int64_t)raw_batch_idx * H * FF_DK * FF_DV +
                    (int64_t)head_idx * FF_DK * FF_DV;
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tc_dkdv); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tc_dkdv); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(tc_dkdv); ++vv) {
          auto crd = tc_dkdv(vv, m, n);
#if FF_REG_CHAIN
          int dv = cute::get<0>(crd), dk = cute::get<1>(crd);
#else
          int dk = cute::get<0>(crd), dv = cute::get<1>(crd);
#endif
          if (raw_state != nullptr) {
            const int64_t raw_off =
                ((int64_t)raw_state_idx * H + head_idx) * FF_DV * FF_DK +
                (int64_t)(dv_base + dv) * FF_DK + dk;
            fq::state_from_f32(raw_state[raw_off], rHreg(vv, m, n));
          } else {
            fq::state_from_f32(ht_out[hbase + dk * FF_DV + dv_base + dv],
                               rHreg(vv, m, n));
          }
        }
      }
    }
  }
#endif
