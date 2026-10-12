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
 * FlashKDApro PPU1.5 CuTe port — PyTorch extension binding.
 *
 * flashkdapro = flashkda + the affine-K1 / DK-split-dedicated-leader / split-V-O
 * optimizations. Exposes flashkda_fwd (full forward) and flashkda_prepare (K1).
 *
 * flashkda_fwd always runs Kernel1 (prepare) then Kernel2 (recurrence). The
 * per-shape dispatch to the "better-case-range" fast paths (register-resident
 * nows recurrence, DK/DV split, dedicated-leader) is decided inside
 * launch_flashkda_recurrence (see kernels/kernel2_recurrence.cu); the host does
 * not branch on shape. The affine-associativity K1 (P=kr^T@INV@diag(beta),
 * O=Mqk@INV@diag(beta)) is applied for the promoted shape and shrinks the
 * per-chunk serial work.
 *
 * The one dispatch decision the host DOES own is CP (context-parallel), because
 * it needs Host-visible per-sequence lengths and three extra device buffers.
 * Serving supplies trusted Host offsets and reuses a bounded plan/table cache;
 * standalone callers fall back to a varlen D2H read.  When engaged, Kernel2 is
 * replaced by the three-pass pipeline pre_scan -> merge -> segment (see the CP
 * launcher comments in kernels/kernel2_recurrence_nows.cu and kernel3_cp.cu).
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "../../../ppu_torch_stream.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "prefill/flashkdapro/common/launch_api.h"
#include "prefill/flashkdapro/common/bt64_workspace_abi.h"
#include "prefill/flashkdapro/common/runtime_arch.h"
#include "op_forge_case.h"
#include "../../sanity_check.h"
#include "../../../bypass_exec.h"
#include "sanity_checks.h"

namespace {

int g_ppu_lib_show_params = 0;

void init_show_params() {
    g_ppu_lib_show_params = pla::instrument::show_params();
}

namespace sp = pla::sanity::prefill;

// Upstream appends an aligned (N+1)-entry tile-prefix table to the workspace.
// Keep the same internal ABI in bf16 allocation units; the K1 launcher derives
// the table address from the end of the per-(head,tile) byte-image slots.
int64_t tile_prefix_units(int N, bool is_varlen) {
    if (!is_varlen) return 0;
    int64_t bytes = (int64_t(N + 1) * int64_t(sizeof(int32_t)) + 127) / 128 * 128;
    return bytes / int64_t(sizeof(at::BFloat16));
}

// ===================== CP (context-parallel) planning =====================
// CP trades total work for a shorter critical path: the recurrence is re-run
// 2.3x (a bias chain + a transition chain in pre_scan, then the real segment
// pass) but no chain is longer than one segment.  It therefore only pays off
// when the serial chain -- not the machine -- is the limit, i.e. when the plain
// grid (N, H, DV/BV) leaves the CUs starved.
//
// The gate below is a concurrency argument, not a shape whitelist:
//   chains  = N * H * (DV/BV)    independent CTAs the plain launch can offer
//   target  = CP_CTAS_PER_CU * cu_count   chains needed to saturate the CUs
//   mult    = ceil(target / chains)       how much CP would have to multiply it
// If mult is 1 the machine is already full and CP can only lose; CP is engaged
// once mult is large enough to repay the 2.3x re-run, and the segment length is
// floored so the merge chain and the per-segment fixed costs stay negligible.
//
// MEASURED CROSSOVER (PPU1.5 ZW-M890P card4, bf16, no_state == with_state to
// within noise, cp_crossover_sweep.py, oracle = best of a bv x segment-count
// sweep, quoted as best/serial for the two gate regimes slow|fast; "slow" keeps
// the pre_scan transition chain alive for the whole segment, "fast" lets it hit
// its exact zero-state early exit at once, so real traffic sits between them):
//
//   varlen, N=1 (what a Kimi-K3 B=1 prefill actually calls)
//     H=6    T=2K 1.24|1.25  4K 1.49|1.64  8K 1.73|2.20
//     H=12   T=2K 0.92|1.05  4K 1.04|1.25  8K 1.22|1.38  16K 1.33|1.51
//            T=32K 1.45|1.65
//     H=24   T=8K 0.85|0.97  16K 0.95|1.06  32K 1.01|1.10
//   fixed shape, N=1
//     H=6    T=4K 1.03|1.18  8K 1.19|1.44  16K 1.34|1.64  32K 1.58|1.83
//     H=12   T=4K 0.79|0.91  8K 0.91|1.05  16K 0.98|1.15  32K 1.11|1.25
//     H=24   T=4K 0.77|0.88  8K 0.81|0.97  16K 0.92|1.04  32K 0.99|1.08
//   varlen, N>=2
//     H=12,24 at N=2,4:  0.54-1.06 over T=4K..32K -- the fan-in already gives
//                  the plain launch N times the chains, so there is nothing to
//                  fill and CP only pays for the extra passes.
//     H=6 at N=2:  0.91|1.05 at 2K per sequence but 1.21|1.36 at 8K and
//                  1.33|1.49 at 16K -- still only 48 chains, so it wins exactly
//                  like the N=1/H=12 varlen case.  Hence the gate keys on the
//                  chain count and the per-sequence length, not on N.
//
// So CP is engaged for H<=12 only, when the plain launch offers at most 96
// chains and the longest sequence clears a floor that grows with that chain
// count (24 chains -> 2K tokens, 48 -> 4K, 96 -> 32K).  The one post-merge-tune
// exception is fixed N=1/H=12: its new BV16/256-thread merge moves the measured
// slow-decay crossover from 32K to 16K; 8K still regresses and stays off.
// H=24 is deliberately
// excluded: its best case is 1.10x in the most favourable gate regime and it
// regresses to 0.85x in the least, so there is nothing there worth the extra
// passes and the auxiliary state.
//
// CALIBRATION.  CP_CTAS_PER_CU=10 remains the generic occupancy target.  For
// Kimi-K3's production H=6/12, however, 13 segments is the PPU1.5 wave-aligned
// point: segment grid size is 13*H*4, exactly 8/16 waves on 39 CUs.  A new
// slow/fast, fixed/varlen, N=1/2 sweep found this point no worse at the capped
// short shapes and 2--16% faster at the long shapes than the old ceil(target /
// chains) result (9 segments for H12, 17 for H6).  Keep the generic formula for
// other head counts rather than extrapolating the Kimi calibration.
// CP_MIN_SEG_MULT=3 is the break-even of the 2.3x re-run plus the merge.
// PLA_FLASHKDA_CP_* overrides exist so the sweep needs no rebuild.
constexpr int KDA_CHUNK = 16;
constexpr int CP_CTAS_PER_CU = 10;
constexpr int CP_KIMI_SEG_MULT = 13;
constexpr int CP_MIN_SEG_TILES = 32;   // 512 tokens
constexpr int CP_MIN_SEG_MULT = 3;
constexpr int CP_MAX_H = 12;
constexpr int CP_FIXED_H12_MIN_SEQ_LEN = 16384;
constexpr int CP_RAGGED_H12_N2_MIN_SEQ_LEN = 6144;
constexpr int CP_RAGGED_H12_N3_MIN_SEQ_LEN = 4096;
constexpr int CP_RAGGED_WIDE_MIN_SEQ_LEN = 4096;
constexpr int64_t CP_MAX_AUX_BYTES = int64_t(1) << 30;   // chain + carries

// tc01 AUTO is a conservative additive admission path, matching FlashQLA's
// legacy-plus-TFU model: shapes outside this measured interval remain serial,
// while mode=2 keeps the existing architecture-independent force semantics.
// Exact value "0" restores the prior tc01 AUTO-off behavior for live rollback.
constexpr const char* CP_TC01_AUTO_ENV = "PLA_FLASHKDA_CP_TC01_TFU_MODEL";
constexpr int CP_TC01_H8_N1_MIN_SEQ_LEN = 4096;
constexpr int CP_TC01_H8_N1_LONG_SEQ_LEN = 16384;
constexpr int CP_TC01_H8_SHORT_BV = 16;
constexpr int CP_TC01_H8_SHORT_MULT = 2;
constexpr int CP_TC01_H8_LONG_BV = 128;
constexpr int CP_TC01_H8_LONG_MULT = 16;

// The two axes of the gate, both taken from the measurements above.
//
// cp_plain_chains is the CTA count the plain K2 launch already offers: its DV
// split runs at BV32 for varlen and BV16 for fixed-shape H<=24 (see
// kernel2_recurrence_nows.cu), i.e. 4 resp. 8 slices per (sequence, head).
// Anything wider than 96 is beyond the measured win region -- H=24 fixed (192)
// and H=12 varlen at N=4 (192) both regress -- so it never engages.
//
// cp_min_seq_len is a floor on the LONGEST SEQUENCE, not on the token total:
// what CP shortens is one sequence's chain, so a wide batch of short sequences
// is exactly the case it cannot help (N=2/H=6/2048-per-sequence measures
// 0.91-1.05x while the same 48 chains at 8192 per sequence reach 1.18-1.49x).
constexpr int CP_CHAINS_MAX = 96;

int cp_plain_chains(int H, int N, bool is_varlen) {
    return N * H * (is_varlen ? 4 : 8);
}

int cp_min_seq_len(int chains) {
    if (chains <= 24) return 2048;
    if (chains <= 48) return 4096;
    if (chains <= CP_CHAINS_MAX) return 32768;
    return INT32_MAX;
}

int cp_min_seq_len_for_shape(int H, int N, bool is_varlen, int chains) {
    // H12/N2-3 can still be serial-tail bound when one sequence dominates the
    // packed batch.  plan_cp applies the exact host-visible dominance test;
    // this coarse floor only controls whether the cu_seqlens D2H is worthwhile.
    if (is_varlen && H == 12 && N == 2)
        return CP_RAGGED_H12_N2_MIN_SEQ_LEN;
    if (is_varlen && H == 12 && N == 3)
        return CP_RAGGED_H12_N3_MIN_SEQ_LEN;
    if (is_varlen && (H == 24 || H == 48) && N >= 2 && N <= 3)
        return CP_RAGGED_WIDE_MIN_SEQ_LEN;
    if (is_varlen && H == 12 && N > 3) return INT32_MAX;
    if (!is_varlen && N == 1 && H == 12)
        return CP_FIXED_H12_MIN_SEQ_LEN;
    return cp_min_seq_len(chains);
}

struct CpPlan {
    bool engage = false;
    bool fma_front = false;
    int n_seg = 0;
    int bv = 32;
    // BT64-aligned schedule for the composed BT64xCP pipeline: every non-last
    // segment length is a multiple of 4 CH tiles (one BT64 block), and seg_bt
    // carries the per-segment BT64 descriptor consumed by the segment kernel.
    bool bt64_align = false;
    std::vector<int32_t> seg_tbl;   // n_seg * KDA_SEG_STRIDE
    std::vector<int32_t> seq_seg;   // N * 2 = {first segment, segment count}
    std::vector<int32_t> seg_bt;    // n_seg * KDA_SEG_BT_STRIDE (bt64_align)
};

int cp_env_int(const char* name, int fallback) {
    const char* e = getenv(name);
    return (e && e[0]) ? atoi(e) : fallback;
}

// K1 and K2 share a private workspace ABI. Decide its physical layout and
// raw/affine value semantics once, after the authoritative CP plan is known,
// and pass that immutable decision to both launchers. Diagnostic environments
// are changed only between calls; downstream getenv reads cannot split the
// producer and consumer contracts after this selection.
flashkda::KrWorkspaceLayout select_kr_workspace_layout(
    int T_total, int H, int N, int total_tiles, bool is_varlen,
    const CpPlan& cp) {
    using Layout = flashkda::KrWorkspaceLayout;
    if (cp.engage) return Layout::kNormalRaw;

    const bool is_tc01 =
        flashkda::host_runtime::current_device_is_tc01();
    // Freeze the established native affine value ABI on both architectures.
    // Packed H12/N1 is affine only on tc01; fixed H6/H12 is affine on both.
    const bool native_affine_shape = N == 1 && T_total >= 4096
        && ((!is_varlen && (H == 6 || H == 12))
            || (is_tc01 && is_varlen && H == 12));
    const char* enows = getenv("PLA_FLASHKDA_K2_NOWS");
    const char* edv = getenv("PLA_FLASHKDA_K2_DVSPLIT");
    const char* edk = getenv("PLA_FLASHKDA_K2_DKSPLIT");
    const char* eleader = getenv("PLA_FLASHKDA_K2_DKLEADER");
    const bool nows_ok = !(enows && enows[0] && atoi(enows) == 0);
    const bool affine_schedule_ok = native_affine_shape
        && cp_env_int("PLA_FLASHKDA_K1_THREADS", 128) == 128
        && nows_ok
        && (!(edv && edv[0]) || atoi(edv) == 16)
        && (!(edk && edk[0]) || atoi(edk) == 4)
        && (!(eleader && eleader[0]) || atoi(eleader) != 0);
    const Layout native_layout = affine_schedule_ok
        ? Layout::kNormalAffineP : Layout::kNormalRaw;

    if (!is_tc01) return native_layout;

    // The new contiguous affine-P layout is initially promoted for the packed
    // H12/N1 production ABI. Fixed shapes retain their validated native image.
    if (!is_varlen) return native_layout;

    // H12/N1 already uses the affine P=kr^T*INV*diag(beta) producer and the
    // BV16/DK4/dedicated-leader consumer. Store P as contiguous [DK,CH] without
    // changing its BF16 values or arithmetic. This path is independent of the
    // plain K1 FMA-front policy.
    const bool affine_shape = native_affine_shape;

    // Unset selects the proven automatic contiguous layouts; zero preserves
    // the established native physical image. A nonzero force request never
    // broadens either safe shape gate.
    if (cp_env_int("PLA_FLASHKDA_TC01_KR_TRANSPOSE", 1) == 0)
        return native_layout;

    if (affine_shape && affine_schedule_ok)
        return Layout::kContiguousAffineP;

    // The short packed H12/N1 rows are recurrence-latency bound but below the
    // affine crossover. A complete 105-case paired replay promoted BV16/DK2:
    // every row wins (minimum 1.052x, roughly 2x at 3--4K tokens). Keep raw
    // arithmetic and only materialize kr^T contiguously for that exact schedule.
    const bool short_h12_n1 = N == 1 && H == 12 && T_total < 4096;
    if (short_h12_n1) {
        const bool threads_ok =
            cp_env_int("PLA_FLASHKDA_K1_THREADS", 128) == 128;
        const bool bv16_ok = !(edv && edv[0]) || atoi(edv) == 16;
        const bool dk2_ok = !(edk && edk[0]) || atoi(edk) == 2;
        const bool no_leader =
            !(eleader && eleader[0]) || atoi(eleader) == 0;
        if (threads_ok && nows_ok && bv16_ok && dk2_ok && no_leader)
            return Layout::kContiguousRawKrT;
        return native_layout;
    }

    // Long H96/N<=3 recurrence remains dominated by the same strided kr^T
    // fragment loads. Keep each established schedule (N1 BV16/DK1, N2/N3
    // full-DV) and change only the private A image. The complete T>=8192
    // replay promoted this exact fan-in interval: 389/389 rows won, whereas
    // N>=4 was effectively flat and included small regressions.
    const bool long_h96 = H == 96 && N <= 3 && T_total >= 8192;
    if (long_h96) {
        const bool threads_ok =
            cp_env_int("PLA_FLASHKDA_K1_THREADS", 128) == 128;
        // N1 retains its established BV16/DK1 split; N>=2 retains full DV.
        const int expected_bv = N == 1 ? 16 : 0;
        const bool dv_ok = !(edv && edv[0]) || atoi(edv) == expected_bv;
        const bool dk_ok = !(edk && edk[0]) || atoi(edk) == 1;
        const bool no_leader =
            !(eleader && eleader[0]) || atoi(eleader) == 0;
        if (threads_ok && nows_ok && dv_ok && dk_ok && no_leader)
            return Layout::kContiguousRawKrT;
        return native_layout;
    }

    // These full-DV/NT128/DK1 rows keep their native schedule and change only
    // the private kr image.  Complete paired replays promoted three intervals:
    // H48/N3: 185/185, 1.0972x aggregate, 1.0211x minimum;
    // H24/N3: 160/160, 1.0918x aggregate, 1.0748x minimum;
    // H12/N9: 40/40 accuracy at warmup5/repeat20, 1.0287x aggregate, with
    //          the lone nominal loss measuring 0.012%.  Adjacent N stayed flat
    //          or contained material regressions and retain the native image.
    const bool promoted_full_dv_contiguous =
        (H == 48 && N == 3 && T_total >= 8192)
        || (H == 24 && N == 3 && T_total > 8192)
        || (H == 12 && N == 9 && T_total > 8192);
    if (promoted_full_dv_contiguous) {
        const bool threads_ok =
            cp_env_int("PLA_FLASHKDA_K1_THREADS", 128) == 128;
        const bool full_dv_ok =
            cp_env_int("PLA_FLASHKDA_K2_DVSPLIT", 0) == 0;
        const bool dk1_ok =
            cp_env_int("PLA_FLASHKDA_K2_DKSPLIT", 1) == 1;
        const bool no_leader =
            cp_env_int("PLA_FLASHKDA_K2_DKLEADER", 0) == 0;
        if (threads_ok && nows_ok && full_dv_ok && dk1_ok && no_leader)
            return Layout::kContiguousRawKrT;
        return native_layout;
    }

    // These five schedules are the long low-concurrency rows whose K2 state
    // update repeatedly reads kr^T through tc01's costly 256-byte column
    // stride.  Each maps to one existing nows specialization below.
    const bool bv32_dk2 = N == 2 && (H == 12 || H == 48);
    const bool bv32_dk1 = N == 3 && H == 12;
    const bool bv16_dk1 = (N == 2 && H == 24) || (N == 1 && H == 48);
    const bool target_shape = is_varlen && T_total >= 8192
        && (bv32_dk2 || bv32_dk1 || bv16_dk1)
        && int64_t(total_tiles) * int64_t(H) >= 384;
    if (!target_shape) return native_layout;

    // The transposed producer exists only in the 128-thread/CPC1 FMA front.
    // Respect every override that can steer K1 away from that specialization.
    if (cp_env_int("PLA_FLASHKDA_K1_FMA_SPLIT", 1) == 0
        || cp_env_int("PLA_FLASHKDA_K1_THREADS", 128) != 128
        || cp_env_int("PLA_FLASHKDA_K1_CPC", 1) != 1)
        return native_layout;

    // Each shape maps to one exact tc01 varlen nows BV/DK specialization. Keep
    // explicit K2 controls authoritative and fall back to the native raw ABI
    // whenever a caller asks for another schedule.
    const int expected_bv = bv16_dk1 ? 16 : 32;
    const int expected_dk = bv32_dk2 ? 2 : 1;
    if (!nows_ok
        || cp_env_int("PLA_FLASHKDA_K2_DVSPLIT", expected_bv) != expected_bv
        || cp_env_int("PLA_FLASHKDA_K2_DKSPLIT", expected_dk) != expected_dk
        || cp_env_int("PLA_FLASHKDA_K2_DKLEADER", 0) != 0)
        return native_layout;
    return Layout::kContiguousRawKrT;
}

bool use_tc01_bt64_candidate(
    int T_total, int H, int N, bool is_varlen,
    const CpPlan& cp, flashkda::KrWorkspaceLayout kr_layout) {
    const char* env = std::getenv("PLA_FLASHKDA_TC01_BT64");
    const bool opt_in = env && env[0] == '1' && env[1] == '\0';
    // When unset, auto-enable the validated GLM5.3 H8/H4, N1 ranges; an explicit value other than 1 keeps it off.
    if (env && !opt_in) return false;
    if (!flashkda::host_runtime::current_device_is_tc01()) return false;
    const bool cp_forced = cp_env_int("PLA_FLASHKDA_CP", 1) == 2;
    // GLM5.3 TP8 batch=1: every case with T>=256 is faster than the existing CH16/CP schedule; forced CP keeps its original semantics.
    const bool glm_h8_n1_auto = H == 8 && N == 1 && T_total >= 256
        && !cp_forced;
    // TP16 keeps the plain BT64 recurrence; the P1 state layout is enabled only on the TP8 entry.
    const bool glm_h4_n1_auto = H == 4 && N == 1 && T_total >= 256
        && !cp_forced;
    // H8 packed N2-4 and K3 H12/H24 N4 are still experimental entries, enabled only when BT64=1.
    const bool h8_packed_probe = H == 8 && N >= 2 && N <= 4 && T_total >= 64
        && !cp_forced;
    const bool k3_h12_n4_probe = H == 12 && N == 4 && T_total >= 8192
        && !cp.engage && !cp_forced;
    const bool k3_h24_n4_probe = H == 24 && N == 4 && T_total >= 8192
        && !cp.engage && !cp_forced;
    if (!is_varlen) return false;
    if (!(glm_h8_n1_auto || glm_h4_n1_auto
            || (opt_in && (h8_packed_probe || k3_h12_n4_probe || k3_h24_n4_probe))))
        return false;
    // Candidate 1 consumes the raw K1 value ABI. Both tc01 physical images are
    // supported by its streamed Kr loader; affine P/O has already overwritten
    // INV/beta and is therefore not a valid producer.
    return kr_layout == flashkda::KrWorkspaceLayout::kNormalRaw
        || kr_layout == flashkda::KrWorkspaceLayout::kContiguousRawKrT;
}

// Snapshot every environment-controlled planner input once per invocation.
// Besides making cache keys complete, this prevents one launch from observing
// a mixture of old and new values when an A/B harness updates the overrides.
struct CpConfig {
    int mode = 1;
    int bv = 32;
    int min_seg_tiles = CP_MIN_SEG_TILES;
    int max_h = CP_MAX_H;
    bool balance = false;
    bool mult_set = false;
    int mult = 0;
    bool tc01_auto_enabled = true;
    bool tc01_h8_n1_candidate = false;
    bool tc01_h8_n1_auto = false;
    bool k1_fma_enabled = true;
    int k1_threads = 128;
    int k1_cpc = 1;
    // Requested PLA_FLASHKDA_TC01_BT64_CP switch and its effective form: the
    // BT64-aligned segment schedule only exists inside the tc01 H8/N1 AUTO
    // candidate window.
    bool bt64_cp = false;
    bool bt64_align = false;

    bool operator==(const CpConfig& other) const {
        return mode == other.mode && bv == other.bv
            && min_seg_tiles == other.min_seg_tiles
            && max_h == other.max_h && balance == other.balance
            && mult_set == other.mult_set
            && (!mult_set || mult == other.mult)
            && tc01_auto_enabled == other.tc01_auto_enabled
            && tc01_h8_n1_candidate == other.tc01_h8_n1_candidate
            && tc01_h8_n1_auto == other.tc01_h8_n1_auto
            && k1_fma_enabled == other.k1_fma_enabled
            && k1_threads == other.k1_threads
            && k1_cpc == other.k1_cpc
            && bt64_cp == other.bt64_cp
            && bt64_align == other.bt64_align;
    }
};

CpConfig cp_config_from_env() {
    CpConfig c;
    c.mode = cp_env_int("PLA_FLASHKDA_CP", 1);
    c.bv = cp_env_int("PLA_FLASHKDA_CP_BV", 32);
    if (c.bv != 16 && c.bv != 32 && c.bv != 128) c.bv = 32;
    c.min_seg_tiles = cp_env_int(
        "PLA_FLASHKDA_CP_SEG_TILES", CP_MIN_SEG_TILES);
    c.max_h = cp_env_int("PLA_FLASHKDA_CP_MAX_H", CP_MAX_H);
    c.balance = cp_env_int("PLA_FLASHKDA_CP_BALANCE", 0) != 0;
    const char* forced_mult = getenv("PLA_FLASHKDA_CP_MULT");
    c.mult_set = forced_mult != nullptr && forced_mult[0] != '\0';
    if (c.mult_set) c.mult = atoi(forced_mult);
    const char* tc01_auto = getenv(CP_TC01_AUTO_ENV);
    c.tc01_auto_enabled = !(
        tc01_auto != nullptr && tc01_auto[0] == '0'
        && tc01_auto[1] == '\0');
    const char* k1_fma = getenv("PLA_FLASHKDA_K1_FMA_SPLIT");
    c.k1_fma_enabled = !(k1_fma && k1_fma[0] && atoi(k1_fma) == 0);
    c.k1_threads = cp_env_int("PLA_FLASHKDA_K1_THREADS", 128);
    c.k1_cpc = cp_env_int("PLA_FLASHKDA_K1_CPC", 1);
    // Exact "1" opts the tc01 H8/N1 BT64 pipeline into the CP-composed segment
    // schedule; PLA_FLASHKDA_CP_MULT then sweeps the segment count.
    const char* bt64_cp = getenv("PLA_FLASHKDA_TC01_BT64_CP");
    c.bt64_cp = bt64_cp && bt64_cp[0] == '1' && bt64_cp[1] == '\0';
    return c;
}

CpConfig effective_cp_config(
    flashkda::host_runtime::PpuArchitecture architecture,
    int max_seq_len, int H, int N, bool is_varlen,
    const CpConfig& requested) {
    CpConfig effective = requested;
    const bool tc01_h8_n1_candidate =
        requested.mode == 1 && requested.tc01_auto_enabled
        && requested.k1_fma_enabled && requested.k1_threads == 128
        && requested.k1_cpc == 1
        && architecture == flashkda::host_runtime::PpuArchitecture::kTc01
        && is_varlen && H == 8 && N == 1;
    if (!tc01_h8_n1_candidate) return effective;

    effective.tc01_h8_n1_candidate = true;
    if (max_seq_len < CP_TC01_H8_N1_MIN_SEQ_LEN) return effective;

    effective.tc01_h8_n1_auto = true;
    effective.bt64_align = requested.bt64_cp;
    // The composed BT64xCP pipeline sweeps PLA_FLASHKDA_CP_MULT explicitly:
    // an operator-provided mult wins over the AUTO model, while bv keeps
    // following the sequence-length model for the CH16 pre_scan pass.
    if (requested.bt64_cp && requested.mult_set) {
        effective.bv = max_seq_len < CP_TC01_H8_N1_LONG_SEQ_LEN
            ? CP_TC01_H8_SHORT_BV : CP_TC01_H8_LONG_BV;
        return effective;
    }
    effective.mult_set = true;
    if (max_seq_len < CP_TC01_H8_N1_LONG_SEQ_LEN) {
        effective.bv = CP_TC01_H8_SHORT_BV;
        effective.mult = CP_TC01_H8_SHORT_MULT;
    } else {
        effective.bv = CP_TC01_H8_LONG_BV;
        effective.mult = CP_TC01_H8_LONG_MULT;
    }
    return effective;
}

struct CpPlanKey {
    int device = -1;
    int architecture = -1;
    int cu_count = 0;
    int T_total = 0;
    int H = 0;
    int N = 0;
    bool is_varlen = false;
    CpConfig config;
    std::vector<int64_t> raw_cu;

    bool operator==(const CpPlanKey& other) const {
        return device == other.device && architecture == other.architecture
            && cu_count == other.cu_count && T_total == other.T_total
            && H == other.H && N == other.N
            && is_varlen == other.is_varlen && config == other.config
            && raw_cu == other.raw_cu;
    }
};

using CpStreamKey = std::pair<int, int64_t>;

struct CachedCpPlan {
    CpPlan plan;
    torch::Tensor seg_tbl_d;
    torch::Tensor seq_seg_d;
    torch::Tensor seg_bt_d;         // defined only for bt64_align plans
    std::shared_ptr<at::cuda::CUDAEvent> ready_event;
    std::set<CpStreamKey> active_streams;
    std::mutex stream_mutex;
};

struct CpPlanCacheEntry {
    CpPlanKey key;
    std::shared_ptr<CachedCpPlan> value;
};

struct CpPlanCacheState {
    std::mutex mutex;
    std::vector<CpPlanCacheEntry> entries;  // MRU first
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t d2h_reads = 0;
    uint64_t h2d_tables = 0;
};

// Intentionally process-lifetime: destroying CUDA tensors from a DSO static
// after the CUDA runtime has started shutting down is not safe.  Capacity is
// tiny and the process releases the retained allocations at exit.
CpPlanCacheState& cp_plan_cache_state() {
    static auto* state = new CpPlanCacheState();
    return *state;
}

int cp_plan_cache_capacity() {
    return std::max(0, std::min(
        cp_env_int("PLA_FLASHKDA_CP_PLAN_CACHE_SIZE", 4), 16));
}

// seq_len_host must hold N lengths; tok_begin_host the N absolute token offsets.
CpPlan plan_cp(const std::vector<int64_t>& seq_len,
               const std::vector<int64_t>& tok_begin,
               int T_total, int H, bool is_varlen, int cu_count,
               const CpConfig& config) {
    CpPlan p;
    const int N = int(seq_len.size());
    // 0 = off, 1 = auto (default), 2 = force whenever a split is possible.
    const int mode = config.mode;
    if (mode == 0 || N == 0) return p;

    // BV32 unconditionally.  This used to mirror the plain K2's auto_bv, which
    // picks BV16 for fixed-shape H<=24 -- but BV16 was measurably the wrong tile
    // for CP: on the fixed-shape sweep it dragged AUTO down to 0.77-0.86x of
    // serial while the same shapes reach 1.11-1.25x at BV32, and BV32 was the
    // oracle's choice on every point of every sweep.  CP has one third of the
    // chains per pass to hide, so it wants the wider tile, not the finer split.
    p.bv = config.bv;

    std::vector<int> seq_tiles(N);
    std::vector<int> ws_tile_base(N);   // == K1's tile_prefix (exclusive scan)
    // Independent ceil(seq_len/BT64) exclusive scan.  The BT64 block index of
    // a segment cannot be derived from the CH16 workspace prefix because every
    // sequence tail is ragged against the 64-token grid.
    std::vector<int> seq_bt_base(N);
    int64_t tile_sum = 0;
    int64_t bt_sum = 0;
    int max_seq_tiles = 0;
    for (int i = 0; i < N; ++i) {
        seq_tiles[i] = int((seq_len[i] + KDA_CHUNK - 1) / KDA_CHUNK);
        ws_tile_base[i] = int(tile_sum);
        tile_sum += seq_tiles[i];
        seq_bt_base[i] = int(bt_sum);
        bt_sum += (seq_len[i] + flashkda::BT64_CHUNK - 1)
            / flashkda::BT64_CHUNK;
        max_seq_tiles = std::max(max_seq_tiles, seq_tiles[i]);
    }
    const bool ragged_small_batch = is_varlen && N >= 2 && N <= 3
        && (H == 12 || H == 24 || H == 48);
    // The H12/N3 serial path remains below the 1.20 acceptance target even for
    // balanced triples, so all sufficiently long triples use CP.  H24/H48
    // already have enough plain chains for balanced batches; only their
    // measured dominant-tail window benefits after paying the CP passes.
    const int64_t max_seq_tokens = int64_t(max_seq_tiles) * KDA_CHUNK;
    const bool ragged_cp_shape = ragged_small_batch && (
        (H == 12 && (N == 3 || max_seq_tokens * 5 >= int64_t(T_total) * 3))
        || (H != 12 && N == 2
            && max_seq_tokens * 5 >= int64_t(T_total) * 4)
        || (H != 12 && N == 3
            && max_seq_tokens * 20 >= int64_t(T_total) * 13));

    const int zsplit = 128 / p.bv;
    const int64_t chains = int64_t(N) * H * zsplit;
    const int64_t target = int64_t(CP_CTAS_PER_CU) * std::max(cu_count, 1);
    int mult = int((target + chains - 1) / chains);
    if (H == 6 || H == 12) mult = CP_KIMI_SEG_MULT;
    const int min_seg_tiles = config.min_seg_tiles;
    // Never shorten the longest sequence past the segment floor.
    mult = std::min(mult, std::max(1, max_seq_tiles / std::max(min_seg_tiles, 1)));
    if (mode == 1 && ragged_cp_shape && !config.mult_set) {
        // Proportional segmentation below turns `mult` into a target segment
        // length.  Pick it so the resulting segment CTA grid lands near the
        // measured complete-wave points.  The optimal total-segment target
        // falls as H grows because every segment already fans out over H*4
        // BV32 chains: H12/N3=13, H24=9, and H48=8.  Thirteen also avoids
        // the 11-vs-13 equidistant tie seen on two-nearly-equal H12 triples.
        // This automatically chooses 5 for a 97/3 split, 4 for 68/32, and 6
        // for a representative 64/26/10 split instead of a shape whitelist.
        // Short (~8K-token) N2 chains need one more complete resident wave to
        // amortize the fixed CP passes; long N2 chains are fastest at six.
        const int target_segments = H == 12
            ? (N == 2 ? (max_seq_tiles <= 512 ? 10 : 6) : 13)
            : (H == 24 ? 9 : 8);
        int best_mult = 2, best_distance = INT32_MAX;
        const int max_mult = std::min(
            CP_KIMI_SEG_MULT,
            std::max(2, max_seq_tiles / std::max(min_seg_tiles, 1)));
        for (int candidate = 2; candidate <= max_mult; ++candidate) {
            const int target_tiles = std::max(
                min_seg_tiles,
                (max_seq_tiles + candidate - 1) / candidate);
            int predicted_segments = 0;
            for (int tiles : seq_tiles) {
                const int desired = std::max(
                    1, (tiles + target_tiles - 1) / target_tiles);
                const int floor_cap = std::max(
                    1, tiles / std::max(min_seg_tiles, 1));
                predicted_segments += std::min(
                    candidate, std::min(desired, floor_cap));
            }
            const int distance = std::abs(predicted_segments - target_segments);
            if (distance < best_distance) {
                best_distance = distance;
                best_mult = candidate;
            }
        }
        mult = best_mult;
    }
    if (config.mult_set) mult = config.mult;
    if (mult < 2) return p;
    // AUTO only engages inside the measured win region: the plain launch must be
    // narrow enough to leave the CUs starved, the longest sequence long enough
    // for the split to repay the extra passes, and the head count inside the
    // range where CP was measured to win at all.  mode==2 (the sweep's force)
    // keeps only the structural mult>=2 requirement.
    const int plain_chains = cp_plain_chains(H, N, is_varlen);
    const int min_seq_len = cp_min_seq_len_for_shape(
        H, N, is_varlen, plain_chains);
    if (mode == 1 && config.tc01_h8_n1_candidate
        && !config.tc01_h8_n1_auto)
        return p;
    if (mode == 1
        && ((H > config.max_h
             && !ragged_cp_shape)
            || (plain_chains > CP_CHAINS_MAX && !ragged_cp_shape)
            || (ragged_small_batch && !ragged_cp_shape)
            || int64_t(max_seq_tiles) * KDA_CHUNK < min_seq_len
            || (!config.tc01_h8_n1_auto
                && mult < CP_MIN_SEG_MULT)))
        return p;

    // ---- segmentation ----
    // Split each sequence into whole CH tiles; only its LAST segment may end on
    // a partial tile, which is exactly the invariant the segment kernel's
    // `actual = min(CH, seg_len - t*CH)` tail handling expects.
    //
    // The proportional mode is an A/B-able load-balancing planner for ragged
    // batches.  The old planner assigns the same segment count to every long
    // sequence, which needlessly re-runs a 10K tail five times merely because a
    // different sequence is 22K.  Instead, derive a target segment length from
    // the longest chain and give each sequence only enough segments to meet it.
    // The min-segment cap keeps every segment amortized just as before.
    const bool proportional = ragged_cp_shape || mode == 2 || config.balance;
    const int target_seg_tiles = std::max(
        min_seg_tiles, (max_seq_tiles + mult - 1) / mult);
    // CH16 tiles per BT64 block; the bt64_align quantization unit.
    constexpr int kBt64Tiles = flashkda::BT64_CHUNK / KDA_CHUNK;
    static_assert(flashkda::BT64_CHUNK % KDA_CHUNK == 0,
                  "BT64 blocks must be whole CH tiles");
    p.seq_seg.resize(size_t(N) * 2);
    for (int i = 0; i < N; ++i) {
        const int tiles = seq_tiles[i];
        const int max_segs_by_floor = std::max(
            1, tiles / std::max(min_seg_tiles, 1));
        const int desired_segs = proportional
            ? std::max(1, (tiles + target_seg_tiles - 1) / target_seg_tiles)
            : mult;
        int segs = std::min(mult, std::min(desired_segs, max_segs_by_floor));
        p.seq_seg[size_t(i) * 2 + 0] =
            int32_t(p.seg_tbl.size() / flashkda::KDA_SEG_STRIDE);
        p.seq_seg[size_t(i) * 2 + 1] = int32_t(segs);
        const int base = tiles / segs, rem = tiles % segs;
        int tile_off = 0;
        for (int s = 0; s < segs; ++s) {
            const bool last = (s == segs - 1);
            int seg_tiles = base + (s < rem ? 1 : 0);
            if (config.bt64_align && !last) {
                // Quantize every non-last segment to whole BT64 blocks.  The
                // floor keeps the CP invariant: segs <= tiles/min_seg_tiles
                // gives a remaining average >= min_seg_tiles (32) tiles per
                // segment at every step, the round-down to a 4-tile multiple
                // drops at most 3, and induction on tile_off keeps the tail
                // segment non-empty above the same floor.
                const int left = segs - s;
                const int share = (tiles - tile_off) / left;
                seg_tiles = std::max(kBt64Tiles, share / kBt64Tiles * kBt64Tiles);
            }
            const int64_t begin = tok_begin[i] + int64_t(tile_off) * KDA_CHUNK;
            const int64_t len = last
                ? seq_len[i] - int64_t(tile_off) * KDA_CHUNK
                : int64_t(seg_tiles) * KDA_CHUNK;
            const size_t seg_base = p.seg_tbl.size();
            p.seg_tbl.resize(seg_base + flashkda::KDA_SEG_STRIDE);
            p.seg_tbl[seg_base + flashkda::KDA_SEG_TOK] = int32_t(begin);
            p.seg_tbl[seg_base + flashkda::KDA_SEG_LEN] = int32_t(len);
            // Absolute workspace tile index.  K1 lays tiles out as
            // tile_prefix[seq] + local_tile for varlen and 0 + tile for N==1,
            // and tile_prefix is the same exclusive ceil-div scan built here.
            p.seg_tbl[seg_base + flashkda::KDA_SEG_WSBASE] =
                int32_t(ws_tile_base[i] + tile_off);
            // Slot in final_state, or -1 to skip the 32 KiB state store.  It
            // doubles as pre_scan's "this segment has no consumer" marker.
            p.seg_tbl[seg_base + flashkda::KDA_SEG_OUTSLOT] =
                last ? int32_t(i) : int32_t(-1);
            if (config.bt64_align) {
                // tile_off is always a whole BT64 block here: it starts at 0
                // and every preceding segment length was quantized, so the
                // local BT64 block index is exact.
                const size_t bt_desc = p.seg_bt.size();
                p.seg_bt.resize(bt_desc + flashkda::KDA_SEG_BT_STRIDE);
                p.seg_bt[bt_desc + flashkda::KDA_SEG_BT_TOK] = int32_t(begin);
                p.seg_bt[bt_desc + flashkda::KDA_SEG_BT_LEN] = int32_t(len);
                p.seg_bt[bt_desc + flashkda::KDA_SEG_BT_WSBASE] =
                    int32_t(ws_tile_base[i] + tile_off);
                p.seg_bt[bt_desc + flashkda::KDA_SEG_BT_BTBASE] =
                    int32_t(seq_bt_base[i] + tile_off / kBt64Tiles);
                p.seg_bt[bt_desc + flashkda::KDA_SEG_BT_OUTSLOT] =
                    last ? int32_t(i) : int32_t(-1);
            }
            tile_off += seg_tiles;
        }
    }
    p.n_seg = int(p.seg_tbl.size() / flashkda::KDA_SEG_STRIDE);
    if (p.n_seg <= N) { p.n_seg = 0; p.seg_tbl.clear(); p.seq_seg.clear();
                        p.seg_bt.clear(); return p; }

    // chain_state [2,n_seg,H,128,128] + carries [n_seg,H,128,128] bf16.
    const int64_t aux_bytes =
        int64_t(3) * p.n_seg * H * 128 * 128 * int64_t(sizeof(at::BFloat16));
    if (aux_bytes > CP_MAX_AUX_BYTES) {
        p.n_seg = 0; p.seg_tbl.clear(); p.seq_seg.clear();
        p.seg_bt.clear(); return p;
    }
    p.engage = true;
    p.fma_front = config.tc01_h8_n1_auto;
    p.bt64_align = config.bt64_align;
    return p;
}

// PPU CU count is an occupancy input to the CP segment planner only.  It is
// deliberately independent of architecture detection, which uses the runtime
// compute-capability attributes in common/runtime_arch.h.
int cp_cu_count() {
    const int device = flashkda::host_runtime::current_device();
    auto query = [device]() {
        const int count = flashkda::host_runtime::device_attribute(
            device, hggcDevAttrMultiProcessorCount,
            "hggcDeviceGetAttribute(hggcDevAttrMultiProcessorCount)");
        if (count <= 0) {
            throw std::runtime_error(
                std::string("FlashKDApro CP planner: logical PPU device ")
                + std::to_string(device)
                + " reported invalid multiprocessor count "
                + std::to_string(count));
        }
        return count;
    };
    if (device < 0 || device >= flashkda::host_runtime::kMaxCachedDevices) {
        return query();
    }
    static std::once_flag queried[
        flashkda::host_runtime::kMaxCachedDevices];
    static int cached[flashkda::host_runtime::kMaxCachedDevices] = {};
    std::call_once(queried[device], [&]() { cached[device] = query(); });
    return cached[device];
}

// The complete host-only candidate gate.  Raw offsets are only resolved after
// this passes, so non-target shapes never pay a varlen cu_seqlens D2H read.
bool cp_dispatch_candidate(
    int T_total, int H, int N, bool is_varlen,
    flashkda::host_runtime::PpuArchitecture architecture,
    const CpConfig& config) {
    const int mode = config.mode;
    // tc01 AUTO is an additive, measured gate. The prior blanket-disable remains
    // available through the exact-zero rollback, and every non-H8/N1 shape stays
    // serial. mode==2 remains the architecture-independent diagnostic override.
    if (mode == 1
        && architecture == flashkda::host_runtime::PpuArchitecture::kTc01) {
        return config.tc01_auto_enabled && config.k1_fma_enabled
            && config.k1_threads == 128 && config.k1_cpc == 1
            && is_varlen && H == 8 && N == 1
            && T_total >= CP_TC01_H8_N1_MIN_SEQ_LEN;
    }
    const bool candidate = mode != 0 && T_total > 0
        && (mode == 2
            ? T_total >= 2 * KDA_CHUNK
            // T_total >= the per-sequence floor is a necessary condition for the
            // real max_seq_len test in plan_cp, so this only skips the varlen
            // cu_seqlens D2H read for shapes that could never engage.
            : ((H <= config.max_h
                || (is_varlen && N >= 2 && N <= 3
                    && (H == 12 || H == 24 || H == 48)))
               && (cp_plain_chains(H, N, is_varlen) <= CP_CHAINS_MAX
                   || (is_varlen && N >= 2 && N <= 3
                       && (H == 12 || H == 24 || H == 48)))
               && T_total >= cp_min_seq_len_for_shape(
                   H, N, is_varlen, cp_plain_chains(H, N, is_varlen))));
    return candidate;
}

CpPlan plan_cp_dispatch(
    int T_total, int H, int N, bool is_varlen,
    const CpConfig& config,
    const std::function<void(std::vector<int64_t>&,
                             std::vector<int64_t>&)>& fill) {
    CpPlan cp;
    const auto architecture = flashkda::host_runtime::current_architecture();
    if (!cp_dispatch_candidate(
            T_total, H, N, is_varlen, architecture, config))
        return cp;
    std::vector<int64_t> seq_len(N), tok_begin(N);
    fill(seq_len, tok_begin);
    const int64_t max_seq_len64 = *std::max_element(
        seq_len.begin(), seq_len.end());
    const int max_seq_len = int(std::min<int64_t>(
        max_seq_len64, std::numeric_limits<int>::max()));
    const CpConfig effective = effective_cp_config(
        architecture, max_seq_len, H, N, is_varlen, config);
    const int cu = cp_cu_count();
    return plan_cp(
        seq_len, tok_begin, T_total, H, is_varlen, cu, effective);
}

void note_cp_cache_d2h() {
    auto& state = cp_plan_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.d2h_reads;
}

std::vector<int64_t> resolve_raw_cu(
    int T_total, int N, bool is_varlen, const torch::Tensor& cu_t,
    const c10::optional<std::vector<int64_t>>& raw_cu_opt) {
    if (!is_varlen) return {0, int64_t(T_total)};
    if (raw_cu_opt.has_value()) {
        const auto& raw_cu = raw_cu_opt.value();
        TORCH_CHECK(
            raw_cu.size() == size_t(N + 1),
            "raw_cu_seqlens must have N+1 entries, got ", raw_cu.size(),
            " for N=", N);
        return raw_cu;
    }

    if (pla::sanity::enabled() && cu_t.is_cuda()) {
        const pla::sanity::Context sanity("flashkda CP planning", cu_t);
        sanity.check(sanity.values_enabled(),
                     "CUDA graph capture requires raw_cu_seqlens host metadata for CP planning");
    }
    note_cp_cache_d2h();
    auto cu_cpu = cu_t.to(torch::kCPU);
    std::vector<int64_t> raw_cu(size_t(N + 1));
    if (cu_cpu.dtype() == torch::kInt32) {
        const int32_t* p = cu_cpu.data_ptr<int32_t>();
        for (int i = 0; i <= N; ++i) raw_cu[size_t(i)] = p[i];
    } else {
        const int64_t* p = cu_cpu.data_ptr<int64_t>();
        for (int i = 0; i <= N; ++i) raw_cu[size_t(i)] = p[i];
    }
    return raw_cu;
}

void activate_cached_cp_plan(
    const std::shared_ptr<CachedCpPlan>& cached,
    const c10::cuda::CUDAStream& stream) {
    if (!cached->plan.engage || !cached->ready_event) return;
    const CpStreamKey key{
        int(stream.device_index()), int64_t(stream.id())};
    std::lock_guard<std::mutex> lock(cached->stream_mutex);
    if (cached->active_streams.count(key)) return;

    // Event wait orders the descriptor uploads before a new consumer stream;
    // allocator stream recording keeps an evicted table alive until every
    // already-enqueued CP kernel on this stream has retired.
    cached->ready_event->block(stream);
    c10::cuda::CUDACachingAllocator::recordStream(
        cached->seg_tbl_d.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(
        cached->seq_seg_d.storage().data_ptr(), stream);
    if (cached->seg_bt_d.defined()) {
        c10::cuda::CUDACachingAllocator::recordStream(
            cached->seg_bt_d.storage().data_ptr(), stream);
    }
    cached->active_streams.insert(key);
}

std::shared_ptr<CachedCpPlan> build_cached_cp_plan(
    const CpPlanKey& key, const torch::Tensor& device_ref,
    const c10::cuda::CUDAStream& stream, bool retain_for_cache) {
    const int N = key.N;
    std::vector<int64_t> seq_len(static_cast<size_t>(N));
    std::vector<int64_t> tok_begin(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        tok_begin[size_t(i)] = key.raw_cu[size_t(i)];
        seq_len[size_t(i)] =
            key.raw_cu[size_t(i + 1)] - key.raw_cu[size_t(i)];
    }

    auto cached = std::make_shared<CachedCpPlan>();
    cached->plan = plan_cp(
        seq_len, tok_begin, key.T_total, key.H, key.is_varlen,
        key.cu_count, key.config);
    if (!cached->plan.engage) return cached;

    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32);
    cached->seg_tbl_d = torch::from_blob(
        cached->plan.seg_tbl.data(),
        {int64_t(cached->plan.seg_tbl.size())}, opt_i32).to(device_ref.device());
    cached->seq_seg_d = torch::from_blob(
        cached->plan.seq_seg.data(),
        {int64_t(cached->plan.seq_seg.size())}, opt_i32).to(device_ref.device());
    int uploaded_tables = 2;
    if (cached->plan.bt64_align && !cached->plan.seg_bt.empty()) {
        cached->seg_bt_d = torch::from_blob(
            cached->plan.seg_bt.data(),
            {int64_t(cached->plan.seg_bt.size())},
            opt_i32).to(device_ref.device());
        ++uploaded_tables;
    }
    {
        auto& state = cp_plan_cache_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.h2d_tables += uploaded_tables;
    }

    if (retain_for_cache) {
        cached->ready_event = std::make_shared<at::cuda::CUDAEvent>();
        cached->ready_event->record(stream);
        c10::cuda::CUDACachingAllocator::recordStream(
            cached->seg_tbl_d.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(
            cached->seq_seg_d.storage().data_ptr(), stream);
        if (cached->seg_bt_d.defined()) {
            c10::cuda::CUDACachingAllocator::recordStream(
                cached->seg_bt_d.storage().data_ptr(), stream);
        }
        cached->active_streams.insert({
            int(stream.device_index()), int64_t(stream.id())});
    }
    return cached;
}

std::shared_ptr<CachedCpPlan> get_or_build_cp_plan(
    int T_total, int H, int N, bool is_varlen, const torch::Tensor& cu_t,
    const c10::optional<std::vector<int64_t>>& raw_cu_opt,
    const torch::Tensor& device_ref, const c10::cuda::CUDAStream& stream) {
    const CpConfig requested = cp_config_from_env();
    const int device = device_ref.get_device();
    const auto architecture = flashkda::host_runtime::current_architecture();
    if (!cp_dispatch_candidate(
            T_total, H, N, is_varlen, architecture, requested))
        return std::make_shared<CachedCpPlan>();

    const std::vector<int64_t> raw_cu = resolve_raw_cu(
        T_total, N, is_varlen, cu_t, raw_cu_opt);
    int64_t max_seq_len64 = 0;
    for (int i = 0; i < N; ++i) {
        max_seq_len64 = std::max(
            max_seq_len64, raw_cu[size_t(i + 1)] - raw_cu[size_t(i)]);
    }
    const int max_seq_len = int(std::min<int64_t>(
        max_seq_len64, std::numeric_limits<int>::max()));
    const CpConfig config = effective_cp_config(
        architecture, max_seq_len, H, N, is_varlen, requested);

    CpPlanKey key;
    key.device = device;
    key.architecture = int(architecture);
    key.cu_count = cp_cu_count();
    key.T_total = T_total;
    key.H = H;
    key.N = N;
    key.is_varlen = is_varlen;
    key.config = config;
    key.raw_cu = raw_cu;

    const int capacity = cp_plan_cache_capacity();
    auto& state = cp_plan_cache_state();
    if (capacity > 0) {
        std::shared_ptr<CachedCpPlan> hit;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            while (state.entries.size() > size_t(capacity)) {
                state.entries.pop_back();
                ++state.evictions;
            }
            for (size_t i = 0; i < state.entries.size(); ++i) {
                if (state.entries[i].key == key) {
                    ++state.hits;
                    CpPlanCacheEntry promoted = std::move(state.entries[i]);
                    state.entries.erase(state.entries.begin() + i);
                    state.entries.insert(
                        state.entries.begin(), std::move(promoted));
                    hit = state.entries.front().value;
                    break;
                }
            }
            if (!hit) ++state.misses;
        }
        if (hit) {
            activate_cached_cp_plan(hit, stream);
            return hit;
        }
    } else {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.entries.clear();
        ++state.misses;
    }

    auto built = build_cached_cp_plan(
        key, device_ref, stream, capacity > 0);
    if (capacity == 0) return built;

    std::shared_ptr<CachedCpPlan> selected = built;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        // A second thread may have published the same complete plan while the
        // H2D upload ran outside the lock. Prefer the published entry.
        for (size_t i = 0; i < state.entries.size(); ++i) {
            if (state.entries[i].key == key) {
                selected = state.entries[i].value;
                CpPlanCacheEntry promoted = std::move(state.entries[i]);
                state.entries.erase(state.entries.begin() + i);
                state.entries.insert(
                    state.entries.begin(), std::move(promoted));
                break;
            }
        }
        if (selected == built) {
            state.entries.insert(
                state.entries.begin(), CpPlanCacheEntry{key, built});
            while (state.entries.size() > size_t(capacity)) {
                state.entries.pop_back();
                ++state.evictions;
            }
        }
    }
    if (selected != built) activate_cached_cp_plan(selected, stream);
    return selected;
}

void clear_cp_plan_cache() {
    auto& state = cp_plan_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.entries.clear();
    state.hits = state.misses = state.evictions = 0;
    state.d2h_reads = state.h2d_tables = 0;
}

py::dict cp_plan_cache_stats() {
    auto& state = cp_plan_cache_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    py::dict d;
    d["capacity"] = cp_plan_cache_capacity();
    d["size"] = state.entries.size();
    d["hits"] = state.hits;
    d["misses"] = state.misses;
    d["evictions"] = state.evictions;
    d["d2h_reads"] = state.d2h_reads;
    d["h2d_tables"] = state.h2d_tables;
    return d;
}

// flashkda_prepare: Kernel 1. Inputs are the FLA-style [T_total,H,D] contiguous
// tensors (q,k,g bf16), beta [T_total,H] bf16, A_log [H] f32, dt_bias [H,D] f32.
// Returns the 6 workspace tensors for validation against the torch reference.
std::vector<torch::Tensor> flashkda_prepare(
    torch::Tensor q, torch::Tensor k, torch::Tensor g, torch::Tensor beta,
    torch::Tensor A_log, torch::Tensor dt_bias,
    double scale, double lower_bound,
    c10::optional<torch::Tensor> cu_seqlens_opt) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashkda_prepare",
            {{"q", &q}, {"k", &k}, {"g", &g}, {"beta", &beta}, {"A_log", &A_log},
             {"dt_bias", &dt_bias}, {"cu_seqlens", cu_seqlens_opt.has_value() ? &cu_seqlens_opt.value() : nullptr}});
        const int64_t tiles = (q.size(0) + 15) / 16 +
            (cu_seqlens_opt.has_value() ? cu_seqlens_opt->numel() - 1 : 0);
        const int64_t nht = q.size(1) * tiles;
        auto options = q.options().dtype(torch::kBFloat16);
        return {pla::bypass::output({nht, 16, 128}, options),
                pla::bypass::output({nht, 16, 128}, options),
                pla::bypass::output({nht, 16, 128}, options),
                pla::bypass::output({nht, 128}, options.dtype(torch::kFloat32)),
                pla::bypass::output({nht, 16, 16}, options),
                pla::bypass::output({nht, 16, 16}, options)};
    }
    const pla::sanity::Context sanity("flashkda_prepare", q);
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16, "q bf16 CUDA");
    TORCH_CHECK(q.dim() == 3, "q must be [T_total,H,D]");
    // CUDA launch state belongs to q's device even when the caller has another
    // device selected. The scoped guard also restores the caller's selection.
    const c10::cuda::CUDAGuard device_guard(q.device());
    if (sanity)
        sanity_prepare_inputs(sanity, q, k, g, beta, A_log, dt_bias,
                              scale, lower_bound, cu_seqlens_opt);
    int T_total = q.size(0), H = q.size(1), Dd = q.size(2);
    TORCH_CHECK(Dd == 128, "D must be 128");
    constexpr int CHUNK = 16;
    const hggcStream_t stream = pla::current_stream();
    // Resolve the debug workspace ABI before allocating or launching work.
    // A cold capture must fail before attempting the probe's eager-only D2H.
    const bool kr_preswizzled = flashkda_kr_is_preswizzled(stream);

    bool is_varlen = cu_seqlens_opt.has_value();
    int N; const int32_t* cu = nullptr;
    torch::Tensor cu_t;
    int total_tiles;
    if (is_varlen) {
        cu_t = cu_seqlens_opt.value();
        TORCH_CHECK(cu_t.dtype() == torch::kInt32 || cu_t.dtype() == torch::kLong,
                    "cu_seqlens must be int32 or int64");
        if (cu_t.dtype() == torch::kLong) cu_t = cu_t.to(torch::kInt32);
        N = cu_t.numel() - 1;
        cu = cu_t.data_ptr<int32_t>();
        total_tiles = (T_total + CHUNK - 1) / CHUNK + N;   // upper bound
    } else {
        N = 1;
        total_tiles = (T_total + CHUNK - 1) / CHUNK;
    }

    // Consume the public [T_total,H] layout directly.  The kernels read only
    // one 16-element beta vector per chunk, so materializing a full transposed
    // copy costs more than the small strided reads on PPU1.5.
    TORCH_CHECK(beta.is_cuda() && beta.device() == q.device()
                    && beta.dtype() == torch::kBFloat16 && beta.dim() == 2
                    && beta.size(0) == T_total && beta.size(1) == H
                    && beta.stride(1) == 1 && beta.stride(0) >= H
                    && beta.stride(0) <= std::numeric_limits<int>::max(),
                "beta must be bf16 CUDA [T_total,H] contiguous in H");

    auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    int64_t nht = int64_t(H) * total_tiles;
    // Per-slot byte images followed by upstream-compatible 128B-aligned prefix.
    auto ws_storage = torch::zeros(
        {nht * flashkda::WS_SLOT_UNITS + tile_prefix_units(N, is_varlen)},
        opt_bf16);
    auto ws = ws_storage.narrow(0, 0, nht * flashkda::WS_SLOT_UNITS)
                  .view({nht, flashkda::WS_SLOT_UNITS});

    float gate_scale = float(lower_bound * 1.4426950408889634);
    launch_flashkda_prepare(
        reinterpret_cast<const void*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(g.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
        float(scale), gate_scale, T_total, H, int(beta.stride(0)),
        N, total_tiles, cu, is_varlen,
        0, total_tiles, 0, H, nullptr,
        reinterpret_cast<void*>(ws.data_ptr<at::BFloat16>()),
        /*force_plain=*/false, /*cp_fma_front=*/false, /*fp16_kd_qd=*/false,
        flashkda::KrWorkspaceLayout::kNormalRaw,
        stream);

    // Reconstruct the six logical fields through the shared private ABI.
    auto ws_kd = ws.narrow(
        1, flashkda::WS_OFF_KD_U, flashkda::WS_KD_U)
        .reshape({nht, CHUNK, 128}).contiguous();
    auto ws_qd = ws.narrow(
        1, flashkda::WS_OFF_QD_U, flashkda::WS_KD_U)
        .reshape({nht, CHUNK, 128}).contiguous();
    auto kr_raw = ws.narrow(
        1, flashkda::WS_OFF_KR_U, flashkda::WS_KD_U).contiguous();
    auto ws_inv = ws.narrow(
        1, flashkda::WS_OFF_INV_U, flashkda::WS_INV_U)
        .reshape({nht, CHUNK, CHUNK}).contiguous();
    auto ws_mqk = ws.narrow(
        1, flashkda::WS_OFF_MQK_U, flashkda::WS_INV_U)
        .reshape({nht, CHUNK, CHUNK}).contiguous();
    auto ws_gt = ws.narrow(
        1, flashkda::WS_OFF_GT_U, flashkda::WS_GT_U)
        .contiguous().view(torch::kFloat32).reshape({nht, 128}).contiguous();

    torch::Tensor ws_kr;
    if (kr_preswizzled) {
        // Recover logical row-major from tc02's probe-verified
        // CUBE_H=16/CUBE_W=32 physical image.
        std::vector<int64_t> src_h(flashkda::WS_KD_U);
        for (int j = 0; j < flashkda::WS_KD_U; ++j) {
            int cube = j >> 9, local = j & 511, r = local >> 5, kk = local & 31;
            int shift = ((r >> 1) & 3) << 3;
            src_h[j] = int64_t(r * 128 + cube * 32 + (kk ^ shift));
        }
        auto src_idx = torch::from_blob(
            src_h.data(), {flashkda::WS_KD_U}, torch::kLong)
            .to(kr_raw.device());
        ws_kr = torch::empty_like(kr_raw);
        ws_kr.index_copy_(1, src_idx, kr_raw);
    } else {
        ws_kr = kr_raw;
    }
    ws_kr = ws_kr.reshape({nht, CHUNK, 128}).contiguous();

    return {ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk};
}

// flashkda_fwd: full forward = prepare (Kernel1) + recurrence (Kernel2).
// Inputs FLA-style [T_total,H,D] (q,k,v,g bf16), beta [T_total,H] bf16,
// A_log [H] f32, dt_bias [H,D] f32.
//
// Out-params, matching upstream flash_kda.fwd: the caller owns `out` and the
// optional `final_state`, so the state dtype (bf16 or fp32) is the caller's
// choice and the final state is only computed when a tensor is supplied. The
// recurrence always carries the state in bf16 internally (FlashKDA design); an
// fp32 state is rounded on load and widened on store, at the boundary only.
void flashkda_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, torch::Tensor A_log, torch::Tensor dt_bias,
    torch::Tensor out,
    double scale, double lower_bound,
    c10::optional<torch::Tensor> initial_state_opt,
    c10::optional<torch::Tensor> final_state_opt,
    c10::optional<torch::Tensor> cu_seqlens_opt,
    c10::optional<torch::Tensor> state_indices_opt,
    c10::optional<std::vector<int64_t>> raw_cu_seqlens_opt) {
    if (pla::bypass::enabled()) {
        const c10::cuda::OptionalCUDAGuard guard(q.is_cuda()
            ? c10::optional<c10::Device>(q.device()) : c10::nullopt);
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashkda_fwd",
            {{"q", &q}, {"k", &k}, {"v", &v}, {"g", &g}, {"beta", &beta},
             {"A_log", &A_log}, {"dt_bias", &dt_bias}, {"out", &out},
             {"initial_state", initial_state_opt.has_value() ? &initial_state_opt.value() : nullptr},
             {"final_state", final_state_opt.has_value() ? &final_state_opt.value() : nullptr},
             {"cu_seqlens", cu_seqlens_opt.has_value() ? &cu_seqlens_opt.value() : nullptr},
             {"state_indices", state_indices_opt.has_value() ? &state_indices_opt.value() : nullptr}},
            "scale=" + std::to_string(scale) + " lower_bound=" + std::to_string(lower_bound));
        if (pla::instrument::show_params()) pla::instrument::dump_case(cu_seqlens_opt.has_value() ? cu_seqlens_opt->numel() - 1 : 1,
            q.size(0), q.size(2), v.size(2), q.size(1), v.size(1),
            get_dtype_string(q), true,
            cu_seqlens_opt.has_value() ? &cu_seqlens_opt.value() : nullptr,
            static_cast<float>(lower_bound), true);
        return;
    }
    const pla::sanity::Context sanity("flashkda_fwd", q);
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.dim() == 3,
                "q bf16 CUDA [T,H,D]");
    const c10::cuda::CUDAGuard device_guard(q.device());
    if (sanity) {
        sanity_prepare_inputs(sanity, q, k, g, beta, A_log, dt_bias,
                              scale, lower_bound, cu_seqlens_opt, raw_cu_seqlens_opt);
        sanity_v(sanity, q, v);
        sanity.tensor(out, "out");
        sanity.check(!sp::overlaps(out, v) && !sp::overlaps(out, beta),
                     "out must not overlap recurrence inputs v/beta");
        const int64_t N = cu_seqlens_opt.has_value() ? cu_seqlens_opt->numel() - 1 : 1;
        for (const auto& named :
             std::initializer_list<std::pair<const char*, c10::optional<torch::Tensor>>>{
                 {"initial_state", initial_state_opt}, {"final_state", final_state_opt}}) {
            if (!named.second.has_value()) continue;
            const auto& state = named.second.value();
            sanity.tensor(state, named.first);
            sanity.check(state.dim() == 4, named.first, " must have rank four");
            sanity.check(!sp::overlaps(state, out), named.first, " must not overlap out");
            if (state_indices_opt.has_value())
                sp::slots(sanity, state_indices_opt.value(), N, state.size(0),
                          "state_indices", final_state_opt.has_value());
        }
        if (initial_state_opt.has_value() && final_state_opt.has_value())
            sanity.check(!sp::overlaps(initial_state_opt.value(), final_state_opt.value()) ||
                             sp::same_view(initial_state_opt.value(), final_state_opt.value()),
                         "initial/final state may alias only as the exact same tensor view");
        if (final_state_opt.has_value())
            sanity.check(!sp::overlaps(final_state_opt.value(), v) &&
                             !sp::overlaps(final_state_opt.value(), beta),
                         "final_state must not overlap recurrence inputs v/beta");
    }
    int T_total = q.size(0), H = q.size(1), Dd = q.size(2);
    TORCH_CHECK(Dd == 128, "D must be 128");
    constexpr int CHUNK = 16;

    bool is_varlen = cu_seqlens_opt.has_value();
    int N; const int32_t* cu = nullptr; torch::Tensor cu_t; int total_tiles;
    if (is_varlen) {
        cu_t = cu_seqlens_opt.value();
        TORCH_CHECK(cu_t.dtype() == torch::kInt32 || cu_t.dtype() == torch::kLong,
                    "cu_seqlens must be int32 or int64");
        if (cu_t.dtype() == torch::kLong) cu_t = cu_t.to(torch::kInt32);
        TORCH_CHECK(cu_t.is_cuda() && cu_t.device() == q.device() &&
                        cu_t.dtype() == torch::kInt32 && cu_t.dim() == 1 &&
                        cu_t.is_contiguous() && cu_t.numel() >= 2,
                    "cu_seqlens must be contiguous int32 CUDA [N+1] on q.device");
        N = cu_t.numel() - 1; cu = cu_t.data_ptr<int32_t>();
        total_tiles = (T_total + CHUNK - 1) / CHUNK + N;
    } else {
        N = 1; total_tiles = (T_total + CHUNK - 1) / CHUNK;
    }

    if (g_ppu_lib_show_params) {
        const torch::Tensor* dump_cu_seqlens = is_varlen ? &cu_t : nullptr;
        std::cout << generate_nvtx_tag(
            N, T_total, Dd, Dd, H, H, get_dtype_string(q), true,
            dump_cu_seqlens, static_cast<float>(lower_bound), true) << std::endl;
    }

    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16,
                "out must be bf16 CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == T_total && out.size(1) == H &&
                    out.size(2) == Dd,
                "out must be [T_total,H,D] matching q");

    // initial_state / final_state are caller-owned and may each be bf16 or fp32;
    // when both are given their dtypes must match (upstream contract). Neither
    // is made contiguous here on purpose: final_state is written through, so
    // silently substituting a compacted copy would discard the result.
    bool has_state_in = initial_state_opt.has_value();
    bool has_state_out = final_state_opt.has_value();
    bool indexed_state = state_indices_opt.has_value();
    bool state_is_fp32 = false;
    auto check_state = [&](torch::Tensor const& s, const char* name) {
        TORCH_CHECK(s.is_cuda() && s.device() == q.device(),
                    name, " must be CUDA on q.device");
        TORCH_CHECK(s.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(s.dtype() == torch::kBFloat16 || s.dtype() == torch::kFloat32,
                    name, " must be bf16 or fp32");
        TORCH_CHECK(s.dim() == 4
                        && (indexed_state ? s.size(0) >= 1 : s.size(0) == N) &&
                        s.size(1) == H &&
                        s.size(2) == 128 && s.size(3) == 128,
                    name, indexed_state ? " must be [slots,H,128,128]"
                                        : " must be [N,H,128,128]");
        if (s.dtype() == torch::kFloat32) state_is_fp32 = true;
    };
    if (has_state_in) check_state(initial_state_opt.value(), "initial_state");
    if (has_state_out) check_state(final_state_opt.value(), "final_state");
    if (has_state_in && has_state_out) {
        TORCH_CHECK(initial_state_opt->dtype() == final_state_opt->dtype(),
                    "initial_state and final_state must have the same dtype");
    }
    const int32_t* state_indices = nullptr;
    torch::Tensor state_indices_t;
    if (indexed_state) {
        TORCH_CHECK(has_state_in || has_state_out,
                    "state_indices requires initial_state or final_state");
        state_indices_t = state_indices_opt.value();
        TORCH_CHECK(state_indices_t.is_cuda() && state_indices_t.device() == q.device()
                        && state_indices_t.dtype() == torch::kInt32
                        && state_indices_t.dim() == 1
                        && state_indices_t.numel() == N
                        && state_indices_t.is_contiguous(),
                    "state_indices must be contiguous int32 CUDA [N] on q.device");
        // Serving owns the cache allocator and guarantees that indices are in
        // range and unique within the batch. Keep this hot path device-only:
        // a D2H validation would serialize every prefill launch.
        state_indices = state_indices_t.data_ptr<int32_t>();
    }

    TORCH_CHECK(beta.is_cuda() && beta.device() == q.device()
                    && beta.dtype() == torch::kBFloat16 && beta.dim() == 2
                    && beta.size(0) == T_total && beta.size(1) == H
                    && beta.stride(1) == 1 && beta.stride(0) >= H
                    && beta.stride(0) <= std::numeric_limits<int>::max(),
                "beta must be bf16 CUDA [T_total,H] contiguous in H");
    auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    int64_t nht = int64_t(H) * total_tiles;
    // Per-slot byte images followed by upstream-compatible 128B-aligned prefix.
    auto ws_storage = torch::empty(
        {nht * flashkda::WS_SLOT_UNITS + tile_prefix_units(N, is_varlen)},
        opt_bf16);
    auto ws = ws_storage.narrow(0, 0, nht * flashkda::WS_SLOT_UNITS)
                  .view({nht, flashkda::WS_SLOT_UNITS});

    // K2 carries recurrent values in bf16 internally. The caller's bf16/fp32
    // state buffer is consumed and produced directly by the recurrence kernels;
    // fp32 conversion happens only for the elements owned by each CTA. Indexed
    // serving pools therefore avoid gather/scatter and full-pool dtype staging.
    const void* init_ptr = nullptr;
    if (has_state_in) {
        init_ptr = state_is_fp32
            ? reinterpret_cast<const void*>(initial_state_opt->data_ptr<float>())
            : reinterpret_cast<const void*>(initial_state_opt->data_ptr<at::BFloat16>());
    }
    void* final_ptr = nullptr;
    if (has_state_out) {
        final_ptr = state_is_fp32
            ? reinterpret_cast<void*>(final_state_opt->data_ptr<float>())
            : reinterpret_cast<void*>(final_state_opt->data_ptr<at::BFloat16>());
    }

    float gate_scale = float(lower_bound * 1.4426950408889634);
    const auto torch_stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    const hggcStream_t stream = pla::current_stream(q.get_device());

    // Serving already owns a trusted Host copy of cu_seqlens. It lets the
    // common exact-length path skip the padding kernel without a D2H read.
    // Standalone callers without that metadata use a lightweight device check;
    // either way, padding logic never inflates the prepare-kernel ABI/resources.
    if (is_varlen) {
        int valid_tokens = -1;
        bool launch_padding = true;
        if (raw_cu_seqlens_opt.has_value()) {
            const auto& raw_cu = raw_cu_seqlens_opt.value();
            TORCH_CHECK(
                raw_cu.size() == size_t(N + 1),
                "raw_cu_seqlens must have N+1 entries, got ", raw_cu.size(),
                " for N=", N);
            const int64_t endpoint = raw_cu.back();
            launch_padding = endpoint >= 0 && endpoint < T_total;
            if (launch_padding) valid_tokens = int(endpoint);
        }
        if (launch_padding) {
            launch_flashkda_zero_padded_output_tail(
                reinterpret_cast<void*>(out.data_ptr<at::BFloat16>()),
                T_total, H, N, cu, valid_tokens, stream);
        }
    }

    // ---- CP plan -----------------------------------------------------------
    // Trusted Host offsets let serving reuse a value-keyed plan across every
    // layer without synchronising cu_seqlens back to the CPU.  The first miss
    // builds and uploads both descriptor tables before K1; subsequent hits
    // only establish stream ordering/lifetime on a newly seen stream.
    auto cp_cached = get_or_build_cp_plan(
        T_total, H, N, is_varlen, cu_t, raw_cu_seqlens_opt,
        q, torch_stream);
    const CpPlan& cp = cp_cached->plan;
    const torch::Tensor& seg_tbl_d = cp_cached->seg_tbl_d;
    const torch::Tensor& seq_seg_d = cp_cached->seq_seg_d;
    const torch::Tensor& seg_bt_d = cp_cached->seg_bt_d;
    const flashkda::KrWorkspaceLayout kr_layout =
        select_kr_workspace_layout(
        T_total, H, N, total_tiles, is_varlen, cp);
    const bool fp16_state = cp.engage
        && !flashkda::host_runtime::current_device_is_tc01()
        && cp_env_int("PLA_FLASHKDA_CP_STATE_FP16", 0) != 0;
    auto launch_prepare_range = [&](int tile_offset, int tile_count,
                                    int head_offset, int head_count,
                                    int* ready, hggcStream_t target_stream) {
        launch_flashkda_prepare(
            reinterpret_cast<const void*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(g.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
            A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
            float(scale), gate_scale, T_total, H, int(beta.stride(0)),
            N, total_tiles, cu, is_varlen,
            tile_offset, tile_count, head_offset, head_count, ready,
            reinterpret_cast<void*>(ws.data_ptr<at::BFloat16>()),
            /*force_plain=*/cp.engage, /*cp_fma_front=*/cp.fma_front,
            /*fp16_kd_qd=*/fp16_state, kr_layout, target_stream);
    };

    // A rejected launch (over-budget smem, bad grid) would otherwise be silent
    // and leave `out` holding whatever the caching allocator last put there,
    // which can look exactly like a correct result.  Fail loudly instead.  The
    // BT64 stages throw from their own launchers; the reused CH16 pre_scan and
    // merge passes rely on this check.
    auto cp_check = [&](const char* pass) {
        hggcError_t e = hggcPeekAtLastError();
        TORCH_CHECK(e == hggcSuccess, "flashkdapro CP ", pass,
                    " launch failed: ", hggcGetErrorString(e),
                    " (n_seg=", cp.n_seg, " H=", H, " bv=", cp.bv, ")");
    };

    bool bt64_ready = use_tc01_bt64_candidate(
        T_total, H, N, is_varlen, cp, kr_layout);
    const int64_t total_bt64 =
        (int64_t(T_total) + flashkda::BT64_CHUNK - 1)
            / flashkda::BT64_CHUNK + N;
    torch::Tensor bt_ws, bt_x, bt_h;
    if (bt64_ready) {
        // Only catch allocation failures; kernel errors must propagate explicitly and must not fall back with a partial state.
        try {
            bt_ws = torch::empty(
                {total_bt64, H, flashkda::BT64_WS_SLOT_UNITS}, opt_bf16);
            bt_x = torch::empty({T_total, H, 128}, opt_bf16);
            bt_h = torch::empty({total_bt64, H, 128, 128}, opt_bf16);
        } catch (const c10::OutOfMemoryError&) {
            bt_ws = torch::Tensor();
            bt_x = torch::Tensor();
            bt_h = torch::Tensor();
            bt64_ready = false;
        }
    }

    launch_prepare_range(0, total_tiles, 0, H, nullptr, stream);

    // ---- BT64xCP composed pipeline ---------------------------------------
    // cp.bt64_align only engages inside the tc01 H8/N1 AUTO window under the
    // PLA_FLASHKDA_TC01_BT64_CP opt-in, so this branch is the GLM batch=1
    // long-sequence shape by construction.  The CH16 pre_scan/merge passes
    // are reused unchanged; only the serial stage switches from the CH16
    // segment recurrence to the BT64 segment recurrence, and the BT64
    // parallel front/back stages keep their original order around it.
    const bool bt64_cp = bt64_ready && cp.engage && cp.bt64_align
        && !cp.seg_bt.empty() && seg_bt_d.defined();
    if (bt64_cp) {
        // chain_state/carries follow the plain CP branch: an allocation
        // failure propagates instead of falling back with a partial state.
        auto chain_state = torch::empty({2, cp.n_seg, H, 128, 128}, opt_bf16);
        auto carries = torch::empty({cp.n_seg, H, 128, 128}, opt_bf16);
        if (cp_env_int("PLA_FLASHKDA_CP_VERBOSE", 0)) {
            fprintf(stderr,
                    "[flashkdapro] BT64xCP engaged: T=%d H=%d N=%d n_seg=%d bv=%d\n",
                    T_total, H, N, cp.n_seg, cp.bv);
        }

        launch_flashkda_bt64_cp_front_tc01(
            reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_h.data_ptr<at::BFloat16>()),
            T_total, H, N, total_tiles, int(total_bt64), cu, kr_layout,
            stream);

        launch_flashkda_cp_prescan(
            reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(chain_state.data_ptr<at::BFloat16>()),
            /*fp16_state=*/false,
            T_total, H, cp.n_seg, total_tiles,
            seg_tbl_d.data_ptr<int32_t>(), cp.bv, stream);
        cp_check("pre_scan");

        launch_flashkda_cp_merge(
            reinterpret_cast<const void*>(chain_state.data_ptr<at::BFloat16>()),
            init_ptr,
            reinterpret_cast<void*>(carries.data_ptr<at::BFloat16>()),
            /*fp16_state=*/false,
            state_indices, state_is_fp32,
            seq_seg_d.data_ptr<int32_t>(), H, N, cp.n_seg, stream);
        cp_check("merge");

        launch_flashkda_bt64_cp_segment_tc01(
            reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(bt_ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_x.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_h.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(carries.data_ptr<at::BFloat16>()),
            final_ptr, state_indices, state_is_fp32,
            T_total, H, total_tiles, int(total_bt64),
            seg_bt_d.data_ptr<int32_t>(), cp.n_seg, has_state_out, stream);

        launch_flashkda_bt64_cp_back_tc01(
            reinterpret_cast<void*>(out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(bt_x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(bt_h.data_ptr<at::BFloat16>()),
            T_total, H, N, total_tiles, int(total_bt64), cu, stream);
        return;
    }

    if (bt64_ready) {
        launch_flashkda_bt64_prefill_tc01(
            reinterpret_cast<void*>(out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_ws.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_x.data_ptr<at::BFloat16>()),
            reinterpret_cast<void*>(bt_h.data_ptr<at::BFloat16>()),
            init_ptr, final_ptr, state_indices, state_indices,
            state_is_fp32, state_is_fp32,
            T_total, H, N, total_tiles, int(total_bt64), cu,
            has_state_in, has_state_out, kr_layout, stream);
        return;
    }

    if (!cp.engage) {
        launch_flashkda_recurrence(
            reinterpret_cast<void*>(out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
            reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
            init_ptr, final_ptr,
            state_indices, state_indices, state_is_fp32, state_is_fp32,
            T_total, H, N, total_tiles, cu, is_varlen, has_state_in,
            has_state_out, kr_layout, stream);
        return;
    }

    // ---- CP: pre_scan -> merge -> segment ----------------------------------
    // chain_state holds both affine parts of every non-final segment; the merge
    // composes them into one carry per segment; the segment pass is then an
    // ordinary with-state recurrence over n_seg independent chains.  chain_state
    // is torch::empty on purpose: pre_scan skips each sequence's last segment
    // (it has no consumer) and the merge never reads those slots.
    auto opt_cp_state = fp16_state
        ? opt_bf16.dtype(torch::kFloat16) : opt_bf16;
    auto chain_state = torch::empty({2, cp.n_seg, H, 128, 128}, opt_cp_state);
    auto carries = torch::empty({cp.n_seg, H, 128, 128}, opt_cp_state);

    if (cp_env_int("PLA_FLASHKDA_CP_VERBOSE", 0)) {
        fprintf(stderr, "[flashkdapro] CP engaged: T=%d H=%d N=%d n_seg=%d bv=%d\n",
                T_total, H, N, cp.n_seg, cp.bv);
    }

    launch_flashkda_cp_prescan(
        reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
        chain_state.data_ptr(), fp16_state,
        T_total, H, cp.n_seg, total_tiles,
        seg_tbl_d.data_ptr<int32_t>(), cp.bv, stream);
    cp_check("pre_scan");

    launch_flashkda_cp_merge(
        chain_state.data_ptr(), init_ptr, carries.data_ptr(), fp16_state,
        state_indices, state_is_fp32,
        seq_seg_d.data_ptr<int32_t>(), H, N, cp.n_seg, stream);
    cp_check("merge");

    launch_flashkda_cp_segment(
        reinterpret_cast<void*>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
        carries.data_ptr(), fp16_state, final_ptr,
        state_indices, state_is_fp32,
        T_total, H, cp.n_seg, total_tiles,
        seg_tbl_d.data_ptr<int32_t>(), cp.bv, stream);
    cp_check("segment");

}

// flashkda_cp_stages: debug view of the CP pipeline's first two passes.  Runs
// K1 + pre_scan + merge and stops, returning {chain_state, carries}.  The carry
// of segment i must equal the serial final_state of the prefix ending at that
// segment's first token, which is what makes the merge independently testable
// instead of only observable through the final output.
std::vector<torch::Tensor> flashkda_cp_stages(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, torch::Tensor A_log, torch::Tensor dt_bias,
    double scale, double lower_bound,
    c10::optional<torch::Tensor> initial_state_opt,
    c10::optional<torch::Tensor> cu_seqlens_opt,
    c10::optional<std::vector<int64_t>> raw_cu_seqlens_opt) {
    if (pla::bypass::enabled()) {
        if (pla::instrument::show_params()) pla::instrument::dump_metadata("flashkda_cp_stages",
            {{"q", &q}, {"k", &k}, {"v", &v}, {"g", &g}, {"beta", &beta},
             {"A_log", &A_log}, {"dt_bias", &dt_bias},
             {"initial_state", initial_state_opt.has_value() ? &initial_state_opt.value() : nullptr},
             {"cu_seqlens", cu_seqlens_opt.has_value() ? &cu_seqlens_opt.value() : nullptr}});
        const c10::cuda::OptionalCUDAGuard guard(q.is_cuda()
            ? c10::optional<c10::Device>(q.device()) : c10::nullopt);
        auto options = q.options().dtype(torch::kBFloat16);
        return {pla::bypass::output({2, 0, q.size(1), 128, 128}, options),
                pla::bypass::output({0, q.size(1), 128, 128}, options)};
    }
    const pla::sanity::Context sanity("flashkda_cp_stages", q);
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.dim() == 3,
                "q bf16 CUDA [T,H,D]");
    const c10::cuda::CUDAGuard device_guard(q.device());
    if (sanity) {
        sanity_prepare_inputs(sanity, q, k, g, beta, A_log, dt_bias,
                              scale, lower_bound, cu_seqlens_opt, raw_cu_seqlens_opt);
        sanity_v(sanity, q, v);
        if (initial_state_opt.has_value()) {
            const auto& state = initial_state_opt.value();
            sanity.tensor(state, "initial_state");
            const int64_t N = cu_seqlens_opt.has_value() ? cu_seqlens_opt->numel() - 1 : 1;
            sanity.check(state.dim() == 4 && state.size(0) == N && state.size(1) == q.size(1) &&
                             state.size(2) == 128 && state.size(3) == 128 &&
                             (state.scalar_type() == torch::kFloat32 ||
                              state.scalar_type() == torch::kBFloat16),
                         "initial_state must be bf16/fp32 [N,H,128,128]");
        }
    }
    const int T_total = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 128, "D must be 128");

    const bool is_varlen = cu_seqlens_opt.has_value();
    int N = 1; const int32_t* cu = nullptr; torch::Tensor cu_t; int total_tiles;
    if (is_varlen) {
        cu_t = cu_seqlens_opt.value();
        TORCH_CHECK(cu_t.dtype() == torch::kInt32 || cu_t.dtype() == torch::kLong,
                    "cu_seqlens must be int32 or int64");
        if (cu_t.dtype() == torch::kLong) cu_t = cu_t.to(torch::kInt32);
        N = cu_t.numel() - 1; cu = cu_t.data_ptr<int32_t>();
        total_tiles = (T_total + KDA_CHUNK - 1) / KDA_CHUNK + N;
    } else {
        total_tiles = (T_total + KDA_CHUNK - 1) / KDA_CHUNK;
    }

    torch::Tensor init_state;
    const void* init_ptr = nullptr;
    if (initial_state_opt.has_value()) {
        init_state = initial_state_opt.value().to(torch::kBFloat16).contiguous();
        init_ptr = reinterpret_cast<const void*>(init_state.data_ptr<at::BFloat16>());
    }

    const auto torch_stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    auto cp_cached = get_or_build_cp_plan(
        T_total, H, N, is_varlen, cu_t, raw_cu_seqlens_opt,
        q, torch_stream);
    const CpPlan& cp = cp_cached->plan;
    TORCH_CHECK(cp.engage,
                "flashkda_cp_stages: CP did not engage for T=", T_total,
                " H=", H, " N=", N, "; set PLA_FLASHKDA_CP=2 and a segment "
                "floor small enough to split this shape");

    const torch::Tensor& seg_tbl_d = cp_cached->seg_tbl_d;
    const torch::Tensor& seq_seg_d = cp_cached->seq_seg_d;

    TORCH_CHECK(beta.is_cuda() && beta.device() == q.device()
                    && beta.dtype() == torch::kBFloat16 && beta.dim() == 2
                    && beta.size(0) == T_total && beta.size(1) == H
                    && beta.stride(1) == 1 && beta.stride(0) >= H
                    && beta.stride(0) <= std::numeric_limits<int>::max(),
                "beta must be bf16 CUDA [T_total,H] contiguous in H");
    auto opt_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    const int64_t nht = int64_t(H) * total_tiles;
    auto ws_storage = torch::empty(
        {nht * flashkda::WS_SLOT_UNITS + tile_prefix_units(N, is_varlen)},
        opt_bf16);
    auto ws = ws_storage.narrow(0, 0, nht * flashkda::WS_SLOT_UNITS)
                  .view({nht, flashkda::WS_SLOT_UNITS});
    // Zeroed so an unwritten slot reads as zero instead of stale allocator bytes.
    const bool fp16_state =
        !flashkda::host_runtime::current_device_is_tc01()
        && cp_env_int("PLA_FLASHKDA_CP_STATE_FP16", 0) != 0;
    auto opt_cp_state = fp16_state
        ? opt_bf16.dtype(torch::kFloat16) : opt_bf16;
    auto chain_state = torch::zeros({2, cp.n_seg, H, 128, 128}, opt_cp_state);
    auto carries = torch::zeros({cp.n_seg, H, 128, 128}, opt_cp_state);

    const float gate_scale = float(lower_bound * 1.4426950408889634);
    const hggcStream_t stream = pla::current_stream(q.get_device());

    launch_flashkda_prepare(
        reinterpret_cast<const void*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(g.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
        float(scale), gate_scale, T_total, H, int(beta.stride(0)),
        N, total_tiles, cu, is_varlen,
        0, total_tiles, 0, H, nullptr,
        reinterpret_cast<void*>(ws.data_ptr<at::BFloat16>()),
        /*force_plain=*/true, /*cp_fma_front=*/cp.fma_front,
        /*fp16_kd_qd=*/fp16_state,
        flashkda::KrWorkspaceLayout::kNormalRaw, stream);

    launch_flashkda_cp_prescan(
        reinterpret_cast<const void*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<const void*>(ws.data_ptr<at::BFloat16>()),
        chain_state.data_ptr(), fp16_state,
        T_total, H, cp.n_seg, total_tiles,
        seg_tbl_d.data_ptr<int32_t>(), cp.bv, stream);

    launch_flashkda_cp_merge(
        chain_state.data_ptr(), init_ptr, carries.data_ptr(), fp16_state,
        /*init_state_indices=*/nullptr, /*init_state_is_fp32=*/false,
        seq_seg_d.data_ptr<int32_t>(), H, N, cp.n_seg, stream);

    hggcError_t e = hggcPeekAtLastError();
    TORCH_CHECK(e == hggcSuccess, "flashkda_cp_stages launch failed: ",
                hggcGetErrorString(e));
    return {chain_state, carries};
}

// flashkda_cp_plan: host-only probe of the CP dispatch decision.  Same inputs
// and same env overrides as flashkda_fwd, no kernels launched — the accuracy
// tests use it to assert CP really engaged, and the crossover sweep uses it to
// report the segmentation it measured.
py::dict flashkda_cp_plan(int64_t T_total, int64_t H,
                          c10::optional<torch::Tensor> cu_seqlens_opt,
                          c10::optional<std::vector<int64_t>>
                              raw_cu_seqlens_opt) {
    if (pla::sanity::enabled()) {
        TORCH_CHECK(T_total > 0 && T_total <= std::numeric_limits<int>::max() &&
                        H > 0 && H <= std::numeric_limits<int>::max(),
                    "PLA_SANITY_CHECK [flashkda_cp_plan]: T_total/H must be positive int32 values");
        if (cu_seqlens_opt.has_value()) {
            const auto& cu = cu_seqlens_opt.value();
            TORCH_CHECK(cu.dim() == 1 && cu.is_contiguous() && cu.numel() >= 2 &&
                            cu.numel() < std::numeric_limits<int>::max() &&
                            (cu.is_cuda() || cu.device().is_cpu()) &&
                            (cu.scalar_type() == torch::kInt32 || cu.scalar_type() == torch::kInt64),
                        "PLA_SANITY_CHECK [flashkda_cp_plan]: cu_seqlens must be contiguous int32/int64 [N+1]");
            if (cu.is_cuda()) {
                const pla::sanity::Context sanity("flashkda_cp_plan", cu);
                const auto values = sp::offsets(sanity, cu, T_total, "cu_seqlens", true);
                if (raw_cu_seqlens_opt.has_value() && sanity.values_enabled())
                    sanity.check(values == raw_cu_seqlens_opt.value(),
                                 "raw_cu_seqlens must equal device cu_seqlens");
            } else if (raw_cu_seqlens_opt.has_value()) {
                const auto host = cu.to(torch::kInt64);
                const auto* data = host.data_ptr<int64_t>();
                const std::vector<int64_t> values(data, data + host.numel());
                TORCH_CHECK(values == raw_cu_seqlens_opt.value(),
                            "PLA_SANITY_CHECK [flashkda_cp_plan]: raw_cu_seqlens must equal CPU cu_seqlens");
            }
        }
    }
    const bool is_varlen = cu_seqlens_opt.has_value();
    c10::cuda::OptionalCUDAGuard device_guard;
    torch::Tensor cu_t;
    int N = 1;
    if (is_varlen) {
        cu_t = cu_seqlens_opt.value();
        if (cu_t.is_cuda()) device_guard.set_device(cu_t.device());
        N = int(cu_t.numel()) - 1;
    }
    const auto raw_cu = resolve_raw_cu(
        int(T_total), N, is_varlen, cu_t, raw_cu_seqlens_opt);
    if (pla::sanity::enabled()) {
        TORCH_CHECK(raw_cu.size() == size_t(N + 1) && raw_cu.front() == 0,
                    "PLA_SANITY_CHECK [flashkda_cp_plan]: invalid raw_cu_seqlens shape/start");
        for (size_t i = 1; i < raw_cu.size(); ++i)
            TORCH_CHECK(raw_cu[i] >= raw_cu[i - 1] && raw_cu[i] <= T_total,
                        "PLA_SANITY_CHECK [flashkda_cp_plan]: offsets must be monotonic within T_total");
        TORCH_CHECK(N <= std::numeric_limits<int>::max() / H,
                    "PLA_SANITY_CHECK [flashkda_cp_plan]: N*H exceeds kernel int32 ABI");
    }
    const CpConfig config = cp_config_from_env();
    CpPlan cp = plan_cp_dispatch(
        int(T_total), int(H), N, is_varlen, config,
        [&](std::vector<int64_t>& seq_len, std::vector<int64_t>& tok_begin) {
            for (int i = 0; i < N; ++i) {
                tok_begin[i] = raw_cu[size_t(i)];
                seq_len[i] = raw_cu[size_t(i + 1)] - raw_cu[size_t(i)];
            }
        });
    py::dict d;
    d["engage"] = cp.engage;
    d["fma_front"] = cp.fma_front;
    d["n_seg"] = cp.n_seg;
    d["bv"] = cp.bv;
    d["cu_count"] = cp_cu_count();
    // Longest segment in CH tiles: the CP critical path, and the quantity the
    // AUTO gate is really trying to minimise.
    int max_seg_tiles = 0;
    for (int s = 0; s < cp.n_seg; ++s) {
        const int len =
            cp.seg_tbl[size_t(s) * flashkda::KDA_SEG_STRIDE +
                       flashkda::KDA_SEG_LEN];
        max_seg_tiles = std::max(max_seg_tiles,
                                 (len + KDA_CHUNK - 1) / KDA_CHUNK);
    }
    d["max_seg_tiles"] = max_seg_tiles;
    // The exact tables the kernels are handed, so a test can verify the
    // segmentation rather than infer it.  bt64_align/seg_bt expose the
    // BT64-quantized schedule of the composed BT64xCP pipeline.
    d["seg_tbl"] = cp.seg_tbl;
    d["seq_seg"] = cp.seq_seg;
    d["bt64_align"] = cp.bt64_align;
    d["seg_bt"] = cp.seg_bt;
    return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    (void)pla::bypass::enabled();
    (void)pla::instrument::show_params();


    init_show_params();

    m.def("flashkda_prepare", &flashkda_prepare,
          "FlashKDA Kernel1 prepare (returns 6 workspace tensors)",
          py::arg("q"), py::arg("k"), py::arg("g"), py::arg("beta"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("scale"),
          py::arg("lower_bound"),
          py::arg("cu_seqlens") = c10::optional<torch::Tensor>());
    m.def("flashkda_fwd", &flashkda_fwd,
          "FlashKDA full forward (prepare + recurrence); writes out and, "
          "when supplied, final_state",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("out"), py::arg("scale"),
          py::arg("lower_bound"),
          py::arg("initial_state") = c10::optional<torch::Tensor>(),
          py::arg("final_state") = c10::optional<torch::Tensor>(),
          py::arg("cu_seqlens") = c10::optional<torch::Tensor>(),
          py::arg("state_indices") = c10::optional<torch::Tensor>(),
          py::arg("raw_cu_seqlens") =
              c10::optional<std::vector<int64_t>>());
    m.def("flashkda_cp_plan", &flashkda_cp_plan,
          "Probe the CP dispatch decision for a shape (no kernel launch)",
          py::arg("T_total"), py::arg("H"),
          py::arg("cu_seqlens") = c10::optional<torch::Tensor>(),
          py::arg("raw_cu_seqlens") =
              c10::optional<std::vector<int64_t>>());
    m.def("flashkda_cp_stages", &flashkda_cp_stages,
          "CP debug: K1 + pre_scan + merge only -> (chain_state, carries)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
          py::arg("A_log"), py::arg("dt_bias"), py::arg("scale"),
          py::arg("lower_bound"),
          py::arg("initial_state") = c10::optional<torch::Tensor>(),
          py::arg("cu_seqlens") = c10::optional<torch::Tensor>(),
          py::arg("raw_cu_seqlens") =
              c10::optional<std::vector<int64_t>>());
    m.def("flashkda_cp_plan_cache_clear", &clear_cp_plan_cache,
          "Clear the bounded FlashKDA CP plan cache and reset its counters");
    m.def("flashkda_cp_plan_cache_stats", &cp_plan_cache_stats,
          "Return FlashKDA CP plan cache hit/transfer counters");
}
