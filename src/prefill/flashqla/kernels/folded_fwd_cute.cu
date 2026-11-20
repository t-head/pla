// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Compiled by the normal FlashQLA extension; no separate library or ABI.
#define FLASHQLA_FOLDED_DEFINITIONS_ONLY 1
#define FLASHQLA_FWD_NAMESPACE flashqla_folded
#include "fused_fwd_cute.cu"
#undef FLASHQLA_FOLDED_DEFINITIONS_ONLY
#undef FLASHQLA_FWD_NAMESPACE

#define FLASHQLA_CORRECT_KERNEL_ONLY
#define FLASHQLA_CORRECT_NAMESPACE flashqla_folded_correction
#include "correct_h0_cute.cu"
#undef FLASHQLA_CORRECT_NAMESPACE
#undef FLASHQLA_CORRECT_KERNEL_ONLY

#include <mutex>
#include <stdexcept>
#include <type_traits>

namespace flashqla_folded {
constexpr int FoldedAElements = FF_CS * (FF_DK + FF_CS);
constexpr int FoldedQPad = 2048;

// Private folded forward: dead intermediates are absent and V holds one half.
struct FoldedForwardSmemLayout {
  static constexpr int K = 0;
  static constexpr int V = K + FF_NBUF * FF_KELEMS * 2;
  static constexpr int Q = V + FF_NBUF * FF_CS * 64 * 2 + FoldedQPad;
  static constexpr int A = Q + FF_NBUF * FF_KELEMS * 2;
  static constexpr int G = A + FF_NBUF * FoldedAElements * 2;
  static constexpr int TOTAL = G + FF_NBUF * FF_CS * 4;
  static constexpr int KT=0,AGSWZ=0,HSTMT=0,WSWZ=0,VDSWZ=0,GVDSWZ=0;
  static constexpr int PG=0,B=0,GEXP=0,GREV=0,GLOG=0;
};
static_assert(FoldedForwardSmemLayout::TOTAL ==
    FF_NBUF * (2 * FF_KELEMS + FF_CS * 64 + FoldedAElements) * 2 +
    FF_NBUF * FF_CS * 4 + FoldedQPad);

// State preparation consumes K, V, R and decay only. In particular it never
// publishes Q or the output coefficient M, and needs no shared accumulators.
template <int Groups>
struct FoldedStateSmemLayout {
  static constexpr int K = 0;
  static constexpr int V = K + FF_NBUF * FF_KELEMS * 2;
  static constexpr int A = V + FF_NBUF * FF_CS * 64 * Groups * 2;
  static constexpr int G = A + FF_NBUF * FF_CS * FF_DK * 2;
  static constexpr int TOTAL = G + FF_NBUF * FF_CS * 4;
  static constexpr int Q = 0, KT = 0, AGSWZ = 0, HSTMT = 0, WSWZ = 0;
  static constexpr int VDSWZ = 0, GVDSWZ = 0, PG = 0, B = 0;
  static constexpr int GEXP = 0, GREV = 0, GLOG = 0;
};
static_assert(FoldedStateSmemLayout<1>::TOTAL == 41216);
static_assert(FoldedStateSmemLayout<2>::TOTAL == 49408);

// The folded path can use a coalesced scan and BF16 gate rounding. The
// generic path keeps its original scan in chunk_cumsum_cute.cu.
__device__ __forceinline__ float folded_gate_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return __shfl_sync(0xffffffff, value, 0);
}

template <int H, int Hg, int FixedQKStride, int FixedGateStride, bool PackedQK>
__global__ void folded_gated_scan(
    const __ppu_bfloat16* __restrict__ a,
    const __ppu_bfloat16* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const int* __restrict__ cu_seqlens, const int* __restrict__ chunk_indices,
    float* __restrict__ g_cum, float* __restrict__ beta,
    const __ppu_bfloat16* __restrict__ q,
    const __ppu_bfloat16* __restrict__ k,
    __ppu_bfloat16* __restrict__ q_norm, __ppu_bfloat16* __restrict__ k_norm,
    int num_chunks, int T, int64_t stride_a_runtime, int64_t stride_b_runtime,
    int64_t stride_q_runtime, int64_t stride_k_runtime, float eps) {
  using Element = __ppu_bfloat16;
  constexpr int CS = 32;
  constexpr int D = 128;
  constexpr int QK_PER_BLOCK = 16;
  const int64_t stride_a = FixedGateStride ? FixedGateStride : stride_a_runtime;
  const int64_t stride_b = FixedGateStride ? FixedGateStride : stride_b_runtime;
  const int64_t stride_q = FixedQKStride ? FixedQKStride : stride_q_runtime;
  const int64_t stride_k = FixedQKStride ? FixedQKStride : stride_k_runtime;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // Q/K normalization keeps the same one-warp-per-row ownership and exact
  // bf16/fp16 rounding point as the materialized-gate path.
  int qk_idx = blockIdx.x * QK_PER_BLOCK + warp * 2;
  for (int item = 0; item < 2; ++item, ++qk_idx) {
    if (qk_idx < T * Hg) {
      const int token = qk_idx / Hg;
      const int head = qk_idx - token * Hg;
      const int64_t q_base = (int64_t)token * stride_q + head * D;
      const int64_t k_base = (int64_t)token * stride_k + head * D;
      const int64_t out_base = (int64_t)qk_idx * D;
      if constexpr (PackedQK) {
        using Pair = __ppu_bfloat162;
        const Pair* qp = reinterpret_cast<const Pair*>(q + q_base);
        const Pair* kp = reinterpret_cast<const Pair*>(k + k_base);
        Pair* qo = reinterpret_cast<Pair*>(q_norm + out_base);
        Pair* ko = reinterpret_cast<Pair*>(k_norm + out_base);
        float q_values[D / 32], k_values[D / 32];
        float q_sq = 0.f, k_sq = 0.f;
#pragma unroll
        for (int pair = lane; pair < D / 2; pair += 32) {
          const float2 qv = __bfloat1622float2(qp[pair]);
          const float2 kv = __bfloat1622float2(kp[pair]);
          const int i = (pair / 32) * 2;
          q_values[i] = qv.x;
          q_values[i + 1] = qv.y;
          k_values[i] = kv.x;
          k_values[i + 1] = kv.y;
          q_sq += qv.x * qv.x + qv.y * qv.y;
          k_sq += kv.x * kv.x + kv.y * kv.y;
        }
        const float qi = rsqrtf(folded_gate_warp_sum(q_sq) + eps);
        const float ki = rsqrtf(folded_gate_warp_sum(k_sq) + eps);
#pragma unroll
        for (int pair = lane; pair < D / 2; pair += 32) {
          const int i = (pair / 32) * 2;
          qo[pair] = __float22bfloat162_rn(make_float2(q_values[i] * qi, q_values[i + 1] * qi));
          ko[pair] = __float22bfloat162_rn(make_float2(k_values[i] * ki, k_values[i + 1] * ki));
        }
      } else {
        float q_sq = 0.0f;
        float k_sq = 0.0f;
#pragma unroll
        for (int d = lane; d < D; d += 32) {
          float qv = __bfloat162float(q[q_base + d]);
          float kv = __bfloat162float(k[k_base + d]);
          q_sq += qv * qv;
          k_sq += kv * kv;
        }
        float q_inv = rsqrtf(folded_gate_warp_sum(q_sq) + eps);
        float k_inv = rsqrtf(folded_gate_warp_sum(k_sq) + eps);
#pragma unroll
        for (int d = lane; d < D; d += 32) {
          q_norm[out_base + d] = __float2bfloat16_rn(__bfloat162float(q[q_base + d]) * q_inv);
          k_norm[out_base + d] = __float2bfloat16_rn(__bfloat162float(k[k_base + d]) * k_inv);
        }
      }
    }
  }

  // Each warp owns four tokens; lanes own contiguous heads. Global gate
  // loads/stores stay contiguous while eight warp totals provide carry.
  __shared__ float totals[8 * 32];
  if ((int)blockIdx.x < num_chunks) {
    const int chunk_id = blockIdx.x;
    const int seq = chunk_indices[chunk_id * 2];
    const int local_chunk = chunk_indices[chunk_id * 2 + 1];
    const int begin = cu_seqlens[seq] + local_chunk * CS;
    const int end = min(begin + CS, cu_seqlens[seq + 1]);
    const int head = lane;
    float negative_scale = 0.0f, bias = 0.0f;
    if (head < H) {
      negative_scale = -expf(A_log[head]);
      bias = dt_bias[head];
    }
    float prefixes[4], betas[4];
    float local = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int token = begin + warp * 4 + item;
      float gate = 0.0f, beta_value = 0.0f;
      if (head < H && token < end) {
        const float x = __bfloat162float(a[(int64_t)token * stride_a + head]) + bias;
        const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
        gate = negative_scale * softplus;
        const float raw_beta = __bfloat162float(b[(int64_t)token * stride_b + head]);
        beta_value = __bfloat162float(__float2bfloat16_rn(1.0f / (1.0f + expf(-raw_beta))));
      }
      local += gate;
      prefixes[item] = local;
      betas[item] = beta_value;
    }
    if (head < H)
      totals[warp * 32 + head] = local;
    __syncthreads();
    if (head < H) {
      float carry = 0.0f;
#pragma unroll
      for (int previous = 0; previous < 8; ++previous)
        if (previous < warp)
          carry += totals[previous * 32 + head];
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int token = begin + warp * 4 + item;
        if (token < end) {
          const int64_t offset = (int64_t)token * H + head;
          g_cum[offset] = carry + prefixes[item];
          beta[offset] = betas[item];
        }
      }
    }
  }
}

} // namespace flashqla_folded

#if defined(FLASHQLA_ARCH_150)
namespace flashqla_folded {
using FoldE = __ppu_bfloat16;

// A channel-major R chunk is packed [DK,actual], including partial chunks.
template <typename Element>
__device__ __forceinline__ void ff_aiu_load_channel_major_r(
    Element* dst,const Element* src,int actual) {
#if defined(FLASHQLA_ARCH_150)
  using Load=flashqla::PpuAiuLoad<32*32*16,
      typename fq::ElementTraits<Element>::CuteType,false,32,32,true>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t*>(src),128,actual,actual);
  CUTE_UNROLL
  for(int cube=0;cube<4;++cube) {
    uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst+cube*1024));
    Load::copy(reinterpret_cast<void*>(static_cast<uintptr_t>(address)),
        reinterpret_cast<const void*>(src),desc,0,cube*32);
  }
#endif
}

// The first half of each coefficient tile is channel-major R. Output
// kernels also load M, while the transition kernel consumes R alone.
template <typename Element, bool StateOnly, int Width>
__device__ __forceinline__ void folded_load_tail(
    Element* Kd, Element* Vd, Element* Qd, Element* Ad, float* gd,
    const Element* q, const Element* k, const Element* v, const Element* a,
    const float* g, const float* beta, int Hg, int kv_head, int head,
    int64_t v_stride, int tokens, int v_offset, int start, int actual,
    bool zero_v) {
  const int64_t coeff_row = (int64_t)head * tokens + start;
  const int64_t qk_row = ((int64_t)start * Hg + kv_head) * FF_DK;
  if constexpr (!StateOnly)
    ff_aiu_swzl_write_kq_cs32<Element>(Qd, q + qk_row, Hg * FF_DK, actual);
  ff_aiu_swzl_write_kq_cs32<Element>(Kd, k + qk_row, Hg * FF_DK, actual);
  ff_aiu_load_2d_b16(reinterpret_cast<Element*>(gd), g + coeff_row,
                     2, actual, 4);
  if (!zero_v)
    ff_aiu_load_2d_b16(Vd, v + (int64_t)start * v_stride + head * FF_DV + v_offset,
                       Width, actual, v_stride * 2, FF_CS);
  ff_aiu_load_channel_major_r(Ad, a + coeff_row * FF_DK, actual);
  if constexpr (!StateOnly)
    ff_aiu_load_2d_b16(Ad + FF_CS * FF_DK,
                       reinterpret_cast<const Element*>(beta) + coeff_row * FF_CS,
                       FF_CS, actual, FF_CS * 2, FF_CS);
  ff_aiu_commit();
}

template <typename Element, int Hg, int Groups, bool StateOnly>
__device__ __forceinline__ void folded_preload_chunks(
    int n_chunks, int start, int first_slot, int seq_end, int warp, int lane,
    Element* Q_all, Element* K_all, Element* V_all, Element* A_all, float* g_all,
    const Element* q, const Element* k, const Element* v, const Element* a,
    const float* g, const float* beta, int kv_head, int head, int64_t v_stride,
    bool zero_v, int tokens, int v_offset) {
  constexpr int Width = 64 * Groups;
  constexpr int AElements = StateOnly ? FF_CS * FF_DK : FoldedAElements;
  constexpr int k_warp = Groups == 1 ? 1 : 4;
  constexpr int v_warp = Groups == 1 ? 2 : 1;
  constexpr int a_warp = Groups == 1 ? 2 : 5;
  constexpr int g_warp = Groups == 1 ? 3 : 2;
  for (int i = 0; i < n_chunks; ++i) {
    const int s = start + i * FF_CS;
    if (min(FF_CS, seq_end - s) < FF_CS) return;
    const int slot = (first_slot + i) % FF_NBUF;
    const int64_t coeff_row = (int64_t)head * tokens + s;
    const int64_t qk_row = ((int64_t)s * Hg + kv_head) * FF_DK;
    if (!StateOnly && warp == 0 && lane == 0)
      ff_aiu_swzl_write_kq_cs32<Element>(
          Q_all + slot * FF_KELEMS, q + qk_row, Hg * FF_DK, FF_CS);
    if (warp == k_warp && lane == 0)
      ff_aiu_swzl_write_kq_cs32<Element>(
          K_all + slot * FF_KELEMS, k + qk_row, Hg * FF_DK, FF_CS);
    if (!zero_v && warp == v_warp && lane == 0)
      ff_aiu_load_2d_b16(V_all + slot * FF_CS * Width,
                         v + (int64_t)s * v_stride + head * FF_DV + v_offset,
                         Width, FF_CS, v_stride * 2);
    if (warp == a_warp && lane == 0) {
      Element* coeff = A_all + slot * AElements;
      ff_aiu_load_channel_major_r(coeff, a + coeff_row * FF_DK, FF_CS);
      if constexpr (!StateOnly)
        ff_aiu_load_2d_b16(coeff + FF_CS * FF_DK,
                           reinterpret_cast<const Element*>(beta) + coeff_row * FF_CS,
                           FF_CS, FF_CS, FF_CS * 2);
    }
    if (warp == g_warp && lane == 0)
      ff_aiu_load_2d_b16(reinterpret_cast<Element*>(g_all + slot * FF_CS),
                         g + coeff_row, 2, FF_CS, 4);
    ff_aiu_commit();
  }
}

template <int H, int Groups, bool Prepare>
struct FoldedGeometry {
  using Smem = std::conditional_t<Prepare, FoldedStateSmemLayout<Groups>,
                                  FoldedForwardSmemLayout>;
  static constexpr int VPitch = Prepare ? Groups * 64 : 64;
  static constexpr int AElements = Prepare ? FF_CS * FF_DK : FoldedAElements;
  static constexpr bool PrefetchAfterDrain = H < 32;

  __device__ __forceinline__ static int dv_base(int prepare_task, int gid) {
    if constexpr (Prepare)
      return (prepare_task % (2 / Groups) + gid) * 64;
    return (int(blockIdx.x) & 1) * 64;
  }

  __device__ __forceinline__ static int v_base(int gid) {
    return Prepare ? gid * 64 : 0;
  }
};

__device__ __forceinline__ int folded_prepare_start(
    int start, int end, int requested, bool transition) {
  const int total = (end - start + FF_CS - 1) / FF_CS;
  const int chunks = min(max(requested, 0), total);
  if (transition && !(chunks > 0 && chunks == total)) return -1;
  return min(end, start + (total - chunks) * FF_CS);
}

template <typename Tensor, typename Coordinates>
__device__ __forceinline__ void folded_seed_state(
    Tensor& state, const Coordinates& coordinates, int dv_base, const float* seed) {
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(coordinates); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(coordinates); ++n) {
      CUTE_UNROLL
      for (int vv = 0; vv < cute::size<0>(coordinates); ++vv) {
        const auto coord = coordinates(vv, m, n);
        const int dv = dv_base + cute::get<0>(coord);
        const int dk = cute::get<1>(coord);
        state(vv, m, n) = seed[dv * FF_DK + dk];
      }
    }
  }
}

template <typename Tensor, typename Coordinates>
__device__ __forceinline__ void folded_transition_state(
    Tensor& state, const Coordinates& coordinates, int dv_base) {
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(coordinates); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(coordinates); ++n) {
      CUTE_UNROLL
      for (int vv = 0; vv < cute::size<0>(coordinates); ++vv) {
        const auto coord = coordinates(vv, m, n);
        state(vv, m, n) =
            cute::get<1>(coord) == dv_base + cute::get<0>(coord) ? 1.0f : 0.0f;
      }
    }
  }
}

template <typename StateElement>
__device__ __forceinline__ void folded_store_prepared(
    StateElement* output, float* seed, int64_t pool_base, int local, float value) {
  if (seed != nullptr)
    seed[local] = value;
  else
    fq::state_from_f32(output[pool_base + local], value);
}

__device__ __forceinline__ void folded_pad_gates(float* gates, int actual, int tx) {
  if (actual < FF_CS) {
    if (tx < FF_CS && tx >= actual) gates[tx] = gates[actual - 1];
    __syncthreads();
  }
}

template <typename Element, bool Prepare>
__device__ __forceinline__ float folded_residual(
    const Element* values, int index, float gate, float u, bool transition) {
  float residual = fq::element_to_float<Element>(values[index]) - gate * u;
  if constexpr (Prepare) {
    if (transition) residual = -gate * u;
  }
  return residual;
}

template <typename Element, bool Prepare, typename State, typename Output>
__device__ __forceinline__ void folded_update(
    const uint32_t lhs[FF_CS / 16][1][4], const Element* coefficients,
    int lane, int group, State& state, Output& output) {
  if (((lane / 32 + group) & 1) != 0)
    ff_gemm_regA_ktB_m4_acc<Element, State, true, false>(lhs, coefficients, lane, state);
  else
    ff_gemm_regA_ktB_m4_acc<Element, State, false, false>(lhs, coefficients, lane, state);
  if constexpr (!Prepare)
    ff_gemm_regA_swzlB_m4<FF_CS, FF_CS, Element>(
        lhs, coefficients + FF_CS * FF_DK, lane, output, /*clear_acc=*/false);
}

__device__ __forceinline__ int64_t folded_m_offset(int begin, int head, int col,
                                                   int tokens) {
  return (int64_t)tokens * 24 * 128 + ((int64_t)head * tokens + begin) * 32 +
         col;
}
__device__ __forceinline__ int64_t folded_g_offset(int begin, int head,
                                                   int tokens) {
  return (int64_t)tokens * 24 * 160 + ((int64_t)head * tokens + begin) * 2;
}
__global__ void __launch_bounds__(384, 2)
    gdn_folded_grouped_kernel(const FoldE *q, const FoldE *k, const FoldE *a,
                              const float *beta, const float *g,
                              FoldE *coefficients, const int *cu,
                              const int *chunks, int count, int tokens,
                              float scale) {
  constexpr int C = 32, D = 128, H = 24, Hg = 8;
  const int task = blockIdx.x, chunk = task / Hg, key_head = task % Hg;
  if (chunk >= count)
    return;
  const int seq = chunks ? chunks[chunk * 2] : 0;
  const int ci = chunks ? chunks[chunk * 2 + 1] : chunk;
  const int begin = cu[seq] + ci * C, actual = min(C, cu[seq + 1] - begin);
  if (actual <= 0)
    return;
  const int group = threadIdx.x / 128, tid = threadIdx.x % 128,
            head = key_head * 3 + group;
  __shared__ __align__(16) FoldE kb[C * D], qb[C * D], ab[3 * C * C],
      ag[3 * C * C], pg[3 * C * C];
  __shared__ float qk[C * C], gates[3 * C], bets[3 * C], reverse[3 * C];
  FoldE *ai = ab + group * C * C;
  FoldE *agi = ag + group * C * C;
  FoldE *pgi = pg + group * C * C;
  float *gs = gates + group * C;
  float *bs = bets + group * C;
  float *rs = reverse + group * C;
  if (threadIdx.x == 0) {
    flashqla_folded::ff_aiu_swzl_write_kq_cs32<FoldE>(
        kb, k + (int64_t)begin * Hg * D + key_head * D, Hg * D, actual);
    flashqla_folded::ff_aiu_swzl_write_kq_cs32<FoldE>(
        qb, q + (int64_t)begin * Hg * D + key_head * D, Hg * D, actual);
  }

  if (tid == 0)
    flashqla_folded::ff_aiu_load_2d_b16(
        ai, a + (int64_t)begin * H * C + head * C, C, actual, H * C * 2, C);

  flashqla_folded::ff_aiu_commit();
  if (tid < C) {
    gs[tid] = tid < actual ? g[(int64_t)(begin + tid) * H + head] : 0.0f;
    bs[tid] = tid < actual ? beta[(int64_t)(begin + tid) * H + head] : 0.0f;
  }
  flashqla_folded::ff_aiu_wait_group<0>();
  __syncthreads();
  using Mma = flashqla::TiledMmaM32T<FoldE>;
  Mma mma;
  auto thr = mma.get_thread_slice(tid);
  if (group == 0) {
    auto acc = flashqla::make_acc<C, C, Mma, FoldE>(tid);

    flashqla::flashqla_gemm_ss_swzlAB_m32<C, C, D, decltype(acc), FoldE>(
        qb, kb, tid, acc);
    flashqla::store_acc_to_smem_fp32<Mma>(acc, qk, C, C, C, tid);
  }
  __syncthreads();

  if (tid < C)
    rs[tid] = exp2f((gs[actual - 1] - gs[tid]) * flashqla_folded::FF_LOG2E);
  CUTE_UNROLL
  for (int step = 0; step < C * C / 128; ++step) {
    const int z = tid + step * 128;
    const int row = z / C, col = z % C;
    float av = 0.0f, pv = 0.0f;
    if (row >= col && row < actual && col < actual) {
      const float ratio =
          exp2f((gs[row] - gs[col]) * flashqla_folded::FF_LOG2E);
      av = flashqla::element_to_float<FoldE>(ai[z]) * ratio * bs[col];
      pv = scale * ratio * qk[z];
    }
    auto ag_native = flashqla::make_smem_swzA_cs32<C, C>(agi);
    auto pg_native = flashqla::make_smem_swzA_cs32<C, C>(pgi);
    ag_native(col, row) = typename flashqla::ElementTraits<FoldE>::CuteType(av);
    pg_native(row, col) = typename flashqla::ElementTraits<FoldE>::CuteType(pv);
  }
  __syncthreads();
  {
    auto sa = flashqla::make_smem_tensor_h<C, C>(ai);
    auto sat = flashqla::transpose_tensor(sa);
    auto acc = flashqla::make_acc<D, C, Mma, FoldE>(tid);
    cute::clear(acc);
    flashqla::flashqla_gemm_ss_swzlA_trans_acc_m32<D, C, decltype(sat),
                                                   decltype(acc), FoldE>(
        kb, sat, tid, acc);
    auto coords = thr.partition_C(cute::make_identity_tensor(
        cute::make_shape(cute::Int<D>{}, cute::Int<C>{})));
    auto rr = cute::recast<float>(acc);
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(coords); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(coords); ++n) {
        CUTE_UNROLL
        for (int j = 0; j < cute::size<0>(coords); j += 2) {
          const auto xy0 = coords(j, m, n), xy1 = coords(j + 1, m, n);
          const int dk0 = cute::get<0>(xy0), cs0 = cute::get<1>(xy0);
          const int dk1 = cute::get<0>(xy1), cs1 = cute::get<1>(xy1);
          const FoldE x0 = flashqla::float_to_element<FoldE>(rr(j, m, n) *
                                                             bs[cs0] * rs[cs0]);
          const FoldE x1 = flashqla::float_to_element<FoldE>(rr(j + 1, m, n) *
                                                             bs[cs1] * rs[cs1]);
          const int64_t base = ((int64_t)head * tokens + begin) * D;
          if (actual == C) {
            const uint32_t pair =
                flashqla::pack_b16x2_u32(flashqla::element_bits16<FoldE>(x0),
                                         flashqla::element_bits16<FoldE>(x1));
            *reinterpret_cast<uint32_t *>(coefficients + base + dk0 * C + cs0) =
                pair;
          } else {
            if (cs0 < actual)
              coefficients[base + dk0 * actual + cs0] = x0;
            if (cs1 < actual)
              coefficients[base + dk1 * actual + cs1] = x1;
          }
        }
      }
    }
  }
  {
    auto acc = flashqla::make_acc<C, C, Mma, FoldE>(tid);
    flashqla::flashqla_gemm_ss_swzlAB_m32<C, C, C, decltype(acc), FoldE>(
        pgi, agi, tid, acc);
    auto coords = thr.partition_C(cute::make_identity_tensor(
        cute::make_shape(cute::Int<C>{}, cute::Int<C>{})));
    auto rm = cute::recast<float>(acc);
    const int64_t base = folded_m_offset(begin, head, 0, tokens);
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(coords); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(coords); ++n) {
        CUTE_UNROLL
        for (int j = 0; j < cute::size<0>(coords); j += 2) {
          const auto xy = coords(j, m, n);
          const int row = cute::get<0>(xy), col = cute::get<1>(xy);
          if (row < actual) {
            const auto offset = fq::SmemLayoutSwzA_cs32<C, C>{}(
                cute::make_coord(row, col));
            *reinterpret_cast<__ppu_bfloat162 *>(coefficients + base + offset) =
                __floats2bfloat162_rn(rm(j, m, n), rm(j + 1, m, n));
          }
        }
      }
    }
  }
  const int row = tid;
  if (row < actual)
    *reinterpret_cast<float *>(coefficients +
                               folded_g_offset(begin + row, head, tokens)) =
        exp2f(gs[row] * flashqla_folded::FF_LOG2E);
}

__device__ __forceinline__ int64_t folded_m_offset(int begin, int head, int col,
                                                   int tokens, int heads) {
  return (int64_t)tokens * heads * 128 + ((int64_t)head * tokens + begin) * 32 +
         col;
}
__device__ __forceinline__ int64_t folded_g_offset(int begin, int head,
                                                   int tokens, int heads) {
  return (int64_t)tokens * heads * 160 + ((int64_t)head * tokens + begin) * 2;
}

// Keep the existing BF16 kernel first: HGGC 2.2 otherwise increased its
// register allocation from 96 to 104 in the combined translation unit.
// Form the inverse and both folded coefficients before any BF16 rounding.
// R involves cancellation for correlated keys: rounding the inverse first
// introduces a small, persistent state update that accumulates across chunks.
__device__ __forceinline__ auto folded_fp32_mma() {
  using namespace cute;
  return make_tiled_mma(MMA_Atom<PPU0015_16x16x8_F32F32F32F32_TN>{},
                        Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _8>{});
}

template <int Begin = 0, int End = -1, typename TA, typename TB>
__device__ __forceinline__ auto folded_fp32_product(TA a, TB b, int lane) {
  using namespace cute;
  auto mma = folded_fp32_mma();
  auto thr = mma.get_slice(lane);
  auto tile = make_tensor<float>(Layout<Shape<_16, _16>>{});
  auto acc = thr.make_fragment_C(thr.partition_C(tile));
  clear(acc);
  CUTE_UNROLL
  for (int i = Begin; i < (End < 0 ? size<1>(a) / 8 : End); ++i) {
    auto aa = local_tile(a, Shape<_16, _8>{}, make_coord(0, i));
    auto bb = local_tile(b, Shape<_16, _8>{}, make_coord(0, i));
    auto ra = thr.partition_fragment_A(aa), rb = thr.partition_fragment_B(bb);
    copy(AutoVectorizingCopy{}, thr.partition_A(aa), ra);
    copy(AutoVectorizingCopy{}, thr.partition_B(bb), rb);
    gemm(thr, ra, rb, acc);
  }
  return acc;
}

template <int Groups, int StaticH = 0, int StaticHg = 0>
__global__ void __launch_bounds__(128 * Groups, 2)
    gdn_folded_precise_kernel(const FoldE *q, const FoldE *k, FoldE *a,
                              const float *beta, const float *g,
                              FoldE *coefficients, const int *cu,
                              const int *chunks, int count, int tokens,
                              float scale, int heads, int qk_heads) {
  using namespace cute;
  constexpr int C = 32, D = 128, LD = 40, KD = 136, QLD = 36;
  const int H = StaticH ? StaticH : heads, Hg = StaticHg ? StaticHg : qk_heads;
  const int tiles = H / Hg / Groups, tasks_per_chunk = Hg * tiles;
  const int task = blockIdx.x, chunk = task / tasks_per_chunk;
  const int key_head = (task % tasks_per_chunk) / tiles;
  if (chunk >= count)
    return;
  const int seq = chunks ? chunks[chunk * 2] : 0;
  const int ci = chunks ? chunks[chunk * 2 + 1] : chunk;
  const int begin = cu[seq] + ci * C, actual = min(C, cu[seq + 1] - begin);
  if (actual <= 0)
    return;
  const int group = threadIdx.x / 128, tid = threadIdx.x % 128, warp = tid / 32,
            lane = tid % 32,
            head = key_head * (H / Hg) + (task % tiles) * Groups + group;
  // Q/K are dead after the two Gram products; reuse their storage for FP32 K.
  __shared__ __align__(16) union {
    struct {
      FoldE k[C * D], q[C * D];
    } input;
    float k[C * KD];
  } shared;
  __shared__ float gram[C * C], qk[C * QLD], inverse[Groups * C * LD];
  __shared__ float gates[Groups * C], bets[Groups * C], reverse[Groups * C],
      temporary[Groups * 16 * 20];
  float *inv = inverse + group * C * LD;
  float *cross = temporary + group * 16 * 20;
  float *gs = gates + group * C, *bs = bets + group * C,
        *rs = reverse + group * C;
  if (threadIdx.x == 0) {
    ff_aiu_swzl_write_kq_cs32<FoldE>(shared.input.k,
                                     k + (int64_t)begin * Hg * D + key_head * D,
                                     Hg * D, actual);
    ff_aiu_swzl_write_kq_cs32<FoldE>(shared.input.q,
                                     q + (int64_t)begin * Hg * D + key_head * D,
                                     Hg * D, actual);
  }
  ff_aiu_commit();
  if (tid < C) {
    gs[tid] = tid < actual ? g[(int64_t)(begin + tid) * H + head] : 0.f;
    bs[tid] = tid < actual ? beta[(int64_t)(begin + tid) * H + head] : 0.f;
  }
  ff_aiu_wait_group<0>();
  __syncthreads();
  using Mma = flashqla::TiledMmaM32T<FoldE>;
  // Warp 2 produces the strictly upper 16x16 tile of K K^T and Q K^T.
  // The triangular solve and the causal coefficient products never use it.
  if (warp != 2) {
    if constexpr (Groups == 1) {
      CUTE_UNROLL
      for (int which = 0; which < 2; ++which) {
        auto acc = flashqla::make_acc<C, C, Mma, FoldE>(tid);
        flashqla::flashqla_gemm_ss_swzlAB_m32<C, C, D, decltype(acc), FoldE>(
            which ? shared.input.q : shared.input.k, shared.input.k, tid, acc);
        flashqla::store_acc_to_smem_fp32<Mma>(acc, which ? qk : gram, C, C,
                                              which ? QLD : C, tid);
      }
    } else {
      if (group < 2) {
        auto acc = flashqla::make_acc<C, C, Mma, FoldE>(tid);
        flashqla::flashqla_gemm_ss_swzlAB_m32<C, C, D, decltype(acc), FoldE>(
            group ? shared.input.q : shared.input.k, shared.input.k, tid, acc);
        flashqla::store_acc_to_smem_fp32<Mma>(acc, group ? qk : gram, C, C,
                                              group ? QLD : C, tid);
      }
    }
  }
  __syncthreads();
  if (tid < C)
    rs[tid] = exp2f((gs[actual - 1] - gs[tid]) * FF_LOG2E);
  // Reload unswizzled K as FP32 for the coefficient products.
  const bool vector_aligned =
      reinterpret_cast<uintptr_t>(k) % alignof(uint2) == 0;
  for (int i = threadIdx.x * 4; i < C * D; i += blockDim.x * 4) {
    const int row = i / D, dk = i % D;
    float4 values = {0.f, 0.f, 0.f, 0.f};
    if (row < actual) {
      const FoldE *src = k + (int64_t)(begin + row) * Hg * D + key_head * D + dk;
      if (vector_aligned) {
        const uint2 packed = *reinterpret_cast<const uint2 *>(src);
        const auto *pairs = reinterpret_cast<const __ppu_bfloat162 *>(&packed);
        const float2 lo = __bfloat1622float2(pairs[0]);
        const float2 hi = __bfloat1622float2(pairs[1]);
        values = {lo.x, lo.y, hi.x, hi.y};
      } else {
        values = {fq::element_to_float(src[0]), fq::element_to_float(src[1]),
                  fq::element_to_float(src[2]), fq::element_to_float(src[3])};
      }
    }
    *reinterpret_cast<float4 *>(shared.k + row * KD + dk) = values;
  }
  for (int i = threadIdx.x; i < C * C; i += blockDim.x)
    if (i / C < i % C)
      qk[(i / C) * QLD + i % C] = 0.f;
  if (warp == 1)
    for (int i = lane; i < 256; i += 32)
      cross[(i / 16) * 20 + i % 16] =
          gram[(i / 16 + 16) * C + i % 16] * bs[i / 16 + 16];
  if (warp == 0) {
    // Solve two diagonal 16x16 blocks concurrently, then form the cross block
    // with FP32 Tensor Cell products. Keep the inverse in FP32 until R/M.
    const int half = lane / 16, col = lane % 16;
    float column[16];
    CUTE_UNROLL
    for (int row = 0; row < 16; ++row) {
      float sum = 0.f;
      CUTE_UNROLL
      for (int j = 0; j < row; ++j)
        sum += gram[(half * 16 + row) * C + half * 16 + j] * column[j];
      column[row] =
          col == row ? 1.f : (col < row ? -bs[half * 16 + row] * sum : 0.f);
      inv[(half * 16 + row) * LD + half * 16 + col] = column[row];
    }
  }
  __syncthreads();
  if (warp == 0) {
    auto a1 = make_tensor(make_smem_ptr(inv + 16 * LD + 16),
                          Layout<Shape<_16, _16>, Stride<Int<LD>, _1>>{});
    auto l10 = make_tensor(make_smem_ptr(cross),
                           Layout<Shape<_16, _16>, Stride<_1, Int<20>>>{});
    auto acc = folded_fp32_product(a1, l10, lane);
    auto mma = folded_fp32_mma();
    auto t = mma.get_slice(lane);
    auto tmp = make_tensor(make_smem_ptr(cross),
                           Layout<Shape<_16, _16>, Stride<Int<20>, _1>>{});
    copy(AutoVectorizingCopy{}, acc, t.partition_C(tmp));
    __syncwarp();
    auto a0t = make_tensor(make_smem_ptr(inv),
                           Layout<Shape<_16, _16>, Stride<_1, Int<LD>>>{});
    acc = folded_fp32_product(tmp, a0t, lane);
    for (int i = 0; i < size(acc); ++i)
      acc(i) = -acc(i);
    auto dst = make_tensor(make_smem_ptr(inv + 16 * LD),
                           Layout<Shape<_16, _16>, Stride<Int<LD>, _1>>{});
    copy(AutoVectorizingCopy{}, acc, t.partition_C(dst));
  }
  __syncthreads();
  // The products skip the structurally zero upper-right block of A.
  // Materialize those zeros only when exporting A.
  for (int i = tid * 2; i < actual * C; i += 256) {
    const int row = i / C, col = i % C;
    const bool zero = row < 16 && col >= 16;
    const auto lo = fq::float_to_element<FoldE>(zero ? 0.f : inv[row * LD + col]);
    const auto hi = fq::float_to_element<FoldE>(zero ? 0.f : inv[row * LD + col + 1]);
    const int64_t offset = (int64_t)(begin + row) * H * C + head * C + col;
    *reinterpret_cast<uint32_t *>(a + offset) =
        fq::pack_b16x2_u32(fq::element_bits16<FoldE>(lo), fq::element_bits16<FoldE>(hi));
  }
  auto mma = folded_fp32_mma();
  auto thr = mma.get_slice(lane);
  auto coords = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
  // R^T = K^T A diag(beta * exp(g_end - g)). A is common
  // to all four output tiles handled by this warp.
  const int r_col0 = (warp % 2) * 16;
  auto inverse_tile = make_tensor(make_smem_ptr(inv + r_col0),
                                  Layout<Shape<_16, _32>, Stride<_1, Int<LD>>>{});
  auto inverse_regs = thr.partition_fragment_B(inverse_tile);
  auto inverse_smem = thr.partition_B(inverse_tile);
  const auto b_layout = thr.partition_fragment_B(
      local_tile(inverse_tile, Shape<_16, _8>{}, make_coord(0, 0))).layout();
  static_assert(decltype(size<2>(inverse_regs))::value == 4);
  CUTE_UNROLL
  for (int k_slice = 0; k_slice < 4; ++k_slice)
    if (r_col0 == 0 || k_slice >= 2)
      copy(AutoVectorizingCopy{}, inverse_smem(_, _, k_slice),
           inverse_regs(_, _, k_slice));
  for (int tile = warp; tile < 16; tile += 4) {
    const int dk_begin = (tile / 2) * 16, col0 = r_col0;
    auto ka = make_tensor(make_smem_ptr(shared.k + dk_begin),
                          Layout<Shape<_16, _32>, Stride<_1, Int<KD>>>{});
    auto out = make_tensor<float>(Layout<Shape<_16, _16>>{});
    auto acc = thr.make_fragment_C(thr.partition_C(out));
    clear(acc);
    CUTE_UNROLL
    for (int k_slice = 0; k_slice < 4; ++k_slice) {
      if (r_col0 == 0 || k_slice >= 2) {
        auto k_tile = local_tile(ka, Shape<_16, _8>{}, make_coord(0, k_slice));
        auto k_regs = thr.partition_fragment_A(k_tile);
        copy(AutoVectorizingCopy{}, thr.partition_A(k_tile), k_regs);
        auto a_slice = inverse_regs(_, _, k_slice);
        auto a_regs = make_tensor(a_slice.data(), b_layout);
        gemm(thr, k_regs, a_regs, acc);
      }
    }
    CUTE_UNROLL
    for (int i = 0; i < size(acc); i += 2) {
      const auto xy0 = coords(i), xy1 = coords(i + 1);
      const int dk0 = dk_begin + get<0>(xy0), cs0 = col0 + get<1>(xy0);
      const int dk1 = dk_begin + get<0>(xy1), cs1 = col0 + get<1>(xy1);
      const FoldE x0 =
          flashqla::float_to_element<FoldE>(acc(i) * bs[cs0] * rs[cs0]);
      const FoldE x1 =
          flashqla::float_to_element<FoldE>(acc(i + 1) * bs[cs1] * rs[cs1]);
      const int64_t base = ((int64_t)head * tokens + begin) * D;
      if (actual == C) {
        *reinterpret_cast<uint32_t *>(coefficients + base + dk0 * C + cs0) =
            flashqla::pack_b16x2_u32(flashqla::element_bits16<FoldE>(x0),
                                     flashqla::element_bits16<FoldE>(x1));
      } else {
        if (cs0 < actual)
          coefficients[base + dk0 * actual + cs0] = x0;
        if (cs1 < actual)
          coefficients[base + dk1 * actual + cs1] = x1;
      }
    }
  }

  // M = scale * diag(exp(g)) tril(Q K^T) A diag(beta * exp(-g)).
  const int row0 = (warp / 2) * 16, col0 = (warp % 2) * 16;
  auto qa = make_tensor(make_smem_ptr(qk + row0 * QLD),
                        Layout<Shape<_16, _32>, Stride<Int<QLD>, _1>>{});
  auto ia = make_tensor(make_smem_ptr(inv + col0),
                        Layout<Shape<_16, _32>, Stride<_1, Int<LD>>>{});
  auto out_tile = make_tensor<float>(Layout<Shape<_16, _16>>{});
  auto acc = thr.make_fragment_C(thr.partition_C(out_tile));
  clear(acc);
  if (warp == 0)
    acc = folded_fp32_product<0, 2>(qa, ia, lane);
  else if (warp == 2)
    acc = folded_fp32_product(qa, ia, lane);
  else if (warp == 3)
    acc = folded_fp32_product<2>(qa, ia, lane);
  CUTE_UNROLL
  for (int i = 0; i < size(acc); i += 2) {
    // Each adjacent accumulator pair occupies an aligned column pair.
    const auto xy = coords(i);
    const int row = row0 + get<0>(xy), col = col0 + get<1>(xy);
    if (row < actual) {
      const float lo_weight =
          row >= col && col < actual
              ? scale * bs[col] * exp2f((gs[row] - gs[col]) * FF_LOG2E)
              : 0.f;
      const float hi_weight =
          row > col && col + 1 < actual
              ? scale * bs[col + 1] * exp2f((gs[row] - gs[col + 1]) * FF_LOG2E)
              : 0.f;
      const auto lo = fq::float_to_element<FoldE>(acc(i) * lo_weight);
      const auto hi = fq::float_to_element<FoldE>(acc(i + 1) * hi_weight);
      const int64_t offset = folded_m_offset(begin, head, 0, tokens, H) +
          fq::SmemLayoutSwzA_cs32<C, C>{}(make_coord(row, col));
      *reinterpret_cast<uint32_t *>(coefficients + offset) =
          fq::pack_b16x2_u32(fq::element_bits16<FoldE>(lo), fq::element_bits16<FoldE>(hi));
    }
  }
  if (tid < actual)
    *reinterpret_cast<float *>(coefficients +
                               folded_g_offset(begin + tid, head, tokens, H)) =
        exp2f(gs[tid] * FF_LOG2E);
}
template <typename Element>
__forceinline__ __device__ void
folded_load_k_cube(Element *smem, const Element *gmem, int row_stride_elems,
                   int rows, int cube) {
#if defined(__HGGC_ARCH__)
  using AiuWriteCube =
      flashqla::PpuAiuLoad<32 * 32 * 16,
                           typename fq::ElementTraits<Element>::CuteType,
                           /*Trans=*/false,
                           /*CUBE_H=*/32, /*CUBE_W=*/32, /*Swzl=*/true>;
  cute::AiuDesc desc;
  desc.init(reinterpret_cast<const uint8_t *>(gmem), /*h=*/rows, /*w=*/128,
            /*stride_w=*/row_stride_elems);
  uint32_t dst_shared =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem + cube * 32 * 32));
  AiuWriteCube::copy(
      reinterpret_cast<void *>(static_cast<uintptr_t>(dst_shared)),
      reinterpret_cast<const void *>(gmem), desc,
      /*coord_w=*/cube * 32, /*coord_h=*/0);
#else
  (void)smem;
  (void)gmem;
  (void)row_stride_elems;
  (void)rows;
  (void)cube;
#endif
}

__device__ __forceinline__ void kkt_block16_gemm(const float *a, const float *b,
                                                 float *out, int lane) {
  using namespace cute;
  auto mma = make_tiled_mma(MMA_Atom<PPU0015_16x16x8_F32F32F32F32_TN>{},
                            Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _8>{});
  auto thr = mma.get_slice(lane);
  auto ta = make_tensor(make_smem_ptr(a),
                        Layout<Shape<_16, _16>, Stride<Int<20>, _1>>{});
  auto tb = make_tensor(make_smem_ptr(b),
                        Layout<Shape<_16, _16>, Stride<_1, Int<20>>>{});
  auto tc = make_tensor(make_smem_ptr(out),
                        Layout<Shape<_16, _16>, Stride<Int<20>, _1>>{});
  auto acc = thr.make_fragment_C(thr.partition_C(tc));
  clear(acc);
  CUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    auto aa = local_tile(ta, Shape<_16, _8>{}, make_coord(0, i));
    auto bb = local_tile(tb, Shape<_16, _8>{}, make_coord(0, i));
    auto ra = thr.partition_fragment_A(aa), rb = thr.partition_fragment_B(bb);
    copy(AutoVectorizingCopy{}, thr.partition_A(aa), ra);
    copy(AutoVectorizingCopy{}, thr.partition_B(bb), rb);
    gemm(thr, ra, rb, acc);
  }
  // All source elements were consumed before these stores. Alias with either
  // input is allowed; one warp owns the complete tile and synchronizes here.
  copy(AutoVectorizingCopy{}, acc, thr.partition_C(tc));
  __syncwarp();
}

__global__ __launch_bounds__(128, 6) void gdn_folded_kkt_bf16_kernel(
    __ppu_bfloat16 *output, const __ppu_bfloat16 *k, const float *beta,
    const int *cu, const int *chunks, int count, int tokens) {
  using E = __ppu_bfloat16;
  constexpr int C = 32, D = 128, H = 24, Hg = 8, LD = 20;
  const int task = blockIdx.x, chunk = task / Hg, key_head = task % Hg;
  if (chunk >= count)
    return;
  const int seq = chunks ? chunks[2 * chunk] : 0;
  const int ci = chunks ? chunks[2 * chunk + 1] : chunk;
  const int begin = cu[seq] + ci * C, actual = min(C, cu[seq + 1] - begin);
  if (actual <= 0)
    return;
  const int tx = threadIdx.x, warp = tx / 32, lane = tx % 32;
  extern __shared__ __align__(16) char storage[];
  E *kb = reinterpret_cast<E *>(storage);
  float *gram = reinterpret_cast<float *>(storage + C * D * 2);
  if (lane == 0) {
    const E *source = k + (int64_t)begin * Hg * D + key_head * D;
    folded_load_k_cube<E>(kb, source, Hg * D, actual, warp);
  }
  ff_aiu_commit();
  ff_aiu_wait_group<0>();
  __syncthreads();
  auto acc = fq::make_acc<C, C, fq::TiledMmaM32T<E>, E>(tx);
  fq::flashqla_gemm_ss_swzlAB_m32<C, C, D, decltype(acc), E>(kb, kb, tx, acc);
  fq::store_acc_to_smem_fp32<fq::TiledMmaM32T<E>>(acc, gram, C, C, C, tx);
  __syncthreads();
  if (warp >= 3)
    return;
  // K is dead. Two heads reuse its8KiB; the third follows the Gram tile.
  float *work = reinterpret_cast<float *>(
      storage + (warp < 2 ? warp * 3 * 16 * LD * 4 : C * D * 2 + C * C * 4));
  float *a0 = work;
  float *a1 = a0 + 16 * LD;
  float *cross = a1 + 16 * LD;
  const int head = key_head * 3 + warp, half = lane / 16, col = lane % 16;
  float *diagonal = half ? a1 : a0;
  diagonal[col * LD + 16] =
      lane < actual ? beta[(int64_t)(begin + lane) * H + head] : 0.f;
  __syncwarp();
  float column[16];
  CUTE_UNROLL
  for (int i = 0; i < 16; ++i)
    column[i] = 0.f;
  CUTE_UNROLL
  for (int row = 0; row < 16; ++row) {
    float sum = 0.f;
    CUTE_UNROLL
    for (int j = 0; j < row; ++j)
      sum += gram[(half * 16 + row) * C + half * 16 + j] * column[j];
    column[row] =
        col == row ? 1.f : (col < row ? -diagonal[row * LD + 16] * sum : 0.f);
    diagonal[row * LD + col] = column[row];
  }
  __syncwarp();
  if (actual > 16) {
    for (int i = lane; i < 256; i += 32) {
      const int row = i / 16, col = i % 16;
      cross[row * LD + col] = gram[(row + 16) * C + col] * a1[row * LD + 16];
    }
    __syncwarp();
    kkt_block16_gemm(a1, cross, cross, lane);
    kkt_block16_gemm(cross, a0, cross, lane);
  }
  struct alignas(16) Bf16x8 {
    __ppu_bfloat162 a, b, c, d;
  };
  for (int vec = lane; vec < actual * (C / 8); vec += 32) {
    const int row = vec / (C / 8), col = 8 * (vec % (C / 8));
    if (begin + row >= tokens)
      continue;
    const bool negate = row >= 16 && col < 16;
    const float *src = nullptr;
    if (row < 16 && col < 16)
      src = a0 + row * LD + col;
    else if (row >= 16 && col >= 16)
      src = a1 + (row - 16) * LD + col - 16;
    else if (negate)
      src = cross + (row - 16) * LD + col;
    float4 lo{0.f, 0.f, 0.f, 0.f}, hi{0.f, 0.f, 0.f, 0.f};
    if (src != nullptr) {
      const auto *chunks = reinterpret_cast<const float4 *>(src);
      lo = chunks[0];
      hi = chunks[1];
      if (negate) {
        lo = {-lo.x, -lo.y, -lo.z, -lo.w};
        hi = {-hi.x, -hi.y, -hi.z, -hi.w};
      }
    }
    const int64_t offset = ((int64_t)(begin + row) * H + head) * C + col;
    *reinterpret_cast<Bf16x8 *>(output + offset) =
        {__floats2bfloat162_rn(lo.x, lo.y), __floats2bfloat162_rn(lo.z, lo.w),
         __floats2bfloat162_rn(hi.x, hi.y), __floats2bfloat162_rn(hi.z, hi.w)};
  }
}
template <typename Element, typename TC>
__device__ __forceinline__ void
folded_hk_m4(const uint32_t h_regs[FF_DK / 16][1][4], const Element *Kswz,
             int tidx, TC &acc) {
  using Load =
      cute::PPU0015_TSM_LD_SWZL<typename fq::ElementTraits<Element>::CuteType,
                                32, 32, true, false, 4>;
  cute::clear(acc);
  auto r = cute::recast<float>(acc);
  CUTE_UNROLL
  for (int kt = 0; kt < FF_DK / 16; ++kt) {
    uint32_t b[2][4];
    CUTE_UNROLL
    for (int n = 0; n < 2; ++n)
      Load::copy(reinterpret_cast<int *>(b[n]),
                 reinterpret_cast<void *>(fq::ff_swzl_read_base<Element>(Kswz)),
                 (kt % 2) * 16, n * 16, kt / 2, 0);
    CUTE_UNROLL
    for (int n = 0; n < 2; ++n) {
      fq::ppu_mma_impl<Element>(
          r(0, 0, n), r(1, 0, n), r(2, 0, n), r(3, 0, n), r(4, 0, n),
          r(5, 0, n), r(6, 0, n), r(7, 0, n), h_regs[kt][0][0],
          h_regs[kt][0][1], h_regs[kt][0][2], h_regs[kt][0][3], b[n][0],
          b[n][1], b[n][2], b[n][3], r(0, 0, n), r(1, 0, n), r(2, 0, n),
          r(3, 0, n), r(4, 0, n), r(5, 0, n), r(6, 0, n), r(7, 0, n));
    }
  }
}

template <int HV, int HGV, bool StateBf16>
__global__ void __launch_bounds__(HV < 32 ? 128 : 256, 2)
    gdn_folded_state_kernel(const __ppu_bfloat16 *k, const __ppu_bfloat16 *v,
                            const __ppu_bfloat16 *a, const float *g,
                            const int *cu_seqlens, const int *num_warmup_chunks,
                            void *ht, void *mt, int batch_size, int num_tokens,
                            int64_t v_token_stride) {
  using Element = __ppu_bfloat16;
  constexpr int H_TPL = HV, Hg_TPL = HGV, NGRP = HV < 32 ? 1 : 2, DVH_TPL = 64;
  constexpr bool AUX_P_TPL = false, STATE_BF16 = StateBf16;
  const Element *q = k;
  const float *beta = g;
  Element *o = nullptr;
  const fq::StatePool<StateBf16> *h0 = nullptr;
  constexpr const float *seed_state = nullptr;
  constexpr float *seed_result = nullptr;
  // Adjacent CTAs consume the same segment, head and K/R tile.
  const bool compute_mt = (blockIdx.x & 1) != 0;
  const int prepare_task = blockIdx.x / 2;
  fq::StatePool<StateBf16> *ht_out =
      static_cast<fq::StatePool<StateBf16> *>(compute_mt ? mt : ht);
  const int *cp_seq_map = nullptr;
  const int *raw_cu_seqlens = cu_seqlens;
  const int raw_batch_size = batch_size;
  const float scale = 1.0f;
  constexpr bool FF_PREPARE_MODE = true;
  const int folded_sequence = prepare_task / (2 / NGRP);
#define FF_BODY_FOLDED 1
#define FF_BODY_PREPARE 1
#define FF_BODY_LEGACY_TC01 1
#include "fused_fwd_cute_body.inc"
#undef FF_BODY_LEGACY_TC01
#undef FF_BODY_PREPARE
#undef FF_BODY_FOLDED
}
template <bool STATE_BF16>
__global__ void __launch_bounds__(128, 2)
    gdn_folded_seeded_state_kernel(const __ppu_bfloat16 *k, const __ppu_bfloat16 *v,
                            const __ppu_bfloat16 *a, const float *g,
                            const int *cu_seqlens, const int *num_warmup_chunks,
                            void *ht, void *mt, int batch_size, int num_tokens,
                            int64_t v_token_stride, const float *initial_pool,
                            const int *first_slot, float *corrected) {
  using Element = __ppu_bfloat16;
  constexpr int H_TPL = 24, Hg_TPL = 8, NGRP = 1, DVH_TPL = 64;
  constexpr bool AUX_P_TPL = false;
  const Element *q = k;
  const float *beta = g;
  Element *o = nullptr;
  const fq::StatePool<STATE_BF16> *h0 = nullptr;
  // Compact the launch by removing the first segment's transition CTAs.
  constexpr int FirstTiles = 2 / NGRP;
  const int block = blockIdx.x;
  const int task = block + min(block, FirstTiles);
  const bool compute_mt = (task & 1) != 0;
  const int prepare_task = task / 2;
  const bool seed = prepare_task < FirstTiles;
  const int64_t state_elems = FF_DK * FF_DV;
  const float *seed_state = seed
      ? initial_pool + ((int64_t)first_slot[0] * H_TPL + blockIdx.y) * state_elems
      : nullptr;
  float *seed_result = seed ? corrected + (H_TPL + blockIdx.y) * state_elems
                            : nullptr;
  fq::StatePool<STATE_BF16> *ht_out =
      static_cast<fq::StatePool<STATE_BF16> *>(compute_mt ? mt : ht);
  const int *cp_seq_map = nullptr;
  const int *raw_cu_seqlens = cu_seqlens;
  const int raw_batch_size = batch_size;
  const float scale = 1.0f;
  constexpr bool FF_PREPARE_MODE = true;
  const int folded_sequence = prepare_task / (2 / NGRP);
#define FF_BODY_FOLDED 1
#define FF_BODY_PREPARE 1
#define FF_BODY_LEGACY_TC01 1
#include "fused_fwd_cute_body.inc"
#undef FF_BODY_LEGACY_TC01
#undef FF_BODY_PREPARE
#undef FF_BODY_FOLDED
}
// CP composes affine state transitions. Preserve their FP32 values across
// segment boundaries; rounding each transition to BF16 accumulates with length.
template <bool SeedFirst>
__global__ void __launch_bounds__(128, 2)
    gdn_folded_correct_kernel(float *out, const float *pool, const float *ht,
                              const float *mt, const signed char *fallback,
                              const int *mapping, const int *indices, int H) {
  using namespace cute;
  constexpr int D = 128, LD = 132;
  const int v0 = blockIdx.x % 8 * 16, head = blockIdx.x / 8 % H,
            seq = blockIdx.x / (8 * H), lane = threadIdx.x % 32,
            warp = threadIdx.x / 32;
  __shared__ __align__(64) float matrix[64 * LD];
  const uintptr_t mt_address = reinterpret_cast<uintptr_t>(mt);
  const bool mt_use_aiu = mt_address % 64 == 0;
  const bool mt_aligned = mt_address % alignof(float4) == 0;
  __shared__ float state[16 * LD];
  auto mma = folded_fp32_mma();
  auto thr = mma.get_slice(lane);
  auto tile = make_tensor<float>(Layout<Shape<_16, _16>>{});
  auto proto = thr.make_fragment_C(thr.partition_C(tile));
  auto coords = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
  constexpr int N = decltype(size(proto))::value;
  float resident[2][N];
  CUTE_UNROLL
  for (int pass = 0; pass < 2; ++pass) {
    CUTE_UNROLL
    for (int i = 0; i < N; ++i) {
      const auto xy = coords(i);
      const int dk = pass * 64 + warp * 16 + get<0>(xy), dv = v0 + get<1>(xy);
      resident[pass][i] =
          pool[((int64_t)indices[seq] * H + head) * D * D + dv * D + dk];
    }
  }
  const int first = mapping[seq];
  for (int cp = first; cp < mapping[seq + 1]; ++cp) {
    const int64_t base = ((int64_t)cp * H + head) * D * D;
    CUTE_UNROLL
    for (int pass = 0; pass < 2; ++pass) {
      CUTE_UNROLL
      for (int i = 0; i < N; ++i) {
        const auto xy = coords(i);
        const int dk = pass * 64 + warp * 16 + get<0>(xy), dv = get<1>(xy);
        out[base + dk * D + v0 + dv] = resident[pass][i];
        if constexpr (SeedFirst) {
          if (cp == first) {
            // Seeded prepare already wrote the first continuation state.
            resident[pass][i] = out[base + (int64_t)H * D * D + dk * D + v0 + dv];
          } else {
            state[dv * LD + dk] = resident[pass][i];
          }
        } else {
          state[dv * LD + dk] = resident[pass][i];
        }
      }
    }
    if (cp + 1 == mapping[seq + 1])
      break;
    if constexpr (SeedFirst) {
      if (cp == first)
        continue;
    }
    CUTE_UNROLL
    for (int pass = 0; pass < 2; ++pass) {
      if (fallback[cp * H + head]) {
        constexpr int Vec = 4, Width = D / Vec;
        const float *source = mt + base + pass * 64 * D;
        if (mt_use_aiu) {
          // Express the FP32 row as adjacent pairs of b16 elements; the
          // copy engine transfers both halves without numerical conversion.
          if (threadIdx.x == 0) {
            CUTE_UNROLL
            for (int row = 0; row < 64; row += 32) {
              const auto dst = static_cast<uint32_t>(
                  __cvta_generic_to_shared(matrix + row * LD));
              const float *src = source + row * D;
              asm volatile(
                  "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
                  " [%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, %13;\n"
                  :: "r"(dst), "l"(src), "r"(2 * D), "r"(32), "r"(1),
                     "r"(2 * LD), "r"(32), "r"(1),
                     "l"(int64_t(D * sizeof(float))), "l"(int64_t(0)),
                     "r"(0), "r"(0), "r"(0), "r"(0)
                  : "memory");
            }
            ff_aiu_commit();
            ff_aiu_wait_group<0>();
          }
        } else if (mt_aligned) {
          for (int i = threadIdx.x; i < 64 * Width; i += blockDim.x)
            *reinterpret_cast<float4 *>(matrix + (i / Width) * LD + (i % Width) * Vec) =
                *reinterpret_cast<const float4 *>(source + i * Vec);
        } else {
          for (int i = threadIdx.x; i < 64 * D; i += blockDim.x)
            matrix[(i / D) * LD + i % D] = source[i];
        }
        __syncthreads();
        auto a = make_tensor(make_smem_ptr(matrix + warp * 16 * LD),
                             Layout<Shape<_16, Int<D>>, Stride<Int<LD>, _1>>{});
        auto b = make_tensor(make_smem_ptr(state),
                             Layout<Shape<_16, Int<D>>, Stride<Int<LD>, _1>>{});
        auto acc = folded_fp32_product(a, b, lane);
        CUTE_UNROLL
        for (int i = 0; i < N; ++i)
          resident[pass][i] = acc(i);
        __syncthreads();
      } else {
        CUTE_UNROLL
        for (int i = 0; i < N; ++i)
          resident[pass][i] = 0.f;
      }
      CUTE_UNROLL
      for (int i = 0; i < N; ++i) {
        const auto xy = coords(i);
        const int dk = pass * 64 + warp * 16 + get<0>(xy), dv = v0 + get<1>(xy);
        resident[pass][i] += ht[base + dk * D + dv];
      }
    }
  }
}
} // namespace flashqla_folded
#else
// Every registered kernel has an image for both PPU generations. Host dispatch
// admits only tc02; the tc01 placeholders preserve fat-binary registration.
namespace flashqla_folded {
using FoldE = __ppu_bfloat16;
__global__ void gdn_folded_kkt_bf16_kernel(FoldE *, const FoldE *,
                                           const float *, const int *,
                                           const int *, int, int) {}
__global__ void gdn_folded_grouped_kernel(const FoldE *, const FoldE *,
                                          const FoldE *, const float *,
                                          const float *, FoldE *, const int *,
                                          const int *, int, int, float) {}
template <int Groups, int StaticH = 0, int StaticHg = 0>
__global__ void gdn_folded_precise_kernel(const FoldE *, const FoldE *, FoldE *,
                                          const float *, const float *, FoldE *,
                                          const int *, const int *, int, int,
                                          float, int, int) {}
template <int HV, int HGV, bool StateBf16>
__global__ void gdn_folded_state_kernel(const FoldE *, const FoldE *,
                                        const FoldE *, const float *,
                                        const int *, const int *, void *,
                                        void *, int, int, int64_t) {}
template <bool StateBf16>
__global__ void gdn_folded_seeded_state_kernel(const FoldE *, const FoldE *,
                                        const FoldE *, const float *,
                                        const int *, const int *, void *,
                                        void *, int, int, int64_t, const float *,
                                        const int *, float *) {}
template <bool SeedFirst>
__global__ void gdn_folded_correct_kernel(float *, const float *, const float *,
                                          const float *, const signed char *,
                                          const int *, const int *, int) {}
} // namespace flashqla_folded
#endif

namespace flashqla_folded {
template <typename Element, int H_TPL, int Hg_TPL, int NGRP,
          int DVH_TPL = FF_DVH, bool AUX_P_TPL = false,
          bool STATE_BF16 = false>
__global__ void __launch_bounds__(FF_WG * (AUX_P_TPL ? 2 : NGRP), FF_MIN_BLOCKS)
gdn_folded_forward_kernel(
    Element* __restrict__ o,
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const Element* __restrict__ a,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const fq::StatePool<STATE_BF16>* __restrict__ h0,
    fq::StatePool<STATE_BF16>* __restrict__ ht_out,
    fq::StatePool<STATE_BF16>* __restrict__ raw_state,
    const fq::StatePool<STATE_BF16>* __restrict__ raw_state_input,
    const int* __restrict__ raw_state_indices,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ cp_seq_map,
    const int* __restrict__ raw_cu_seqlens,
    const int* __restrict__ num_warmup_chunks,
    const signed char* __restrict__ fallback_mask,
    int batch_size, int num_tokens, int raw_batch_size,
    float scale, int64_t v_token_stride)
{
#if defined(FLASHQLA_ARCH_150)
  constexpr bool FF_PREPARE_MODE = false;
  constexpr bool compute_mt = false;
  constexpr int prepare_task = 0;
  const int folded_sequence = blockIdx.x / 2;
#define FF_BODY_FOLDED 1
#define FF_BODY_INDEXED_TC01 1
#include "fused_fwd_cute_body.inc"
#undef FF_BODY_INDEXED_TC01
#undef FF_BODY_FOLDED
#endif
}

}  // namespace flashqla_folded

namespace {

template <int H, int Hg, int QKStride, int GateStride, bool PackedQK>
void launch_gdn_folded_gated_scan_impl(const void* a, const void* b, const float* alog,
    const float* bias, const int* cu, const int* chunks, float* cumulative, float* beta,
    const void* q, const void* k, void* qnorm, void* knorm, int count, int tokens, int64_t stride_a,
    int64_t stride_b, int64_t stride_q, int64_t stride_k, hggcStream_t stream) {
  const int qk_blocks = (tokens * Hg + 15) / 16;
  const int gate_blocks = count;
  const int blocks = qk_blocks > gate_blocks ? qk_blocks : gate_blocks;
  flashqla_folded::folded_gated_scan<H, Hg, QKStride, GateStride, PackedQK><<<blocks, 256, 0, stream>>>(static_cast<const __ppu_bfloat16*>(a),
      static_cast<const __ppu_bfloat16*>(b), alog, bias, cu, chunks, cumulative, beta,
      static_cast<const __ppu_bfloat16*>(q), static_cast<const __ppu_bfloat16*>(k),
      static_cast<__ppu_bfloat16*>(qnorm), static_cast<__ppu_bfloat16*>(knorm), count, tokens,
      stride_a, stride_b, stride_q, stride_k, 1e-6f);
}

template <int H, int Hg>
void launch_gdn_folded_gated_scan_for_heads(const void* a, const void* b, const float* alog,
    const float* bias, const int* cu, const int* chunks, float* cumulative, float* beta,
    const void* q, const void* k, void* qnorm, void* knorm, int count, int tokens, int64_t stride_a,
    int64_t stride_b, int64_t stride_q, int64_t stride_k, hggcStream_t stream) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
                              reinterpret_cast<uintptr_t>(qnorm) |
                              reinterpret_cast<uintptr_t>(knorm);
  const bool packed = (alignment & 3) == 0 && (stride_q & 1) == 0 && (stride_k & 1) == 0;
  if (!packed)
    launch_gdn_folded_gated_scan_impl<H, Hg, 0, 0, false>(a, b, alog, bias, cu, chunks, cumulative, beta,
        q, k, qnorm, knorm, count, tokens, stride_a, stride_b, stride_q, stride_k, stream);
  else if (stride_q == (2 * Hg + H) * 128 && stride_k == (2 * Hg + H) * 128 &&
           stride_a == 2 * H && stride_b == 2 * H)
    launch_gdn_folded_gated_scan_impl<H, Hg, (2 * Hg + H) * 128, 2 * H, true>(a, b, alog, bias, cu, chunks, cumulative,
        beta, q, k, qnorm, knorm, count, tokens, stride_a, stride_b, stride_q, stride_k, stream);
  else
    launch_gdn_folded_gated_scan_impl<H, Hg, 0, 0, true>(a, b, alog, bias, cu, chunks, cumulative, beta, q,
        k, qnorm, knorm, count, tokens, stride_a, stride_b, stride_q, stride_k, stream);
}

} // namespace

void launch_gdn_folded_gated_scan(const void* a, const void* b, const float* alog,
    const float* bias, const int* cu, const int* chunks, float* cumulative, float* beta,
    const void* q, const void* k, void* qnorm, void* knorm, int count, int tokens, int H, int Hg,
    int64_t stride_a, int64_t stride_b, int64_t stride_q, int64_t stride_k, hggcStream_t stream) {
    if (H == 24 && Hg == 8)
        launch_gdn_folded_gated_scan_for_heads<24, 8>(a, b, alog, bias, cu, chunks,
            cumulative, beta, q, k, qnorm, knorm, count, tokens, stride_a, stride_b,
            stride_q, stride_k, stream);
    else if (H == 12 && Hg == 4)
        launch_gdn_folded_gated_scan_for_heads<12, 4>(a, b, alog, bias, cu, chunks,
            cumulative, beta, q, k, qnorm, knorm, count, tokens, stride_a, stride_b,
            stride_q, stride_k, stream);
    else
        throw std::invalid_argument("unsupported folded gating head configuration");
}

#define FOLDED_HEAD_CONFIGS(M)                                                 \
  M(128, 16)                                                                   \
  M(64, 16)                                                                    \
  M(64, 8)                                                                     \
  M(48, 16)                                                                    \
  M(32, 16)                                                                    \
  M(32, 8)                                                                     \
  M(32, 4)                                                                     \
  M(24, 8)                                                                     \
  M(16, 16)                                                                    \
  M(16, 8)                                                                     \
  M(16, 4)                                                                     \
  M(16, 2)                                                                     \
  M(12, 4)                                                                     \
  M(8, 4)                                                                      \
  M(8, 2)                                                                      \
  M(8, 1)                                                                      \
  M(6, 2)                                                                      \
  M(4, 2)

void launch_folded_coefficients_cute(const void *q, const void *k, void *a,
                                     const float *beta, const float *g,
                                     void *coefficients, const int *cu,
                                     const int *chunks, int count, int tokens,
                                     int H, int Hg, float scale, bool precise,
                                     hggcStream_t stream) {
  using namespace flashqla_folded;
  if (!precise) {
    gdn_folded_kkt_bf16_kernel<<<count * 8, 128, 16128, stream>>>(
        static_cast<FoldE *>(a), static_cast<const FoldE *>(k), beta, cu,
        chunks, count, tokens);
    gdn_folded_grouped_kernel<<<count * 8, 384, 0, stream>>>(
        static_cast<const FoldE *>(q), static_cast<const FoldE *>(k),
        static_cast<const FoldE *>(a), beta, g,
        static_cast<FoldE *>(coefficients), cu, chunks, count, tokens, scale);
    return;
  }
  const int ratio = H / Hg;
#define LAUNCH_PRECISE(G, SH, SG)                                              \
  gdn_folded_precise_kernel<G, SH, SG>                                         \
      <<<count * Hg *(ratio / G), 128 * G, 0, stream>>>(                       \
          static_cast<const FoldE *>(q), static_cast<const FoldE *>(k),        \
          static_cast<FoldE *>(a), beta, g,                                    \
          static_cast<FoldE *>(coefficients), cu, chunks, count, tokens,       \
          scale, H, Hg)
  if (H == 24 && Hg == 8) {
    LAUNCH_PRECISE(3, 24, 8);
  } else if (H == 12 && Hg == 4) {
    LAUNCH_PRECISE(3, 12, 4);
  } else if (ratio == 1) {
    LAUNCH_PRECISE(1, 0, 0);
  } else if (ratio == 3) {
    LAUNCH_PRECISE(3, 0, 0);
  } else {
    LAUNCH_PRECISE(2, 0, 0);
  }
#undef LAUNCH_PRECISE
}

template <int H, int Hg, bool StateBf16>
void launch_folded_state(const void *k, const void *v, const void *coefficients,
                         const int *cu, const int *warmup, void *ht, void *mt,
                         int batch, int tokens, int64_t v_stride,
                         hggcStream_t stream) {
  using namespace flashqla_folded;
  const float *g = reinterpret_cast<const float *>(
      static_cast<const FoldE *>(coefficients) + (int64_t)tokens * H * 160);
  const auto status = hggcFuncSetAttribute(
      gdn_folded_state_kernel<H, Hg, StateBf16>,
      hggcFuncAttributeMaxDynamicSharedMemorySize, 130 * 1024);
  if (status != hggcSuccess)
    throw std::runtime_error(hggcGetErrorString(status));
  constexpr int Groups = H < 32 ? 1 : 2;
  gdn_folded_state_kernel<H, Hg, StateBf16>
      <<<dim3(batch * 4 / Groups, H), dim3(128, Groups),
         FoldedStateSmemLayout<Groups>::TOTAL,
         stream>>>(static_cast<const FoldE *>(k), static_cast<const FoldE *>(v),
                   static_cast<const FoldE *>(coefficients), g, cu, warmup, ht,
                   mt, batch, tokens, v_stride);
}

template <bool StateBf16>
void launch_folded_seeded_state(const void *k, const void *v,
                              const void *coefficients, const int *cu,
                              const int *warmup, void *ht, void *mt, int batch,
                              int tokens, int64_t v_stride, hggcStream_t stream,
                              const float *initial_pool, const int *first_slot,
                              float *corrected) {
  using namespace flashqla_folded;
  constexpr int H = 24;
  const auto *coeff = static_cast<const FoldE *>(coefficients);
  const auto *g = reinterpret_cast<const float *>(coeff + (int64_t)tokens * H * 160);
  const auto status = hggcFuncSetAttribute(
      gdn_folded_seeded_state_kernel<StateBf16>,
      hggcFuncAttributeMaxDynamicSharedMemorySize, 130 * 1024);
  if (status != hggcSuccess)
    throw std::runtime_error(hggcGetErrorString(status));
  gdn_folded_seeded_state_kernel<StateBf16>
      <<<dim3(batch * 4 - 2, H), 128, FoldedStateSmemLayout<1>::TOTAL, stream>>>(
          static_cast<const FoldE *>(k), static_cast<const FoldE *>(v), coeff,
          g, cu, warmup, ht, mt, batch, tokens, v_stride, initial_pool, first_slot,
          corrected);
}

void launch_folded_prepare_h_cute(const void *k, const void *v,
                                  const void *coefficients, const int *cu,
                                  const int *warmup, void *ht, void *mt,
                                  int batch, int tokens, int H, int Hg,
                                  int64_t v_stride, bool precise,
                                  hggcStream_t stream, const float *initial_pool,
                                  const int *first_slot, float *corrected) {
  if (corrected != nullptr) {
    if (precise)
      launch_folded_seeded_state<false>(k, v, coefficients, cu, warmup, ht, mt,
          batch, tokens, v_stride, stream, initial_pool, first_slot, corrected);
    else
      launch_folded_seeded_state<true>(k, v, coefficients, cu, warmup, ht, mt,
          batch, tokens, v_stride, stream, initial_pool, first_slot, corrected);
    return;
  }
  if (!precise && H == 24 && Hg == 8) {
    launch_folded_state<24, 8, true>(k, v, coefficients, cu, warmup, ht, mt,
                                     batch, tokens, v_stride, stream);
    return;
  }
#define STATE_CASE(HV, HGV)                                                    \
  if (H == HV && Hg == HGV) {                                                  \
    if (precise)                                                               \
      launch_folded_state<HV, HGV, false>(k, v, coefficients, cu, warmup, ht,  \
                                          mt, batch, tokens, v_stride,         \
                                          stream);                             \
    else                                                                       \
      launch_folded_state<HV, HGV, true>(k, v, coefficients, cu, warmup, ht,   \
                                         mt, batch, tokens, v_stride, stream); \
    return;                                                                    \
  }
  FOLDED_HEAD_CONFIGS(STATE_CASE)
#undef STATE_CASE
  throw std::invalid_argument("unsupported folded state head configuration");
}

template <int H, int Hg>
void launch_folded_forward(void *output, const void *q, const void *k,
                           const void *v, const void *coefficients,
                           const int *cu, const int *mapping, const int *raw_cu,
                           const float *h0, float *pool, const int *indices,
                           int batch, int tokens, int raw_batch, int device,
                           int64_t v_stride, float scale, hggcStream_t stream) {
  using E = __ppu_bfloat16;
  using namespace flashqla_folded;
  const E *a = static_cast<const E *>(coefficients);
  const float *m =
      reinterpret_cast<const float *>(a + (int64_t)tokens * H * 128);
  const float *g =
      reinterpret_cast<const float *>(a + (int64_t)tokens * H * 160);
  static std::once_flag configured[64];
  auto configure = [] {
    hggcFuncSetAttribute(gdn_folded_forward_kernel<E, H, Hg, 1, 64, false, false>,
                         hggcFuncAttributeMaxDynamicSharedMemorySize, 130 * 1024);
  };
  if (device >= 0 && device < 64)
    std::call_once(configured[device], configure);
  else
    configure();
  gdn_folded_forward_kernel<E, H, Hg, 1, 64, false, false>
      <<<dim3(batch * 2, H), dim3(128, 1), FoldedForwardSmemLayout::TOTAL,
         stream>>>(static_cast<E *>(output), static_cast<const E *>(q),
                   static_cast<const E *>(k), static_cast<const E *>(v), a, m,
                   g, h0, nullptr, pool, nullptr, indices, cu, mapping, raw_cu,
                   nullptr, nullptr, batch, tokens, raw_batch, scale, v_stride);
}

void launch_folded_fwd_cute(void *output, const void *q, const void *k,
                            const void *v, const void *coefficients,
                            const int *cu, const int *mapping,
                            const int *raw_cu, const float *h0, float *pool,
                            const int *indices, int batch, int tokens,
                            int raw_batch, int device, int H, int Hg,
                            int64_t v_stride, float scale,
                            hggcStream_t stream) {
#define FORWARD_CASE(HV, HGV)                                                  \
  if (H == HV && Hg == HGV) {                                                  \
    launch_folded_forward<HV, HGV>(                                            \
        output, q, k, v, coefficients, cu, mapping, raw_cu, h0, pool, indices, \
        batch, tokens, raw_batch, device, v_stride, scale, stream);            \
    return;                                                                    \
  }
  FOLDED_HEAD_CONFIGS(FORWARD_CASE)
#undef FORWARD_CASE
  throw std::invalid_argument("unsupported folded forward head configuration");
}
#undef FOLDED_HEAD_CONFIGS

void launch_folded_correct_fp32(void *corrected, const void *pool,
                                const void *ht, const void *mt,
                                const signed char *fallback, const int *mapping,
                                const int *indices, int raw_batch, int H,
                                hggcStream_t stream, bool seed_first) {
  using flashqla_folded::gdn_folded_correct_kernel;
  auto *out = static_cast<float *>(corrected);
  const auto *initial = static_cast<const float *>(pool);
  const auto *affine = static_cast<const float *>(ht);
  const auto *linear = static_cast<const float *>(mt);
  const int blocks = raw_batch * H * 8;
  if (seed_first)
    gdn_folded_correct_kernel<true><<<blocks, 128, 0, stream>>>(
        out, initial, affine, linear, fallback, mapping, indices, H);
  else
    gdn_folded_correct_kernel<false><<<blocks, 128, 0, stream>>>(
        out, initial, affine, linear, fallback, mapping, indices, H);
}

namespace {

struct FoldedFirstSegment {
  template <typename State, typename Coordinates>
  __device__ __forceinline__ static void load(
      State& state, const Coordinates& coordinates, const void* corrected,
      int segment, int head, int dv_tile, int heads) {
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(coordinates); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(coordinates); ++n) {
        CUTE_UNROLL
        for (int vv = 0; vv < cute::size<0>(coordinates); ++vv) {
          const auto coord = coordinates(vv, m, n);
          const int64_t offset = flashqla_folded_correction::ch0_off(
              segment, head, cute::get<0>(coord), dv_tile,
              cute::get<1>(coord), heads);
          state(vv, m, n) = static_cast<const float*>(corrected)[offset];
        }
      }
    }
  }
};

template <typename FirstSegment>
void launch_folded_correct_bf16(
    void* corrected, const void* pool, const void* ht, const void* mt,
    const signed char* fallback, const int* mapping, const int* indices,
    int cp_batch, int raw_batch, int heads, int device, hggcStream_t stream) {
  using namespace flashqla_folded_correction;
  using E = __ppu_bfloat16;
  constexpr int Groups = 2;
  constexpr int Padding = 8;
  const int bytes = ch0_smem_bytes(Groups, false) +
                    (Groups * CH0_DVT + CH0_DK) * Padding * sizeof(E);
  ch0_configure_dynamic_smem<E, Groups, false, 4, Padding, FirstSegment>(device, bytes);
  correct_h0_cute_kernel<E, Groups, false, 4, Padding, FirstSegment>
      <<<raw_batch * heads * 2, 256, bytes, stream>>>(
          corrected, pool, indices, static_cast<const E*>(ht), static_cast<const E*>(mt),
          fallback, mapping, cp_batch, raw_batch, heads, 1, 1, false, false);
}

}  // namespace

void launch_gdn_correct_native(
    void* corrected, const void* pool, const void* ht, const void* mt,
    const signed char* fallback, const int* mapping, const int* indices,
    int cp_batch, int raw_batch, int heads, int device, hggcStream_t stream,
    bool seed_first) {
  const auto launch = seed_first ? launch_folded_correct_bf16<FoldedFirstSegment>
                                 : launch_folded_correct_bf16<void>;
  launch(corrected, pool, ht, mt, fallback, mapping, indices,
         cp_batch, raw_batch, heads, device, stream);
}
