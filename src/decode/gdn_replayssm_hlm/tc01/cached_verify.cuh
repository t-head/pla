// SPDX-License-Identifier: Apache-2.0
#pragma once

// Swap the column group in the accumulator with the within-group lane index.
template <class Word>
__device__ __forceinline__ void tc01_fold_transpose(Word (&x)[4], int lane) {
  const unsigned mask = 0xffffffffu;
  const bool odd = lane & 1, upper = lane & 2;
  const Word a = __shfl_xor_sync(mask, odd ? x[0] : x[1], 1, 4);
  const Word b = __shfl_xor_sync(mask, odd ? x[2] : x[3], 1, 4);
  if (odd) { x[0] = a; x[2] = b; }
  else { x[1] = a; x[3] = b; }
  const Word c = __shfl_xor_sync(mask, upper ? x[0] : x[2], 2, 4);
  const Word d = __shfl_xor_sync(mask, upper ? x[1] : x[3], 2, 4);
  if (upper) { x[0] = c; x[1] = d; }
  else { x[2] = c; x[3] = d; }
}

// Each four-lane group owns four K8 vectors of one checkpoint row.
// Transpose the K-pair register index with the K8-slice lane index.
template <int VGroup, class BFragment>
__device__ __forceinline__ void tc01_checkpoint_b_words(BFragment& first, BFragment& second,
                                                        uint4 slice, int lane) {
  static_assert(sizeof(first.x) == 2 * sizeof(uint4));
  uint32_t words[] = {slice.x, slice.y, slice.z, slice.w};
  tc01_fold_transpose(words, lane);
  __builtin_memcpy(first.x + VGroup * 4, words, 2 * sizeof(uint32_t));
  __builtin_memcpy(second.x + VGroup * 4, words + 2, 2 * sizeof(uint32_t));
}

// State staging and projection outputs share storage only across synchronized phases.
template <class S, int T, int BV, bool ParallelGram, bool StageBF16,
          bool GroupCheckpointLoads = false, bool ReuseFold = false,
          bool PackedT4 = false, class Params>
__global__ __launch_bounds__(128) void tc01_verify_kernel(Params p) {
  constexpr int R = 16, K = 128, CAP = 16;
  constexpr bool SPLIT = std::is_same_v<S, float>;
  constexpr bool WideDenseT4 = StageBF16 && T == 4 && BV == 128 &&
                               !std::is_same_v<Params, CachedVerify>;
  constexpr bool FusedGramEpilogue = (WideDenseT4 && !GroupCheckpointLoads) || PackedT4;
  static_assert(!PackedT4 || (!SPLIT && ParallelGram && T == 4 && BV == 64 &&
                                !std::is_same_v<Params, CachedVerify>));
  static_assert(!StageBF16 || (!SPLIT && !ParallelGram));
  static_assert(!GroupCheckpointLoads || WideDenseT4);
  static_assert(!ReuseFold || (!SPLIT && BV == 64 && T == 8 && !std::is_same_v<Params, CachedVerify>));
  constexpr int StageK = 32, StageStride = StageK + 8;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.y, hv = blockIdx.z, head = hv / (p.value_heads / p.heads);
  const int64_t active_slot = index(p.slots, row);
  if constexpr (!PackedT4) {
    if (active_slot <= p.null_slot) return;
  }
  const int64_t slot = PackedT4 ? 0 : active_slot;
  const int64_t replay = std::is_same_v<Params, CachedVerify> ? index(p.replay, row) : slot;
  if constexpr (!PackedT4) {
    assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  }
  const int64_t cr = index(p.lengths, replay), br = index(p.buffers, replay);
  if constexpr (!PackedT4) {
    assert(cr >= 0 && cr <= CAP && br >= 0 && br < 2);
    if (p.offsets.data) assert(index(p.offsets, replay) == 0);
  }
  const int count = PackedT4 ? min(CAP, max(0, int(cr))) : int(cr);
  const int bank = PackedT4 ? int(br) & 1 : int(br);
  const int vbase = blockIdx.x * BV;
  const bool overflow = count + T > CAP;
  const int write_bank = overflow ? 1 - bank : bank, wp = overflow ? 0 : count;
  const int64_t kb = replay * p.keys.stride[0] + bank * p.keys.stride[1] + head * K;
  const int64_t ub = replay * p.values.stride[0] + bank * p.values.stride[1] + hv * 128;
  const int64_t gb = replay * p.gates.stride[0] + bank * p.gates.stride[1] + hv * CAP;
  // Pack normalized Q and K in the TensorCell lane order before Verify.
  // Fold can reuse this tile once the Verify warps finish reading it.
  __shared__ __align__(32) BF x[PackedT4 ? CAP * BV : R * K], old_k[CAP * K];
  auto* packed_x = reinterpret_cast<uint32_t*>(x);
  static_assert(!PackedT4 || CAP * BV * sizeof(BF) >= (K / 16) * 32 * sizeof(uint2));
  __shared__ __align__(32) BF coeff[R * CAP], old_u[CAP * BV];
  // Dense BF16 T=4 shares storage between eight projection rows and packed Fold keys.
  constexpr int ProjectionElements = (FusedGramEpilogue && !PackedT4)
      ? 2 * T * BV : (SPLIT ? 3 : 2) * R * BV;
  __shared__ __align__(32) float projections[ProjectionElements];
  float* proj = projections;
  __shared__ __align__(32) float corr[R * 32];
  __shared__ float g[T], total[T], beta[T], gh[CAP], origin, decay[T], fold_decay[CAP];
  __shared__ float prefix[T];
  BF input_v[T];
  if constexpr (PackedT4) {
    if (warp >= 2) {
      const int v_out = (warp - 2) * (BV / 2) + lane;
      const BF* input = static_cast<const BF*>(p.v.data) +
          int64_t(row * T) * p.v.stride[0] + hv * K + vbase + v_out;
      #pragma unroll
      for (int t = 0; t < T; ++t)
        input_v[t] = lane < BV / 2 ? input[t * p.v.stride[0]] : BF(0.f);
    }
  } else if constexpr (ReuseFold) {
    if (tid < BV) {
      const BF* src = static_cast<const BF*>(p.v.data) +
          int64_t(row * T) * p.v.stride[0] + hv * K + vbase + tid;
      #pragma unroll
      for (int t = 0; t < T; ++t) input_v[t] = src[t * p.v.stride[0]];
    }
  }
  uint2 q_bits{}, k_bits{};
  if constexpr (PackedT4) {
    const int64_t qr = int64_t(row * T + warp) * p.q.stride[0] + head * K + lane * 4;
    const int64_t kr = int64_t(row * T + warp) * p.k.stride[0] + head * K + lane * 4;
    q_bits = *reinterpret_cast<const uint2*>(static_cast<const BF*>(p.q.data) + qr);
    k_bits = *reinterpret_cast<const uint2*>(static_cast<const BF*>(p.k.data) + kr);
  }
  if constexpr (!PackedT4) {
    for (int i = 2 * T * K + tid; i < R * K; i += 128) x[i] = BF(0.f);
  }
  if constexpr (PackedT4) {
    if (count) {
      const int row_k = tid / 16, col_k = tid % 16 * 8;
      const int row_u = tid / 8, col_u = tid % 8 * 8;
      const BF* src_k = static_cast<const BF*>(p.keys.data) + kb + col_k;
      const BF* src_u = static_cast<const BF*>(p.values.data) + ub + vbase + col_u;
      uint4 k0{}, k1{}, u{};
      if (row_k < count)
        k0 = *reinterpret_cast<const uint4*>(src_k + row_k * p.keys.stride[2]);
      if (row_k + 8 < count)
        k1 = *reinterpret_cast<const uint4*>(src_k + (row_k + 8) * p.keys.stride[2]);
      if (row_u < count)
        u = *reinterpret_cast<const uint4*>(src_u + row_u * p.values.stride[2]);
      // Let the three independent reads issue before publishing the operands.
      asm volatile("" :: "r"(k0.x), "r"(k0.y), "r"(k0.z), "r"(k0.w),
                           "r"(k1.x), "r"(k1.y), "r"(k1.z), "r"(k1.w),
                           "r"(u.x), "r"(u.y), "r"(u.z), "r"(u.w) : "memory");
      *reinterpret_cast<uint4*>(old_k + row_k * K + col_k) = k0;
      *reinterpret_cast<uint4*>(old_k + (row_k + 8) * K + col_k) = k1;
      *reinterpret_cast<uint4*>(old_u + row_u * BV + col_u) = u;
    }
  } else {
    auto copy = [&](const auto& tensor, int64_t offset, bool valid, BF* dst) {
      if constexpr (ReuseFold) {
        uint4 words{};
        if (valid) words = *reinterpret_cast<const uint4*>(static_cast<const BF*>(tensor.data) + offset);
        *reinterpret_cast<uint4*>(dst) = words;
      } else {
        cached_copy_bf8(tensor, offset, valid ? 8 : 0, dst);
      }
    };
    if (count) {
      #pragma unroll
      for (int i = tid * 8; i < CAP * K; i += 128 * 8) {
        copy(p.keys, kb + (i / K) * p.keys.stride[2] + i % K, i / K < count, old_k + i);
      }
      #pragma unroll
      for (int i = tid * 8; i < CAP * BV; i += 128 * 8) {
        copy(p.values, ub + (i / BV) * p.values.stride[2] + vbase + i % BV, i / BV < count, old_u + i);
      }
    }
  }
  if constexpr (PackedT4) {
    if (warp == 0 || warp == 1) {
      const float old = lane < count ? static_cast<const float*>(p.gates.data)[gb + lane] : 0.f;
      const float start = __shfl_sync(0xffffffffu, old, count ? count - 1 : 0);
      if (warp == 1 && lane < CAP) gh[lane] = old;
      if (warp == 0 && lane == 0) origin = start;
    }
  } else {
    if (tid < CAP) gh[tid] = tid < count ? read<float>(p.gates, gb + tid) : 0.f;
    if (tid == 0) origin = count ? read<float>(p.gates, gb + count - 1) : 0.f;
  }
  for (int t = warp; t < T; t += 4) {
    float sq = 0.f, sk = 0.f, raw_q[4], raw_k[4];
    const int64_t qo = int64_t(row * T + t) * p.q.stride[0] + head * K + lane * 4;
    const int64_t ko = int64_t(row * T + t) * p.k.stride[0] + head * K + lane * 4;
    if constexpr (PackedT4) {
      asm volatile("" :: "r"(q_bits.x), "r"(q_bits.y), "r"(k_bits.x), "r"(k_bits.y));
      raw_q[0] = __uint_as_float(q_bits.x << 16);
      raw_q[1] = __uint_as_float(q_bits.x & 0xffff0000u);
      raw_q[2] = __uint_as_float(q_bits.y << 16);
      raw_q[3] = __uint_as_float(q_bits.y & 0xffff0000u);
      raw_k[0] = __uint_as_float(k_bits.x << 16);
      raw_k[1] = __uint_as_float(k_bits.x & 0xffff0000u);
      raw_k[2] = __uint_as_float(k_bits.y << 16);
      raw_k[3] = __uint_as_float(k_bits.y & 0xffff0000u);
    } else if constexpr (std::is_same_v<Params, CachedVerify>) {
      read_vector<BF, 4>(p.q, qo, 4, raw_q);
      read_vector<BF, 4>(p.k, ko, 4, raw_k);
    } else {
      load_bf16x4(static_cast<const BF*>(p.q.data) + qo, raw_q);
      load_bf16x4(static_cast<const BF*>(p.k.data) + ko, raw_k);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      sq += raw_q[i] * raw_q[i];
      sk += raw_k[i] * raw_k[i];
    }
    sq = sum_warp(sq); sk = sum_warp(sk);
    const float qs = p.scale / norm_denominator(sq, p.norm_policy, p.epsilon);
    const float ks = 1.f / norm_denominator(sk, p.norm_policy, p.epsilon);
    uint32_t keys[2], queries[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      keys[i] = cached_bf16_pair(raw_k[2 * i] * ks, raw_k[2 * i + 1] * ks);
      queries[i] = cached_bf16_pair(raw_q[2 * i] * qs, raw_q[2 * i + 1] * qs);
    }
    if constexpr (PackedT4) {
      // Each lane in the MMA owns the same K pair in both halves of K16.
      const int tile = lane / 4, pair = (lane % 2) * 2, half = (lane % 4) / 2;
      const int i = tile * 64 + (t * 4 + pair) * 2 + half;
      packed_x[i] = keys[0];
      packed_x[i + 2] = keys[1];
      packed_x[i + T * 8] = queries[0];
      packed_x[i + T * 8 + 2] = queries[1];
    } else {
      *reinterpret_cast<uint2*>(x + t * K + lane * 4) = uint2{keys[0], keys[1]};
      *reinterpret_cast<uint2*>(x + (T+t) * K + lane * 4) = uint2{queries[0], queries[1]};
    }
    if (lane == 0) {
      float gt = read_any(p.a, int64_t(row * T + t) * p.a.stride[0] + hv);
      float bt = read_any(p.b, int64_t(row * T + t) * p.b.stride[0] + hv);
      if (!p.processed_gates) {
        const float arg = gt + read<float>(p.dt_bias, hv * p.dt_bias.stride[0]);
        gt = -expf(read<float>(p.A_log, hv * p.A_log.stride[0])) *
            (arg <= 20.f ? 0.6931471805599453f * __log2f(__fadd_rn(1.f, __expf(arg))) : arg);
        bt = 1.f / (1.f + expf(-bt));
      }
      g[t] = gt; beta[t] = bt;
    }
  }
  if constexpr (PackedT4) {
    if (active_slot <= p.null_slot) return;
    assert(active_slot == 0 && cr >= 0 && cr <= CAP && br >= 0 && br < 2);
  }
  __syncthreads();
  if constexpr (!FusedGramEpilogue) {
    if (tid == 0) {
      float acc = 0.f;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        acc += g[t];
        total[t] = origin + acc;
        if constexpr (ReuseFold) prefix[t] = acc;
      }
    }
  }
  if (p.materialize_overflow && overflow && tid < count) fold_decay[tid] = expf(origin - gh[tid]);
  {
    awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major> a;
    awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major> b;
    awmma::fragment<awmma::accumulator,16,16,16,float> c;
    if (warp < 2 && (warp == 0 || count)) {
      awmma::fill_fragment(c,0.f);
      #pragma unroll
      for (int kk = 0; kk < K; kk += 16) {
        if constexpr (PackedT4) {
          const uint2 x_pair = reinterpret_cast<const uint2*>(packed_x)[(kk / 16) * 32 + lane];
          const uint4 x_rows{x_pair.x, x_pair.y, 0, 0};
          static_assert(sizeof(a.x) == sizeof(x_rows) && sizeof(b.x) == sizeof(x_rows));
          __builtin_memcpy(a.x, &x_rows, sizeof(x_rows));
          if (warp == 0) __builtin_memcpy(b.x, &x_rows, sizeof(x_rows));
          else awmma::load_matrix_sync(b, old_k + kk, K);
        } else {
          awmma::load_matrix_sync(a,x + kk,K);
          awmma::load_matrix_sync(b,(warp == 0 ? x : old_k) + kk,K);
        }
        awmma::mma_sync(c,a,b,c);
      }
      if constexpr (FusedGramEpilogue) {
        // The first four registers of each lane contain rows 0–7 of the tc01
        // M16N16 accumulator. Each lane owns columns spaced four apart.
        const int r = lane / 4, col = lane % 4;
        const int token = r % T;
        const float prefix0 = g[0];
        const float prefix1 = prefix0 + g[1];
        const float prefix2 = prefix1 + g[2];
        const float prefix3 = prefix2 + g[3];
        const float start = origin;
        auto total_at = [&](int i) {
          return start + (i < 2 ? (i == 0 ? prefix0 : prefix1) : (i == 2 ? prefix2 : prefix3));
        };
        const float row_total = total_at(token);
        if (warp == 0) {
          const float col_total = total_at(col);
          if (lane < T) {
            total[col] = col_total;
            decay[col] = expf(col_total);
          }
          const float value = r < T ? beta[token] * c.x[0] : c.x[0];
          corr[r * 32 + col] = value * expf(row_total - col_total);
        } else {
          #pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int j = col + 4 * e;
            coeff[r * CAP + j] = BF(j < count ? c.x[e] * expf(row_total - gh[j]) : 0.f);
          }
        }
      } else {
        awmma::store_matrix_sync(corr + warp * 16, c, 32, awmma::mem_row_major);
      }
    }
  }
  constexpr int M = T <= 4 ? 8 : 16;
  constexpr int N = 256 / M;
  using AF = awmma::fragment<awmma::matrix_a,M,N,16,BF,awmma::row_major>;
  using BFg = awmma::fragment<awmma::matrix_b,M,N,16,BF,awmma::col_major>;
  using CF = awmma::fragment<awmma::accumulator,M,N,16,float>;
  constexpr int PROJECT_WARPS = ParallelGram ? 2 : 4;
  const int worker = ParallelGram ? warp - 2 : warp;
  constexpr int PER_WARP = (BV + PROJECT_WARPS * N - 1) / (PROJECT_WARPS * N);
  constexpr int LOAD = sizeof(S) == 2 ? 8 : 4;
  CF high[PER_WARP], low[PER_WARP];
  #pragma unroll
  for (int s = 0; s < PER_WARP; ++s) {
    awmma::fill_fragment(high[s], 0.f);
    if constexpr (SPLIT) awmma::fill_fragment(low[s], 0.f);
  }
  const auto* state = static_cast<const S*>(p.state.data) +
      slot * p.state.stride[0] + (int64_t(hv) * 128 + vbase) * K;
  if constexpr (StageBF16) {
    // A warp shares a padded staging tile across the two K=16 products.
    // With four adjacent K pairs per V row, stride 40 distributes rows
    // over all eight groups of TSM banks on tc01.
    constexpr int StageElements = N * StageStride;
    static_assert(FusedGramEpilogue || BV * StageStride * sizeof(BF) <= sizeof(projections));
    static_assert(M * BV * sizeof(float) <= sizeof(projections));
    if constexpr (FusedGramEpilogue) {
      static_assert(PER_WARP == 1 && BV / N == 4);
      const BF* src = state + (warp * N + lane / 4) * K + (lane % 4) * 8;
      #pragma unroll 1
      for (int kk = 0; kk < K; kk += StageK) {
        BFg b0, b1;
        const uint4 a = *reinterpret_cast<const uint4*>(src + kk);
        const uint4 b = *reinterpret_cast<const uint4*>(src + kk + 8 * K);
        const uint4 c = *reinterpret_cast<const uint4*>(src + kk + 16 * K);
        const uint4 d = *reinterpret_cast<const uint4*>(src + kk + 24 * K);
        asm volatile("" :: "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
                            "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w),
                            "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w),
                            "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w) : "memory");
        tc01_checkpoint_b_words<0>(b0, b1, a, lane);
        tc01_checkpoint_b_words<1>(b0, b1, b, lane);
        tc01_checkpoint_b_words<2>(b0, b1, c, lane);
        tc01_checkpoint_b_words<3>(b0, b1, d, lane);
        AF a0, a1;
        awmma::load_matrix_sync(a0, x + kk, K);
        awmma::load_matrix_sync(a1, x + kk + 16, K);
        awmma::mma_sync(high[0], a0, b0, high[0]);
        awmma::mma_sync(high[0], a1, b1, high[0]);
      }
    } else if (warp < BV / N) {
      BF* tile = reinterpret_cast<BF*>(projections) + warp * StageElements;
      #pragma unroll 2
      for (int kk = 0; kk < K; kk += StageK) {
        AF a0, a1;
        awmma::load_matrix_sync(a0, x + kk, K);
        awmma::load_matrix_sync(a1, x + kk + 16, K);
        #pragma unroll
        for (int s = 0; s < PER_WARP; ++s) {
          const int vt = warp + s * 4;
          if (vt < BV / N) {
            if constexpr (GroupCheckpointLoads) {
              // Keep four aligned state loads in flight before storing the tile.
              const int row = lane / 4, col = (lane % 4) * 8;
              const BF* src = state + (vt * N + row) * K + kk + col;
              BF* dst = tile + row * StageStride + col;
              const uint4 a = *reinterpret_cast<const uint4*>(src);
              const uint4 b = *reinterpret_cast<const uint4*>(src + 8 * K);
              const uint4 c = *reinterpret_cast<const uint4*>(src + 16 * K);
              const uint4 d = *reinterpret_cast<const uint4*>(src + 24 * K);
              asm volatile("" :: "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
                                  "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w),
                                  "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w),
                                  "r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w) : "memory");
              *reinterpret_cast<uint4*>(dst) = a;
              *reinterpret_cast<uint4*>(dst + 8 * StageStride) = b;
              *reinterpret_cast<uint4*>(dst + 16 * StageStride) = c;
              *reinterpret_cast<uint4*>(dst + 24 * StageStride) = d;
            } else {
              #pragma unroll
              for (int i = lane * 8; i < N * StageK; i += 32 * 8) {
                const int row = i / StageK, col = i % StageK;
                const int soff = (vt * N + row) * K + kk + col;
                if constexpr (std::is_same_v<Params, CachedVerify>) {
                  cached_copy_bf8(p.state, state - static_cast<const S*>(p.state.data) + soff,
                                  8, tile + row * StageStride + col);
                } else {
                  *reinterpret_cast<uint4*>(tile + row * StageStride + col) =
                      *reinterpret_cast<const uint4*>(state + soff);
                }
              }
            }
            __syncwarp();
            BFg b0, b1;
            awmma::load_matrix_sync(b0, tile, StageStride);
            awmma::load_matrix_sync(b1, tile + 16, StageStride);
            awmma::mma_sync(high[s], a0, b0, high[s]);
            awmma::mma_sync(high[s], a1, b1, high[s]);
            __syncwarp();
          }
        }
      }
    }
  } else if (worker >= 0 && worker < BV / N) {
    #pragma unroll
    for (int kk = 0; kk < K; kk += 16) {
      AF a;
      if constexpr (PackedT4) {
        const uint2 packed = reinterpret_cast<const uint2*>(packed_x)[(kk / 16) * 32 + lane];
        static_assert(sizeof(a.x) == sizeof(packed));
        __builtin_memcpy(a.x, &packed, sizeof(packed));
      } else {
        awmma::load_matrix_sync(a, x + kk, K);
      }
      #pragma unroll
      for (int s = 0; s < PER_WARP; ++s) {
        const int vt = worker + s * PROJECT_WARPS;
        if (vt >= 0 && vt < BV / N) {
          BFg hi, lo;
          if constexpr (SPLIT) {
            // tc01 column-major B owns K pairs 8 apart at V rows 8 apart.
            const S* tile = state + vt * N * K + kk;
            const int kpair = (lane % 4) * 2, vrow = lane / 4;
            #pragma unroll
            for (int j = 0; j < N / 8; ++j) {
              #pragma unroll
              for (int i = 0; i < 2; ++i) {
                const S* src = tile + (vrow + 8 * j) * K + kpair + 8 * i;
                const float2 pair = *reinterpret_cast<const float2*>(src);
                const int e = 4 * j + 2 * i;
                hi.x[e] = BF(pair.x);
                hi.x[e + 1] = BF(pair.y);
                lo.x[e] = BF(pair.x - float(hi.x[e]));
                lo.x[e + 1] = BF(pair.y - float(hi.x[e + 1]));
              }
            }
          } else {
            awmma::load_matrix_sync(hi, state + vt * N * K + kk, K);
          }
          awmma::mma_sync(high[s], a, hi, high[s]);
          if constexpr (SPLIT) awmma::mma_sync(low[s], a, lo, low[s]);
        }
      }
    }
  }
  __syncthreads();
  if constexpr (!FusedGramEpilogue) {
    for (int i = tid; i < R * CAP; i += 128) {
      const int r = i / CAP, j = i % CAP, t = r % T;
      coeff[i] = BF(r < 2 * T && j < count ? corr[r * 32 + 16 + j] * expf(total[t] - gh[j]) : 0.f);
    }
    if (tid < T) decay[tid] = expf(total[tid]);
    for (int i = tid; i < 2 * T * T; i += 128) {
      const int r = i / T, j = i % T, t = r % T;
      const float value = corr[r * 32 + j];
      corr[r * 32 + j] = (r < T ? beta[t] * value : value) * expf(total[t]-total[j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int s = 0; s < PER_WARP; ++s) {
    const int vt = worker + s * PROJECT_WARPS;
    if (vt >= 0 && vt < BV / N) {
      CF h;
      awmma::fill_fragment(h, 0.f);
      if (count) {
        AF a;
        awmma::fragment<awmma::matrix_b,M,N,16,BF,awmma::row_major> b;
        awmma::load_matrix_sync(a, coeff, CAP);
        awmma::load_matrix_sync(b, old_u + vt * N, BV);
        awmma::mma_sync(h, a, b, h);
      }
      #pragma unroll
      for (int e = 0; e < CF::num_elements; ++e) {
        // tc01 accumulator layout for the exact M16N16 and M8N32 shapes.
        const int row = lane / 4 + (M == 16 && e >= 4 ? 8 : 0);
        const float projection = high[s].x[e] + (SPLIT ? low[s].x[e] : 0.f);
        h.x[e] += decay[row % T] * projection;
      }
      awmma::store_matrix_sync(proj + vt * N, h, BV, awmma::mem_row_major);
    }
  }
  // Each dense T=4 warp produces and consumes its own 32 value columns.
  if constexpr (FusedGramEpilogue) {
    static_assert(PER_WARP == 1 && PROJECT_WARPS * N == BV);
    __syncwarp();
  } else {
    __syncthreads();
  }
  if (PackedT4 ? warp >= 2 && lane < BV / 2 : tid < BV) {
    const int local_v = PackedT4 ? (warp - 2) * (BV / 2) + lane : tid;
    const int column = hv * 128 + vbase + local_v;
    const int64_t input_row = int64_t(row) * T;
    const int64_t value_base = replay * p.values.stride[0] + write_bank * p.values.stride[1];
    float u[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      const float input = (PackedT4 || ReuseFold) ? float(input_v[t]) : read<BF>(p.v, (input_row + t) * p.v.stride[0] + column);
      float update = beta[t] * (input - proj[t * BV + local_v]);
      #pragma unroll
      for (int j = 0; j < t; ++j) update -= corr[t * 32 + j] * u[j];
      u[t] = update;
      float output = proj[(T + t) * BV + local_v];
      #pragma unroll
      for (int j = 0; j <= t; ++j) output += corr[(T+t) * 32 + j] * u[j];
      write<BF>(p.out, (input_row + t) * p.out.stride[0] + column, output);
      write<BF>(p.values, value_base + (wp + t) * p.values.stride[2] + column, update);
    }
  }
  if (blockIdx.x == 0) {
    if (hv % (p.value_heads / p.heads) == 0) {
      if constexpr (PackedT4) {
        const int tile = lane / 4, pair = (lane % 2) * 2, half = (lane % 4) / 2;
        const int i = tile * 64 + (warp * 4 + pair) * 2 + half;
        BF* dst = static_cast<BF*>(p.keys.data) + replay * p.keys.stride[0] +
            write_bank * p.keys.stride[1] + (wp + warp) * p.keys.stride[2] + head * K + lane * 4;
        *reinterpret_cast<uint2*>(dst) = uint2{packed_x[i], packed_x[i + 2]};
      } else {
        for (int i = tid; i < T * K; i += 128)
          static_cast<BF*>(p.keys.data)[replay * p.keys.stride[0] + write_bank * p.keys.stride[1] + (wp+i/K) * p.keys.stride[2] + head * K + i%K] = x[i];
      }
    }
    if (tid == 0) {
      float acc = 0.f;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        if constexpr (ReuseFold) acc = prefix[t];
        else acc += g[t];
        write<float>(p.gates,replay * p.gates.stride[0] + write_bank * p.gates.stride[1] + hv * CAP + wp+t,overflow ? acc : total[t]);
      }
    }
  }
  if (p.materialize_overflow && overflow) {
    // The old bank is read-only in this CTA. The new candidates are in the
    // opposite bank, so Fold reads exactly the prefix that caused overflow.
    __syncthreads();
    for (int i = tid; i < CAP * BV; i += 128) {
      const int j = i / BV;
      const float scaled = j < count ? float(old_u[i]) * fold_decay[j] : 0.f;
      const BF high = BF(scaled);
      old_u[i] = high;
      x[i] = BF(scaled - float(high));
    }
    using FoldBF = awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major>;
    // Verify has finished reading projections. Reuse that storage for one
    // packed copy of each key tile, shared by all warps and value tiles.
    auto* fold_keys = reinterpret_cast<uint4*>(proj);
    if constexpr ((FusedGramEpilogue && !PackedT4) || ReuseFold) {
      static_assert(sizeof(FoldBF{}.x) == sizeof(uint4));
      static_assert((K / 16) * 32 * sizeof(uint4) <= sizeof(projections));
      #pragma unroll
      for (int tile = 0; tile < K / (16 * 4); ++tile) {
        const int kt = warp + tile * 4;
        FoldBF b;
        awmma::load_matrix_sync(b, old_k + kt * 16, K);
        uint4 packed;
        __builtin_memcpy(&packed, b.x, sizeof(packed));
        fold_keys[kt * 32 + lane] = packed;
      }
    }
    __syncthreads();
    using FoldCF = awmma::fragment<awmma::accumulator,16,16,16,float>;
    static_assert(FoldCF::num_elements == 8, "Fold expects eight accumulator elements per lane");
    const float state_decay = expf(origin);
    for (int vt = warp; vt < BV / 16; vt += 4) {
      awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major> a_hi,a_lo;
      awmma::load_matrix_sync(a_hi,old_u+vt*16,BV);
      awmma::load_matrix_sync(a_lo,x+vt*16,BV);
      if constexpr (!std::is_same_v<Params, CachedVerify> && std::is_same_v<S, BF>) {
        BF* target = static_cast<BF*>(p.state.data) +
            slot * p.state.stride[0] + (int64_t(hv) * 128 + vbase + vt * 16) * K;
        #pragma unroll 2
        for (int kt = 0; kt < K; kt += 16) {
          FoldBF b;
          FoldCF hi, lo;
          awmma::fill_fragment(hi, 0.f);
          awmma::fill_fragment(lo, 0.f);
          if constexpr ((FusedGramEpilogue && !PackedT4) || ReuseFold) {
            const uint4 packed = fold_keys[(kt / 16) * 32 + lane];
            __builtin_memcpy(b.x, &packed, sizeof(packed));
          } else {
            awmma::load_matrix_sync(b, old_k + kt, K);
          }
          uint2 cp[2];
          if constexpr (ReuseFold) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
              const BF* src = target + (lane / 4 + 8 * half) * K + kt + 4 * (lane % 4);
              cp[half] = *reinterpret_cast<const uint2*>(src);
            }
          }
          awmma::mma_sync(hi, a_hi, b, hi);
          awmma::mma_sync(lo, a_lo, b, lo);
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            float delta[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) delta[e] = hi.x[4*half+e] + lo.x[4*half+e];
            tc01_fold_transpose(delta, lane);
            BF* dst = target + (lane / 4 + 8 * half) * K + kt + 4 * (lane % 4);
            float checkpoint[4];
            if constexpr (ReuseFold) {
              checkpoint[0] = __uint_as_float(cp[half].x << 16);
              checkpoint[1] = __uint_as_float(cp[half].x & 0xffff0000u);
              checkpoint[2] = __uint_as_float(cp[half].y << 16);
              checkpoint[3] = __uint_as_float(cp[half].y & 0xffff0000u);
            } else {
              load_bf16x4(dst, checkpoint);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e) checkpoint[e] = checkpoint[e] * state_decay + delta[e];
            *reinterpret_cast<uint2*>(dst) = uint2{cached_bf16_pair(checkpoint[0], checkpoint[1]),
                                                   cached_bf16_pair(checkpoint[2], checkpoint[3])};
          }
        }
      } else {
        float* fold_scratch = projections + R * BV;
        for (int kt = 0; kt < K; kt += 32) {
          #pragma unroll
          for (int part = 0; part < 2; ++part) {
            awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major> b;
            FoldCF hi,lo,sum;
            awmma::fill_fragment(hi,0.f); awmma::fill_fragment(lo,0.f);
            awmma::load_matrix_sync(b,old_k+kt+part*16,K);
            awmma::mma_sync(hi,a_hi,b,hi);
            awmma::mma_sync(lo,a_lo,b,lo);
            #pragma unroll
            for (int e = 0; e < 8; ++e) sum.x[e] = hi.x[e]+lo.x[e];
            awmma::store_matrix_sync((part == 0 ? proj : fold_scratch)+vt*256,sum,16,awmma::mem_row_major);
          }
          __syncwarp();
          for (int i = lane*LOAD; i < 512; i += 32*LOAD) {
            const int row = i / 32, col = i % 32;
            const int64_t off = slot*p.state.stride[0]+(int64_t(hv)*128+vbase+vt*16+row)*K+kt+col;
            float values[LOAD];
            read_vector<S,LOAD>(p.state,off,LOAD,values);
            const float* scratch = col < 16 ? proj : fold_scratch;
            #pragma unroll
            for (int e = 0; e < LOAD; ++e) {
              const float delta = scratch[vt*256+row*16+col%16+e];
              values[e] = values[e]*state_decay+delta;
            }
            cached_fold_store_vector<S,LOAD>(p.state,off,values);
          }
          __syncwarp();
        }
      }
    }
  }
}
