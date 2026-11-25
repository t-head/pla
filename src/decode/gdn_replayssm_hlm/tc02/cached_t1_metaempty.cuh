// SPDX-License-Identifier: Apache-2.0
// The supported 16x16 A and 16x8 B operands have two populated rows
// or columns. The lane/word layout is verified by probe_operands.cu.
template <bool Correlation, class Fragment>
__device__ __forceinline__ void load_pair_operand(
    Fragment& fragment, const BF* input, int stride, int lane) {
  constexpr int Words = Correlation ? 4 : 2;
  static_assert(sizeof(fragment.x) == Words*sizeof(uint32_t));
  uint32_t bits[Words]{};
  if (lane < 8) {
    const BF* pair = input + (lane/4)*stride + 2*(lane%4);
    __builtin_memcpy(&bits[0], pair, sizeof(bits[0]));
    __builtin_memcpy(&bits[Correlation ? 2 : 1], pair + 8, sizeof(bits[0]));
  }
  __builtin_memcpy(fragment.x, bits, sizeof(bits));
}

// SPDX-License-Identifier: Apache-2.0
// BF16 fixed-length T1, K=V=128, L=16. Each warp owns 32 consecutive V rows.
template <int StageStride, bool SkipEmpty, bool WarpStages, bool I32, bool AlignedInputs, int FlatHeads>
__global__ __launch_bounds__(128) void cached_decode_t1(CachedT1 p) {
  constexpr int K = 128, V = 128, L = 16, WarpV = 32, KStep = 32;
  constexpr int StageElements = WarpV * StageStride;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  constexpr int HVDiv = FlatHeads == 0 ? 1 : 4*FlatHeads;
  const int row = FlatHeads ? blockIdx.y/HVDiv : blockIdx.y;
  const int hv = FlatHeads ? blockIdx.y%HVDiv : blockIdx.z;
  const int group = AlignedInputs ? 4 : p.value_heads / p.heads, head = hv / group;
  const int v0 = warp * WarpV;
  const int64_t slot = AlignedInputs ? static_cast<const int32_t*>(p.slots.data)[row]
                                    : meta_index<I32>(p.slots, row);
  if (slot <= p.null_slot) return;
  const int64_t replay = AlignedInputs
      ? slot : meta_index<I32>(p.replay, row);
  assert(slot >= 0 && slot < p.state_slots && replay >= 0 && replay < p.replay_slots);
  const int64_t count64 = AlignedInputs ? static_cast<const int32_t*>(p.lengths.data)[replay]
                                       : meta_index<I32>(p.lengths, replay);
  const int64_t bank64 = AlignedInputs ? static_cast<const int32_t*>(p.buffers.data)[replay]
                                      : meta_index<I32>(p.buffers, replay);
  assert(count64 >= 0 && count64 <= L && bank64 >= 0 && bank64 < 2);
  if (!AlignedInputs && p.offsets.data) assert(meta_index<I32>(p.offsets, replay) == 0);
  const int count = int(count64), bank = int(bank64);
  const bool overflow = count == L;
  const int dst_bank = overflow ? 1 - bank : bank, wp = overflow ? 0 : count;
  const bool use_history = !SkipEmpty || count != 0;

  const int64_t key_base = replay*p.keys.stride[0] + bank*p.keys.stride[1] + head*K;
  const int64_t u_base = replay*p.values.stride[0] + bank*p.values.stride[1] + hv*V;
  const int64_t gate_base = replay*p.gates.stride[0] + bank*p.gates.stride[1] + hv*L;
  const int64_t state_base = slot*p.state.stride[0] + int64_t(hv)*V*K;

  struct Gate {
    float origin, token, total, decay, beta, qk, qk_decay;
  };
  __shared__ Gate gates;
  __shared__ __align__(32) BF key_query[2*K], coeff[2*L];
  __shared__ __align__(128) BF old_keys[L*K], old_u[L*V];
  __shared__ __align__(32) float correlations[16*L];
  __shared__ __align__(32) BF state_stages[2][4][StageElements];
  __shared__ float old_g[L];

  auto copy_history = [](const auto& view, int64_t offset, int valid, BF* destination) {
    if constexpr (AlignedInputs) {
      uint4 bits{};
      if (valid)
        bits = *reinterpret_cast<const uint4*>(static_cast<const BF*>(view.data) + offset);
      __builtin_memcpy(destination, &bits, sizeof(bits));
    } else {
      cached_copy_bf8(view, offset, valid, destination);
    }
  };
  if (use_history) {
    if constexpr (AlignedInputs) {
      if (count != 0) {
        if (warp == 1) {
          auto load_history = [&](const auto& view, int64_t offset, BF* tile, int rows) {
            const auto address = static_cast<uint32_t>(__cvta_generic_to_shared(tile));
            const BF* source = static_cast<const BF*>(view.data) + offset;
            asm volatile(
                "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
                " [%0], [%1], {%2, %3, %4}, {%2, %5, %4}, {%6, %8}, {%7, %7, %7}, %7;"
                :: "r"(address), "l"(source), "r"(K), "r"(rows), "r"(1), "r"(L),
                   "l"(view.stride[2]*int64_t(sizeof(BF))), "r"(0), "l"(int64_t(0))
                : "memory");
          };
          load_history(p.keys, key_base, old_keys, count);
          load_history(p.values, u_base, old_u, count);
          __pipeline_commit();
        }
      } else {
        const uint4 zero{};
        #pragma unroll
        for (int i = tid*8; i < L*K; i += 128*8) {
          __builtin_memcpy(old_keys + i, &zero, sizeof(zero));
          __builtin_memcpy(old_u + i, &zero, sizeof(zero));
        }
      }
    } else {
      #pragma unroll
      for (int i = tid*8; i < L*K; i += 128*8)
        copy_history(p.keys, key_base + (i/K)*p.keys.stride[2] + i%K,
                        i/K < count ? 8 : 0, old_keys + i);
      #pragma unroll
      for (int i = tid*8; i < L*V; i += 128*8)
        copy_history(p.values, u_base + (i/V)*p.values.stride[2] + i%V,
                        i/V < count ? 8 : 0, old_u + i);
    }
    if (tid < L) old_g[tid] = tid < count ? read<float>(p.gates, gate_base + tid) : 0.f;
  }

  if (warp == 0) {
    float q[4], k[4], q2 = 0.f, k2 = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = lane + 32*i;
      q[i] = read<BF>(p.q, int64_t(row)*p.q.stride[0] + head*K + c);
      k[i] = read<BF>(p.k, int64_t(row)*p.k.stride[0] + head*K + c);
      q2 += q[i]*q[i];
      k2 += k[i]*k[i];
    }
    const float qs = p.scale / norm_denominator(sum_warp(q2), AlignedInputs ? 2 : p.norm_policy, p.epsilon);
    const float ks = 1.f / norm_denominator(sum_warp(k2), AlignedInputs ? 2 : p.norm_policy, p.epsilon);
    float qk = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = lane + 32*i;
      const BF k_bf = BF(k[i]*ks), q_bf = BF(q[i]*qs);
      key_query[c] = k_bf;
      key_query[K + c] = q_bf;
      qk += float(k_bf)*float(q_bf);
    }
    qk = sum_warp(qk);
    if (lane == 0) gates.qk = qk;
  }
  if (warp == 2 && lane == 0) {
    const int64_t a_offset = int64_t(row)*p.a.stride[0] + hv;
    float g = AlignedInputs ? read<float>(p.a, a_offset) : read_any(p.a, a_offset);
    const int64_t b_offset = int64_t(row)*p.b.stride[0] + hv;
    float b = AlignedInputs ? read<float>(p.b, b_offset) : read_any(p.b, b_offset);
    if (!AlignedInputs && !p.processed_gates) {
      const float arg = g + read<float>(p.dt_bias, hv*p.dt_bias.stride[0]);
      g = -expf(read<float>(p.A_log, hv*p.A_log.stride[0])) *
          (arg <= 20.f ? 0.6931471805599453f*__log2f(__fadd_rn(1.f, __expf(arg))) : arg);
      b = 1.f / (1.f + expf(-b));
    }
    const float origin = count ? read<float>(p.gates, gate_base + count - 1) : 0.f;
    const float total = origin + g;
    gates.origin = origin;
    gates.token = g;
    gates.total = total;
    gates.decay = expf(total);
    gates.beta = b;
    gates.qk_decay = expf(total-total);
  }
  if constexpr (AlignedInputs) {
    if (warp == 1 && use_history && count != 0) __pipeline_wait_prior(0);
  }
  __syncthreads();

  using CorrA = awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major>;
  using CorrB = awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major>;
  using CorrC = awmma::fragment<awmma::accumulator,16,16,16,float>;
  using StateA = awmma::fragment<awmma::matrix_a,32,8,16,BF,awmma::row_major>;
  using HistA = awmma::fragment<awmma::matrix_a,32,8,16,BF,awmma::col_major>;
  using ProjectB = awmma::fragment<awmma::matrix_b,32,8,16,BF,awmma::col_major>;
  using ProjectC = awmma::fragment<awmma::accumulator,32,8,16,float>;

  const BF* state = static_cast<const BF*>(p.state.data) + state_base;
  const bool aligned = AlignedInputs || (reinterpret_cast<uintptr_t>(state) & 15u) == 0;
  auto copy_stage = [&](int phase, BF* destination) {
    static_assert(KStep == 32);
    const unsigned row_lane = unsigned(lane) >> 2;
    const unsigned column = (unsigned(lane) & 3)*8;
    const unsigned global_base = (unsigned(v0) + row_lane)*K + unsigned(phase)*KStep + column;
    BF* const shared_base = destination + row_lane*StageStride + column;
    #pragma unroll
    for (unsigned slab = 0; slab < WarpV; slab += 8) {
      const unsigned offset = global_base + slab*K;
      BF* target = shared_base + slab*StageStride;
      if (aligned)
        __pipeline_memcpy_async(target, state + offset, 16);
      else
        cached_copy_bf8(p.state, state_base + offset, 8, target);
    }
    if (aligned) __pipeline_commit();
  };

  copy_stage(0, state_stages[0][warp]);
  // Compute coefficients on one warp while state staging is in progress.
  if (use_history && warp == 0) {
    CorrC c;
    awmma::fill_fragment(c, 0.f);
    #pragma unroll
    for (int offset = 0; offset < K; offset += 16) {
      CorrA a;
      CorrB b;
      load_pair_operand<true>(a, key_query + offset, K, lane);
      awmma::load_matrix_sync(b, old_keys + offset, K);
      awmma::mma_sync(c, a, b, c);
    }
    awmma::store_matrix_sync(correlations, c, L, awmma::mem_row_major);
    __syncwarp();
    const int j = lane % L;
    coeff[lane] = j < count ? BF(correlations[lane]*expf(gates.total-old_g[j])) : BF(0.f);
  }

  ProjectC projected;
  awmma::fill_fragment(projected, 0.f);
  #pragma unroll 1
  for (int Phase = 0; Phase < K/KStep; ++Phase) {
    if (Phase < K/KStep - 1)
      copy_stage(Phase + 1, state_stages[1 - Phase%2][warp]);
    if (aligned) {
      if (Phase < K/KStep - 1) __pipeline_wait_prior(1);
      else __pipeline_wait_prior(0);
    }
    if constexpr (WarpStages) __syncwarp();
    else __syncthreads();

    #pragma unroll
    for (int k = 0; k < KStep; k += 16) {
      StateA a;
      ProjectB b;
      awmma::load_matrix_sync(a, state_stages[Phase%2][warp] + k, StageStride);
      load_pair_operand<false>(b, key_query + Phase*KStep + k, K, lane);
      awmma::mma_sync(projected, a, b, projected);
    }
    if constexpr (WarpStages) __syncwarp();
    else __syncthreads();
  }
  __syncthreads();  // All warps may now consume the history coefficients.

  float* projected_values = reinterpret_cast<float*>(state_stages[0][warp]);
  awmma::store_matrix_sync(projected_values, projected, WarpV, awmma::mem_col_major);
  __syncwarp();
  const float kp = projected_values[lane], qp = projected_values[WarpV + lane];

  float kh = 0.f, qh = 0.f;
  if (use_history) {
    HistA a;
    ProjectB b;
    ProjectC historical;
    awmma::fill_fragment(historical, 0.f);
    awmma::load_matrix_sync(a, old_u + v0, V);
    load_pair_operand<false>(b, coeff, L, lane);
    awmma::mma_sync(historical, a, b, historical);
    float* history_values = reinterpret_cast<float*>(state_stages[1][warp]);
    awmma::store_matrix_sync(history_values, historical, WarpV, awmma::mem_col_major);
    __syncwarp();
    kh = history_values[lane];
    qh = history_values[WarpV + lane];
  }
  const int v = v0 + lane;
  const float value = read<BF>(p.v, int64_t(row)*p.v.stride[0] + hv*V + v);
  const float u = gates.beta * (value - gates.decay*kp - kh);
  const float output = gates.decay*qp + qh + (gates.qk*gates.qk_decay)*u;
  write<BF>(p.out, int64_t(row)*p.out.stride[0] + hv*V + v, output);
  write<BF>(p.values, replay*p.values.stride[0] + dst_bank*p.values.stride[1] +
            wp*p.values.stride[2] + hv*V + v, u);
  if (hv % group == 0)
    static_cast<BF*>(p.keys.data)[replay*p.keys.stride[0] + dst_bank*p.keys.stride[1] +
                                  wp*p.keys.stride[2] + head*K + tid] = key_query[tid];
  if (tid == 0)
    write<float>(p.gates, replay*p.gates.stride[0] + dst_bank*p.gates.stride[1] +
                 hv*L + wp, overflow ? gates.token : gates.total);

  // Reuse both staging buffers for the same two-dot K32 Fold. Each warp has
  // consumed its own 32 rows of the old U and state before repurposing them.
  if (p.materialize_overflow && overflow) {
    __syncwarp();
    auto* low_u = state_stages[1][warp];
    auto* delta = reinterpret_cast<float*>(state_stages[0][warp]);
    #pragma unroll
    for (int j = 0; j < L; ++j) {
      BF& prior = old_u[j*V + v];
      const float scaled = float(prior)*expf(gates.origin-old_g[j]);
      const BF upper = BF(scaled);
      prior = upper;
      low_u[j*WarpV + lane] = BF(scaled-float(upper));
    }
    __syncwarp();
    using FoldA = awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major>;
    using FoldB = awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major>;
    const float decay = expf(gates.origin);
    for (int tile_v = 0; tile_v < WarpV; tile_v += 16) {
      FoldA u_hi, u_lo;
      awmma::load_matrix_sync(u_hi, old_u + v0 + tile_v, V);
      awmma::load_matrix_sync(u_lo, low_u + tile_v, WarpV);
      for (int k = 0; k < K; k += KStep) {
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
          FoldB keys;
          CorrC high, low;
          awmma::fill_fragment(high, 0.f);
          awmma::fill_fragment(low, 0.f);
          awmma::load_matrix_sync(keys, old_keys + k + 16*part, K);
          awmma::mma_sync(high, u_hi, keys, high);
          awmma::mma_sync(low, u_lo, keys, low);
          #pragma unroll
          for (int i = 0; i < CorrC::num_elements; ++i) high.x[i] += low.x[i];
          awmma::store_matrix_sync(delta + part*256, high, 16, awmma::mem_row_major);
        }
        __syncwarp();
        #pragma unroll
        for (int i = lane*8; i < 16*KStep; i += 32*8) {
          const int tile_row = i/KStep, tile_k = i%KStep;
          const int64_t offset = state_base + (v0 + tile_v + tile_row)*K + k + tile_k;
          const float* addend = delta + (tile_k/16)*256 + tile_row*16 + tile_k%16;
          float values[8];
          read_vector<BF,8>(p.state, offset, 8, values);
          #pragma unroll
          for (int e = 0; e < 8; ++e) values[e] = values[e]*decay + addend[e];
          cached_fold_store_vector<BF,8>(p.state, offset, values);
        }
        __syncwarp();
      }
    }
  }
}
