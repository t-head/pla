// SPDX-License-Identifier: Apache-2.0
template <int TileV, int StatePadding = 0, int Heads = 0, bool Single = false, class Params>
__global__ __launch_bounds__(2*TileV) void cached_decode_t1_tiny(Params p) {
  constexpr int K = 128, V = 128, L = 16, WarpV = 16, Threads = 2*TileV;
  constexpr int Warps = TileV/WarpV, StateStride = K + StatePadding;
  static_assert(TileV == 16 || TileV == 32 || TileV == 64 || TileV == 128);
  static_assert(StatePadding % 8 == 0);
  constexpr int HistoryWarp = Warps == 1 ? 0 : 1;
  constexpr int GateWarp = Warps > 2 ? 2 : HistoryWarp;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = Single ? 0 : blockIdx.y, hv = blockIdx.z, head = Heads == 1 ? 0 : hv/4;
  const int vbase = blockIdx.x*TileV, local_v = warp*WarpV;
  const int64_t slot = static_cast<const int32_t*>(p.slots.data)[row];
  if (slot <= p.null_slot) return;
  assert(slot >= 0 && slot < p.state_slots && slot < p.replay_slots);
  const int64_t state_offset = slot*p.state.stride[0] + int64_t(hv*V + vbase)*K;
  BF* const state = static_cast<BF*>(p.state.data) + state_offset;

  struct Gate { float origin, g, total, decay, beta, qk; };
  __shared__ Gate gates;
  __shared__ __align__(128) BF state_stage[TileV*StateStride];
  __shared__ __align__(128) BF old_k[L*K], old_u[L*TileV], low_u[L*TileV];
  __shared__ __align__(32) BF kq[2*K], coeff[2*L];
  __shared__ float history_g[L];

  const bool copies_state = StatePadding != 0 || warp == 0;
  if constexpr (StatePadding == 0) {
    if (copies_state)
      cached_bulk_bf16(state_stage, state, K, TileV, K, TileV, 0, K);
  } else {
    #pragma unroll
    for (unsigned i = unsigned(tid)*8; i < TileV*K; i += Threads*8)
      __pipeline_memcpy_async(state_stage + (i/K)*StateStride + i%K, state + i, 16);
  }

  if (warp == 0) {
    float q[4], k[4], q2 = 0.f, k2 = 0.f;
    #pragma unroll
    for (int i=0; i<4; ++i) {
      const int c = lane + 32*i;
      q[i] = read<BF>(p.q, int64_t(row)*p.q.stride[0] + head*K + c);
      k[i] = read<BF>(p.k, int64_t(row)*p.k.stride[0] + head*K + c);
      q2 += q[i]*q[i];
      k2 += k[i]*k[i];
    }
    const float qs = p.scale/(sqrtf(sum_warp(q2)) + p.epsilon);
    const float ks = 1.f/(sqrtf(sum_warp(k2)) + p.epsilon);
    float qk = 0.f;
    #pragma unroll
    for (int i=0; i<4; ++i) {
      const int c = lane + 32*i;
      const BF kn = BF(k[i]*ks), qn = BF(q[i]*qs);
      kq[c] = kn;
      kq[K+c] = qn;
      qk += float(kn)*float(qn);
    }
    qk = sum_warp(qk);
    if (lane == 0) gates.qk = qk;
  }

  const int count = static_cast<const int32_t*>(p.lengths.data)[slot];
  const int bank = static_cast<const int32_t*>(p.buffers.data)[slot];
  assert(count >= 0 && count <= L && bank >= 0 && bank < 2);
  const bool has_history = count != 0, overflow = count == L;
  const int64_t kb = slot*p.keys.stride[0] + bank*p.keys.stride[1] + head*K;
  const int64_t ub = slot*p.values.stride[0] + bank*p.values.stride[1] + hv*V;
  const int64_t gb = slot*p.gates.stride[0] + bank*p.gates.stride[1] + hv*L;
  const bool copies_history = has_history && warp == HistoryWarp;
  if (copies_history) {
    cached_bulk_bf16(old_k, static_cast<const BF*>(p.keys.data)+kb, K, count, K, L, 0, p.keys.stride[2]);
    cached_bulk_bf16(old_u, static_cast<const BF*>(p.values.data)+ub, V, count, TileV, L, vbase, p.values.stride[2]);
  }
  if (copies_state || copies_history) __pipeline_commit();
  if (tid < count) history_g[tid] = read<float>(p.gates,gb+tid);

  if (warp == GateWarp && lane == 0) {
    const float origin = has_history ? read<float>(p.gates,gb+count-1) : 0.f;
    const float g = read<float>(p.a, int64_t(row)*p.a.stride[0] + hv);
    const float total = origin+g;
    gates.origin = origin;
    gates.g = g;
    gates.total = total;
    gates.decay = expf(total);
    gates.beta = read<float>(p.b, int64_t(row)*p.b.stride[0] + hv);
  }
  if (copies_state || copies_history) __pipeline_wait_prior(0);
  if constexpr (Warps == 1) __syncwarp();
  else __syncthreads();

  using RowA = awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::row_major>;
  using ColA = awmma::fragment<awmma::matrix_a,16,16,16,BF,awmma::col_major>;
  using ColB = awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::col_major>;
  using RowB = awmma::fragment<awmma::matrix_b,16,16,16,BF,awmma::row_major>;
  using Acc = awmma::fragment<awmma::accumulator,16,16,16,float>;

  if (has_history && warp == 0) {
    Acc correlation;
    awmma::fill_fragment(correlation,0.f);
    #pragma unroll 1
    for (int k=0; k<K; k+=16) {
      RowA a;
      ColB b;
      cached_two_rows<true>(a,kq+k,K,lane);
      load_b16(b,old_k+k,K,lane);
      awmma::mma_sync(correlation,a,b,correlation);
    }
    // For the first two C16 rows, each lane owns four columns.
    if (lane < 8) {
      const int t = lane/4, start = (lane%4)*2;
      #pragma unroll
      for (int i=0; i<4; ++i) {
        const int j = start + (i/2)*8 + i%2;
        coeff[t*L+j] = j < count ?
            BF(correlation.x[(i/2)*4+i%2]*expf(gates.total-history_g[j])) : BF(0.f);
      }
    }
  }

  Acc projected, history;
  awmma::fill_fragment(projected,0.f);
  awmma::fill_fragment(history,0.f);
  const BF* const staged_row = state_stage + local_v*StateStride;
  #pragma unroll 1
  for (int k=0; k<K; k+=16) {
    RowA a;
    ColB b;
    load_a16(a,staged_row+k,StateStride,lane);
    cached_two_rows<false>(b,kq+k,K,lane);
    awmma::mma_sync(projected,a,b,projected);
  }
  if (has_history) {
    if constexpr (Warps == 1) __syncwarp();
    else __syncthreads();
    ColA a;
    ColB b;
    load_fold_operand<true>(a, old_u+local_v, TileV, lane);
    cached_two_rows<false>(b, coeff, L, lane);
    awmma::mma_sync(history,a,b,history);
  }

  const int dst = overflow ? 1-bank : bank, wp = overflow ? 0 : count;
  if ((lane&3) == 0) {
    const float qk = gates.qk*expf(gates.total-gates.total);
    #pragma unroll
    for (int i=0; i<2; ++i) {
      const int v = vbase + local_v + lane/4 + i*8;
      const float value = read<BF>(p.v,int64_t(row)*p.v.stride[0]+hv*V+v);
      const float u = gates.beta*(value-gates.decay*projected.x[2*i]-history.x[2*i]);
      const float out = gates.decay*projected.x[2*i+1] + history.x[2*i+1] + qk*u;
      write<BF>(p.out,int64_t(row)*p.out.stride[0]+hv*V+v,out);
      write<BF>(p.values,slot*p.values.stride[0]+dst*p.values.stride[1]+wp*p.values.stride[2]+hv*V+v,u);
    }
  }
  if (blockIdx.x == 0) {
    if ((hv&3) == 0)
      for (int i=tid; i<K; i+=Threads)
        static_cast<BF*>(p.keys.data)[slot*p.keys.stride[0]+dst*p.keys.stride[1]+
                                     wp*p.keys.stride[2]+head*K+i] = kq[i];
    if (tid == 0)
      write<float>(p.gates,slot*p.gates.stride[0]+dst*p.gates.stride[1]+hv*L+wp,overflow ? gates.g : gates.total);
  }

  if (p.materialize_overflow && overflow) {
    __syncwarp();
    #pragma unroll
    for (int i=lane; i<L*WarpV; i+=32) {
      const int t = i/WarpV;
      const int offset = t*TileV + local_v + i%WarpV;
      const float scaled = float(old_u[offset])*expf(gates.origin-history_g[t]);
      const BF upper = BF(scaled);
      old_u[offset] = upper;
      low_u[offset] = BF(scaled-float(upper));
    }
    __syncwarp();
    ColA upper, lower;
    load_fold_operand<true>(upper,old_u+local_v,TileV,lane);
    load_fold_operand<true>(lower,low_u+local_v,TileV,lane);
    const float decay = expf(gates.origin);
    const int lane_row = lane/4, lane_column = (lane%4)*2;
    #pragma unroll 1
    for (int k=0; k<K; k+=16) {
      RowB key;
      Acc high, low;
      load_fold_operand<false>(key,old_k+k,K,lane);
      awmma::fill_fragment(high,0.f);
      awmma::fill_fragment(low,0.f);
      awmma::mma_sync(high,upper,key,high);
      awmma::mma_sync(low,lower,key,low);
      #pragma unroll
      for (int i=0; i<4; ++i) {
        const int r = lane_row + (i%2)*8;
        const int c = k + lane_column + (i/2)*8;
        const uint32_t bits = *reinterpret_cast<const uint32_t*>(staged_row + r*StateStride+c);
        const float x0 = __uint_as_float(bits<<16), x1 = __uint_as_float(bits&0xffff0000u);
        const float d0 = high.x[2*i]+low.x[2*i], d1 = high.x[2*i+1]+low.x[2*i+1];
        const uint32_t b0 = __bfloat16_as_ushort(BF(x0*decay + d0));
        const uint32_t b1 = __bfloat16_as_ushort(BF(x1*decay + d1));
        *reinterpret_cast<uint32_t*>(state + (local_v+r)*K+c) = b0 | (b1<<16);
      }
    }
  }
}
